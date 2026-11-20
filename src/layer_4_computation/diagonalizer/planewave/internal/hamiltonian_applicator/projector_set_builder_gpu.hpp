#pragma once

/**
 * GPUProjectorSetBuilder — builds Kleinman-Bylander projector sets on a
 * device.
 *
 * A projector is a product of three things over the plane waves of a k-point:
 *
 *   <k+G|β> = (1/√Ω) (-i)^l · β(|k+G|) Y_lm(k+G) · S_atom(k+G)
 *
 * The radial function times the spherical harmonic depends on the species and
 * the channel (beta function and m) but not on the atom; the structure factor
 * S depends on the atom but not on the channel. A cell of many atoms of few
 * species therefore has few distinct factors and many products of them: 64
 * silicon atoms with 18 channels each are 1,152 projectors made from 64
 * structure factors and 18 form factors. The host builder forms the products
 * on the host, and a device run then uploads all of them. This builder
 * computes only the factors on the host, uploads those, and forms the products
 * on the device, into the set's device copy: the set's largest array never
 * exists on the host unless something asks for it there.
 *
 * The products use what every device backend already has, and are written
 * straight into the set. One atom at a time, the kernel that applies the
 * kinetic energy (a real diagonal applied to a batch of vectors) multiplies the
 * real part of the atom's structure factor into the form factors of all its
 * channels at once, and its imaginary part likewise; the device BLAS adds the
 * second to the first as the imaginary part and scales each shell by its
 * constant (1/√Ω)(-i)^l. The operations are the host builder's in the host
 * builder's order, real factor times structure factor and then the constant,
 * so the two builders give the same projectors.
 *
 * Forming them one channel at a time for every atom, and copying each atom's
 * vector to its place, was tried first: 1,152 device-to-device copies per
 * k-point on the 64-atom cell, 0.26 s on an A100, slower than uploading the
 * finished set.
 *
 * The workspaces are kept between builds and shared, so builds are serialized.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/profiling/stage_timer.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

namespace q3 {

/// ProjectorSetBuilder that forms the projectors on a device; see the file comment.
class GPUProjectorSetBuilder : public ProjectorSetBuilder {
  public:
    /**
     * @param device_backend  The device the sets are built on. Must not be null.
     * @param device_blas     The device BLAS of that backend. Must outlive this object.
     */
    GPUProjectorSetBuilder(const Crystal& crystal_structure,
                           const PlanewaveBasis& planewave_basis,
                           const NormConservingPlanewaveBasis& pseudopotential_basis,
                           std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend,
                           const execution::gpu::blas::GPUBLAS& device_blas)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis)
        , device_backend_(require(std::move(device_backend)))
        , device_blas_(device_blas)
        , device_structure_factors_(0, device_backend_)
        , device_imaginary_products_(0, device_backend_) {}

    /// See ProjectorSetBuilder::build(); the set's device copy is written, its host copy is not.
    void build(const Vector3D& kpoint_crystal_coordinates,
               const std::vector<int>& g_vector_indices,
               int number_of_plane_waves,
               KleinmanBylanderProjectorSet& projector_set) const override {
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        const auto& pseudopotential_set = pseudopotential_basis_.pseudopotential_set();
        const auto& beta_function_table = pseudopotential_basis_.beta_function_table();
        const int number_of_atoms = crystal_structure_.number_of_atoms();
        const int number_of_species = static_cast<int>(pseudopotential_set.size());
        const std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves);

        Complex* device_projectors =
            projector_set.begin_device_build(kpoint_crystal_coordinates,
                                             crystal_structure_,
                                             pseudopotential_basis_.total_number_of_projectors(),
                                             number_of_plane_waves,
                                             device_backend_);
        if (number_of_plane_waves == 0 || number_of_atoms == 0) {
            return;
        }
        std::optional<StageTimer> stage_timer(std::in_place, "setup.projectors.device.factors");

        const Matrix3x3& reciprocal_lattice_vectors =
            crystal_structure_.reciprocal_lattice_vectors();
        Vector3D kpoint_cartesian;
        for (int dimension = 0; dimension < 3; dimension++) {
            kpoint_cartesian[dimension] =
                kpoint_crystal_coordinates.x * reciprocal_lattice_vectors[0][dimension] +
                kpoint_crystal_coordinates.y * reciprocal_lattice_vectors[1][dimension] +
                kpoint_crystal_coordinates.z * reciprocal_lattice_vectors[2][dimension];
        }
        const Real two_pi_over_lattice_parameter =
            crystal_structure_.two_pi_over_lattice_parameter();
        const Real inverse_sqrt_cell_volume = 1.0 / std::sqrt(crystal_structure_.cell_volume());

        // k+G, and its length in atomic units, for every plane wave.
        std::vector<std::array<Real, 3>> k_plus_g_cartesian(plane_waves);
        std::vector<Real> k_plus_g_magnitude_squared(plane_waves);
        std::vector<Real> k_plus_g_magnitude(plane_waves);
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            Vector3D k_plus_g =
                kpoint_cartesian + planewave_basis_.g_vectors()[g_vector_indices[g_index]].g_cart;
            k_plus_g_cartesian[g_index] = {k_plus_g.x, k_plus_g.y, k_plus_g.z};
            Real magnitude_squared =
                k_plus_g.x * k_plus_g.x + k_plus_g.y * k_plus_g.y + k_plus_g.z * k_plus_g.z;
            k_plus_g_magnitude_squared[g_index] = magnitude_squared;
            k_plus_g_magnitude[g_index] =
                std::sqrt(magnitude_squared) * two_pi_over_lattice_parameter;
        }

        // Where each atom's projectors start in the set.
        std::vector<int> first_projector_of_atom(static_cast<std::size_t>(number_of_atoms), 0);
        int projector_count = 0;
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            int species_index = crystal_structure_.species_type_index(atom_index);
            first_projector_of_atom[static_cast<std::size_t>(atom_index)] = projector_count;
            for (const auto& projector : pseudopotential_set[species_index].projectors) {
                projector_count += 2 * projector.angular_momentum + 1;
            }
        }

        /**
         * The structure factors, exp(-2πi (k+G)·r) per atom, computed on the host and
         * uploaded in one copy: for each atom its real part and then its imaginary part.
         */
        std::vector<Real> structure_factors(2 * plane_waves *
                                            static_cast<std::size_t>(number_of_atoms));
        int atom_threads = execution::threading::thread_count_for(number_of_atoms, 1);
#pragma omp parallel for schedule(static) num_threads(atom_threads) if (atom_threads > 1)
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            Vector3D atomic_position =
                crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
            Real k_dot_r = kpoint_cartesian.dot(atomic_position);
            Real* real_part =
                structure_factors.data() + static_cast<std::size_t>(2 * atom_index) * plane_waves;
            Real* imaginary_part = real_part + plane_waves;
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                Real phase_angle =
                    constants::two_pi *
                    (planewave_basis_.g_vectors()[g_vector_indices[g_index]].g_cart.dot(
                         atomic_position) +
                     k_dot_r);
                Complex structure_factor = compute_phase_factor(phase_angle);
                real_part[g_index] = structure_factor.real();
                imaginary_part[g_index] = structure_factor.imag();
            }
        }
        stage_timer.emplace("setup.projectors.device.upload");
        ensure_capacity(device_structure_factors_, structure_factors.size());
        device_backend_->copy(device_structure_factors_.data(),
                              structure_factors.data(),
                              structure_factors.size() * sizeof(Real));
        stage_timer.emplace("setup.projectors.device.factors");

        // Y_lm(k+G) for every plane wave, up to the largest l of any species.
        int maximum_angular_momentum = 0;
        for (int species_index = 0; species_index < number_of_species; species_index++) {
            for (const auto& projector : pseudopotential_set[species_index].projectors) {
                maximum_angular_momentum =
                    std::max(maximum_angular_momentum, projector.angular_momentum);
            }
        }
        int number_of_lm_components =
            (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);
        std::vector<std::vector<Real>> spherical_harmonics;
        compute_real_spherical_harmonics_batch(number_of_lm_components,
                                               number_of_plane_waves,
                                               k_plus_g_cartesian,
                                               k_plus_g_magnitude_squared,
                                               spherical_harmonics);

        /**
         * Each species' form factors, β(|k+G|) Y_lm(k+G) per channel, uploaded in one copy
         * per species. They are real; the device kernel takes a batch of complex vectors, so
         * they go up with a zero imaginary part.
         */
        while (static_cast<int>(device_form_factors_.size()) < number_of_species) {
            device_form_factors_.emplace_back(0, device_backend_);
        }
        std::vector<int> channels_of_species(static_cast<std::size_t>(number_of_species), 0);
        std::size_t largest_species_elements = 0;
        std::vector<Real> radial_values(plane_waves);
        std::vector<Complex> form_factors;
        for (int species_index = 0; species_index < number_of_species; species_index++) {
            const auto& projectors = pseudopotential_set[species_index].projectors;
            int number_of_channels = 0;
            for (const auto& projector : projectors) {
                number_of_channels += 2 * projector.angular_momentum + 1;
            }
            channels_of_species[static_cast<std::size_t>(species_index)] = number_of_channels;
            form_factors.assign(plane_waves * static_cast<std::size_t>(number_of_channels),
                                Complex(0.0, 0.0));
            int channel = 0;
            for (int beta_index = 0; beta_index < static_cast<int>(projectors.size());
                 beta_index++) {
                int angular_momentum = projectors[beta_index].angular_momentum;
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    radial_values[g_index] = beta_function_table.interpolate(
                        species_index, beta_index, k_plus_g_magnitude[g_index]);
                }
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    int lm_index =
                        harmonic_combined_degree_order_index(angular_momentum, magnetic_projection);
                    Complex* form_factor =
                        form_factors.data() + static_cast<std::size_t>(channel) * plane_waves;
                    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                        form_factor[g_index] =
                            radial_values[g_index] * spherical_harmonics[g_index][lm_index];
                    }
                    channel++;
                }
            }
            stage_timer.emplace("setup.projectors.device.upload");
            execution::gpu::device::GPUDeviceBuffer<Complex>& device_form_factors =
                device_form_factors_[static_cast<std::size_t>(species_index)];
            ensure_capacity(device_form_factors, form_factors.size());
            if (!form_factors.empty()) {
                device_backend_->copy(device_form_factors.data(),
                                      form_factors.data(),
                                      form_factors.size() * sizeof(Complex));
            }
            stage_timer.emplace("setup.projectors.device.factors");
            largest_species_elements = std::max(largest_species_elements, form_factors.size());
        }
        ensure_capacity(device_imaginary_products_, largest_species_elements);

        /**
         * The set is not cleared first: every element of it is written below. Clearing it,
         * to have the device take its memory in one piece, was three quarters of the build:
         * 0.56 s of 0.73 s for the four sets of a 64-atom silicon cell on an A100, where the
         * products themselves take 0.03 s.
         */
        stage_timer.emplace("setup.projectors.device.products");
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            int species_index = crystal_structure_.species_type_index(atom_index);
            int number_of_channels = channels_of_species[static_cast<std::size_t>(species_index)];
            if (number_of_channels == 0) {
                continue;
            }
            const Complex* species_form_factors =
                device_form_factors_[static_cast<std::size_t>(species_index)].data();
            const Real* real_part = device_structure_factors_.data() +
                                    static_cast<std::size_t>(2 * atom_index) * plane_waves;
            Complex* atom_projectors =
                device_projectors +
                static_cast<std::size_t>(
                    first_projector_of_atom[static_cast<std::size_t>(atom_index)]) *
                    plane_waves;
            // F ∘ S = F ∘ Re[S] + i · F ∘ Im[S], every channel of the atom at once.
            diagonalizer::planewave::gpu::launch_apply_kinetic_energy(number_of_channels,
                                                                      number_of_plane_waves,
                                                                      real_part,
                                                                      species_form_factors,
                                                                      atom_projectors);
            diagonalizer::planewave::gpu::launch_apply_kinetic_energy(
                number_of_channels,
                number_of_plane_waves,
                real_part + plane_waves,
                species_form_factors,
                device_imaginary_products_.data());
            device_blas_.axpy(number_of_channels * number_of_plane_waves,
                              Complex(0.0, 1.0),
                              device_imaginary_products_.data(),
                              atom_projectors);
            // The constant (1/√Ω)(-i)^l of each shell: one beta function with its 2l+1 components.
            int channel = 0;
            for (const auto& projector : pseudopotential_set[species_index].projectors) {
                int shell_channels = 2 * projector.angular_momentum + 1;
                Complex combined_prefactor =
                    inverse_sqrt_cell_volume *
                    compute_angular_phase_factor(projector.angular_momentum);
                device_blas_.scale(shell_channels * number_of_plane_waves,
                                   combined_prefactor,
                                   atom_projectors +
                                       static_cast<std::size_t>(channel) * plane_waves);
                channel += shell_channels;
            }
        }
        device_backend_->synchronize();
    }

  private:
    [[nodiscard]] static std::shared_ptr<execution::gpu::device::GPUDeviceBackend>
    require(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend) {
        if (!device_backend) {
            throw std::invalid_argument(
                "GPUProjectorSetBuilder: the device backend must not be null");
        }
        return device_backend;
    }

    /// Grow `buffer` to hold `elements`; never shrink it.
    template <typename Element>
    void ensure_capacity(execution::gpu::device::GPUDeviceBuffer<Element>& buffer,
                         std::size_t elements) const {
        if (buffer.size() < elements) {
            buffer = execution::gpu::device::GPUDeviceBuffer<Element>(elements, device_backend_);
        }
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
    const execution::gpu::blas::GPUBLAS& device_blas_;
    mutable std::mutex workspace_mutex_;
    /// Per atom, the real part of its structure factor and then the imaginary part.
    mutable execution::gpu::device::GPUDeviceBuffer<Real> device_structure_factors_;
    /// One species' form factors per entry, plane waves × channels, real values held as complex.
    mutable std::vector<execution::gpu::device::GPUDeviceBuffer<Complex>> device_form_factors_;
    /// Workspace for Im[S] ∘ form factors of one atom.
    mutable execution::gpu::device::GPUDeviceBuffer<Complex> device_imaginary_products_;
};

} // namespace q3
