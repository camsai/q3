#pragma once

/**
 * GPUProjectorStrainOverlapCalculator — projector strain overlaps formed on
 * a device, without building the projectors or their derivatives.
 *
 * A projector and each of its six strain derivatives is a product of a real
 * form factor F over the plane waves, a constant c = (1/√Ω)(-i)^l, and the
 * atom's structure factor S:
 *
 *   column(G) = c · F(G) · S_atom(G)
 *
 * F depends on the species, the channel (beta function and m) and the kind
 * (projector, or derivative along one strain component), not on the atom; S
 * depends on the atom alone. The overlap with a band is therefore
 *
 *   <column|ψ> = conj(c) · Σ_G F(G) · [conj(S_atom(G)) ψ(G)]
 *
 * so an atom's overlaps of every kind and channel are one small matrix product:
 * the species' form factors, kinds × channels of them, against the bands
 * multiplied through by the conjugate structure factor. The host builder writes
 * out seven columns per projector, each as long as the plane waves, and a device
 * run would have to upload them: on a 64-atom silicon cell that is 6 GB per
 * k-point. Here the form factors of a species go up once per k-point, the
 * structure factors of a block's atoms once per block, and the bands are already
 * on the device.
 *
 * It uses what every device backend has: the kernel that applies the kinetic
 * energy, which is a real diagonal applied to a batch of vectors, for the real
 * and the imaginary part of conj(S)·ψ; the device BLAS to combine the two and
 * for the product. The constant is applied on the host, to the overlaps, which
 * are small.
 *
 * The workspaces are kept between blocks and k-points: a device buffer taken
 * fresh is unified memory the device has not touched, and its first use faults
 * every page in.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_strain_overlap_calculator.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/band_matrix_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

namespace q3 {

/// ProjectorStrainOverlapCalculator that never builds the columns; see the file comment.
class GPUProjectorStrainOverlapCalculator : public ProjectorStrainOverlapCalculator {
  public:
    /**
     * @param device_backend  The device the products run on. Must not be null.
     * @param device_blas     The device BLAS of that backend. Must outlive this object.
     */
    GPUProjectorStrainOverlapCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveBasis& pseudopotential_basis,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend,
        const execution::gpu::blas::GPUBLAS& device_blas)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis)
        , device_backend_(require(std::move(device_backend)))
        , device_blas_(device_blas)
        , device_gathered_bands_(0, device_backend_)
        , device_conjugate_structure_factors_(0, device_backend_)
        , device_phased_bands_(0, device_backend_)
        , device_phased_bands_imaginary_(0, device_backend_)
        , device_overlaps_(0, device_backend_) {}

    /// See ProjectorStrainOverlapCalculator::begin_kpoint(); uploads each species' form factors.
    void begin_kpoint(const ProjectorStrainInputs& inputs,
                      int /*maximum_projectors_per_block*/) override {
        inputs_ = &inputs;
        const int number_of_plane_waves = inputs.number_of_plane_waves;
        const std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves);
        const Real inverse_sqrt_cell_volume = 1.0 / std::sqrt(crystal_structure_.cell_volume());
        const Real inverse_two_pi_over_lattice_parameter =
            1.0 / crystal_structure_.two_pi_over_lattice_parameter();
        const int number_of_species =
            static_cast<int>(pseudopotential_basis_.pseudopotential_set().size());
        while (static_cast<int>(species_.size()) < number_of_species) {
            species_.emplace_back(device_backend_);
        }
        block_first_atom_ = 0;
        block_end_atom_ = 0;
        block_projectors_ = 0;

        std::vector<Real> beta_at_g(plane_waves);
        std::vector<Real> beta_derivative_at_g(plane_waves);
        for (int species_index = 0; species_index < number_of_species; species_index++) {
            SpeciesFormFactors& species = species_[static_cast<std::size_t>(species_index)];
            species.number_of_channels =
                pseudopotential_basis_.number_of_projector_channels(species_index);
            const std::size_t number_of_form_factors =
                static_cast<std::size_t>(number_of_kinds * species.number_of_channels);
            species.prefactors.assign(static_cast<std::size_t>(species.number_of_channels),
                                      Complex(0.0, 0.0));
            species.form_factors.assign(plane_waves * number_of_form_factors, Complex(0.0, 0.0));
            int channel = 0;
            for (int beta_index = 0;
                 beta_index < pseudopotential_basis_.number_of_projectors(species_index);
                 beta_index++) {
                int angular_momentum =
                    pseudopotential_basis_.beta_angular_momentum(species_index, beta_index);
                Complex prefactor =
                    inverse_sqrt_cell_volume * compute_angular_phase_factor(angular_momentum);
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    beta_derivative_at_g[g_index] = pseudopotential_basis_.interpolate_dbeta(
                        species_index, beta_index, (*inputs.magnitude_bohr)[g_index]);
                    beta_at_g[g_index] = pseudopotential_basis_.interpolate_beta(
                        species_index, beta_index, (*inputs.magnitude_bohr)[g_index]);
                }
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    int lm_index =
                        harmonic_combined_degree_order_index(angular_momentum, magnetic_projection);
                    species.prefactors[static_cast<std::size_t>(channel)] = prefactor;
                    Complex* projector_form_factor =
                        species.form_factors.data() +
                        static_cast<std::size_t>(channel) * plane_waves;
                    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                        projector_form_factor[g_index] =
                            beta_at_g[g_index] * (*inputs.spherical_harmonics)[g_index][lm_index];
                    }
                    for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                        int direction_a = strain_direction_a[strain_pair];
                        int direction_b = strain_direction_b[strain_pair];
                        Complex* derivative_form_factor =
                            species.form_factors.data() +
                            static_cast<std::size_t>(
                                (strain_pair + 1) * species.number_of_channels + channel) *
                                plane_waves;
                        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                            Real spherical_harmonic =
                                (*inputs.spherical_harmonics)[g_index][lm_index];
                            /**
                             * The radial part and the angular part of the derivative, as in
                             * the file comment of ProjectorStrainOverlapCalculator; β is
                             * divided by 2π/a for the angular part, whose spherical-harmonic
                             * derivatives are in (2π/a)⁻¹.
                             */
                            derivative_form_factor[g_index] =
                                beta_derivative_at_g[g_index] * spherical_harmonic *
                                    (*inputs.radial_strain_weights)[strain_pair][g_index] +
                                beta_at_g[g_index] * inverse_two_pi_over_lattice_parameter *
                                    (*inputs.spherical_harmonic_derivatives)[direction_a][g_index]
                                                                            [lm_index] *
                                    (*inputs.kpoint_plus_g_components[direction_b])[g_index];
                        }
                    }
                    channel++;
                }
            }
            ensure_capacity(species.device_form_factors, species.form_factors.size());
            if (!species.form_factors.empty()) {
                device_backend_->copy(species.device_form_factors.data(),
                                      species.form_factors.data(),
                                      species.form_factors.size() * sizeof(Complex));
            }
        }
    }

    /// See ProjectorStrainOverlapCalculator::compute_block(); one small product per atom.
    void compute_block(int first_atom,
                       int end_atom,
                       int number_of_projectors,
                       const PlanewaveKohnShamQuantumSystem& quantum_system,
                       int kpoint_state_index,
                       const std::vector<int>& band_indices,
                       Complex* overlaps) override {
        block_first_atom_ = first_atom;
        block_end_atom_ = end_atom;
        block_projectors_ = number_of_projectors;
        const int number_of_bands = static_cast<int>(band_indices.size());
        const int number_of_block_atoms = end_atom - first_atom;
        if (number_of_bands == 0 || number_of_block_atoms <= 0 || number_of_projectors == 0) {
            return;
        }
        const ProjectorStrainInputs& inputs = *inputs_;
        const int number_of_plane_waves = inputs.number_of_plane_waves;
        const std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves);
        const std::size_t band_elements = plane_waves * static_cast<std::size_t>(number_of_bands);

        const Complex* device_bands = bands_as_one_matrix_gpu(quantum_system,
                                                              kpoint_state_index,
                                                              number_of_plane_waves,
                                                              band_indices,
                                                              device_backend_,
                                                              device_gathered_bands_);

        /**
         * conj(S) of each atom of the block as two real vectors, its real and its imaginary
         * part, computed on the host and uploaded in one copy.
         */
        conjugate_structure_factors_.resize(2 * plane_waves *
                                            static_cast<std::size_t>(number_of_block_atoms));
        int atom_threads = execution::threading::thread_count_for(number_of_block_atoms, 1);
#pragma omp parallel for schedule(static) num_threads(atom_threads) if (atom_threads > 1)
        for (int block_atom = 0; block_atom < number_of_block_atoms; block_atom++) {
            Real* real_part = conjugate_structure_factors_.data() +
                              static_cast<std::size_t>(2 * block_atom) * plane_waves;
            Real* imaginary_part = real_part + plane_waves;
            Vector3D atomic_position =
                crystal_structure_.convert_from_crystal_to_cartesian(first_atom + block_atom);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                Complex structure_factor = structure_factor_at(atomic_position, g_index);
                real_part[g_index] = structure_factor.real();
                imaginary_part[g_index] = -structure_factor.imag();
            }
        }
        ensure_capacity(device_conjugate_structure_factors_, conjugate_structure_factors_.size());
        device_backend_->copy(device_conjugate_structure_factors_.data(),
                              conjugate_structure_factors_.data(),
                              conjugate_structure_factors_.size() * sizeof(Real));

        // Where each atom's overlaps go in the device buffer: kinds × channels rows per band.
        std::vector<std::size_t> atom_overlap_offset(
            static_cast<std::size_t>(number_of_block_atoms));
        std::size_t number_of_overlaps = 0;
        for (int block_atom = 0; block_atom < number_of_block_atoms; block_atom++) {
            atom_overlap_offset[static_cast<std::size_t>(block_atom)] = number_of_overlaps;
            number_of_overlaps +=
                static_cast<std::size_t>(number_of_kinds *
                                         species_of(first_atom + block_atom).number_of_channels) *
                static_cast<std::size_t>(number_of_bands);
        }
        ensure_capacity(device_phased_bands_, band_elements);
        ensure_capacity(device_phased_bands_imaginary_, band_elements);
        ensure_capacity(device_overlaps_, number_of_overlaps);

        for (int block_atom = 0; block_atom < number_of_block_atoms; block_atom++) {
            const SpeciesFormFactors& species = species_of(first_atom + block_atom);
            const int number_of_form_factors = number_of_kinds * species.number_of_channels;
            if (number_of_form_factors == 0) {
                continue;
            }
            const Real* device_real_part = device_conjugate_structure_factors_.data() +
                                           static_cast<std::size_t>(2 * block_atom) * plane_waves;
            // conj(S)·ψ = Re[conj(S)]·ψ + i · Im[conj(S)]·ψ, every band at once.
            diagonalizer::planewave::gpu::launch_apply_kinetic_energy(number_of_bands,
                                                                      number_of_plane_waves,
                                                                      device_real_part,
                                                                      device_bands,
                                                                      device_phased_bands_.data());
            diagonalizer::planewave::gpu::launch_apply_kinetic_energy(
                number_of_bands,
                number_of_plane_waves,
                device_real_part + plane_waves,
                device_bands,
                device_phased_bands_imaginary_.data());
            device_blas_.axpy(number_of_bands * number_of_plane_waves,
                              Complex(0.0, 1.0),
                              device_phased_bands_imaginary_.data(),
                              device_phased_bands_.data());
            // (kinds × channels, bands) = Fᵀ · [conj(S)·ψ]; F is real, so no conjugation.
            device_blas_.gemm('T',
                              'N',
                              number_of_form_factors,
                              number_of_bands,
                              number_of_plane_waves,
                              Complex(1.0, 0.0),
                              species.device_form_factors.data(),
                              number_of_plane_waves,
                              device_phased_bands_.data(),
                              number_of_plane_waves,
                              Complex(0.0, 0.0),
                              device_overlaps_.data() +
                                  atom_overlap_offset[static_cast<std::size_t>(block_atom)],
                              number_of_form_factors);
        }
        device_backend_->synchronize();
        host_overlaps_.resize(number_of_overlaps);
        device_backend_->copy(
            host_overlaps_.data(), device_overlaps_.data(), number_of_overlaps * sizeof(Complex));

        // Into the caller's layout, with the constant each column carries.
        const std::size_t block_rows = static_cast<std::size_t>(number_of_kinds) *
                                       static_cast<std::size_t>(number_of_projectors);
        int atom_first_projector = 0;
        for (int block_atom = 0; block_atom < number_of_block_atoms; block_atom++) {
            const SpeciesFormFactors& species = species_of(first_atom + block_atom);
            const int number_of_channels = species.number_of_channels;
            const std::size_t number_of_form_factors =
                static_cast<std::size_t>(number_of_kinds * number_of_channels);
            const Complex* atom_overlaps =
                host_overlaps_.data() + atom_overlap_offset[static_cast<std::size_t>(block_atom)];
            for (int band = 0; band < number_of_bands; band++) {
                for (int kind = 0; kind < number_of_kinds; kind++) {
                    for (int channel = 0; channel < number_of_channels; channel++) {
                        overlaps[static_cast<std::size_t>(kind * number_of_projectors +
                                                          atom_first_projector + channel) +
                                 static_cast<std::size_t>(band) * block_rows] =
                            std::conj(species.prefactors[static_cast<std::size_t>(channel)]) *
                            atom_overlaps[static_cast<std::size_t>(kind * number_of_channels +
                                                                   channel) +
                                          static_cast<std::size_t>(band) * number_of_form_factors];
                    }
                }
            }
            atom_first_projector += number_of_channels;
        }
    }

    /// See ProjectorStrainOverlapCalculator::block_values_at_plane_wave(); from the factors.
    void block_values_at_plane_wave(int plane_wave_index,
                                    std::vector<Complex>& values) const override {
        values.assign(static_cast<std::size_t>(number_of_kinds * block_projectors_),
                      Complex(0.0, 0.0));
        const std::size_t plane_waves = static_cast<std::size_t>(inputs_->number_of_plane_waves);
        int atom_first_projector = 0;
        for (int atom_index = block_first_atom_; atom_index < block_end_atom_; atom_index++) {
            const SpeciesFormFactors& species = species_of(atom_index);
            Complex structure_factor = structure_factor_at(
                crystal_structure_.convert_from_crystal_to_cartesian(atom_index), plane_wave_index);
            for (int kind = 0; kind < number_of_kinds; kind++) {
                for (int channel = 0; channel < species.number_of_channels; channel++) {
                    Real form_factor =
                        species
                            .form_factors[static_cast<std::size_t>(
                                              kind * species.number_of_channels + channel) *
                                              plane_waves +
                                          static_cast<std::size_t>(plane_wave_index)]
                            .real();
                    values[static_cast<std::size_t>(kind * block_projectors_ +
                                                    atom_first_projector + channel)] =
                        species.prefactors[static_cast<std::size_t>(channel)] * form_factor *
                        structure_factor;
                }
            }
            atom_first_projector += species.number_of_channels;
        }
    }

  private:
    /// One species' form factors at the present k-point, on the host and on the device.
    struct SpeciesFormFactors {
        explicit SpeciesFormFactors(
            const std::shared_ptr<execution::gpu::device::GPUDeviceBackend>& device_backend)
            : device_form_factors(0, device_backend) {}

        /// Channels of the species: its beta functions, each with its 2l+1 components.
        int number_of_channels = 0;
        /// (1/√Ω)(-i)^l per channel.
        std::vector<Complex> prefactors;
        /**
         * plane waves × (kinds × channels), column (kind, channel) at kind × channels + channel.
         * Real values held as complex numbers, which is what the device product takes.
         */
        std::vector<Complex> form_factors;
        execution::gpu::device::GPUDeviceBuffer<Complex> device_form_factors;
    };

    [[nodiscard]] static std::shared_ptr<execution::gpu::device::GPUDeviceBackend>
    require(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend) {
        if (!device_backend) {
            throw std::invalid_argument(
                "GPUProjectorStrainOverlapCalculator: the device backend must not be null");
        }
        return device_backend;
    }

    [[nodiscard]] const SpeciesFormFactors& species_of(int atom_index) const {
        return species_[static_cast<std::size_t>(
            crystal_structure_.species_type_index(atom_index))];
    }

    /// exp(-2πi (k+G)·r) of an atom at one plane wave of the present k-point.
    [[nodiscard]] Complex structure_factor_at(const Vector3D& atomic_position,
                                              int plane_wave_index) const {
        const auto& g_vector = planewave_basis_.g_vectors()[(
            *inputs_->plane_wave_indices)[static_cast<std::size_t>(plane_wave_index)]];
        Real phase = constants::two_pi * (g_vector.g_cart.dot(atomic_position) +
                                          inputs_->kpoint_cartesian.dot(atomic_position));
        return compute_phase_factor(phase);
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
    const ProjectorStrainInputs* inputs_ = nullptr;
    std::vector<SpeciesFormFactors> species_;
    /// The block computed last, for block_values_at_plane_wave().
    int block_first_atom_ = 0;
    int block_end_atom_ = 0;
    int block_projectors_ = 0;
    std::vector<Real> conjugate_structure_factors_;
    std::vector<Complex> host_overlaps_;
    execution::gpu::device::GPUDeviceBuffer<Complex> device_gathered_bands_;
    execution::gpu::device::GPUDeviceBuffer<Real> device_conjugate_structure_factors_;
    execution::gpu::device::GPUDeviceBuffer<Complex> device_phased_bands_;
    execution::gpu::device::GPUDeviceBuffer<Complex> device_phased_bands_imaginary_;
    execution::gpu::device::GPUDeviceBuffer<Complex> device_overlaps_;
};

} // namespace q3
