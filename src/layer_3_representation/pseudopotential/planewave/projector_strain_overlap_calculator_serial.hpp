#pragma once

/**
 * SerialProjectorStrainOverlapCalculator — projector strain overlaps formed from
 * columns built on the host.
 *
 * For a block of atoms it builds the block's projectors and their six strain
 * derivatives as columns over the plane waves, seven kinds side by side in one
 * array, and contracts all of them with the bands in one product through a
 * BandProjectionCalculator.
 *
 * The array is one workspace for the whole calculation, grown to the largest
 * block and never shrunk. A block carries a limited number of projectors (the
 * caller chooses how many) so that its seven arrays stay below the bands: the
 * whole cell at once is what took a 54-atom silicon cell from 2.3 GiB through
 * the SCF to over 4 GiB in the stress.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "layer_0_core/execution/gpu/device/device_mirrored_array.hpp"
#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_strain_overlap_calculator.hpp"

namespace q3 {

/// ProjectorStrainOverlapCalculator that builds its columns on the host; see the file comment.
class SerialProjectorStrainOverlapCalculator : public ProjectorStrainOverlapCalculator {
  public:
    /// The crystal, the bases and the band projection calculator must outlive this object.
    SerialProjectorStrainOverlapCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveBasis& pseudopotential_basis,
        const BandProjectionCalculator& band_projections)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_basis_(pseudopotential_basis)
        , band_projections_(band_projections) {}

    /// See ProjectorStrainOverlapCalculator::begin_kpoint(); makes room for the largest block.
    void begin_kpoint(const ProjectorStrainInputs& inputs,
                      int maximum_projectors_per_block) override {
        inputs_ = &inputs;
        std::size_t number_of_elements = static_cast<std::size_t>(number_of_kinds) *
                                         static_cast<std::size_t>(maximum_projectors_per_block) *
                                         static_cast<std::size_t>(inputs.number_of_plane_waves);
        if (columns_.size() < number_of_elements) {
            columns_.resize(number_of_elements);
        }
        block_rows_ = 0;
    }

    /// See ProjectorStrainOverlapCalculator::compute_block(); columns on the host, one product.
    void compute_block(int first_atom,
                       int end_atom,
                       int number_of_projectors,
                       const PlanewaveKohnShamQuantumSystem& quantum_system,
                       int kpoint_state_index,
                       const std::vector<int>& band_indices,
                       Complex* overlaps) override {
        const auto& pseudopotential_basis = pseudopotential_basis_;
        const ProjectorStrainInputs& geometry = *inputs_;
        const std::vector<int>& plane_wave_indices_kpoint = *geometry.plane_wave_indices;
        const std::array<std::vector<Real>, 6>& radial_strain_weights =
            *geometry.radial_strain_weights;
        const Vector3D& kpoint_coordinates_cartesian = geometry.kpoint_cartesian;
        int number_of_plane_waves_kpoint = geometry.number_of_plane_waves;
        Real inverse_sqrt_cell_volume = 1.0 / std::sqrt(crystal_structure_.cell_volume());
        Real inverse_two_pi_over_lattice_parameter =
            1.0 / crystal_structure_.two_pi_over_lattice_parameter();
        std::size_t plane_waves = static_cast<std::size_t>(number_of_plane_waves_kpoint);

        /**
         * Where each atom's projectors start within the block, so that the atoms can be
         * built side by side: every column belongs to one atom, and every value in it is
         * computed from that atom's inputs alone, so sharing the atoms out over threads
         * changes no bit of the result. Serial, this loop was most of the stress on a
         * large cell: seven columns per projector, each as long as the plane waves.
         */
        int number_of_block_atoms = end_atom - first_atom;
        std::vector<int> atom_first_projector(static_cast<std::size_t>(number_of_block_atoms));
        {
            int projector_count = 0;
            for (int block_atom = 0; block_atom < number_of_block_atoms; block_atom++) {
                atom_first_projector[static_cast<std::size_t>(block_atom)] = projector_count;
                projector_count += pseudopotential_basis.number_of_projector_channels(
                    crystal_structure_.species_type_index(first_atom + block_atom));
            }
        }

        // Written on the host: a device copy of the workspace is stale from here on.
        Complex* columns = columns_.mutable_host();
        int atom_threads = execution::threading::thread_count_for(number_of_block_atoms, 1);
#pragma omp parallel for schedule(static) num_threads(atom_threads) if (atom_threads > 1)
        for (int block_atom = 0; block_atom < number_of_block_atoms; block_atom++) {
            int atom_index = first_atom + block_atom;
            std::vector<Complex> structure_factor(number_of_plane_waves_kpoint);
            std::vector<Real> beta_derivative_at_g(number_of_plane_waves_kpoint);
            std::vector<Real> beta_at_g(number_of_plane_waves_kpoint);
            int projector_index = atom_first_projector[static_cast<std::size_t>(block_atom)];
            int index_type = crystal_structure_.species_type_index(atom_index);
            Vector3D atomic_position =
                crystal_structure_.convert_from_crystal_to_cartesian(atom_index);
            for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
                const auto& g_vector =
                    planewave_basis_.g_vectors()[plane_wave_indices_kpoint[g_index]];
                Real phase =
                    constants::two_pi * (g_vector.g_cart.dot(atomic_position) +
                                         kpoint_coordinates_cartesian.dot(atomic_position));
                structure_factor[g_index] = compute_phase_factor(phase);
            }
            for (int index_beta = 0;
                 index_beta < pseudopotential_basis.number_of_projectors(index_type);
                 index_beta++) {
                int angular_momentum =
                    pseudopotential_basis.beta_angular_momentum(index_type, index_beta);
                Complex prefactor =
                    inverse_sqrt_cell_volume * compute_angular_phase_factor(angular_momentum);
                for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
                    beta_derivative_at_g[g_index] = pseudopotential_basis.interpolate_dbeta(
                        index_type, index_beta, (*geometry.magnitude_bohr)[g_index]);
                    beta_at_g[g_index] = pseudopotential_basis.interpolate_beta(
                        index_type, index_beta, (*geometry.magnitude_bohr)[g_index]);
                }
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    int lm_index =
                        harmonic_combined_degree_order_index(angular_momentum, magnetic_projection);
                    Complex* projector =
                        columns + static_cast<std::size_t>(projector_index) * plane_waves;
                    std::array<Complex*, 6> strain_projector;
                    for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                        strain_projector[strain_pair] =
                            columns +
                            static_cast<std::size_t>((strain_pair + 1) * number_of_projectors +
                                                     projector_index) *
                                plane_waves;
                    }
                    for (int g_index = 0; g_index < number_of_plane_waves_kpoint; g_index++) {
                        Complex phased = structure_factor[g_index] * prefactor;
                        Real spherical_harmonic =
                            (*geometry.spherical_harmonics)[g_index][lm_index];
                        projector[g_index] = beta_at_g[g_index] * spherical_harmonic * phased;
                        Complex radial =
                            beta_derivative_at_g[g_index] * spherical_harmonic * phased;
                        /**
                         * β divided by 2π/a to match the unit convention of the
                         * spherical-harmonic derivatives, which are in (2π/a)⁻¹.
                         */
                        Complex angular_scale =
                            beta_at_g[g_index] * inverse_two_pi_over_lattice_parameter * phased;
                        for (int strain_pair = 0; strain_pair < 6; strain_pair++) {
                            int direction_a = strain_direction_a[strain_pair];
                            int direction_b = strain_direction_b[strain_pair];
                            strain_projector[strain_pair][g_index] =
                                radial * radial_strain_weights[strain_pair][g_index] +
                                angular_scale *
                                    (*geometry.spherical_harmonic_derivatives)[direction_a][g_index]
                                                                              [lm_index] *
                                    (*geometry.kpoint_plus_g_components[direction_b])[g_index];
                        }
                    }
                    projector_index++;
                }
            }
        }

        /**
         * All seven kinds against the bands in one product:
         * overlaps(7·Nproj, Nbands) = columns†(7·Nproj, NPW) Ψ(NPW, Nbands).
         */
        block_rows_ = number_of_kinds * number_of_projectors;
        band_projections_.project(columns_,
                                  block_rows_,
                                  number_of_plane_waves_kpoint,
                                  quantum_system,
                                  kpoint_state_index,
                                  band_indices,
                                  overlaps);
    }

    /// See ProjectorStrainOverlapCalculator::block_values_at_plane_wave(); read from the columns.
    void block_values_at_plane_wave(int plane_wave_index,
                                    std::vector<Complex>& values) const override {
        values.resize(static_cast<std::size_t>(block_rows_));
        const Complex* columns = columns_.host();
        const std::size_t plane_waves = static_cast<std::size_t>(inputs_->number_of_plane_waves);
        for (int row = 0; row < block_rows_; row++) {
            values[static_cast<std::size_t>(row)] =
                columns[static_cast<std::size_t>(row) * plane_waves +
                        static_cast<std::size_t>(plane_wave_index)];
        }
    }

  private:
    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const NormConservingPlanewaveBasis& pseudopotential_basis_;
    const BandProjectionCalculator& band_projections_;
    const ProjectorStrainInputs* inputs_ = nullptr;
    /// The block's projectors and strain derivatives, seven kinds of column side by side.
    execution::gpu::device::DeviceMirroredArray<Complex> columns_;
    /// Rows of the block computed last: number_of_kinds times its projectors.
    int block_rows_ = 0;
};

} // namespace q3
