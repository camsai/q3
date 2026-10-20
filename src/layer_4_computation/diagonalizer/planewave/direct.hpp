#pragma once

/**
 * PlanewaveDirectDiagonalizer — dense LAPACK-based planewave eigenvalue solver.
 *
 * Builds the full Hamiltonian matrix H(G,G') and solves via LAPACK's
 * ZHEEV/ZHEGV. O(N_pw³) cost, but exact — used as the reference solver
 * for small systems and for validating iterative solvers.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/execution/serial.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_4_computation/diagonalizer/planewave/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/factory.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/factory.hpp"

namespace q3 {

/**
 * Dense (direct) diagonalization using LAPACK.
 * The OverlapOperator selects the standard (NC) or generalized (US) eigenproblem.
 */
class PlanewaveDirectDiagonalizer : public PlanewaveDiagonalizer {
  public:
    inline static const execution::SerialExecutionPolicy default_serial_execution_policy{};

    explicit PlanewaveDirectDiagonalizer(
        const execution::ExecutionPolicy& execution_policy = default_serial_execution_policy)
        : subspace_diagonalizer_(SubspaceDiagonalizerFactory::create(execution_policy))
        , execution_policy_(execution_policy) {}
    void solve(int kpoint_index_state,
               int kpoint_index,
               int number_of_plane_waves,
               int number_of_bands,
               const std::vector<int>& g_vector_indices,
               const std::vector<Real>& kinetic_energy_squared,
               const std::vector<Real>& real_space_potential,
               const Crystal& crystal_structure,
               const KPointsGrid& kpoint_grid,
               const PlanewaveBasis& planewave_basis,
               const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
               PlanewaveKohnShamQuantumSystem& quantum_system) override {
        // Build full Hamiltonian matrix H[i + j*number_of_plane_waves] (column-major for LAPACK)
        std::vector<Complex> hamiltonian_matrix(number_of_plane_waves * number_of_plane_waves,
                                                Complex(0.0, 0.0));

        // Kinetic energy (diagonal)
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            hamiltonian_matrix[g_index + g_index * number_of_plane_waves] =
                kinetic_energy_squared[g_index];
        }

        // Local potential via FFT
        {
            int number_of_grid_points = planewave_basis.fft_rho().total_grid_points();
            int fft_dimension_x = planewave_basis.fft_rho().grid_dimension_1();
            int fft_dimension_y = planewave_basis.fft_rho().grid_dimension_2();
            int fft_dimension_z = planewave_basis.fft_rho().grid_dimension_3();
            for (int plane_wave_index_jg = 0; plane_wave_index_jg < number_of_plane_waves;
                 plane_wave_index_jg++) {
                std::vector<Complex> psi_g(number_of_grid_points, Complex(0.0, 0.0));
                const auto& g_vector_j =
                    planewave_basis.g_vectors()[g_vector_indices[plane_wave_index_jg]];
                int miller_x_j = static_cast<int>(std::round(g_vector_j.miller.x));
                int miller_y_j = static_cast<int>(std::round(g_vector_j.miller.y));
                int miller_z_j = static_cast<int>(std::round(g_vector_j.miller.z));
                if (std::abs(miller_x_j) <= fft_dimension_x / 2 &&
                    std::abs(miller_y_j) <= fft_dimension_y / 2 &&
                    std::abs(miller_z_j) <= fft_dimension_z / 2) {
                    int fft_grid_index = planewave_basis.fft_rho().g_vector_to_fft_grid_index(
                        miller_x_j, miller_y_j, miller_z_j);
                    psi_g[fft_grid_index] = Complex(1.0, 0.0);
                }

                planewave_basis.fft_rho().inverse(psi_g.data());

                for (int grid_point_index = 0; grid_point_index < number_of_grid_points;
                     grid_point_index++) {
                    psi_g[grid_point_index] = Complex(real_space_potential[grid_point_index], 0.0) *
                                              psi_g[grid_point_index];
                }

                planewave_basis.fft_rho().forward(psi_g.data());

                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    const auto& g_vector_i = planewave_basis.g_vectors()[g_vector_indices[g_index]];
                    int miller_x_i = static_cast<int>(std::round(g_vector_i.miller.x));
                    int miller_y_i = static_cast<int>(std::round(g_vector_i.miller.y));
                    int miller_z_i = static_cast<int>(std::round(g_vector_i.miller.z));
                    if (std::abs(miller_x_i) <= fft_dimension_x / 2 &&
                        std::abs(miller_y_i) <= fft_dimension_y / 2 &&
                        std::abs(miller_z_i) <= fft_dimension_z / 2) {
                        int fft_grid_index_i = planewave_basis.fft_rho().g_vector_to_fft_grid_index(
                            miller_x_i, miller_y_i, miller_z_i);
                        hamiltonian_matrix[g_index + plane_wave_index_jg * number_of_plane_waves] +=
                            psi_g[fft_grid_index_i];
                    }
                }
            }
        }

        /**
         * Which spin channel this call belongs to; see the same derivation in
         * the Davidson solver. The nonlocal D is per channel for ultrasoft and
         * PAW, and `real_space_potential` above is already this channel's.
         */
        int spin_channel = kpoint_grid.number_of_kpoints() > 0
                               ? kpoint_index_state / kpoint_grid.number_of_kpoints()
                               : 0;

        // Nonlocal potential contribution — use cached projectors if available
        Matrix<Complex> computed_projectors;
        const Matrix<Complex>* kleinman_bylander_potential_ptr = nullptr;
        if (kpoint_index < static_cast<int>(cached_projector_map_.size()) &&
            cached_projector_map_[kpoint_index] != nullptr) {
            kleinman_bylander_potential_ptr = cached_projector_map_[kpoint_index];
        } else {
            Vector3D kpoint_coordinates = kpoint_grid[kpoint_index].coordinates;
            compute_kleinman_bylander_projectors(
                kpoint_coordinates,
                crystal_structure,
                planewave_basis,
                pseudopotential_quantum_system.pseudopotential_basis().pseudopotential_set(),
                pseudopotential_quantum_system.pseudopotential_basis().beta_function_table(),
                g_vector_indices,
                number_of_plane_waves,
                pseudopotential_quantum_system.pseudopotential_basis().total_number_of_projectors(),
                computed_projectors);
            kleinman_bylander_potential_ptr = &computed_projectors;
        }
        const auto& kleinman_bylander_potential = *kleinman_bylander_potential_ptr;

        int combined_projector_index = 0;
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            int species_type_index = crystal_structure.species_type_index(atom_index);
            int number_of_projectors =
                pseudopotential_quantum_system.pseudopotential_basis().number_of_projectors(
                    species_type_index);

            int projector_index_start = combined_projector_index;
            std::vector<int> projector_angular_momenta, projector_magnetic_projections,
                projector_beta_indices;
            for (int beta_index = 0; beta_index < number_of_projectors; beta_index++) {
                int angular_momentum =
                    pseudopotential_quantum_system.pseudopotential_basis().beta_angular_momentum(
                        species_type_index, beta_index);
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum)) {
                    projector_angular_momenta.push_back(angular_momentum);
                    projector_magnetic_projections.push_back(magnetic_projection);
                    projector_beta_indices.push_back(beta_index);
                    combined_projector_index++;
                }
            }
            int number_of_projectors_for_atom = static_cast<int>(projector_angular_momenta.size());

            for (int projector_index_a = 0; projector_index_a < number_of_projectors_for_atom;
                 projector_index_a++) {
                for (int projector_index_b = 0; projector_index_b < number_of_projectors_for_atom;
                     projector_index_b++) {
                    Real d_matrix_element_value =
                        pseudopotential_quantum_system.d_matrix_element(atom_index,
                                                                        species_type_index,
                                                                        projector_index_a,
                                                                        projector_index_b,
                                                                        spin_channel);
                    if (std::abs(d_matrix_element_value) <
                        math::tolerances::d_matrix_element_threshold)
                        continue;

                    int combined_projector_index_a = projector_index_start + projector_index_a;
                    int combined_projector_index_b = projector_index_start + projector_index_b;

                    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                        for (int plane_wave_index_jg = 0;
                             plane_wave_index_jg < number_of_plane_waves;
                             plane_wave_index_jg++) {
                            hamiltonian_matrix[g_index +
                                               plane_wave_index_jg * number_of_plane_waves] +=
                                d_matrix_element_value *
                                kleinman_bylander_potential[combined_projector_index_a][g_index] *
                                std::conj(kleinman_bylander_potential[combined_projector_index_b]
                                                                     [plane_wave_index_jg]);
                        }
                    }
                }
            }
        }

        // The overlap operator selects the standard (NC, S=I) or generalized (US) eigenproblem.
        auto overlap_operator = create_overlap_operator(crystal_structure,
                                                        pseudopotential_quantum_system,
                                                        kleinman_bylander_potential,
                                                        execution_policy_);

        std::vector<Complex> overlap_matrix;
        overlap_operator->build_dense_matrix(number_of_plane_waves, overlap_matrix);

        /**
         * Reduce Hamiltonian and overlap matrices across band group.
         * No-op when group_size == 1 (serial); MPI_Allreduce when distributed.
         */
        execution_policy_.reduce_within_group(hamiltonian_matrix.data(),
                                              number_of_plane_waves * number_of_plane_waves);
        execution_policy_.reduce_within_group(overlap_matrix.data(),
                                              number_of_plane_waves * number_of_plane_waves);

        std::vector<Real> lapack_eigenvalues(number_of_plane_waves);
        std::vector<Complex> eigenvectors(number_of_plane_waves * number_of_plane_waves);

        /**
         * Direct diagonalizer solves the FULL matrix problem.
         * We reuse the SubspaceDiagonalizer interface here as well.
         * NOTE: always pass is_gamma_only=false because the direct diag builds a standard
         * complex Hermitian H matrix via explicit G-space convolution — the imaginary
         * part of H[i,j] = V(G_i - G_j) is non-zero even for gamma-only calculations.
         * The real-symmetric solver (dsygv) would discard the imaginary part and produce
         * incorrect eigenvalues.
         */
        subspace_diagonalizer_->solve(number_of_plane_waves,
                                      hamiltonian_matrix.data(),
                                      overlap_matrix.data(),
                                      lapack_eigenvalues.data(),
                                      eigenvectors.data(),
                                      false);

        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            quantum_system.set_eigenvalue(
                kpoint_index_state, band_index, lapack_eigenvalues[band_index]);
            Complex* band_data =
                quantum_system.wavefunction_band_data(kpoint_index_state, band_index);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                band_data[g_index] = eigenvectors[band_index * number_of_plane_waves + g_index];
            }
        }
    }

    [[nodiscard]] std::string name() const override {
        return "direct";
    }

    void set_cached_projectors(int kpoint_index, const q3::Matrix<Complex>* projectors) override {
        if (kpoint_index >= static_cast<int>(cached_projector_map_.size())) {
            cached_projector_map_.resize(kpoint_index + 1, nullptr);
        }
        cached_projector_map_[kpoint_index] = projectors;
    }

  private:
    std::unique_ptr<SubspaceDiagonalizer> subspace_diagonalizer_;
    const execution::ExecutionPolicy& execution_policy_;
    std::vector<const q3::Matrix<Complex>*> cached_projector_map_;
};

} // namespace q3
