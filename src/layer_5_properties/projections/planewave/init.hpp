#pragma once

/**
 * PlanewaveWavefunctionProjectionCalculator — atomic-orbital projections for PDOS / Löwdin
 * population analysis.
 *
 * Ties together the shared atomic-orbital G-space construction
 * (atomic_orbital_builder.hpp) and Löwdin orthogonalization
 * (lowdin_orthogonalization.hpp) into the actual projection weights QE's
 * projwfc.x calls "proj": for each k-point, each band, and each ortho-atomic
 * orbital i, computes |⟨φ'_i|S|ψ_nk⟩|². These weights are exactly what's
 * needed for PDOS energy binning (Python side, mirroring q3.workflows.dos's
 * existing Gaussian broadening) and Löwdin charges (Σ_nk f_nk·w_k·proj_i).
 *
 * S is the overlap operator of the generalized eigenproblem the run solves.
 * It is the identity for norm-conserving pseudopotentials, where the two
 * metrics coincide and the weight is the plain |⟨φ'_i|ψ_nk⟩|². It is not for
 * ultrasoft and PAW: those relax norm conservation, so charge lives in the
 * augmentation term S = 1 + Σ_ij q_ij |β_i⟩⟨β_j| as well as in the pseudo
 * wavefunction, the Kohn-Sham states are S-orthonormal rather than
 * orthonormal, and a projection taken in the plain metric conserves no charge
 * -- the Löwdin charges stop summing to the valence electron count
 * (Vanderbilt, PRB 41, 7892 (1990); Blöchl, PRB 50, 17953 (1994); Kresse &
 * Joubert, PRB 59, 1758 (1999)). The atomic orbitals are Löwdin-orthogonalized
 * in the same metric, ⟨φ_i|S|φ_j⟩, for the same reason. This is the convention
 * QE's projwfc.x projects in, and the one Q3's own ortho-atomic DFT+U
 * projectors already use.
 *
 * Which metric a run needs is not this class's decision: it is handed the
 * operator (null for S = I), because building the ultrasoft one needs the live
 * SCF state and a k-point's Kleinman-Bylander projector matrix. The same
 * split the Wannier trial-orbital projections use.
 *
 * One instance is built once per calculation (tabulates the atomic-orbital
 * radial table), then queried per k-point -- the atomic orbitals and their
 * Löwdin orthogonalization are k-point-dependent (different structure
 * factors / |k+G|), so must be rebuilt per k-point, unlike the projection
 * weights which are the whole point of calling this per k-point.
 */

#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/orbital_metric.hpp"

namespace q3 {

class PlanewaveWavefunctionProjectionCalculator {
  public:
    PlanewaveWavefunctionProjectionCalculator(const Crystal& crystal_structure,
                                              const PseudopotentialSet& pseudopotential_set)
        : orbital_labels_(enumerate_atomic_orbitals(crystal_structure, pseudopotential_set)) {
        atomic_wavefunction_table_ = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure.cell_volume(), q_spacing_, number_of_q_points_);
    }

    /**
     * Stable per-orbital (atom, shell, l, m) labels, in the same order as every
     * projection row returned below.
     */
    [[nodiscard]] const std::vector<AtomicOrbitalLabel>& orbital_labels() const {
        return orbital_labels_;
    }

    [[nodiscard]] int number_of_orbitals() const {
        return static_cast<int>(orbital_labels_.size());
    }

    /**
     * Compute |<phi'_i|S|psi_nk>|^2 for one k-point: projections[band_index][orbital_index].
     * `kpoint_state_index` indexes quantum_system's flattened (spin, kpoint) storage,
     * matching PlanewaveKohnShamQuantumSystem::wavefunction_band_data's convention.
     * `overlap_operator` is null for norm-conserving pseudopotentials (S = I);
     * ultrasoft and PAW callers pass create_overlap_operator()'s result for this
     * k-point.
     */
    [[nodiscard]] std::vector<std::vector<Real>>
    compute_projections_at_kpoint(int kpoint_index,
                                  int kpoint_state_index,
                                  int number_of_plane_waves,
                                  int number_of_bands,
                                  const std::vector<int>& g_vector_indices,
                                  const Crystal& crystal_structure,
                                  const KPointsGrid& kpoint_grid,
                                  const PlanewaveBasis& planewave_basis,
                                  const PseudopotentialSet& pseudopotential_set,
                                  const PlanewaveKohnShamQuantumSystem& quantum_system,
                                  const OverlapOperator* overlap_operator) const {
        std::vector<std::vector<Complex>> orbitals;
        build_atomic_orbitals_g_space(kpoint_index,
                                      number_of_plane_waves,
                                      g_vector_indices,
                                      crystal_structure,
                                      kpoint_grid,
                                      planewave_basis,
                                      pseudopotential_set,
                                      atomic_wavefunction_table_,
                                      q_spacing_,
                                      number_of_q_points_,
                                      orbitals);
        lowdin_orthogonalize_orbitals(
            orbitals, compute_orbital_overlap_matrix_in_s_metric(orbitals, overlap_operator));

        /**
         * S is Hermitian, so <phi'_i|S|psi> = <S phi'_i|psi>: applying it once
         * to the orthogonalized orbitals leaves the band loop below the same
         * contraction it performs when S is the identity, rather than a second
         * operator application per band.
         */
        std::vector<std::vector<Complex>> orbitals_in_s_metric =
            apply_overlap_operator_to_orbitals(orbitals, overlap_operator);

        int orbital_count = static_cast<int>(orbitals.size());
        std::vector<std::vector<Real>> projections(number_of_bands,
                                                   std::vector<Real>(orbital_count, 0.0));

        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            const Complex* band_data =
                quantum_system.wavefunction_band_data(kpoint_state_index, band_index);
            for (int orbital_index = 0; orbital_index < orbital_count; orbital_index++) {
                Complex overlap(0.0, 0.0);
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    overlap += std::conj(orbitals_in_s_metric[orbital_index][g_index]) *
                               band_data[g_index];
                }
                projections[band_index][orbital_index] = std::norm(overlap);
            }
        }
        return projections;
    }

  private:
    std::vector<AtomicOrbitalLabel> orbital_labels_;
    std::vector<std::vector<std::vector<Real>>> atomic_wavefunction_table_;
    Real q_spacing_ = 0.01;
    int number_of_q_points_ = 900;
};

} // namespace q3
