#pragma once

/**
 * PlanewaveTrialOrbitalProjectionCalculator — the A_mn(k) = <psi_mk|S|g_n>
 * projections a Wannier initial guess needs, and the S-metric Löwdin
 * orthogonalization of the trial orbitals that makes projectability-based
 * disentanglement meaningful (p_nk = sum_m |A_mn(k)|^2 in [0,1] only when
 * the trial orbitals are orthonormal in the S metric: O_ij = <g_i|S|g_j>).
 *
 * Where the trial orbitals come from is not this class's concern: it is
 * handed a PlanewaveLocalOrbitalSet -- the pseudopotentials' atomic
 * wavefunctions, hydrogenic orbitals at explicit centres, anything else that
 * renders itself in plane waves -- and does the same two things to any of
 * them. That is what lets the interface builder have one path instead of
 * one per orbital source.
 *
 * Differs from the PDOS / Löwdin population analysis in one way that
 * matters for Wannier functions: it returns the complex projection amplitude
 * rather than its squared modulus, because a Wannier initial guess needs the
 * phase and PDOS does not. The S metric itself is shared -- both hand the
 * same overlap operator to orbital_metric.hpp.
 */

#include <complex>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/orbital_metric.hpp"

namespace q3 {

class PlanewaveTrialOrbitalProjectionCalculator {
  public:
    explicit PlanewaveTrialOrbitalProjectionCalculator(
        const PlanewaveLocalOrbitalSet& trial_orbitals)
        : trial_orbitals_(trial_orbitals) {}

    [[nodiscard]] int number_of_orbitals() const {
        return trial_orbitals_.number_of_orbitals();
    }

    /**
     * Compute A_mn(k) = <psi_mk|S|g'_n>, where g'_n is the trial-orbital
     * set Löwdin-orthogonalized in the S metric: projections[band_index]
     * [orbital_index]. `overlap_operator` is null for norm-conserving
     * pseudopotentials (S = I); ultrasoft/PAW callers pass
     * create_overlap_operator()'s result, built from the live SCF state
     * at this k-point (the projector matrix it needs is k-point
     * dependent, so it cannot be cached across k-points inside this
     * calculator).
     */
    [[nodiscard]] std::vector<std::vector<Complex>>
    compute_projections_at_kpoint(int kpoint_index,
                                  int kpoint_state_index,
                                  int number_of_plane_waves,
                                  int number_of_bands,
                                  const std::vector<int>& g_vector_indices,
                                  const Crystal& crystal_structure,
                                  const KPointsGrid& kpoint_grid,
                                  const PlanewaveBasis& planewave_basis,
                                  const PlanewaveKohnShamQuantumSystem& quantum_system,
                                  const OverlapOperator* overlap_operator,
                                  int first_band_index = 0) const {
        std::vector<std::vector<Complex>> orbitals;
        trial_orbitals_.build(kpoint_index,
                              number_of_plane_waves,
                              g_vector_indices,
                              crystal_structure,
                              kpoint_grid,
                              planewave_basis,
                              orbitals);

        std::vector<Complex> overlap_in_s_metric =
            compute_orbital_overlap_matrix_in_s_metric(orbitals, overlap_operator);
        lowdin_orthogonalize_orbitals(orbitals, overlap_in_s_metric);

        int orbital_count = static_cast<int>(orbitals.size());
        std::vector<std::vector<Complex>> s_applied_orbitals =
            apply_overlap_operator_to_orbitals(orbitals, overlap_operator);

        std::vector<std::vector<Complex>> projections(
            number_of_bands, std::vector<Complex>(orbital_count, Complex(0.0, 0.0)));

        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            const Complex* band_data = quantum_system.wavefunction_band_data(
                kpoint_state_index, first_band_index + band_index);
            for (int orbital_index = 0; orbital_index < orbital_count; orbital_index++) {
                Complex overlap(0.0, 0.0);
                const std::vector<Complex>& s_applied_orbital = s_applied_orbitals[orbital_index];
                for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                    overlap += std::conj(band_data[g_index]) * s_applied_orbital[g_index];
                }
                projections[band_index][orbital_index] = overlap;
            }
        }
        return projections;
    }

  private:
    const PlanewaveLocalOrbitalSet& trial_orbitals_;
};

} // namespace q3
