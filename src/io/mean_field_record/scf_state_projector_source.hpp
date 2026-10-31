#pragma once

/**
 * ScfStateMeanFieldProjectorSource — Kleinman-Bylander projectors of a live SCF state.
 *
 * Builds <k+G|β> with compute_kleinman_bylander_projectors(), the routine the
 * Hamiltonian applicator itself calls, so the projectors a `VKB` file carries
 * are the ones the eigenvalues in `WFN` were computed with. One k-point is
 * cached at a time: the writer walks the k-points in order and needs each once.
 */

#include "io/mean_field_record/projector_source.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

/// MeanFieldProjectorSource over a live SCF state; see the file comment.
class ScfStateMeanFieldProjectorSource : public MeanFieldProjectorSource {
  public:
    /// Holds a reference, as ScfStateMeanFieldBandSource does: the state outlives the record.
    explicit ScfStateMeanFieldProjectorSource(
        const NormConservingPlanewaveKohnShamSCFState& scf_state)
        : scf_state_(scf_state) {}

    /// See MeanFieldProjectorSource::projectors(); recomputed only when the k-point changes.
    [[nodiscard]] const Matrix<Complex>& projectors(int kpoint_index) const override {
        if (kpoint_index != cached_kpoint_index_) {
            const auto& pseudopotential_basis = scf_state_.quantum_system().pseudopotential_basis();
            const PlanewaveBasis& planewave_basis = *scf_state_.planewave_basis;
            compute_kleinman_bylander_projectors(
                (*scf_state_.kpoint_grid)[kpoint_index].coordinates,
                *scf_state_.crystal_structure,
                planewave_basis,
                pseudopotential_basis.pseudopotential_set(),
                pseudopotential_basis.beta_function_table(),
                planewave_basis.kpoint_g_vector_indices()[kpoint_index],
                planewave_basis.kpoint_number_of_plane_waves()[kpoint_index],
                pseudopotential_basis.total_number_of_projectors(),
                cached_projectors_);
            cached_kpoint_index_ = kpoint_index;
        }
        return cached_projectors_;
    }

  private:
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    mutable Matrix<Complex> cached_projectors_;
    mutable int cached_kpoint_index_ = -1;
};

} // namespace q3
