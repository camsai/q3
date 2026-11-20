#pragma once

/**
 * ScfStateMeanFieldProjectorSource — Kleinman-Bylander projectors of a live SCF state.
 *
 * Reads <k+G|β> from the SCF state's projector sets, the ones the Hamiltonian
 * applicator itself read, so the projectors a `VKB` file carries are the ones
 * the eigenvalues in `WFN` were computed with. The record hands projectors out
 * as a matrix of its own, so one k-point's set is copied into one at a time:
 * the writer walks the k-points in order and needs each once.
 */

#include <algorithm>
#include <cstddef>

#include "io/mean_field_record/projector_source.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
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
            const KleinmanBylanderProjectorSet& projector_set =
                scf_state_.projector_set(kpoint_index);
            const std::size_t number_of_projectors =
                static_cast<std::size_t>(projector_set.number_of_projectors());
            const std::size_t number_of_plane_waves =
                static_cast<std::size_t>(projector_set.number_of_plane_waves());
            cached_projectors_.resize(number_of_projectors, number_of_plane_waves);
            std::copy(projector_set.host(),
                      projector_set.host() + number_of_projectors * number_of_plane_waves,
                      cached_projectors_.data());
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
