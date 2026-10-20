#pragma once

/**
 * PlanewaveKohnShamWavefunctionCheckpoint — planewave-level base.
 *
 * Knows about planewave representation (G-vectors, k-points, bands)
 * but not about the specific binary format.
 */

#include "io/wavefunction/kohn_sham/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

class PlanewaveKohnShamWavefunctionCheckpoint : public KohnShamWavefunctionCheckpoint {
  public:
    explicit PlanewaveKohnShamWavefunctionCheckpoint(
        const NormConservingPlanewaveKohnShamSCFState& scf_state)
        : scf_state_(scf_state)
        , crystal_structure_(*scf_state.crystal_structure)
        , kpoint_grid_(*scf_state.kpoint_grid) {}

  protected:
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
    const Crystal& crystal_structure_;
    const KPointsGrid& kpoint_grid_;
};

} // namespace q3
