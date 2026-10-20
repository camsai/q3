#pragma once

/**
 * PlanewaveKohnShamChargeDensityCheckpoint — planewave-level base.
 *
 * Knows about planewave representation (G-vectors, Miller indices)
 * but not about the specific binary format (QE, HDF5, etc.)
 */

#include "io/charge_density/kohn_sham/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

class PlanewaveKohnShamChargeDensityCheckpoint : public KohnShamChargeDensityCheckpoint {
  public:
    explicit PlanewaveKohnShamChargeDensityCheckpoint(
        const NormConservingPlanewaveKohnShamSCFState& scf_state)
        : scf_state_(scf_state) {}

  protected:
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
};

} // namespace q3
