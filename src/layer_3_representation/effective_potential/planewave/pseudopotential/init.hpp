#pragma once

#include "layer_3_representation/effective_potential/planewave/init.hpp"

namespace q3 {

/**
 * NormConservingPlanewaveEffectivePotential — pseudopotential effective potential.
 *
 * Base class for effective potentials under pseudopotentials in planewave basis.
 * For norm-conserving, this is identical to PlanewaveEffectivePotential since the D-matrix is
 * static. For ultrasoft, it is overridden by UltrasoftPlanewaveEffectivePotential to
 * manage D_eff.
 */
class NormConservingPlanewaveEffectivePotential : public PlanewaveEffectivePotential {
  public:
    using PlanewaveEffectivePotential::PlanewaveEffectivePotential;
    virtual ~NormConservingPlanewaveEffectivePotential() override = default;
};

} // namespace q3
