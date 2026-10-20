#pragma once

/**
 * EffectivePotential — abstract effective potential.
 *
 * The self-consistent field acting on electrons.
 * Pure data container. Computation is done by calculators in layer 4.
 */

#include "layer_2_quantum_system/core/spin_resolved_entity.hpp"

namespace q3 {

/// Abstract effective potential — the self-consistent field acting on electrons.
class EffectivePotential : public SpinResolvedEntity {
  public:
    virtual ~EffectivePotential() = default;
};

} // namespace q3
