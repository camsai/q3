#pragma once

/**
 * ChargeDensity — abstract charge density.
 *
 * A scalar field property of a quantum system.
 * Pure data container. Computation is done by calculators in layer 4.
 */

#include "layer_0_core/types/init.hpp"
#include "layer_2_quantum_system/core/spin_resolved_entity.hpp"

namespace q3 {

/// Abstract charge density — a scalar field property of a quantum system.
class ChargeDensity : public SpinResolvedEntity {
  public:
    virtual ~ChargeDensity() = default;

    [[nodiscard]] virtual Real get_total_charge() const = 0;
    virtual void reset() = 0;
};

} // namespace q3
