#pragma once

/**
 * KohnShamChargeDensityCheckpoint — Kohn-Sham level charge density checkpoint.
 *
 * Knows about KS-level concepts (spin components, SCF convergence)
 * but not about the basis representation.
 */

#include "io/charge_density/init.hpp"

namespace q3 {

class KohnShamChargeDensityCheckpoint : public ChargeDensityCheckpoint {
  public:
    ~KohnShamChargeDensityCheckpoint() override = default;
};

} // namespace q3
