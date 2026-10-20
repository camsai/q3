#pragma once

/**
 * Diagonalizer — abstract interface for eigenvalue solvers.
 *
 * Provides the generic base for all eigensolvers. Planewave-specific
 * solve() methods, cached projector management, and smooth potential
 * cache invalidation are declared in the PlanewaveDiagonalizer subclass.
 */

#include <string>

#include "layer_0_core/types/init.hpp"

namespace q3 {

// Abstract interface for eigenvalue solvers.
class Diagonalizer {
  public:
    virtual ~Diagonalizer() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * Set the convergence tolerance for iterative eigensolvers.
     * No-op for direct solvers; overridden by Davidson and similar.
     */
    virtual void set_tolerance(Real /*tolerance*/) {}

    /**
     * Set whether this is the first SCF iteration.
     * No-op by default; overridden by solvers like Davidson to control convergence checks.
     */
    virtual void set_first_iteration(bool /*is_first_iteration*/) {}
};

} // namespace q3
