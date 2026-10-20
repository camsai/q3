#pragma once

/**
 * create_planewave_diagonalizer — factory function for planewave eigenvalue solvers.
 *
 * Supported solvers: "davidson" (iterative, O(N_pw·N_bands²)) and
 * "direct" (dense LAPACK, O(N_pw³), default for small systems).
 *
 * An ExecutionPolicy is injected into the diagonalizer to provide
 * band-group collective operations (dot_product, norm) — serial BLAS
 * in serial builds; BLAS + MPI_Allreduce in MPI builds.
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/execution/serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/davidson.hpp"
#include "layer_4_computation/diagonalizer/planewave/direct.hpp"
#include "layer_4_computation/diagonalizer/planewave/init.hpp"

namespace q3 {

/**
 * Create a diagonalizer by name with an optional execution policy.
 *
 * @param name             "direct" or "davidson".
 * @param execution_policy Band-group collective operations policy.
 *                         Defaults to a static serial policy for backward compatibility.
 */
inline std::unique_ptr<PlanewaveDiagonalizer>
create_planewave_diagonalizer(const std::string& name,
                              const execution::ExecutionPolicy& execution_policy =
                                  PlanewaveDavidsonDiagonalizer::default_serial_execution_policy) {
    if (name == "davidson") {
        return std::make_unique<PlanewaveDavidsonDiagonalizer>(execution_policy);
    }
    if (name == "direct") {
        return std::make_unique<PlanewaveDirectDiagonalizer>(execution_policy);
    }
    throw std::runtime_error("Unknown diagonalizer: '" + name +
                             "'. Supported values: 'direct', 'davidson'.");
}

} // namespace q3
