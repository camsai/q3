#pragma once

/**
 * SubspaceDiagonalizerFactory — chooses the best subspace solver.
 *
 * Automatically selects SerialSubspaceDiagonalizer for serial execution
 * and MPISubspaceDiagonalizer (ScaLAPACK) for parallel execution.
 *
 * Dispatch is based on the runtime group_size() of the execution policy.
 * The MPI subspace header self-guards with Q3_WITH_MPI, so this factory
 * includes it unconditionally. In non-MPI builds, MPISubspaceDiagonalizer
 * is not defined and the MPI branch is unreachable (group_size() is always 1).
 */

#include "layer_0_core/execution/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/mpi.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/serial.hpp"

#include <memory>
#include <stdexcept>

namespace q3 {

class SubspaceDiagonalizerFactory {
  public:
    static std::unique_ptr<SubspaceDiagonalizer>
    create(const execution::ExecutionPolicy& execution_policy) {
        if (execution_policy.group_size() > 1) {
            return create_mpi_subspace_diagonalizer(execution_policy);
        }
        return std::make_unique<SerialSubspaceDiagonalizer>();
    }

  private:
    /**
     * Create an MPI subspace diagonalizer from the execution policy.
     *
     * This is a separate function so that the MPISubspaceDiagonalizer type
     * reference is isolated behind the Q3_WITH_MPI self-guard in mpi.hpp.
     * In non-MPI builds, group_size() is always 1 so this is never called,
     * and the linker never needs to resolve the MPI symbols.
     */
    static std::unique_ptr<SubspaceDiagonalizer> create_mpi_subspace_diagonalizer(
        [[maybe_unused]] const execution::ExecutionPolicy& execution_policy) {
#ifdef Q3_WITH_MPI
        const auto& mpi_policy =
            dynamic_cast<const execution::MPIExecutionPolicy&>(execution_policy);
        return std::make_unique<MPISubspaceDiagonalizer>(mpi_policy);
#else
        throw std::runtime_error(
            "group_size() > 1 requires MPI support, but Q3 was built without Q3_WITH_MPI.");
#endif
    }
};

} // namespace q3
