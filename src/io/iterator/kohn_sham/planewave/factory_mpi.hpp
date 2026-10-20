#pragma once

/**
 * create_scf_iterator_mpi — MPI SCF iterator factory.
 *
 * Two overloads:
 *
 *   1. create_scf_iterator_mpi(input)
 *      Constructs a PlanewaveMPIExecutionPolicy internally (convenience form).
 *      MPI is initialized inside the factory, AFTER parse_input in main().
 *      Preserved for backward compatibility.
 *
 *   2. create_scf_iterator_mpi(input, policy)
 *      Accepts an already-constructed, already-configured policy.
 *      Use this overload from main() to ensure MPI_Init fires before any
 *      other code (including parse_input), so that:
 *        - log::init_mpi() runs immediately and silences non-root ranks
 *        - MPI_Finalize is guaranteed by policy's destructor even if
 *          parse_input throws
 *
 * This is the entry point for multi-process MPI runs. The only io/ files
 * that touch <mpi.h> are this one (transitively via PlanewaveMPIExecutionPolicy)
 * and io/logger_mpi.hpp (directly, for the rank query).
 */

#include <memory>

#include "io/input/data.hpp"
#include "io/logger_mpi.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/mpi.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/init.hpp"

namespace q3 {

/**
 * Create an MPI-capable Kohn-Sham SCF iterator from an already-initialized policy.
 *
 * This overload is preferred from main(): construct PlanewaveMPIExecutionPolicy
 * before parse_input so that MPI_Init fires first and log::init_mpi() silences
 * non-root ranks before any output is produced.
 *
 * @param input   Parsed input parameters.
 * @param policy  Fully constructed, topology-configured execution policy.
 * @return        A fully initialized, execution-policy-owning SCF iterator.
 */
inline std::unique_ptr<PlanewaveKohnShamSCFIterator>
create_scf_iterator_mpi(const InputData& input,
                        std::unique_ptr<execution::PlanewaveMPIExecutionPolicy> policy) {
    const execution::PlanewaveMPIExecutionPolicy& policy_ref = *policy;
    std::unique_ptr<PlanewaveKohnShamSCFIterator> iterator =
        create_kohn_sham_iterator(input, policy_ref);
    iterator->set_owned_execution_policy(std::move(policy));
    return iterator;
}

/**
 * Create an MPI-capable Kohn-Sham SCF iterator from parsed InputData.
 *
 * Convenience overload: constructs PlanewaveMPIExecutionPolicy internally,
 * calls MPI_Init and log::init_mpi(), then delegates to the shared factory.
 *
 * @param input  Parsed input parameters.
 * @return       A fully initialized, execution-policy-owning SCF iterator.
 */
inline std::unique_ptr<PlanewaveKohnShamSCFIterator>
create_scf_iterator_mpi(const InputData& input) {
    auto policy = std::make_unique<execution::PlanewaveMPIExecutionPolicy>();
    policy->configure_topology(input.number_of_kpoint_pools);
    log::init_mpi();
    return create_scf_iterator_mpi(input, std::move(policy));
}

} // namespace q3
