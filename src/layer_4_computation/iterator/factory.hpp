#pragma once

/**
 * Top-level iterator factory — dispatches by theory.
 *
 * create_iterator(InputData) is the main entry point for production use.
 * Reads input.theory to select the method (default: "kohn_sham") and
 * delegates to the appropriate sub-factory.
 *
 * Uses PlanewaveSerialExecutionPolicy by default (serial, single-process).
 * For MPI runs, use io/iterator/kohn_sham/planewave/factory_mpi.hpp which supplies
 * PlanewaveMPIExecutionPolicy.
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "io/input/data.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

namespace q3 {

/**
 * Create a serial SCF iterator from parsed InputData.
 *
 * Dispatches by input.theory:
 *   - "kohn_sham" (default): delegates to create_kohn_sham_iterator(input, policy).
 *
 * Returns a PlanewaveKohnShamSCFIterator so callers can access result()
 * without knowing the concrete subclass.
 *
 * Ownership of the execution policy is transferred into the returned iterator.
 */
inline std::unique_ptr<PlanewaveKohnShamSCFIterator> create_iterator(const InputData& input) {
    if (input.theory == "kohn_sham") {
        auto policy = std::make_unique<execution::PlanewaveSerialExecutionPolicy>();
        const execution::PlanewaveSerialExecutionPolicy& policy_ref = *policy;
        auto iterator = create_kohn_sham_iterator(input, policy_ref);
        iterator->set_owned_execution_policy(std::move(policy));
        return iterator;
    }
    throw std::runtime_error("Unsupported theory: " + input.theory);
}

} // namespace q3
