#pragma once

/**
 * create_flavor — factory for Q3 calculation flavors.
 *
 * Dispatches by calculation type string to the appropriate Flavor subclass.
 * run_flavor() below is the entry point callers should reach for: it is the one
 * place a completed result is stamped with its provenance.
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "io/provenance/factory.hpp"
#include "layer_4_computation/flavors/base.hpp"
#include "layer_4_computation/flavors/non_self_consistent_field.hpp"
#include "layer_4_computation/flavors/self_consistent_field.hpp"
#include "layer_4_computation/flavors/verification.hpp"

namespace q3 {

/**
 * Create a Flavor from a calculation type string.
 *
 * `iterator_factory` is the entry point's execution environment -- the MPI binaries'
 * execution policy, the GPU binaries' diagonalizer -- and reaches every flavor
 * that builds an iterator, so a unit of a workflow runs on the same setup
 * whichever flavor it is. Empty means serial, on the CPU.
 *
 * Supported types:
 *   - "self_consistent_field"     → SelfConsistentFieldFlavor
 *   - "non_self_consistent_field" → NonSelfConsistentFieldFlavor
 *   - "verification"              → VerificationFlavor
 *
 * A "relaxation" ("relax", "vc-relax") never reaches this factory: parse_input()
 * runs it as one self_consistent_field step of the relaxation q3.workflows drives.
 *
 * "band_structure" ("bands") and "density_of_states" ("dos") are deliberately
 * NOT single-flavor calculation types (see main.cpp's
 * dispatch comment): each is a multi-step workflow that chains
 * self_consistent_field/non_self_consistent_field runs from the Python layer
 * (q3.workflows.band_structure/dos). The YAML schema and parser accept
 * them (io/input/parser/yaml.hpp normalizes the short aliases and validates
 * the workflow-specific control keys, e.g. band_path), because that Python
 * layer reads the same input file -- but the engine itself never receives
 * these strings as calculation_type. A caller that reaches this factory
 * with one of them is running the file directly instead of through its
 * workflow function, so the error says exactly that instead of just "unsupported".
 */
inline std::unique_ptr<Flavor> create_flavor(const std::string& calculation_type,
                                             IteratorFactory iterator_factory = {}) {
    if (calculation_type == "self_consistent_field") {
        return std::make_unique<SelfConsistentFieldFlavor>(std::move(iterator_factory));
    }
    if (calculation_type == "non_self_consistent_field") {
        return std::make_unique<NonSelfConsistentFieldFlavor>(std::move(iterator_factory));
    }
    if (calculation_type == "verification") {
        /**
         * A verification compares the engine against a reference on one
         * process; an MPI or GPU setup handed in here would be silently
         * dropped, so it is refused instead.
         */
        if (iterator_factory) {
            throw std::runtime_error(
                "create_flavor: 'verification' runs on the serial q3 binary only; run this "
                "input with q3.");
        }
        return std::make_unique<VerificationFlavor>();
    }
    if (calculation_type == "band_structure") {
        throw std::runtime_error(
            "create_flavor: 'band_structure' is a multi-step workflow (SCF, then NSCF along "
            "the requested k-point path), orchestrated by the Python layer, not a single "
            "engine flavor -- the q3 binary never runs it directly. Run this file through "
            "q3.workflows.run_band_structure_workflow(scf_input_path=<this file>) instead, "
            "or set calculation: self_consistent_field for a single-point SCF run.");
    }
    if (calculation_type == "interpolation") {
        throw std::runtime_error(
            "create_flavor: 'interpolation' evaluates a model a previous NSCF run saved "
            "(control.interpolation on calculation: nscf), which is post-processing rather "
            "than an engine flavor -- the q3 binary never runs it directly. Run this file "
            "with `q3_utility interpolate <file>` instead.");
    }
    if (calculation_type == "density_of_states") {
        throw std::runtime_error(
            "create_flavor: 'density_of_states' is a multi-step workflow (SCF, then a dense "
            "NSCF k-mesh, then Gaussian-broadened binning), orchestrated by the Python layer, "
            "not a single engine flavor -- the q3 binary never runs it directly. Run this "
            "file through q3.workflows.run_dos_workflow(scf_input_path=...) instead, or set "
            "calculation: self_consistent_field for a single-point SCF run.");
    }
    throw std::runtime_error("create_flavor: unsupported calculation type '" + calculation_type +
                             "'. Supported: self_consistent_field, non_self_consistent_field, "
                             "verification");
}

/**
 * Create the flavor named by the input and run it.
 *
 * A convenience for the callers that dispatch by name; the provenance stamp is
 * not applied here but in Flavor::run(), which every path reaches, including
 * the MPI and GPU entry points that construct their flavor directly rather than
 * through this factory.
 */
[[nodiscard]] inline generated::Results run_flavor(const InputData& input) {
    auto flavor = create_flavor(input.calculation);
    return flavor->run(input);
}

} // namespace q3
