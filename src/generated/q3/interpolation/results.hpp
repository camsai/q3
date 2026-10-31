#pragma once

// @generated from interpolation/results.json — DO NOT EDIT

#include <string>

#include <nlohmann/json.hpp>

#include "generated/q3/interpolation/results/construction.hpp"
#include "generated/q3/interpolation/results/properties.hpp"
#include "generated/q3/interpolation/results/quality.hpp"

namespace q3::generated {

/**
 * The file `q3_utility interpolate` writes: <outdir>/<prefix>_interpolation_results.json, the model a saved interpolation record holds — how it was built, how big it is and what it says about itself — and its evaluation along a k-point list and, when asked, on a dense mesh for a density of states. `required` is the PRODUCER's guarantee — present in every file the utility writes. It is not a reader's precondition: the Python model (q3.io.interpolation.results) defaults every field to None, because the unit-test stub and older files carry a subset. The keys q3.workflows.wannier adds after the evaluation (quality.status, quality.reason, quality.ladder, quality.band_distance) are optional here, so one schema describes the file before and after the workflow annotates it. The evaluated quantities under `properties` (the Fermi level, the eigenvalues, the density of states) are the shapes the SCF results file uses, each carrying its unit in its own `units` field (ESSE's fermi_energy and density_of_states records and results.json's eigenvalues block). The diagnostics of the evaluated model (construction and quality) still carry theirs in the key names (bohr and bohr squared for the geometry, rydberg for energies, electronvolts where the manifest speaks them) until the input manifest, which shares their vocabulary, moves to unit fields with them; ESSE has no unit for an area yet.
 *
 * Schema: interpolation/results.json
 */
struct InterpolationResults {
    // Always "interpolation": the calculation this file is the result of.
    std::string calculation;
    // The provenance stamp of the evaluated manifest, carried through as it stands; its shape is the SCF results file's (docs/io_reference.md, section 4.1) and is not described again here.
    nlohmann::json provenance;
    // What the model gave at the k-points it was evaluated at, grouped as the SCF results file groups its own: the Fermi level, the eigenvalues and the density of states, each carrying its unit in its own `units` field.
    InterpolationResultsProperties properties;
    // How the evaluated model was built and how big it is: the method, the record it was read from, its number of bands, and what the construction measured of it (the spread of its Wannier functions and, when the subspace was disentangled, the classifier's verdict).
    InterpolationResultsConstruction construction;
    // What the model says about itself with no reference calculation, and — added afterwards by q3.workflows.wannier, which runs the calculation they need — the band distance against a direct one and the verdict it earned.
    InterpolationResultsQuality quality;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InterpolationResults, calculation, provenance, properties, construction, quality)

} // namespace q3::generated
