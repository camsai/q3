#pragma once

// @generated from interpolation_results.json — DO NOT EDIT

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

/**
 * The interpolated density of states on control.interpolation.dos_mesh; present when a mesh was asked for.
 */
struct InterpolationResultsDensityOfStates {
    // The dense mesh the eigenvalues were interpolated on, [n1, n2, n3].
    std::vector<int64_t> mesh;
    // The gaussian broadening applied, in rydberg (control.dos_broadening).
    double broadening_ry{};
    // The energy grid, in rydberg.
    std::vector<double> energies_ry;
    // States per rydberg per cell, spin included, one per energy.
    std::vector<double> dos;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsDensityOfStates& data) {
    json_output = nlohmann::json::object();
    json_output["mesh"] = data.mesh;
    json_output["broadening_ry"] = data.broadening_ry;
    json_output["energies_ry"] = data.energies_ry;
    json_output["dos"] = data.dos;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsDensityOfStates& data) {
    json_input.at("mesh").get_to(data.mesh);
    json_input.at("broadening_ry").get_to(data.broadening_ry);
    json_input.at("energies_ry").get_to(data.energies_ry);
    json_input.at("dos").get_to(data.dos);
}

/**
 * The Marzari-Vanderbilt spread of the model's Wannier functions at the gauge it was built with, in bohr and bohr squared. Omega_I is reported always; the rest whenever the record carries the neighbour overlaps that define it and the mesh has no two-point axis — the projection gauge has a spread too, and a driver comparing two models wants it either way.
 */
struct InterpolationResultsSpread {
    // Omega_I: the part of the spread the gauge cannot change. From the spread when it was evaluated, otherwise from the subspace iteration when there was one; zero for an isolated manifold without overlaps.
    double gauge_invariant_bohr_squared{};
    // Whether the gauge was rotated by the spread minimiser at all.
    bool localized{};
    // Omega = Omega_I + Omega_D + Omega_OD at the final gauge.
    std::optional<double> total_bohr_squared;
    // What fixed the branch of each Berry phase: "trial orbitals" when the record carries their positions, "projection gauge" otherwise.
    std::optional<std::string> guiding_centers;
    // The record's trial-orbital positions, Cartesian bohr, one [x, y, z] per function; empty when it carries none.
    std::optional<std::vector<std::vector<double>>> trial_orbital_centers_bohr;
    // Omega_D + Omega_OD at the projection gauge, before any rotation: what localisation bought is this minus the final value.
    std::optional<double> initial_minimizable_bohr_squared;
    // The Wannier centres at the final gauge, Cartesian bohr, one [x, y, z] per function.
    std::optional<std::vector<std::vector<double>>> centers_bohr;
    // Omega_n per function: <r^2>_n - |<r>_n|^2, summing to the total.
    std::optional<std::vector<double>> spreads_bohr_squared;
    // How many iterations the spread minimiser ran; present when it ran.
    std::optional<int64_t> localization_iterations;
    // Whether the minimiser reached a stationary gauge rather than merely stopped; false after a stall, where the gauge is the best one found but not a minimum.
    std::optional<bool> localization_converged;
    // The norm of the spread gradient where the minimiser stopped, a diagnostic beside the answer rather than the convergence test.
    std::optional<double> localization_gradient_norm;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsSpread& data) {
    json_output = nlohmann::json::object();
    json_output["gauge_invariant_bohr_squared"] = data.gauge_invariant_bohr_squared;
    json_output["localized"] = data.localized;
    if (data.total_bohr_squared) {
        json_output["total_bohr_squared"] = *data.total_bohr_squared;
    }
    if (data.guiding_centers) {
        json_output["guiding_centers"] = *data.guiding_centers;
    }
    if (data.trial_orbital_centers_bohr) {
        json_output["trial_orbital_centers_bohr"] = *data.trial_orbital_centers_bohr;
    }
    if (data.initial_minimizable_bohr_squared) {
        json_output["initial_minimizable_bohr_squared"] = *data.initial_minimizable_bohr_squared;
    }
    if (data.centers_bohr) {
        json_output["centers_bohr"] = *data.centers_bohr;
    }
    if (data.spreads_bohr_squared) {
        json_output["spreads_bohr_squared"] = *data.spreads_bohr_squared;
    }
    if (data.localization_iterations) {
        json_output["localization_iterations"] = *data.localization_iterations;
    }
    if (data.localization_converged) {
        json_output["localization_converged"] = *data.localization_converged;
    }
    if (data.localization_gradient_norm) {
        json_output["localization_gradient_norm"] = *data.localization_gradient_norm;
    }
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsSpread& data) {
    json_input.at("gauge_invariant_bohr_squared").get_to(data.gauge_invariant_bohr_squared);
    json_input.at("localized").get_to(data.localized);
    if (json_input.contains("total_bohr_squared")) {
        data.total_bohr_squared = json_input.at("total_bohr_squared").get<double>();
    }
    if (json_input.contains("guiding_centers")) {
        data.guiding_centers = json_input.at("guiding_centers").get<std::string>();
    }
    if (json_input.contains("trial_orbital_centers_bohr")) {
        data.trial_orbital_centers_bohr = json_input.at("trial_orbital_centers_bohr").get<std::vector<std::vector<double>>>();
    }
    if (json_input.contains("initial_minimizable_bohr_squared")) {
        data.initial_minimizable_bohr_squared = json_input.at("initial_minimizable_bohr_squared").get<double>();
    }
    if (json_input.contains("centers_bohr")) {
        data.centers_bohr = json_input.at("centers_bohr").get<std::vector<std::vector<double>>>();
    }
    if (json_input.contains("spreads_bohr_squared")) {
        data.spreads_bohr_squared = json_input.at("spreads_bohr_squared").get<std::vector<double>>();
    }
    if (json_input.contains("localization_iterations")) {
        data.localization_iterations = json_input.at("localization_iterations").get<int64_t>();
    }
    if (json_input.contains("localization_converged")) {
        data.localization_converged = json_input.at("localization_converged").get<bool>();
    }
    if (json_input.contains("localization_gradient_norm")) {
        data.localization_gradient_norm = json_input.at("localization_gradient_norm").get<double>();
    }
}

/**
 */
struct InterpolationResultsExcludedBand {
    // The band, one-based as the manifest counts them.
    int64_t band{};
    // At how many k-points it was excluded.
    int64_t kpoints{};
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsExcludedBand& data) {
    json_output = nlohmann::json::object();
    json_output["band"] = data.band;
    json_output["kpoints"] = data.kpoints;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsExcludedBand& data) {
    json_input.at("band").get_to(data.band);
    json_input.at("kpoints").get_to(data.kpoints);
}

/**
 * What the classifier decided when the subspace was chosen by projectability disentanglement: the window it resolved and which bands the singular-gauge rule kept out of the freeze. Present only when the model was disentangled.
 */
struct InterpolationResultsDisentanglement {
    // Whether a band crosses the Fermi level, which is what makes the automatic window E_F + 2 eV rather than 0.
    bool metallic{};
    // How many bands cross the Fermi level.
    int64_t bands_crossing_fermi{};
    // The frozen window that was applied, in electronvolts above the Fermi level.
    double frozen_window_above_fermi_ev{};
    // Whether the window was resolved from the record (true) or given in the manifest (false).
    bool frozen_window_automatic{};
    // Bands inside the frozen window whose projectability fell below the minimum at some k-points, and were excluded from the freeze there rather than frozen: the trial orbitals cannot represent them, and freezing them would make the gauge singular.
    std::vector<InterpolationResultsExcludedBand> bands_excluded_from_freeze;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsDisentanglement& data) {
    json_output = nlohmann::json::object();
    json_output["metallic"] = data.metallic;
    json_output["bands_crossing_fermi"] = data.bands_crossing_fermi;
    json_output["frozen_window_above_fermi_ev"] = data.frozen_window_above_fermi_ev;
    json_output["frozen_window_automatic"] = data.frozen_window_automatic;
    json_output["bands_excluded_from_freeze"] = data.bands_excluded_from_freeze;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsDisentanglement& data) {
    json_input.at("metallic").get_to(data.metallic);
    json_input.at("bands_crossing_fermi").get_to(data.bands_crossing_fermi);
    json_input.at("frozen_window_above_fermi_ev").get_to(data.frozen_window_above_fermi_ev);
    json_input.at("frozen_window_automatic").get_to(data.frozen_window_automatic);
    json_input.at("bands_excluded_from_freeze").get_to(data.bands_excluded_from_freeze);
}

/**
 * Statistics of the per-function spreads; present when a spread was evaluated.
 */
struct InterpolationResultsQualitySpread {
    double median_bohr_squared{};
    // The population variance of the spreads across functions: the published no-reference criterion.
    double variance_bohr_squared{};
    double maximum_bohr_squared{};
    // Zero-based indices of functions above five times the median.
    std::vector<int64_t> outliers;
    // Zero-based indices of functions whose spread exceeds ten square angstrom, whatever the others do.
    std::vector<int64_t> above_ten_angstrom_squared;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsQualitySpread& data) {
    json_output = nlohmann::json::object();
    json_output["median_bohr_squared"] = data.median_bohr_squared;
    json_output["variance_bohr_squared"] = data.variance_bohr_squared;
    json_output["maximum_bohr_squared"] = data.maximum_bohr_squared;
    json_output["outliers"] = data.outliers;
    json_output["above_ten_angstrom_squared"] = data.above_ten_angstrom_squared;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsQualitySpread& data) {
    json_input.at("median_bohr_squared").get_to(data.median_bohr_squared);
    json_input.at("variance_bohr_squared").get_to(data.variance_bohr_squared);
    json_input.at("maximum_bohr_squared").get_to(data.maximum_bohr_squared);
    json_input.at("outliers").get_to(data.outliers);
    json_input.at("above_ten_angstrom_squared").get_to(data.above_ten_angstrom_squared);
}

/**
 * The Wannier centres against the atoms of the manifest's material, over periodic images; present when a spread was evaluated and the manifest names atoms.
 */
struct InterpolationResultsQualityCenters {
    // Per function, the distance to the nearest atom.
    std::vector<double> nearest_atom_distance_bohr;
    // Per function, the distance to the nearest midpoint of a bond.
    std::vector<double> nearest_bond_midpoint_distance_bohr;
    // The shortest interatomic distance, which defines a bond here as anything within 20 percent of it.
    double shortest_bond_bohr{};
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsQualityCenters& data) {
    json_output = nlohmann::json::object();
    json_output["nearest_atom_distance_bohr"] = data.nearest_atom_distance_bohr;
    json_output["nearest_bond_midpoint_distance_bohr"] = data.nearest_bond_midpoint_distance_bohr;
    json_output["shortest_bond_bohr"] = data.shortest_bond_bohr;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsQualityCenters& data) {
    json_input.at("nearest_atom_distance_bohr").get_to(data.nearest_atom_distance_bohr);
    json_input.at("nearest_bond_midpoint_distance_bohr").get_to(data.nearest_bond_midpoint_distance_bohr);
    json_input.at("shortest_bond_bohr").get_to(data.shortest_bond_bohr);
}

/**
 * How far H(R) has decayed by the edge of its Wigner-Seitz domain. A Bloch sum that has not decayed by its last term is the direct cause of error between mesh points.
 */
struct InterpolationResultsHamiltonianDecay {
    // |R| of the outermost shell of lattice vectors.
    double outermost_shell_distance_bohr{};
    // How many lattice vectors share that distance.
    int64_t outermost_shell_size{};
    // The largest Bloch-sum term w_R |H_mn(R)| on that shell.
    double outermost_shell_maximum_ry{};
    // The largest term that varies with k: any R away from the origin, or m != n on site, so a rigid shift of the energy zero cannot move it.
    double largest_hopping_ry{};
    // outermost_shell_maximum_ry / largest_hopping_ry; zero when there is nothing to divide by.
    double ratio{};
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsHamiltonianDecay& data) {
    json_output = nlohmann::json::object();
    json_output["outermost_shell_distance_bohr"] = data.outermost_shell_distance_bohr;
    json_output["outermost_shell_size"] = data.outermost_shell_size;
    json_output["outermost_shell_maximum_ry"] = data.outermost_shell_maximum_ry;
    json_output["largest_hopping_ry"] = data.largest_hopping_ry;
    json_output["ratio"] = data.ratio;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsHamiltonianDecay& data) {
    json_input.at("outermost_shell_distance_bohr").get_to(data.outermost_shell_distance_bohr);
    json_input.at("outermost_shell_size").get_to(data.outermost_shell_size);
    json_input.at("outermost_shell_maximum_ry").get_to(data.outermost_shell_maximum_ry);
    json_input.at("largest_hopping_ry").get_to(data.largest_hopping_ry);
    json_input.at("ratio").get_to(data.ratio);
}

/**
 */
struct InterpolationResultsLadderAttempt {
    // The rung of the ladder that made the attempt: "default", "unconverged", "stalled", "window", "thresholds" or "mesh".
    std::string rung;
    // The k-mesh of the record the attempt evaluated.
    std::vector<int64_t> grid;
    // The control.interpolation keys the attempt changed from the defaults, as it wrote them; for the mesh rung, the spacing it recaptured at (kpoint_spacing_inverse_angstrom).
    nlohmann::json settings;
    // What became of the attempt: "accept" or "retry" for an evaluation the gate judged, "error:<route>" for one that raised (the route is "unconverged", "localization", "infeasible" or "unknown", from the engine's diagnostic code), and "recapture" for the mesh rung, which evaluates nothing.
    std::string status;
    // Why, in words: the gate's verdict or the engine's own error line.
    std::string reason;
    // The band distance of the attempt, in electronvolts; null when the evaluation raised.
    nlohmann::json eta_ev;
    // The largest pointwise band difference of the attempt, in electronvolts; null when the evaluation raised.
    nlohmann::json eta_maximum_ev;
    // Whether the attempt's gauge was localised; null when the evaluation raised.
    nlohmann::json localized;
    // Whether that localisation converged; null when the evaluation raised or did not localise.
    nlohmann::json localization_converged;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsLadderAttempt& data) {
    json_output = nlohmann::json::object();
    json_output["rung"] = data.rung;
    json_output["grid"] = data.grid;
    json_output["settings"] = data.settings;
    json_output["status"] = data.status;
    json_output["reason"] = data.reason;
    json_output["eta_ev"] = data.eta_ev;
    json_output["eta_maximum_ev"] = data.eta_maximum_ev;
    json_output["localized"] = data.localized;
    json_output["localization_converged"] = data.localization_converged;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsLadderAttempt& data) {
    json_input.at("rung").get_to(data.rung);
    json_input.at("grid").get_to(data.grid);
    json_input.at("settings").get_to(data.settings);
    json_input.at("status").get_to(data.status);
    json_input.at("reason").get_to(data.reason);
    json_input.at("eta_ev").get_to(data.eta_ev);
    json_input.at("eta_maximum_ev").get_to(data.eta_maximum_ev);
    json_input.at("localized").get_to(data.localized);
    json_input.at("localization_converged").get_to(data.localization_converged);
}

/**
 * Added by q3.workflows.wannier: the band distance of this model against a direct calculation on the reference k-points.
 */
struct InterpolationResultsBandDistance {
    // The band distance eta, in electronvolts.
    double eta_ev{};
    // The largest pointwise band difference, in electronvolts.
    double eta_maximum_ev{};
    // How many (k-point, band) pairs entered the distance.
    int64_t number_of_states{};
    // What the reference k-points were: "path" or "half-shifted mesh".
    std::string reference;
    // How many reference k-points were compared.
    int64_t number_of_kpoints{};
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsBandDistance& data) {
    json_output = nlohmann::json::object();
    json_output["eta_ev"] = data.eta_ev;
    json_output["eta_maximum_ev"] = data.eta_maximum_ev;
    json_output["number_of_states"] = data.number_of_states;
    json_output["reference"] = data.reference;
    json_output["number_of_kpoints"] = data.number_of_kpoints;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsBandDistance& data) {
    json_input.at("eta_ev").get_to(data.eta_ev);
    json_input.at("eta_maximum_ev").get_to(data.eta_maximum_ev);
    json_input.at("number_of_states").get_to(data.number_of_states);
    json_input.at("reference").get_to(data.reference);
    json_input.at("number_of_kpoints").get_to(data.number_of_kpoints);
}

/**
 * What the model says about itself with no reference calculation, and — added afterwards by q3.workflows.wannier, which runs the calculation they need — the band distance against a direct one and the verdict it earned.
 */
struct InterpolationResultsQuality {
    // Statistics of the per-function spreads; present when a spread was evaluated.
    std::optional<InterpolationResultsQualitySpread> spread;
    // The Wannier centres against the atoms of the manifest's material, over periodic images; present when a spread was evaluated and the manifest names atoms.
    std::optional<InterpolationResultsQualityCenters> centers;
    // How far H(R) has decayed by the edge of its Wigner-Seitz domain. A Bloch sum that has not decayed by its last term is the direct cause of error between mesh points.
    InterpolationResultsHamiltonianDecay hamiltonian_decay;
    // Repeated from spread.localization_converged when the gauge was minimised, so the block a driver judges by is complete on its own.
    std::optional<bool> localization_converged;
    // Added by q3.workflows.wannier: "accept" or "reject", the verdict of the retry ladder on this model.
    std::optional<std::string> status;
    // Added by q3.workflows.wannier: why the verdict was reached, in words.
    std::optional<std::string> reason;
    // Added by q3.workflows.wannier: every evaluation the retry ladder made of the saved record, in order, ending with the one this file describes.
    std::optional<std::vector<InterpolationResultsLadderAttempt>> ladder;
    // Added by q3.workflows.wannier: the band distance of this model against a direct calculation on the reference k-points.
    std::optional<InterpolationResultsBandDistance> band_distance;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsQuality& data) {
    json_output = nlohmann::json::object();
    if (data.spread) {
        json_output["spread"] = *data.spread;
    }
    if (data.centers) {
        json_output["centers"] = *data.centers;
    }
    json_output["hamiltonian_decay"] = data.hamiltonian_decay;
    if (data.localization_converged) {
        json_output["localization_converged"] = *data.localization_converged;
    }
    if (data.status) {
        json_output["status"] = *data.status;
    }
    if (data.reason) {
        json_output["reason"] = *data.reason;
    }
    if (data.ladder) {
        json_output["ladder"] = *data.ladder;
    }
    if (data.band_distance) {
        json_output["band_distance"] = *data.band_distance;
    }
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsQuality& data) {
    if (json_input.contains("spread")) {
        data.spread = json_input.at("spread").get<InterpolationResultsQualitySpread>();
    }
    if (json_input.contains("centers")) {
        data.centers = json_input.at("centers").get<InterpolationResultsQualityCenters>();
    }
    json_input.at("hamiltonian_decay").get_to(data.hamiltonian_decay);
    if (json_input.contains("localization_converged")) {
        data.localization_converged = json_input.at("localization_converged").get<bool>();
    }
    if (json_input.contains("status")) {
        data.status = json_input.at("status").get<std::string>();
    }
    if (json_input.contains("reason")) {
        data.reason = json_input.at("reason").get<std::string>();
    }
    if (json_input.contains("ladder")) {
        data.ladder = json_input.at("ladder").get<std::vector<InterpolationResultsLadderAttempt>>();
    }
    if (json_input.contains("band_distance")) {
        data.band_distance = json_input.at("band_distance").get<InterpolationResultsBandDistance>();
    }
}

/**
 * The model that was evaluated: how it was built, how big it is, and what it says about itself.
 */
struct InterpolationResultsInterpolation {
    // The record's method name ("wannier").
    std::string method;
    // How the model was built, in words: the method, the subspace choice, whether and how far the gauge was localised, and the model's size.
    std::string description;
    // The path of the record that was evaluated.
    std::string record;
    // The number of bands the model interpolates: the number of Wannier functions.
    int64_t number_of_bands{};
    // The Wigner-Seitz weight identity: the sum of the Bloch-sum weights over the model's lattice vectors, which equals the number of k-points of the mesh the record was built on.
    double total_bloch_sum_weight{};
    // Any further diagnostic the interpolator reports, by name. The local-orbital interpolator reports none beyond total_bloch_sum_weight, which has its own key because readers and the reference name it.
    std::optional<std::map<std::string, double>> diagnostics;
    // The Marzari-Vanderbilt spread of the model's Wannier functions at the gauge it was built with, in bohr and bohr squared. Omega_I is reported always; the rest whenever the record carries the neighbour overlaps that define it and the mesh has no two-point axis — the projection gauge has a spread too, and a driver comparing two models wants it either way.
    InterpolationResultsSpread spread;
    // What the classifier decided when the subspace was chosen by projectability disentanglement: the window it resolved and which bands the singular-gauge rule kept out of the freeze. Present only when the model was disentangled.
    std::optional<InterpolationResultsDisentanglement> disentanglement;
    // What the model says about itself with no reference calculation, and — added afterwards by q3.workflows.wannier, which runs the calculation they need — the band distance against a direct one and the verdict it earned.
    InterpolationResultsQuality quality;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResultsInterpolation& data) {
    json_output = nlohmann::json::object();
    json_output["method"] = data.method;
    json_output["description"] = data.description;
    json_output["record"] = data.record;
    json_output["number_of_bands"] = data.number_of_bands;
    json_output["total_bloch_sum_weight"] = data.total_bloch_sum_weight;
    if (data.diagnostics) {
        json_output["diagnostics"] = *data.diagnostics;
    }
    json_output["spread"] = data.spread;
    if (data.disentanglement) {
        json_output["disentanglement"] = *data.disentanglement;
    }
    json_output["quality"] = data.quality;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResultsInterpolation& data) {
    json_input.at("method").get_to(data.method);
    json_input.at("description").get_to(data.description);
    json_input.at("record").get_to(data.record);
    json_input.at("number_of_bands").get_to(data.number_of_bands);
    json_input.at("total_bloch_sum_weight").get_to(data.total_bloch_sum_weight);
    if (json_input.contains("diagnostics")) {
        data.diagnostics = json_input.at("diagnostics").get<std::map<std::string, double>>();
    }
    json_input.at("spread").get_to(data.spread);
    if (json_input.contains("disentanglement")) {
        data.disentanglement = json_input.at("disentanglement").get<InterpolationResultsDisentanglement>();
    }
    json_input.at("quality").get_to(data.quality);
}

/**
 * The file `q3_utility interpolate` writes: <outdir>/<prefix>_interpolation_results.json, the evaluation of a saved interpolation record along a k-point list and, when asked, on a dense mesh for a density of states. `required` is the PRODUCER's guarantee — present in every file the utility writes. It is not a reader's precondition: the Python model (q3.io.interpolation_results) defaults every field to None, because the unit-test stub and older files carry a subset. The keys q3.workflows.wannier adds after the evaluation (quality.status, quality.reason, quality.ladder, quality.band_distance) are optional here, so one schema describes the file before and after the workflow annotates it. Units are carried in the key names: bohr and bohr squared for the Wannier geometry, rydberg for energies, electronvolts where the manifest speaks them.
 *
 * Schema: interpolation_results.json
 */
struct InterpolationResults {
    // Always "interpolation": the calculation this file is the result of.
    std::string calculation;
    // The provenance stamp of the evaluated manifest, carried through as it stands; its shape is the SCF results file's (docs/io_reference.md, section 4.1) and is not described again here.
    nlohmann::json provenance;
    // The Fermi level the NSCF that built the record computed over its own full-grid k-set, in rydberg — the highest occupied level when every requested band is occupied. q3.workflows.interpolation reports the SCF's value instead, which is the authoritative one.
    double fermi_energy_ry{};
    // Crystal coordinates of the evaluated k-points, one [k1, k2, k3] each; present when bands were evaluated.
    std::optional<std::vector<std::vector<double>>> kpoints;
    // The interpolated eigenvalues shaped [k-point][spin][band] in rydberg, as the SCF results file shapes them so the Python extractors read both unchanged; present with kpoints.
    std::optional<std::vector<std::vector<std::vector<double>>>> eigenvalues_ry;
    // The interpolated density of states on control.interpolation.dos_mesh; present when a mesh was asked for.
    std::optional<InterpolationResultsDensityOfStates> density_of_states;
    // The model that was evaluated: how it was built, how big it is, and what it says about itself.
    InterpolationResultsInterpolation interpolation;
};

inline void to_json(nlohmann::json& json_output, const InterpolationResults& data) {
    json_output = nlohmann::json::object();
    json_output["calculation"] = data.calculation;
    json_output["provenance"] = data.provenance;
    json_output["fermi_energy_ry"] = data.fermi_energy_ry;
    if (data.kpoints) {
        json_output["kpoints"] = *data.kpoints;
    }
    if (data.eigenvalues_ry) {
        json_output["eigenvalues_ry"] = *data.eigenvalues_ry;
    }
    if (data.density_of_states) {
        json_output["density_of_states"] = *data.density_of_states;
    }
    json_output["interpolation"] = data.interpolation;
}

inline void from_json(const nlohmann::json& json_input, InterpolationResults& data) {
    json_input.at("calculation").get_to(data.calculation);
    json_input.at("provenance").get_to(data.provenance);
    json_input.at("fermi_energy_ry").get_to(data.fermi_energy_ry);
    if (json_input.contains("kpoints")) {
        data.kpoints = json_input.at("kpoints").get<std::vector<std::vector<double>>>();
    }
    if (json_input.contains("eigenvalues_ry")) {
        data.eigenvalues_ry = json_input.at("eigenvalues_ry").get<std::vector<std::vector<std::vector<double>>>>();
    }
    if (json_input.contains("density_of_states")) {
        data.density_of_states = json_input.at("density_of_states").get<InterpolationResultsDensityOfStates>();
    }
    json_input.at("interpolation").get_to(data.interpolation);
}

} // namespace q3::generated
