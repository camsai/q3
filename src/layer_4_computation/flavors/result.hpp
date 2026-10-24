#pragma once

/**
 * FlavorResult — raw output data from a completed flavor run.
 *
 * Holds all computed quantities as primitive types and standard containers.
 * Does NOT depend on any types in layer_5_properties/results/. Serialization
 * to results.json is provided via write_results_json().
 */

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/json_file.hpp"
#include "io/logger.hpp"
#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_2_quantum_system/derived/magnetization.hpp"

namespace q3 {

struct FlavorResult {
    // ─── Identity ────────────────────────────────────────────────────────────

    std::string flavor;

    /**
     * How this result was produced: engine build, pseudopotential identities,
     * and the model/method/precision the run was asked for. Built by
     * build_provenance_json() and attached by run_flavor(), so a flavor itself
     * never has to carry it. Empty when a FlavorResult is constructed directly
     * (unit tests, and any future caller that bypasses run_flavor), in which
     * case the key is omitted from the JSON rather than emitted blank.
     */
    nlohmann::json provenance;

    /**
     * The structure the run used, as an ESSE material (build_material_json()),
     * attached by Flavor::run() beside the provenance; omitted when empty.
     */
    nlohmann::json material;

    // ─── Convergence ─────────────────────────────────────────────────────────

    bool converged = false;

    /**
     * How many self-consistent field iterations the run performed: the
     * iteration it converged on, or its whole budget when it did not converge.
     *
     * `converged` says whether a run failed, not how far it got, and without
     * this nothing else in the results said so: the budget is in the
     * provenance, as what the run was asked to do, while what it did was only
     * in the log.
     *
     * Set wherever a calculation iterates to self-consistency: the SCF flavor
     * and the bindings' SCFSolver. Unset otherwise -- a non-self-consistent
     * pass, or a verification restart whose convergence exit is disabled -- in
     * which case the key is omitted rather than emitted as a zero that reads as
     * a run that never started.
     */
    std::optional<int> number_of_self_consistent_field_iterations;

    // ─── Timings ─────────────────────────────────────────────────────────────

    /**
     * Where the run spent its time, in milliseconds, taken from
     * ProfilingCounters at the end of the flavor.
     *
     * These are emitted because the numbers existed only in the human-readable
     * log, so anything wanting them -- a scaling study, a regression check,
     * a comparison against another code -- had to regular-expression them back
     * out of console output. That breaks silently: a run that aborts has no
     * timing block to match, and a scraper reports a blank rather than a
     * failure. results.json already carries the execution topology and the
     * provenance a measurement needs to be interpretable, so the timings
     * belong beside them.
     *
     * Empty when nothing ran (a --check invocation, or a flavor that does no
     * SCF), in which case the key is omitted rather than emitted as zeros.
     */
    nlohmann::json timings;

    // ─── Execution topology ──────────────────────────────────────────────────

    int number_of_processes = 1;
    int number_of_kpoint_pools = 1;
    int band_group_size = 1;
    bool reciprocal_space_vectors_are_distributed = false;
    /**
     * OpenMP threads each process may use (OMP_NUM_THREADS), 1 without OpenMP.
     * A single-process run's speed depends on it as much as an MPI run's does
     * on its rank count, and a result used to leave it out.
     */
    int number_of_threads_per_process = execution::threading::maximum_threads();
    /**
     * Largest peak resident set over the ranks, in bytes.
     *
     * Reported as a maximum rather than a sum because every rank holds its own
     * copy of the workspace: what a node has to fit is the largest rank, not
     * the average. Zero when no platform source could be read.
     */
    std::uint64_t peak_resident_memory_bytes = 0;

    // ─── Energies (Ry) ───────────────────────────────────────────────────────

    Real total_energy_in_ry = 0.0;

    // Energy breakdown (all in Ry)
    Real band_energy_in_ry = 0.0;
    Real hartree_energy_in_ry = 0.0;
    Real exchange_correlation_energy_in_ry = 0.0;
    Real ewald_energy_in_ry = 0.0;
    Real double_counting_correction_in_ry = 0.0;
    Real smearing_entropy_contribution_in_ry = 0.0;
    Real hubbard_energy_in_ry = 0.0;

    /**
     * Semi-empirical dispersion (van der Waals) correction to the total energy.
     *
     * Already inside total_energy_in_ry; reported separately for the same
     * reason the log prints it on its own line -- it is the whole of what the
     * correction did, and a layered-material result is uninterpretable without
     * being able to subtract it. Exactly zero when the run asked for no
     * dispersion scheme, in which case the key is omitted.
     */
    Real dispersion_energy_in_ry = 0.0;

    /**
     * PAW one-center contribution Σ_a (E¹_a − Ẽ¹_a). 0 for NC and ultrasoft.
     *
     * Reported separately rather than folded into the total because it is the
     * first rung of the PAW failure ladder: QE prints the same quantity as
     * `one-center paw contrib.`, so comparing it directly separates a
     * plane-wave-side discrepancy from a one-center one before any finer
     * decomposition is needed.
     */
    Real one_center_energy_in_ry = 0.0;

    /**
     * ─── Isolated-system correction ─────────────────────────────────────────
     *
     * Scheme-agnostic mirror of IsolatedSystemCorrectionResult (kept as plain
     * std types here so this layer-4 header does not include the layer-5
     * interface). `applied` is false when control.isolated_system_correction
     * is "none"/unset; `terms_in_ry` are the scheme's labeled additive energy
     * terms (Makov-Payne: "first_order", "second_order"), each already signed
     * such that corrected = total_energy_in_ry + sum(terms).
     */

    bool isolated_system_correction_applied = false;
    std::string isolated_system_correction_method = "none";
    std::vector<std::pair<std::string, Real>> isolated_system_correction_terms_in_ry;

    // ─── Forces (Ry/bohr) ────────────────────────────────────────────────────

    /// Per-atom Hellmann-Feynman forces in Cartesian Ry/bohr.
    std::vector<Vector3D> forces_in_ry_per_bohr;

    /**
     * ─── Known-incomplete derivatives ───────────────────────────────────────
     *
     * Set when a term Q3 cannot yet compute has been left out of the forces or
     * the stress. The reason is carried, not just a flag, so a consumer can say
     * *what* is missing.
     *
     * These exist because a warning in the log is invisible to anything that
     * reads the results programmatically. A geometry relaxation driven by
     * forces with a term missing converges — to the wrong geometry, quietly.
     * The Python relax and vc-relax workflows refuse to run on a result that
     * carries one of these, which turns a silently wrong answer into a stop.
     */
    bool forces_are_incomplete = false;
    std::string forces_incomplete_reason;
    bool stress_is_incomplete = false;
    std::string stress_incomplete_reason;

    /**
     * ─── Per-atom force breakdown (Ry/bohr) ─────────────────────────────────
     *
     * The same decomposition the stress already carried: one named term per
     * physical mechanism, each a full per-atom vector rather than the summary
     * root-mean-square the log prints beside it.
     *
     * A total force that disagrees with a reference says only that something is
     * wrong; the term that disagrees says what. That diagnosis was available
     * only by reading the console block, so every check of it had to parse
     * stdout -- and a run that aborts before the block prints leaves a scraper
     * reporting a blank instead of a failure.
     */
    struct ForceContribution {
        std::string name;
        std::vector<Vector3D> values_in_ry_per_bohr;
    };
    std::vector<ForceContribution> force_contributions;

    // ─── Stress tensor (kbar) ────────────────────────────────────────────────

    /**
     * 3×3 stress tensor in kbar. Zero-initialized; computed only when
     * forces_and_stress is enabled in the input.
     */
    Real stress_in_kbar[3][3]{};

    struct StressContribution {
        std::string name;
        Real values_in_kbar[3][3]{};
    };
    std::vector<StressContribution> stress_contributions;

    /**
     * Hydrostatic pressure Tr(σ)/3, in kbar.
     *
     * Carried from the stress calculator rather than re-derived from
     * stress_in_kbar here, so the file and the log report one number rather
     * than two: the calculator takes the trace before converting to kbar, and
     * taking it afterwards need not agree in the last bits. Zero whenever the
     * stress itself is (no forces_and_stress, or a topology that skips it),
     * exactly as stress_in_kbar is a zero tensor there.
     */
    Real pressure_in_kbar = 0.0;

    // ─── Eigenvalues ─────────────────────────────────────────────────────────

    /**
     * Eigenvalues per k-point per spin per band in Ry.
     * Dimensions: [number_of_kpoints][number_of_spin_components][number_of_bands]
     */
    std::vector<std::vector<std::vector<Real>>> eigenvalues_in_ry;

    /**
     * Occupation numbers f_nk in [0, 1] (per spin channel; NOT pre-multiplied by
     * spin degeneracy -- that factor lives in kpoint_weights, matching how
     * total-electron counting already works). Same dimensions and (single
     * spin-component-0) limitation as eigenvalues_in_ry above. Empty unless
     * explicitly populated by the flavor (all flavors that populate
     * eigenvalues_in_ry also populate this).
     */
    std::vector<std::vector<std::vector<Real>>> occupations;

    // ─── SCF history ─────────────────────────────────────────────────────────

    /**
     * One row per SCF iteration, in order, recorded where the iteration line is
     * printed so the two can never disagree.
     *
     * Exists so a companion output that lists every iteration can be written
     * after the run without re-deriving anything: a result that carried only
     * the converged numbers could not produce one.
     */
    struct ScfIteration {
        int iteration = 0;
        Real total_energy_in_ry = 0.0;
        Real energy_change_in_ry = 0.0;
        Real density_residual = 0.0;
    };
    std::vector<ScfIteration> scf_iterations;

    // ─── K-point weights ─────────────────────────────────────────────────────

    /**
     * Integration weight for each k-point in the irreducible Brillouin zone.
     * Normalized so that the sum equals 2.0 (accounting for spin degeneracy).
     * Dimensions: [number_of_kpoints]
     */
    std::vector<Real> kpoint_weights;

    /**
     * The same k-points' fractional coordinates, one per weight.
     *
     * Needed by companion outputs that list the k-points before the
     * eigenvalues and index the eigenvalue sets against that list.
     */
    std::vector<Vector3D> kpoint_coordinates;

    // ─── Fermi energy ────────────────────────────────────────────────────────

    Real fermi_energy_in_ry = 0.0;

    /**
     * ─── Magnetization (Bohr magnetons per cell) ────────────────────────────
     *
     * Both of the numbers the convergence report prints: ∫(ρ↑−ρ↓) and
     * ∫|ρ↑−ρ↓|. The total alone cannot describe an antiferromagnet -- it is
     * zero both for a converged AFM state and for one that collapsed to
     * non-magnetic -- so a consumer that has only the total cannot answer "did
     * the moments survive?" from the results file either.
     *
     * Set by a spin-polarized run that converged, which is where the iterator
     * integrates them for the report, and carried over as the iterator holds
     * it; empty for everything else (spin-unpolarized, or a run that never
     * reached the convergence block), and the keys are then omitted rather than
     * emitted as a zero that reads as a collapsed moment.
     */
    std::optional<Magnetization> magnetization;

    /**
     * ─── DFT+U on-site occupations ──────────────────────────────────────────
     *
     * Tr[n] per Hubbard manifold, per spin channel: the number of electrons the
     * manifold's projectors actually see, and the first thing to look at when a
     * DFT+U result looks wrong -- it says immediately whether the projection
     * captured the shell. For ultrasoft and PAW it depends on the overlap
     * operator: measured on fcc nickel, 8.83 with it and 2.71 without.
     *
     * The failure it diagnoses is a converged, plausible, wrong answer, which
     * is exactly the kind that has to be checkable without a human reading the
     * console. Empty when the run has no Hubbard manifolds.
     *
     * `atom_index` is 0-based, matching orbital_atom_index below; the log
     * prints it 1-based.
     */
    struct HubbardOccupation {
        int atom_index = 0;
        std::vector<Real> occupation_trace_per_spin;
    };
    std::vector<HubbardOccupation> hubbard_occupations;

    /**
     * ─── SCF energy trajectory (Ry) ──────────────────────────────────────────
     *
     * Total energy after each completed iteration, oldest first. A single
     * number says where a run landed; the sequence says whether it got there,
     * which is what a restart has to demonstrate -- a density that is really a
     * fixed point of the SCF map does not move when iterated again.
     *
     * Populated by the verification flavor. Empty elsewhere, so a consumer can
     * tell "did not report a trajectory" from "reported one that is flat".
     */
    std::vector<Real> scf_energies_in_ry;

    /**
     * ─── Atomic-orbital (ortho-atomic/Löwdin) projections ──────────────────────
     *
     * Populated only when input.compute_projections is set (NSCF calculations).
     * |<phi'_i|psi_nk>|^2 for every k-point/band/orbital -- see
     * layer_5_properties/projections/planewave/init.hpp. Kept as primitive
     * vectors (not the layer_3 AtomicOrbitalLabel struct) per this file's
     * no-layer-5/layer-3-dependency design above.
     */

    /**
     * Dimensions: [number_of_kpoints][number_of_bands][number_of_orbitals].
     * Same single-spin-component limitation as eigenvalues_in_ry.
     */
    std::vector<std::vector<std::vector<Real>>> orbital_projections;

    /**
     * Per-orbital labels, parallel arrays indexed by orbital (last dimension
     * of orbital_projections above). Empty unless orbital_projections is.
     */
    std::vector<int> orbital_atom_index;
    std::vector<int> orbital_species_type_index;
    std::vector<std::string> orbital_shell_label;
    std::vector<int> orbital_angular_momentum;
    std::vector<int> orbital_magnetic_quantum_number;

    // ─── Serialization ───────────────────────────────────────────────────────

    /**
     * ESSE-legal canonical units for every value this struct serializes,
     * keyed by category (not per-field, since every energy-like field shares
     * one unit and likewise for forces/stress/eigenvalues) -- see
     * TODOs.md §3.1 "Add units fields". Values chosen to match what the
     * existing unit-suffixed JSON key names (total_energy_ry, etc.) already
     * imply, so this is purely additive documentation of behavior that
     * hasn't changed, not a unit change.
     */
    static nlohmann::json units_json() {
        return {
            {"energy", "Ry"},
            {"forces", "Ry/bohr"},
            {"stress", "kbar"},
            {"pressure", "kbar"},
            {"magnetization", "uB"},
            {"eigenvalues", "Ry"},
        };
    }

    [[nodiscard]] nlohmann::json execution_json() const {
        return {
            {"number_of_processes", number_of_processes},
            {"number_of_threads_per_process", number_of_threads_per_process},
            {"number_of_kpoint_pools", number_of_kpoint_pools},
            {"band_group_size", band_group_size},
            {"reciprocal_space_vectors_are_distributed", reciprocal_space_vectors_are_distributed},
            {"peak_resident_memory_bytes", peak_resident_memory_bytes},
        };
    }

    void add_force_contribution(const std::string& name,
                                const std::vector<Vector3D>& values_in_ry_per_bohr) {
        force_contributions.push_back({name, values_in_ry_per_bohr});
    }

    [[nodiscard]] nlohmann::json force_contributions_json() const {
        nlohmann::json contributions_json;
        for (const auto& contribution : force_contributions) {
            nlohmann::json vectors_json = nlohmann::json::array();
            for (const auto& force_vector : contribution.values_in_ry_per_bohr) {
                vectors_json.push_back({force_vector.x, force_vector.y, force_vector.z});
            }
            contributions_json[contribution.name] = vectors_json;
        }
        return contributions_json;
    }

    [[nodiscard]] nlohmann::json hubbard_occupations_json() const {
        nlohmann::json occupations_json = nlohmann::json::array();
        for (const auto& occupation : hubbard_occupations) {
            occupations_json.push_back({
                {"atom_index", occupation.atom_index},
                {"occupation_trace", occupation.occupation_trace_per_spin},
            });
        }
        return occupations_json;
    }

    /**
     * ─── Band edges ─────────────────────────────────────────────────────────
     *
     * The highest eigenvalue at or below the Fermi energy, the lowest above it,
     * and the distance between them. Derived here rather than in each flavor
     * because every flavor that reports eigenvalues already reports the Fermi
     * energy beside them, and a second derivation would be a second definition:
     * the Python band-structure check carried one of its own, and a consumer
     * reading the results file had to invent a third.
     *
     * The Fermi energy is the divider, so the answer is only as good as it is:
     * across a gap it is the mid-gap value and the edges are the real ones; in
     * a metal it lies inside a band and the "gap" comes out as the local level
     * spacing, which is the honest report of "no gap" for a discrete k-point
     * sampling. Both edges have to exist for a gap to be reported at all --
     * a run whose bands are all occupied has no conduction edge to measure to.
     */
    struct BandEdges {
        bool has_valence_band_maximum = false;
        bool has_conduction_band_minimum = false;
        Real valence_band_maximum_in_ry = 0.0;
        Real conduction_band_minimum_in_ry = 0.0;
    };

    [[nodiscard]] BandEdges band_edges() const {
        BandEdges edges;
        Real valence_band_maximum_in_ry = -std::numeric_limits<Real>::infinity();
        Real conduction_band_minimum_in_ry = std::numeric_limits<Real>::infinity();
        for (const auto& kpoint_eigenvalues : eigenvalues_in_ry) {
            for (const auto& spin_eigenvalues : kpoint_eigenvalues) {
                for (const Real eigenvalue_in_ry : spin_eigenvalues) {
                    if (eigenvalue_in_ry <= fermi_energy_in_ry) {
                        edges.has_valence_band_maximum = true;
                        valence_band_maximum_in_ry =
                            std::max(valence_band_maximum_in_ry, eigenvalue_in_ry);
                    } else {
                        edges.has_conduction_band_minimum = true;
                        conduction_band_minimum_in_ry =
                            std::min(conduction_band_minimum_in_ry, eigenvalue_in_ry);
                    }
                }
            }
        }
        if (edges.has_valence_band_maximum) {
            edges.valence_band_maximum_in_ry = valence_band_maximum_in_ry;
        }
        if (edges.has_conduction_band_minimum) {
            edges.conduction_band_minimum_in_ry = conduction_band_minimum_in_ry;
        }
        return edges;
    }

    void add_stress_contribution(const std::string& name, const Real (&values_in_kbar)[3][3]) {
        StressContribution contribution;
        contribution.name = name;
        for (int row_index = 0; row_index < 3; row_index++) {
            for (int column_index = 0; column_index < 3; column_index++) {
                contribution.values_in_kbar[row_index][column_index] =
                    values_in_kbar[row_index][column_index];
            }
        }
        stress_contributions.push_back(contribution);
    }

    [[nodiscard]] nlohmann::json stress_contributions_json() const {
        nlohmann::json contributions_json;
        for (const auto& contribution : stress_contributions) {
            nlohmann::json tensor_json = nlohmann::json::array();
            for (int row_index = 0; row_index < 3; row_index++) {
                nlohmann::json row_json = nlohmann::json::array();
                for (int column_index = 0; column_index < 3; column_index++) {
                    row_json.push_back(contribution.values_in_kbar[row_index][column_index]);
                }
                tensor_json.push_back(row_json);
            }
            contributions_json[contribution.name] = tensor_json;
        }
        return contributions_json;
    }

    /**
     * Append the isolated-system-correction block (if one was applied) to a
     * results JSON object: one nested "isolated_system_correction" object
     * keyed by scheme, so a future scheme (Martyna-Tuckerman, ...) adds terms
     * without new top-level keys. Term labels become "<label>_ry" keys.
     * Shared by write_results_json() and to_json_string().
     */
    /**
     * Record any known-missing force or stress term, so a consumer reading the
     * JSON sees the same thing a reader of the log would. Absent entirely when
     * nothing is missing, which keeps the common case clean.
     */
    void append_incomplete_derivative_json(nlohmann::json& json_output) const {
        if (forces_are_incomplete) {
            json_output["forces_are_incomplete"] = true;
            json_output["forces_incomplete_reason"] = forces_incomplete_reason;
        }
        if (stress_is_incomplete) {
            json_output["stress_is_incomplete"] = true;
            json_output["stress_incomplete_reason"] = stress_incomplete_reason;
        }
    }

    void append_isolated_system_correction_json(nlohmann::json& json_output) const {
        if (!isolated_system_correction_applied) {
            return;
        }
        nlohmann::json correction_json;
        correction_json["method"] = isolated_system_correction_method;
        Real correction_total_in_ry = 0.0;
        for (const auto& [label, value] : isolated_system_correction_terms_in_ry) {
            correction_json[label + "_ry"] = value;
            correction_total_in_ry += value;
        }
        correction_json["total_ry"] = correction_total_in_ry;
        correction_json["corrected_total_energy_ry"] = total_energy_in_ry + correction_total_in_ry;
        json_output["isolated_system_correction"] = correction_json;
    }

    /**
     * Build the flat JSON summary of all computed results.
     *
     * Values are stored in their native units (Ry for energies, Ry/bohr for
     * forces, kbar for stress) -- see units_json() above, also emitted as the
     * "units" key below.
     *
     * The single source of the results schema. write_results_json() and
     * to_json_string() are both thin wrappers around it, so the file a run
     * leaves on disk and the string the Python bindings return cannot drift
     * apart -- they used to be two hand-maintained copies of the same seventy
     * lines, and a key added to one would silently be missing from the other.
     */
    [[nodiscard]] nlohmann::json results_json() const {
        nlohmann::json json_output;
        json_output["flavor"] = flavor;
        json_output["converged"] = converged;
        if (number_of_self_consistent_field_iterations.has_value()) {
            json_output["self_consistent_field_iterations"] =
                number_of_self_consistent_field_iterations.value();
        }
        if (!timings.is_null() && !timings.empty()) {
            json_output["timings_ms"] = timings;
        }
        json_output["units"] = units_json();
        json_output["execution"] = execution_json();
        if (!provenance.is_null()) {
            json_output["provenance"] = provenance;
        }
        if (!material.is_null()) {
            json_output["material"] = material;
        }

        // Energies
        json_output["total_energy_ry"] = total_energy_in_ry;
        json_output["band_energy_ry"] = band_energy_in_ry;
        json_output["hartree_energy_ry"] = hartree_energy_in_ry;
        json_output["exchange_correlation_energy_ry"] = exchange_correlation_energy_in_ry;
        json_output["ewald_energy_ry"] = ewald_energy_in_ry;
        json_output["double_counting_correction_ry"] = double_counting_correction_in_ry;
        json_output["smearing_entropy_contribution_ry"] = smearing_entropy_contribution_in_ry;
        if (hubbard_energy_in_ry != 0.0) {
            json_output["hubbard_energy_ry"] = hubbard_energy_in_ry;
        }
        if (dispersion_energy_in_ry != 0.0) {
            json_output["dispersion_energy_ry"] = dispersion_energy_in_ry;
        }
        if (one_center_energy_in_ry != 0.0) {
            json_output["one_center_energy_ry"] = one_center_energy_in_ry;
        }

        append_isolated_system_correction_json(json_output);

        // Fermi energy
        json_output["fermi_energy_ry"] = fermi_energy_in_ry;

        if (magnetization) {
            json_output["total_magnetization_bohr_magneton"] =
                magnetization->total_in_bohr_magneton;
            json_output["absolute_magnetization_bohr_magneton"] =
                magnetization->absolute_in_bohr_magneton;
        }

        if (!hubbard_occupations.empty()) {
            json_output["hubbard_occupations"] = hubbard_occupations_json();
        }

        // SCF energy trajectory
        if (!scf_energies_in_ry.empty()) {
            json_output["scf_energies_ry"] = scf_energies_in_ry;
        }

        // Forces
        nlohmann::json forces_json = nlohmann::json::array();
        for (const auto& force_vector : forces_in_ry_per_bohr) {
            forces_json.push_back({force_vector.x, force_vector.y, force_vector.z});
        }
        json_output["forces_ry_per_bohr"] = forces_json;
        if (!force_contributions.empty()) {
            json_output["force_contributions_ry_per_bohr"] = force_contributions_json();
        }
        append_incomplete_derivative_json(json_output);

        // Stress
        nlohmann::json stress_json = nlohmann::json::array();
        for (int row_index = 0; row_index < 3; row_index++) {
            nlohmann::json stress_row = nlohmann::json::array();
            for (int column_index = 0; column_index < 3; column_index++) {
                stress_row.push_back(stress_in_kbar[row_index][column_index]);
            }
            stress_json.push_back(stress_row);
        }
        json_output["stress_kbar"] = stress_json;
        json_output["pressure_kbar"] = pressure_in_kbar;
        if (!stress_contributions.empty()) {
            json_output["stress_contributions_kbar"] = stress_contributions_json();
        }

        // Eigenvalues
        json_output["eigenvalues_ry"] = eigenvalues_in_ry;

        // Band edges, from the eigenvalues and the Fermi energy above
        const BandEdges edges = band_edges();
        if (edges.has_valence_band_maximum) {
            json_output["valence_band_maximum_ry"] = edges.valence_band_maximum_in_ry;
        }
        if (edges.has_conduction_band_minimum) {
            json_output["conduction_band_minimum_ry"] = edges.conduction_band_minimum_in_ry;
        }
        if (edges.has_valence_band_maximum && edges.has_conduction_band_minimum) {
            json_output["band_gap_ry"] =
                edges.conduction_band_minimum_in_ry - edges.valence_band_maximum_in_ry;
        }

        // Occupations
        if (!occupations.empty()) {
            json_output["occupations"] = occupations;
        }

        // K-point weights
        json_output["kpoint_weights"] = kpoint_weights;

        // Atomic-orbital projections
        if (!orbital_projections.empty()) {
            json_output["orbital_projections"] = orbital_projections;
            nlohmann::json orbital_labels_json = nlohmann::json::array();
            for (size_t orbital_index = 0; orbital_index < orbital_shell_label.size();
                 orbital_index++) {
                orbital_labels_json.push_back({
                    {"atom_index", orbital_atom_index[orbital_index]},
                    {"species_type_index", orbital_species_type_index[orbital_index]},
                    {"shell_label", orbital_shell_label[orbital_index]},
                    {"angular_momentum", orbital_angular_momentum[orbital_index]},
                    {"magnetic_quantum_number", orbital_magnetic_quantum_number[orbital_index]},
                });
            }
            json_output["orbital_labels"] = orbital_labels_json;
        }

        return json_output;
    }

    /**
     * Write the flat JSON summary of all computed results to a file.
     *
     * Format is defined by the Q3 results.json schema in the architecture
     * documentation.
     */
    void write_results_json(const std::string& output_path) const {
        write_json_file(output_path, results_json());
    }

    /**
     * Serialize all results to a JSON string.
     *
     * Returns the same content as write_results_json() but as an in-memory
     * string rather than a file.  Used by the nanobind Python binding to
     * return results directly without file I/O.
     */
    [[nodiscard]] std::string to_json_string() const {
        return results_json().dump(4) + "\n";
    }
};

} // namespace q3
