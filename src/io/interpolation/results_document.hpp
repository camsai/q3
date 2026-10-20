#pragma once

/**
 * InterpolationResultsDocument — the results file of `q3_utility interpolate`,
 * assembled block by block onto the struct generated from its schema, and
 * written once.
 *
 * One contract, one serialisation. schemas/q3/interpolation_results.json is
 * the source of truth and src/generated/q3/interpolation_results.hpp its
 * C++ form; this class owns that struct and is the only place the engine's
 * types are mapped onto it, so nothing builds the file's JSON by hand. The
 * Python model, q3.io.interpolation_results, reads the same schema, and a
 * key renamed on one side without the other is caught by the parity test
 * there rather than by a driver at campaign speed.
 *
 * Every block written here is the producer's guarantee the schema states
 * with `required`. What the workflow adds afterwards -- quality.status,
 * .reason, .ladder, .band_distance -- is optional in the schema and never
 * set here.
 */

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "generated/q3/interpolation_results.hpp"
#include "io/json_file.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/local_orbital/hamiltonian_decay.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/quality_report.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/init.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/local_orbital.hpp"
#include "layer_5_properties/band_structure/interpolated/init.hpp"
#include "layer_5_properties/density_of_states/interpolated/init.hpp"

namespace q3 {

class InterpolationResultsDocument {
  public:
    /// A results file starts from what identifies the run it belongs to.
    InterpolationResultsDocument(nlohmann::json provenance, Real fermi_energy_rydberg) {
        results_.calculation = "interpolation";
        results_.provenance = std::move(provenance);
        results_.fermi_energy_ry = fermi_energy_rydberg;
    }

    /**
     * The model that was evaluated: its method, where its record is, and
     * what the interpolator says of itself. The Wigner-Seitz weight identity
     * has its own key, because the readers and the reference name it;
     * anything else an interpolator reports goes under `diagnostics`.
     */
    void describe_model(const std::string& method,
                        const std::string& record_path,
                        const QuantumStateInterpolator& interpolator) {
        generated::InterpolationResultsInterpolation& model = results_.interpolation;
        model.method = method;
        model.description = interpolator.description();
        model.record = record_path;
        model.number_of_bands = interpolator.number_of_bands();
        for (const auto& [name, value] : interpolator.diagnostics()) {
            if (name == LocalOrbitalQuantumStateInterpolator::TOTAL_BLOCH_SUM_WEIGHT) {
                model.total_bloch_sum_weight = value;
                continue;
            }
            if (!model.diagnostics) {
                model.diagnostics.emplace();
            }
            (*model.diagnostics)[name] = value;
        }
    }

    /**
     * The Wannier construction: the spread and the centres, reported whether
     * or not the gauge was localised; the classifier's verdict when the
     * subspace was disentangled; and what the model says about itself with
     * no reference bands in hand. The band distance against a direct
     * calculation is the workflow's to add, since it costs a calculation
     * the utility does not run; `quality` is where it lands.
     */
    void add_construction(const WannierConstructionReport& construction,
                          const WannierQualityReport& quality) {
        generated::InterpolationResultsInterpolation& model = results_.interpolation;
        model.spread = spread_block(construction.spread);
        if (construction.disentanglement.performed) {
            model.disentanglement = disentanglement_block(construction.disentanglement);
        }
        model.quality = quality_block(quality, construction.spread);
    }

    /// The k-points evaluated and the eigenvalues there, shaped [k-point][spin][band].
    void add_bands(const std::vector<Vector3D>& kpoints, const InterpolatedBandStructure& bands) {
        results_.kpoints = triplets(kpoints);
        std::vector<std::vector<std::vector<double>>> blocks;
        blocks.reserve(bands.eigenvalues_rydberg.size());
        for (const std::vector<Real>& at_kpoint : bands.eigenvalues_rydberg) {
            blocks.push_back({at_kpoint});
        }
        results_.eigenvalues_ry = std::move(blocks);
    }

    void add_density_of_states(const std::array<int, 3>& mesh,
                               Real broadening_rydberg,
                               const InterpolatedDensityOfStates& density) {
        generated::InterpolationResultsDensityOfStates block;
        block.mesh = {mesh[0], mesh[1], mesh[2]};
        block.broadening_ry = broadening_rydberg;
        block.energies_ry = density.energies_rydberg;
        block.dos = density.density_of_states;
        results_.density_of_states = std::move(block);
    }

    [[nodiscard]] const generated::InterpolationResults& results() const {
        return results_;
    }

    [[nodiscard]] nlohmann::json json() const {
        return results_;
    }

    void write(const std::string& path) const {
        write_json_file(path, json());
    }

  private:
    /// Vectors as the file stores them: one [x, y, z] per entry.
    [[nodiscard]] static std::vector<std::vector<double>>
    triplets(const std::vector<Vector3D>& vectors) {
        std::vector<std::vector<double>> rows;
        rows.reserve(vectors.size());
        for (const Vector3D& vector : vectors) {
            rows.push_back({vector.x, vector.y, vector.z});
        }
        return rows;
    }

    /// Zero-based indices as the file's integers.
    [[nodiscard]] static std::vector<std::int64_t> indices(const std::vector<int>& values) {
        return std::vector<std::int64_t>(values.begin(), values.end());
    }

    /// Only what was evaluated is written, and the localisation's numbers only after one.
    [[nodiscard]] static generated::InterpolationResultsSpread
    spread_block(const SpreadReport& report) {
        generated::InterpolationResultsSpread block;
        block.gauge_invariant_bohr_squared = report.gauge_invariant_bohr_squared;
        block.localized = report.localized;
        if (report.evaluated) {
            block.total_bohr_squared = report.total_bohr_squared;
            block.guiding_centers =
                report.guided_by_trial_orbitals ? "trial orbitals" : "projection gauge";
            block.trial_orbital_centers_bohr = triplets(report.trial_orbital_centers_bohr);
            block.initial_minimizable_bohr_squared = report.initial_minimizable_bohr_squared;
            block.centers_bohr = triplets(report.centers_bohr);
            block.spreads_bohr_squared = report.spreads_bohr_squared;
        }
        if (report.localized) {
            block.localization_iterations = report.localization_iterations;
            block.localization_converged = report.localization_converged;
            block.localization_gradient_norm = report.localization_gradient_norm;
        }
        return block;
    }

    /**
     * What the classifier decided: the window it resolved and which bands
     * the singular-gauge rule kept out of the freeze -- the one thing about
     * a model that a dataset without the right shell makes unavoidable, and
     * that should never be silent. Bands are one-based, as the manifest
     * counts them; the window is in electronvolts, as the manifest gives it.
     */
    [[nodiscard]] static generated::InterpolationResultsDisentanglement
    disentanglement_block(const DisentanglementReport& report) {
        const BandClassificationSummary& verdict = report.classification;
        generated::InterpolationResultsDisentanglement block;
        block.metallic = verdict.metallic;
        block.bands_crossing_fermi = verdict.number_of_bands_crossing_fermi;
        block.frozen_window_above_fermi_ev =
            verdict.frozen_window_above_fermi_rydberg * constants::RY_TO_EV;
        block.frozen_window_automatic = verdict.frozen_window_automatic;
        for (const BandExcludedFromFreeze& band : verdict.bands_excluded_from_freeze) {
            generated::InterpolationResultsExcludedBand excluded;
            excluded.band = band.band_index + 1;
            excluded.kpoints = band.number_of_kpoints;
            block.bands_excluded_from_freeze.push_back(excluded);
        }
        return block;
    }

    [[nodiscard]] static generated::InterpolationResultsHamiltonianDecay
    hamiltonian_decay_block(const HamiltonianDecay& decay) {
        generated::InterpolationResultsHamiltonianDecay block;
        block.outermost_shell_distance_bohr = decay.outermost_shell_distance;
        block.outermost_shell_size = decay.outermost_shell_size;
        block.outermost_shell_maximum_ry = decay.outermost_shell_maximum;
        block.largest_hopping_ry = decay.largest_hopping;
        block.ratio = decay.ratio;
        return block;
    }

    /// The spread statistics and the centres only when the report measured them; the decay always.
    [[nodiscard]] static generated::InterpolationResultsQuality
    quality_block(const WannierQualityReport& report, const SpreadReport& spread) {
        generated::InterpolationResultsQuality block;
        if (!report.spreads_bohr_squared.empty()) {
            generated::InterpolationResultsQualitySpread statistics;
            statistics.median_bohr_squared = report.spread_median_bohr_squared;
            statistics.variance_bohr_squared = report.spread_variance_bohr_squared;
            statistics.maximum_bohr_squared = report.spread_maximum_bohr_squared;
            statistics.outliers = indices(report.spread_outliers);
            statistics.above_ten_angstrom_squared =
                indices(report.spreads_above_ten_angstrom_squared);
            block.spread = statistics;
        }
        if (!report.nearest_atom_distance_bohr.empty()) {
            generated::InterpolationResultsQualityCenters centers;
            centers.nearest_atom_distance_bohr = report.nearest_atom_distance_bohr;
            centers.nearest_bond_midpoint_distance_bohr =
                report.nearest_bond_midpoint_distance_bohr;
            centers.shortest_bond_bohr = report.shortest_bond_bohr;
            block.centers = centers;
        }
        block.hamiltonian_decay = hamiltonian_decay_block(report.decay);
        if (spread.localized) {
            block.localization_converged = spread.localization_converged;
        }
        return block;
    }

    generated::InterpolationResults results_;
};

} // namespace q3
