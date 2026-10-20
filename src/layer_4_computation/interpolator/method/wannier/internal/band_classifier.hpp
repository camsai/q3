#pragma once

/**
 * BandClassifier — which bands are kept exactly, which compete for the
 * remaining room, and which are thrown away, at each k-point.
 *
 * The entangled case has more bands in the window than Wannier functions, so
 * something must choose. This is the first half of that choice, and it is made
 * from two quantities the record already carries: the projectability
 * p_nk = sum_m |A_mn(k)|^2, and the eigenvalue.
 *
 *   frozen     p >= projectability_maximum, OR e <= E_F + frozen_window.
 *              Kept exactly; never rotated by the optimisation.
 *   discarded  not frozen, and p < projectability_minimum. Removed from the
 *              working basis entirely, so the optimiser cannot select it back.
 *   candidate  everything else; offered to the optimisation.
 *
 * The energy rule is not redundant with the projectability rule, and a metal
 * is why. The states at a transition metal's Fermi level are the
 * free-electron-like ones with the LOWEST projectability -- copper's sp band
 * at E_F is exactly such a state -- so freezing on projectability alone
 * discards them and leaves the model with a hole at the Fermi surface. That
 * failure is invisible to the mesh-point round trip, which is exact for any
 * subspace, and shows up only in the density of states at E_F.
 *
 * The energy window has a published value, E_F + 2 eV (Qiao, Pizzi, Marzari,
 * npj Comput. Mater. 9, 208 (2023)), and `automatic_frozen_window` applies it
 * to metal and insulator alike. It is the same window the band distance that
 * judges the model is measured on, and the two must coincide: a state the
 * distance weighs has to be in the model exactly, which only freezing
 * guarantees. Measured on silicon at 12x12x12, freezing the occupied manifold
 * alone gives a band distance of 220 meV -- every millivolt of it from the
 * lowest conduction band, which the subspace iteration does not pick on its
 * own -- against 14 meV at E_F + 2 eV and 5 meV at E_F + 3 eV. For an
 * insulator the engine's Fermi level sits in the gap, so the window reaches
 * the conduction states the distance is measured on and no further. An
 * explicit value overrides it, and a value below the whole manifold freezes
 * nothing by energy, which is how a test exercises the optimisation on its
 * own. Whether the system is a metal is still decided here -- a band whose
 * energies straddle the Fermi level across the mesh, below it at one k-point
 * and above it at another -- and reported, because a driver reads it.
 *
 * THE SINGULAR-GAUGE RULE. The energy rule and the projectability rule can
 * disagree: a state inside the window with a projectability below
 * `projectability_minimum` is one the trial orbitals cannot represent, and
 * freezing it puts a direction in the subspace that the projection A(k) has
 * no column for. The gauge U = A (A^dagger A)^(-1/2) is then singular by
 * construction, not by accident -- copper's norm-conserving dataset has no 4p
 * shell, its sp band above E_F has zero projectability at seventeen k-points,
 * and a 2 eV window froze it and collapsed the gauge every time. Such a band
 * is EXCLUDED from the freeze and reported, never frozen: the model has a
 * hole where the dataset has none of the character the state needs, and the
 * report says so at which band and how many k-points. It is a fact about the
 * dataset, and hiding it behind a hand-lowered window was the alternative.
 *
 * Feasibility is checked here rather than left to fail deep in the iteration:
 * at every k-point the frozen count must not exceed the number of Wannier
 * functions, and frozen plus candidate must reach it. Either violation is a
 * typed error naming the k-point and the counts, which is what a retry ladder
 * needs to act on.
 */

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/band_classification_thresholds.hpp"

namespace q3 {

class BandClassifier {
  public:
    /// Inverted thresholds describe no rule, and are refused here rather than at each k-point.
    explicit BandClassifier(BandClassificationThresholds thresholds) : thresholds_(thresholds) {
        if (!(thresholds.projectability_minimum < thresholds.projectability_maximum)) {
            throw std::invalid_argument("BandClassifier: projectability_minimum (" +
                                        std::to_string(thresholds.projectability_minimum) +
                                        ") must be below projectability_maximum (" +
                                        std::to_string(thresholds.projectability_maximum) + ")");
        }
    }

    /**
     * The classification of `record`'s captured window for as many Wannier
     * functions as it has trial orbitals. Its projectabilities and
     * eigenvalues are one row per k-point, both indexed by band.
     */
    [[nodiscard]] BandClassification classify(const WannierInputs& record) const {
        require_classifiable(record);
        const int number_of_wannier_functions = record.number_of_trial_orbitals;

        BandClassification classification;
        BandClassificationSummary& verdict = classification.summary;
        verdict.number_of_bands_crossing_fermi =
            bands_crossing_fermi(record.eigenvalues_rydberg, record.fermi_energy_rydberg);
        verdict.metallic = verdict.number_of_bands_crossing_fermi > 0;
        verdict.frozen_window_automatic = thresholds_.automatic_frozen_window;
        verdict.frozen_window_above_fermi_rydberg =
            thresholds_.automatic_frozen_window
                ? BandClassificationThresholds::AUTOMATIC_WINDOW_RYDBERG
                : thresholds_.frozen_window_above_fermi_rydberg;
        const Real frozen_ceiling =
            record.fermi_energy_rydberg + verdict.frozen_window_above_fermi_rydberg;
        std::map<int, int> kpoints_excluded_from_freeze;

        classification.kpoints.reserve(record.projectabilities.size());
        for (std::size_t kpoint_index = 0; kpoint_index < record.projectabilities.size();
             kpoint_index++) {
            const std::vector<Real>& projectability = record.projectabilities[kpoint_index];
            const std::vector<Real>& energies = record.eigenvalues_rydberg[kpoint_index];

            KPointBandClassification entry;
            entry.roles.reserve(projectability.size());
            for (std::size_t band = 0; band < projectability.size(); band++) {
                const bool in_window = energies[band] <= frozen_ceiling;
                const BandRole role = role_of(projectability[band], in_window);
                entry.roles.push_back(role);
                if (role == BandRole::Discarded) {
                    if (in_window) {
                        kpoints_excluded_from_freeze[static_cast<int>(band)]++;
                    }
                    continue;
                }
                if (role == BandRole::Frozen) {
                    entry.frozen_positions.push_back(static_cast<int>(entry.kept_bands.size()));
                }
                entry.kept_bands.push_back(static_cast<int>(band));
            }
            require_feasible(entry, kpoint_index, number_of_wannier_functions);
            classification.kpoints.push_back(std::move(entry));
        }
        for (const auto& [band_index, number_of_kpoints] : kpoints_excluded_from_freeze) {
            verdict.bands_excluded_from_freeze.push_back({band_index, number_of_kpoints});
        }
        return classification;
    }

    /**
     * How many bands lie below the Fermi level at one k-point and above it at
     * another, by more than the crossing margin on each side. Bands are
     * counted by index across the k-points that carry them, so a window whose
     * band count varies between k-points is judged on the bands every k-point
     * has.
     */
    [[nodiscard]] static int
    bands_crossing_fermi(const std::vector<std::vector<Real>>& eigenvalues_rydberg,
                         Real fermi_energy_rydberg) {
        std::size_t common_bands = eigenvalues_rydberg.empty() ? 0 : eigenvalues_rydberg[0].size();
        for (const std::vector<Real>& energies : eigenvalues_rydberg) {
            common_bands = energies.size() < common_bands ? energies.size() : common_bands;
        }
        int crossing = 0;
        for (std::size_t band = 0; band < common_bands; band++) {
            bool below = false;
            bool above = false;
            for (const std::vector<Real>& energies : eigenvalues_rydberg) {
                below = below ||
                        energies[band] < fermi_energy_rydberg -
                                             BandClassificationThresholds::CROSSING_MARGIN_RYDBERG;
                above = above ||
                        energies[band] > fermi_energy_rydberg +
                                             BandClassificationThresholds::CROSSING_MARGIN_RYDBERG;
            }
            if (below && above) {
                crossing++;
            }
        }
        return crossing;
    }

  private:
    /**
     * The rule for one band: frozen by a projectability at the maximum or
     * above, or by lying inside the energy window while the trial orbitals can
     * reach it; a candidate when they can reach it and neither freezes it;
     * discarded otherwise -- which, inside the window, is the singular-gauge
     * rule keeping a state out of the freeze.
     */
    [[nodiscard]] BandRole role_of(Real projectability, bool in_window) const {
        const bool reachable = projectability >= thresholds_.projectability_minimum;
        if (projectability >= thresholds_.projectability_maximum || (in_window && reachable)) {
            return BandRole::Frozen;
        }
        return reachable ? BandRole::Candidate : BandRole::Discarded;
    }

    /// The frozen bands must fit in the subspace, and the kept ones must be able to fill it.
    static void require_feasible(const KPointBandClassification& entry,
                                 std::size_t kpoint_index,
                                 int number_of_wannier_functions) {
        if (entry.number_frozen() > number_of_wannier_functions) {
            throw WannierDiagnosticError(
                WannierDiagnostic::frozen_window_overfull,
                "disentanglement: k-point " + std::to_string(kpoint_index) + " freezes " +
                    std::to_string(entry.number_frozen()) + " bands but only " +
                    std::to_string(number_of_wannier_functions) +
                    " Wannier functions were asked for. Lower "
                    "control.interpolation.frozen_window_above_fermi_ev, raise "
                    "projectability_maximum, or ask for more Wannier functions.");
        }
        if (entry.number_kept() < number_of_wannier_functions) {
            throw WannierDiagnosticError(
                WannierDiagnostic::kept_bands_underfull,
                "disentanglement: k-point " + std::to_string(kpoint_index) + " keeps only " +
                    std::to_string(entry.number_kept()) + " bands, fewer than the " +
                    std::to_string(number_of_wannier_functions) +
                    " Wannier functions asked for. Lower "
                    "control.interpolation.projectability_minimum, or widen "
                    "control.interpolation.bands so more states are available.");
        }
    }

    static void require_classifiable(const WannierInputs& record) {
        const std::vector<std::vector<Real>>& projectabilities = record.projectabilities;
        const std::vector<std::vector<Real>>& eigenvalues_rydberg = record.eigenvalues_rydberg;
        if (projectabilities.size() != eigenvalues_rydberg.size()) {
            throw std::invalid_argument("BandClassifier::classify: the projectabilities cover " +
                                        std::to_string(projectabilities.size()) +
                                        " k-points and the eigenvalues " +
                                        std::to_string(eigenvalues_rydberg.size()));
        }
        if (projectabilities.empty()) {
            throw std::invalid_argument("BandClassifier::classify: no k-points");
        }
        if (record.number_of_trial_orbitals <= 0) {
            throw std::invalid_argument(
                "BandClassifier::classify: number_of_wannier_functions must be positive");
        }
        for (std::size_t kpoint_index = 0; kpoint_index < projectabilities.size(); kpoint_index++) {
            if (projectabilities[kpoint_index].size() != eigenvalues_rydberg[kpoint_index].size()) {
                throw std::invalid_argument(
                    "BandClassifier::classify: k-point " + std::to_string(kpoint_index) + " has " +
                    std::to_string(projectabilities[kpoint_index].size()) +
                    " projectabilities and " +
                    std::to_string(eigenvalues_rydberg[kpoint_index].size()) + " eigenvalues");
            }
        }
    }

    BandClassificationThresholds thresholds_;
};

} // namespace q3
