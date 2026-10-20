#pragma once

/**
 * InterpolationSpecification — the `control.interpolation` block of an input.
 *
 * One block serves two steps of the same feature, and which step is meant is
 * decided by `control.calculation`, not by the block:
 *
 *  - On an NSCF run the block is a BUILD request: the engine constructs a
 *    small model of the band structure from its converged states -- for
 *    `method: wannier`, the interface matrices on the full k-grid -- and saves
 *    it as `<prefix>.save/interpolation.q3ip`, plus whatever interoperability
 *    formats `export_formats` lists. `grid` is required here: the k-points arrive as
 *    an explicit `control.kpoints_list`, which carries no grid shape of its
 *    own, and the neighbour arithmetic needs one.
 *  - On `calculation: interpolation` the block is an EVALUATION request,
 *    consumed by `q3_utility interpolate`, never by the engine: read the saved
 *    model and evaluate it along `control.band_path` / `control.kpoints_list`
 *    and, when `dos_mesh` is set, on that dense mesh for a density of states.
 *
 * Method-specific keys (`projection_centers`, `projection_decay_bohr`) sit in
 * the same block rather than in a per-method sub-block so that an input reads
 * as one decision -- "interpolate, this way". With one method there is nothing
 * to refuse yet; when a second lands, the parser owns the rule that a key the
 * chosen method does not use is an error rather than an ignored field.
 */

#include <array>
#include <map>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/band_classification_thresholds.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/disentanglement_settings.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/spread_minimization_settings.hpp"

namespace q3 {

struct InterpolationSpecification {
    /// "" when no interpolation was asked for; otherwise the method name ("wannier").
    std::string method;

    /// The full k-point grid `control.kpoints_list` enumerates; {0,0,0} = not stated.
    std::array<int, 3> grid = {0, 0, 0};

    /**
     * The 1-based inclusive band range the model is built from; {0,0} = not
     * stated, meaning the lowest bands.
     *
     * A range exists because the manifold worth modelling does not always
     * start at band 1. A pseudopotential that carries semicore states puts
     * them underneath it -- gallium with 3d in valence adds five bands below
     * GaAs's four sp valence bands -- and those lower bands are a different,
     * separately isolated manifold rather than something to interpolate
     * alongside. Selecting a range is NOT disentanglement: it picks a
     * contiguous set that must still be isolated from what surrounds it at
     * every k-point. A manifold that is entangled with its neighbours, as a
     * transition metal's d and s bands are, is not fixed by any choice of
     * range.
     */
    std::array<int, 2> bands = {0, 0};

    /**
     * Wannier: centres of the s-type trial orbitals in crystal coordinates.
     * Empty selects the pseudopotentials' own atom-centred orbitals -- the
     * wrong shape whenever the Wannier functions are not on atoms (silicon's
     * valence manifold lives on the bonds), so the example states them.
     */
    std::vector<Vector3D> projection_centers;
    /// Wannier: decay length of those orbitals in bohr (Wannier90's `zona` is 1/a).
    Real projection_decay_bohr = 1.0;

    /**
     * Build step: which of each species' pseudo-atomic shells become trial
     * orbitals, as 0-based indices in the order the pseudopotential file lists
     * its atomic wavefunctions. A species not named keeps every shell its
     * pseudopotential does not mark unusable (a negative occupation, or a
     * zero-norm radial function), which is the whole set and the default.
     *
     * An augmented dataset lists more than its atom's valence: a PAW file's
     * unoccupied second s and p partial waves, an ultrasoft file's empty 4s and
     * 4p projector channels. As trial orbitals they double the functions asked
     * for and the band window sized from them. The automatic workflow decides
     * which shells to leave out and writes them here; the engine honours the
     * list and does not guess.
     */
    std::map<std::string, std::vector<int>> trial_shells;

    /**
     * How the N states the gauge is built from are chosen out of the band
     * window: "" or "none" requires the window to be exactly those states,
     * "projectability" classifies and then optimises the subspace.
     */
    std::string disentanglement;
    /// The classifier's thresholds; the defaults are BandClassificationThresholds' own.
    Real projectability_minimum = BandClassificationThresholds{}.projectability_minimum;
    Real projectability_maximum = BandClassificationThresholds{}.projectability_maximum;
    /**
     * In electronvolts; see the schema for why this is not redundant with the
     * above. Read only when `frozen_window_is_automatic` is false.
     */
    Real frozen_window_above_fermi_ev = 0.0;
    /**
     * `frozen_window_above_fermi_ev: auto`, the default: the classifier
     * applies the published window for atomic projectors, E_F + 2 eV, to metal
     * and insulator alike (BandClassifier documents why the two are not
     * treated differently). A number given explicitly overrides it.
     */
    bool frozen_window_is_automatic = true;

    /**
     * Evaluation step: the spread minimiser's iteration budget. Three thousand
     * is far more than a well-posed case needs (copper converges in about
     * sixty); the retry ladder doubles it after a stall, which is the one
     * reason it is a key rather than a constant. The default is
     * SpreadMinimizationSettings' own.
     */
    int localization_maximum_iterations = SpreadMinimizationSettings{}.maximum_iterations;

    /**
     * Evaluation step: the subspace iteration's budget and its relative
     * convergence tolerance; the defaults are DisentanglementSettings' own. A disentanglement
     * that exhausts the budget throws with its last five spreads, and the retry
     * ladder loosens the tolerance to a margin above the change among them: the
     * iteration is deterministic, so the retry stops no later than the failure
     * did, where a larger budget would cost as much again.
     */
    int disentanglement_maximum_iterations = DisentanglementSettings{}.maximum_iterations;
    Real disentanglement_convergence_tolerance = DisentanglementSettings{}.convergence_tolerance;

    /**
     * How the gauge inside the chosen subspace is fixed.
     *
     *   auto     minimise the spread when the record carries the neighbour
     *            overlaps, and keep the projection gauge when it does not.
     *   maximal  minimise it, and fail if the record cannot support that.
     *   none     keep the projection gauge.
     *
     * The default is "auto" rather than "maximal" because the spread is defined
     * through M_mn(k,b), and a record imported from another code's projections
     * and eigenvalues alone does not carry them. Defaulting to a demand would
     * make every such record an error for asking for the default -- so the
     * default is a preference and "maximal" is the demand. Localising is the
     * preference because the projection gauge is exact at the mesh points
     * whatever it does, which means a model built without it looks perfect
     * where it was fitted and is only as good as its trial orbitals everywhere
     * else, which is where an interpolator is actually used.
     */
    std::string localization = "auto";

    /// Interoperability formats to write beside the native record ("wannier90").
    std::vector<std::string> export_formats;

    /// Evaluation step: the dense k-mesh for the interpolated DOS; {0,0,0} = none.
    std::array<int, 3> dos_mesh = {0, 0, 0};

    [[nodiscard]] bool requested() const {
        return !method.empty();
    }
    [[nodiscard]] bool has_grid() const {
        return grid[0] > 0 && grid[1] > 0 && grid[2] > 0;
    }
    [[nodiscard]] bool disentangles() const {
        return disentanglement == "projectability";
    }
    [[nodiscard]] bool has_band_selection() const {
        return bands[0] > 0 && bands[1] > 0;
    }
    /// 0-based index of the first selected band; 0 when no range was given.
    [[nodiscard]] int first_band_index() const {
        return has_band_selection() ? bands[0] - 1 : 0;
    }
    /// How many bands the range holds; 0 when no range was given.
    [[nodiscard]] int number_of_selected_bands() const {
        return has_band_selection() ? bands[1] - bands[0] + 1 : 0;
    }
    [[nodiscard]] bool wants_density_of_states() const {
        return dos_mesh[0] > 0 && dos_mesh[1] > 0 && dos_mesh[2] > 0;
    }
};

} // namespace q3
