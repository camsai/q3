#pragma once

/**
 * WannierInputs — everything the Wannier construction needs from a
 * plane-wave calculation, as plain data: the k-mesh and its neighbour
 * shells, the overlaps M_mn(k,b), the trial-orbital projections A_mn(k),
 * the eigenvalues, and the projectabilities.
 *
 * Mirrors the design of io/mean_field_record/init.hpp: the record is
 * purely descriptive (no plane-wave basis, no pseudopotential set), so it
 * can come from a live NSCF run or be read back from the record a run saved
 * (`<prefix>.save/interpolation.q3ip`), and the same construction runs
 * either way. Unlike MeanFieldRecord it
 * additionally carries the trial-orbital projections and per-band
 * projectabilities that the augmentation-aware kernels
 * (PlanewaveBlochOverlapCalculator, PlanewaveTrialOrbitalProjectionCalculator)
 * compute from the live SCF/NSCF state -- MeanFieldRecord has no
 * pseudopotential set to derive them from.
 */

#include <array>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"

namespace q3 {

struct [[nodiscard]] WannierInputs {
    // ── k-grid ───────────────────────────────────────────────────────
    /**
     * Monkhorst-Pack grid dimensions this record's k-points were
     * generated on -- a full, unreduced grid: the construction sums over
     * every k-point's neighbours, and has no path through a symmetry-reduced
     * wedge.
     */
    std::array<int, 3> kpoint_grid_dimensions = {0, 0, 0};
    /**
     * Crystal (fractional) coordinates, one per physical k-point, in the
     * same enumeration order KPointsGrid::generate() produces.
     */
    std::vector<Vector3D> kpoint_coordinates;

    // ── Neighbor shells ──────────────────────────────────────────────
    /**
     * b-vector shells and finite-difference weights for this grid
     * (KPointNeighborShells::compute() output).
     */
    std::vector<KPointNeighborDirection> neighbor_directions;

    // ── Overlaps and projections ─────────────────────────────────────
    /**
     * M_mn(k,b) = <u_mk|u_n,k+b>, one entry per (k, b, m, n): indexed
     * [kpoint_index][direction_index][neighbor_band_index * number_of_bands
     * + band_index] -- the k-side band m varies fastest, i.e. each (k,b)
     * block is column-major with m as the row index, the dense-linear-
     * algebra convention the localizer expects and the order a `.mmn`
     * block is written in.
     */
    std::vector<std::vector<std::vector<Complex>>> overlap_matrices;
    /**
     * A_mn(k) = <psi_mk|S|g_n>, one entry per k-point: indexed
     * [kpoint_index][orbital_index * number_of_bands + band_index] -- the
     * band m varies fastest (column-major with m as the row index), the
     * layout PlanewaveWannierCapture packs, ProjectionGauge reads
     * and the `.amn` writer streams.
     */
    std::vector<std::vector<Complex>> projection_matrices;

    // ── Bands ────────────────────────────────────────────────────────
    int number_of_bands = 0;
    int number_of_trial_orbitals = 0;
    /// Eigenvalues in Rydberg, indexed [kpoint_index][band_index].
    std::vector<std::vector<Real>> eigenvalues_rydberg;
    /// p_nk = sum_m |A_mn(k)|^2, indexed [kpoint_index][band_index].
    std::vector<std::vector<Real>> projectabilities;
    Real fermi_energy_rydberg = 0.0;
    int number_of_spin_components = 1;
    /**
     * Where each trial orbital was placed, Cartesian, in BOHR, one per orbital
     * in orbital order -- the same frame and unit as the b-vectors, so
     * b . r is dimensionless without a conversion. These are the localiser's
     * guiding centres. Empty means the record predates them or was imported
     * from a format that does not carry them; the localiser then falls back to
     * the projection gauge's own centres and says so.
     */
    std::vector<Vector3D> trial_orbital_centers_bohr;

    [[nodiscard]] int number_of_kpoints() const {
        return static_cast<int>(kpoint_coordinates.size());
    }

    /// Whether the record carries M_mn(k,b): the spread and the subspace iteration need it.
    [[nodiscard]] bool has_overlaps() const {
        return !overlap_matrices.empty();
    }
    /// Whether the record carries p_nk: classification by projectability needs it.
    [[nodiscard]] bool has_projectabilities() const {
        return !projectabilities.empty();
    }
    /// Whether every trial orbital's position is known: the localiser's guiding centres.
    [[nodiscard]] bool has_trial_orbital_centers() const {
        return !trial_orbital_centers_bohr.empty() &&
               static_cast<int>(trial_orbital_centers_bohr.size()) == number_of_trial_orbitals;
    }
};

} // namespace q3
