#pragma once

/**
 * WannierQualityReport — what a run can say about its own model with no
 * reference calculation in hand.
 *
 * A driver that has to accept or reject a model unattended reads numbers, and
 * the ones that can be had for free -- from the model alone, in the time it
 * takes to evaluate it -- are three:
 *
 *   The spreads, function by function. The totals say how localised the set
 *   is; the per-function values say which one is not. The published
 *   no-reference criterion (the automated Wannier90 workflows judge a model
 *   with no bands to compare against by the VARIANCE of its spreads) is
 *   reported, with two outlier flags beside it.
 *
 *   The centres against the atoms. A Wannier function built from atomic
 *   projections sits on its atom, one built from bond projections on its bond;
 *   a centre far from both is the classic sign of a model that will
 *   interpolate badly, and it is visible without a band structure.
 *
 *   How far H(R) has decayed by the edge of its domain (HamiltonianDecay).
 *
 * Plain data; WannierQualityAssessor measures it. The lengths are in bohr, as
 * the record's lattice is, and the field names say so.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/local_orbital/hamiltonian_decay.hpp"

namespace q3 {

struct [[nodiscard]] WannierQualityReport {
    std::vector<Real> spreads_bohr_squared;
    Real spread_median_bohr_squared = 0.0;
    /// Population variance of the spreads across functions, the no-reference criterion.
    Real spread_variance_bohr_squared = 0.0;
    Real spread_maximum_bohr_squared = 0.0;
    /// Zero-based indices of functions far above the median spread.
    std::vector<int> spread_outliers;
    /// Zero-based indices of functions above ten square angstrom.
    std::vector<int> spreads_above_ten_angstrom_squared;

    /**
     * Per function, the distance to the nearest atom over all periodic images; empty without
     * atoms.
     */
    std::vector<Real> nearest_atom_distance_bohr;
    /// Per function, the distance to the nearest midpoint of a bond; empty without atoms.
    std::vector<Real> nearest_bond_midpoint_distance_bohr;
    /// The shortest interatomic distance, which defines what a bond is here.
    Real shortest_bond_bohr = 0.0;

    HamiltonianDecay decay;
};

} // namespace q3
