#pragma once

/**
 * SpreadReport — the spread and the centres at the final gauge, and how the
 * localisation that produced it ended.
 *
 * Reported whether or not the gauge was localised: the projection gauge has a
 * spread too, and a driver comparing two models wants it either way.
 * Evaluated whenever the record carries the neighbour overlaps and the mesh
 * has no two-point axis.
 */

#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/spread_terms.hpp"

namespace q3 {

struct [[nodiscard]] SpreadReport {
    /**
     * The report of a spread that WAS evaluated: `terms` at the final gauge,
     * what its phases were guided by, and the minimisable part at the
     * projection gauge. The localisation's own numbers are its caller's to
     * add; a gauge that was not localised leaves them at their defaults.
     */
    [[nodiscard]] static SpreadReport evaluated_as(const SpreadTerms& terms,
                                                   bool guided_by_trial_orbitals,
                                                   std::vector<Vector3D> trial_orbital_centers_bohr,
                                                   Real initial_minimizable_bohr_squared) {
        SpreadReport report;
        report.evaluated = true;
        report.gauge_invariant_bohr_squared = terms.gauge_invariant_bohr_squared;
        report.total_bohr_squared = terms.total_bohr_squared();
        report.centers_bohr = terms.centers_bohr;
        report.spreads_bohr_squared = terms.spreads_bohr_squared;
        report.guided_by_trial_orbitals = guided_by_trial_orbitals;
        report.trial_orbital_centers_bohr = std::move(trial_orbital_centers_bohr);
        report.initial_minimizable_bohr_squared = initial_minimizable_bohr_squared;
        return report;
    }

    /// Whether the numbers below were evaluated at all; false leaves them at their zeros.
    bool evaluated = false;
    /**
     * Omega_I, bohr squared. From the spread when it was evaluated, and
     * otherwise from the subspace iteration when there was one: it is the
     * one spread term a disentangled model has even on a mesh where the
     * centres are undefined.
     */
    Real gauge_invariant_bohr_squared = 0.0;
    Real total_bohr_squared = 0.0;
    std::vector<Vector3D> centers_bohr;
    /// Omega_n per function, bohr squared.
    std::vector<Real> spreads_bohr_squared;
    /// Whether the phase branches were fixed by the trial-orbital positions in the record.
    bool guided_by_trial_orbitals = false;
    /// The record's trial-orbital positions, Cartesian bohr; empty when it carries none.
    std::vector<Vector3D> trial_orbital_centers_bohr;
    /**
     * Omega_D + Omega_OD at the projection gauge, before any rotation: what
     * localisation bought is this minus the final value.
     */
    Real initial_minimizable_bohr_squared = 0.0;
    /// Whether the gauge was rotated by the spread minimiser at all.
    bool localized = false;
    int localization_iterations = 0;
    /**
     * Whether the minimisation actually reached a stationary gauge. False
     * after a stall, where the gauge is the best one found but not a minimum
     * -- and where calling the model "maximally localized" would be a claim
     * the run did not earn.
     */
    bool localization_converged = false;
    Real localization_gradient_norm = 0.0;
};

} // namespace q3
