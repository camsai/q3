#pragma once

/**
 * HubbardOccupationMixer — damping and Anderson acceleration for the DFT+U
 * occupation matrix.
 *
 * Q3 rebuilds n from the output states every iteration and feeds it straight
 * to V_U. Undamped, that is a feedback loop the density mixer has to fight,
 * and it is why FeO with U on Fe 3d ran 100 iterations without converging.
 * Damping fixes that but is not free: a damped n lags the fresh one, the SCF
 * stops while the lag is threshold-sized, and E_U reads the lagging matrix --
 * a *first* order energy error, because E_U is not stationary in n the way the
 * total energy is in the density.
 *
 * Both of this class's decisions follow from that tension:
 *
 *   - damping switches on by *evidence* (non-monotone progress), so a system
 *     that never needed it takes the fresh matrix bit-for-bit and pays nothing;
 *   - the Anderson step switches off *before* the fixed point, so the final
 *     approach runs on the plain damped path the recorded references were
 *     measured on.
 *
 * Kept apart from the SCF iterator so it can be exercised on synthetic
 * matrices: everything here is a pure function of the sequence of matrices it
 * is handed, with no plane waves, k-points or Hamiltonian in sight.
 */

#include <algorithm>
#include <complex>
#include <limits>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/// Per-manifold, per-spin occupation matrices, each flattened row-major.
using HubbardOccupationMatrices = std::vector<std::vector<std::vector<Complex>>>;

class HubbardOccupationMixer {
  public:
    /**
     * Blend `fresh` into the retained state and return what should drive V_U.
     *
     * `mixing_beta` is the damping factor once damping is active;
     * `convergence_threshold` gates both SCF exit (via `residual()`) and the
     * point at which the Anderson step stands down.
     */
    const HubbardOccupationMatrices&
    blend(const HubbardOccupationMatrices& fresh, Real mixing_beta, Real convergence_threshold) {
        if (!matches_layout(damped_, fresh)) {
            /**
             * First call, or a manifold layout change mid-run. Both caches are
             * indexed the same way, so both are dropped rather than indexed
             * into while mismatched.
             */
            damped_ = fresh;
            residuals_.clear();
            residual_ = std::numeric_limits<Real>::max();
            return damped_;
        }

        /**
         * How far n still is from its own fixed point, measured before
         * anything is blended: ||n_fresh - n_damped||^2, zero exactly when the
         * freshly accumulated matrix already agrees with the one driving V_U.
         * Same squared-magnitude convention `density_residual_` uses.
         */
        Real mismatch = 0.0;
        for_each_element(fresh, [&](size_t manifold, size_t spin, size_t element) {
            mismatch +=
                std::norm(fresh[manifold][spin][element] - damped_[manifold][spin][element]);
        });

        /**
         * What earns the damping is non-monotone progress: an occupation
         * matrix that moves *further* from its fixed point than it was the
         * iteration before. Monotone runs record zero such events ever; FeO,
         * which does not converge in 100 iterations undamped, trips the
         * counter almost immediately. Counted cumulatively rather than
         * consecutively, because an oscillation alternates rather than growing
         * every step.
         */
        if (!damping_active_) {
            /**
             * ">=", not ">": while damping is off the mixer passes the fresh
             * matrix straight through, so a clean period-2 oscillation
             * produces a *constant* mismatch rather than a growing one. Firing
             * only on strict growth would leave exactly the sustained
             * oscillation this exists to break undetected forever. Anything
             * that fails to improve counts.
             */
            if (mismatch >= previous_mismatch_)
                ++nonmonotone_iterations_;
            if (nonmonotone_iterations_ >= constants::hubbard_damping_trigger_iterations)
                damping_active_ = true;
        }
        previous_mismatch_ = mismatch;
        residual_ = mismatch;

        Real damping = damping_active_ ? mixing_beta : 1.0;

        /**
         * Anderson accelerates the hard middle of the SCF and then steps
         * aside. Its extrapolation changes the path n takes, and a path change
         * moves where the run stops by the size of the convergence threshold --
         * which is the scale cross-basis energies are compared at. Restricting
         * it to iterations where n is still further out than that threshold
         * keeps the acceleration where the cost is and leaves the approach to
         * the fixed point on the plain damped path.
         */
        bool use_anderson = damping_active_ && matches_layout(residuals_, fresh) &&
                            mismatch > convergence_threshold;

        Real anderson_theta = 0.0;
        if (use_anderson) {
            /**
             * Two-point secant: <r, r - r_prev> / |r - r_prev|^2, the one-term
             * quasi-Newton step. Clamped, because the secant is exact only
             * where the residual map is locally linear and can otherwise ask
             * for an arbitrarily long step; at the clamp the update is still a
             * damped step in the descent direction rather than a leap.
             */
            Real numerator = 0.0;
            Real denominator = 0.0;
            for_each_element(fresh, [&](size_t manifold, size_t spin, size_t element) {
                Complex residual =
                    fresh[manifold][spin][element] - damped_[manifold][spin][element];
                Complex residual_change = residual - residuals_[manifold][spin][element];
                numerator += (std::conj(residual) * residual_change).real();
                denominator += std::norm(residual_change);
            });
            anderson_theta = denominator > constants::hubbard_anderson_secant_minimum_denominator
                                 ? numerator / denominator
                                 : 0.0;
            anderson_theta =
                std::max(-constants::hubbard_anderson_theta_limit,
                         std::min(constants::hubbard_anderson_theta_limit, anderson_theta));
        }

        // Stash this iteration's residual before the update overwrites `damped_`.
        residuals_.resize(fresh.size());
        for (size_t manifold = 0; manifold < fresh.size(); ++manifold) {
            residuals_[manifold].resize(fresh[manifold].size());
            for (size_t spin = 0; spin < fresh[manifold].size(); ++spin) {
                std::vector<Complex> residual(fresh[manifold][spin].size());
                for (size_t element = 0; element < residual.size(); ++element)
                    residual[element] =
                        fresh[manifold][spin][element] - damped_[manifold][spin][element];
                for (size_t element = 0; element < residual.size(); ++element) {
                    Complex step = damping * residual[element];
                    if (use_anderson)
                        step *= (1.0 - anderson_theta);
                    damped_[manifold][spin][element] += step;
                }
                residuals_[manifold][spin] = std::move(residual);
            }
        }
        return damped_;
    }

    /// ||n_fresh - n_damped||^2 from the last blend; huge until one has run.
    [[nodiscard]] Real residual() const {
        return residual_;
    }

    /// Whether non-monotone progress has switched damping on for this run.
    [[nodiscard]] bool damping_active() const {
        return damping_active_;
    }

    /// The retained damped state, for callers that need it after `blend`.
    [[nodiscard]] const HubbardOccupationMatrices& damped() const {
        return damped_;
    }

  private:
    /**
     * Whether a cache is indexed exactly like `fresh`, all three levels deep.
     * The outer size alone is not enough: a layout change can leave the
     * manifold count matching while an inner row is shorter, and both caches
     * are indexed element by element.
     */
    [[nodiscard]] static bool matches_layout(const HubbardOccupationMatrices& cache,
                                             const HubbardOccupationMatrices& fresh) {
        if (cache.size() != fresh.size())
            return false;
        for (size_t manifold = 0; manifold < fresh.size(); ++manifold) {
            if (cache[manifold].size() != fresh[manifold].size())
                return false;
            for (size_t spin = 0; spin < fresh[manifold].size(); ++spin)
                if (cache[manifold][spin].size() != fresh[manifold][spin].size())
                    return false;
        }
        return true;
    }

    template <typename Visitor>
    static void for_each_element(const HubbardOccupationMatrices& matrices, Visitor&& visit) {
        for (size_t manifold = 0; manifold < matrices.size(); ++manifold)
            for (size_t spin = 0; spin < matrices[manifold].size(); ++spin)
                for (size_t element = 0; element < matrices[manifold][spin].size(); ++element)
                    visit(manifold, spin, element);
    }

    HubbardOccupationMatrices damped_;
    HubbardOccupationMatrices residuals_;
    Real previous_mismatch_{std::numeric_limits<Real>::max()};
    Real residual_{std::numeric_limits<Real>::max()};
    int nonmonotone_iterations_ = 0;
    bool damping_active_ = false;
};

} // namespace q3
