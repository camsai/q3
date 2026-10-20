#pragma once

/**
 * BroydenMixer — modified Broyden (type-2) density mixer in reciprocal space.
 *
 * Implements the modified Broyden mixing scheme for SCF convergence
 * acceleration, operating in reciprocal space with metric preconditioning.
 * Maintains a history of density residuals and uses quasi-Newton updates
 * to improve convergence.
 */

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/iterator/internal/mixing/reciprocal_space/init.hpp"

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/linear_algebra/lapack.hpp"

namespace q3 {
using namespace q3::lapack;

/**
 * Modified Broyden mixing in G-space with Hartree-metric inner product.
 *
 * Implements D.D. Johnson's modified Broyden algorithm (PRB 38, 12807, 1988).
 * Preconditioning (e.g. Kerker/Thomas-Fermi, PRB 23, 3082, 1981) is applied to
 * the residuals BEFORE they enter the Broyden history, so the Gram matrix is
 * built in preconditioned space. That placement is this implementation's own
 * choice: it keeps long-wavelength charge-sloshing modes out of the Broyden
 * subspace itself, not merely out of the final search direction.
 *
 * Metric weights are supplied through initialize() before the first mix() call.
 */
class BroydenMixer : public ReciprocalSpaceMixer {
  public:
    explicit BroydenMixer(Real mixing_parameter = 0.7, int history_depth = 8)
        : ReciprocalSpaceMixer(mixing_parameter, history_depth) {}

    /**
     * Perform one mixing step in G-space.
     *
     * One step of the Johnson modified-Broyden update, in the order applied:
     *   1. Compute raw residual F = ρ_out - ρ_in
     *   2. Apply Kerker preconditioning: F_prec = K(G) · F  (if enabled)
     *   3. Compute delta vectors in preconditioned space:
     *        ΔF = F_prec_prev - F_prec   (preconditioned residual delta)
     *        Δρ = ρ_in_prev - ρ_in       (raw density delta)
     *   4. Precompute the metric-weighted twin of each ΔF (build_metric_scaled_copy)
     *   5. Build Gram matrix β_{ij} = ⟨ΔF_i|ΔF_j⟩_metric  (no w0² regularization)
     *   6. Solve for γ_i: γ = β⁻¹ · ⟨ΔF_i|F_prec⟩_metric
     *   7. Correct: ρ_in -= Σ γ_i Δρ_i;  F_prec -= Σ γ_i ΔF_i
     *   8. Mix: ρ_new = ρ_in + α · F_prec
     *
     * Step 2 ahead of step 3 is what makes Kerker filter long-wavelength modes
     * out of the Broyden subspace, not just out of the final search direction:
     * the stored history lives entirely in preconditioned space. The pairs are
     * stored raw, with no normalization -- see the storage site below.
     */
    void mix_in_reciprocal_space(std::vector<std::vector<Complex>>& rho_in_g,
                                 const std::vector<std::vector<Complex>>& rho_out_g,
                                 int number_of_spin_components) override {
        iteration_counter_++;

        int number_of_spin = static_cast<int>(rho_in_g.size());
        int number_of_g_vectors = static_cast<int>(rho_in_g[0].size());

        /**
         * Determine the boundary between smooth and hard G-vectors.
         * Broyden operates only on [0:smooth_g_vector_count], hard G-vectors get linear mixing.
         */
        int smooth_g_vector_count =
            (number_of_smooth_g_vectors_ > 0 && number_of_smooth_g_vectors_ < number_of_g_vectors)
                ? number_of_smooth_g_vectors_
                : number_of_g_vectors;

        // Compute raw residual F = ρ_out - ρ_in (full G range for linear mixing of hard G)
        MixingRecord raw_residual;
        raw_residual.data.resize(number_of_spin);
        for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
            raw_residual.data[spin_index].resize(number_of_g_vectors);
            for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
                raw_residual.data[spin_index][g_index] =
                    rho_out_g[spin_index][g_index] - rho_in_g[spin_index][g_index];
            }
        }

        /**
         * Residual norm ⟨F|F⟩ in the shared two-channel metric, restricted to the
         * smooth G-vector set -- the range the Broyden history spans. This value is
         * mixer-local: it drives the restart heuristic below, NOT the SCF convergence
         * test, which computes its own residual (see
         * PlanewaveKohnShamSCFIterator::compute_local_residual_between). The two use
         * the same metric and the same G range, so they are comparable.
         */
        Real residual_norm = compute_metric_dot_product(
            raw_residual.data, raw_residual.data, number_of_spin, smooth_g_vector_count);
        last_residual_norm_ = residual_norm;

        // Track best residual for restart detection.
        if (residual_norm < best_residual_norm_ || best_residual_norm_ < 0.0) {
            best_residual_norm_ = residual_norm;
            stagnation_counter_ = 0;
        } else {
            stagnation_counter_++;
        }

        // Growth: residual norm > 5× best means the Jacobian approximation is degrading.
        bool growth_detected = best_residual_norm_ > 0.0 &&
                               residual_norm > 5.0 * best_residual_norm_ &&
                               iteration_counter_ > history_depth_;
        // Stagnation: no improvement in 2×ndim iterations.
        bool stagnation_detected =
            stagnation_counter_ >= 2 * history_depth_ && iteration_counter_ > history_depth_;

        if (growth_detected || stagnation_detected) {
            log::info("     Broyden mixer: {} (dr2={:.2e}, best={:.2e}, stagnation={}), "
                      "restarting history",
                      growth_detected ? "residual growth" : "stagnation",
                      residual_norm,
                      best_residual_norm_,
                      stagnation_counter_);
            delta_residuals_.clear();
            delta_inputs_.clear();
            metric_scaled_delta_residuals_.clear();
            has_previous_ = false;
            iteration_counter_ = 1;
            best_residual_norm_ = residual_norm;
            stagnation_counter_ = 0;
        }

        /**
         * The preconditioned residual F_prec enters the Broyden history,
         * so the Gram matrix is built in preconditioned space. This prevents
         * long-wavelength charge-sloshing modes from corrupting the subspace.
         * When preconditioning is disabled, F_prec = F (identity transform).
         */
        MixingRecord preconditioned_residual;
        preconditioned_residual.data.resize(number_of_spin);
        for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
            /**
             * Thomas-Fermi preconditioning (Kerker, or local-TF) screens the Coulomb response of
             * the TOTAL charge channel only. The magnetization channel (spin >= 1 in
             * the (total, m) representation) has no Coulomb kernel — damping its
             * long-wavelength components would freeze slow moment relaxation. The
             * preconditioner is the Thomas-Fermi screening of the Coulomb kernel,
             * and the magnetization has no Coulomb kernel to screen.
             */
            bool apply_preconditioning = (spin_index == 0) && (preconditioner_ != nullptr);
            preconditioned_residual.data[spin_index] = raw_residual.data[spin_index];
            if (apply_preconditioning) {
                preconditioner_->apply(preconditioned_residual.data[spin_index]);
            }
        }

        Real preconditioned_residual_norm = compute_metric_dot_product(preconditioned_residual.data,
                                                                       preconditioned_residual.data,
                                                                       number_of_spin,
                                                                       smooth_g_vector_count);

        auto log_diagnostics = [&](int number_of_history_entries) {
            log::debug("     Broyden diagnostics: raw_dr2={:.8e} preconditioned_dr2={:.8e} "
                       "history_entries={}",
                       residual_norm,
                       preconditioned_residual_norm,
                       number_of_history_entries);
        };

        int iterations_used = std::min(iteration_counter_ - 1, history_depth_);

        if (iteration_counter_ > 1 && has_previous_) {
            MixingRecord delta_residual_new, delta_input_new;
            delta_residual_new.data.resize(number_of_spin);
            delta_input_new.data.resize(number_of_spin);
            for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
                delta_residual_new.data[spin_index].resize(smooth_g_vector_count);
                delta_input_new.data[spin_index].resize(smooth_g_vector_count);
                for (int g_index = 0; g_index < smooth_g_vector_count; g_index++) {
                    // ΔF = F_prec_prev - F_prec_current (residual delta, preconditioned space)
                    delta_residual_new.data[spin_index][g_index] =
                        previous_preconditioned_residual_.data[spin_index][g_index] -
                        preconditioned_residual.data[spin_index][g_index];
                    // Δρ = ρ_in_prev - ρ_in_current (input-density delta, density space)
                    delta_input_new.data[spin_index][g_index] =
                        previous_input_.data[spin_index][g_index] - rho_in_g[spin_index][g_index];
                }
            }

            /**
             * Delta vectors are stored WITHOUT normalization: raw (un-normalized)
             * ΔF/Δρ vectors, with DSYTRF Bunch-Kaufman pivoting left to
             * handle near-singularity. Normalization was removed because it caused
             * gamma = ⟨ΔF̂|F⟩ to be unbounded: when convergence is slow (||F|| >> ||ΔF||),
             * gamma >> 1, leading to oscillatory divergence of the density.
             * Without normalization, gamma = ⟨ΔF|F⟩/||ΔF||² ≤ ||F||/||ΔF||, naturally bounded.
             */

            /**
             * Precompute the metric-weighted twin of ΔF once, when it enters history,
             * instead of re-applying the metric weight on every pairwise access below.
             * See build_metric_scaled_copy() for why this is safe (see TODOs.md §2.1
             * "Accelerate Broyden G-space operations").
             */
            MixingRecord scaled_delta_residual_new =
                build_metric_scaled_copy(delta_residual_new, number_of_spin, smooth_g_vector_count);

            // Store in circular buffer
            if (static_cast<int>(delta_residuals_.size()) < history_depth_) {
                delta_residuals_.push_back(std::move(delta_residual_new));
                delta_inputs_.push_back(std::move(delta_input_new));
                metric_scaled_delta_residuals_.push_back(std::move(scaled_delta_residual_new));
            } else {
                int position = (iteration_counter_ - 2) % history_depth_;
                delta_residuals_[position] = std::move(delta_residual_new);
                delta_inputs_[position] = std::move(delta_input_new);
                metric_scaled_delta_residuals_[position] = std::move(scaled_delta_residual_new);
            }
        }

        /**
         * Save current preconditioned residual and input for next iteration's delta computation.
         * A secant pair has to relate a residual to the input that produced it, so both are
         * snapshotted here, before the Broyden correction below overwrites rho_in_g in place.
         */
        previous_preconditioned_residual_.data.resize(number_of_spin);
        previous_input_.data.resize(number_of_spin);
        for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
            previous_preconditioned_residual_.data[spin_index].resize(smooth_g_vector_count);
            previous_input_.data[spin_index].resize(smooth_g_vector_count);
            for (int g_index = 0; g_index < smooth_g_vector_count; g_index++) {
                previous_preconditioned_residual_.data[spin_index][g_index] =
                    preconditioned_residual.data[spin_index][g_index];
                previous_input_.data[spin_index][g_index] = rho_in_g[spin_index][g_index];
            }
        }
        has_previous_ = true;

        // If we have history, do the Broyden update (smooth G-vectors only)
        if (iterations_used > 0 && !delta_residuals_.empty()) {
            int actual_history = static_cast<int>(delta_residuals_.size());

            /**
             * Build Gram matrix β_{ij} = ⟨ΔF_i|ΔF_j⟩_metric (preconditioned delta residuals).
             * One side of each pairwise dot product uses the precomputed metric-scaled
             * history entry (metric_scaled_delta_residuals_), so the metric weight is
             * applied once per history entry (O(history_depth)) rather than once per
             * pair (O(history_depth^2)) -- see build_metric_scaled_copy() below. The
             * actual reduction is a plain, branch-free complex dot product
             * (q3_complex_conjugate_dot_product, BLAS Level-1 style).
             */
            std::vector<Real> gram_matrix(actual_history * actual_history, 0.0);
            for (int history_row = 0; history_row < actual_history; history_row++) {
                for (int history_column = history_row; history_column < actual_history;
                     history_column++) {
                    Real dot =
                        dot_product_of_scaled_and_raw(metric_scaled_delta_residuals_[history_row],
                                                      delta_residuals_[history_column],
                                                      number_of_spin_components);
                    gram_matrix[history_row + history_column * actual_history] = dot;
                    gram_matrix[history_column + history_row * actual_history] = dot;
                }
            }

            /**
             * No diagonal regularization: the Gram matrix is inverted as built, with no w0²
             * added. Two mechanisms cover the near-singular case instead -- DSYTRF's
             * Bunch-Kaufman pivoting, and the fall-back to linear mixing on a non-zero
             * LAPACK info below. Adding w0² was incorrect and interacted badly with the
             * normalization that has also been removed.
             */

            /**
             * Invert β via the LAPACK symmetric-indefinite path: DSYTRF factorizes
             * with Bunch-Kaufman pivoting, DSYTRI inverts from that factorization
             * (DSYTRI's contract is to consume DSYTRF's output). Not Cholesky: β is
             * only positive SEMI-definite once the stored ΔF go nearly linearly
             * dependent.
             */
            std::vector<int> pivot_indices(actual_history);
            std::vector<Real> lapack_work(actual_history);
            int info;

            dsytrf_("U",
                    &actual_history,
                    gram_matrix.data(),
                    &actual_history,
                    pivot_indices.data(),
                    lapack_work.data(),
                    &actual_history,
                    &info);
            if (info != 0) {
                log::warn("Broyden mixing: DSYTRF failed (info={}), falling back to linear", info);
                apply_linear_mixing(
                    rho_in_g, preconditioned_residual, number_of_spin, number_of_g_vectors);
                log_diagnostics(actual_history);
                return;
            }

            dsytri_("U",
                    &actual_history,
                    gram_matrix.data(),
                    &actual_history,
                    pivot_indices.data(),
                    lapack_work.data(),
                    &info);
            if (info != 0) {
                log::warn("Broyden mixing: DSYTRI failed (info={}), falling back to linear", info);
                apply_linear_mixing(
                    rho_in_g, preconditioned_residual, number_of_spin, number_of_g_vectors);
                log_diagnostics(actual_history);
                return;
            }

            // Symmetrize the inverse (DSYTRI only fills upper triangle)
            for (int history_row = 0; history_row < actual_history; history_row++) {
                for (int history_column = history_row + 1; history_column < actual_history;
                     history_column++) {
                    gram_matrix[history_column + history_row * actual_history] =
                        gram_matrix[history_row + history_column * actual_history];
                }
            }

            /**
             * Compute projection γ_i = Σ_j β⁻¹_{ij} · ⟨ΔF_j | F_prec⟩_metric
             * The projection uses the preconditioned current residual, because that is the
             * space the history is stored in -- both sides of ⟨ΔF_j|F⟩ must live in the
             * same space for β⁻¹ to be the right operator to apply.
             * Reuses the same metric_scaled_delta_residuals_ entries built above --
             * no additional metric-weight application needed here.
             */
            std::vector<Real> projection(actual_history);
            for (int history_index = 0; history_index < actual_history; history_index++) {
                projection[history_index] =
                    dot_product_of_scaled_and_raw(metric_scaled_delta_residuals_[history_index],
                                                  preconditioned_residual,
                                                  number_of_spin_components);
            }

            /**
             * Apply the Broyden update over the smooth G-vectors only. On a dual grid the
             * hard G-vectors carry the augmentation charge; they stay outside the
             * quasi-Newton subspace and are linear-mixed by the caller
             * (ReciprocalSpaceMixer::mix) instead:
             *   ρ_in  -= Σ γ_i · Δρ_i      (density correction in density space)
             *   F_prec -= Σ γ_i · ΔF_i     (preconditioned residual correction)
             * Each inner update is a scaled vector addition (y += alpha*x), so it maps
             * directly onto BLAS ZAXPY rather than a hand-rolled per-element loop.
             */
            const int unit_stride = 1;
            for (int history_index = 0; history_index < actual_history; history_index++) {
                Real gamma = 0.0;
                for (int inner_index = 0; inner_index < actual_history; inner_index++) {
                    gamma += gram_matrix[inner_index + history_index * actual_history] *
                             projection[inner_index];
                }
                Complex negative_gamma(-gamma, 0.0);

                for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
                    blas::zaxpy_(&smooth_g_vector_count,
                                 &negative_gamma,
                                 delta_inputs_[history_index].data[spin_index].data(),
                                 &unit_stride,
                                 rho_in_g[spin_index].data(),
                                 &unit_stride);
                    blas::zaxpy_(&smooth_g_vector_count,
                                 &negative_gamma,
                                 delta_residuals_[history_index].data[spin_index].data(),
                                 &unit_stride,
                                 preconditioned_residual.data[spin_index].data(),
                                 &unit_stride);
                }
            }
        }

        /**
         * Final mixing step: ρ_new = ρ_in + α · F_prec (smooth G-vectors only).
         * Hard G-vectors are outside the Broyden subspace and are handled exclusively by
         * the outer ReciprocalSpaceMixer::mix() via beta-weighted linear mixing. Limiting
         * this call to smooth_g_vector_count prevents double-application of linear mixing
         * to hard G.
         */
        apply_linear_mixing(
            rho_in_g, preconditioned_residual, number_of_spin, smooth_g_vector_count);
        log_diagnostics(static_cast<int>(delta_residuals_.size()));
    }

    void reset() override {
        iteration_counter_ = 0;
        delta_residuals_.clear();
        delta_inputs_.clear();
        metric_scaled_delta_residuals_.clear();
        has_previous_ = false;
        best_residual_norm_ = -1.0;
        stagnation_counter_ = 0;
    }

    [[nodiscard]] std::string name() const override {
        return "broyden";
    }

  private:
    int iteration_counter_{0};

    struct MixingRecord {
        std::vector<std::vector<Complex>> data;
    };

    void apply_linear_mixing(std::vector<std::vector<Complex>>& rho_in_g,
                             const MixingRecord& residual,
                             int number_of_spin,
                             int number_of_g_vectors) {
        const int unit_stride = 1;
        Complex alpha(mixing_parameter_, 0.0);
        for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
            /**
             * Charge channel (spin 0): start from first_nonzero_g_index_ to preserve the
             * G=0 component (total charge). For standard (augmented) mixing, the G=0
             * residual is always zero (both input and output carry the same total charge
             * N_el/Ω), so this has no effect. For smooth-only mixing (US PP), the smooth
             * wf density has N_smooth < N_el at G=0; skipping G=0 prevents charge drift
             * through the Broyden linear mixing step.
             * Magnetization channel (spin 1 in the (total, m) representation): G=0 is the
             * total moment and MUST be mixed — the moment is not conserved and skipping
             * it freezes moment relaxation entirely.
             */
            int start_g_index = (spin_index == 0) ? first_nonzero_g_index_ : 0;
            int count = number_of_g_vectors - start_g_index;
            // rho_in_g += mixing_parameter_ * residual (BLAS ZAXPY over the active range).
            blas::zaxpy_(&count,
                         &alpha,
                         residual.data[spin_index].data() + start_g_index,
                         &unit_stride,
                         rho_in_g[spin_index].data() + start_g_index,
                         &unit_stride);
        }
    }

    /**
     * Per-(spin, G) scale factor of the Hartree-metric inner product implemented by
     * ReciprocalSpaceMixer::compute_metric_dot_product() -- see that function's doc
     * comment for the metric definition. Exposed here so the scale can be baked into
     * a history vector ONCE (see build_metric_scaled_copy()) instead of being
     * recomputed on every pairwise dot product in the O(history_depth^2) Gram matrix
     * loop. Called only O(history_depth) times per iteration (once per new history
     * entry), so the per-element branch here is not performance-sensitive.
     */
    Real metric_scale_factor(int spin_index, int g_index) const {
        Real gamma_factor = is_gamma_only_ ? 2.0 : 1.0;
        if (spin_index == 0) {
            return (g_index < first_nonzero_g_index_)
                       ? 0.0
                       : charge_metric_weights_[g_index] * gamma_factor;
        }
        Real g_factor = (g_index < first_nonzero_g_index_) ? 1.0 : gamma_factor;
        return magnetization_metric_weight_ * g_factor;
    }

    /**
     * Build a metric-scaled copy of a MixingRecord: scaled[s][g] = metric_scale_factor(s,g)
     * * record[s][g]. Since the metric weight is real, w(g)*Re(conj(a)*b) and
     * Re(conj(w(g)*a)*b) are the same value mathematically, so scaling one side of a
     * pairwise dot product reproduces the same weighted inner product as scaling neither
     * side and applying the weight per-element inside the dot loop -- see
     * dot_product_of_scaled_and_raw(). NOT guaranteed bit-identical to the original
     * per-pair-weighted formulation (floating-point multiplication does not distribute
     * exactly over addition for arbitrary weights), only mathematically equivalent to
     * numerical precision -- see this branch's regression tests for the tolerance this
     * is validated to. Charge-channel entries below first_nonzero_g_index_ are zeroed
     * (not skipped), which contributes exact 0.0 terms to the subsequent dot-product sum,
     * matching the ranged sum used elsewhere in this file.
     */
    MixingRecord
    build_metric_scaled_copy(const MixingRecord& record, int number_of_spin, int g_count) const {
        MixingRecord scaled;
        scaled.data.resize(number_of_spin);
        for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
            scaled.data[spin_index].resize(g_count);
            for (int g_index = 0; g_index < g_count; g_index++) {
                scaled.data[spin_index][g_index] =
                    metric_scale_factor(spin_index, g_index) * record.data[spin_index][g_index];
            }
        }
        return scaled;
    }

    /**
     * ⟨scaled_a | raw_b⟩ in the mixing metric, given scaled_a is already
     * metric-scaled (see build_metric_scaled_copy()). Reduces to a plain complex
     * conjugate dot product per spin channel -- a BLAS Level-1-style operation
     * (q3_complex_conjugate_dot_product, same helper used by
     * layer_0_core/execution/serial.hpp) with no per-element weight lookup or
     * branch, unlike ReciprocalSpaceMixer::compute_metric_dot_product().
     */
    Real dot_product_of_scaled_and_raw(const MixingRecord& scaled_a,
                                       const MixingRecord& raw_b,
                                       int number_of_spin) const {
        Real result = 0.0;
        for (int spin_index = 0; spin_index < number_of_spin; spin_index++) {
            int g_count = static_cast<int>(scaled_a.data[spin_index].size());
            Complex dot = blas::q3_complex_conjugate_dot_product(
                g_count, scaled_a.data[spin_index].data(), raw_b.data[spin_index].data());
            result += dot.real();
        }
        return result;
    }

    std::vector<MixingRecord> delta_residuals_;
    std::vector<MixingRecord> delta_inputs_;
    std::vector<MixingRecord> metric_scaled_delta_residuals_;
    MixingRecord previous_preconditioned_residual_;
    MixingRecord previous_input_;
    bool has_previous_{false};
    Real best_residual_norm_{-1.0};
    int stagnation_counter_{0};
};

} // namespace q3
