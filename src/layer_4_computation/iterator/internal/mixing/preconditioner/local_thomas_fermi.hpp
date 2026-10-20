#pragma once

/**
 * LocalThomasFermiPreconditioner — Thomas-Fermi screening from the LOCAL density.
 *
 * Kerker screens every G with one number for the whole cell. That is the right
 * average answer for a homogeneous system and the wrong answer wherever the
 * density is not homogeneous: a slab screens like a metal inside the slab and
 * not at all in the vacuum above it, and a single q0 cannot describe both.
 * Local Thomas-Fermi replaces the constant q0² with a screening wavevector
 * built from the density at each point,
 *
 *     k_TF²(r) = 4 k_F(r) / π,    k_F(r) = (3 π² ρ(r))^(1/3)
 *
 * (Hartree atomic units: k_TF² = 4π e² N(E_F) with the free-electron density
 * of states N(E_F) = k_F/π², so the e² and the π² cancel to 4 k_F/π. The
 * expression is in bohr⁻² for ρ in bohr⁻³, with no further unit factor.)
 *
 * The preconditioned residual is then the solution of
 *
 *     [ G² + k_TF²(r) ] x = G² F
 *
 * where G² is diagonal in reciprocal space and k_TF²(r) is diagonal in real
 * space, so the operator is diagonal in neither and has to be inverted rather
 * than multiplied. Setting k_TF²(r) to a constant collapses this to
 * x = G²F/(G²+q0²), which is Kerker exactly -- the identity the unit tests pin.
 *
 * Method reference: Raczkowski, Canning & Wang, "Thomas-Fermi charge mixing for
 * obtaining self-consistency in density functional calculations", PRB 64,
 * 121101(R) (2001). The conjugate-gradient solve, its diagonal inner
 * preconditioner, the vacuum floor and the initial guess below are this
 * implementation's own choices and are not taken from any other code.
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_4_computation/iterator/internal/mixing/preconditioner/init.hpp"

namespace q3 {

/// Residual preconditioner solving [G² + k_TF²(r)] x = G² F by conjugate gradients.
class LocalThomasFermiPreconditioner : public Preconditioner {
  public:
    /**
     * @param g_squared              |G|² per G-vector, in bohr⁻².
     * @param q0_squared             floor for k_TF², in bohr⁻² (see below).
     * @param first_nonzero_g_index  index of the first G ≠ 0 component.
     * @param to_reciprocal_space    real-space → G-space density transform.
     * @param to_real_space          G-space → real-space density transform.
     * @param reduce_within_pool     sums values over the ranks sharing this
     *                               pool's real-space grid; empty when one
     *                               rank holds all of it.
     *
     * The floor is the existing `mixing.q0` key rather than a second knob. It
     * matters because ρ → 0 gives k_TF → 0, which is correct physics -- there is
     * nothing to screen in vacuum -- but leaves the operator singular in exactly
     * the cells (slabs, molecules in a box) this preconditioner exists to help.
     * Flooring at q0² recovers Kerker's behaviour in the vacuum region and the
     * local screening everywhere else.
     */
    LocalThomasFermiPreconditioner(
        std::vector<Real> g_squared,
        Real q0_squared,
        int first_nonzero_g_index,
        std::function<void(const std::vector<Real>&, std::vector<Complex>&)> to_reciprocal_space,
        std::function<void(const std::vector<Complex>&, std::vector<Real>&)> to_real_space,
        std::function<void(std::vector<Real>&)> reduce_within_pool = {})
        : g_squared_(std::move(g_squared))
        , q0_squared_(q0_squared)
        , first_nonzero_g_index_(first_nonzero_g_index)
        , to_reciprocal_space_(std::move(to_reciprocal_space))
        , to_real_space_(std::move(to_real_space))
        , reduce_within_pool_(std::move(reduce_within_pool)) {
        /**
         * Until the first update() the local screening is unknown, so the
         * operator falls back to the uniform q0² -- i.e. to Kerker. That keeps
         * apply() well defined if it is ever called before a density exists.
         */
        mean_screening_ = q0_squared_;
    }

    [[nodiscard]] std::string name() const override {
        return "local-tf";
    }

    /**
     * Recompute k_TF²(r) and its cell average from the spin-summed input density.
     *
     * The density is this rank's part of the real-space grid. Where ranks share
     * a pool that is only some of the planes, and the average taken over them
     * alone differed from rank to rank. The average sets the diagonal of the
     * conjugate-gradient solve, which then took different steps on different
     * ranks around the same collective transforms and hung. Sum and point
     * count are therefore completed within the pool before they are divided.
     */
    void update(const std::vector<Real>& density_real) override {
        screening_real_.resize(density_real.size());
        Real sum = 0.0;
        for (std::size_t index = 0; index < density_real.size(); ++index) {
            /**
             * FFT round-off leaves small negative values in low-density regions.
             * A negative density would give a negative k_TF² and an indefinite
             * operator, so clamp before the cube root rather than after.
             */
            Real density = std::max(density_real[index], 0.0);
            Real fermi_wavevector = std::cbrt(3.0 * M_PI * M_PI * density);
            Real screening = 4.0 * fermi_wavevector / M_PI;
            screening_real_[index] = std::max(screening, q0_squared_);
            sum += screening_real_[index];
        }
        std::vector<Real> sum_and_count = {sum, static_cast<Real>(density_real.size())};
        if (reduce_within_pool_) {
            reduce_within_pool_(sum_and_count);
        }
        mean_screening_ =
            sum_and_count[1] > 0.0 ? sum_and_count[0] / sum_and_count[1] : q0_squared_;
        has_screening_ = true;
    }

    void apply(std::vector<Complex>& residual_g) const override {
        /**
         * Asked of a flag rather than of screening_real_: a rank that holds no
         * real-space planes has an empty screening after update() like any
         * other, and taking the shortcut there would skip the collective
         * transforms every other rank runs below.
         */
        if (!has_screening_) {
            /**
             * No density supplied yet: fall back to the uniform-screening
             * answer G²F/(G²+q0²), which is what the operator reduces to anyway
             * -- Kerker exactly.
             */
            for (std::size_t g_index = first_index(); g_index < residual_g.size(); ++g_index) {
                residual_g[g_index] *=
                    g_squared_[g_index] / (g_squared_[g_index] + mean_screening_);
            }
            return;
        }

        std::size_t count = residual_g.size();
        std::vector<Complex> right_hand_side(count, Complex(0.0, 0.0));
        for (std::size_t g_index = first_index(); g_index < count; ++g_index) {
            right_hand_side[g_index] = g_squared_[g_index] * residual_g[g_index];
        }

        /**
         * Initial guess: the exact solution of the uniform-screening problem,
         * which is free and is already correct wherever k_TF² is close to its
         * cell average -- the error it leaves is only the *variation* of the
         * screening. This is deliberately not warm-started from the previous
         * SCF step: the residual shrinks by orders of magnitude as the SCF
         * converges, so a stale iterate has the wrong scale and is a worse
         * starting point than this one.
         */
        std::vector<Complex> solution(count, Complex(0.0, 0.0));
        apply_diagonal_inverse(right_hand_side, solution);

        std::vector<Complex> operator_times_solution(count, Complex(0.0, 0.0));
        apply_operator(solution, operator_times_solution);

        std::vector<Complex> gradient(count, Complex(0.0, 0.0));
        for (std::size_t g_index = first_index(); g_index < count; ++g_index) {
            gradient[g_index] = right_hand_side[g_index] - operator_times_solution[g_index];
        }

        Real right_hand_side_norm = std::sqrt(dot_product(right_hand_side, right_hand_side));
        if (right_hand_side_norm <= 0.0) {
            for (std::size_t g_index = first_index(); g_index < count; ++g_index) {
                residual_g[g_index] = solution[g_index];
            }
            return;
        }
        Real tolerance = relative_tolerance_ * right_hand_side_norm;

        std::vector<Complex> preconditioned_gradient(count, Complex(0.0, 0.0));
        apply_diagonal_inverse(gradient, preconditioned_gradient);
        std::vector<Complex> search_direction = preconditioned_gradient;
        Real gradient_dot = dot_product(gradient, preconditioned_gradient);

        std::vector<Complex> operator_times_direction(count, Complex(0.0, 0.0));
        int iteration = 0;
        for (; iteration < maximum_iterations_; ++iteration) {
            if (std::sqrt(dot_product(gradient, gradient)) <= tolerance)
                break;

            apply_operator(search_direction, operator_times_direction);
            Real curvature = dot_product(search_direction, operator_times_direction);
            if (curvature <= 0.0)
                break; // Not expected: the operator is positive definite on G ≠ 0.

            Real step = gradient_dot / curvature;
            for (std::size_t g_index = first_index(); g_index < count; ++g_index) {
                solution[g_index] += step * search_direction[g_index];
                gradient[g_index] -= step * operator_times_direction[g_index];
            }

            apply_diagonal_inverse(gradient, preconditioned_gradient);
            Real gradient_dot_new = dot_product(gradient, preconditioned_gradient);
            Real direction_weight = gradient_dot_new / gradient_dot;
            gradient_dot = gradient_dot_new;
            for (std::size_t g_index = first_index(); g_index < count; ++g_index) {
                search_direction[g_index] =
                    preconditioned_gradient[g_index] + direction_weight * search_direction[g_index];
            }
        }

        if (iteration == maximum_iterations_ &&
            std::sqrt(dot_product(gradient, gradient)) > tolerance) {
            /**
             * A truncated solve is not wrong, but it is no longer local-TF: it
             * sits somewhere between Kerker and the identity, and the SCF would
             * only show it as unexplained slow convergence. Say so.
             */
            log::info("  local-TF preconditioner: conjugate gradients hit the {}-iteration cap "
                      "(relative residual {:.3e}); the preconditioner is only partly applied",
                      maximum_iterations_,
                      std::sqrt(dot_product(gradient, gradient)) / right_hand_side_norm);
        }

        /**
         * Copy back the G != 0 components only. G = 0 is left exactly as the
         * caller passed it: it is fixed by charge conservation, and the solve
         * never represented it.
         */
        for (std::size_t g_index = first_index(); g_index < count; ++g_index) {
            residual_g[g_index] = solution[g_index];
        }
    }

    /// Cell-averaged k_TF², in bohr⁻². Exposed for tests.
    [[nodiscard]] Real mean_screening() const {
        return mean_screening_;
    }

  private:
    [[nodiscard]] std::size_t first_index() const {
        return static_cast<std::size_t>(first_nonzero_g_index_);
    }

    /**
     * Real inner product over the G ≠ 0 components.
     *
     * On a gamma-only half-sphere every retained component stands for the pair
     * (G, −G) and so carries the same weight 2. A uniform factor cancels in
     * every conjugate-gradient ratio, so it is left out rather than carried:
     * G = 0, the one component whose weight differs, is excluded from the solve
     * altogether.
     */
    [[nodiscard]] Real dot_product(const std::vector<Complex>& left,
                                   const std::vector<Complex>& right) const {
        Real result = 0.0;
        for (std::size_t g_index = first_index(); g_index < left.size(); ++g_index) {
            result += std::real(std::conj(left[g_index]) * right[g_index]);
        }
        return result;
    }

    /// x ← [G² + ⟨k_TF²⟩]⁻¹ v, the operator's diagonal part; exact when k_TF² is uniform.
    void apply_diagonal_inverse(const std::vector<Complex>& input,
                                std::vector<Complex>& output) const {
        output.assign(input.size(), Complex(0.0, 0.0));
        for (std::size_t g_index = first_index(); g_index < input.size(); ++g_index) {
            output[g_index] = input[g_index] / (g_squared_[g_index] + mean_screening_);
        }
    }

    /**
     * y ← [G² + k_TF²(r)] x, one FFT pair.
     *
     * The real-space multiply repopulates G = 0 even when x has none, so the
     * result is projected back onto G ≠ 0. That projection is what keeps the
     * operator symmetric positive definite on the subspace the solve runs in,
     * and it is why charge is conserved: the G = 0 component of the residual is
     * never touched.
     */
    void apply_operator(const std::vector<Complex>& input, std::vector<Complex>& output) const {
        std::vector<Real> input_real;
        to_real_space_(input, input_real);
        std::size_t points = std::min(input_real.size(), screening_real_.size());
        for (std::size_t index = 0; index < points; ++index) {
            input_real[index] *= screening_real_[index];
        }
        to_reciprocal_space_(input_real, output);

        output.resize(input.size());
        for (std::size_t g_index = 0; g_index < first_index(); ++g_index) {
            output[g_index] = Complex(0.0, 0.0);
        }
        for (std::size_t g_index = first_index(); g_index < input.size(); ++g_index) {
            output[g_index] += g_squared_[g_index] * input[g_index];
        }
    }

    std::vector<Real> g_squared_;
    Real q0_squared_{1.0};
    int first_nonzero_g_index_{0};
    std::function<void(const std::vector<Real>&, std::vector<Complex>&)> to_reciprocal_space_;
    std::function<void(const std::vector<Complex>&, std::vector<Real>&)> to_real_space_;
    std::function<void(std::vector<Real>&)> reduce_within_pool_;
    std::vector<Real> screening_real_;
    Real mean_screening_{1.0};
    bool has_screening_{false};

    /// Cap and exit tolerance for the conjugate-gradient solve.
    static constexpr int maximum_iterations_ = 50;
    static constexpr Real relative_tolerance_ = 1.0e-8;
};

} // namespace q3
