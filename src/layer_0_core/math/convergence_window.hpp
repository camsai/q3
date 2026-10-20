#pragma once

/**
 * ConvergenceWindow — an iteration has converged when its objective has
 * stopped moving, relatively, for a window of consecutive steps.
 *
 * One step within tolerance proves nothing: a damped iteration passes
 * through near-stationary points on its way, and a line search that has
 * shrunk its step produces a value that moves by nothing while the gradient
 * is still large. Requiring the last `window` differences all to be inside
 * the tolerance is the least that distinguishes settled from paused. The
 * test is relative to the current value, so the same tolerance serves a
 * spread of ten and a spread of ten thousand.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

class ConvergenceWindow {
  public:
    ConvergenceWindow(Real relative_tolerance, int window)
        : relative_tolerance_(relative_tolerance), window_(window) {
        if (!(relative_tolerance > 0.0)) {
            throw std::invalid_argument(
                "ConvergenceWindow: the relative tolerance must be positive, got " +
                std::to_string(relative_tolerance));
        }
        if (window < 1) {
            throw std::invalid_argument("ConvergenceWindow: the window must be at least one "
                                        "step, got " +
                                        std::to_string(window));
        }
    }

    /// The objective at the end of one more step.
    void record(Real value) {
        history_.push_back(value);
    }

    /**
     * Whether the last `window` differences are all within the tolerance of
     * the current value. Needs `window + 1` values, so the first steps never
     * count as converged.
     */
    [[nodiscard]] bool has_converged() const {
        if (static_cast<int>(history_.size()) <= window_) {
            return false;
        }
        const Real current = history_.back();
        const Real scale =
            std::max(std::abs(current), math::tolerances::relative_change_scale_floor);
        for (int step = 1; step <= window_; step++) {
            const Real earlier = history_[history_.size() - 1 - static_cast<std::size_t>(step)];
            if (std::abs(current - earlier) / scale >= relative_tolerance_) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] const std::vector<Real>& history() const {
        return history_;
    }

    [[nodiscard]] int number_of_steps() const {
        return static_cast<int>(history_.size());
    }

    /// The last `count` values, oldest first.
    [[nodiscard]] std::vector<Real> recent_history(int count = 5) const {
        const std::size_t start = history_.size() > static_cast<std::size_t>(count)
                                      ? history_.size() - static_cast<std::size_t>(count)
                                      : 0;
        return std::vector<Real>(history_.begin() + static_cast<std::ptrdiff_t>(start),
                                 history_.end());
    }

    /// The last `count` values, oldest first, as "a, b, c" with six decimals each.
    [[nodiscard]] std::string recent_history_text(int count = 5) const {
        return history_text(recent_history(count));
    }

    /**
     * `values` as "a, b, c" with six decimals each: the one form in which an
     * error names the spreads it reached, in its prose and on its own line.
     */
    [[nodiscard]] static std::string history_text(const std::vector<Real>& values) {
        std::string text;
        for (std::size_t index = 0; index < values.size(); index++) {
            text += (index == 0 ? "" : ", ") + std::to_string(values[index]);
        }
        return text;
    }

  private:
    Real relative_tolerance_;
    int window_;
    std::vector<Real> history_;
};

} // namespace q3
