#pragma once

/**
 * Anderson (Pulay) density mixer in real space.
 *
 * Implements the residual-minimization variant of Pulay/Anderson mixing
 * with configurable history depth. Falls back to linear mixing when the
 * LAPACK least-squares solve fails or when only one history entry exists.
 */

#include <algorithm>
#include <cmath>

#include "io/logger.hpp"
#include "layer_4_computation/iterator/internal/mixing/real_space/init.hpp"

#include "layer_0_core/math/linear_algebra/lapack.hpp"

namespace q3 {
using namespace q3::lapack;

/**
 * Anderson (Pulay) mixing with configurable history depth (real-space).
 * Uses the residual minimization approach for faster SCF convergence.
 */
class AndersonMixer : public RealSpaceMixer {
  public:
    explicit AndersonMixer(Real mixing_parameter = 0.7, int history_depth = 8)
        : RealSpaceMixer(mixing_parameter, history_depth) {}

    std::vector<Real> mix_one_component(const std::vector<Real>& input_density,
                                        const std::vector<Real>& output_density) override {
        int number_of_points = static_cast<int>(input_density.size());

        // Compute residual
        std::vector<Real> residual(number_of_points);
        for (int point_index = 0; point_index < number_of_points; point_index++) {
            residual[point_index] = output_density[point_index] - input_density[point_index];
        }

        if (preconditioner_) {
            preconditioner_(residual);
        }

        // Store in history
        density_history_.push_back(input_density);
        residual_history_.push_back(residual);

        // Trim to max history depth
        while (static_cast<int>(density_history_.size()) > history_depth_) {
            density_history_.erase(density_history_.begin());
            residual_history_.erase(residual_history_.begin());
        }

        int number_of_history_entries = static_cast<int>(density_history_.size());

        // If only one entry, fall back to linear mixing
        if (number_of_history_entries <= 1) {
            std::vector<Real> mixed_density(number_of_points);
            for (int point_index = 0; point_index < number_of_points; point_index++) {
                mixed_density[point_index] =
                    input_density[point_index] + mixing_parameter_ * residual[point_index];
            }
            return mixed_density;
        }

        /**
         * Build overlap matrix A[i,j] = <R_i - R_last | R_j - R_last>
         * and RHS b[i] = <R_i - R_last | R_last>
         *
         * All dot products are accumulated locally first, then reduced together
         * in a single MPI collective call to avoid O(history^2) synchronizations.
         */
        int number_of_differences = number_of_history_entries - 1;
        std::vector<Real> overlap_matrix(number_of_differences * number_of_differences,
                                         0.0); // column-major
        std::vector<Real> right_hand_side(number_of_differences, 0.0);

        const auto& residual_last = residual_history_.back();

        // Pre-compute delta residuals for all history entries.
        std::vector<std::vector<Real>> delta_residuals(number_of_differences,
                                                       std::vector<Real>(number_of_points));
        for (int difference_index = 0; difference_index < number_of_differences;
             difference_index++) {
            for (int point_index = 0; point_index < number_of_points; point_index++) {
                delta_residuals[difference_index][point_index] =
                    residual_history_[difference_index][point_index] - residual_last[point_index];
            }
        }

        // Accumulate all dot products locally (no MPI communication yet).
        for (int difference_index = 0; difference_index < number_of_differences;
             difference_index++) {
            const auto& delta_residual_i = delta_residuals[difference_index];

            Real rhs_dot = 0.0;
            for (int point_index = 0; point_index < number_of_points; point_index++) {
                rhs_dot -= delta_residual_i[point_index] * residual_last[point_index];
            }
            right_hand_side[difference_index] = rhs_dot;

            for (int column_index = 0; column_index < number_of_differences; column_index++) {
                Real dot = 0.0;
                const auto& delta_residual_j = delta_residuals[column_index];
                for (int point_index = 0; point_index < number_of_points; point_index++) {
                    dot += delta_residual_i[point_index] * delta_residual_j[point_index];
                }
                overlap_matrix[difference_index + column_index * number_of_differences] =
                    dot; // column-major
            }
        }

        /**
         * Single batched MPI reduction: pack rhs and upper triangle of overlap into one vector,
         * reduce once, then unpack. This is exactly one MPI_Allreduce regardless of history depth.
         */
        if (global_reduce_) {
            // Pack: [rhs[0..n-1], overlap[0,0], overlap[1,0], overlap[0,1], ...]
            int overlap_size = number_of_differences * number_of_differences;
            std::vector<Real> reduce_buffer(number_of_differences + overlap_size);
            for (int i = 0; i < number_of_differences; i++) {
                reduce_buffer[i] = right_hand_side[i];
            }
            for (int i = 0; i < overlap_size; i++) {
                reduce_buffer[number_of_differences + i] = overlap_matrix[i];
            }
            global_reduce_(reduce_buffer);
            // Unpack
            for (int i = 0; i < number_of_differences; i++) {
                right_hand_side[i] = reduce_buffer[i];
            }
            for (int i = 0; i < overlap_size; i++) {
                overlap_matrix[i] = reduce_buffer[number_of_differences + i];
            }
        }

        // Solve A * gamma = b via LAPACK least-squares
        int number_of_right_hand_sides = 1;
        int lapack_workspace_size = 4 * number_of_differences;
        std::vector<Real> lapack_workspace(lapack_workspace_size);
        int info;
        dgels_("N",
               &number_of_differences,
               &number_of_differences,
               &number_of_right_hand_sides,
               overlap_matrix.data(),
               &number_of_differences,
               right_hand_side.data(),
               &number_of_differences,
               lapack_workspace.data(),
               &lapack_workspace_size,
               &info);

        if (info != 0) {
            log::warn("Anderson mixing: LAPACK dgels failed (info={}), falling back to linear",
                      info);
            std::vector<Real> mixed_density(number_of_points);
            for (int point_index = 0; point_index < number_of_points; point_index++) {
                mixed_density[point_index] =
                    input_density[point_index] + mixing_parameter_ * residual[point_index];
            }
            return mixed_density;
        }

        // right_hand_side now contains the solution gamma

        // Construct optimal input: rho_opt = rho_last + Σ gamma_i (rho_i - rho_last)

        // Construct optimal residual: res_opt = res_last + Σ gamma_i (res_i - res_last)
        std::vector<Real> density_optimal(number_of_points), residual_optimal(number_of_points);
        const auto& density_last = density_history_.back();

        for (int point_index = 0; point_index < number_of_points; point_index++) {
            density_optimal[point_index] = density_last[point_index];
            residual_optimal[point_index] = residual_last[point_index];
        }

        for (int difference_index = 0; difference_index < number_of_differences;
             difference_index++) {
            Real gamma_i = right_hand_side[difference_index];
            for (int point_index = 0; point_index < number_of_points; point_index++) {
                density_optimal[point_index] +=
                    gamma_i *
                    (density_history_[difference_index][point_index] - density_last[point_index]);
                residual_optimal[point_index] +=
                    gamma_i *
                    (residual_history_[difference_index][point_index] - residual_last[point_index]);
            }
        }

        // Final mixed density: rho_new = rho_opt + beta * res_opt
        std::vector<Real> mixed_density(number_of_points);
        for (int point_index = 0; point_index < number_of_points; point_index++) {
            mixed_density[point_index] =
                density_optimal[point_index] + mixing_parameter_ * residual_optimal[point_index];
        }

        return mixed_density;
    }

    void reset() override {
        density_history_.clear();
        residual_history_.clear();
    }

    [[nodiscard]] std::string name() const override {
        return "anderson";
    }

  private:
    // History buffers
    std::vector<std::vector<Real>> density_history_;
    std::vector<std::vector<Real>> residual_history_;
};

} // namespace q3
