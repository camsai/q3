#pragma once

/**
 * InterpolatedDensityOfStatesCalculator — a Gaussian-broadened density of
 * states from a QuantumStateInterpolator evaluated on a dense uniform k-mesh.
 *
 * Bands and DOS are the same operation asked for different k-points:
 * evaluate the model, then either keep the eigenvalues ordered along a path
 * or histogram them over a mesh. That is why an interpolated DOS costs
 * almost nothing once the model exists -- evaluating it at 10^5 k-points is
 * cheap next to the calculation that produced it -- and why this calculator
 * is written against the interpolator interface rather than any one
 * method's model.
 *
 * Gaussian broadening matches what q3.workflows.dos does for directly
 * computed eigenvalues, so a DOS from this class and a direct one can be
 * compared at identical broadening and the difference attributed to the
 * model rather than to the smearing. Q3 has no tetrahedron integrator; when
 * one exists it belongs beside this as an alternative strategy.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/init.hpp"

namespace q3 {

/// A density of states on a uniform energy grid.
struct [[nodiscard]] InterpolatedDensityOfStates {
    std::vector<Real> energies_rydberg;
    /// States per Rydberg per unit cell, spin degeneracy NOT included.
    std::vector<Real> density_of_states;
};

class InterpolatedDensityOfStatesCalculator {
  public:
    explicit InterpolatedDensityOfStatesCalculator(const QuantumStateInterpolator& interpolator)
        : interpolator_(interpolator) {}

    /**
     * DOS on a uniform Gamma-centred `mesh_x` x `mesh_y` x `mesh_z` mesh,
     * broadened by `broadening_rydberg` onto `energy_step_count` points.
     *
     * The energy window spans the eigenvalue range plus a margin on each
     * side. The margin is at least `energy_margin_rydberg` and at least the
     * Gaussian's own cutoff of 8 sigma: a fixed margin smaller than the
     * broadening clips the tails of the band-edge states, and the integral
     * below stops returning the band count -- which it must, since the
     * normalisation is one state per band per k-point divided by the
     * k-point count. Nothing here doubles for spin; a caller wanting states
     * per cell for an unpolarised system multiplies by two, the convention
     * q3.workflows.dos follows.
     */
    [[nodiscard]] InterpolatedDensityOfStates compute(int mesh_x,
                                                      int mesh_y,
                                                      int mesh_z,
                                                      Real broadening_rydberg,
                                                      int energy_step_count = 1000,
                                                      Real energy_margin_rydberg = 0.05) const {
        if (mesh_x <= 0 || mesh_y <= 0 || mesh_z <= 0) {
            throw std::invalid_argument(
                "InterpolatedDensityOfStatesCalculator::compute: every mesh dimension must be "
                "positive");
        }
        if (broadening_rydberg <= 0.0) {
            throw std::invalid_argument(
                "InterpolatedDensityOfStatesCalculator::compute: broadening must be positive");
        }
        if (energy_step_count < 2) {
            throw std::invalid_argument(
                "InterpolatedDensityOfStatesCalculator::compute: need at least 2 energy points");
        }

        std::vector<std::vector<Real>> sampled_eigenvalues;
        sampled_eigenvalues.reserve(static_cast<std::size_t>(mesh_x) * mesh_y * mesh_z);
        for (int index_x = 0; index_x < mesh_x; index_x++) {
            for (int index_y = 0; index_y < mesh_y; index_y++) {
                for (int index_z = 0; index_z < mesh_z; index_z++) {
                    Vector3D kpoint(static_cast<Real>(index_x) / mesh_x,
                                    static_cast<Real>(index_y) / mesh_y,
                                    static_cast<Real>(index_z) / mesh_z);
                    sampled_eigenvalues.push_back(interpolator_.eigenvalues_at(kpoint));
                }
            }
        }

        Real minimum_energy = sampled_eigenvalues.front().front();
        Real maximum_energy = sampled_eigenvalues.front().front();
        for (const std::vector<Real>& row : sampled_eigenvalues) {
            for (Real value : row) {
                minimum_energy = std::min(minimum_energy, value);
                maximum_energy = std::max(maximum_energy, value);
            }
        }
        const Real cutoff = 8.0 * broadening_rydberg;
        const Real margin = std::max(energy_margin_rydberg, cutoff);
        minimum_energy -= margin;
        maximum_energy += margin;

        InterpolatedDensityOfStates result;
        result.energies_rydberg.resize(energy_step_count);
        result.density_of_states.assign(energy_step_count, 0.0);
        const Real energy_step =
            (maximum_energy - minimum_energy) / static_cast<Real>(energy_step_count - 1);
        for (int step = 0; step < energy_step_count; step++) {
            result.energies_rydberg[step] = minimum_energy + step * energy_step;
        }

        /**
         * Unit-area Gaussian per state. Contributions beyond the 8 sigma
         * cutoff are below 1e-14 of the peak and skipped: with 10^5 k-points
         * times a few bands times 10^3 energy points, evaluating every pair
         * would dominate the runtime for no numerical gain.
         */
        const Real normalisation = 1.0 / (broadening_rydberg * std::sqrt(constants::two_pi) *
                                          static_cast<Real>(sampled_eigenvalues.size()));
        for (const std::vector<Real>& row : sampled_eigenvalues) {
            for (Real eigenvalue : row) {
                int first = static_cast<int>(
                    std::ceil((eigenvalue - cutoff - minimum_energy) / energy_step));
                int last = static_cast<int>(
                    std::floor((eigenvalue + cutoff - minimum_energy) / energy_step));
                first = std::max(first, 0);
                last = std::min(last, energy_step_count - 1);
                for (int step = first; step <= last; step++) {
                    const Real offset =
                        (result.energies_rydberg[step] - eigenvalue) / broadening_rydberg;
                    result.density_of_states[step] +=
                        normalisation * std::exp(-0.5 * offset * offset);
                }
            }
        }
        return result;
    }

  private:
    const QuantumStateInterpolator& interpolator_;
};

} // namespace q3
