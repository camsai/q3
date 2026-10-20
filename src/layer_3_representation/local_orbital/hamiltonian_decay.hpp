#pragma once

/**
 * HamiltonianDecay — how far a real-space Hamiltonian H(R) has decayed by the
 * outermost shell of its lattice vectors.
 *
 * A local-orbital model is a Fourier sum over the lattice vectors of its
 * domain, exact at the k-points it was built from whatever the sum's last
 * terms are. Between those points the error IS those terms. So the largest
 * term on the outermost shell, against the largest hopping anywhere, is the
 * one number that predicts off-mesh error without a reference calculation --
 * and unlike a spread it is in energy units and can be read as one.
 *
 * The denominator is the largest term that varies with k (any R other than
 * zero, or an off-diagonal on-site element) rather than the largest on-site
 * element, because the on-site diagonal carries the absolute zero of energy,
 * which a pseudopotential sets arbitrarily; a ratio built on it would change
 * under a rigid shift of every eigenvalue.
 *
 * The lengths are in whatever unit the domain's Cartesian vectors are in and
 * the magnitudes in the Hamiltonian's; nothing here converts. It holds for a
 * model built by any method, which is why it sits beside the Hamiltonian it
 * measures rather than with one of them.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/local_orbital/init.hpp"

namespace q3 {

struct [[nodiscard]] HamiltonianDecay {
    /// |R| of the outermost shell, in the lattice's length unit.
    Real outermost_shell_distance = 0.0;
    /// How many lattice vectors share that distance (to the registry's shell width).
    int outermost_shell_size = 0;
    /// The largest Bloch-sum term w_R |H_mn(R)| on that shell, in the Hamiltonian's unit.
    Real outermost_shell_maximum = 0.0;
    /// The largest term that varies with k: R away from the origin, or m != n on site.
    Real largest_hopping = 0.0;
    /// outermost_shell_maximum / largest_hopping; zero when there is nothing to divide by.
    Real ratio = 0.0;

    [[nodiscard]] static HamiltonianDecay measured_on(const LocalOrbitalHamiltonian& hamiltonian) {
        const std::vector<LocalOrbitalLatticeVector>& vectors = hamiltonian.lattice_vectors();
        const std::vector<std::vector<Complex>>& blocks = hamiltonian.hamiltonian_blocks();
        if (vectors.empty() || blocks.size() != vectors.size()) {
            throw std::runtime_error("HamiltonianDecay::measured_on: the Hamiltonian has no "
                                     "blocks on its lattice vectors");
        }
        const std::size_t dimension = static_cast<std::size_t>(hamiltonian.number_of_orbitals());

        HamiltonianDecay decay;
        for (const LocalOrbitalLatticeVector& vector : vectors) {
            decay.outermost_shell_distance =
                std::max(decay.outermost_shell_distance, vector.cartesian.norm());
        }
        const Real shell_floor =
            decay.outermost_shell_distance * (1.0 - math::tolerances::lattice_shell_relative_width);

        for (std::size_t vector_index = 0; vector_index < vectors.size(); vector_index++) {
            const LocalOrbitalLatticeVector& vector = vectors[vector_index];
            const bool on_site = vector.translation == IntegerVector3D{0, 0, 0};
            const bool on_outermost_shell = vector.cartesian.norm() >= shell_floor;
            if (on_outermost_shell) {
                decay.outermost_shell_size++;
            }
            const std::vector<Complex>& block = blocks[vector_index];
            for (std::size_t element = 0; element < block.size(); element++) {
                /// Column-major, so an element is on the diagonal when its row is its column.
                const bool diagonal = element % dimension == element / dimension;
                const Real magnitude = vector.bloch_sum_weight * std::abs(block[element]);
                if (!on_site || !diagonal) {
                    decay.largest_hopping = std::max(decay.largest_hopping, magnitude);
                }
                if (on_outermost_shell) {
                    decay.outermost_shell_maximum =
                        std::max(decay.outermost_shell_maximum, magnitude);
                }
            }
        }
        decay.ratio = decay.largest_hopping > 0.0
                          ? decay.outermost_shell_maximum / decay.largest_hopping
                          : 0.0;
        return decay;
    }
};

} // namespace q3
