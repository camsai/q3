#pragma once

/**
 * Local potential tabulation free functions.
 *
 * Pure functions that compute the local pseudopotential in G-space:
 * - Direct radial evaluation at a single q value
 * - Per-species tabulation at all G-vectors
 * - Total local potential with structure factors
 */

#include <cmath>
#include <complex>
#include <vector>

#include "layer_0_core/math/integration/simpson.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/reciprocal_projector_resampling.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

/**
 * Whether a dataset gives its local potential reciprocally, with the bare Coulomb tail
 * already subtracted, rather than radially.
 *
 * A dataset may tabulate V_loc reciprocally for the plane-wave grid as well as giving a
 * radial potential that describes something else (the potential inside the augmentation
 * sphere), so this is a branch on what the dataset carries, not a shortcut. Such a table
 * stores V(q) + 4πZe²/q², the divergence subtracted; at q = 0 its entry is the finite
 * α_Z constant.
 */
inline bool local_potential_is_tabulated_reciprocally(const Pseudopotential& pseudopotential) {
    const auto* paw = dynamic_cast<const PawPseudopotential*>(&pseudopotential);
    return paw != nullptr && paw->has_reciprocal_local_potential();
}

/**
 * The long-range part of V_loc(q): the transform of the Coulomb tail -Z e²/r, which
 * diverges as 1/q² at small q.
 *
 * A radially given potential has the Gaussian-screened tail -Z e² erf(r)/r subtracted in
 * real space (see calculate_local_potential_short_range_at_q), and this is its exact
 * transform, -4πZe² exp(-q²/4)/(Ωq²). A reciprocal table subtracted the bare tail, so for
 * it this is -4πZe²/(Ωq²). Zero at q = 0, where the finite α_Z constant stands for the
 * whole potential.
 *
 * It is kept out of every interpolation table. A 4-point Lagrange fit across a 1/q²
 * divergence is off by roughly (dq/q)⁴ of the term itself — a relative 1e-3 at the
 * smallest |G| of a 20 Å cell with dq = 0.01 — and the error lands in every G-vector
 * near the origin, which only cells with a long vacuum direction have.
 */
inline Real calculate_local_potential_long_range_at_q(Real valence_charge,
                                                      bool bare_coulomb_tail,
                                                      Real q_value,
                                                      Real cell_volume) {
    if (q_value < math::tolerances::zero_vector_magnitude_threshold)
        return 0.0;
    if (bare_coulomb_tail)
        return -valence_charge * constants::electron_charge_squared_rydberg * constants::four_pi /
               (cell_volume * q_value * q_value);
    return -valence_charge * constants::electron_charge_squared_rydberg * constants::four_pi *
           std::exp(-q_value * q_value * 0.25) / (cell_volume * q_value * q_value);
}

/**
 * d/dq of calculate_local_potential_long_range_at_q, which the local stress needs:
 * 2C/q³ for the bare tail -C/q², and C exp(-q²/4) (2/q³ + 1/(2q)) for the screened one,
 * with C = 4πZe²/Ω. Zero at q = 0.
 */
inline Real calculate_local_potential_long_range_derivative_at_q(Real valence_charge,
                                                                 bool bare_coulomb_tail,
                                                                 Real q_value,
                                                                 Real cell_volume) {
    if (q_value < math::tolerances::zero_vector_magnitude_threshold)
        return 0.0;
    Real coulomb_over_q_squared = valence_charge * constants::electron_charge_squared_rydberg *
                                  constants::four_pi / (cell_volume * q_value * q_value);
    if (bare_coulomb_tail)
        return 2.0 * coulomb_over_q_squared / q_value;
    return coulomb_over_q_squared * std::exp(-q_value * q_value * 0.25) *
           (2.0 / q_value + 0.5 * q_value);
}

/**
 * The short-range part of V_loc(q): everything but the long-range tail.
 *
 * It is smooth in q down to and including q = 0, which is what lets it be tabulated
 * on a uniform q-grid and interpolated. For a radially given potential it is the
 * transform of r V_loc(r) + Z e² erf(r), which vanishes beyond the core, integrated with
 * Simpson's rule over the truncated mesh; at q = 0 the kernel sin(qr)/q is its limit r.
 * For a reciprocal table it is the table itself, resampled at q.
 */
inline Real calculate_local_potential_short_range_at_q(const Pseudopotential& pseudopotential,
                                                       int mesh_size,
                                                       Real q_value,
                                                       Real cell_volume) {
    if (local_potential_is_tabulated_reciprocally(pseudopotential)) {
        const auto& paw = dynamic_cast<const PawPseudopotential&>(pseudopotential);
        std::vector<Real> single_q{q_value};
        std::vector<Real> tabulated;
        resample_uniform_reciprocal_table(paw.potential_local_reciprocal,
                                          paw.reciprocal_potential_grid_spacing,
                                          0,
                                          single_q,
                                          tabulated);
        return tabulated[0] / cell_volume;
    }

    bool at_zero_q = q_value < math::tolerances::zero_vector_magnitude_threshold;
    std::vector<Real> integrand(mesh_size);
    for (int radial_index = 0; radial_index < mesh_size; radial_index++) {
        Real radius = pseudopotential.radial_grid[radial_index];
        Real screened_potential =
            radius * pseudopotential.potential_local.radial_values[radial_index] +
            pseudopotential.valence_charge * constants::electron_charge_squared_rydberg *
                std::erf(radius);
        integrand[radial_index] = at_zero_q
                                      ? screened_potential * radius
                                      : screened_potential * std::sin(q_value * radius) / q_value;
    }
    Real integrated_sum =
        simpson_rule(integrand, pseudopotential.radial_grid_derivative, mesh_size);
    return constants::four_pi * integrated_sum / cell_volume;
}

/**
 * Compute local pseudopotential V_loc(q) via direct radial integration.
 *
 * This evaluates the Fourier-transformed local potential at a single |G| value: the
 * short-range part plus the long-range part for q > 0, and the finite α_Z constant at
 * q = 0, the G = 0 term of a neutral cell.
 */
inline Real calculate_local_potential_at_q(const Pseudopotential& pseudopotential,
                                           int mesh_size,
                                           Real q_value,
                                           Real cell_volume) {
    bool bare_coulomb_tail = local_potential_is_tabulated_reciprocally(pseudopotential);
    if (q_value >= math::tolerances::zero_vector_magnitude_threshold) {
        return calculate_local_potential_short_range_at_q(
                   pseudopotential, mesh_size, q_value, cell_volume) +
               calculate_local_potential_long_range_at_q(
                   pseudopotential.valence_charge, bare_coulomb_tail, q_value, cell_volume);
    }

    // A reciprocal table's own q = 0 entry is already α_Z.
    if (bare_coulomb_tail)
        return calculate_local_potential_short_range_at_q(
            pseudopotential, mesh_size, q_value, cell_volume);

    std::vector<Real> integrand(mesh_size);
    for (int radial_index = 0; radial_index < mesh_size; radial_index++) {
        integrand[radial_index] =
            pseudopotential.radial_grid[radial_index] *
            (pseudopotential.radial_grid[radial_index] *
                 pseudopotential.potential_local.radial_values[radial_index] +
             pseudopotential.valence_charge * constants::electron_charge_squared_rydberg);
    }
    Real integrated_sum =
        simpson_rule(integrand, pseudopotential.radial_grid_derivative, mesh_size);
    return constants::four_pi * integrated_sum / cell_volume;
}

/// Result of local potential tabulation: per-species V_loc(G) and total with structure factors.
struct LocalPotentialTabulationResult {
    std::vector<std::vector<Real>> per_species;
    std::vector<Complex> total;
};

/**
 * Tabulate local potential in G-space for all species and compute the total.
 *
 * Evaluates V_loc(|G|) directly at each G-vector (no intermediate q-grid),
 * then combines with structure factors to produce the total local potential.
 */
inline LocalPotentialTabulationResult
tabulate_local_potential_in_g_space(const Crystal& crystal_structure,
                                    const PlanewaveBasis& planewave_basis,
                                    const PseudopotentialSet& pseudopotential_set,
                                    const std::vector<int>& mesh_sizes,
                                    const std::vector<Real>& g_vector_magnitudes,
                                    Real cell_volume) {

    int number_of_species_types = static_cast<int>(pseudopotential_set.size());
    int number_of_g_vectors = planewave_basis.number_of_g_vectors();

    LocalPotentialTabulationResult result;

    // Compute Vloc(G) directly at actual G-vectors
    result.per_species.resize(number_of_species_types);
    for (int type_index = 0; type_index < number_of_species_types; type_index++) {
        result.per_species[type_index].resize(number_of_g_vectors);
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
            Real q_value = g_vector_magnitudes[g_vector_index];
            result.per_species[type_index][g_vector_index] = calculate_local_potential_at_q(
                pseudopotential_set[type_index], mesh_sizes[type_index], q_value, cell_volume);
        }
    }

    // Compute structure factor and total Vloc(G)
    result.total.resize(number_of_g_vectors, Complex(0.0, 0.0));
    for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
        for (int type_index = 0; type_index < number_of_species_types; type_index++) {
            Complex structure_factor(0.0, 0.0);
            for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms();
                 atom_index++) {
                if (crystal_structure.species_type_index(atom_index) == type_index) {
                    Vector3D atomic_position =
                        crystal_structure.convert_from_crystal_to_cartesian(atom_index);
                    const auto& g_vector = planewave_basis.g_vectors()[g_vector_index];
                    Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                    structure_factor += compute_phase_factor(phase);
                }
            }
            result.total[g_vector_index] +=
                result.per_species[type_index][g_vector_index] * structure_factor;
        }
    }

    return result;
}

} // namespace q3
