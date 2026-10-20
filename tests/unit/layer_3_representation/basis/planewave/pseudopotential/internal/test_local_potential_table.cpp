#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "layer_0_core/math/interpolation/uniform_q_grid.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/local_potential_table.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"

using namespace q3;

/**
 * Default-constructed LocalPotentialTable should have a default q_grid
 * with zero points and zero spacing.
 */
TEST(LocalPotentialTable, DefaultConstruction) {
    LocalPotentialTable table;
    const UniformQGrid& grid = table.q_grid();
    EXPECT_EQ(grid.number_of_q_points, 0);
    EXPECT_DOUBLE_EQ(grid.q_spacing, 0.0);
}

/**
 * Verify that the q_grid accessor returns consistent values
 * from a default-constructed table.
 */
TEST(LocalPotentialTable, DefaultQGridPointValuesEmpty) {
    LocalPotentialTable table;
    std::vector<Real> q_point_values = table.q_grid().q_point_values();
    EXPECT_TRUE(q_point_values.empty());
}

// ── Small wavevectors ───────────────────────────────────────────────

/**
 * The whole of V_loc(q) used to be tabulated, 1/q² divergence included, and a 4-point
 * fit across that divergence is wrong near the origin: a relative 1e-3 at the smallest
 * |G| of a 20 Å cell. Only cells with a long vacuum direction reach those G-vectors, and
 * there it added an energy growing roughly as c⁵ with the cell length. The fixture is a
 * potential whose transform is known exactly: V(r) = -Z e² erf(r/r_c)/r has
 *     V(q) = -4πZe² exp(-q² r_c²/4) / (Ω q²),   and   α_Z = 4πZe² r_c² / (4 Ω).
 * The cell volume is the 3.5 x 3.5 x 20 Å box the defect was found in, and the
 * wavevectors include its smallest |G| (2π/c) and points inside the first grid interval.
 */
namespace {

constexpr Real valence_charge = 4.0;
constexpr Real screening_radius = 0.6;
constexpr Real cell_volume = 6.614041435977716 * 6.614041435977716 * 37.79452249130124;
constexpr Real q_spacing = 0.01;

/// 4πZe²/Ω, the scale every value here carries.
Real coulomb_scale() {
    return constants::four_pi * valence_charge * constants::electron_charge_squared_rydberg /
           cell_volume;
}

Real exact_local_potential(Real q) {
    return -coulomb_scale() * std::exp(-q * q * screening_radius * screening_radius / 4.0) /
           (q * q);
}

Real exact_local_potential_derivative(Real q) {
    Real a = screening_radius * screening_radius / 4.0;
    return coulomb_scale() * std::exp(-a * q * q) * (2.0 * a / q + 2.0 / (q * q * q));
}

/// A logarithmic radial mesh, as UPF files carry, with the potential above on it.
std::unique_ptr<Pseudopotential> screened_coulomb_pseudopotential() {
    auto pseudopotential = std::make_unique<Pseudopotential>();
    pseudopotential->element = "C";
    pseudopotential->valence_charge = valence_charge;
    const int points = 1100;
    const Real first_radius = 1.0e-4;
    const Real logarithmic_step = 0.0125;
    pseudopotential->mesh_size = points;
    for (int index = 0; index < points; index++) {
        Real radius = first_radius * std::exp(logarithmic_step * index);
        pseudopotential->radial_grid.push_back(radius);
        pseudopotential->radial_grid_derivative.push_back(logarithmic_step * radius);
        pseudopotential->potential_local.radial_values.push_back(
            -valence_charge * constants::electron_charge_squared_rydberg *
            std::erf(radius / screening_radius) / radius);
    }
    return pseudopotential;
}

/// The mesh the basis integrates over: the first point beyond 10 bohr, made odd.
int truncated_mesh_size(const Pseudopotential& pseudopotential) {
    int size = pseudopotential.mesh_size;
    for (int index = 0; index < pseudopotential.mesh_size; index++) {
        if (pseudopotential.radial_grid[index] > 10.0) {
            size = index + 1;
            break;
        }
    }
    return 2 * ((size + 1) / 2) - 1;
}

const std::vector<Real> small_and_ordinary_wavevectors = {
    0.004, 0.0123, 0.0718, 0.1333, 0.16625, 0.19, 0.2471, 0.33249, 1.2345, 5.4321};

} // namespace

TEST(LocalPotentialTable, SmallWavevectorsMatchTheExactTransform) {
    PseudopotentialSet set;
    set.push_back(screened_coulomb_pseudopotential());
    int mesh = truncated_mesh_size(set[0]);
    LocalPotentialTable table;
    table.initialize(UniformQGrid::from_maximum_q(8.0, q_spacing), set, {mesh}, cell_volume, 1);

    for (Real q : small_and_ordinary_wavevectors) {
        EXPECT_NEAR(table.interpolate(0, q), exact_local_potential(q), 1e-9 * coulomb_scale())
            << "q = " << q;
        EXPECT_NEAR(calculate_local_potential_at_q(set[0], mesh, q, cell_volume),
                    exact_local_potential(q),
                    1e-9 * coulomb_scale())
            << "q = " << q;
    }
}

TEST(LocalPotentialTable, ZeroWavevectorKeepsTheAlphaZConstant) {
    PseudopotentialSet set;
    set.push_back(screened_coulomb_pseudopotential());
    int mesh = truncated_mesh_size(set[0]);
    LocalPotentialTable table;
    table.initialize(UniformQGrid::from_maximum_q(8.0, q_spacing), set, {mesh}, cell_volume, 1);

    EXPECT_DOUBLE_EQ(table.interpolate(0, 0.0),
                     calculate_local_potential_at_q(set[0], mesh, 0.0, cell_volume));
    // The mesh starts at 1e-4 bohr, so the integral misses 5e-9 of the scale below it.
    EXPECT_NEAR(table.interpolate(0, 0.0),
                coulomb_scale() * screening_radius * screening_radius / 4.0,
                1e-8 * coulomb_scale());
}

TEST(LocalPotentialTable, DerivativeAtSmallWavevectorsMatchesTheExactTransform) {
    /**
     * The local stress takes dV_loc/dq from the table. Differentiating an interpolant of
     * the divergence was off by more than the value itself at these wavevectors.
     */
    PseudopotentialSet set;
    set.push_back(screened_coulomb_pseudopotential());
    int mesh = truncated_mesh_size(set[0]);
    LocalPotentialTable table;
    table.initialize(UniformQGrid::from_maximum_q(8.0, q_spacing), set, {mesh}, cell_volume, 1);

    for (Real q : small_and_ordinary_wavevectors) {
        if (q < q_spacing)
            continue;
        EXPECT_NEAR(table.interpolate_derivative(0, q),
                    exact_local_potential_derivative(q),
                    1e-6 * coulomb_scale())
            << "q = " << q;
    }
}

TEST(LocalPotentialTable, ReciprocallyTabulatedDatasetsGetTheBareTailBack) {
    /**
     * A reciprocal table gives V(q) + 4πZe²/q² on its own grid, finite at q = 0 where it is α_Z.
     * The table must add -4πZe²/(Ωq²) back exactly, not interpolate it.
     */
    auto paw = std::make_unique<PawPseudopotential>();
    paw->element = "C";
    paw->valence_charge = valence_charge;
    paw->reciprocal_potential_grid_spacing = 0.05;
    for (int index = 0; index < 400; index++) {
        Real q = index * paw->reciprocal_potential_grid_spacing;
        paw->potential_local_reciprocal.push_back(25.0 * std::exp(-0.3 * q * q) - 4.0);
    }
    PseudopotentialSet set;
    set.push_back(std::move(paw));
    LocalPotentialTable table;
    table.initialize(UniformQGrid::from_maximum_q(8.0, q_spacing), set, {0}, cell_volume, 1);

    EXPECT_DOUBLE_EQ(table.interpolate(0, 0.0), 21.0 / cell_volume);
    /**
     * What is left after the tail is the table's own fit of the dataset's smooth part,
     * resampled from a coarser grid: a few parts in 1e7 of its size, 25/Ω.
     */
    const Real smooth_part_scale = 25.0 / cell_volume;
    for (Real q : small_and_ordinary_wavevectors) {
        Real direct = calculate_local_potential_at_q(set[0], 0, q, cell_volume);
        EXPECT_NEAR(direct + coulomb_scale() / (q * q),
                    calculate_local_potential_short_range_at_q(set[0], 0, q, cell_volume),
                    1e-12 * coulomb_scale() / (q * q))
            << "q = " << q;
        EXPECT_NEAR(table.interpolate(0, q), direct, 1e-6 * smooth_part_scale) << "q = " << q;
    }
}
