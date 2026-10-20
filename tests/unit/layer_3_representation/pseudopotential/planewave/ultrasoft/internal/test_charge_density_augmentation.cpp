#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/math/spherical/bessel.hpp"
#include "layer_3_representation/pseudopotential/planewave/ultrasoft/internal/charge_density_augmentation.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"

using namespace q3;

// ── Default construction ────────────────────────────────────────────

TEST(AugmentationQTable, DefaultConstructionYieldsZeroMaxAngularMomentum) {
    AugmentationQTable table;
    EXPECT_EQ(table.maximum_angular_momentum_q(), 0);
}

TEST(AugmentationQTable, DefaultConstructedInterpolationReturnsZero) {
    AugmentationQTable table;

    // Interpolation on an uninitialized table should return 0.0
    EXPECT_DOUBLE_EQ(table.interpolate_qrad(0, 0, 0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(table.interpolate_qrad_derivative(0, 0, 0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(table.interpolate_qfunc(0, 0, 1.0), 0.0);
}

// ── Helper: load Si USPP and build a PseudopotentialSet ─────────────

static PseudopotentialSet load_si_uspp_set() {
    auto pseudopotential =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");
    PseudopotentialSet set;
    set.push_back(std::move(pseudopotential));
    return set;
}

// ── Augmentation charge symmetry: Q_ij(q,L) == Q_ji(q,L) ───────────

TEST(AugmentationQTable, AugmentationChargeSymmetryInProjectorIndices) {
    auto pseudopotential_set = load_si_uspp_set();
    const auto& ultrasoft_pseudopotential = pseudopotential_set.as_ultrasoft(0);

    int number_of_projectors = ultrasoft_pseudopotential.number_of_projectors;
    ASSERT_GT(number_of_projectors, 1) << "Need at least 2 projectors to test symmetry";

    // Use the mesh size for the radial grid
    int mesh_size = pseudopotential_set[0].mesh_size;
    std::vector<int> mesh_sizes = {mesh_size};

    // Parameters matching typical DFT settings
    int number_of_q_points = 201;
    Real q_spacing = 0.01;
    Real cell_volume = 270.0; // approximate Si FCC cell volume in Bohr^3

    AugmentationQTable table;
    table.init(pseudopotential_set, mesh_sizes, number_of_q_points, q_spacing, cell_volume);

    // Verify maximum angular momentum is positive for US-PP
    EXPECT_GT(table.maximum_angular_momentum_q(), 0)
        << "Ultrasoft PP should have non-zero maximum angular momentum for Q";

    int maximum_angular_momentum = table.maximum_angular_momentum_q();

    /**
     * Test symmetry: Q(species=0, ij, L, q) == Q(species=0, ji, L, q)
     * for all projector pairs and angular momentum channels
     */
    for (int angular_momentum = 0; angular_momentum <= maximum_angular_momentum;
         angular_momentum++) {
        for (int beta_index_i = 0; beta_index_i < number_of_projectors; beta_index_i++) {
            for (int beta_index_j = beta_index_i + 1; beta_index_j < number_of_projectors;
                 beta_index_j++) {
                int projector_pair_index_ij = beta_index_i * number_of_projectors + beta_index_j;
                int projector_pair_index_ji = beta_index_j * number_of_projectors + beta_index_i;

                // Sample at multiple q values
                for (int q_index = 1; q_index < number_of_q_points - 2; q_index += 20) {
                    Real q_value = q_index * q_spacing;

                    Real value_ij = table.interpolate_qrad(
                        0, projector_pair_index_ij, angular_momentum, q_value);
                    Real value_ji = table.interpolate_qrad(
                        0, projector_pair_index_ji, angular_momentum, q_value);

                    EXPECT_DOUBLE_EQ(value_ij, value_ji)
                        << "Q_ij != Q_ji for projectors (" << beta_index_i << ", " << beta_index_j
                        << "), L=" << angular_momentum << ", q=" << q_value;
                }
            }
        }
    }
}

TEST(AugmentationQTable, QFunctionSymmetryInProjectorIndices) {
    auto pseudopotential_set = load_si_uspp_set();
    const auto& ultrasoft_pseudopotential = pseudopotential_set.as_ultrasoft(0);

    int number_of_projectors = ultrasoft_pseudopotential.number_of_projectors;
    int mesh_size = pseudopotential_set[0].mesh_size;
    std::vector<int> mesh_sizes = {mesh_size};

    int number_of_q_points = 201;
    Real q_spacing = 0.01;
    Real cell_volume = 270.0;

    AugmentationQTable table;
    table.init(pseudopotential_set, mesh_sizes, number_of_q_points, q_spacing, cell_volume);

    // Test symmetry of the total Q function (L=0)
    for (int beta_index_i = 0; beta_index_i < number_of_projectors; beta_index_i++) {
        for (int beta_index_j = beta_index_i + 1; beta_index_j < number_of_projectors;
             beta_index_j++) {
            int projector_pair_index_ij = beta_index_i * number_of_projectors + beta_index_j;
            int projector_pair_index_ji = beta_index_j * number_of_projectors + beta_index_i;

            for (int q_index = 1; q_index < number_of_q_points - 2; q_index += 25) {
                Real q_value = q_index * q_spacing;

                Real value_ij = table.interpolate_qfunc(0, projector_pair_index_ij, q_value);
                Real value_ji = table.interpolate_qfunc(0, projector_pair_index_ji, q_value);

                EXPECT_DOUBLE_EQ(value_ij, value_ji)
                    << "Qfunc_ij != Qfunc_ji for projectors (" << beta_index_i << ", "
                    << beta_index_j << "), q=" << q_value;
            }
        }
    }
}

TEST(AugmentationQTable, DerivativeSymmetryInProjectorIndices) {
    auto pseudopotential_set = load_si_uspp_set();
    const auto& ultrasoft_pseudopotential = pseudopotential_set.as_ultrasoft(0);

    int number_of_projectors = ultrasoft_pseudopotential.number_of_projectors;
    int mesh_size = pseudopotential_set[0].mesh_size;
    std::vector<int> mesh_sizes = {mesh_size};

    int number_of_q_points = 201;
    Real q_spacing = 0.01;
    Real cell_volume = 270.0;

    AugmentationQTable table;
    table.init(pseudopotential_set, mesh_sizes, number_of_q_points, q_spacing, cell_volume);

    int maximum_angular_momentum = table.maximum_angular_momentum_q();

    // Derivative symmetry: dQ_ij/dq == dQ_ji/dq
    for (int angular_momentum = 0; angular_momentum <= maximum_angular_momentum;
         angular_momentum++) {
        for (int beta_index_i = 0; beta_index_i < number_of_projectors; beta_index_i++) {
            for (int beta_index_j = beta_index_i + 1; beta_index_j < number_of_projectors;
                 beta_index_j++) {
                int projector_pair_index_ij = beta_index_i * number_of_projectors + beta_index_j;
                int projector_pair_index_ji = beta_index_j * number_of_projectors + beta_index_i;

                for (int q_index = 1; q_index < number_of_q_points - 2; q_index += 30) {
                    Real q_value = q_index * q_spacing;

                    Real derivative_ij = table.interpolate_qrad_derivative(
                        0, projector_pair_index_ij, angular_momentum, q_value);
                    Real derivative_ji = table.interpolate_qrad_derivative(
                        0, projector_pair_index_ji, angular_momentum, q_value);

                    EXPECT_DOUBLE_EQ(derivative_ij, derivative_ji)
                        << "dQ_ij/dq != dQ_ji/dq for projectors (" << beta_index_i << ", "
                        << beta_index_j << "), L=" << angular_momentum << ", q=" << q_value;
                }
            }
        }
    }
}

// ── NC pseudopotentials have no augmentation ────────────────────────

TEST(AugmentationQTable, NormConservingPseudopotentialHasNoAugmentation) {
    auto pseudopotential =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF");

    EXPECT_EQ(pseudopotential->pseudopotential_type_enum(), PseudopotentialType::norm_conserving);
    EXPECT_FALSE(pseudopotential->has_augmentation_charges())
        << "Norm-conserving PP should not have augmentation charges";
}

// ── Non-zero values at finite q ─────────────────────────────────────

TEST(AugmentationQTable, AugmentationChargesAreNonZeroAtFiniteQ) {
    auto pseudopotential_set = load_si_uspp_set();

    int mesh_size = pseudopotential_set[0].mesh_size;
    std::vector<int> mesh_sizes = {mesh_size};

    int number_of_q_points = 201;
    Real q_spacing = 0.01;
    Real cell_volume = 270.0;

    AugmentationQTable table;
    table.init(pseudopotential_set, mesh_sizes, number_of_q_points, q_spacing, cell_volume);

    // At least some Q values should be non-zero for a USPP
    bool found_nonzero = false;
    int number_of_projectors = pseudopotential_set.as_ultrasoft(0).number_of_projectors;

    for (int beta_index_i = 0; beta_index_i < number_of_projectors && !found_nonzero;
         beta_index_i++) {
        int projector_pair_index = beta_index_i * number_of_projectors + beta_index_i;
        Real value = table.interpolate_qrad(0, projector_pair_index, 0, 0.5);
        if (std::abs(value) > 1e-15) {
            found_nonzero = true;
        }
    }

    EXPECT_TRUE(found_nonzero) << "At least some diagonal Q values should be non-zero for a USPP";
}

// ── q_with_l="false": the flat function must serve every allowed L ──

TEST(AugmentationQTable, FlatFunctionFallbackAppliesAtEveryAllowedAngularMomentumNotJustZero) {
    /**
     * `q_with_l="false"` pseudopotentials (the older un-decomposed
     * augmentation format) carry a single un-decomposed Q_ij(r) per
     * projector pair, with no per-L breakdown at all. The established
     * convention for these is to reuse that one function at *every* L the
     * triangle rule admits for the pair: the flat function is the pair's
     * whole augmentation content, so each allowed multipole draws on it.
     * This constructs a 2-projector case (s
     * and d, angular momenta 0 and 2) where the off-diagonal pair's *only*
     * allowed L is 2 -- there is no L=0 term for an s-d pair at all -- so a
     * reader that only ever falls back to the flat function at L==0 gives
     * an off-diagonal augmentation charge that is identically zero for
     * every q, silently dropping the pair's entire contribution.
     */
    auto pseudopotential = std::make_unique<UltrasoftPseudopotential>();
    pseudopotential->number_of_projectors = 2;
    pseudopotential->projectors = {
        Projector{/*angular_momentum=*/0, 0, {}},
        Projector{/*angular_momentum=*/2, 0, {}},
    };
    /**
     * `determine_maximum_angular_momentum_q` (charge_density_augmentation.hpp)
     * falls back to `2 * maximum_angular_momentum` only when
     * `number_of_q_angular_momenta` is unset (0) -- which it is here, since
     * this pseudopotential has no PP_QIJL data at all. Leaving the base
     * class's `maximum_angular_momentum` at its default of 0 would size the
     * table for L=0 only and this test would never reach L=2 regardless of
     * the fix under test.
     */
    pseudopotential->maximum_angular_momentum = 2;

    int mesh_size = 101;
    pseudopotential->mesh_size = mesh_size;
    pseudopotential->radial_grid.resize(mesh_size);
    pseudopotential->radial_grid_derivative.resize(mesh_size, 0.01);
    for (int i = 0; i < mesh_size; i++) {
        pseudopotential->radial_grid[i] = i * 0.01;
    }

    // Flat (L-undecomposed) Q_ij(r): a smooth bump, same shape for every pair.
    std::vector<Real> flat_function(mesh_size);
    for (int i = 0; i < mesh_size; i++) {
        Real r = pseudopotential->radial_grid[i];
        flat_function[i] = std::exp(-r * r);
    }
    pseudopotential->augmentation_function = {
        flat_function, flat_function, flat_function, flat_function}; // (0,0) (0,1) (1,0) (1,1)
    pseudopotential->augmentation_overlap_integrals = {{1.0, 0.5}, {0.5, 1.0}};
    /**
     * augmentation_function_by_angular_momentum deliberately left empty --
     * this is what the UPF reader leaves it as when no PP_QIJL data exists.
     */

    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(std::move(pseudopotential));

    AugmentationQTable table;
    table.init(pseudopotential_set,
               {mesh_size},
               /*number_of_q_points=*/201,
               /*q_spacing=*/0.01,
               /*cell_volume=*/270.0);

    /**
     * Off-diagonal (s, d) pair: only L=2 is triangle-rule-allowed, and it
     * must come from the flat function, not be silently dropped.
     */
    int off_diagonal_pair = 0 * 2 + 1;
    Real off_diagonal_at_l2 = table.interpolate_qrad(0, off_diagonal_pair, 2, 0.5);
    EXPECT_GT(std::abs(off_diagonal_at_l2), 1e-10)
        << "s-d augmentation charge at its only allowed L (2) must not be zero "
        << "when the pseudopotential only provides the flat, un-decomposed function";

    /**
     * Diagonal (d, d) pair at L=2 (allowed alongside L=0 and L=4 for l=l'=2)
     * must equally not be silently dropped.
     */
    int diagonal_pair = 1 * 2 + 1;
    Real diagonal_at_l2 = table.interpolate_qrad(0, diagonal_pair, 2, 0.5);
    EXPECT_GT(std::abs(diagonal_at_l2), 1e-10)
        << "d-d augmentation charge at L=2 must not be zero when only the flat "
        << "function is available";
}
