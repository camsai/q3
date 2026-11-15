/**
 * LocalDensityGLayout and the compact density transforms built on it.
 *
 * On a grid that does not distribute, every G-vector is local: the layout is the identity, a
 * compact vector is the full one, and the compact transforms must reproduce the full ones
 * exactly. The distributed case -- each rank a share, sums completed across the pool -- is
 * exercised end to end by the MPI topology-invariance validation, which compares energies
 * and forces against a one-rank run.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/transform/local_density_g_layout.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

Crystal two_atom_cell() {
    Crystal crystal;
    crystal.set_from_lattice_type(2, 10.26);
    crystal.add_species(AtomSpecies{"Si", 28.08, 4.0, ""});
    crystal.add_atom({0, {0.0, 0.0, 0.0}});
    crystal.add_atom({0, {0.27, 0.23, 0.25}});
    crystal.compute_derived_quantities();
    return crystal;
}

std::vector<Real> pseudo_random_field(int points, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<Real> uniform(0.0, 1.0);
    std::vector<Real> field(static_cast<std::size_t>(points));
    for (auto& value : field) {
        value = uniform(generator);
    }
    return field;
}

} // namespace

TEST(LocalDensityGLayout, SerialGridHoldsEveryGVectorInOrder) {
    Crystal crystal = two_atom_cell();
    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal, 10.0, 40.0, false, false);
    const LocalDensityGLayout& layout = basis.local_density_g_layout();

    ASSERT_EQ(layout.size(), basis.number_of_g_vectors());
    EXPECT_TRUE(layout.holds_every_g_vector());
    for (int entry = 0; entry < layout.size(); ++entry) {
        EXPECT_EQ(layout.global_g_indices()[entry], entry);
        EXPECT_EQ(layout.mirror_fft_indices()[entry], -1) << "no mirrors off the gamma half sphere";
    }
    EXPECT_EQ(layout.count_below(0), 0);
    EXPECT_EQ(layout.count_below(7), 7);
    EXPECT_EQ(layout.count_below(basis.number_of_g_vectors() + 5), layout.size());

    std::vector<Real> full(static_cast<std::size_t>(basis.number_of_g_vectors()));
    for (std::size_t index = 0; index < full.size(); ++index) {
        full[index] = 0.5 * static_cast<Real>(index);
    }
    EXPECT_EQ(layout.compact(full), full);
}

TEST(LocalDensityGLayout, CompactTransformsMatchTheFullOnes) {
    Crystal crystal = two_atom_cell();
    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal, 10.0, 40.0, false, false);
    const LocalDensityGLayout& layout = basis.local_density_g_layout();

    std::vector<Real> field = pseudo_random_field(basis.fft_rho().local_grid_points(), 3);
    std::vector<Complex> full_g;
    std::vector<Complex> compact_g;
    basis.rho_r_to_g(field, full_g);
    basis.rho_r_to_compact_g(field, compact_g);
    ASSERT_EQ(static_cast<int>(compact_g.size()), layout.size());
    for (int entry = 0; entry < layout.size(); ++entry) {
        EXPECT_EQ(compact_g[entry], full_g[layout.global_g_indices()[entry]]);
    }

    std::vector<Real> from_full;
    std::vector<Real> from_compact;
    basis.rho_g_to_r(full_g, from_full);
    basis.rho_compact_g_to_r(compact_g, from_compact);
    ASSERT_EQ(from_full.size(), from_compact.size());
    for (std::size_t index = 0; index < from_full.size(); ++index) {
        EXPECT_EQ(from_full[index], from_compact[index]);
    }
}

TEST(LocalDensityGLayout, GammaOnlyEntriesCarryTheirMirror) {
    Crystal crystal = two_atom_cell();
    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal, 10.0, 40.0, /*is_gamma_only=*/true, false);
    const LocalDensityGLayout& layout = basis.local_density_g_layout();
    const int first_nonzero = basis.first_nonzero_g_index();
    for (int entry = 0; entry < layout.size(); ++entry) {
        if (layout.global_g_indices()[entry] < first_nonzero) {
            EXPECT_EQ(layout.mirror_fft_indices()[entry], -1) << "G = 0 is its own mirror";
        } else {
            EXPECT_GE(layout.mirror_fft_indices()[entry], 0);
        }
    }

    // The half sphere's transform writes conj at -G: compact and full agree there too.
    std::vector<Real> field = pseudo_random_field(basis.fft_rho().local_grid_points(), 5);
    std::vector<Complex> full_g;
    std::vector<Complex> compact_g;
    basis.rho_r_to_g(field, full_g);
    basis.rho_r_to_compact_g(field, compact_g);
    std::vector<Real> from_full;
    std::vector<Real> from_compact;
    basis.rho_g_to_r(full_g, from_full);
    basis.rho_compact_g_to_r(compact_g, from_compact);
    for (std::size_t index = 0; index < from_full.size(); ++index) {
        EXPECT_EQ(from_full[index], from_compact[index]);
    }
}

TEST(LocalDensityGLayout, CompletingSumsIsANoOpWhenEveryRankHoldsEverything) {
    Crystal crystal = two_atom_cell();
    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal, 10.0, 40.0, false, false);
    std::vector<Real> sums{1.5, -2.0, 3.25};
    basis.complete_density_g_sums(sums);
    EXPECT_EQ(sums, (std::vector<Real>{1.5, -2.0, 3.25}));
    std::vector<Vector3D> forces{{1.0, 2.0, 3.0}};
    basis.complete_density_g_sums(forces);
    EXPECT_EQ(forces[0][0], 1.0);
    EXPECT_EQ(forces[0][2], 3.0);
}
