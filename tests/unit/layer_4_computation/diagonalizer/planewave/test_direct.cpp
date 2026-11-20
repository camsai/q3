#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set_store.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder.hpp"
#include "layer_4_computation/diagonalizer/planewave/direct.hpp"

using namespace q3;

// ── Name and basic properties ───────────────────────────────────────

TEST(PlanewaveDirectDiagonalizer, NameReturnsDirect) {
    PlanewaveDirectDiagonalizer diagonalizer;
    EXPECT_EQ(diagonalizer.name(), "direct");
}

// ── The projector set store ─────────────────────────────────────────

namespace {

/// A builder that is never asked to build: these tests only hand the store over.
class UnusedProjectorSetBuilder : public ProjectorSetBuilder {
  public:
    void build(const Vector3D& /*kpoint_crystal_coordinates*/,
               const std::vector<int>& /*g_vector_indices*/,
               int /*number_of_plane_waves*/,
               KleinmanBylanderProjectorSet& /*projector_set*/) const override {
        throw std::logic_error("UnusedProjectorSetBuilder: build() was called");
    }
};

/// A store for one k-point, never read.
struct UnreadProjectorSetStore {
    Crystal crystal_structure;
    KPointsGrid kpoint_grid;
    std::vector<std::vector<int>> g_vector_indices_by_kpoint;
    std::unique_ptr<KleinmanBylanderProjectorSetStore> store;

    UnreadProjectorSetStore() {
        kpoint_grid.generate(1, 1, 1, 0, 0, 0);
        g_vector_indices_by_kpoint.resize(1);
        store = std::make_unique<KleinmanBylanderProjectorSetStore>(
            crystal_structure,
            kpoint_grid,
            g_vector_indices_by_kpoint,
            std::make_unique<UnusedProjectorSetBuilder>());
    }
};

} // namespace

TEST(PlanewaveDirectDiagonalizer, TakesAProjectorSetStoreWithoutReadingIt) {
    PlanewaveDirectDiagonalizer diagonalizer;
    UnreadProjectorSetStore first;
    UnreadProjectorSetStore second;

    // Handing a store over, and replacing it, builds nothing: sets are built when solve() asks.
    diagonalizer.set_projector_set_store(first.store.get());
    diagonalizer.set_projector_set_store(second.store.get());

    EXPECT_FALSE(first.store->is_built(0));
    EXPECT_FALSE(second.store->is_built(0));
}

TEST(PlanewaveDirectDiagonalizer, AProjectorSetStoreCanBeTakenBack) {
    PlanewaveDirectDiagonalizer diagonalizer;
    UnreadProjectorSetStore unread;

    diagonalizer.set_projector_set_store(unread.store.get());
    diagonalizer.set_projector_set_store(nullptr);
    // Taking back a store that was never given is fine too.
    diagonalizer.set_projector_set_store(nullptr);

    EXPECT_FALSE(unread.store->is_built(0));
}

// ── Default virtual method no-ops ───────────────────────────────────

TEST(PlanewaveDirectDiagonalizer, SetToleranceIsNoOp) {
    PlanewaveDirectDiagonalizer diagonalizer;
    // Should not crash; direct solver ignores tolerance
    diagonalizer.set_tolerance(1e-8);
}

TEST(PlanewaveDirectDiagonalizer, SetFirstIterationIsNoOp) {
    PlanewaveDirectDiagonalizer diagonalizer;
    // Should not crash; direct solver ignores this flag
    diagonalizer.set_first_iteration(true);
    diagonalizer.set_first_iteration(false);
}

TEST(PlanewaveDirectDiagonalizer, InvalidateSmoothPotentialCacheIsNoOp) {
    PlanewaveDirectDiagonalizer diagonalizer;
    // Should not crash
    diagonalizer.invalidate_smooth_potential_cache();
}
