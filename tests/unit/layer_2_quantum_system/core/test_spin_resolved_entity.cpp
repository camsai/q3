#include "layer_2_quantum_system/core/spin_resolved_entity.hpp"
#include <gtest/gtest.h>

using namespace q3;

/**
 * Concrete subclass exposing SpinResolvedEntity's protected member
 * for testing purposes.
 */
class TestableSpinResolvedEntity : public SpinResolvedEntity {
  public:
    /**
     * Construct with a given number of spin components.
     */
    explicit TestableSpinResolvedEntity(int spin_components) {
        number_of_spin_components_ = spin_components;
    }

    TestableSpinResolvedEntity() = default;
};

// ─── SpinResolvedEntity ─────────────────────────────────────────────────────

TEST(SpinResolvedEntity, DefaultNumberOfSpinComponentsIsOne) {
    TestableSpinResolvedEntity entity;
    EXPECT_EQ(entity.number_of_spin_components(), 1);
}

TEST(SpinResolvedEntity, DefaultIsNotSpinPolarized) {
    TestableSpinResolvedEntity entity;
    EXPECT_FALSE(entity.is_spin_polarized());
}

TEST(SpinResolvedEntity, TwoSpinComponentsIsPolarized) {
    TestableSpinResolvedEntity entity(2);
    EXPECT_EQ(entity.number_of_spin_components(), 2);
    EXPECT_TRUE(entity.is_spin_polarized());
}

TEST(SpinResolvedEntity, MoreThanTwoSpinComponentsIsPolarized) {
    TestableSpinResolvedEntity entity(4);
    EXPECT_EQ(entity.number_of_spin_components(), 4);
    EXPECT_TRUE(entity.is_spin_polarized());
}

TEST(SpinResolvedEntity, CanBeDeletedThroughBasePointer) {
    SpinResolvedEntity* entity = new TestableSpinResolvedEntity(2);
    EXPECT_EQ(entity->number_of_spin_components(), 2);
    EXPECT_TRUE(entity->is_spin_polarized());
    delete entity;
}
