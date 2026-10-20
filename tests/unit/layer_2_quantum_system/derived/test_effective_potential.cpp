#include "layer_2_quantum_system/derived/effective_potential.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace q3;

/**
 * Concrete subclass of EffectivePotential for testing.
 *
 * Exposes the protected spin component member for test construction.
 */
class TestableEffectivePotential : public EffectivePotential {
  public:
    /**
     * Construct with a given number of spin components.
     */
    explicit TestableEffectivePotential(int spin_components) {
        number_of_spin_components_ = spin_components;
    }

    TestableEffectivePotential() = default;
};

// ─── EffectivePotential ─────────────────────────────────────────────────────

TEST(EffectivePotential, CanBeUsedThroughBasePointer) {
    std::unique_ptr<EffectivePotential> potential = std::make_unique<TestableEffectivePotential>(2);

    EXPECT_EQ(potential->number_of_spin_components(), 2);
    EXPECT_TRUE(potential->is_spin_polarized());
}

TEST(EffectivePotential, PolymorphicDeletionThroughBasePointer) {
    SpinResolvedEntity* entity = new TestableEffectivePotential(2);
    EXPECT_TRUE(entity->is_spin_polarized());
    delete entity;
}
