#include "layer_2_quantum_system/derived/charge_density.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace q3;

/**
 * Concrete implementation of ChargeDensity for testing.
 *
 * Stores a fixed total charge and tracks whether reset() was called,
 * allowing tests to exercise the abstract interface.
 */
class TestableChargeDensity : public ChargeDensity {
  public:
    /**
     * Construct with a given number of spin components and total charge.
     */
    explicit TestableChargeDensity(int spin_components, Real total_charge = 0.0)
        : total_charge_(total_charge), was_reset_(false) {
        number_of_spin_components_ = spin_components;
    }

    [[nodiscard]] Real get_total_charge() const override {
        return total_charge_;
    }

    void reset() override {
        total_charge_ = 0.0;
        was_reset_ = true;
    }

    [[nodiscard]] bool was_reset() const {
        return was_reset_;
    }

    void set_total_charge(Real charge) {
        total_charge_ = charge;
    }

  private:
    Real total_charge_;
    bool was_reset_;
};

// ─── ChargeDensity ─────────────────────────────────────────────────────────

TEST(ChargeDensity, GetTotalChargeReturnsStoredValue) {
    TestableChargeDensity density(1, 8.0);
    EXPECT_DOUBLE_EQ(density.get_total_charge(), 8.0);
}

TEST(ChargeDensity, ResetSetsChargeToZero) {
    TestableChargeDensity density(1, 10.0);
    EXPECT_DOUBLE_EQ(density.get_total_charge(), 10.0);

    density.reset();
    EXPECT_DOUBLE_EQ(density.get_total_charge(), 0.0);
}

TEST(ChargeDensity, ResetTracksInvocation) {
    TestableChargeDensity density(1, 5.0);
    EXPECT_FALSE(density.was_reset());

    density.reset();
    EXPECT_TRUE(density.was_reset());
}

TEST(ChargeDensity, CanBeUsedThroughBasePointer) {
    std::unique_ptr<ChargeDensity> density = std::make_unique<TestableChargeDensity>(2, 12.0);

    EXPECT_EQ(density->number_of_spin_components(), 2);
    EXPECT_TRUE(density->is_spin_polarized());
    EXPECT_DOUBLE_EQ(density->get_total_charge(), 12.0);
}
