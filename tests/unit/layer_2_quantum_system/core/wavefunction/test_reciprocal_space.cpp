#include "layer_2_quantum_system/core/wavefunction/reciprocal_space.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace q3;

/**
 * Concrete implementation of ReciprocalSpaceWavefunction for testing.
 *
 * Returns a fixed complex value for any reciprocal-space vector,
 * allowing us to test the abstract interface without real physics.
 */
class ConstantReciprocalWavefunction : public ReciprocalSpaceWavefunction {
  public:
    /**
     * Construct with a fixed return value for evaluate().
     */
    explicit ConstantReciprocalWavefunction(Complex value) : value_(value) {}

    [[nodiscard]] Complex evaluate(const ReciprocalSpaceVector& g_vector) const override {
        // Suppress unused parameter — constant wavefunction ignores input
        (void)g_vector;
        return value_;
    }

    [[nodiscard]] int number_of_components() const override {
        return 1;
    }

  private:
    Complex value_;
};

// ─── ReciprocalSpaceWavefunction ────────────────────────────────────────────

TEST(ReciprocalSpaceWavefunction, CanBeInstantiatedThroughConcreteSubclass) {
    ConstantReciprocalWavefunction wavefunction(Complex(1.0, 0.0));
    ReciprocalSpaceVector g_vector;
    g_vector.coordinates = {0.0, 0.0, 0.0};

    Complex result = wavefunction.evaluate(g_vector);
    EXPECT_DOUBLE_EQ(result.real(), 1.0);
    EXPECT_DOUBLE_EQ(result.imag(), 0.0);
}

TEST(ReciprocalSpaceWavefunction, EvaluateReturnsComplexValue) {
    ConstantReciprocalWavefunction wavefunction(Complex(0.5, -0.3));
    ReciprocalSpaceVector g_vector;
    g_vector.coordinates = {1.0, 2.0, 3.0};

    Complex result = wavefunction.evaluate(g_vector);
    EXPECT_DOUBLE_EQ(result.real(), 0.5);
    EXPECT_DOUBLE_EQ(result.imag(), -0.3);
}

TEST(ReciprocalSpaceWavefunction, NumberOfComponentsIsOne) {
    ConstantReciprocalWavefunction wavefunction(Complex(0.0, 0.0));
    EXPECT_EQ(wavefunction.number_of_components(), 1);
}

TEST(ReciprocalSpaceWavefunction, CanBeUsedThroughBasePointer) {
    std::unique_ptr<ReciprocalSpaceWavefunction> wavefunction =
        std::make_unique<ConstantReciprocalWavefunction>(Complex(2.0, 1.0));

    ReciprocalSpaceVector g_vector;
    g_vector.coordinates = {0.5, 0.5, 0.5};

    Complex result = wavefunction->evaluate(g_vector);
    EXPECT_DOUBLE_EQ(result.real(), 2.0);
    EXPECT_DOUBLE_EQ(result.imag(), 1.0);
}

TEST(ReciprocalSpaceWavefunction, CanBeUsedThroughWavefunctionBasePointer) {
    std::unique_ptr<Wavefunction> wavefunction =
        std::make_unique<ConstantReciprocalWavefunction>(Complex(0.0, 0.0));

    EXPECT_EQ(wavefunction->number_of_components(), 1);
}

TEST(ReciprocalSpaceWavefunction, EvaluateWithZeroVector) {
    ConstantReciprocalWavefunction wavefunction(Complex(3.14, -2.72));
    ReciprocalSpaceVector zero_vector;

    Complex result = wavefunction.evaluate(zero_vector);
    EXPECT_DOUBLE_EQ(result.real(), 3.14);
    EXPECT_DOUBLE_EQ(result.imag(), -2.72);
}
