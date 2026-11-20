#include <gtest/gtest.h>

#include <cmath>

#include "layer_0_core/math/contracted_multiply_add.hpp"
#include "layer_0_core/types/init.hpp"

using namespace q3;

namespace {

/**
 * Operands whose fused and twice-rounded results differ: the exact product
 * (1 + ε)(1 − ε) = 1 − ε² needs more bits than a double has, so rounding it
 * before adding −1 gives zero, while the fused operation keeps −ε².
 */
const Real epsilon = std::ldexp(1.0, -30);
const Real factor_a = 1.0 + epsilon;
const Real factor_b = 1.0 - epsilon;
const Real addend = -1.0;

} // namespace

TEST(ContractedMultiplyAdd, TheTwoRoundingsOfTheTestOperandsDiffer) {
    volatile Real rounded_product = factor_a * factor_b;
    const Real twice_rounded = rounded_product + addend;

    EXPECT_EQ(twice_rounded, 0.0);
    EXPECT_EQ(std::fma(factor_a, factor_b, addend), -epsilon * epsilon);
}

TEST(ContractedMultiplyAdd, IsFusedExactlyWhereTheTargetHasAFusedMultiplyAdd) {
    const Real result = contracted_multiply_add(factor_a, factor_b, addend);

#ifdef FP_FAST_FMA
    EXPECT_EQ(result, -epsilon * epsilon);
#else
    EXPECT_EQ(result, 0.0);
#endif
}

TEST(ContractedMultiplyAdd, AgreesWithPlainArithmeticWhenTheProductIsExact) {
    EXPECT_EQ(contracted_multiply_add(3.0, 0.5, 0.25), 1.75);
    EXPECT_EQ(contracted_multiply_add(0.0, 7.0, -2.0), -2.0);
}
