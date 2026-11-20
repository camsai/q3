#pragma once

/**
 * a·b + c with the choice between one rounding and two written down.
 *
 * Q3 is compiled with -ffp-contract=fast, which lets the compiler turn a
 * product followed by a sum into one fused multiply-add wherever the target
 * has the instruction. It decides loop by loop, and decides differently when
 * the code around a loop changes: the band-energy sum was fused while each
 * state was a large object, and was no longer fused once the states shrank to
 * an eigenvalue and an occupation, with the loop's own source untouched. The
 * two results differ in the last bit.
 *
 * A sum that must not move when storage is rearranged calls this instead of
 * leaving the choice to the optimizer. It makes the choice a scalar loop under
 * -ffp-contract=fast makes: fused where the target has a fused multiply-add
 * (FP_FAST_FMA), a rounded product and a rounded sum where it has none, so
 * neither kind of target sees a result change.
 */

#include <cmath>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// factor_a · factor_b + addend, fused where the target has a fused multiply-add.
[[nodiscard]] inline Real contracted_multiply_add(Real factor_a, Real factor_b, Real addend) {
#ifdef FP_FAST_FMA
    return std::fma(factor_a, factor_b, addend);
#else
    return factor_a * factor_b + addend;
#endif
}

} // namespace q3
