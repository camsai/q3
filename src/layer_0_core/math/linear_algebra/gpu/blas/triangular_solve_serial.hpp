#pragma once

/**
 * Triangular solves through the host BLAS, for device backends whose unified
 * memory the host can address directly.
 *
 * Kept out of gpu/blas/init.hpp on purpose: that header is included by the
 * Metal backend beside Accelerate, and Accelerate's prototypes for these
 * symbols differ from Q3's. A backend that wants the host solve includes this
 * file from its own source instead.
 */

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {
namespace execution {
namespace gpu {
namespace blas {

/// ZTRSM on host-addressable memory; flags as in BLAS.
inline void triangular_solve_serial(char side,
                                    char upper_or_lower,
                                    char transpose_a,
                                    char unit_diagonal,
                                    int m,
                                    int n,
                                    Complex alpha,
                                    const Complex* matrix_a,
                                    int leading_dimension_a,
                                    Complex* matrix_b,
                                    int leading_dimension_b) {
    q3::blas::ztrsm_(&side,
                     &upper_or_lower,
                     &transpose_a,
                     &unit_diagonal,
                     &m,
                     &n,
                     &alpha,
                     matrix_a,
                     &leading_dimension_a,
                     matrix_b,
                     &leading_dimension_b);
}

/// DTRSM on host-addressable memory; flags as in BLAS.
inline void triangular_solve_serial(char side,
                                    char upper_or_lower,
                                    char transpose_a,
                                    char unit_diagonal,
                                    int m,
                                    int n,
                                    Real alpha,
                                    const Real* matrix_a,
                                    int leading_dimension_a,
                                    Real* matrix_b,
                                    int leading_dimension_b) {
    q3::blas::dtrsm_(&side,
                     &upper_or_lower,
                     &transpose_a,
                     &unit_diagonal,
                     &m,
                     &n,
                     &alpha,
                     matrix_a,
                     &leading_dimension_a,
                     matrix_b,
                     &leading_dimension_b);
}

} // namespace blas
} // namespace gpu
} // namespace execution
} // namespace q3
