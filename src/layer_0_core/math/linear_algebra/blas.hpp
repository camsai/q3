#pragma once

/**
 * BLAS declarations for complex double-precision routines.
 *
 * These are the standard Fortran BLAS interfaces used by the Davidson
 * diagonalizer for matrix-matrix and matrix-vector operations. On macOS,
 * these are provided by the Accelerate framework (same as LAPACK).
 */

#include <complex>

namespace q3::blas {

extern "C" {

/**
 * ZGEMM: Complex double-precision general matrix-matrix multiply.
 * C := alpha * op(A) * op(B) + beta * C
 */
void zgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const std::complex<double>* alpha,
            const std::complex<double>* a,
            const int* lda,
            const std::complex<double>* b,
            const int* ldb,
            const std::complex<double>* beta,
            std::complex<double>* c,
            const int* ldc);

/**
 * DGEMM: Real double-precision general matrix-matrix multiply.
 * C := alpha * op(A) * op(B) + beta * C
 *
 * Used in gamma-only inner products where the result is real-valued.
 * Two DGEMM calls on the real and imaginary parts of the complex data
 * give a 2× FLOP reduction compared to a single ZGEMM.
 */
void dgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const double* alpha,
            const double* a,
            const int* lda,
            const double* b,
            const int* ldb,
            const double* beta,
            double* c,
            const int* ldc);

/**
 * ZGEMV: Complex double-precision general matrix-vector multiply.
 * y := alpha * op(A) * x + beta * y
 */
void zgemv_(const char* trans,
            const int* m,
            const int* n,
            const std::complex<double>* alpha,
            const std::complex<double>* a,
            const int* lda,
            const std::complex<double>* x,
            const int* incx,
            const std::complex<double>* beta,
            std::complex<double>* y,
            const int* incy);

/**
 * DGEMV: Real double-precision general matrix-vector multiply.
 * y := alpha * op(A) * x + beta * y
 */
void dgemv_(const char* trans,
            const int* m,
            const int* n,
            const double* alpha,
            const double* a,
            const int* lda,
            const double* x,
            const int* incx,
            const double* beta,
            double* y,
            const int* incy);

} // extern "C"

/**
 * ZDOTC wrapper: Complex double-precision dot product with conjugation.
 * result := Σ conj(x[element_index]) * y[element_index]
 *
 * Uses manual loop to avoid zdotc_ ABI incompatibility on macOS Accelerate
 * (hidden return-value pointer). BLAS Level-1 operations on vectors of
 * typical DFT size (~100-1000) are memory-bound, so the manual loop
 * performs comparably to the BLAS call.
 */
inline std::complex<double> q3_complex_conjugate_dot_product(int n,
                                                             const std::complex<double>* x,
                                                             const std::complex<double>* y) {
    std::complex<double> result(0.0, 0.0);
    for (int element_index = 0; element_index < n; element_index++) {
        result += std::conj(x[element_index]) * y[element_index];
    }
    return result;
}

extern "C" {

/**
 * ZAXPY: Complex double-precision vector addition with scalar.
 * y := alpha * x + y
 */
void zaxpy_(const int* n,
            const std::complex<double>* alpha,
            const std::complex<double>* x,
            const int* incx,
            std::complex<double>* y,
            const int* incy);

/**
 * ZDSCAL: Scale a complex double-precision vector by a real scalar.
 * x := alpha * x
 */
void zdscal_(const int* n, const double* alpha, std::complex<double>* x, const int* incx);

/**
 * DZNRM2: Euclidean norm of a complex double-precision vector.
 * result := sqrt( Σ |x[i]|² )
 */
double dznrm2_(const int* n, const std::complex<double>* x, const int* incx);

/**
 * DPOTRF: Real double-precision Cholesky factorization.
 */
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);

/**
 * ZPOTRF: Complex double-precision Cholesky factorization.
 */
void zpotrf_(const char* uplo, const int* n, std::complex<double>* a, const int* lda, int* info);

/**
 * DTRSM: Real double-precision triangular matrix solve.
 */
void dtrsm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const double* alpha,
            const double* a,
            const int* lda,
            double* b,
            const int* ldb);

/**
 * ZTRSM: Complex double-precision triangular matrix solve.
 */
void ztrsm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const std::complex<double>* alpha,
            const std::complex<double>* a,
            const int* lda,
            std::complex<double>* b,
            const int* ldb);

} // extern "C"

} // namespace q3::blas
