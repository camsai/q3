#pragma once

/**
 * Centralized LAPACK declarations for the Q3 codebase.
 *
 * All LAPACK extern "C" function declarations are collected here to avoid
 * duplication across translation units. Each routine links against the
 * system LAPACK library (e.g., macOS Accelerate or a linked LAPACK build).
 *
 * Routines included:
 *   - zheev_  : complex Hermitian eigensolver (standard)
 *   - zhegv_  : complex Hermitian eigensolver (generalized)
 *   - dsygv_  : real symmetric eigensolver (generalized)
 *   - dgels_  : real least-squares solver
 *   - dsytrf_ : real symmetric factorization (Bunch-Kaufman)
 *   - dsytri_ : real symmetric matrix inverse from dsytrf_ factors
 *   - zgesvd_ : complex singular value decomposition
 *   - dgesvd_ : real singular value decomposition
 *   - zgeqp3_ : complex QR factorization with column pivoting
 *   - zgetrf_ : complex general matrix LU factorization
 *   - zgetri_ : complex general matrix inverse from zgetrf_ factors
 */

#include <complex>

namespace q3::lapack {

extern "C" {

/**
 * ZHEEV: Complex Hermitian eigensolver (standard eigenproblem).
 * Computes all eigenvalues and, optionally, eigenvectors of a complex
 * Hermitian matrix A: A * v = lambda * v.
 */
void zheev_(const char* jobz,
            const char* uplo,
            const int* n,
            std::complex<double>* a,
            const int* lda,
            double* w,
            std::complex<double>* work,
            const int* lwork,
            double* rwork,
            int* info);

/**
 * ZHEGV: Complex Hermitian eigensolver (generalized eigenproblem).
 * Solves A * v = lambda * B * v for Hermitian A and positive-definite B.
 */
void zhegv_(const int* itype,
            const char* jobz,
            const char* uplo,
            const int* n,
            std::complex<double>* a,
            const int* lda,
            std::complex<double>* b,
            const int* ldb,
            double* w,
            std::complex<double>* work,
            const int* lwork,
            double* rwork,
            int* info);

/**
 * DSYGV: Real symmetric eigensolver (generalized eigenproblem).
 * Solves A * v = lambda * B * v for symmetric A and positive-definite B.
 */
void dsygv_(const int* itype,
            const char* jobz,
            const char* uplo,
            const int* n,
            double* a,
            const int* lda,
            double* b,
            const int* ldb,
            double* w,
            double* work,
            const int* lwork,
            int* info);

/**
 * DGELS: Real least-squares solver.
 * Solves overdetermined or underdetermined real linear systems using
 * QR or LQ factorization of A.
 */
void dgels_(const char* trans,
            const int* m,
            const int* n,
            const int* nrhs,
            double* a,
            const int* lda,
            double* b,
            const int* ldb,
            double* work,
            const int* lwork,
            int* info);

/**
 * DSYTRF: Real symmetric matrix factorization (Bunch-Kaufman).
 * Computes the factorization of a real symmetric matrix using the
 * Bunch-Kaufman diagonal pivoting method.
 */
void dsytrf_(const char* uplo,
             const int* n,
             double* a,
             const int* lda,
             int* ipiv,
             double* work,
             const int* lwork,
             int* info);

/**
 * DSYTRI: Real symmetric matrix inverse from DSYTRF factors.
 * Computes the inverse of a real symmetric matrix from the Bunch-Kaufman
 * factorization computed by DSYTRF.
 */
void dsytri_(const char* uplo,
             const int* n,
             double* a,
             const int* lda,
             const int* ipiv,
             double* work,
             int* info);

/**
 * ZGESVD: Complex singular value decomposition.
 * Computes the singular value decomposition A = U * Sigma * V**H of a
 * complex M-by-N matrix A, optionally computing the left and/or right
 * singular vectors.
 */
void zgesvd_(const char* jobu,
             const char* jobvt,
             const int* m,
             const int* n,
             std::complex<double>* a,
             const int* lda,
             double* s,
             std::complex<double>* u,
             const int* ldu,
             std::complex<double>* vt,
             const int* ldvt,
             std::complex<double>* work,
             const int* lwork,
             double* rwork,
             int* info);

/**
 * DGESVD: Real singular value decomposition.
 * Computes the singular value decomposition A = U * Sigma * V**T of a
 * real M-by-N matrix A, optionally computing the left and/or right
 * singular vectors.
 */
void dgesvd_(const char* jobu,
             const char* jobvt,
             const int* m,
             const int* n,
             double* a,
             const int* lda,
             double* s,
             double* u,
             const int* ldu,
             double* vt,
             const int* ldvt,
             double* work,
             const int* lwork,
             int* info);

/**
 * ZGEQP3: Complex QR factorization with column pivoting.
 * Computes a QR factorization with column pivoting of a complex M-by-N
 * matrix A: A * P = Q * R, using Level 3 BLAS.
 */
void zgeqp3_(const int* m,
             const int* n,
             std::complex<double>* a,
             const int* lda,
             int* jpvt,
             std::complex<double>* tau,
             std::complex<double>* work,
             const int* lwork,
             double* rwork,
             int* info);

/**
 * ZGETRF: Complex general matrix LU factorization.
 * Computes an LU factorization of a general M-by-N complex matrix A using
 * partial pivoting with row interchanges: A = P * L * U.
 */
void zgetrf_(
    const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv, int* info);

/**
 * ZGETRI: Complex general matrix inverse from ZGETRF factors.
 * Computes the inverse of a complex matrix using the LU factorization
 * computed by ZGETRF.
 */
void zgetri_(const int* n,
             std::complex<double>* a,
             const int* lda,
             const int* ipiv,
             std::complex<double>* work,
             const int* lwork,
             int* info);

} // extern "C"

} // namespace q3::lapack
