#pragma once

/**
 * ScaLAPACK declarations for distributed complex double-precision routines.
 *
 * Provides the Fortran interfaces for BLACS grid management and distributed
 * eigensolvers (PZHEGVX).
 */

#include <complex>

#include <mpi.h>

namespace q3::scalapack {

extern "C" {

/**
 * BLACS grid management, C entry points.
 *
 * These must not be mixed with the Fortran `blacs_grid*_` entry points.
 * Csys2blacs_handle returns a handle in the C layer's numbering space, and
 * handing that to Fortran blacs_gridinit_ makes BLACS read it as a
 * communicator: it returned 1 here, and MPICH then aborted inside
 * MPI_Comm_group with "Invalid communicator, comm=0x1". Going through
 * Cblacs_gridinit instead gives each communicator its own grid correctly.
 *
 * Csys2blacs_handle takes a C MPI_Comm, hence <mpi.h> above; typing it int
 * would compile only where MPI_Comm is an integer (MPICH yes, OpenMPI no).
 * Note also that Cblacs_gridinfo and Cblacs_gridexit take the context BY
 * VALUE, unlike their Fortran counterparts which take a pointer.
 *
 * The system handle from Csys2blacs_handle is a separate resource from the grid
 * context Cblacs_gridinit derives from it, and each needs releasing:
 * Cblacs_gridexit for the grid, Cfree_blacs_system_handle for the handle.
 */
int Csys2blacs_handle(MPI_Comm mpi_comm);
void Cfree_blacs_system_handle(int system_handle);
void Cblacs_gridinit(int* context, const char* layout, int nprow, int npcol);
void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
void Cblacs_gridexit(int context);

/**
 * BLACS: Exit from BLACS.
 */
void blacs_exit_(const int* continue_mpi);

/**
 * DESCINIT: Initialize a ScaLAPACK descriptor.
 */
void descinit_(int* desc,
               const int* m,
               const int* n,
               const int* mb,
               const int* nb,
               const int* irsrc,
               const int* icsrc,
               const int* ictxt,
               const int* lld,
               int* info);

/**
 * NUMROC: Compute number of rows/cols on a local process.
 */
int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);

/**
 * PZHEGVX: Parallel generalized Hermitian-definite eigenvalue problem.
 *
 * The expert driver, because it is the only one ScaLAPACK offers for the
 * generalized problem. Q3 previously declared and called `pzhegv_`, the simple
 * driver by analogy with LAPACK's ZHEGV -- but ScaLAPACK has no PZHEGV. The API
 * provides PZHEEV, PZHEEVD and PZHEEVX for the standard problem and PZHEGVX
 * alone for the generalized one, so every ScaLAPACK-enabled build failed to
 * link with `undefined reference to pzhegv_' against any conforming
 * implementation, Cray libsci included.
 *
 * Beyond PZHEGV's arguments this takes the eigenvalue selection (`range`, `vl`,
 * `vu`, `il`, `iu`, `abstol`), returns how many it found (`m`) and how many
 * eigenvectors it computed (`nz`), takes a reorthogonalization threshold
 * (`orfac`), and needs integer workspace plus the `ifail`, `iclustr` and `gap`
 * diagnostic arrays.
 */
void pzhegvx_(const int* ibtype,
              const char* jobz,
              const char* range,
              const char* uplo,
              const int* n,
              std::complex<double>* a,
              const int* ia,
              const int* ja,
              const int* desca,
              std::complex<double>* b,
              const int* ib,
              const int* jb,
              const int* descb,
              const double* vl,
              const double* vu,
              const int* il,
              const int* iu,
              const double* abstol,
              int* m,
              int* nz,
              double* w,
              const double* orfac,
              std::complex<double>* z,
              const int* iz,
              const int* jz,
              const int* descz,
              std::complex<double>* work,
              const int* lwork,
              double* rwork,
              const int* lrwork,
              int* iwork,
              const int* liwork,
              int* ifail,
              int* iclustr,
              double* gap,
              int* info);

} // extern "C"

} // namespace q3::scalapack
