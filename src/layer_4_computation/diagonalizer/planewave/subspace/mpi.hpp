#pragma once

/**
 * MPISubspaceDiagonalizer — ScaLAPACK implementation of SubspaceDiagonalizer.
 *
 * Solves the generalized eigenvalue problem on a distributed process grid
 * using pzhegvx. Assumes the input matrices are already globally reduced
 * (redundant) on all ranks of the communicator, and redistributes them
 * to a block-cyclic layout for ScaLAPACK.
 *
 * This file self-guards with Q3_WITH_MPI so it can be safely included
 * from the factory without requiring the consumer to add #ifdef guards.
 */

#ifdef Q3_WITH_MPI

#include "io/logger.hpp"
#include "layer_0_core/execution/mpi.hpp"
#include "layer_0_core/math/linear_algebra/scalapack.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/serial.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace q3 {

class MPISubspaceDiagonalizer : public SubspaceDiagonalizer {
  public:
    explicit MPISubspaceDiagonalizer(const execution::MPIExecutionPolicy& policy)
        : policy_(policy) {}

    int solve(int basis_size,
              Complex* hamiltonian_matrix,
              Complex* overlap_matrix,
              Real* eigenvalues,
              Complex* eigenvector_matrix,
              bool is_gamma_only = false) override {
        if (basis_size == 0)
            return 0;

        MPI_Comm comm = policy_.get_group_communicator();
        if (comm == MPI_COMM_NULL)
            return -1;

        /**
         * A replicated group solves locally, never collectively.
         *
         * disable_group_distribution() -- the replicated-G fallback, taken when
         * a rank's FFT slab holds no plane waves -- sets group_size() to 1 but
         * deliberately leaves the group communicator with all its ranks, so
         * MPI_Comm_size() below still reports the full group. In that state
         * every rank holds the whole problem and steps through Davidson on its
         * own, reaching its own basis size at its own time. Entering a
         * collective solve from there had each rank calling group-wide BLACS
         * and MPI_Allreduce independently, with mismatched counts, which
         * aborted the job in MPI_Allreduce with "Other MPI error" and a
         * segmentation fault on every rank. It reproduced at any problem size
         * and only ever with ScaLAPACK enabled -- the same width completes
         * without it, because that path is already a local solve.
         *
         * MPIExecutionPolicy documents this: branch on group_size() <= 1, not
         * on group_rank_index() == 0.
         */
        if (policy_.group_size() <= 1) {
            SerialSubspaceDiagonalizer replicated_solver;
            return replicated_solver.solve(basis_size,
                                           hamiltonian_matrix,
                                           overlap_matrix,
                                           eigenvalues,
                                           eigenvector_matrix,
                                           is_gamma_only);
        }

#ifdef Q3_WITH_SCALAPACK
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        /**
         * 1. Set up BLACS grid
         * We use a square-ish grid for best ScaLAPACK performance.
         */
        int nprow = static_cast<int>(std::sqrt(size));
        while (size % nprow != 0)
            nprow--;
        int npcol = size / nprow;

        /**
         * Stay inside the C BLACS API. This previously took the handle from
         * Csys2blacs_handle (C) into blacs_gridinit_ (Fortran), which reads its
         * argument in a different numbering space: the handle came back as 1
         * and BLACS then used it as a communicator, so MPICH aborted in
         * MPI_Comm_group with "Invalid communicator, comm=0x1". It also went
         * through blacs_get_(handle, 0, &ctxt) first, which asks for the
         * *default* system context -- built on MPI_COMM_WORLD, not on `comm` --
         * so the grid would have spanned world ranks while nprow*npcol was
         * sized from `comm`, leaving the surplus ranks outside the grid.
         *
         * None of this had ever run: Q3 called pzhegv_, which is not in the
         * ScaLAPACK API, so no ScaLAPACK-enabled build linked.
         */
        int blacs_system_handle = scalapack::Csys2blacs_handle(comm);
        int blacs_context = blacs_system_handle;
        scalapack::Cblacs_gridinit(&blacs_context, "R", nprow, npcol);
        int myrow, mycol;
        scalapack::Cblacs_gridinfo(blacs_context, &nprow, &npcol, &myrow, &mycol);

        /**
         * Every rank of `comm` is in the grid, so this is defensive. A rank
         * outside the grid has no context to exit -- BLACS treats exiting one
         * as fatal rather than as a no-op -- but the system handle is still
         * this rank's to release.
         */
        if (myrow < 0 || mycol < 0) {
            scalapack::Cfree_blacs_system_handle(blacs_system_handle);
            return 0;
        }

        // 2. Set up descriptors
        int nb = 32; // Block size
        int desc[9];
        int info;
        int lld = scalapack::numroc_(&basis_size, &nb, &myrow, &zero_, &nprow);
        lld = std::max(1, lld);
        scalapack::descinit_(
            desc, &basis_size, &basis_size, &nb, &nb, &zero_, &zero_, &blacs_context, &lld, &info);

        // 3. Redistribute/Copy redundant data to local ScaLAPACK blocks
        int local_rows = scalapack::numroc_(&basis_size, &nb, &myrow, &zero_, &nprow);
        int local_cols = scalapack::numroc_(&basis_size, &nb, &mycol, &zero_, &npcol);
        std::vector<Complex> local_h(static_cast<size_t>(local_rows) * local_cols);
        std::vector<Complex> local_s(static_cast<size_t>(local_rows) * local_cols);
        std::vector<Complex> local_z(static_cast<size_t>(local_rows) * local_cols);

        for (int j_local = 0; j_local < local_cols; ++j_local) {
            // Map local column index to global column index
            int j_global = (j_local / nb) * npcol * nb + mycol * nb + (j_local % nb);
            for (int i_local = 0; i_local < local_rows; ++i_local) {
                int i_global = (i_local / nb) * nprow * nb + myrow * nb + (i_local % nb);
                local_h[j_local * local_rows + i_local] =
                    hamiltonian_matrix[j_global * basis_size + i_global];
                local_s[j_local * local_rows + i_local] =
                    overlap_matrix[j_global * basis_size + i_global];
            }
        }

        // 4. Solve
        int ibtype = 1; // H*x = lambda*S*x
        char jobz = 'V';
        char range = 'A'; // every eigenvalue; vl/vu/il/iu are then unused
        char uplo = 'U';
        int one = 1;

        /**
         * PZHEGVX selection and tolerance arguments. abstol <= 0 asks for the
         * default (ulp * ||A||), which is what the simple driver used to apply
         * implicitly. orfac is the cluster threshold above which eigenvectors
         * are reorthogonalized; 1e-3 is ScaLAPACK's own default.
         */
        double vl = 0.0;
        double vu = 0.0;
        int il = 1;
        int iu = basis_size;
        double abstol = 0.0;
        double orfac = 1.0e-3;
        int number_of_eigenvalues_found = 0;
        int number_of_eigenvectors_computed = 0;

        /**
         * Diagnostics PZHEGV did not have. iclustr and gap are sized by the
         * process grid rather than the matrix: iclustr holds a start/end index
         * pair per process and gap one value per process.
         */
        std::vector<int> ifail(static_cast<size_t>(std::max(1, basis_size)));
        std::vector<int> iclustr(static_cast<size_t>(2 * std::max(1, nprow * npcol)));
        std::vector<double> gap(static_cast<size_t>(std::max(1, nprow * npcol)));

        auto* cast_h = reinterpret_cast<std::complex<double>*>(local_h.data());
        auto* cast_s = reinterpret_cast<std::complex<double>*>(local_s.data());
        auto* cast_z = reinterpret_cast<std::complex<double>*>(local_z.data());

        /**
         * Workspace query: lwork/lrwork/liwork = -1 returns the sizes in the
         * first element of each workspace array.
         */
        Complex work_query;
        Real rwork_query = 0.0;
        int iwork_query = 0;
        int lwork = -1;
        int lrwork = -1;
        int liwork = -1;
        auto* cast_work_query = reinterpret_cast<std::complex<double>*>(&work_query);

        scalapack::pzhegvx_(&ibtype,
                            &jobz,
                            &range,
                            &uplo,
                            &basis_size,
                            cast_h,
                            &one,
                            &one,
                            desc,
                            cast_s,
                            &one,
                            &one,
                            desc,
                            &vl,
                            &vu,
                            &il,
                            &iu,
                            &abstol,
                            &number_of_eigenvalues_found,
                            &number_of_eigenvectors_computed,
                            eigenvalues,
                            &orfac,
                            cast_z,
                            &one,
                            &one,
                            desc,
                            cast_work_query,
                            &lwork,
                            &rwork_query,
                            &lrwork,
                            &iwork_query,
                            &liwork,
                            ifail.data(),
                            iclustr.data(),
                            gap.data(),
                            &info);

        /**
         * The real workspace the query reports is only enough when no
         * eigenvalue cluster needs reorthogonalizing; ScaLAPACK's own
         * documentation says so and recommends padding when jobz='V' and
         * orfac > 0. Under-allocating here does not fail loudly -- it returns
         * info with the "could not reorthogonalize" bit set and eigenvectors
         * that are not orthogonal, which would show up as a wrong subspace
         * rather than an error. The pad is proportional to the work a cluster
         * spanning the whole basis would need.
         */
        lwork = std::max(1, static_cast<int>(work_query.real()));
        lrwork = std::max(1, static_cast<int>(rwork_query)) + basis_size * basis_size;
        liwork = std::max(1, iwork_query);

        std::vector<Complex> work(static_cast<size_t>(lwork));
        std::vector<Real> rwork(static_cast<size_t>(lrwork));
        std::vector<int> iwork(static_cast<size_t>(liwork));

        auto* cast_work = reinterpret_cast<std::complex<double>*>(work.data());

        scalapack::pzhegvx_(&ibtype,
                            &jobz,
                            &range,
                            &uplo,
                            &basis_size,
                            cast_h,
                            &one,
                            &one,
                            desc,
                            cast_s,
                            &one,
                            &one,
                            desc,
                            &vl,
                            &vu,
                            &il,
                            &iu,
                            &abstol,
                            &number_of_eigenvalues_found,
                            &number_of_eigenvectors_computed,
                            eigenvalues,
                            &orfac,
                            cast_z,
                            &one,
                            &one,
                            desc,
                            cast_work,
                            &lwork,
                            rwork.data(),
                            &lrwork,
                            iwork.data(),
                            &liwork,
                            ifail.data(),
                            iclustr.data(),
                            gap.data(),
                            &info);

        /**
         * A short solve is as wrong as a failed one, and the expert driver can
         * return fewer than n of either: info carries "eigenvectors did not
         * converge" in bit 0 and "could not reorthogonalize" in bit 1. The
         * caller treats a non-zero return as a failure to fall back on, so
         * report a short result that way rather than leaving the tail of
         * eigenvalues and eigenvectors at whatever they held.
         */
        if (info == 0 && (number_of_eigenvalues_found != basis_size ||
                          number_of_eigenvectors_computed != basis_size)) {
            log::warn("ScaLAPACK subspace solve returned {} of {} eigenvalues and {} of {} "
                      "eigenvectors; treating as a failure so the caller can fall back.",
                      number_of_eigenvalues_found,
                      basis_size,
                      number_of_eigenvectors_computed,
                      basis_size);
            info = -1;
        }

        /**
         * 5. Redistribute results back to all ranks (redundant)
         * Eigenvalues are already redundant (returned on all ranks by pzhegvx).
         * Eigenvectors (local_z) need to be gathered.
         */
        std::vector<Complex> global_z(static_cast<size_t>(basis_size) * basis_size,
                                      Complex(0.0, 0.0));
        for (int j_local = 0; j_local < local_cols; ++j_local) {
            int j_global = (j_local / nb) * npcol * nb + mycol * nb + (j_local % nb);
            for (int i_local = 0; i_local < local_rows; ++i_local) {
                int i_global = (i_local / nb) * nprow * nb + myrow * nb + (i_local % nb);
                global_z[j_global * basis_size + i_global] =
                    local_z[j_local * local_rows + i_local];
            }
        }

        /**
         * Global reduction (sum) to make eigenvector_matrix redundant.
         * Each rank contributed its local pieces, so a SUM reduction gathers the full matrix.
         */
        execution::sum_complex(global_z.data(), eigenvector_matrix, basis_size * basis_size, comm);

        scalapack::Cblacs_gridexit(blacs_context);
        scalapack::Cfree_blacs_system_handle(blacs_system_handle);
        return info;
#else
        // Fallback: Redundant serial solve on each rank.
        SerialSubspaceDiagonalizer serial_solver;
        return serial_solver.solve(basis_size,
                                   hamiltonian_matrix,
                                   overlap_matrix,
                                   eigenvalues,
                                   eigenvector_matrix,
                                   is_gamma_only);
#endif
    }

    /**
     * ScaLAPACK solve() is collective (BLACS grid + pzhegvx + Allreduce span the
     * whole group communicator); the redundant-serial fallback is local-only,
     * and so is a replicated group, which solve() routes to the serial path.
     * This must agree with solve(): the subspace manager reads it to decide
     * whether the group root broadcasts the result, and a solver that claims to
     * be collective while solving locally would leave that decision resting on
     * a false premise.
     */
    bool is_collective() const override {
#ifdef Q3_WITH_SCALAPACK
        return policy_.group_size() > 1;
#else
        return false;
#endif
    }

    std::string name() const override {
        return "Parallel (ScaLAPACK)";
    }

  private:
    const execution::MPIExecutionPolicy& policy_;
    const int zero_ = 0;
};

} // namespace q3

#endif // Q3_WITH_MPI
