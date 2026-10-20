/**
 * MPI unit tests for SerialPlanewaveSubspaceManager's redundant-solve path.
 *
 * Run via: mpirun -np N ./build_mpi/q3_tests_mpi   (N >= 2 to be meaningful)
 *
 * These cover the case where band distribution is disabled at run time, after
 * the subspace manager has already been constructed. The execution policy then
 * reports group_size() == 1 while every rank keeps its pool-local
 * group_rank_index(), so a solve gated on group_rank_index() == 0 paired with a
 * broadcast gated on group_size() > 1 would leave the non-root ranks holding
 * nothing at all -- and diverging from the root at the next collective.
 *
 * They also cover the gamma trick under a reciprocal-space distribution: every
 * inner product is 2 Re over the stored half sphere with the origin removed
 * once, and when the half sphere is split across the pool only the rank whose
 * list begins with G = 0 may remove it. The distributed result is checked
 * against one manager holding the whole half sphere.
 */

#include "layer_0_core/execution/mpi.hpp"
#include "layer_0_core/execution/serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/serial.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/mpi.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <vector>

namespace q3 {
namespace {

/**
 * A two-by-two generalized problem with an identity overlap, whose eigenvalues
 * are 1 and 3. Small enough that every rank solving it redundantly is free, and
 * fixed so the expected answer needs no reference implementation.
 */
void build_two_by_two_problem(std::vector<Complex>& subspace_hamiltonian,
                              std::vector<Complex>& subspace_overlap) {
    subspace_hamiltonian = {
        Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)};
    subspace_overlap = {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};
}

TEST(SerialPlanewaveSubspaceManagerMPI, EveryRankSolvesWhenBandDistributionIsDisabled) {
    execution::PlanewaveMPIExecutionPolicy execution_policy;
    execution_policy.configure_topology(1);

    if (execution_policy.world_size() < 2) {
        GTEST_SKIP() << "needs at least 2 ranks to distinguish root from non-root";
    }

    /**
     * Construct the manager while the group is still distributed, exactly as the
     * iterator does: the diagonalizer is built before the plane-wave data that
     * later triggers the fallback.
     */
    SerialPlanewaveSubspaceManager subspace_manager(execution_policy);

    execution_policy.disable_band_distribution();
    ASSERT_EQ(execution_policy.group_size(), 1);

    const int basis_size = 2;
    std::vector<Complex> subspace_hamiltonian;
    std::vector<Complex> subspace_overlap;
    build_two_by_two_problem(subspace_hamiltonian, subspace_overlap);

    std::vector<Real> eigenvalues(basis_size, 0.0);
    std::vector<Complex> eigenvector_matrix(basis_size * basis_size, Complex(0.0, 0.0));

    const int info = subspace_manager.diagonalize_subspace(basis_size,
                                                           basis_size,
                                                           false,
                                                           subspace_hamiltonian.data(),
                                                           subspace_overlap.data(),
                                                           eigenvalues.data(),
                                                           eigenvector_matrix.data());

    EXPECT_EQ(info, 0);
    EXPECT_NEAR(eigenvalues[0], 1.0, 1.0e-10);
    EXPECT_NEAR(eigenvalues[1], 3.0, 1.0e-10);

    /**
     * The eigenvector matrix must not be left at its zero initialization. That
     * is the observable symptom of the missed broadcast: arithmetic downstream
     * stays finite, so nothing crashes -- the ranks simply stop agreeing.
     */
    Real largest_magnitude = 0.0;
    for (const Complex& element : eigenvector_matrix) {
        largest_magnitude = std::max(largest_magnitude, std::abs(element));
    }
    EXPECT_GT(largest_magnitude, 1.0e-10);
}

TEST(SerialPlanewaveSubspaceManagerMPI, AllRanksAgreeOnTheEigenvaluesTheySolved) {
    execution::PlanewaveMPIExecutionPolicy execution_policy;
    execution_policy.configure_topology(1);

    if (execution_policy.world_size() < 2) {
        GTEST_SKIP() << "needs at least 2 ranks to compare results across ranks";
    }

    SerialPlanewaveSubspaceManager subspace_manager(execution_policy);
    execution_policy.disable_band_distribution();

    const int basis_size = 2;
    std::vector<Complex> subspace_hamiltonian;
    std::vector<Complex> subspace_overlap;
    build_two_by_two_problem(subspace_hamiltonian, subspace_overlap);

    std::vector<Real> eigenvalues(basis_size, 0.0);
    std::vector<Complex> eigenvector_matrix(basis_size * basis_size, Complex(0.0, 0.0));
    subspace_manager.diagonalize_subspace(basis_size,
                                          basis_size,
                                          false,
                                          subspace_hamiltonian.data(),
                                          subspace_overlap.data(),
                                          eigenvalues.data(),
                                          eigenvector_matrix.data());

    std::vector<Real> root_eigenvalues = eigenvalues;
    MPI_Bcast(root_eigenvalues.data(), basis_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int index = 0; index < basis_size; index++) {
        EXPECT_NEAR(eigenvalues[index], root_eigenvalues[index], 1.0e-12)
            << "rank " << execution_policy.world_rank_index() << " disagrees with the root at "
            << index;
    }
}

} // namespace

// ─── Gamma trick with the origin on one rank ────────────────────────────────

namespace {

/**
 * A gamma-only half sphere of `global_plane_waves` entries, cut into contiguous
 * blocks across the pool so that G = 0 -- entry 0, kept real -- lands on rank 0
 * alone. Deterministic, so every rank rebuilds the whole set and can run the
 * replicated reference itself.
 */
struct DistributedHalfSphere {
    int global_plane_waves = 0;
    int local_plane_waves = 0;
    int local_offset = 0;
    int number_of_vectors = 0;
    std::vector<Complex> global_vectors;
    std::vector<Complex> local_vectors;

    [[nodiscard]] const Complex* global_column(int vector_index) const {
        return &global_vectors[static_cast<std::size_t>(vector_index) * global_plane_waves];
    }
    [[nodiscard]] Complex* local_column(int vector_index) {
        return &local_vectors[static_cast<std::size_t>(vector_index) * local_plane_waves];
    }
};

Complex half_sphere_entry(int g_index, int vector_index, double seed) {
    if (g_index == 0) {
        return Complex(std::cos(1.3 * vector_index + seed), 0.0);
    }
    return Complex(std::cos(0.37 * g_index + 1.3 * vector_index + seed),
                   std::sin(0.11 * g_index * (vector_index + 1) + 0.7 * vector_index + seed));
}

DistributedHalfSphere build_half_sphere(
    int group_size, int group_rank, int global_plane_waves, int number_of_vectors, double seed) {
    DistributedHalfSphere sphere;
    sphere.global_plane_waves = global_plane_waves;
    sphere.number_of_vectors = number_of_vectors;
    int base = global_plane_waves / group_size;
    int remainder = global_plane_waves % group_size;
    sphere.local_plane_waves = base + (group_rank < remainder ? 1 : 0);
    sphere.local_offset = group_rank * base + std::min(group_rank, remainder);

    sphere.global_vectors.resize(static_cast<std::size_t>(global_plane_waves) * number_of_vectors);
    sphere.local_vectors.resize(static_cast<std::size_t>(sphere.local_plane_waves) *
                                number_of_vectors);
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        for (int g_index = 0; g_index < global_plane_waves; ++g_index) {
            sphere.global_vectors[static_cast<std::size_t>(vector_index) * global_plane_waves +
                                  g_index] = half_sphere_entry(g_index, vector_index, seed);
        }
        for (int local_index = 0; local_index < sphere.local_plane_waves; ++local_index) {
            sphere.local_vectors[static_cast<std::size_t>(vector_index) * sphere.local_plane_waves +
                                 local_index] =
                half_sphere_entry(sphere.local_offset + local_index, vector_index, seed);
        }
    }
    return sphere;
}

} // namespace

TEST(SerialPlanewaveSubspaceManagerMPI, GammaSubspaceMatricesAgreeWithTheReplicatedSum) {
    execution::PlanewaveMPIExecutionPolicy execution_policy;
    execution_policy.configure_topology(1);
    if (execution_policy.world_size() < 2) {
        GTEST_SKIP() << "needs at least 2 ranks to put the origin on one of them";
    }
    ASSERT_EQ(execution_policy.group_size(), execution_policy.world_size());

    const int global_plane_waves = 37;
    const int number_of_vectors = 4;
    const int group_rank = execution_policy.group_rank_index();
    auto basis = build_half_sphere(
        execution_policy.group_size(), group_rank, global_plane_waves, number_of_vectors, 0.0);
    auto hamiltonian_basis = build_half_sphere(
        execution_policy.group_size(), group_rank, global_plane_waves, number_of_vectors, 2.1);

    SerialPlanewaveSubspaceManager distributed(execution_policy);
    distributed.set_gamma_origin_is_local(group_rank == 0);
    std::vector<Complex> subspace_hamiltonian(number_of_vectors * number_of_vectors);
    std::vector<Complex> subspace_overlap(number_of_vectors * number_of_vectors);
    distributed.update_subspace_matrices(0,
                                         number_of_vectors,
                                         number_of_vectors,
                                         number_of_vectors,
                                         basis.local_plane_waves,
                                         /*is_gamma_only=*/true,
                                         basis.local_vectors.data(),
                                         hamiltonian_basis.local_vectors.data(),
                                         basis.local_vectors.data(),
                                         subspace_hamiltonian.data(),
                                         subspace_overlap.data());

    execution::SerialExecutionPolicy serial_policy;
    SerialPlanewaveSubspaceManager replicated(serial_policy);
    std::vector<Complex> reference_hamiltonian(number_of_vectors * number_of_vectors);
    std::vector<Complex> reference_overlap(number_of_vectors * number_of_vectors);
    replicated.update_subspace_matrices(0,
                                        number_of_vectors,
                                        number_of_vectors,
                                        number_of_vectors,
                                        global_plane_waves,
                                        /*is_gamma_only=*/true,
                                        basis.global_vectors.data(),
                                        hamiltonian_basis.global_vectors.data(),
                                        basis.global_vectors.data(),
                                        reference_hamiltonian.data(),
                                        reference_overlap.data());

    for (int element = 0; element < number_of_vectors * number_of_vectors; ++element) {
        EXPECT_NEAR(
            subspace_hamiltonian[element].real(), reference_hamiltonian[element].real(), 1.0e-10)
            << "H element " << element;
        EXPECT_NEAR(subspace_overlap[element].real(), reference_overlap[element].real(), 1.0e-10)
            << "S element " << element;
    }
}

/**
 * The same origin rule inside orthogonalisation: the Cholesky overlaps, the
 * projection onto the existing basis and the final normalisation all carry
 * it, and the real-G=0 cleanup must touch index 0 on the origin's rank only
 * -- everywhere else that entry is an ordinary coefficient.
 */
TEST(SerialPlanewaveSubspaceManagerMPI, GammaOrthogonalisationMatchesTheReplicatedVectors) {
    execution::PlanewaveMPIExecutionPolicy execution_policy;
    execution_policy.configure_topology(1);
    if (execution_policy.world_size() < 2) {
        GTEST_SKIP() << "needs at least 2 ranks to put the origin on one of them";
    }

    const int global_plane_waves = 41;
    const int number_of_vectors = 5;
    const int group_rank = execution_policy.group_rank_index();
    auto basis = build_half_sphere(
        execution_policy.group_size(), group_rank, global_plane_waves, number_of_vectors, 0.5);

    SerialPlanewaveSubspaceManager distributed(execution_policy);
    distributed.set_gamma_origin_is_local(group_rank == 0);
    execution::SerialExecutionPolicy serial_policy;
    SerialPlanewaveSubspaceManager replicated(serial_policy);

    // First three as a batch against nothing, then two more against those three.
    for (auto [start, count] : {std::pair<int, int>{0, 3}, std::pair<int, int>{3, 2}}) {
        int accepted_distributed = distributed.orthogonalize_batch(start,
                                                                   count,
                                                                   basis.local_plane_waves,
                                                                   /*is_gamma_only=*/true,
                                                                   nullptr,
                                                                   basis.local_vectors.data(),
                                                                   basis.local_vectors.data());
        int accepted_replicated = replicated.orthogonalize_batch(start,
                                                                 count,
                                                                 global_plane_waves,
                                                                 /*is_gamma_only=*/true,
                                                                 nullptr,
                                                                 basis.global_vectors.data(),
                                                                 basis.global_vectors.data());
        ASSERT_EQ(accepted_distributed, accepted_replicated);
        ASSERT_EQ(accepted_distributed, count);
    }

    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        const Complex* reference = basis.global_column(vector_index) + basis.local_offset;
        const Complex* local = basis.local_column(vector_index);
        for (int local_index = 0; local_index < basis.local_plane_waves; ++local_index) {
            EXPECT_NEAR(local[local_index].real(), reference[local_index].real(), 1.0e-10)
                << "vector " << vector_index << " entry " << local_index;
            EXPECT_NEAR(local[local_index].imag(), reference[local_index].imag(), 1.0e-10)
                << "vector " << vector_index << " entry " << local_index;
        }
    }
}

// ─── Collapse with an identity overlap held in its own buffer ───────────────

/**
 * An identity overlap lets the CPU workspace alias S|V> to |V>, and then
 * rotating the basis rotates S|V> for free. The device workspace keeps S|V> in
 * a separate buffer even for the identity, and the collapse used to skip that
 * buffer on the identity flag alone, leaving it unrotated: every later
 * orthogonalisation and subspace update then read stale S|V>, and GPU runs that
 * collapsed often converged 1.77 Ry low on the two-atom silicon deck. Whatever
 * the flag says, a separate overlap buffer must come out of the collapse
 * holding S times the rotated basis -- here, the rotated basis itself.
 */
TEST(SerialPlanewaveSubspaceManager, CollapseRotatesASeparateOverlapBufferEvenForTheIdentity) {
    execution::SerialExecutionPolicy serial_policy;
    SerialPlanewaveSubspaceManager manager(serial_policy);

    const int plane_waves = 4;
    const int basis_size = 4;
    const int bands = 2;

    // Orthonormal basis: the unit vectors, so S_sub is the identity.
    std::vector<Complex> basis(static_cast<std::size_t>(plane_waves) * basis_size,
                               Complex(0.0, 0.0));
    for (int column = 0; column < basis_size; ++column) {
        basis[static_cast<std::size_t>(column) * plane_waves + column] = Complex(1.0, 0.0);
    }
    // A Hermitian H with distinct eigenvalues; H|V> is H itself for unit vectors.
    const double h[4][4] = {
        {4.0, 1.0, 0.5, 0.0}, {1.0, 3.0, 0.2, 0.3}, {0.5, 0.2, 1.0, 0.4}, {0.0, 0.3, 0.4, 2.0}};
    std::vector<Complex> hamiltonian_basis(basis.size());
    std::vector<Complex> subspace_hamiltonian(static_cast<std::size_t>(basis_size) * basis_size);
    for (int column = 0; column < basis_size; ++column) {
        for (int row = 0; row < plane_waves; ++row) {
            hamiltonian_basis[static_cast<std::size_t>(column) * plane_waves + row] =
                Complex(h[row][column], 0.0);
            subspace_hamiltonian[static_cast<std::size_t>(column) * basis_size + row] =
                Complex(h[row][column], 0.0);
        }
    }
    std::vector<Complex> subspace_overlap(subspace_hamiltonian.size(), Complex(0.0, 0.0));
    for (int index = 0; index < basis_size; ++index) {
        subspace_overlap[static_cast<std::size_t>(index) * basis_size + index] = Complex(1.0, 0.0);
    }
    // The overlap is the identity, but held in a buffer of its own.
    std::vector<Complex> overlap_basis = basis;
    std::vector<Complex> scratch(static_cast<std::size_t>(bands) * plane_waves);

    int collapsed = manager.collapse_subspace(basis_size,
                                              plane_waves,
                                              bands,
                                              basis_size,
                                              /*is_gamma_only=*/false,
                                              /*is_identity_overlap=*/true,
                                              nullptr,
                                              subspace_hamiltonian.data(),
                                              subspace_overlap.data(),
                                              basis.data(),
                                              hamiltonian_basis.data(),
                                              overlap_basis.data(),
                                              scratch.data());
    ASSERT_EQ(collapsed, bands);

    for (int column = 0; column < bands; ++column) {
        for (int row = 0; row < plane_waves; ++row) {
            std::size_t index = static_cast<std::size_t>(column) * plane_waves + row;
            EXPECT_NEAR(overlap_basis[index].real(), basis[index].real(), 1.0e-12)
                << "S|V> column " << column << " row " << row << " was not rotated";
            EXPECT_NEAR(overlap_basis[index].imag(), basis[index].imag(), 1.0e-12);
        }
    }
    // And the collapse really rotated: the first column is no longer a unit vector.
    EXPECT_GT(std::abs(basis[1]) + std::abs(basis[2]) + std::abs(basis[3]), 1.0e-6);
}

} // namespace q3
