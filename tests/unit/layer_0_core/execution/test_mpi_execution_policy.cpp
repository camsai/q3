/**
 * MPI unit tests for the execution-policy topology and the subspace
 * diagonalizer factory.
 *
 * Run via: mpirun -np N ./build_mpi/q3_tests_mpi   (N = 1, 2, or 4)
 *
 * All tests are world-size-agnostic: they assert invariants that hold for any
 * rank count, and guard rank-count-specific assertions accordingly. MPI is
 * initialized exactly once by a global test environment, so the policies
 * under test never own the MPI lifecycle (their documented test-harness mode)
 * and can be constructed/destroyed freely across test cases.
 */

#include "layer_0_core/execution/mpi.hpp"
#include "layer_0_core/execution/mpi_datatype.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/mpi.hpp"

#include <cstring>

#include <gtest/gtest.h>
#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace q3 {
namespace {

/**
 * Initializes MPI before any test constructs an execution policy and
 * finalizes it after the last test, so policies never own the lifecycle.
 */
class MPITestEnvironment : public ::testing::Environment {
  public:
    void SetUp() override {
        int already_initialized = 0;
        MPI_Initialized(&already_initialized);
        if (!already_initialized) {
            MPI_Init(nullptr, nullptr);
        }
    }
    void TearDown() override {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
};

const ::testing::Environment* const mpi_environment =
    ::testing::AddGlobalTestEnvironment(new MPITestEnvironment);

// ── Layer-0 process group topology ────────────────────────────────────────────

TEST(MPIExecutionPolicyTest, WorldRankAndSizeAreConsistent) {
    execution::MPIExecutionPolicy policy;
    EXPECT_GE(policy.world_size(), 1);
    EXPECT_GE(policy.world_rank_index(), 0);
    EXPECT_LT(policy.world_rank_index(), policy.world_size());
    EXPECT_EQ(policy.number_of_processes(), policy.world_size());
}

TEST(MPIExecutionPolicyTest, DefaultGroupSpansTheWorld) {
    execution::MPIExecutionPolicy policy;
    EXPECT_EQ(policy.group_size(), policy.world_size());
    EXPECT_EQ(policy.group_rank_index(), policy.world_rank_index());
}

TEST(MPIExecutionPolicyTest, ConfigureGroupTopologySingleGroup) {
    execution::MPIExecutionPolicy policy;
    policy.configure_group_topology(1);
    EXPECT_EQ(policy.group_size(), policy.world_size());
    EXPECT_EQ(policy.group_index(), 0);
}

TEST(MPIExecutionPolicyTest, ConfigureGroupTopologyOnePerRank) {
    execution::MPIExecutionPolicy policy;
    policy.configure_group_topology(policy.world_size());
    EXPECT_EQ(policy.group_size(), 1);
    EXPECT_EQ(policy.group_index(), policy.world_rank_index());
}

TEST(MPIExecutionPolicyTest, ConfigureGroupTopologyRejectsOutOfRange) {
    execution::MPIExecutionPolicy policy;
    EXPECT_THROW(policy.configure_group_topology(0), std::invalid_argument);
    EXPECT_THROW(policy.configure_group_topology(policy.world_size() + 1), std::invalid_argument);
}

TEST(MPIExecutionPolicyTest, GroupDotProductMatchesSerialForReplicatedData) {
    execution::MPIExecutionPolicy policy;
    // One group per rank: the group collective must degenerate to the local dot.
    policy.configure_group_topology(policy.world_size());
    const Complex vector[2] = {Complex(1.0, 0.0), Complex(0.0, 1.0)};
    // conj([1, i]) . [1, i] = 1 + 1 = 2
    const Complex dot = policy.dot_product(2, vector, vector);
    EXPECT_NEAR(dot.real(), 2.0, 1e-14);
    EXPECT_NEAR(dot.imag(), 0.0, 1e-14);
}

// ── The datatype complex buffers travel as ────────────────────────────────────

/**
 * These gate camsai/q3#11: Q3 passed MPI_DOUBLE_COMPLEX -- Fortran's DOUBLE
 * COMPLEX -- for std::complex<double>, and an MPI built without Fortran leaves
 * that datatype uninitialized, so the first complex collective aborted with
 * MPI_DATATYPE_NULL. Only a run with two or more ranks in one group reaches
 * these paths at all, which is why nothing before this noticed.
 *
 * On an MPI that does have Fortran these pass either way; the configuration
 * that distinguishes them is the container image, whose MPICH is built
 * --disable-fortran and which CI now runs with ranks sharing a pool.
 *
 * The sum below additionally gates camsai/q3#17, whose failure no CI runner
 * can reproduce: a host MPICH swapped in by `podman-hpc --mpi` refused
 * MPI_SUM on MPI_CXX_DOUBLE_COMPLEX, which every MPI available here accepts.
 * What is testable is the property the fix relies on -- that reducing N
 * complex numbers as 2N doubles is the same arithmetic -- so that is what is
 * asserted, bitwise rather than approximately.
 */
TEST(MPIComplexDatatypeTest, IsInitializedInThisMpiBuild) {
    ASSERT_NE(execution::complex_datatype(), MPI_DATATYPE_NULL);
    int datatype_size = 0;
    ASSERT_EQ(MPI_Type_size(execution::complex_datatype(), &datatype_size), MPI_SUCCESS);
    EXPECT_EQ(datatype_size, static_cast<int>(sizeof(Complex)));
}

TEST(MPIComplexDatatypeTest, SumsComplexDataOverTheGroup) {
    execution::MPIExecutionPolicy policy;
    // One group spanning the world, so every rank contributes to one sum.
    policy.configure_group_topology(1);
    std::vector<Complex> data(3, Complex(1.0, -2.0));
    policy.reduce_within_group(data.data(), static_cast<int>(data.size()));
    const double ranks = policy.world_size();
    for (const Complex& value : data) {
        EXPECT_NEAR(value.real(), ranks, 1e-14);
        EXPECT_NEAR(value.imag(), -2.0 * ranks, 1e-14);
    }
}

TEST(MPIComplexDatatypeTest, SumAsDoublePairsMatchesTheComplexDatatypeBitwise) {
    /**
     * Values chosen not to be exactly representable, so any difference in how
     * the two reductions associate their additions would show up.
     */
    const int count = 257;
    std::vector<Complex> as_double_pairs(count);
    std::vector<Complex> as_complex_datatype(count);
    for (int i = 0; i < count; ++i) {
        const Complex value(0.1 * (i + 1), -0.3 * (i + 1));
        as_double_pairs[i] = value;
        as_complex_datatype[i] = value;
    }

    ASSERT_EQ(execution::sum_complex_in_place(as_double_pairs.data(), count, MPI_COMM_WORLD),
              MPI_SUCCESS);
    ASSERT_EQ(MPI_Allreduce(MPI_IN_PLACE,
                            as_complex_datatype.data(),
                            count,
                            execution::complex_datatype(),
                            MPI_SUM,
                            MPI_COMM_WORLD),
              MPI_SUCCESS);

    EXPECT_EQ(
        std::memcmp(as_double_pairs.data(), as_complex_datatype.data(), count * sizeof(Complex)),
        0);
}

TEST(MPIComplexDatatypeTest, SumWritesToASeparateDestination) {
    const int count = 8;
    const std::vector<Complex> source(count, Complex(1.0, -2.0));
    std::vector<Complex> destination(count, Complex(99.0, 99.0));

    ASSERT_EQ(execution::sum_complex(source.data(), destination.data(), count, MPI_COMM_WORLD),
              MPI_SUCCESS);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    for (const Complex& value : destination) {
        EXPECT_NEAR(value.real(), world_size, 1e-14);
        EXPECT_NEAR(value.imag(), -2.0 * world_size, 1e-14);
    }
}

TEST(MPIComplexDatatypeTest, BroadcastsComplexDataOverTheGroup) {
    execution::MPIExecutionPolicy policy;
    policy.configure_group_topology(1);
    // Only the root holds the value; every other rank must end up with it.
    Complex value = (policy.world_rank_index() == 0) ? Complex(3.0, 4.0) : Complex(0.0, 0.0);
    policy.broadcast_from_group_root(&value, 1);
    EXPECT_NEAR(value.real(), 3.0, 1e-14);
    EXPECT_NEAR(value.imag(), 4.0, 1e-14);
}

TEST(MPIComplexDatatypeTest, GathersOneComplexBlockPerBandRank) {
    execution::MPIExecutionPolicy policy;
    policy.configure_group_topology(1);
    policy.enable_band_distribution();

    constexpr int block_size = 2;
    const int number_of_ranks = policy.world_size();
    std::vector<int> counts(number_of_ranks, block_size);
    std::vector<int> displacements(number_of_ranks);
    for (int rank_index = 0; rank_index < number_of_ranks; rank_index++) {
        displacements[rank_index] = rank_index * block_size;
    }

    // Each rank fills only its own block, as the MPI_IN_PLACE contract requires.
    std::vector<Complex> data(static_cast<size_t>(block_size) * number_of_ranks, Complex(0.0, 0.0));
    for (int element_index = 0; element_index < block_size; element_index++) {
        data[displacements[policy.world_rank_index()] + element_index] =
            Complex(policy.world_rank_index(), element_index);
    }

    policy.allgather_across_bands(data.data(), counts.data(), displacements.data());

    for (int rank_index = 0; rank_index < number_of_ranks; rank_index++) {
        for (int element_index = 0; element_index < block_size; element_index++) {
            const Complex& value = data[rank_index * block_size + element_index];
            EXPECT_NEAR(value.real(), rank_index, 1e-14);
            EXPECT_NEAR(value.imag(), element_index, 1e-14);
        }
    }
}

// ── Planewave pool topology ───────────────────────────────────────────────────

TEST(PlanewaveMPIExecutionPolicyTest, OnePoolPerRankTopology) {
    execution::PlanewaveMPIExecutionPolicy policy;
    policy.configure_topology(policy.world_size());
    EXPECT_EQ(policy.number_of_pools(), policy.world_size());
    EXPECT_EQ(policy.band_size(), 1);
    EXPECT_EQ(policy.pool_index(), policy.world_rank_index());
    EXPECT_EQ(policy.one_process_owns_complete_state(), policy.world_size() == 1);
    EXPECT_TRUE(policy.each_process_owns_whole_kpoints());
}

TEST(PlanewaveMPIExecutionPolicyTest, SinglePoolTopologyIsBandParallel) {
    execution::PlanewaveMPIExecutionPolicy policy;
    policy.configure_topology(1);
    EXPECT_EQ(policy.number_of_pools(), 1);
    EXPECT_EQ(policy.band_size(), policy.world_size());
    EXPECT_EQ(policy.pool_index(), 0);
    // Pool count times ranks-per-pool must recover the world.
    EXPECT_EQ(policy.number_of_pools() * policy.band_size(), policy.world_size());
    EXPECT_EQ(policy.each_process_owns_whole_kpoints(), policy.world_size() == 1);
}

/**
 * The checkpoint names each pool's wfc files by global_kpoint_index(), so it
 * must name the k-point local_kpoint_grid() actually dealt this pool, for
 * every pool and a k-point count the pools do not divide.
 */
TEST(PlanewaveMPIExecutionPolicyTest, GlobalKpointIndexNamesTheKpointThePoolWasDealt) {
    execution::PlanewaveMPIExecutionPolicy policy;
    policy.configure_topology(policy.world_size());
    const int number_of_kpoints = 2 * policy.world_size() + 1;
    KPointsGrid global_kpoint_grid;
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
        KPoint kpoint;
        kpoint.coordinates.x = static_cast<Real>(kpoint_index);
        global_kpoint_grid.add_kpoint(kpoint);
    }

    KPointsGrid local_kpoint_grid = policy.local_kpoint_grid(global_kpoint_grid);
    ASSERT_GT(local_kpoint_grid.number_of_kpoints(), 0);
    for (int local_kpoint_index = 0; local_kpoint_index < local_kpoint_grid.number_of_kpoints();
         ++local_kpoint_index) {
        EXPECT_EQ(static_cast<Real>(policy.global_kpoint_index(local_kpoint_index)),
                  local_kpoint_grid[local_kpoint_index].coordinates.x);
    }
}

TEST(PlanewaveMPIExecutionPolicyTest, ZeroSelectsTheDocumentedSinglePoolDefault) {
    execution::PlanewaveMPIExecutionPolicy policy;
    policy.configure_topology(0);
    EXPECT_EQ(policy.number_of_pools(), 1);
    EXPECT_EQ(policy.band_size(), policy.world_size());
    EXPECT_EQ(policy.pool_index(), 0);
}

TEST(PlanewaveMPIExecutionPolicyTest, GlobalElementCountSumsBandGroupSlabs) {
    execution::PlanewaveMPIExecutionPolicy policy;
    policy.configure_topology(1);
    EXPECT_EQ(policy.global_number_of_elements(1), policy.world_size());

    policy.configure_topology(policy.world_size());
    EXPECT_EQ(policy.global_number_of_elements(1), 1);
}

TEST(PlanewaveMPIExecutionPolicyTest, AssemblesPoolLocalKpointStatesInGlobalOrder) {
    execution::PlanewaveMPIExecutionPolicy policy;
    if (policy.world_size() < 2 || policy.world_size() % 2 != 0) {
        GTEST_SKIP() << "test requires an even world with at least two ranks";
    }
    policy.configure_topology(2);
    const int configured_pool_index = policy.pool_index();
    policy.disable_band_distribution();
    EXPECT_EQ(policy.pool_index(), configured_pool_index);
    /**
     * A group of one in name only: the pool's ranks still share its FFT grid,
     * so they do not hold their k-points whole.
     */
    EXPECT_EQ(policy.band_size(), 1);
    EXPECT_EQ(policy.each_process_owns_whole_kpoints(), policy.world_size() == 2);

    constexpr int local_number_of_physical_kpoints = 2;
    constexpr int number_of_spin_components = 2;
    constexpr int number_of_values_per_state = 1;
    std::vector<Real> local_data(local_number_of_physical_kpoints * number_of_spin_components);
    for (int spin_index = 0; spin_index < number_of_spin_components; ++spin_index) {
        for (int local_kpoint_index = 0; local_kpoint_index < local_number_of_physical_kpoints;
             ++local_kpoint_index) {
            int global_kpoint_index = policy.pool_index() + 2 * local_kpoint_index;
            local_data[spin_index * local_number_of_physical_kpoints + local_kpoint_index] =
                100.0 * spin_index + global_kpoint_index;
        }
    }

    auto global_data = policy.assemble_global_kpoint_state_data(local_data,
                                                                local_number_of_physical_kpoints,
                                                                number_of_spin_components,
                                                                number_of_values_per_state);
    const std::vector<Real> expected = {0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0};
    EXPECT_EQ(global_data, expected);
}

TEST(PlanewaveMPIExecutionPolicyTest, RejectsNonDivisiblePoolCount) {
    execution::PlanewaveMPIExecutionPolicy policy;
    /**
     * A non-divisor of world_size only exists for world_size >= 3
     * (e.g. 2 pools over 3 ranks). Smaller worlds skip this assertion.
     */
    if (policy.world_size() < 3) {
        GTEST_SKIP() << "no non-divisor pool count exists for world_size < 3";
    }
    int non_divisor = 2;
    while (non_divisor < policy.world_size() && policy.world_size() % non_divisor == 0) {
        non_divisor++;
    }
    if (policy.world_size() % non_divisor == 0) {
        GTEST_SKIP() << "world size has no non-divisor in range";
    }
    EXPECT_THROW(policy.configure_topology(non_divisor), std::invalid_argument);
}

// ── Subspace diagonalizer selection and collectivity ──────────────────────────

TEST(SubspaceDiagonalizerFactoryTest, SolveSucceedsUnderCurrentTopology) {
    execution::PlanewaveMPIExecutionPolicy policy;
    /**
     * Default topology: one pool spanning the world (band-parallel when np > 1),
     * which selects the MPI (ScaLAPACK-or-fallback) diagonalizer for np > 1.
     */
    auto diagonalizer = SubspaceDiagonalizerFactory::create(policy);
    ASSERT_NE(diagonalizer, nullptr);

    if (policy.group_size() == 1) {
        EXPECT_FALSE(diagonalizer->is_collective())
            << "serial subspace diagonalizer must not require collective entry";
    }

    /**
     * Regression shape for the rank-guarded-collective deadlock: every rank
     * enters solve() when the solver is collective; only the root must when it
     * is not. Solving A = [2,1;1,2], S = I must give eigenvalues {1, 3} on
     * whichever ranks solve.
     */
    Complex hamiltonian[4] = {
        Complex(2.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0)};
    Complex overlap[4] = {
        Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};
    Real eigenvalues[2] = {0.0, 0.0};
    Complex eigenvectors[4];

    int info = 1;
    if (diagonalizer->is_collective() || policy.group_rank_index() == 0) {
        info = diagonalizer->solve(2, hamiltonian, overlap, eigenvalues, eigenvectors);
        EXPECT_EQ(info, 0);
        EXPECT_NEAR(eigenvalues[0], 1.0, 1e-12);
        EXPECT_NEAR(eigenvalues[1], 3.0, 1e-12);
    }
}

} // namespace
} // namespace q3
