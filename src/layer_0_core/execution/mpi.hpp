#pragma once

/**
 * MPIExecutionPolicy — MPI implementation of ExecutionPolicy.
 *
 * Owns the MPI process lifecycle (MPI_Init / MPI_Finalize) and the
 * communicator topology (world → process groups). Provides group-collective
 * vector operations (dot_product, norm) via MPI_Allreduce over the group
 * communicator.
 *
 * This and mpi_datatype.hpp are the only files in layer_0_core/execution/
 * that include <mpi.h>.
 *
 * This layer is representation-agnostic: it knows nothing about bands,
 * k-points, or any physical basis. What a process group *means* (e.g. a band
 * group of processes diagonalizing the same k-point) is assigned by higher
 * layers — see layer_2_quantum_system/execution/init.hpp.
 *
 * Topology:
 *   World communicator  — all processes
 *   Group communicator  — the subset of processes this rank shares
 *                         collectives with (defaults to world until
 *                         configure_group_topology() is called)
 */

#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/execution/mpi_datatype.hpp"
#include "layer_0_core/execution/serial.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#ifdef Q3_WITH_FFTW_MPI
#include <fftw3-mpi.h>
#endif
#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace q3 {
namespace execution {

class MPIExecutionPolicy : public virtual ExecutionPolicy {
  public:
    /**
     * Initialize MPI (if not already initialized) and set up communicators.
     *
     * Uses MPI_Init(nullptr, nullptr) — valid since MPI-2. If MPI is already
     * initialized (e.g., by a test harness), initialization is skipped and
     * MPI_Finalize will not be called on destruction.
     */
    MPIExecutionPolicy() {
        int already_initialized = 0;
        MPI_Initialized(&already_initialized);
        if (!already_initialized) {
            MPI_Init(nullptr, nullptr);
#ifdef Q3_USE_FFTW_THREADS
            /**
             * FFTW requires fftw_init_threads before fftw_mpi_init when using
             * threaded FFT plans. Only initialize when threads are requested
             * to avoid the ~4% overhead the threading hook adds to all plans.
             */
            {
                const char* omp_env = std::getenv("OMP_NUM_THREADS");
                if (omp_env && std::atoi(omp_env) > 1) {
                    fftw_init_threads();
                }
            }
#endif
#ifdef Q3_WITH_FFTW_MPI
            fftw_mpi_init();
#endif
            owns_mpi_lifecycle_ = true;
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_index_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        /**
         * Default: group communicator is the full world until configure_group_topology()
         * is called. Future multi-rank group parallelism will split this into process groups.
         */
        group_communicator_ = MPI_COMM_WORLD;
        group_rank_index_ = world_rank_index_;
        group_size_ = world_size_;
        group_index_ = 0;
    }

    /**
     * Configure the process group topology by splitting the world communicator.
     *
     * Processes are divided into 'number_of_groups' equal groups of
     * 'world_size / number_of_groups' consecutive ranks. Collective vector
     * operations subsequently reduce over the group communicator.
     *
     * @param number_of_groups  Number of independent process groups.
     */
    void configure_group_topology(int number_of_groups) {
        if (number_of_groups < 1 || number_of_groups > world_size_) {
            throw std::invalid_argument("MPIExecutionPolicy: number_of_groups out of range.");
        }

        // Free old group communicator if it was a split one
        if (group_communicator_ != MPI_COMM_WORLD && group_communicator_ != MPI_COMM_NULL) {
            MPI_Comm_free(&group_communicator_);
        }

        const int group_size = world_size_ / number_of_groups;
        group_index_ = world_rank_index_ / group_size;

        /**
         * Split world into groups based on their index.
         * All ranks in the same group share the same color.
         */
        MPI_Comm_split(MPI_COMM_WORLD, group_index_, world_rank_index_, &group_communicator_);

        MPI_Comm_rank(group_communicator_, &group_rank_index_);
        MPI_Comm_size(group_communicator_, &group_size_);
    }

    /**
     * Disable data distribution within the process group.
     *
     * Resets group_size to 1 so that all group-collective reductions
     * (dot_product, reduce_within_group, and the pool-level reductions built
     * on them) become no-ops. Used by higher layers when their data
     * distribution degenerates and they fall back to replicated data — see
     * PlanewaveExecutionPolicy::disable_band_distribution().
     *
     * The group communicator and configured group index remain intact. Only
     * the active reduction size changes, so higher layers retain the correct
     * k-point-pool identity while working with replicated coefficient data.
     *
     * Note what this leaves behind: group_size() becomes 1 while
     * group_rank_index() keeps its pool-local value, so on ranks other than the
     * pool root the two disagree about whether "this is a group of one". Code
     * choosing which rank performs a redundant computation must branch on
     * group_size() <= 1, not on group_rank_index() == 0.
     */
    void disable_group_distribution() {
        group_size_ = 1;
    }

    /**
     * Switch this pool from distributing reciprocal space to distributing bands.
     *
     * Records how many ranks share the pool, then disables group distribution so
     * that every rank keeps whole coefficient vectors and each reduce_within_group
     * becomes a no-op. The group communicator is deliberately left intact: it is
     * what the band allgather and the density reduction run over.
     */
    void enable_band_distribution() {
        band_distribution_size_ = group_size_;
        band_distribution_rank_index_ = group_rank_index_;
        disable_group_distribution();
    }

    /**
     * Free split communicators and finalize MPI (if we initialized it).
     *
     * Guards every MPI call with MPI_Finalized() so that this destructor is
     * safe to invoke even if MPI has already been finalized (e.g., when the
     * policy object is owned by a unique_ptr that outlives global teardown).
     *
     * The barrier before MPI_Finalize() ensures that all ranks have finished
     * their pending work before any rank finalizes. Without it, on macOS with
     * --oversubscribe, rank 0 can reach MPI_Finalize() while non-root ranks
     * are still processing trailing work (e.g., logging), causing SIGSEGV in
     * OpenMPI's shared-memory cleanup.
     *
     * Teardown is skipped entirely while an exception is in flight, because
     * every step of it is collective -- MPI_Comm_free and MPI_Finalize no less
     * than the barrier -- and each one only completes once every rank calls it.
     * A rank unwinding alone has no reason to believe its peers are unwinding
     * too: they are still inside a physics collective, and will not arrive.
     * Running teardown anyway is what turns a rank-local error -- an
     * unreadable file, a failed allocation, anything one rank sees and the
     * others do not -- into a job that hangs instead of failing, with the
     * message still unsent because the handler that prints it is never
     * reached. Doing nothing here leaves MPI live for that handler, which
     * reports the error and calls MPI_Abort to take the blocked peers down
     * with it (see main_mpi.cpp). Communicators leak, which costs nothing in a
     * process that is about to die and cannot be avoided regardless: there is
     * no non-collective way to release them.
     *
     * A collective failure -- every rank throwing the same thing, such as the
     * shared-pool refusal in the Kohn-Sham factory -- also lands here, on
     * every rank, and takes the same path. That path is safe for it too: each
     * rank reports before it aborts.
     */
    ~MPIExecutionPolicy() override {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized) {
            return;
        }
        if (std::uncaught_exceptions() > 0) {
            return;
        }
        if (group_communicator_ != MPI_COMM_WORLD && group_communicator_ != MPI_COMM_NULL) {
            MPI_Comm_free(&group_communicator_);
        }
        if (owns_mpi_lifecycle_) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
        }
    }

    // Non-copyable — owns MPI resources
    MPIExecutionPolicy(const MPIExecutionPolicy&) = delete;
    MPIExecutionPolicy& operator=(const MPIExecutionPolicy&) = delete;

    // ── ExecutionPolicy implementation ────────────────────────────────────────

    /**
     * Dot product of two (possibly distributed) coefficient vectors.
     *
     * With data distribution, each rank holds a slice of the
     * coefficients. The local dot product is partial and must be summed
     * across group ranks via MPI_Allreduce.
     */
    ComplexValue dot_product(int number_of_elements,
                             const ComplexValue* vector_x,
                             const ComplexValue* vector_y) const override {
        ComplexValue local_result =
            serial_policy_.dot_product(number_of_elements, vector_x, vector_y);
        if (group_size_ > 1) {
            sum_complex_in_place(&local_result, 1, group_communicator_);
        }
        return local_result;
    }

    /**
     * Scaled vector addition: purely local, no communication.
     */
    void add_scaled_vector(int number_of_elements,
                           ComplexValue scaling_factor,
                           const ComplexValue* vector_x,
                           ComplexValue* vector_y) const override {
        serial_policy_.add_scaled_vector(number_of_elements, scaling_factor, vector_x, vector_y);
    }

    /**
     * Norm of a (possibly distributed) coefficient vector.
     *
     * Delegates to dot_product which handles the group reduction.
     */
    Real norm(int number_of_elements, const ComplexValue* vector_x) const override {
        ComplexValue dot = dot_product(number_of_elements, vector_x, vector_x);
        return std::sqrt(std::real(dot));
    }

    bool is_root() const override {
        return world_rank_index_ == 0;
    }

    int group_rank_index() const override {
        return group_rank_index_;
    }

    int group_size() const override {
        return group_size_;
    }

    int number_of_processes() const override {
        return world_size_;
    }

    int global_number_of_elements(int local_number_of_elements) const override {
        int total_number_of_elements = local_number_of_elements;
        if (group_size_ > 1) {
            MPI_Allreduce(
                MPI_IN_PLACE, &total_number_of_elements, 1, MPI_INT, MPI_SUM, group_communicator_);
        }
        return total_number_of_elements;
    }

    /**
     * Allreduce complex data within the process group.
     *
     * With data distribution, inner products (e.g. subspace matrices,
     * orthogonalization overlaps) contract over local elements and
     * need to be summed across group ranks.
     */
    void reduce_within_group(ComplexValue* data, int number_of_elements) const override {
        if (group_size_ <= 1)
            return;
        sum_complex_in_place(data, number_of_elements, group_communicator_);
    }

    /**
     * Allreduce real data within the process group.
     */
    void reduce_within_group(Real* data, int number_of_elements) const override {
        if (group_size_ <= 1)
            return;
        MPI_Allreduce(
            MPI_IN_PLACE, data, number_of_elements, MPI_DOUBLE, MPI_SUM, group_communicator_);
    }

    /**
     * Reduce group-distributed results across the process group.
     *
     * Used specifically by the task-group H|ψ⟩ path where different ranks
     * compute disjoint contributions to the same array and need to sum them.
     * Unlike reduce_within_group (which is a no-op for replicated data),
     * this always performs the Allreduce.
     */
    void reduce_group_distributed_result(ComplexValue* data,
                                         int number_of_elements) const override {
        if (group_size_ <= 1)
            return;
        sum_complex_in_place(data, number_of_elements, group_communicator_);
    }

    void allgather_within_group(const Real* local_data,
                                int local_count,
                                Real* full_data,
                                int /*total_count*/) const override {
        if (group_size_ <= 1) {
            std::memcpy(full_data, local_data, local_count * sizeof(Real));
            return;
        }
        // Gather per-rank counts to compute displacements.
        std::vector<int> receive_counts(group_size_);
        MPI_Allgather(
            &local_count, 1, MPI_INT, receive_counts.data(), 1, MPI_INT, group_communicator_);
        std::vector<int> displacements(group_size_, 0);
        for (int rank = 1; rank < group_size_; ++rank) {
            displacements[rank] = displacements[rank - 1] + receive_counts[rank - 1];
        }
        MPI_Allgatherv(local_data,
                       local_count,
                       MPI_DOUBLE,
                       full_data,
                       receive_counts.data(),
                       displacements.data(),
                       MPI_DOUBLE,
                       group_communicator_);
    }

    // ── Process topology accessors ────────────────────────────────────────────

    int world_rank_index() const {
        return world_rank_index_;
    }
    int world_size() const {
        return world_size_;
    }
    int group_index() const {
        return group_index_;
    }

    MPI_Comm get_world_communicator() const {
        return MPI_COMM_WORLD;
    }
    MPI_Comm get_group_communicator() const {
        return group_communicator_;
    }

    void broadcast_from_group_root(ComplexValue* data, int number_of_elements) const override {
        if (group_size_ <= 1) {
            return;
        }
        MPI_Bcast(data, number_of_elements, complex_datatype(), 0, group_communicator_);
    }

    void broadcast_from_group_root(Real* data, int number_of_elements) const override {
        if (group_size_ <= 1) {
            return;
        }
        MPI_Bcast(data, number_of_elements, MPI_DOUBLE, 0, group_communicator_);
    }

    int band_distribution_size() const override {
        return band_distribution_size_;
    }

    int band_distribution_rank_index() const override {
        return band_distribution_rank_index_;
    }

    void allgather_across_bands(ComplexValue* data,
                                const int* counts,
                                const int* displacements) const override {
        if (band_distribution_size_ <= 1)
            return;
        /**
         * Counts arrive in complex elements because that is the unit bands are
         * partitioned in; the complex datatype makes that the wire unit too, so
         * there is no doubling here. MPI_IN_PLACE says the local block is
         * already at its own displacement, which is the contract this method
         * documents.
         *
         * The send arguments are ignored under MPI_IN_PLACE, but they remain
         * arguments an implementation may check, so they name the same datatype
         * rather than a null one.
         */
        MPI_Allgatherv(MPI_IN_PLACE,
                       0,
                       complex_datatype(),
                       data,
                       counts,
                       displacements,
                       complex_datatype(),
                       group_communicator_);
    }

    void broadcast_across_bands(ComplexValue* data, int number_of_elements) const override {
        if (band_distribution_size_ <= 1 || number_of_elements <= 0)
            return;
        MPI_Bcast(data, number_of_elements, complex_datatype(), 0, group_communicator_);
    }

    void broadcast_across_bands(int* data, int number_of_elements) const override {
        if (band_distribution_size_ <= 1 || number_of_elements <= 0)
            return;
        MPI_Bcast(data, number_of_elements, MPI_INT, 0, group_communicator_);
    }

    void broadcast_across_bands(Real* data, int number_of_elements) const override {
        if (band_distribution_size_ <= 1 || number_of_elements <= 0)
            return;
        MPI_Bcast(data, number_of_elements, MPI_DOUBLE, 0, group_communicator_);
    }

    void reduce_across_bands(Real* data, int number_of_elements) const override {
        if (band_distribution_size_ <= 1 || number_of_elements <= 0)
            return;
        MPI_Allreduce(
            MPI_IN_PLACE, data, number_of_elements, MPI_DOUBLE, MPI_SUM, group_communicator_);
    }

  protected:
    int band_distribution_size_ = 1;
    int band_distribution_rank_index_ = 0;
    MPI_Comm group_communicator_ = MPI_COMM_NULL;
    int group_rank_index_ = 0;
    int group_size_ = 1;
    int group_index_ = 0;

  private:
    SerialExecutionPolicy serial_policy_;
    int world_rank_index_ = 0;
    int world_size_ = 1;
    bool owns_mpi_lifecycle_ = false;
};

} // namespace execution
} // namespace q3
