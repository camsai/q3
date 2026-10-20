#pragma once

/**
 * MPIFFTGrid exists only when Q3 was linked against FFTW-MPI. Without it the
 * MPI execution policy falls back to SerialFFTGrid, which supports k-point
 * pools but not band groups -- see create_fft_grid() in
 * layer_4_computation/iterator/kohn_sham/planewave/execution/mpi.hpp.
 */
#ifdef Q3_WITH_FFTW_MPI

#include <cstdlib>
#include <fftw3-mpi.h>
#include <thread>
#include <vector>

#include "layer_0_core/execution/mpi_datatype.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"

namespace q3 {

/**
 * MPIFFTGrid — distributed 3D FFT using FFTW MPI.
 *
 * Splits the 3D grid across MPI ranks using slab decomposition
 * (first dimension distributed). Each rank owns a contiguous range
 * of planes [local_offset, local_offset + local_n1).
 */
class MPIFFTGrid : public FFTGrid {
  public:
    explicit MPIFFTGrid(MPI_Comm communicator) : communicator_(communicator) {}

    ~MPIFFTGrid() override {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized) {
            return;
        }
        if (plan_forward_)
            fftw_destroy_plan(plan_forward_);
        if (plan_inverse_)
            fftw_destroy_plan(plan_inverse_);
        if (plan_forward_batched_)
            fftw_destroy_plan(plan_forward_batched_);
        if (plan_inverse_batched_)
            fftw_destroy_plan(plan_inverse_batched_);
    }

    // Non-copyable
    MPIFFTGrid(const MPIFFTGrid&) = delete;
    MPIFFTGrid& operator=(const MPIFFTGrid&) = delete;
    MPIFFTGrid(MPIFFTGrid&&) = default;
    MPIFFTGrid& operator=(MPIFFTGrid&&) = default;

    void init(int dimension_1,
              int dimension_2,
              int dimension_3,
              bool use_measured_planning = false) override {
        grid_dimension_1_ = dimension_1;
        grid_dimension_2_ = dimension_2;
        grid_dimension_3_ = dimension_3;
        total_grid_points_ = dimension_1 * dimension_2 * dimension_3;

        ptrdiff_t local_n1, local_offset;
        ptrdiff_t alloc_local = fftw_mpi_local_size_3d(
            dimension_1, dimension_2, dimension_3, communicator_, &local_n1, &local_offset);

        local_dimension_1_ = static_cast<int>(local_n1);
        local_offset_ = static_cast<int>(local_offset);
        local_grid_points_ = local_dimension_1_ * dimension_2 * dimension_3;

        data_.resize(static_cast<std::size_t>(alloc_local));

        unsigned int planning_flags = use_measured_planning ? FFTW_MEASURE : FFTW_ESTIMATE;

        plan_forward_ = fftw_mpi_plan_dft_3d(dimension_1,
                                             dimension_2,
                                             dimension_3,
                                             reinterpret_cast<fftw_complex*>(data_.data()),
                                             reinterpret_cast<fftw_complex*>(data_.data()),
                                             communicator_,
                                             FFTW_FORWARD,
                                             planning_flags);

        plan_inverse_ = fftw_mpi_plan_dft_3d(dimension_1,
                                             dimension_2,
                                             dimension_3,
                                             reinterpret_cast<fftw_complex*>(data_.data()),
                                             reinterpret_cast<fftw_complex*>(data_.data()),
                                             communicator_,
                                             FFTW_BACKWARD,
                                             planning_flags);
    }

    [[nodiscard]] int grid_dimension_1() const override {
        return grid_dimension_1_;
    }
    [[nodiscard]] int grid_dimension_2() const override {
        return grid_dimension_2_;
    }
    [[nodiscard]] int grid_dimension_3() const override {
        return grid_dimension_3_;
    }

    [[nodiscard]] int total_grid_points() const override {
        return total_grid_points_;
    }

    [[nodiscard]] int local_grid_points() const override {
        return local_grid_points_;
    }

    /// Slabs split a plane from its mirror, which is what breaks gamma-only.
    [[nodiscard]] bool co_locates_mirrors() const override {
        return false;
    }

    [[nodiscard]] int local_offset() const override {
        return local_offset_;
    }

    void reduce_g_space(Complex* data, int number_of_elements) const override {
        execution::sum_complex_in_place(data, number_of_elements, communicator_);
    }

    Complex* input_buffer() const override {
        return data_.data();
    }

    Complex* output_buffer() const override {
        // In-place plan: input and output share the same buffer
        return data_.data();
    }

    void execute_inverse_raw_on_buffers() const override {
        ProfilingCounters::instance().fft_call_count++;
        fftw_execute(plan_inverse_);
    }

    void execute_forward_on_buffers() const override {
        ProfilingCounters::instance().fft_call_count++;
        fftw_execute(plan_forward_);
    }

    void prepare_forward_from_output() const override {
        // In-place plan: input and output are the same buffer — no copy needed
    }

    void forward(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        std::copy(data, data + local_grid_points_, data_.begin());
        fftw_execute(plan_forward_);
        std::copy(data_.begin(), data_.begin() + local_grid_points_, data);
    }

    void inverse(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        std::copy(data, data + local_grid_points_, data_.begin());
        fftw_execute(plan_inverse_);

        Real inverse_total_points = 1.0 / total_grid_points_;
        for (int point_index = 0; point_index < local_grid_points_; point_index++) {
            data[point_index] = data_[point_index] * inverse_total_points;
        }
    }

    void inverse_raw(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        std::copy(data, data + local_grid_points_, data_.begin());
        fftw_execute(plan_inverse_);
        std::copy(data_.begin(), data_.begin() + local_grid_points_, data);
    }

    [[nodiscard]] int
    g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const override {
        int wrapped_index_1 = (miller_1 >= 0) ? miller_1 : miller_1 + grid_dimension_1_;
        int wrapped_index_2 = (miller_2 >= 0) ? miller_2 : miller_2 + grid_dimension_2_;
        int wrapped_index_3 = (miller_3 >= 0) ? miller_3 : miller_3 + grid_dimension_3_;

        if (wrapped_index_1 < local_offset_ ||
            wrapped_index_1 >= local_offset_ + local_dimension_1_) {
            return -1;
        }

        int local_i1 = wrapped_index_1 - local_offset_;
        return (local_i1 * grid_dimension_2_ + wrapped_index_2) * grid_dimension_3_ +
               wrapped_index_3;
    }

    /**
     * Returns the batch_size for which batched FFTW MPI plans are currently
     * initialized, or 0 if no batched plans exist yet.
     */
    int initialized_batched_plan_size() const override {
        return batched_count_;
    }

    /**
     * Pre-create FFTW MPI plans for batched transforms of a given batch size.
     *
     * Uses fftw_mpi_plan_many_dft to create plans that transform batch_size
     * independent 3D grids in a single MPI-coordinated call. This amortizes
     * the MPI communication overhead across multiple transforms.
     *
     * If the grid has not yet been initialized via init(), this call is a no-op.
     * Calling again with a different batch_size destroys the old plans and
     * creates new ones.
     */
    void initialize_batched_plans(int batch_size,
                                  bool use_measured_planning = false) const override {
        if (grid_dimension_1_ == 0) {
            return; // Grid not initialized yet
        }

        // Destroy previous batched plans if they exist
        if (plan_forward_batched_) {
            fftw_destroy_plan(plan_forward_batched_);
            plan_forward_batched_ = nullptr;
        }
        if (plan_inverse_batched_) {
            fftw_destroy_plan(plan_inverse_batched_);
            plan_inverse_batched_ = nullptr;
        }

        batched_count_ = batch_size;

        // Query allocation size for the batched layout
        ptrdiff_t local_n1_batched, local_offset_batched;
        const ptrdiff_t batched_dimensions[3] = {
            grid_dimension_1_, grid_dimension_2_, grid_dimension_3_};
        ptrdiff_t alloc_local_batched = fftw_mpi_local_size_many(3,
                                                                 batched_dimensions,
                                                                 static_cast<ptrdiff_t>(batch_size),
                                                                 FFTW_MPI_DEFAULT_BLOCK,
                                                                 communicator_,
                                                                 &local_n1_batched,
                                                                 &local_offset_batched);

        data_batched_.resize(static_cast<std::size_t>(alloc_local_batched));

        unsigned int planning_flags = use_measured_planning ? FFTW_MEASURE : FFTW_ESTIMATE;

        ptrdiff_t dimensions[3] = {grid_dimension_1_, grid_dimension_2_, grid_dimension_3_};

        plan_forward_batched_ =
            fftw_mpi_plan_many_dft(3,
                                   dimensions,
                                   static_cast<ptrdiff_t>(batch_size),
                                   FFTW_MPI_DEFAULT_BLOCK,
                                   FFTW_MPI_DEFAULT_BLOCK,
                                   reinterpret_cast<fftw_complex*>(data_batched_.data()),
                                   reinterpret_cast<fftw_complex*>(data_batched_.data()),
                                   communicator_,
                                   FFTW_FORWARD,
                                   planning_flags);

        plan_inverse_batched_ =
            fftw_mpi_plan_many_dft(3,
                                   dimensions,
                                   static_cast<ptrdiff_t>(batch_size),
                                   FFTW_MPI_DEFAULT_BLOCK,
                                   FFTW_MPI_DEFAULT_BLOCK,
                                   reinterpret_cast<fftw_complex*>(data_batched_.data()),
                                   reinterpret_cast<fftw_complex*>(data_batched_.data()),
                                   communicator_,
                                   FFTW_BACKWARD,
                                   planning_flags);
    }

    /**
     * Batched forward FFT using fftw_mpi_plan_many_dft.
     *
     * Expects batch_size independent 3D grids packed contiguously in data,
     * each of size local_grid_points(). The batch_size must match the value
     * passed to initialize_batched_plans().
     *
     * Falls back to per-transform loop if batched plans are not initialized
     * or if batch_size does not match.
     */
    void forward_batched(Complex* data, int batch_size) const override {
        if (!plan_forward_batched_ || batch_size != batched_count_) {
            FFTGrid::forward_batched(data, batch_size);
            return;
        }
        ProfilingCounters::instance().fft_call_count += batch_size;

        /**
         * Convert from contiguous layout (transform t at offset t*N) to FFTW's
         * interleaved howmany-tuple layout (element i of transform t at offset i*B+t)
         */
        const int local_points = local_grid_points_;
        std::fill(data_batched_.begin(),
                  data_batched_.begin() + static_cast<std::size_t>(local_points) * batch_size,
                  Complex(0.0, 0.0));
        for (int transform_index = 0; transform_index < batch_size; ++transform_index) {
            const Complex* source = data + transform_index * local_points;
            for (int point_index = 0; point_index < local_points; ++point_index) {
                data_batched_[point_index * batch_size + transform_index] = source[point_index];
            }
        }

        fftw_execute(plan_forward_batched_);

        // Convert back from interleaved to contiguous layout
        for (int transform_index = 0; transform_index < batch_size; ++transform_index) {
            Complex* destination = data + transform_index * local_points;
            for (int point_index = 0; point_index < local_points; ++point_index) {
                destination[point_index] =
                    data_batched_[point_index * batch_size + transform_index];
            }
        }
    }

    /**
     * Batched inverse FFT (raw, no normalization) using fftw_mpi_plan_many_dft.
     *
     * Expects batch_size independent 3D grids packed contiguously in data,
     * each of size local_grid_points(). The batch_size must match the value
     * passed to initialize_batched_plans().
     *
     * Falls back to per-transform loop if batched plans are not initialized
     * or if batch_size does not match.
     */
    void inverse_raw_batched(Complex* data, int batch_size) const override {
        if (!plan_inverse_batched_ || batch_size != batched_count_) {
            FFTGrid::inverse_raw_batched(data, batch_size);
            return;
        }
        ProfilingCounters::instance().fft_call_count += batch_size;

        // Convert from contiguous layout to FFTW's interleaved howmany-tuple layout
        const int local_points = local_grid_points_;
        std::fill(data_batched_.begin(),
                  data_batched_.begin() + static_cast<std::size_t>(local_points) * batch_size,
                  Complex(0.0, 0.0));
        for (int transform_index = 0; transform_index < batch_size; ++transform_index) {
            const Complex* source = data + transform_index * local_points;
            for (int point_index = 0; point_index < local_points; ++point_index) {
                data_batched_[point_index * batch_size + transform_index] = source[point_index];
            }
        }

        fftw_execute(plan_inverse_batched_);

        // Convert back from interleaved to contiguous layout
        for (int transform_index = 0; transform_index < batch_size; ++transform_index) {
            Complex* destination = data + transform_index * local_points;
            for (int point_index = 0; point_index < local_points; ++point_index) {
                destination[point_index] =
                    data_batched_[point_index * batch_size + transform_index];
            }
        }
    }

  private:
    MPI_Comm communicator_;
    int grid_dimension_1_{}, grid_dimension_2_{}, grid_dimension_3_{};
    int total_grid_points_{};
    int local_dimension_1_{}, local_offset_{}, local_grid_points_{};
    fftw_plan plan_forward_{nullptr};
    fftw_plan plan_inverse_{nullptr};
    mutable std::vector<Complex> data_;

    // Batched plan state
    mutable fftw_plan plan_forward_batched_{nullptr};
    mutable fftw_plan plan_inverse_batched_{nullptr};
    mutable int batched_count_{0};
    mutable std::vector<Complex> data_batched_;
};

} // namespace q3

#endif // Q3_WITH_FFTW_MPI
