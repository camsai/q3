#pragma once

/**
 * The build picks one serial FFT backend (CMake option Q3_FFT_BACKEND), and
 * SerialFFTGrid names it. With PocketFFT selected this header never includes
 * <fftw3.h>, so nothing in the binary refers to FFTW.
 */
#ifdef Q3_FFT_BACKEND_POCKETFFT

#include "layer_3_representation/basis/planewave/internal/fft/serial_pocket_fft_grid.hpp"

namespace q3 {
using SerialFFTGrid = SerialPocketFFTGrid;
} // namespace q3

#else

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <thread>
#include <vector>

#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"

namespace q3 {

/**
 * SerialFFTGrid — node-local 3D FFT implementation using FFTW3.
 *
 * This implementation assumes the entire grid resides in the memory of a
 * single process. It supports OpenMP multi-threading within FFTW if
 * Q3_USE_FFTW_THREADS is defined.
 */
class SerialFFTGrid : public FFTGrid {
    /**
     * One thread's scratch and the plans bound to it. Slot 0 is not one of
     * these -- it is `work_` / `output_buffer_` and the plans beside them, so
     * that the single-slot case allocates and plans exactly what it used to.
     */
    struct TransformSlot {
        mutable std::vector<Complex> input;
        mutable std::vector<Complex> output;
        fftw_plan plan_forward{nullptr};
        fftw_plan plan_inverse{nullptr};
    };

  public:
    SerialFFTGrid() = default;
    ~SerialFFTGrid() override {
        if (plan_forward_)
            fftw_destroy_plan(plan_forward_);
        if (plan_inverse_)
            fftw_destroy_plan(plan_inverse_);
        for (TransformSlot& slot : additional_slots_) {
            if (slot.plan_forward)
                fftw_destroy_plan(slot.plan_forward);
            if (slot.plan_inverse)
                fftw_destroy_plan(slot.plan_inverse);
        }
    }

    // FFTW plans are non-copyable resources
    SerialFFTGrid(const SerialFFTGrid&) = delete;
    SerialFFTGrid& operator=(const SerialFFTGrid&) = delete;
    SerialFFTGrid(SerialFFTGrid&&) = default;
    SerialFFTGrid& operator=(SerialFFTGrid&&) = default;

    void init(int dimension_1,
              int dimension_2,
              int dimension_3,
              bool use_measured_planning = false) override {
        grid_dimension_1_ = dimension_1;
        grid_dimension_2_ = dimension_2;
        grid_dimension_3_ = dimension_3;
        int total_points = dimension_1 * dimension_2 * dimension_3;
        work_.resize(total_points);

        /**
         * How many transforms callers may drive at once. One slot -- the
         * historical behaviour -- when there is one thread to run them on, so
         * a single-threaded run allocates nothing extra and executes the same
         * plan on the same buffers it always did.
         *
         * Each additional slot costs two more grids of scratch. That is small
         * beside the wavefunction storage a grid this size implies, and it is
         * what buys a band loop that scales. The alternative -- FFTW's own
         * intra-transform threading, which is what this class used to do with
         * `OMP_NUM_THREADS` -- pays a fork and join inside every one of the
         * thousands of transforms a Davidson iteration issues, and the
         * measurements in docs/performance_nersc_perlmutter.md show it
         * bottoming out at eight threads and then getting worse than serial.
         * That document names threading across bands with serial transforms as
         * the scalable axis; this is it.
         */
        int requested_slots = execution::threading::maximum_threads();

        /**
         * A ceiling on what the extra slots may cost, because the thread count
         * is the machine's and the grid size is the calculation's, and their
         * product is nobody's decision. Sixty-four threads on a 128^3 grid
         * would otherwise claim four gibibytes of scratch that no input asked
         * for. Past this point the slots stop being added and the band loop
         * runs on fewer threads than the machine has -- which costs some
         * parallelism, and is the right way round: a slower run is recoverable
         * and an out-of-memory run is not.
         */
        constexpr std::size_t maximum_additional_scratch_bytes = 1024ull * 1024ull * 1024ull;
        std::size_t bytes_per_slot = 2 * static_cast<std::size_t>(total_points) * sizeof(Complex);
        int affordable_slots =
            1 + static_cast<int>(maximum_additional_scratch_bytes /
                                 std::max<std::size_t>(bytes_per_slot, std::size_t{1}));
        requested_slots = std::min(requested_slots, affordable_slots);

#ifdef Q3_USE_FFTW_THREADS
        static bool fftw_threads_initialized = false;
        int number_of_fft_threads = 1;
        const char* omp_threads_env = std::getenv("OMP_NUM_THREADS");
        if (omp_threads_env != nullptr) {
            number_of_fft_threads = std::atoi(omp_threads_env);
            if (number_of_fft_threads < 1) {
                number_of_fft_threads = 1;
            }
        }
        /**
         * Threads go to whole transforms, not to the inside of one. A plan
         * that forks its own team inside a band-parallel region would
         * oversubscribe the machine, and -- because a two-thread transform
         * decomposes differently from a one-thread transform -- would also
         * make a band's last bits depend on which slot it landed in. One
         * thread per transform keeps every slot numerically the same plan.
         */
        if (requested_slots > 1) {
            number_of_fft_threads = 1;
        }
        if (number_of_fft_threads > 1) {
            if (!fftw_threads_initialized) {
                fftw_init_threads();
                fftw_threads_initialized = true;
            }
            fftw_plan_with_nthreads(number_of_fft_threads);
        }
#endif

        unsigned int planning_flags = use_measured_planning ? FFTW_MEASURE : FFTW_ESTIMATE;
        output_buffer_.resize(total_points);

        /**
         * Out-of-place plans: input from work_, output to output_buffer_.
         * Out-of-place avoids internal data shuffling and is typically faster.
         */
        plan_forward_ = fftw_plan_dft_3d(dimension_1,
                                         dimension_2,
                                         dimension_3,
                                         reinterpret_cast<fftw_complex*>(work_.data()),
                                         reinterpret_cast<fftw_complex*>(output_buffer_.data()),
                                         FFTW_FORWARD,
                                         planning_flags);

        plan_inverse_ = fftw_plan_dft_3d(dimension_1,
                                         dimension_2,
                                         dimension_3,
                                         reinterpret_cast<fftw_complex*>(work_.data()),
                                         reinterpret_cast<fftw_complex*>(output_buffer_.data()),
                                         FFTW_BACKWARD,
                                         planning_flags);

        /**
         * Slots 1..N-1. Each gets its own plan over its own arrays rather than
         * borrowing slot 0's through fftw_execute_dft, because that shortcut
         * is only valid when the new arrays share the plan's alignment class
         * -- a property of the allocator, not of anything checked here.
         * Planning is serial by construction: FFTW's planner is not
         * re-entrant, and this runs once at setup, before any parallel region.
         */
        additional_slots_.resize(static_cast<std::size_t>(std::max(requested_slots - 1, 0)));
        for (TransformSlot& slot : additional_slots_) {
            slot.input.resize(total_points);
            slot.output.resize(total_points);
            slot.plan_forward =
                fftw_plan_dft_3d(dimension_1,
                                 dimension_2,
                                 dimension_3,
                                 reinterpret_cast<fftw_complex*>(slot.input.data()),
                                 reinterpret_cast<fftw_complex*>(slot.output.data()),
                                 FFTW_FORWARD,
                                 planning_flags);
            slot.plan_inverse =
                fftw_plan_dft_3d(dimension_1,
                                 dimension_2,
                                 dimension_3,
                                 reinterpret_cast<fftw_complex*>(slot.input.data()),
                                 reinterpret_cast<fftw_complex*>(slot.output.data()),
                                 FFTW_BACKWARD,
                                 planning_flags);
        }
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
        return grid_dimension_1_ * grid_dimension_2_ * grid_dimension_3_;
    }

    void forward(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        int total_points = grid_dimension_1_ * grid_dimension_2_ * grid_dimension_3_;
        // Copy to the plan's input buffer (guaranteed alignment), execute, copy back
        std::memcpy(work_.data(), data, total_points * sizeof(Complex));
        fftw_execute(plan_forward_);
        std::memcpy(data, output_buffer_.data(), total_points * sizeof(Complex));
    }

    void inverse(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        int total_points = grid_dimension_1_ * grid_dimension_2_ * grid_dimension_3_;
        std::memcpy(work_.data(), data, total_points * sizeof(Complex));
        fftw_execute(plan_inverse_);
        Real inverse_total_points = 1.0 / total_points;
        for (int point_index = 0; point_index < total_points; point_index++) {
            data[point_index] = output_buffer_[point_index] * inverse_total_points;
        }
    }

    void inverse_raw(Complex* data) const override {
        ProfilingCounters::instance().fft_call_count++;
        int total_points = grid_dimension_1_ * grid_dimension_2_ * grid_dimension_3_;
        std::memcpy(work_.data(), data, total_points * sizeof(Complex));
        fftw_execute(plan_inverse_);
        std::memcpy(data, output_buffer_.data(), total_points * sizeof(Complex));
    }

    [[nodiscard]] int
    g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const override {
        int wrapped_index_1 = (miller_1 >= 0) ? miller_1 : miller_1 + grid_dimension_1_;
        int wrapped_index_2 = (miller_2 >= 0) ? miller_2 : miller_2 + grid_dimension_2_;
        int wrapped_index_3 = (miller_3 >= 0) ? miller_3 : miller_3 + grid_dimension_3_;
        return (wrapped_index_1 * grid_dimension_2_ + wrapped_index_2) * grid_dimension_3_ +
               wrapped_index_3;
    }

    [[nodiscard]] int local_grid_points() const override {
        return total_grid_points();
    }

    [[nodiscard]] int local_offset() const override {
        return 0;
    }

    void reduce_g_space(Complex* /*data*/, int /*number_of_elements*/) const override {
        // No-op for serial: data is already redundant
    }

    Complex* input_buffer() const override {
        return work_.data();
    }

    Complex* output_buffer() const override {
        return output_buffer_.data();
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
        // Out-of-place plan: copy output → input for the next forward transform
        int total_points = grid_dimension_1_ * grid_dimension_2_ * grid_dimension_3_;
        std::memcpy(work_.data(), output_buffer_.data(), total_points * sizeof(Complex));
    }

    /**
     * How many transforms may run at once: one per set of scratch arrays
     * `init()` planned. Slot 0 is the pair every other entry point already
     * uses, so a caller that asks for one slot runs the historical code.
     */
    [[nodiscard]] int concurrent_transform_slots() const override {
        return 1 + static_cast<int>(additional_slots_.size());
    }

    [[nodiscard]] Complex* slot_input_buffer(int slot) const override {
        return slot == 0 ? work_.data()
                         : additional_slots_[static_cast<std::size_t>(slot - 1)].input.data();
    }

    [[nodiscard]] Complex* slot_output_buffer(int slot) const override {
        return slot == 0 ? output_buffer_.data()
                         : additional_slots_[static_cast<std::size_t>(slot - 1)].output.data();
    }

    void execute_inverse_raw_on_slot(int slot) const override {
        ProfilingCounters::instance().fft_call_count++;
        fftw_execute(slot == 0
                         ? plan_inverse_
                         : additional_slots_[static_cast<std::size_t>(slot - 1)].plan_inverse);
    }

    void execute_forward_on_slot(int slot) const override {
        ProfilingCounters::instance().fft_call_count++;
        fftw_execute(slot == 0
                         ? plan_forward_
                         : additional_slots_[static_cast<std::size_t>(slot - 1)].plan_forward);
    }

    /**
     * The same copy-in, transform, copy-out `inverse_raw()` performs, driven
     * through slot `slot`'s scratch and plan. Slot 0 is `inverse_raw()` itself.
     */
    void inverse_raw_on_slot(Complex* data, int slot) const override {
        ProfilingCounters::instance().fft_call_count++;
        std::size_t total_points =
            static_cast<std::size_t>(grid_dimension_1_) * grid_dimension_2_ * grid_dimension_3_;
        std::memcpy(slot_input_buffer(slot), data, total_points * sizeof(Complex));
        fftw_execute(slot == 0
                         ? plan_inverse_
                         : additional_slots_[static_cast<std::size_t>(slot - 1)].plan_inverse);
        std::memcpy(data, slot_output_buffer(slot), total_points * sizeof(Complex));
    }

    void prepare_forward_from_output_on_slot(int slot) const override {
        int total_points = grid_dimension_1_ * grid_dimension_2_ * grid_dimension_3_;
        std::memcpy(slot_input_buffer(slot),
                    slot_output_buffer(slot),
                    static_cast<std::size_t>(total_points) * sizeof(Complex));
    }

  private:
    int grid_dimension_1_{}, grid_dimension_2_{}, grid_dimension_3_{};
    fftw_plan plan_forward_{nullptr};
    fftw_plan plan_inverse_{nullptr};
    mutable std::vector<Complex> work_;                   // input workspace for FFTW plan creation
    mutable std::vector<Complex> output_buffer_;          // output buffer for out-of-place FFTs
    mutable std::vector<TransformSlot> additional_slots_; // slots 1..N-1, one per extra thread
};

} // namespace q3

#endif // Q3_FFT_BACKEND_POCKETFFT
