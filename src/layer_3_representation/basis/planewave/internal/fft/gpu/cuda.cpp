/**
 * CudaGPUFFTGrid — GPUFFTGrid implementation using NVIDIA cuFFT.
 *
 * Provides both FP64 (cufftExecZ2Z) and FP32 (cufftExecC2C) complex-to-complex
 * FFT transforms. cuFFT plans are created during init() and destroyed on
 * destruction.
 *
 * For the base FFTGrid interface (grid dimensions, g-vector mapping, buffers),
 * this class composes a SerialFFTGrid internally and delegates all base-class
 * pure virtual calls to it. This avoids diamond inheritance while reusing
 * the complete FFTW-based implementation from SerialFFTGrid.
 *
 * cuFFT operates on device pointers. With unified memory (cudaMallocManaged),
 * the same pointers are accessible from both host and device.
 */

#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>

namespace q3 {
namespace gpu {

namespace {

/**
 * Check a cuFFT return code and throw on error.
 */
void check_cufft_error(cufftResult result, const char* operation) {
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("CudaGPUFFTGrid: ") + operation +
                                 " failed with result " + std::to_string(static_cast<int>(result)));
    }
}

} // anonymous namespace

class CudaGPUFFTGrid : public GPUFFTGrid {
  public:
    /**
     * Construct a CUDA FFT grid backed by the given device.
     *
     * cuFFT plans are not created here — they are created in init() after
     * the grid dimensions are known.
     */
    explicit CudaGPUFFTGrid(
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : GPUFFTGrid(std::move(device_backend)) {}

    ~CudaGPUFFTGrid() override {
        if (plan_fp64_ != 0) {
            cufftDestroy(plan_fp64_);
        }
        if (plan_fp32_ != 0) {
            cufftDestroy(plan_fp32_);
        }
        if (device_buffer_fp32_ != nullptr) {
            cudaFree(device_buffer_fp32_);
        }
        if (device_buffer_fp64_ != nullptr) {
            cudaFree(device_buffer_fp64_);
        }
        if (plan_fp32_batch_ != 0) {
            cufftDestroy(plan_fp32_batch_);
        }
        if (device_buffer_fp32_batch_ != nullptr) {
            cudaFree(device_buffer_fp32_batch_);
        }
        if (plan_fp64_batch_ != 0) {
            cufftDestroy(plan_fp64_batch_);
        }
        if (device_buffer_fp64_batch_ != nullptr) {
            cudaFree(device_buffer_fp64_batch_);
        }
    }

    // ── FFTGrid base interface (delegated to serial_fft_grid_) ───────────

    void init(int dimension_1,
              int dimension_2,
              int dimension_3,
              bool use_measured_planning = false) override {
        // Initialize the serial FFT grid (sets up dimensions and FFTW plans).
        serial_fft_grid_.init(dimension_1, dimension_2, dimension_3, use_measured_planning);

        // Create cuFFT FP64 plan (Z2Z = complex double-to-complex double).
        check_cufft_error(
            cufftPlan3d(&plan_fp64_, dimension_1, dimension_2, dimension_3, CUFFT_Z2Z),
            "cufftPlan3d FP64");

        // Create cuFFT FP32 plan (C2C = complex float-to-complex float).
        check_cufft_error(
            cufftPlan3d(&plan_fp32_, dimension_1, dimension_2, dimension_3, CUFFT_C2C),
            "cufftPlan3d FP32");

        // Allocate persistent device buffers for cuFFT.
        total_grid_points_cached_ = serial_fft_grid_.total_grid_points();

        const size_t buffer_size_fp32 =
            static_cast<size_t>(total_grid_points_cached_) * sizeof(cufftComplex);
        cudaMalloc(&device_buffer_fp32_, buffer_size_fp32);

        const size_t buffer_size_fp64 =
            static_cast<size_t>(total_grid_points_cached_) * sizeof(cufftDoubleComplex);
        cudaMalloc(&device_buffer_fp64_, buffer_size_fp64);
    }

    [[nodiscard]] int grid_dimension_1() const override {
        return serial_fft_grid_.grid_dimension_1();
    }
    [[nodiscard]] int grid_dimension_2() const override {
        return serial_fft_grid_.grid_dimension_2();
    }
    [[nodiscard]] int grid_dimension_3() const override {
        return serial_fft_grid_.grid_dimension_3();
    }
    [[nodiscard]] int total_grid_points() const override {
        return serial_fft_grid_.total_grid_points();
    }
    [[nodiscard]] int local_grid_points() const override {
        return serial_fft_grid_.local_grid_points();
    }
    [[nodiscard]] int local_offset() const override {
        return serial_fft_grid_.local_offset();
    }

    void forward(Complex* data) const override {
        serial_fft_grid_.forward(data);
    }
    void inverse(Complex* data) const override {
        serial_fft_grid_.inverse(data);
    }
    void inverse_raw(Complex* data) const override {
        serial_fft_grid_.inverse_raw(data);
    }

    [[nodiscard]] int
    g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const override {
        return serial_fft_grid_.g_vector_to_fft_grid_index(miller_1, miller_2, miller_3);
    }

    void reduce_g_space(Complex* data, int number_of_elements) const override {
        serial_fft_grid_.reduce_g_space(data, number_of_elements);
    }

    Complex* input_buffer() const override {
        return serial_fft_grid_.input_buffer();
    }
    Complex* output_buffer() const override {
        return serial_fft_grid_.output_buffer();
    }

    void execute_inverse_raw_on_buffers() const override {
        serial_fft_grid_.execute_inverse_raw_on_buffers();
    }
    void execute_forward_on_buffers() const override {
        serial_fft_grid_.execute_forward_on_buffers();
    }
    void prepare_forward_from_output() const override {
        serial_fft_grid_.prepare_forward_from_output();
    }

    // ── FP32 FFT (cuFFT via persistent device buffer) ──────────────────────

    /**
     * Forward FP32 FFT via cufftExecC2C with CUFFT_FORWARD.
     *
     * Uses a persistent device buffer allocated in init() to avoid
     * per-call cudaMalloc/cudaFree overhead.
     */
    void forward_single(ComplexSingle* data) const override {
        const size_t buffer_size =
            static_cast<size_t>(total_grid_points_cached_) * sizeof(cufftComplex);
        cudaMemcpy(device_buffer_fp32_, data, buffer_size, cudaMemcpyDefault);

        check_cufft_error(
            cufftExecC2C(plan_fp32_, device_buffer_fp32_, device_buffer_fp32_, CUFFT_FORWARD),
            "cufftExecC2C forward");

        cudaMemcpy(data, device_buffer_fp32_, buffer_size, cudaMemcpyDefault);
    }

    /**
     * Inverse FP32 FFT (raw, unnormalized) via cufftExecC2C with CUFFT_INVERSE.
     *
     * Uses a persistent device buffer allocated in init() to avoid
     * per-call cudaMalloc/cudaFree overhead.
     */
    void inverse_raw_single(ComplexSingle* data) const override {
        const size_t buffer_size =
            static_cast<size_t>(total_grid_points_cached_) * sizeof(cufftComplex);
        cudaMemcpy(device_buffer_fp32_, data, buffer_size, cudaMemcpyDefault);

        check_cufft_error(
            cufftExecC2C(plan_fp32_, device_buffer_fp32_, device_buffer_fp32_, CUFFT_INVERSE),
            "cufftExecC2C inverse");

        cudaMemcpy(data, device_buffer_fp32_, buffer_size, cudaMemcpyDefault);
    }

    // ── FP64 FFT (cuFFT via persistent device buffer) ──────────────────────

    /**
     * Forward FP64 FFT via cufftExecZ2Z with CUFFT_FORWARD.
     *
     * Uses a persistent device buffer allocated in init() to avoid
     * per-call cudaMalloc/cudaFree overhead.
     */
    void forward_transform(Complex* data) const override {
        const size_t buffer_size =
            static_cast<size_t>(total_grid_points_cached_) * sizeof(cufftDoubleComplex);
        cudaMemcpy(device_buffer_fp64_, data, buffer_size, cudaMemcpyDefault);

        check_cufft_error(
            cufftExecZ2Z(plan_fp64_, device_buffer_fp64_, device_buffer_fp64_, CUFFT_FORWARD),
            "cufftExecZ2Z forward");

        cudaMemcpy(data, device_buffer_fp64_, buffer_size, cudaMemcpyDefault);
    }

    /**
     * Inverse FP64 FFT (raw, unnormalized) via cufftExecZ2Z with CUFFT_INVERSE.
     *
     * Uses a persistent device buffer allocated in init() to avoid
     * per-call cudaMalloc/cudaFree overhead.
     */
    void inverse_transform(Complex* data) const override {
        const size_t buffer_size =
            static_cast<size_t>(total_grid_points_cached_) * sizeof(cufftDoubleComplex);
        cudaMemcpy(device_buffer_fp64_, data, buffer_size, cudaMemcpyDefault);

        check_cufft_error(
            cufftExecZ2Z(plan_fp64_, device_buffer_fp64_, device_buffer_fp64_, CUFFT_INVERSE),
            "cufftExecZ2Z inverse");

        cudaMemcpy(data, device_buffer_fp64_, buffer_size, cudaMemcpyDefault);
    }

    /**
     * ── Batched FP32 FFT (cufftPlanMany) ─────────────────────────────────
     *
     * Transforms multiple contiguous FFT grids in a single kernel launch.
     * The batch plan and device buffer are created lazily on first use and
     * resized if the batch size grows.
     */

    /**
     * Batched forward FP32 FFT via cufftPlanMany + cufftExecC2C.
     *
     * All batch_size grids are transformed in a single cuFFT kernel launch,
     * eliminating per-band dispatch overhead.
     */
    void forward_single_batch(ComplexSingle* data, int batch_size) const override {
        ensure_batch_resources(batch_size);

        const size_t buffer_size =
            static_cast<size_t>(total_grid_points_cached_) * batch_size * sizeof(cufftComplex);
        cudaMemcpy(device_buffer_fp32_batch_, data, buffer_size, cudaMemcpyDefault);

        check_cufft_error(cufftExecC2C(plan_fp32_batch_,
                                       device_buffer_fp32_batch_,
                                       device_buffer_fp32_batch_,
                                       CUFFT_FORWARD),
                          "cufftExecC2C batched forward");

        cudaMemcpy(data, device_buffer_fp32_batch_, buffer_size, cudaMemcpyDefault);
    }

    /**
     * Batched inverse FP32 FFT (raw, unnormalized) via cufftPlanMany + cufftExecC2C.
     *
     * All batch_size grids are transformed in a single cuFFT kernel launch,
     * eliminating per-band dispatch overhead.
     */
    void inverse_raw_single_batch(ComplexSingle* data, int batch_size) const override {
        ensure_batch_resources(batch_size);

        const size_t buffer_size =
            static_cast<size_t>(total_grid_points_cached_) * batch_size * sizeof(cufftComplex);
        cudaMemcpy(device_buffer_fp32_batch_, data, buffer_size, cudaMemcpyDefault);

        check_cufft_error(cufftExecC2C(plan_fp32_batch_,
                                       device_buffer_fp32_batch_,
                                       device_buffer_fp32_batch_,
                                       CUFFT_INVERSE),
                          "cufftExecC2C batched inverse");

        cudaMemcpy(data, device_buffer_fp32_batch_, buffer_size, cudaMemcpyDefault);
    }

    // ── Batched FP64 FFT (cufftPlanMany Z2Z) ─────────────────────────────

    /**
     * Batched forward FP64 FFT via cufftPlanMany + cufftExecZ2Z.
     *
     * All batch_size grids are transformed in a single cuFFT kernel launch.
     */
    void forward_transform_batch(Complex* data, int batch_size) const override {
        ensure_batch_resources_fp64(batch_size);

        const size_t buffer_size = static_cast<size_t>(total_grid_points_cached_) * batch_size *
                                   sizeof(cufftDoubleComplex);
        cudaMemcpy(device_buffer_fp64_batch_, data, buffer_size, cudaMemcpyDefault);

        check_cufft_error(cufftExecZ2Z(plan_fp64_batch_,
                                       device_buffer_fp64_batch_,
                                       device_buffer_fp64_batch_,
                                       CUFFT_FORWARD),
                          "cufftExecZ2Z batched forward");

        cudaMemcpy(data, device_buffer_fp64_batch_, buffer_size, cudaMemcpyDefault);
    }

    /**
     * Batched inverse FP64 FFT (raw, unnormalized) via cufftPlanMany + cufftExecZ2Z.
     *
     * All batch_size grids are transformed in a single cuFFT kernel launch.
     */
    void inverse_transform_batch(Complex* data, int batch_size) const override {
        ensure_batch_resources_fp64(batch_size);

        const size_t buffer_size = static_cast<size_t>(total_grid_points_cached_) * batch_size *
                                   sizeof(cufftDoubleComplex);
        cudaMemcpy(device_buffer_fp64_batch_, data, buffer_size, cudaMemcpyDefault);

        check_cufft_error(cufftExecZ2Z(plan_fp64_batch_,
                                       device_buffer_fp64_batch_,
                                       device_buffer_fp64_batch_,
                                       CUFFT_INVERSE),
                          "cufftExecZ2Z batched inverse");

        cudaMemcpy(data, device_buffer_fp64_batch_, buffer_size, cudaMemcpyDefault);
    }

  private:
    SerialFFTGrid serial_fft_grid_;
    cufftHandle plan_fp64_ = 0;
    cufftHandle plan_fp32_ = 0;

    // Persistent device buffers for single-grid cuFFT — allocated once in init().
    cufftComplex* device_buffer_fp32_ = nullptr;
    cufftDoubleComplex* device_buffer_fp64_ = nullptr;
    int total_grid_points_cached_ = 0;

    // Batched FP32 cuFFT resources — allocated lazily on first batch call.
    mutable cufftHandle plan_fp32_batch_ = 0;
    mutable cufftComplex* device_buffer_fp32_batch_ = nullptr;
    mutable int current_fp32_batch_size_ = 0;

    // Batched FP64 cuFFT resources — allocated lazily on first batch call.
    mutable cufftHandle plan_fp64_batch_ = 0;
    mutable cufftDoubleComplex* device_buffer_fp64_batch_ = nullptr;
    mutable int current_fp64_batch_size_ = 0;

    /**
     * Ensure the batched FP32 cuFFT plan and device buffer are large enough.
     */
    void ensure_batch_resources(int batch_size) const {
        if (batch_size == current_fp32_batch_size_) {
            return;
        }

        if (plan_fp32_batch_ != 0) {
            cufftDestroy(plan_fp32_batch_);
            plan_fp32_batch_ = 0;
        }
        if (device_buffer_fp32_batch_ != nullptr) {
            cudaFree(device_buffer_fp32_batch_);
            device_buffer_fp32_batch_ = nullptr;
        }

        int dimensions[3] = {serial_fft_grid_.grid_dimension_1(),
                             serial_fft_grid_.grid_dimension_2(),
                             serial_fft_grid_.grid_dimension_3()};
        int distance = total_grid_points_cached_;

        check_cufft_error(cufftPlanMany(&plan_fp32_batch_,
                                        3,           // rank (3D)
                                        dimensions,  // [d1, d2, d3]
                                        nullptr,     // inembed (null = contiguous)
                                        1,           // istride
                                        distance,    // idist
                                        nullptr,     // onembed (null = contiguous)
                                        1,           // ostride
                                        distance,    // odist
                                        CUFFT_C2C,   // type
                                        batch_size), // batch
                          "cufftPlanMany FP32 batch");

        const size_t buffer_size =
            static_cast<size_t>(total_grid_points_cached_) * batch_size * sizeof(cufftComplex);
        cudaMalloc(&device_buffer_fp32_batch_, buffer_size);

        current_fp32_batch_size_ = batch_size;
    }

    /**
     * Ensure the batched FP64 cuFFT plan and device buffer are large enough.
     */
    void ensure_batch_resources_fp64(int batch_size) const {
        if (batch_size == current_fp64_batch_size_) {
            return;
        }

        if (plan_fp64_batch_ != 0) {
            cufftDestroy(plan_fp64_batch_);
            plan_fp64_batch_ = 0;
        }
        if (device_buffer_fp64_batch_ != nullptr) {
            cudaFree(device_buffer_fp64_batch_);
            device_buffer_fp64_batch_ = nullptr;
        }

        int dimensions[3] = {serial_fft_grid_.grid_dimension_1(),
                             serial_fft_grid_.grid_dimension_2(),
                             serial_fft_grid_.grid_dimension_3()};
        int distance = total_grid_points_cached_;

        check_cufft_error(cufftPlanMany(&plan_fp64_batch_,
                                        3,           // rank (3D)
                                        dimensions,  // [d1, d2, d3]
                                        nullptr,     // inembed (null = contiguous)
                                        1,           // istride
                                        distance,    // idist
                                        nullptr,     // onembed (null = contiguous)
                                        1,           // ostride
                                        distance,    // odist
                                        CUFFT_Z2Z,   // type
                                        batch_size), // batch
                          "cufftPlanMany FP64 batch");

        const size_t buffer_size = static_cast<size_t>(total_grid_points_cached_) * batch_size *
                                   sizeof(cufftDoubleComplex);
        cudaMalloc(&device_buffer_fp64_batch_, buffer_size);

        current_fp64_batch_size_ = batch_size;
    }
};

std::unique_ptr<GPUFFTGrid>
create_cuda_gpu_fft_grid(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend) {
    return std::make_unique<CudaGPUFFTGrid>(std::move(device_backend));
}

} // namespace gpu
} // namespace q3
