/**
 * main_gpu_hip.cpp — HIP/ROCm GPU-accelerated entry point for Q3.
 *
 * Links the HIP backend factories (device, hipBLAS, hipSOLVER, hipFFT) and
 * delegates the shared entry-point flow to cli::run_gpu_entry_point().
 *
 * Usage: q3_gpu_hip <input_file> [--log-level <level>] [--precision double|mixed]
 */

#include "io/gpu_entry_point.hpp"

// HIP backend factory functions (defined in hip.cpp files, linked at build time)
namespace q3::execution::gpu::blas {
std::unique_ptr<GPUBLAS>
create_hip_gpu_blas(std::shared_ptr<device::GPUDeviceBackend> device_backend);
} // namespace q3::execution::gpu::blas

namespace q3::execution::gpu::lapack {
std::unique_ptr<GPUEigensolver>
create_hip_gpu_eigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend);
} // namespace q3::execution::gpu::lapack

namespace q3::gpu {
std::unique_ptr<GPUFFTGrid>
create_hip_gpu_fft_grid(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend);
} // namespace q3::gpu

int main(int argc, char* argv[]) {
    q3::cli::GPUEntryPointDescriptor descriptor;
    descriptor.backend_label = "HIP GPU";
    descriptor.binary_name = "q3_gpu_hip";
    descriptor.blas_library_name = "hipBLAS";
    descriptor.eigensolver_library_name = "hipSOLVER";
    descriptor.mixed_precision_description = "mixed (FP32 FFT via hipFFT)";
    descriptor.create_blas = q3::execution::gpu::blas::create_hip_gpu_blas;
    descriptor.create_eigensolver = q3::execution::gpu::lapack::create_hip_gpu_eigensolver;
    descriptor.create_fft_grid = q3::gpu::create_hip_gpu_fft_grid;
    return q3::cli::run_gpu_entry_point(argc, argv, descriptor);
}
