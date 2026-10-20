/**
 * main_gpu.cpp — Metal/Accelerate GPU-accelerated entry point for Q3.
 *
 * Links the Metal device backend with Accelerate-based BLAS/LAPACK factories
 * and delegates the shared entry-point flow to cli::run_gpu_entry_point().
 * No native GPU FFT grid: the FP64 path uses the PlanewaveBasis FFTW grid.
 *
 * Usage: q3_gpu <input_file> [--log-level <level>] [--precision double|mixed]
 */

#include "io/gpu_entry_point.hpp"

// Accelerate backend factory functions (defined in metal.cpp files, linked at build time)
namespace q3::execution::gpu::blas {
std::unique_ptr<GPUBLAS>
create_accelerate_gpu_blas(std::shared_ptr<device::GPUDeviceBackend> device_backend);
} // namespace q3::execution::gpu::blas

namespace q3::execution::gpu::lapack {
std::unique_ptr<GPUEigensolver>
create_accelerate_gpu_eigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend);
} // namespace q3::execution::gpu::lapack

int main(int argc, char* argv[]) {
    q3::cli::GPUEntryPointDescriptor descriptor;
    descriptor.backend_label = "GPU";
    descriptor.binary_name = "q3_gpu";
    descriptor.create_blas = q3::execution::gpu::blas::create_accelerate_gpu_blas;
    descriptor.create_eigensolver = q3::execution::gpu::lapack::create_accelerate_gpu_eigensolver;
    // create_fft_grid left empty: Metal has no native GPU FFT backend yet.
    return q3::cli::run_gpu_entry_point(argc, argv, descriptor);
}
