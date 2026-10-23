#pragma once

/**
 * Shared GPU entry-point runner for the q3_gpu* binaries.
 *
 * The four GPU executables (Metal/Accelerate, CUDA, HIP, SYCL) differ only in
 * which backend factories they link and in their banner/usage labels. Each
 * main_gpu*.cpp builds a GPUEntryPointDescriptor and delegates to
 * run_gpu_entry_point(), which owns the shared flow:
 *
 *   parse CLI → init logging → create + validate backends → build serial
 *   iterator → inject GPU Davidson → run SCF flavor → write results →
 *   print profiling summary.
 *
 * Backend factories are passed as std::function so this header stays free of
 * vendor headers; the factories are resolved at link time in each binary.
 */

#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/addon_registration.hpp"
#include "io/cli.hpp"
#include "io/input/parser/factory.hpp"
#include "io/launcher.hpp"
#include "io/logger.hpp"
#include "io/output/companion_outputs.hpp"
#include "layer_0_core/execution/gpu/device/factory.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/davidson.hpp"
#include "layer_4_computation/flavors/self_consistent_field.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

namespace q3 {
namespace cli {

/**
 * Everything that distinguishes one GPU binary from another.
 */
struct GPUEntryPointDescriptor {
    /// Banner label, e.g. "CUDA GPU" → "BEGIN (CUDA GPU): <timestamp>".
    std::string backend_label;
    /// Binary name for the usage line, e.g. "q3_gpu_cuda".
    std::string binary_name;
    /// Vendor BLAS name for the validation log line, e.g. "cuBLAS". Optional.
    std::string blas_library_name;
    /// Vendor eigensolver name for the validation log line, e.g. "cuSOLVER". Optional.
    std::string eigensolver_library_name;
    /// Human-readable description of the mixed-precision mode, e.g. "mixed (FP32 FFT via cuFFT)".
    std::string mixed_precision_description = "mixed (FP32 FFT)";

    using DeviceBackendPointer = std::shared_ptr<execution::gpu::device::GPUDeviceBackend>;

    std::function<std::unique_ptr<execution::gpu::blas::GPUBLAS>(DeviceBackendPointer)> create_blas;
    std::function<std::unique_ptr<execution::gpu::lapack::GPUEigensolver>(DeviceBackendPointer)>
        create_eigensolver;
    /**
     * Optional: backends without a native GPU FFT (Metal) leave this empty and
     * the FP64 path falls back to the PlanewaveBasis FFTW grid.
     */
    std::function<std::unique_ptr<gpu::GPUFFTGrid>(DeviceBackendPointer)> create_fft_grid;
};

// ─── Backend validation ───────────────────────────────────────────────────────

inline void validate_gpu_device_backend(execution::gpu::device::GPUDeviceBackend& device_backend) {
    log::info("GPU Device: {}", device_backend.device_name());
    log::info("GPU Memory: {:.1f} GB",
              static_cast<double>(device_backend.total_memory_bytes()) /
                  (1024.0 * 1024.0 * 1024.0));
    log::info("GPU FP64 support: {}", device_backend.supports_double_precision() ? "yes" : "no");

    // Validate allocate/deallocate cycle
    void* test_pointer = device_backend.allocate_unified(1024);
    device_backend.deallocate(test_pointer);
    log::info("GPU memory allocate/deallocate: OK");
}

inline void validate_gpu_blas(execution::gpu::blas::GPUBLAS& gpu_blas,
                              const std::string& blas_library_name) {
    /**
     * Quick BLAS sanity: compute dot product of [1+0i, 0+1i] with itself
     * Expected: conj([1,i]) · [1,i] = 1*1 + (-i)*i = 1 + 1 = 2
     */
    std::vector<Complex> test_vector = {Complex(1.0, 0.0), Complex(0.0, 1.0)};
    Complex dot_result = gpu_blas.dot_product(2, test_vector.data(), test_vector.data());
    double error = std::abs(dot_result.real() - 2.0) + std::abs(dot_result.imag());
    if (error > 1.0e-14) {
        throw std::runtime_error("GPU BLAS dot product validation failed");
    }
    if (blas_library_name.empty()) {
        log::info("GPU BLAS validation: OK");
    } else {
        log::info("GPU BLAS ({}) validation: OK", blas_library_name);
    }
}

inline void validate_gpu_eigensolver(execution::gpu::lapack::GPUEigensolver& gpu_eigensolver,
                                     const std::string& eigensolver_library_name) {
    // Solve 2x2 Hermitian eigenvalue problem: A = [2,1; 1,2] -> eigenvalues = [1,3]
    std::vector<Complex> matrix = {Complex(2, 0), Complex(1, 0), Complex(1, 0), Complex(2, 0)};
    std::vector<Real> eigenvalues(2);
    gpu_eigensolver.hermitian_eigenvalue_solve(2, matrix.data(), 2, eigenvalues.data());
    double eigen_error = std::abs(eigenvalues[0] - 1.0) + std::abs(eigenvalues[1] - 3.0);
    if (eigen_error > 1.0e-12) {
        throw std::runtime_error("GPU eigensolver validation failed");
    }
    if (eigensolver_library_name.empty()) {
        log::info("GPU eigensolver validation: OK");
    } else {
        log::info("GPU eigensolver ({}) validation: OK", eigensolver_library_name);
    }
}

// ─── Shared entry point ───────────────────────────────────────────────────────

/**
 * Run the full GPU SCF entry-point flow for the backend described by
 * 'descriptor'. Returns the process exit code.
 */
inline int run_gpu_entry_point(int argc, char* argv[], const GPUEntryPointDescriptor& descriptor) {
    q3::addons::register_all();
    auto options = parse_options(argc, argv, descriptor.binary_name);

    q3::log::init(options.log_file);

    /**
     * Answered before the input is resolved, so `--version` works in a
     * directory holding no input.
     */
    if (options.print_version) {
        cli::print_version(descriptor.binary_name);
        return 0;
    }

    /**
     * One process, or none. There is no multi-process mode here: the GPU path
     * builds a serial iterator, so every task of a multi-task step would set up
     * the whole problem on its own device, converge it, and print the same
     * answer -- which is what `srun -n 2 q3_gpu_cuda` did until 2026-09-11,
     * silently, for twice the allocation and 2% more wall time (camsai/q3#15).
     *
     * Refused after --version so `srun q3_gpu_cuda --version` still answers,
     * and before anything touches a device.
     */
    const auto launcher_tasks = detect_launcher_tasks();
    if (launcher_tasks.count > 1) {
        /**
         * The one case where several tasks running the same binary is what the
         * user meant: packing independent calculations into one step. Nothing
         * here can tell that apart from the mistake, so it is opt-in.
         */
        const bool packing_independent_runs = std::getenv("Q3_ALLOW_MULTI_TASK_GPU") != nullptr;
        constexpr const char* message =
            "{} was launched as {} tasks ({}={}), and it has no multi-process mode: each task "
            "would set up the whole problem on its own device and solve all of it, giving {} "
            "identical answers for {}x the allocation. Run it as one task, or use q3_mpi to "
            "spread k-points across ranks on CPU, or q3_mpi_gpu_cuda to spend one device "
            "per rank on a single calculation. Set "
            "Q3_ALLOW_MULTI_TASK_GPU=1 to proceed anyway, which is right only when the tasks "
            "are packing independent calculations into one step.";
        if (!packing_independent_runs) {
            log::error(message,
                       descriptor.binary_name,
                       launcher_tasks.count,
                       launcher_tasks.variable,
                       launcher_tasks.count,
                       launcher_tasks.count,
                       launcher_tasks.count);
            return 1;
        }
        log::warn(message,
                  descriptor.binary_name,
                  launcher_tasks.count,
                  launcher_tasks.variable,
                  launcher_tasks.count,
                  launcher_tasks.count,
                  launcher_tasks.count);
    }

    if (!options.log_level.empty()) {
        log::set_level(options.log_level);
    }

    bool use_mixed_precision = false;
    if (options.precision == "mixed") {
        use_mixed_precision = true;
    } else if (!options.precision.empty() && options.precision != "double") {
        log::error("Unknown precision: {}. Use 'double' or 'mixed'.", options.precision);
        return 1;
    }

    print_timestamped_banner("BEGIN (" + descriptor.backend_label + "): ");

    int exit_code = 0;
    try {
        auto input = read_input(options, descriptor.binary_name);
        /**
         * Step 4 below runs the self-consistent field flavor whatever the
         * input asks for, so an NSCF deck came back as an SCF under the
         * NSCF's name -- the mistake q3_mpi and q3_mpi_gpu_cuda already
         * refuse. It matters more now that a GPU image runs this binary as
         * plain `q3` (#316), where a workflow cannot tell it from the serial
         * build by name. Refuse before any device is touched.
         */
        if (input.calculation != "self_consistent_field") {
            throw std::runtime_error(descriptor.binary_name +
                                     " runs self_consistent_field only, but this input asks for '" +
                                     input.calculation +
                                     "'. Run this input with the serial q3 build, which supports "
                                     "every flavor.");
        }
        if (!input.hubbard.empty()) {
            log::error("DFT+U (hubbard) is not yet supported on GPU. Run on CPU instead.");
            return 1;
        }

        // ── Step 1: Initialize and validate the GPU backends ──────────────
        log::info("Initializing {} backends...", descriptor.backend_label);
        auto device_backend = execution::gpu::device::create_gpu_device_backend();
        validate_gpu_device_backend(*device_backend);

        auto gpu_blas = descriptor.create_blas(device_backend);
        validate_gpu_blas(*gpu_blas, descriptor.blas_library_name);

        auto gpu_eigensolver = descriptor.create_eigensolver(device_backend);
        validate_gpu_eigensolver(*gpu_eigensolver, descriptor.eigensolver_library_name);

        /**
         * Backends without a native GPU FFT run the FP64 path through the
         * PlanewaveBasis FFTW grid (gpu_fft_grid stays null).
         */
        std::unique_ptr<gpu::GPUFFTGrid> gpu_fft_grid;
        if (descriptor.create_fft_grid) {
            gpu_fft_grid = descriptor.create_fft_grid(device_backend);
        }

        log::info("All {} backend validations passed.", descriptor.backend_label);
        log::info("Precision mode: {}",
                  use_mixed_precision ? descriptor.mixed_precision_description : "double (FP64)");
        log::info("");

        // ── Step 2: Create serial execution policy and iterator ───────────
        auto execution_policy = std::make_unique<execution::PlanewaveSerialExecutionPolicy>();
        const execution::PlanewaveSerialExecutionPolicy& policy_ref = *execution_policy;
        auto iterator = create_kohn_sham_iterator(input, policy_ref);
        iterator->set_owned_execution_policy(std::move(execution_policy));

        // ── Step 3: Create GPU Davidson diagonalizer and inject ───────────
        auto gpu_davidson = std::make_unique<PlanewaveDavidsonDiagonalizer>(
            PlanewaveDavidsonDiagonalizer::default_serial_execution_policy,
            *gpu_blas,
            gpu_fft_grid.get(),
            *gpu_eigensolver,
            device_backend);

        if (use_mixed_precision) {
            gpu_davidson->set_use_mixed_precision(true);
        }

        iterator->set_diagonalizer(std::move(gpu_davidson));

        // ── Step 4: Run SCF through the flavor system ─────────────────────
        log::info("     Calculation type: {}", input.calculation);
        auto flavor = std::make_unique<SelfConsistentFieldFlavor>(std::move(iterator));
        auto flavor_result = flavor->run(input);
        flavor_result.write_results_json(input.output_directory + "/" + input.prefix +
                                         "_results.json");
        write_companion_outputs(input, flavor_result);

    } catch (const std::exception& exception) {
        log::error("Error: {}", exception.what());
        exit_code = 1;
    }

    auto& profiling = q3::ProfilingCounters::instance();
    q3::log::info("     Total FFT calls:                {}", profiling.fft_call_count.load());
    q3::log::info("     Total Davidson iterations:      {}",
                  profiling.davidson_iteration_count.load());
    q3::log::info("     Total Hamiltonian applications: {}",
                  profiling.hamiltonian_application_count.load());
    profiling.print_memory_summary();
    profiling.davidson_timing.print_summary();
    print_timestamped_banner("END (" + descriptor.backend_label + "): ");
    return exit_code;
}

} // namespace cli
} // namespace q3
