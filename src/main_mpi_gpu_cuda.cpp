/**
 * main_mpi_gpu_cuda.cpp — CUDA GPU entry point that also speaks MPI.
 *
 * q3_gpu_cuda is one process on one device: main_gpu_cuda.cpp delegates to
 * run_gpu_entry_point(), which hard-codes PlanewaveSerialExecutionPolicy. So
 * `srun -n 8 q3_gpu_cuda` runs eight independent duplicate calculations rather
 * than one calculation across eight devices, and there is no way to spend more
 * than one GPU on a single system.
 *
 * This binary is the other composition: the MPI setup from main_mpi.cpp --
 * execution policy, topology, MPI iterator -- with the CUDA backends and the
 * GPU Davidson injected the way the GPU entry point injects them. Each rank
 * drives its own device and owns a k-point pool, which is the decomposition
 * that already scales on CPU.
 *
 * Device selection. cuda.cpp calls cudaSetDevice(0), so ranks sharing a node
 * must be given different devices from outside:
 *
 *   srun -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 \
 *       bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; q3_mpi_gpu_cuda --pools 8 deck.yml'
 *
 * Without that every rank on a node opens GPU 0 and the run measures four
 * processes contending for one card.
 *
 * Scope. Like q3_mpi this runs self_consistent_field only, and for the same
 * structural reason: it supplies its own execution policy, and that is the one
 * flavor which accepts one.
 */

#include "layer_0_core/profiling/counters.hpp"
#include <memory>
#include <stdexcept>
#include <string>

#include "io/addon_registration.hpp"
#include "io/cli.hpp"
#include "io/input/parser/factory.hpp"
#include "io/iterator/kohn_sham/planewave/factory_mpi.hpp"
#include "io/output/companion_outputs.hpp"
#include "layer_0_core/execution/gpu/device/factory.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/davidson.hpp"
#include "layer_4_computation/flavors/self_consistent_field.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/mpi.hpp"
#include "utils/time.hpp"

using namespace q3;

// CUDA backend factories, defined in the cuda.cpp translation units.
namespace q3::execution::gpu::blas {
std::unique_ptr<GPUBLAS>
create_cuda_gpu_blas(std::shared_ptr<device::GPUDeviceBackend> device_backend);
} // namespace q3::execution::gpu::blas

namespace q3::execution::gpu::lapack {
std::unique_ptr<GPUEigensolver>
create_cuda_gpu_eigensolver(std::shared_ptr<device::GPUDeviceBackend> device_backend);
} // namespace q3::execution::gpu::lapack

namespace q3::gpu {
std::unique_ptr<GPUFFTGrid>
create_cuda_gpu_fft_grid(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend);
} // namespace q3::gpu

int main(int argc, char* argv[]) {
    q3::addons::register_all();
    auto options = cli::parse_options(argc, argv, "q3_mpi_gpu_cuda");

    auto policy = std::make_unique<execution::PlanewaveMPIExecutionPolicy>();
    log::init_mpi(options.log_file);

    if (options.print_version) {
        cli::print_version("q3_mpi_gpu_cuda");
        return 0;
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

    cli::print_timestamped_banner("BEGIN (CUDA GPU + MPI): ");

    int exit_code = 0;
    try {
        auto input = cli::read_input(options, "q3_mpi_gpu_cuda");
        if (!input.hubbard.empty()) {
            log::error("DFT+U (hubbard) is not yet supported on GPU. Run on CPU instead.");
            return 1;
        }
        if (input.calculation != "self_consistent_field") {
            throw std::runtime_error(
                "q3_mpi_gpu_cuda runs self_consistent_field only, but this input asks for '" +
                input.calculation + "'.");
        }

        // ── GPU backends, one device per rank ─────────────────────────────
        auto device_backend = execution::gpu::device::create_gpu_device_backend();
        auto gpu_blas = execution::gpu::blas::create_cuda_gpu_blas(device_backend);
        auto gpu_eigensolver = execution::gpu::lapack::create_cuda_gpu_eigensolver(device_backend);
        auto gpu_fft_grid = gpu::create_cuda_gpu_fft_grid(device_backend);

        // ── MPI topology, as q3_mpi builds it ─────────────────────────────
        int requested_number_of_pools = options.number_of_kpoint_pools_was_set
                                            ? options.number_of_kpoint_pools
                                            : input.number_of_kpoint_pools;
        policy->configure_topology(requested_number_of_pools);
        if (options.distribute_bands) {
            policy->enable_band_distribution();
        }
        int ranks_per_pool = policy->world_size() / std::max(policy->number_of_pools(), 1);
        log::info("MPI topology: {} rank(s), {} k-point pool(s), {} rank(s) per pool, "
                  "one CUDA device each",
                  policy->world_size(),
                  policy->number_of_pools(),
                  ranks_per_pool);

        /**
         * The Davidson takes the execution policy by base reference, so the MPI
         * policy goes in where the serial one does for q3_gpu_cuda. Hold the
         * reference before the unique_ptr is moved into the iterator.
         */
        const execution::ExecutionPolicy& policy_reference = *policy;
        auto iterator = create_scf_iterator_mpi(input, std::move(policy));
        bool is_root = iterator->execution_policy().is_root();

        auto gpu_davidson = std::make_unique<PlanewaveDavidsonDiagonalizer>(
            policy_reference, *gpu_blas, gpu_fft_grid.get(), *gpu_eigensolver, device_backend);
        if (use_mixed_precision) {
            gpu_davidson->set_use_mixed_precision(true);
        }
        iterator->set_diagonalizer(std::move(gpu_davidson));

        auto flavor = std::make_unique<SelfConsistentFieldFlavor>(std::move(iterator));
        cli::print_header();
        log::info("     Calculation type: {}", input.calculation);

        auto flavor_result = flavor->run(input);
        if (is_root) {
            flavor_result.write_results_json(input.output_directory + "/" + input.prefix +
                                             "_results.json");
            write_companion_outputs(input, flavor_result);
            ProfilingCounters::instance().davidson_timing.print_summary();
            cli::print_footer();
        }
    } catch (const std::exception& error) {
        log::error("Error: {}", error.what());
        exit_code = 1;
    }

    cli::print_timestamped_banner("END (CUDA GPU + MPI): ");
    return exit_code;
}
