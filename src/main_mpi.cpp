#include "layer_0_core/profiling/counters.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "io/addon_registration.hpp"
#include "io/cli.hpp"
#include "io/input/parser/factory.hpp"
#include "io/iterator/kohn_sham/planewave/factory_mpi.hpp"
#include "io/output/companion_outputs.hpp"
#include "layer_4_computation/flavors/self_consistent_field.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/mpi.hpp"
#include "utils/time.hpp"

using namespace q3;

// ─── CLI helpers ──────────────────────────────────────────────────────────────

// ─── Entry point ──────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    q3::addons::register_all();
    auto options = cli::parse_options(argc, argv, "q3_mpi");

    auto policy = std::make_unique<execution::PlanewaveMPIExecutionPolicy>();
    log::init_mpi(options.log_file);

    /**
     * Answered before the input is resolved, so `--version` works in a
     * directory holding no input -- which is where it is usually asked. MPI is
     * already up, so init_mpi() has silenced every rank but the root and the
     * line is printed once rather than once per rank; returning here runs the
     * policy's destructor, which finalizes MPI.
     */
    if (options.print_version) {
        cli::print_version("q3_mpi");
        return 0;
    }

    /**
     * Apply the requested log level only after init_mpi(): init replaces the
     * default logger, so applying it earlier would be silently discarded.
     */
    if (!options.log_level.empty()) {
        log::set_level(options.log_level);
    }

    /**
     * Capture the rank now, while MPI is guaranteed live and the policy is
     * guaranteed to still be here. The catch block below needs it to attribute
     * a fatal error, but by then the policy has been moved into the factory and
     * MPI may already be finalized -- and MPI_Comm_rank after MPI_Finalize is
     * itself undefined.
     */
    int world_rank_index = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_index);

    try {
        /**
         * Every rank reads it and applies its verbosity; a non-root rank's console
         * sink is already off, so that reaches only the rank's own log file.
         */
        auto input = cli::read_input(options, "q3_mpi");

        /**
         * An explicit CLI value overrides the input manifest. Otherwise honor
         * control.number_of_kpoint_pools; zero in either source selects the
         * documented one-pool default. Every rank parses the same input and
         * configures the topology before the first physics collective.
         */
        int requested_number_of_pools = options.number_of_kpoint_pools_was_set
                                            ? options.number_of_kpoint_pools
                                            : input.number_of_kpoint_pools;
        policy->configure_topology(requested_number_of_pools);

        /**
         * Switch the pool from splitting reciprocal space to splitting bands.
         *
         * This must happen before the basis is generated: it is what makes
         * band_size() report 1, which in turn keeps the full G-sphere on every
         * rank, selects the serial FFT grid, and leaves stress enabled, which
         * does not survive the reciprocal-space split.
         *
         * It also has to happen after configure_topology, which builds the
         * inter-pool communicator by splitting on the pool-local rank index.
         * That pairing -- rank i of every pool with rank i of the others -- is
         * what makes several pools work alongside distributed bands: within a
         * pool every rank ends up holding the same whole density, because the
         * bands were gathered back, so summing along the inter-pool axis gives
         * every rank the total over all k-points. The pairing was written for
         * the slab split, where rank i owns a matching slab in each pool; whole
         * grids are the degenerate case of that and reduce just as correctly.
         */
        if (options.distribute_bands) {
            policy->enable_band_distribution();
        }

        /**
         * Report the topology before the first physics collective, so a run that
         * stops later still says how its ranks were arranged. Without this the
         * log is silent about rank count, pool count and ranks per pool, which
         * makes "it hangs above N ranks" impossible to diagnose from output
         * alone. The pool count printed here is what was asked for; the factory
         * may still clamp it to the irreducible k-point count.
         */
        const char* pool_source =
            options.number_of_kpoint_pools_was_set
                ? "--pools"
                : (input.number_of_kpoint_pools != 0 ? "control.number_of_kpoint_pools"
                                                     : "default");
        /**
         * Report the true ranks per pool rather than band_size(). The two agree
         * for the reciprocal-space split, where band_size() *is* how many ranks
         * share a pool -- but enabling band distribution sets it to 1 on purpose,
         * to signal that each rank now holds whole coefficient vectors. Printing
         * it here would then claim one rank per pool on a run using four, which
         * is exactly the thing this line exists to make diagnosable.
         */
        int ranks_per_pool = policy->world_size() / std::max(policy->number_of_pools(), 1);
        log::info("MPI topology: {} rank(s), {} k-point pool(s), {} rank(s) per pool ({}), "
                  "splitting {} within a pool",
                  policy->world_size(),
                  policy->number_of_pools(),
                  ranks_per_pool,
                  pool_source,
                  options.distribute_bands ? "bands" : "reciprocal space");

        auto iterator = create_scf_iterator_mpi(input, std::move(policy));
        bool is_root = iterator->execution_policy().is_root();
        /**
         * Only the self-consistent field flavor can run here, and until now a
         * deck asking for anything else got one anyway: this line ran
         * regardless of input.calculation, so `q3_mpi nscf.yml` reported an SCF
         * under an NSCF's name. Refuse instead.
         *
         * The restriction is structural, not an oversight. This binary builds
         * its own iterator so it can inject the MPI execution policy, and
         * SelfConsistentFieldFlavor is the only flavor that accepts one; the
         * others call create_iterator() themselves and would get a serial
         * policy. Lifting it means a composition root that can hand any flavor
         * its execution policy, which is tracked as its own piece of work.
         */
        if (input.calculation != "self_consistent_field") {
            throw std::runtime_error(
                "q3_mpi runs self_consistent_field only, but this input asks for '" +
                input.calculation +
                "'. The MPI binary supplies its own execution policy, which today only the "
                "self-consistent field flavor accepts. Run this input with the serial q3 "
                "binary, which supports every flavor.");
        }

        auto flavor = std::make_unique<SelfConsistentFieldFlavor>(std::move(iterator));

        cli::print_header();
        log::info("     Calculation type: {}", input.calculation);

        auto flavor_result = flavor->run(input);
        if (is_root) {
            flavor_result.write_results_json(input.output_directory + "/" + input.prefix +
                                             "_results.json");
            write_companion_outputs(input, flavor_result);
        }

        if (is_root) {
            auto& profiling = q3::ProfilingCounters::instance();

            /**
             * Reduce counters across all ranks so the summary reflects the total
             * work done across all pools, not just rank 0's contribution.
             */
            long long global_fft_calls = profiling.fft_call_count.load();
            long long global_davidson_iterations = profiling.davidson_iteration_count.load();
            long long global_hamiltonian_applications =
                profiling.hamiltonian_application_count.load();
            MPI_Reduce(
                MPI_IN_PLACE, &global_fft_calls, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE,
                       &global_davidson_iterations,
                       1,
                       MPI_LONG_LONG,
                       MPI_SUM,
                       0,
                       MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE,
                       &global_hamiltonian_applications,
                       1,
                       MPI_LONG_LONG,
                       MPI_SUM,
                       0,
                       MPI_COMM_WORLD);

            q3::log::info("     Total FFT calls (all ranks):              {}", global_fft_calls);
            q3::log::info("     Total Davidson iterations (all ranks):    {}",
                          global_davidson_iterations);
            q3::log::info("     Total Hamiltonian applications (all ranks): {}",
                          global_hamiltonian_applications);

            /**
             * Memory is reported as a spread rather than a sum. Every rank
             * holds its own copy of the workspace, so the interesting figures
             * are the largest rank (what a node must fit) and the smallest
             * (whether the ranks are in fact symmetric, which under
             * --distribute-bands they are meant to be).
             */
            unsigned long long maximum_resident = profiling.peak_resident_bytes_now();
            unsigned long long minimum_resident = maximum_resident;
            MPI_Reduce(MPI_IN_PLACE,
                       &maximum_resident,
                       1,
                       MPI_UNSIGNED_LONG_LONG,
                       MPI_MAX,
                       0,
                       MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE,
                       &minimum_resident,
                       1,
                       MPI_UNSIGNED_LONG_LONG,
                       MPI_MIN,
                       0,
                       MPI_COMM_WORLD);
            if (maximum_resident > 0) {
                q3::log::info("     Peak resident memory per rank:            max {}, min {}",
                              q3::ResidentMemorySample::format_gibibytes(maximum_resident),
                              q3::ResidentMemorySample::format_gibibytes(minimum_resident));
            }
            profiling.davidson_timing.print_summary();
            cli::print_footer();
        } else {
            // Non-root ranks participate in the MPI_Reduce but do not print.
            long long local_fft_calls = q3::ProfilingCounters::instance().fft_call_count.load();
            long long local_davidson_iterations =
                q3::ProfilingCounters::instance().davidson_iteration_count.load();
            long long local_hamiltonian_applications =
                q3::ProfilingCounters::instance().hamiltonian_application_count.load();
            MPI_Reduce(&local_fft_calls, nullptr, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(
                &local_davidson_iterations, nullptr, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_hamiltonian_applications,
                       nullptr,
                       1,
                       MPI_LONG_LONG,
                       MPI_SUM,
                       0,
                       MPI_COMM_WORLD);
            unsigned long long local_resident =
                q3::ProfilingCounters::instance().peak_resident_bytes_now();
            MPI_Reduce(
                &local_resident, nullptr, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(
                &local_resident, nullptr, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        }

    } catch (const std::exception& exception) {
        /**
         * Report from whichever rank threw, on stderr, rather than through
         * log::error(): init_mpi() silences every non-root rank, so a failure
         * raised there would say nothing at all. That is not hypothetical --
         * the factory's shared-pool refusal is raised identically on every
         * rank, and the first to exit nonzero takes the whole job down with it,
         * including rank 0 part-way through printing. The run then exited
         * nonzero with no reason given, intermittently and only under load.
         * Every rank that reaches here now says so, flushed, before it goes.
         */
        log::fatal("[rank " + std::to_string(world_rank_index) +
                   "] Fatal error: " + exception.what());

        /**
         * Abort so that a rank failing alone cannot leave its peers blocked in
         * a collective forever. Reaching this at all depends on the execution
         * policy's destructor running no collective while unwinding: it is
         * destroyed before this handler on every throwing path -- inside
         * create_scf_iterator_mpi, which takes it by value, or with the
         * iterator that owns it afterwards -- and a barrier or MPI_Finalize
         * there would block until peers that are still doing physics arrived,
         * which for a rank-local failure is never. That is why the destructor
         * declines; see ~MPIExecutionPolicy().
         *
         * The guard remains because MPI_Abort after MPI_Finalize is undefined
         * -- in practice OpenMPI buries the message above under a wall of
         * standard-violation warnings and exits 14 instead of the 1 asked for
         * here -- and a finalized MPI means every rank reached the collective
         * MPI_Finalize, so none is left hanging and returning normally is safe.
         */
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);
        if (!is_finalized) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return 1;
    }

    /**
     * MPI_Finalize is called by PlanewaveMPIExecutionPolicy's destructor, which
     * finalizes on this path only: the policy was moved into the iterator, and
     * is destroyed when flavor goes out of scope, with no exception in flight.
     * A policy destroyed while unwinding tears nothing down and leaves MPI live
     * for the handler above.
     */
    return 0;
}
