#pragma once

/**
 * Centralized profiling counters for Q3.
 *
 * Encapsulates all global profiling state (call counts, timing accumulators)
 * into a singleton to avoid scattered mutable globals across headers.
 */

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <string>

#include "io/logger.hpp"
#include "layer_0_core/profiling/memory.hpp"

namespace q3 {

struct ProfilingCounters {
    std::atomic<long long> fft_call_count{0};
    std::atomic<long long> davidson_iteration_count{0};
    std::atomic<long long> hamiltonian_application_count{0};

    /**
     * High-water mark of this rank's resident set, in bytes.
     *
     * Sampled at the phase boundaries that allocate, so that the reported peak
     * covers the setup spike and the first Davidson restart rather than
     * whichever moment the run happened to end on.
     */
    std::atomic<std::uint64_t> peak_resident_bytes{0};

    // Davidson phase timing accumulators (milliseconds)
    struct DavidsonTimingAccumulator {
        double seed_time_ms = 0;
        double ritz_vector_time_ms = 0;
        double residual_time_ms = 0;
        double orthogonalize_time_ms = 0;
        double hamiltonian_application_time_ms = 0;
        double subspace_update_time_ms = 0;
        double collapse_time_ms = 0;
        double diagonalize_time_ms = 0;
        int total_solve_calls = 0;

        // H|ψ⟩ sub-component timing (accumulated across all calls)
        double hamiltonian_fft_time_ms = 0;
        double hamiltonian_kinetic_time_ms = 0;
        double hamiltonian_nonlocal_time_ms = 0;
        /// The projector overlaps <beta|psi> of the nonlocal term, a part of nonlocal above.
        double hamiltonian_projector_overlap_time_ms = 0;

        void print_summary() const {
            log::debug("     Davidson timing breakdown:");
            log::debug("       seed:                     {:8.1f} ms", seed_time_ms);
            log::debug("       ritz_vectors:             {:8.1f} ms", ritz_vector_time_ms);
            log::debug("       residuals:                {:8.1f} ms", residual_time_ms);
            log::debug("       orthogonalize:            {:8.1f} ms", orthogonalize_time_ms);
            log::debug("       hamiltonian_application:  {:8.1f} ms",
                       hamiltonian_application_time_ms);
            log::debug("         fft:                    {:8.1f} ms", hamiltonian_fft_time_ms);
            log::debug("         kinetic:                {:8.1f} ms", hamiltonian_kinetic_time_ms);
            log::debug("         nonlocal:               {:8.1f} ms", hamiltonian_nonlocal_time_ms);
            log::debug("           projector overlap:    {:8.1f} ms",
                       hamiltonian_projector_overlap_time_ms);
            log::debug("       subspace_update:          {:8.1f} ms", subspace_update_time_ms);
            log::debug("       diagonalize:              {:8.1f} ms", diagonalize_time_ms);
            log::debug("       collapse:                 {:8.1f} ms", collapse_time_ms);
            log::debug("       TOTAL:                    {:8.1f} ms", total_time_ms());
        }

        void reset() {
            *this = DavidsonTimingAccumulator{};
        }

        /**
         * The same breakdown print_summary() logs, as data.
         *
         * Caveat for anyone doing arithmetic on these: the four
         * hamiltonian_* sub-component timers are NOT a partition of
         * hamiltonian_application. They are accumulated in the applicator
         * (hamiltonian_applicator/serial.hpp) while the parent is accumulated
         * around the call site in davidson.hpp, and the sub-components are
         * reached from paths the parent timer does not wrap -- a 2-atom
         * silicon run reports 72.9 ms of hamiltonian_fft against 52.2 ms of
         * hamiltonian_application. Use them to compare like with like across
         * configurations, not to decompose the parent.
         *
         * Kept beside the printer so the two cannot drift: a stage added to one
         * and forgotten in the other is the failure mode this guards against.
         * Returns a flat map of stage name to milliseconds; the caller decides
         * where it goes.
         */
        [[nodiscard]] std::map<std::string, double> as_map() const {
            return {{"seed", seed_time_ms},
                    {"ritz_vectors", ritz_vector_time_ms},
                    {"residuals", residual_time_ms},
                    {"orthogonalize", orthogonalize_time_ms},
                    {"hamiltonian_application", hamiltonian_application_time_ms},
                    {"hamiltonian_fft", hamiltonian_fft_time_ms},
                    {"hamiltonian_kinetic", hamiltonian_kinetic_time_ms},
                    {"hamiltonian_nonlocal", hamiltonian_nonlocal_time_ms},
                    {"hamiltonian_projector_overlap", hamiltonian_projector_overlap_time_ms},
                    {"subspace_update", subspace_update_time_ms},
                    {"diagonalize", diagonalize_time_ms},
                    {"collapse", collapse_time_ms},
                    {"total", total_time_ms()}};
        }

        /**
         * The same sum print_summary() reports as TOTAL. The Hamiltonian
         * sub-components (fft, kinetic, nonlocal, allreduce) are deliberately
         * excluded: they partition hamiltonian_application, so adding them
         * again would double-count it.
         */
        [[nodiscard]] double total_time_ms() const {
            return seed_time_ms + ritz_vector_time_ms + residual_time_ms + orthogonalize_time_ms +
                   hamiltonian_application_time_ms + subspace_update_time_ms + collapse_time_ms +
                   diagonalize_time_ms;
        }
    };

    DavidsonTimingAccumulator davidson_timing{};

    /**
     * Take a resident-memory reading and keep it if it is the largest so far.
     *
     * Returns the sample so a caller that wants to log the current value does
     * not have to read the platform source twice.
     */
    ResidentMemorySample record_resident_memory() {
        ResidentMemorySample sample = ResidentMemorySample::sample();
        if (!sample.available) {
            return sample;
        }
        std::uint64_t observed = std::max(sample.current_bytes, sample.peak_bytes);
        std::uint64_t previous = peak_resident_bytes.load(std::memory_order_relaxed);
        while (observed > previous && !peak_resident_bytes.compare_exchange_weak(
                                          previous, observed, std::memory_order_relaxed)) {}
        return sample;
    }

    /**
     * Record a reading and log it against a phase label.
     *
     * One line per allocating phase, at info, so that a run's own output says
     * where its footprint came from. Phases that allocate nothing are not
     * worth a probe; phases inside the Davidson loop log at debug instead,
     * because there are as many of them as there are iterations.
     */
    void log_resident_checkpoint(const std::string& label) {
        ResidentMemorySample sample = record_resident_memory();
        if (!sample.available) {
            return;
        }
        log::info("     memory after {:<22}= {} (peak {})",
                  label,
                  ResidentMemorySample::format_gibibytes(sample.current_bytes),
                  ResidentMemorySample::format_gibibytes(sample.peak_bytes));
    }

    /**
     * Return freed heap to the operating system and record the reading after
     * it, at debug level. Placed at phase boundaries: the peak recorded before
     * the release stands, the current figure afterwards is what the next
     * phase builds on.
     */
    void release_unused_heap(const std::string& label) {
        std::uint64_t released = q3::release_unused_heap();
        ResidentMemorySample sample = record_resident_memory();
        if (!sample.available) {
            return;
        }
        log::debug("     memory released after {:<19}= {} (now {}, peak {})",
                   label,
                   ResidentMemorySample::format_gibibytes(released),
                   ResidentMemorySample::format_gibibytes(sample.current_bytes),
                   ResidentMemorySample::format_gibibytes(sample.peak_bytes));
    }

    /**
     * The same reading at debug level, for probes that fire per SCF iteration
     * or per Davidson restart. The console sink is pinned at info, so these
     * reach a --log-file and nothing else.
     */
    void log_resident_checkpoint_debug(const std::string& label) {
        ResidentMemorySample sample = record_resident_memory();
        if (!sample.available) {
            return;
        }
        log::debug("     memory after {:<22}= {} (peak {})",
                   label,
                   ResidentMemorySample::format_gibibytes(sample.current_bytes),
                   ResidentMemorySample::format_gibibytes(sample.peak_bytes));
    }

    /**
     * The peak so far, after one more reading. The counter only advances when
     * something samples it, and the kernel's high-water mark is folded in at
     * each sample, so a reader that wants the run's true peak -- rather than
     * the peak as of the last probe, which may predate the forces and stress
     * -- takes a fresh sample first. This is what the end-of-run reports use.
     */
    std::uint64_t peak_resident_bytes_now() {
        record_resident_memory();
        return peak_resident_bytes.load(std::memory_order_relaxed);
    }

    /// Print the peak resident set, omitted when no platform source could be read.
    void print_memory_summary() {
        std::uint64_t peak = peak_resident_bytes_now();
        if (peak == 0) {
            return;
        }
        log::info("     Peak resident memory:           {}",
                  ResidentMemorySample::format_gibibytes(peak));
    }

    static ProfilingCounters& instance() {
        static ProfilingCounters counters;
        return counters;
    }

    /**
     * Reset all profiling counters to zero.
     *
     * Must be called by the nanobind module between successive in-process
     * invocations (Mode B) to prevent counter accumulation across calls.
     * Subprocess runs (Mode A) each get a fresh process, so reset() is
     * unnecessary there.
     */
    void reset() {
        fft_call_count.store(0, std::memory_order_relaxed);
        davidson_iteration_count.store(0, std::memory_order_relaxed);
        hamiltonian_application_count.store(0, std::memory_order_relaxed);
        peak_resident_bytes.store(0, std::memory_order_relaxed);
        davidson_timing.reset();
    }

  private:
    ProfilingCounters() = default;
};

} // namespace q3
