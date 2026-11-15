#pragma once

/**
 * StageTimer: wall time of a named stage, added to ProfilingCounters on scope exit.
 *
 * The Davidson accumulators say where the eigensolver spends its time, and nothing said
 * where the rest of a run went: on 128 ranks the setup and the SCF work outside Davidson
 * were 45 of 60 seconds and had no timer at all. A stage timer is one line at the top of
 * a scope, and its total reaches runtime_log.timings_ms under the name it was given, so a
 * scaling study reads it from the results file instead of a debug log.
 *
 * Names are dotted by phase, "setup.<stage>" or "scf.<stage>". Stages may nest (a parent
 * and its parts are both recorded), so the values are not a partition of the run. Timers
 * are started and stopped on the calling thread only; do not open one inside a parallel
 * region.
 */

#include <chrono>
#include <string>

#include "layer_0_core/profiling/counters.hpp"

namespace q3 {

/// Times a named stage from construction to destruction (see the file comment).
class StageTimer {
  public:
    explicit StageTimer(std::string name)
        : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;
    StageTimer(StageTimer&&) = delete;
    StageTimer& operator=(StageTimer&&) = delete;

    ~StageTimer() {
        const auto elapsed = std::chrono::steady_clock::now() - start_;
        ProfilingCounters::instance().add_stage_time(
            name_, std::chrono::duration<double, std::milli>(elapsed).count());
    }

  private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace q3
