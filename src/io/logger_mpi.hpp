#pragma once

/**
 * MPI-aware logger initialization.
 *
 * Include this ONLY in main.cpp (or the MPI entry point).
 * All other source files use "io/logger.hpp" (the serial baseline).
 *
 * Usage:
 *   #include "io/logger_mpi.hpp"
 *   q3::log::init_mpi();  // silences all ranks except rank 0
 */

#include "io/logger.hpp"
#include <mpi.h>

namespace q3 {
namespace log {

/**
 * Initialize the logger for MPI runs.
 *
 * Calls the serial init() then silences stdout on all non-zero ranks.
 * Each rank still writes to its own log file if log_file is provided.
 *
 * @param log_file Optional base path for per-rank file sinks.
 *                 Rank 0 writes to log_file; rank N writes to log_file.N.
 */
inline void init_mpi(const std::string& log_file = "") {
    int rank_index = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_index);

    // Build per-rank file path for non-zero ranks
    std::string effective_log_file = log_file;
    if (!log_file.empty() && rank_index != 0) {
        effective_log_file = log_file + "." + std::to_string(rank_index);
    }

    init(effective_log_file);

    /**
     * Silence console output on all non-root ranks via the logger wrapper —
     * no direct spdlog access outside io/logger.hpp.
     */
    if (rank_index != 0) {
        silence();
    }
}

} // namespace log
} // namespace q3
