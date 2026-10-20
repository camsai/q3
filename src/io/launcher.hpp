#pragma once

/**
 * How many tasks the job launcher started this process as, read from the
 * environment rather than from MPI.
 *
 * The GPU binaries link no MPI -- the published -cuda image has none at all --
 * so they cannot ask MPI_Comm_size how many of them are running. They can ask
 * the launcher, which says so in the environment: srun sets
 * SLURM_STEP_NUM_TASKS for the step it starts, and the MPI launchers set a
 * size variable of their own. That is enough to tell one task from four, which
 * is all a binary with no multi-process mode needs to know.
 *
 * SLURM_NTASKS is deliberately not consulted. `salloc -n 4` exports it to the
 * interactive shell, where a program run directly is still one process, and
 * refusing there would be wrong. SLURM_STEP_NUM_TASKS exists only inside a step
 * srun actually launched.
 */

#include <cstdlib>
#include <functional>
#include <string>

namespace q3 {
namespace cli {

/// What the launcher said, and which variable said it.
struct LauncherTasks {
    /// Tasks in this job step; 1 when no launcher variable was found.
    int count = 1;
    /// The variable the count came from, empty when the count is the default.
    std::string variable;
};

/// Environment reader, so the detection can be tested without setenv.
using EnvironmentLookup = std::function<const char*(const char*)>;

/**
 * Read the task count from `lookup`, taking the largest value any launcher
 * variable reports and remembering which one it was -- a message that names the
 * variable can be checked by the reader, and a wrong guess can be argued with.
 */
inline LauncherTasks detect_launcher_tasks(const EnvironmentLookup& lookup) {
    static constexpr const char* task_count_variables[] = {
        "SLURM_STEP_NUM_TASKS", // srun, for the step rather than the allocation
        "PMI_SIZE",             // MPICH and Cray MPICH, including under srun
        "OMPI_COMM_WORLD_SIZE", // Open MPI's mpirun
        "MV2_COMM_WORLD_SIZE",  // MVAPICH2
    };

    LauncherTasks tasks;
    for (const char* variable : task_count_variables) {
        const char* value = lookup(variable);
        if (value == nullptr || *value == '\0') {
            continue;
        }
        int reported = 0;
        try {
            reported = std::stoi(value);
        } catch (const std::exception&) {
            continue; // a variable that does not hold a number says nothing
        }
        if (reported > tasks.count) {
            tasks.count = reported;
            tasks.variable = variable;
        }
    }
    return tasks;
}

/// The same, against the real environment.
inline LauncherTasks detect_launcher_tasks() {
    return detect_launcher_tasks([](const char* variable) { return std::getenv(variable); });
}

} // namespace cli
} // namespace q3
