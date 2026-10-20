/**
 * Launcher task-count detection: the check that stops a GPU binary from
 * silently solving the same problem once per task (camsai/q3#15).
 *
 * The variables are read through a lookup rather than the real environment, so
 * these cases can be written down exactly rather than arranged with setenv.
 */

#include "io/launcher.hpp"

#include <map>
#include <string>

#include <gtest/gtest.h>

using namespace q3;

namespace {

/// A fake environment: names not in the map read as unset.
cli::EnvironmentLookup environment(const std::map<std::string, std::string>& variables) {
    return [variables](const char* name) -> const char* {
        auto entry = variables.find(name);
        return entry == variables.end() ? nullptr : entry->second.c_str();
    };
}

} // namespace

TEST(LauncherTasksTest, NoLauncherVariablesMeansOneTask) {
    const auto tasks = cli::detect_launcher_tasks(environment({}));
    EXPECT_EQ(tasks.count, 1);
    EXPECT_TRUE(tasks.variable.empty());
}

TEST(LauncherTasksTest, ReadsTheSlurmStepTaskCount) {
    const auto tasks = cli::detect_launcher_tasks(environment({{"SLURM_STEP_NUM_TASKS", "4"}}));
    EXPECT_EQ(tasks.count, 4);
    EXPECT_EQ(tasks.variable, "SLURM_STEP_NUM_TASKS");
}

TEST(LauncherTasksTest, ReadsTheMpiLauncherSizes) {
    EXPECT_EQ(cli::detect_launcher_tasks(environment({{"PMI_SIZE", "2"}})).count, 2);
    EXPECT_EQ(cli::detect_launcher_tasks(environment({{"OMPI_COMM_WORLD_SIZE", "8"}})).count, 8);
    EXPECT_EQ(cli::detect_launcher_tasks(environment({{"MV2_COMM_WORLD_SIZE", "3"}})).count, 3);
}

TEST(LauncherTasksTest, IgnoresTheAllocationWideSlurmVariables) {
    /**
     * `salloc -n 4` exports SLURM_NTASKS to the interactive shell, where a
     * program run without srun is still one process. Refusing there would be
     * wrong, so only step-scoped variables count.
     */
    const auto tasks =
        cli::detect_launcher_tasks(environment({{"SLURM_NTASKS", "4"}, {"SLURM_NPROCS", "4"}}));
    EXPECT_EQ(tasks.count, 1);
    EXPECT_TRUE(tasks.variable.empty());
}

TEST(LauncherTasksTest, TakesTheLargestCountAndNamesIt) {
    const auto tasks =
        cli::detect_launcher_tasks(environment({{"SLURM_STEP_NUM_TASKS", "2"}, {"PMI_SIZE", "6"}}));
    EXPECT_EQ(tasks.count, 6);
    EXPECT_EQ(tasks.variable, "PMI_SIZE");
}

TEST(LauncherTasksTest, OneTaskIsNotAMultiTaskLaunch) {
    const auto tasks =
        cli::detect_launcher_tasks(environment({{"SLURM_STEP_NUM_TASKS", "1"}, {"PMI_SIZE", "1"}}));
    EXPECT_EQ(tasks.count, 1);
    EXPECT_TRUE(tasks.variable.empty());
}

TEST(LauncherTasksTest, AVariableThatIsNotANumberSaysNothing) {
    const auto tasks = cli::detect_launcher_tasks(
        environment({{"SLURM_STEP_NUM_TASKS", ""}, {"PMI_SIZE", "not-a-number"}}));
    EXPECT_EQ(tasks.count, 1);
    EXPECT_TRUE(tasks.variable.empty());
}
