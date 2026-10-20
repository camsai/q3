#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "layer_0_core/version.hpp"

/**
 * Helper to run a shell command and capture its exit code.
 *
 * Returns the exit code of the command. Uses std::system() internally.
 */
static int run_command_and_get_exit_code(const std::string& command) {
    int raw_status = std::system(command.c_str());
#ifdef _WIN32
    return raw_status;
#else
    if (WIFEXITED(raw_status)) {
        return WEXITSTATUS(raw_status);
    }
    return -1;
#endif
}

#ifndef Q3_BINARY_DIR
/**
 * Derive the project root directory from EXAMPLES_DIR.
 *
 * EXAMPLES_DIR is defined as "<project_root>/examples" by CMake.
 * This strips the trailing "/examples" to get the project root.
 */
static std::string get_project_root() {
    std::filesystem::path examples_path(EXAMPLES_DIR);
    return examples_path.parent_path().string();
}
#endif

/**
 * Get the path to the q3 binary.
 *
 * Q3_BINARY_DIR is the build directory CMake wrote the executables to, which is
 * where the binary is whether the build tree sits inside the source tree or on a
 * different filesystem entirely. The <project_root>/build fallback covers a test
 * binary compiled without the definition.
 */
static std::string get_q3_binary_path() {
#ifdef Q3_BINARY_DIR
    return std::string(Q3_BINARY_DIR) + "/q3";
#else
    return get_project_root() + "/build/q3";
#endif
}

// ── CLI smoke test: run q3 on an example input ────────────────────────

TEST(CLISmokeTest, BinaryExistsAndIsExecutable) {
    std::string binary_path = get_q3_binary_path();
    ASSERT_TRUE(std::filesystem::exists(binary_path)) << "q3 binary not found at: " << binary_path;
}

TEST(CLISmokeTest, RunSiNCExampleProducesResults) {
    std::string binary_path = get_q3_binary_path();
    std::string input_file =
        std::string(EXAMPLES_DIR) + "/total_energy--semiconductor--silicon/nc.yml";

    ASSERT_TRUE(std::filesystem::exists(binary_path)) << "q3 binary not found at: " << binary_path;
    ASSERT_TRUE(std::filesystem::exists(input_file)) << "Input file not found at: " << input_file;

    // Create a temporary directory for the working directory
    auto temporary_directory = std::filesystem::temp_directory_path() / "q3_cli_test";
    std::filesystem::create_directories(temporary_directory);

    /**
     * Build the command: run q3 from the temporary directory
     * Redirect stderr to /dev/null to avoid polluting test output
     */
    std::string command = "cd " + temporary_directory.string() + " && " + binary_path + " " +
                          input_file + " > /dev/null 2>&1";

    int exit_code = run_command_and_get_exit_code(command);
    EXPECT_EQ(exit_code, 0) << "q3 binary should exit with code 0";

    /**
     * The output file is <outdir>/<prefix>_results.json
     * From the nc.yml: outdir=./output, prefix=si_nc
     */
    std::filesystem::path results_file = temporary_directory / "output" / "si_nc_results.json";
    ASSERT_TRUE(std::filesystem::exists(results_file))
        << "Results JSON not found at: " << results_file.string();

    // Verify the JSON is valid and contains expected fields
    std::ifstream file_stream(results_file);
    ASSERT_TRUE(file_stream.is_open()) << "Cannot open results JSON";

    nlohmann::json results;
    ASSERT_NO_THROW(results = nlohmann::json::parse(file_stream))
        << "Results file is not valid JSON";

    ASSERT_TRUE(results.contains("total_energy_ry"))
        << "Results JSON must contain 'total_energy_ry' field";

    double total_energy_value = results["total_energy_ry"].get<double>();
    EXPECT_TRUE(std::isfinite(total_energy_value)) << "total_energy_ry must be a finite number";

    // Sanity: total energy for Si should be negative
    EXPECT_LT(total_energy_value, 0.0) << "Total energy for Si should be negative (bound system)";

    ASSERT_TRUE(results.contains("converged")) << "Results JSON must contain 'converged' field";
    ASSERT_TRUE(results["converged"].get<bool>()) << "Si NC SCF should converge";

    // Clean up temporary directory
    std::filesystem::remove_all(temporary_directory);
}

TEST(CLISmokeTest, NoArgumentsReturnsNonZeroExitCode) {
    std::string binary_path = get_q3_binary_path();

    if (!std::filesystem::exists(binary_path)) {
        GTEST_SKIP() << "q3 binary not found, skipping";
    }

    std::string command = binary_path + " > /dev/null 2>&1";
    int exit_code = run_command_and_get_exit_code(command);
    EXPECT_NE(exit_code, 0) << "q3 binary with no arguments should exit with non-zero code";
}

/**
 * `--version` answers without an input, and says which build answered.
 *
 * Two properties, and the first is why the flag exists at all: the question is
 * asked when a result is in doubt, and by scripts recording what produced one,
 * so it must not depend on standing in a directory that holds a valid deck.
 * Run from a temporary directory with no input in it, where every other
 * invocation of this binary exits non-zero.
 */
TEST(CLISmokeTest, VersionIsReportedWithoutAnInput) {
    std::string binary_path = get_q3_binary_path();
    if (!std::filesystem::exists(binary_path)) {
        GTEST_SKIP() << "q3 binary not found, skipping";
    }

    std::filesystem::path empty_directory =
        std::filesystem::temp_directory_path() / "q3_version_no_input";
    std::filesystem::remove_all(empty_directory);
    std::filesystem::create_directories(empty_directory);

    std::filesystem::path output_path = empty_directory / "version.txt";
    std::string command = "cd " + empty_directory.string() + " && " + binary_path +
                          " --version > " + output_path.string() + " 2>&1";
    int exit_code = run_command_and_get_exit_code(command);
    EXPECT_EQ(exit_code, 0) << "--version must succeed where no input exists";

    std::ifstream output_stream(output_path);
    std::string reported((std::istreambuf_iterator<char>(output_stream)),
                         std::istreambuf_iterator<char>());
    EXPECT_NE(reported.find("q3"), std::string::npos) << reported;
    EXPECT_NE(reported.find(q3::version), std::string::npos)
        << "the reported version must be the one the build carries: " << reported;

    std::filesystem::remove_all(empty_directory);
}

/**
 * `--log-file` writes the log to the named path as well as the console.
 *
 * Checked on the `--check` path rather than a full run: the property under
 * test is that the sink is created and written, and that costs an SCF nowhere.
 */
TEST(CLISmokeTest, LogFileReceivesTheLog) {
    std::string binary_path = get_q3_binary_path();
    if (!std::filesystem::exists(binary_path)) {
        GTEST_SKIP() << "q3 binary not found, skipping";
    }

    std::filesystem::path working_directory =
        std::filesystem::temp_directory_path() / "q3_log_file";
    std::filesystem::remove_all(working_directory);
    std::filesystem::create_directories(working_directory);
    std::filesystem::path log_path = working_directory / "run.log";

    std::string input_path =
        std::string(EXAMPLES_DIR) + "/total_energy--semiconductor--silicon/nc.yml";
    std::string command = binary_path + " " + input_path + " --check --log-file " +
                          log_path.string() + " > /dev/null 2>&1";
    run_command_and_get_exit_code(command);

    ASSERT_TRUE(std::filesystem::exists(log_path)) << "--log-file did not create " << log_path;
    EXPECT_GT(std::filesystem::file_size(log_path), 0u) << "the log file was created but empty";

    std::filesystem::remove_all(working_directory);
}
