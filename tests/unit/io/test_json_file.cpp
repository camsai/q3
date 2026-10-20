#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "io/json_file.hpp"

using namespace q3;

namespace {

std::string contents_of(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

} // namespace

TEST(WriteJsonFile, CreatesTheParentDirectoriesAndWritesFourSpaceIndentsWithATrailingNewline) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "q3_test_write_json_file";
    std::filesystem::remove_all(directory);
    const std::filesystem::path path = directory / "nested" / "results.json";

    write_json_file(path.string(), nlohmann::json{{"calculation", "interpolation"}});

    EXPECT_EQ(contents_of(path), "{\n    \"calculation\": \"interpolation\"\n}\n");
    std::filesystem::remove_all(directory);
}

TEST(WriteJsonFile, APathThatCannotBeOpenedIsRefusedByName) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "q3_test_write_json_file_refused";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    try {
        /// A directory cannot be opened as a file.
        write_json_file(directory.string(), nlohmann::json::object());
        FAIL() << "a directory is not a file to write";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(directory.string()), std::string::npos)
            << error.what();
    }
    std::filesystem::remove_all(directory);
}
