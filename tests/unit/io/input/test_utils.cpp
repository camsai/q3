#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "io/input/utils.hpp"

using namespace q3;

TEST(ResolvePath, AbsolutePathReturnedUnchanged) {
    std::string absolute_path = "/usr/local/share/file.json";
    std::string result = detail::resolve_path("/some/base", absolute_path);
    EXPECT_EQ(result, absolute_path);
}

TEST(ResolvePath, RelativePathResolvedToBase) {
    std::string base_directory = "/home/user/data";
    std::string relative_path = "subdir/file.txt";
    std::string result = detail::resolve_path(base_directory, relative_path);
    std::string expected = "/home/user/data/subdir/file.txt";
    EXPECT_EQ(result, expected);
}

TEST(ResolvePath, SimpleFilename) {
    std::string base_directory = "/opt/calculations";
    std::string filename = "input.yaml";
    std::string result = detail::resolve_path(base_directory, filename);
    std::string expected = "/opt/calculations/input.yaml";
    EXPECT_EQ(result, expected);
}

TEST(ResolvePath, EmptyBaseWithRelativePath) {
    std::string result = detail::resolve_path("", "file.txt");
    EXPECT_EQ(result, "file.txt");
}

TEST(ReadJsonFile, ValidJsonFile) {
    // Create a temporary JSON file
    std::string temporary_path =
        std::filesystem::temp_directory_path().string() + "/q3_test_utils.json";
    {
        std::ofstream output_stream(temporary_path);
        output_stream << R"({"key": "value", "number": 42})";
    }
    nlohmann::json parsed = detail::read_json_file(temporary_path);
    EXPECT_EQ(parsed["key"], "value");
    EXPECT_EQ(parsed["number"], 42);
    std::filesystem::remove(temporary_path);
}

TEST(ReadJsonFile, MissingFileThrows) {
    EXPECT_THROW(detail::read_json_file("/nonexistent/path/file.json"), std::runtime_error);
}

TEST(ReadJsonFile, NestedJsonFile) {
    std::string temporary_path =
        std::filesystem::temp_directory_path().string() + "/q3_test_nested.json";
    {
        std::ofstream output_stream(temporary_path);
        output_stream << R"({"outer": {"inner": [1, 2, 3]}})";
    }
    nlohmann::json parsed = detail::read_json_file(temporary_path);
    ASSERT_TRUE(parsed.contains("outer"));
    ASSERT_TRUE(parsed["outer"].contains("inner"));
    EXPECT_EQ(parsed["outer"]["inner"].size(), 3u);
    EXPECT_EQ(parsed["outer"]["inner"][0], 1);
    std::filesystem::remove(temporary_path);
}
