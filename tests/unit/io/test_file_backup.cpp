#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "io/file_backup.hpp"

using namespace q3;

namespace {

std::string contents_of(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path);
    stream << text;
}

/// An empty directory under the system temporary directory, removed when the test ends.
class FileBackupTest : public ::testing::Test {
  protected:
    void SetUp() override {
        directory_ = std::filesystem::temp_directory_path() /
                     ("q3_test_file_backup_" +
                      std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        std::filesystem::remove_all(directory_);
        std::filesystem::create_directories(directory_);
    }
    void TearDown() override {
        std::filesystem::remove_all(directory_);
    }
    std::filesystem::path directory_;
};

} // namespace

TEST_F(FileBackupTest, BackupNamesAppendADashAndTheNumberToTheWholeName) {
    const std::filesystem::path path = directory_ / "si_results.json";
    EXPECT_EQ(backup_path(path, 1), directory_ / "si_results.json-1");
    EXPECT_EQ(backup_path(path, 12), directory_ / "si_results.json-12");
}

TEST_F(FileBackupTest, NothingIsBackedUpWhenThereIsNoFile) {
    const std::filesystem::path path = directory_ / "si_results.json";
    EXPECT_FALSE(back_up_existing_file(path).has_value());
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
}

TEST_F(FileBackupTest, EachBackupTakesTheNextNumberAndKeepsItsContents) {
    const std::filesystem::path path = directory_ / "si_results.json";

    write_text(path, "first");
    EXPECT_EQ(back_up_existing_file(path), directory_ / "si_results.json-1");
    write_text(path, "second");
    EXPECT_EQ(back_up_existing_file(path), directory_ / "si_results.json-2");

    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_EQ(contents_of(directory_ / "si_results.json-1"), "first");
    EXPECT_EQ(contents_of(directory_ / "si_results.json-2"), "second");
}

TEST_F(FileBackupTest, TheLowestUnusedNumberIsTaken) {
    const std::filesystem::path path = directory_ / "si_results.json";
    write_text(directory_ / "si_results.json-1", "");
    write_text(directory_ / "si_results.json-3", "");
    write_text(path, "latest");

    EXPECT_EQ(back_up_existing_file(path), directory_ / "si_results.json-2");
}
