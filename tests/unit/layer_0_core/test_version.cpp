#include "layer_0_core/version.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <string>

using namespace q3;

/// The version is the release identity and must always say something.
TEST(Version, VersionStringIsNotEmpty) {
    EXPECT_FALSE(std::string(version).empty());
}

/**
 * The engine's version and the Python package's version are one release
 * identity written twice, and two version strings that drift apart are worse
 * than one: a results file would name a build the wheel beside it is not.
 */
TEST(Version, MatchesTheVersionDeclaredInPyprojectToml) {
    std::ifstream pyproject(std::string(EXAMPLES_DIR) + "/../pyproject.toml");
    ASSERT_TRUE(pyproject.is_open()) << "cannot open pyproject.toml";

    std::string declared_version;
    std::string line;
    while (std::getline(pyproject, line)) {
        const std::size_t equals_position = line.find('=');
        if (equals_position == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, equals_position);
        if (key.find("version") == std::string::npos) {
            continue;
        }
        const std::size_t first_quote = line.find('"', equals_position);
        const std::size_t last_quote = line.rfind('"');
        if (first_quote == std::string::npos || last_quote <= first_quote) {
            continue;
        }
        declared_version = line.substr(first_quote + 1, last_quote - first_quote - 1);
        break;
    }

    ASSERT_FALSE(declared_version.empty()) << "no version found in pyproject.toml";
    EXPECT_EQ(std::string(version), declared_version);
}

/**
 * The commit is supplied by the build or it is empty; it is never guessed.
 * When the build did supply one it must look like a git object name, because a
 * consumer pattern-matches it.
 */
TEST(Version, CommitIsEitherEmptyOrHexadecimal) {
    const std::string commit_string(commit);
    if (commit_string.empty()) {
        SUCCEED() << "built without -DQ3_GIT_COMMIT, which is the honest default";
        return;
    }
    EXPECT_GE(commit_string.size(), 7u);
    for (char character : commit_string) {
        const bool is_lowercase_hexadecimal =
            (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
        EXPECT_TRUE(is_lowercase_hexadecimal) << "unexpected character: " << character;
    }
}
