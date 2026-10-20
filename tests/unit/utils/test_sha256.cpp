#include "utils/sha256.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "utils/temporary_path.hpp"

using namespace q3;

/**
 * The published test vectors, which are the whole reason an implementation of a
 * standard can be trusted without a dependency. FIPS 180-4 Appendix B.1 and B.2
 * give the one- and two-block messages; the empty-string digest is the
 * universally quoted third.
 */
TEST(SHA256, ReproducesPublishedTestVectors) {
    EXPECT_EQ(SHA256::hexadecimal_digest(""),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(SHA256::hexadecimal_digest("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(
        SHA256::hexadecimal_digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

/// A message longer than one 64-byte block, exercising the multi-block loop.
TEST(SHA256, HashesAMessageSpanningManyBlocks) {
    const std::string message(1000, 'a');
    EXPECT_EQ(SHA256::hexadecimal_digest(message),
              "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

/// Every digest is 64 lowercase hexadecimal characters, which the consumers pattern-match on.
TEST(SHA256, DigestIsSixtyFourLowercaseHexadecimalCharacters) {
    const std::string digest = SHA256::hexadecimal_digest("q3");
    ASSERT_EQ(digest.size(), 64u);
    for (char character : digest) {
        const bool is_lowercase_hexadecimal =
            (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
        EXPECT_TRUE(is_lowercase_hexadecimal) << "unexpected character: " << character;
    }
}

/**
 * Hashing a file must agree with hashing the same bytes in memory, including
 * for contents that are not text: a pseudopotential is read as bytes, and an
 * implementation that stopped at the first null would hash a prefix of the file
 * while reporting the identity of the whole.
 */
TEST(SHA256, FileDigestMatchesTheDigestOfItsBytes) {
    const std::string file_path = q3::test_support::make_temporary_path("_payload.bin");

    const std::string contents("silicon\n\0\1binary bytes", 21);
    {
        std::ofstream file(file_path, std::ios::binary);
        file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    }

    EXPECT_EQ(SHA256::hexadecimal_digest_of_file(file_path), SHA256::hexadecimal_digest(contents));
    std::filesystem::remove(file_path);
}

/// A missing file is an error, not an empty digest that would register as a real answer.
TEST(SHA256, ThrowsWhenTheFileCannotBeOpened) {
    EXPECT_THROW(SHA256::hexadecimal_digest_of_file("/nonexistent/q3/pseudopotential.upf"),
                 std::runtime_error);
}
