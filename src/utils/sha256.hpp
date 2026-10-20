#pragma once

/**
 * SHA-256, from FIPS 180-4.
 *
 * Present because a pseudopotential's identity is the hash of the file it ships
 * as. The registry that consumes Q3's results already identifies a
 * pseudopotential by the SHA-256 of its contents, and adopting that scheme
 * verbatim rather than inventing a second one is a stated requirement -- two
 * identity schemes for the same file turn every cross-check into a mapping
 * exercise.
 *
 * Implemented rather than depended upon. The algorithm is a published standard
 * of about a hundred lines with official test vectors, so an implementation can
 * prove itself; a build dependency for it would have to be provisioned on every
 * runner. FIPS 180-4, Sec. 5.3.3 for the initial hash value, Sec. 4.2.2 for the
 * round constants, Sec. 6.2.2 for the compression function.
 *
 * Not a security primitive here and not offered as one: this identifies files,
 * it does not authenticate them.
 */

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace q3 {

/// Computes the SHA-256 digest of a byte sequence or a file.
class SHA256 {
  public:
    /// Lowercase hexadecimal digest of the given bytes.
    [[nodiscard]] static std::string hexadecimal_digest(const std::string& message) {
        const std::array<std::uint32_t, 8> state = digest(message);

        std::string hexadecimal;
        hexadecimal.reserve(64);
        for (std::uint32_t word : state) {
            char buffer[9];
            std::snprintf(buffer, sizeof(buffer), "%08x", word);
            hexadecimal += buffer;
        }
        return hexadecimal;
    }

    /**
     * Lowercase hexadecimal digest of a file's contents.
     *
     * Read as binary and hashed byte for byte, so the answer matches what any
     * other tool reports for the same file rather than what it reports for some
     * normalization of it.
     *
     * A read error is an error, not a short file. fread() returns a short count
     * for both end-of-file and failure, so a mid-file I/O error on an unchecked
     * loop yields a perfectly well-formed digest -- of a prefix. That is the
     * one outcome this class must never produce: a confidently wrong identity
     * is indistinguishable from a real one, which is the whole argument for
     * hashing contents in the first place.
     */
    [[nodiscard]] static std::string hexadecimal_digest_of_file(const std::string& path) {
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (file == nullptr) {
            throw std::runtime_error("SHA256: cannot open '" + path + "'");
        }

        std::string contents;
        char buffer[65536];
        std::size_t read = 0;
        try {
            while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
                contents.append(buffer, read);
            }
        } catch (...) {
            std::fclose(file);
            throw;
        }
        const bool read_failed = std::ferror(file) != 0;
        std::fclose(file);
        if (read_failed) {
            throw std::runtime_error("SHA256: error reading '" + path + "'");
        }
        return hexadecimal_digest(contents);
    }

  private:
    [[nodiscard]] static std::uint32_t rotate_right(std::uint32_t value, int places) {
        return (value >> places) | (value << (32 - places));
    }

    /// The eight-word state after compressing the padded message. FIPS 180-4 Sec. 6.2.2.
    [[nodiscard]] static std::array<std::uint32_t, 8> digest(const std::string& message) {
        std::array<std::uint32_t, 8> state = {0x6a09e667u,
                                              0xbb67ae85u,
                                              0x3c6ef372u,
                                              0xa54ff53au,
                                              0x510e527fu,
                                              0x9b05688cu,
                                              0x1f83d9abu,
                                              0x5be0cd19u};

        /**
         * Padding: a single 1 bit, then zeros, then the message length in bits
         * as a 64-bit big-endian integer, to a multiple of 64 bytes.
         */
        std::vector<unsigned char> padded(message.begin(), message.end());
        const std::uint64_t length_in_bits = static_cast<std::uint64_t>(message.size()) * 8;
        padded.push_back(0x80);
        while (padded.size() % 64 != 56) {
            padded.push_back(0x00);
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            padded.push_back(static_cast<unsigned char>((length_in_bits >> shift) & 0xffu));
        }

        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t block = 0; block < padded.size(); block += 64) {
            for (int index = 0; index < 16; index++) {
                const std::size_t offset = block + static_cast<std::size_t>(index) * 4;
                schedule[index] = (static_cast<std::uint32_t>(padded[offset]) << 24) |
                                  (static_cast<std::uint32_t>(padded[offset + 1]) << 16) |
                                  (static_cast<std::uint32_t>(padded[offset + 2]) << 8) |
                                  static_cast<std::uint32_t>(padded[offset + 3]);
            }
            for (int index = 16; index < 64; index++) {
                const std::uint32_t previous = schedule[index - 15];
                const std::uint32_t recent = schedule[index - 2];
                const std::uint32_t sigma_zero =
                    rotate_right(previous, 7) ^ rotate_right(previous, 18) ^ (previous >> 3);
                const std::uint32_t sigma_one =
                    rotate_right(recent, 17) ^ rotate_right(recent, 19) ^ (recent >> 10);
                schedule[index] =
                    schedule[index - 16] + sigma_zero + schedule[index - 7] + sigma_one;
            }

            std::array<std::uint32_t, 8> working = state;
            for (int index = 0; index < 64; index++) {
                const std::uint32_t capital_sigma_one = rotate_right(working[4], 6) ^
                                                        rotate_right(working[4], 11) ^
                                                        rotate_right(working[4], 25);
                const std::uint32_t choice = (working[4] & working[5]) ^ (~working[4] & working[6]);
                const std::uint32_t first = working[7] + capital_sigma_one + choice +
                                            round_constants()[index] + schedule[index];
                const std::uint32_t capital_sigma_zero = rotate_right(working[0], 2) ^
                                                         rotate_right(working[0], 13) ^
                                                         rotate_right(working[0], 22);
                const std::uint32_t majority = (working[0] & working[1]) ^
                                               (working[0] & working[2]) ^
                                               (working[1] & working[2]);
                const std::uint32_t second = capital_sigma_zero + majority;

                working[7] = working[6];
                working[6] = working[5];
                working[5] = working[4];
                working[4] = working[3] + first;
                working[3] = working[2];
                working[2] = working[1];
                working[1] = working[0];
                working[0] = first + second;
            }
            for (int index = 0; index < 8; index++) {
                state[index] += working[index];
            }
        }
        return state;
    }

    /// First 32 bits of the fractional parts of the cube roots of the first 64 primes.
    [[nodiscard]] static const std::array<std::uint32_t, 64>& round_constants() {
        static const std::array<std::uint32_t, 64> constants = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
            0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
            0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
            0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        return constants;
    }
};

} // namespace q3
