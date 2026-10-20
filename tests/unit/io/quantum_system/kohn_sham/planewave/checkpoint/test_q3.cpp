#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"

using namespace q3;

/**
 * ── The Q3-native checkpoint ────────────────────────────────────────────
 *
 * `effective-potential.q3cp` is what a non-self-consistent run continues
 * from. Version 3 adds the PAW on-site occupations, without which a PAW run
 * that diagonalizes before building a density has no one-centre term in D;
 * a version-2 file must still read, with those occupations empty, so the
 * flavor that needs them can say so rather than read past the end.
 */

namespace {

KohnShamQuantumSystemSnapshot two_spin_snapshot() {
    KohnShamQuantumSystemSnapshot snapshot;
    snapshot.iteration = 7;
    snapshot.number_of_spin_components = 2;
    snapshot.number_of_bands = 3;
    snapshot.number_of_kpoint_states = 2;
    snapshot.total_energy = -15.5;
    snapshot.converged = true;
    snapshot.charge_density_real = {{0.1, 0.2, 0.3, 0.4}, {0.4, 0.3, 0.2, 0.1}};
    snapshot.eigenvalues.assign(2, 3, 0.0);
    for (int kpoint_state = 0; kpoint_state < 2; kpoint_state++) {
        for (int band = 0; band < 3; band++) {
            snapshot.eigenvalues[kpoint_state][band] = -0.5 + 0.1 * (kpoint_state + band);
        }
    }
    snapshot.effective_potential_real = {{1.0, 2.0, 3.0, 4.0}, {4.0, 3.0, 2.0, 1.0}};
    return snapshot;
}

std::string temporary_path(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

} // namespace

TEST(Q3PlanewaveKohnShamQuantumSystemCheckpoint, CarriesThePawOnSiteOccupations) {
    KohnShamQuantumSystemSnapshot written = two_spin_snapshot();
    // Two spin channels, two atoms, each row a flattened two-channel matrix.
    written.one_center_occupations.assign(2, {});
    for (int spin = 0; spin < 2; spin++) {
        written.one_center_occupations[spin].assign(2, 4, 0.0);
        for (int atom = 0; atom < 2; atom++) {
            for (int entry = 0; entry < 4; entry++) {
                written.one_center_occupations[spin][atom][entry] =
                    0.25 * (spin + 1) + 0.01 * (atom * 4 + entry);
            }
        }
    }
    const std::string path = temporary_path("q3-checkpoint-on-site-occupations.q3cp");
    Q3PlanewaveKohnShamQuantumSystemCheckpoint checkpoint;
    ASSERT_TRUE(checkpoint.save(path, written));

    KohnShamQuantumSystemSnapshot read;
    ASSERT_TRUE(checkpoint.load(path, read));
    std::filesystem::remove(path);

    ASSERT_EQ(read.one_center_occupations.size(), 2u);
    for (int spin = 0; spin < 2; spin++) {
        EXPECT_EQ(read.one_center_occupations[spin].rows(), 2u);
        EXPECT_EQ(read.one_center_occupations[spin].cols(), 4u);
        EXPECT_EQ(read.one_center_occupations[spin].raw(),
                  written.one_center_occupations[spin].raw());
    }
    EXPECT_EQ(read.charge_density_real, written.charge_density_real);
    EXPECT_EQ(read.effective_potential_real, written.effective_potential_real);
    EXPECT_EQ(read.eigenvalues.raw(), written.eigenvalues.raw());
}

TEST(Q3PlanewaveKohnShamQuantumSystemCheckpoint, AVersionTwoFileReadsWithoutOccupations) {
    /**
     * Written as version 3 with no occupations, the section is a single zero
     * count at the end of the file; dropping those bytes and stamping version 2
     * leaves exactly what an older binary wrote.
     */
    const KohnShamQuantumSystemSnapshot written = two_spin_snapshot();
    const std::string path = temporary_path("q3-checkpoint-version-two.q3cp");
    Q3PlanewaveKohnShamQuantumSystemCheckpoint checkpoint;
    ASSERT_TRUE(checkpoint.save(path, written));
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - sizeof(int));
    {
        std::fstream stream(path, std::ios::in | std::ios::out | std::ios::binary);
        stream.seekp(sizeof(uint32_t));
        const uint32_t version_two = 2;
        stream.write(reinterpret_cast<const char*>(&version_two), sizeof(version_two));
    }

    KohnShamQuantumSystemSnapshot read;
    ASSERT_TRUE(checkpoint.load(path, read));
    std::filesystem::remove(path);

    EXPECT_TRUE(read.one_center_occupations.empty());
    EXPECT_EQ(read.effective_potential_real, written.effective_potential_real);
}
