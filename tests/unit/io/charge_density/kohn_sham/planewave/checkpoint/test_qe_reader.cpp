#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/charge_density/kohn_sham/planewave/checkpoint/qe_reader.hpp"

using namespace q3;

namespace {

/// One Fortran unformatted sequential record: [length][payload][length].
void write_record(std::ofstream& stream, const void* data, int32_t number_of_bytes) {
    stream.write(reinterpret_cast<const char*>(&number_of_bytes), 4);
    stream.write(reinterpret_cast<const char*>(data), number_of_bytes);
    stream.write(reinterpret_cast<const char*>(&number_of_bytes), 4);
}

const std::vector<std::array<int32_t, 3>> millers = {{0, 0, 0}, {1, 0, 0}, {-1, 2, -3}};
const std::vector<Complex> density = {
    Complex(1.0, 0.0), Complex(0.25, -0.5), Complex(-0.125, 0.75)};

/**
 * A file in Q3's own dialect: no reciprocal-lattice record, and the Miller
 * indices stored as every first component, then every second, then every
 * third.
 */
std::filesystem::path write_q3_dialect(const std::string& name, int32_t gamma_only = 0) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream stream(path, std::ios::binary);
    const int32_t header[3] = {gamma_only, static_cast<int32_t>(millers.size()), 1};
    write_record(stream, header, sizeof(header));

    std::vector<int32_t> planar(3 * millers.size());
    for (std::size_t index = 0; index < millers.size(); index++) {
        for (int component = 0; component < 3; component++)
            planar[component * millers.size() + index] = millers[index][component];
    }
    write_record(stream, planar.data(), static_cast<int32_t>(planar.size() * sizeof(int32_t)));
    write_record(stream, density.data(), static_cast<int32_t>(density.size() * sizeof(Complex)));
    return path;
}

/**
 * A file in Quantum ESPRESSO's dialect: the reciprocal-lattice record is
 * present, and the Miller indices interleave each G-vector's three components.
 */
std::filesystem::path write_qe_dialect(const std::string& name, int number_of_spin_components) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream stream(path, std::ios::binary);
    const int32_t header[3] = {
        0, static_cast<int32_t>(millers.size()), static_cast<int32_t>(number_of_spin_components)};
    write_record(stream, header, sizeof(header));

    const double reciprocal[9] = {1.5, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 3.5};
    write_record(stream, reciprocal, sizeof(reciprocal));

    std::vector<int32_t> interleaved(3 * millers.size());
    for (std::size_t index = 0; index < millers.size(); index++) {
        for (int component = 0; component < 3; component++)
            interleaved[3 * index + component] = millers[index][component];
    }
    write_record(
        stream, interleaved.data(), static_cast<int32_t>(interleaved.size() * sizeof(int32_t)));
    for (int spin = 0; spin < number_of_spin_components; spin++) {
        std::vector<Complex> channel = density;
        for (Complex& value : channel)
            value *= static_cast<Real>(spin + 1);
        write_record(
            stream, channel.data(), static_cast<int32_t>(channel.size() * sizeof(Complex)));
    }
    return path;
}

} // namespace

TEST(QEChargeDensityReader, ReadsQ3sOwnDialect) {
    const auto path = write_q3_dialect("q3_test_rho_q3.dat");
    QEPlanewaveKohnShamChargeDensityReader reader;
    reader.read(path.string());

    EXPECT_FALSE(reader.has_reciprocal_lattice_vectors());
    EXPECT_FALSE(reader.is_gamma_only());
    EXPECT_EQ(reader.number_of_spin_components(), 1);
    ASSERT_EQ(reader.number_of_g_vectors(), 3);
    for (int index = 0; index < 3; index++) {
        EXPECT_EQ(reader.miller_indices()[index].x, millers[index][0]);
        EXPECT_EQ(reader.miller_indices()[index].y, millers[index][1]);
        EXPECT_EQ(reader.miller_indices()[index].z, millers[index][2]);
    }
    ASSERT_EQ(reader.charge_density_g().size(), 1u);
    for (int index = 0; index < 3; index++)
        EXPECT_EQ(reader.charge_density_g()[0][index], density[index]);

    std::filesystem::remove(path);
}

TEST(QEChargeDensityReader, ReadsQuantumEspressosDialectIncludingItsCell) {
    const auto path = write_qe_dialect("q3_test_rho_qe.dat", 1);
    QEPlanewaveKohnShamChargeDensityReader reader;
    reader.read(path.string());

    ASSERT_TRUE(reader.has_reciprocal_lattice_vectors());
    EXPECT_DOUBLE_EQ(reader.reciprocal_lattice_vectors()[0][0], 1.5);
    EXPECT_DOUBLE_EQ(reader.reciprocal_lattice_vectors()[1][1], 2.5);
    EXPECT_DOUBLE_EQ(reader.reciprocal_lattice_vectors()[2][2], 3.5);

    /**
     * The two dialects store the indices transposed, so reading this file with
     * the other layout would return (0, 1, -1) as the first G-vector. It is the
     * only place the two files differ once the cell record is accounted for.
     */
    ASSERT_EQ(reader.number_of_g_vectors(), 3);
    EXPECT_EQ(reader.miller_indices()[2].x, -1);
    EXPECT_EQ(reader.miller_indices()[2].y, 2);
    EXPECT_EQ(reader.miller_indices()[2].z, -3);

    std::filesystem::remove(path);
}

TEST(QEChargeDensityReader, ReadsEverySpinChannel) {
    const auto path = write_qe_dialect("q3_test_rho_spin.dat", 2);
    QEPlanewaveKohnShamChargeDensityReader reader;
    reader.read(path.string());

    EXPECT_EQ(reader.number_of_spin_components(), 2);
    ASSERT_EQ(reader.charge_density_g().size(), 2u);
    EXPECT_EQ(reader.charge_density_g()[1][1], density[1] * 2.0);

    std::filesystem::remove(path);
}

TEST(QEChargeDensityReader, CarriesTheGammaOnlyFlagThroughToTheCaller) {
    const auto path = write_q3_dialect("q3_test_rho_gamma.dat", /*gamma_only=*/1);
    QEPlanewaveKohnShamChargeDensityReader reader;
    reader.read(path.string());
    /**
     * A gamma-only file stores a half sphere, so whoever transforms it has to
     * fill -G by conjugation. Losing this flag halves the density silently.
     */
    EXPECT_TRUE(reader.is_gamma_only());
    std::filesystem::remove(path);
}

TEST(QEChargeDensityReader, RefusesATruncatedFile) {
    const auto path = write_q3_dialect("q3_test_rho_truncated.dat");
    const auto truncated = std::filesystem::temp_directory_path() / "q3_test_rho_cut.dat";
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 40);
    std::filesystem::rename(path, truncated);

    QEPlanewaveKohnShamChargeDensityReader reader;
    EXPECT_THROW(reader.read(truncated.string()), std::runtime_error);
    std::filesystem::remove(truncated);
}

TEST(QEChargeDensityReader, RefusesAMissingFile) {
    const auto path = std::filesystem::temp_directory_path() / "q3_test_rho_absent.dat";
    std::filesystem::remove(path);
    QEPlanewaveKohnShamChargeDensityReader reader;
    EXPECT_THROW(reader.read(path.string()), std::runtime_error);
}
