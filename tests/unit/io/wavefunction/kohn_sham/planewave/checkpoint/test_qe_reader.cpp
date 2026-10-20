#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/wavefunction/kohn_sham/planewave/checkpoint/qe_reader.hpp"

using namespace q3;

namespace {

void write_record(std::ofstream& stream, const void* data, int32_t number_of_bytes) {
    stream.write(reinterpret_cast<const char*>(&number_of_bytes), 4);
    stream.write(reinterpret_cast<const char*>(data), number_of_bytes);
    stream.write(reinterpret_cast<const char*>(&number_of_bytes), 4);
}

const std::vector<std::array<int32_t, 3>> millers = {{0, 0, 0}, {1, -1, 0}, {0, 2, -1}};
const std::vector<std::vector<Complex>> bands = {
    {Complex(0.5, 0.0), Complex(0.1, 0.2), Complex(-0.3, 0.4)},
    {Complex(0.0, 1.0), Complex(-0.2, 0.0), Complex(0.6, -0.1)}};

/**
 * A `wfc<k>.dat` in Quantum ESPRESSO's layout, which the Q3 writer emits: a
 * packed 44-byte k-point header and interleaved Miller indices. A 48-byte
 * header is the padded one older Q3 builds wrote.
 */
std::filesystem::path write_q3_wavefunction(const std::string& name,
                                            std::size_t header_bytes = 44) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream stream(path, std::ios::binary);

    std::vector<char> header(header_bytes, 0);
    const int32_t kpoint_index = 1;
    const double coordinates[3] = {0.25, -0.5, 0.125};
    const int32_t spin_index = 1;
    const int32_t gamma_only = 0;
    const double scale_factor = 1.0;
    std::memcpy(header.data() + 0, &kpoint_index, sizeof(kpoint_index));
    if (header_bytes == 44) {
        std::memcpy(header.data() + 4, coordinates, sizeof(coordinates));
        std::memcpy(header.data() + 28, &spin_index, sizeof(spin_index));
        std::memcpy(header.data() + 32, &gamma_only, sizeof(gamma_only));
        std::memcpy(header.data() + 36, &scale_factor, sizeof(scale_factor));
    }
    write_record(stream, header.data(), static_cast<int32_t>(header.size()));

    const int32_t dimensions[4] = {static_cast<int32_t>(millers.size()),
                                   static_cast<int32_t>(millers.size()),
                                   1,
                                   static_cast<int32_t>(bands.size())};
    write_record(stream, dimensions, sizeof(dimensions));

    const double reciprocal[9] = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
    write_record(stream, reciprocal, sizeof(reciprocal));

    std::vector<int32_t> interleaved(3 * millers.size());
    for (std::size_t index = 0; index < millers.size(); index++)
        for (int component = 0; component < 3; component++)
            interleaved[3 * index + component] = millers[index][component];
    write_record(
        stream, interleaved.data(), static_cast<int32_t>(interleaved.size() * sizeof(int32_t)));

    for (const auto& band : bands)
        write_record(stream, band.data(), static_cast<int32_t>(band.size() * sizeof(Complex)));
    return path;
}

} // namespace

TEST(QEWavefunctionReader, ReadsTheHeaderTheGSetAndEveryBand) {
    const auto path = write_q3_wavefunction("q3_test_wfc.dat");
    QEPlanewaveKohnShamWavefunctionReader reader;
    reader.read(path.string());

    EXPECT_EQ(reader.number_of_bands(), 2);
    EXPECT_EQ(reader.number_of_plane_waves(), 3);
    EXPECT_FALSE(reader.is_gamma_only());
    EXPECT_DOUBLE_EQ(reader.kpoint_coordinates().x, 0.25);
    EXPECT_DOUBLE_EQ(reader.kpoint_coordinates().y, -0.5);
    EXPECT_DOUBLE_EQ(reader.kpoint_coordinates().z, 0.125);
    EXPECT_DOUBLE_EQ(reader.reciprocal_lattice_vectors()[2][2], 3.0);

    ASSERT_EQ(reader.miller_indices().size(), 3u);
    EXPECT_EQ(reader.miller_indices()[1].x, 1);
    EXPECT_EQ(reader.miller_indices()[1].y, -1);
    EXPECT_EQ(reader.miller_indices()[2].z, -1);

    for (int band = 0; band < 2; band++)
        for (int index = 0; index < 3; index++)
            EXPECT_EQ(reader.band_coefficients(band)[index], bands[band][index]);

    std::filesystem::remove(path);
}

TEST(QEWavefunctionReader, RefusesThePaddedHeaderOfOlderQ3Builds) {
    /**
     * A 48-byte header is the padded one older Q3 builds wrote, whose Miller
     * indices are stored transposed. Nothing later in the file distinguishes
     * the two, so reading on regardless would return a plausible, wrong band
     * rather than an error.
     */
    const auto path = write_q3_wavefunction("q3_test_wfc_padded.dat", /*header_bytes=*/48);
    QEPlanewaveKohnShamWavefunctionReader reader;
    EXPECT_THROW(reader.read(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(QEWavefunctionReader, RefusesATruncatedFile) {
    const auto path = write_q3_wavefunction("q3_test_wfc_truncated.dat");
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 24);
    QEPlanewaveKohnShamWavefunctionReader reader;
    EXPECT_THROW(reader.read(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(QEWavefunctionReader, RefusesAMissingFile) {
    const auto path = std::filesystem::temp_directory_path() / "q3_test_wfc_absent.dat";
    std::filesystem::remove(path);
    QEPlanewaveKohnShamWavefunctionReader reader;
    EXPECT_THROW(reader.read(path.string()), std::runtime_error);
}
