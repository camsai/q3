#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/mean_field_export/berkeleygw/exchange_correlation_potential.hpp"
#include "io/mean_field_record/init.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

namespace {

MeanFieldRecord synthetic_record(int number_of_spin_components) {
    MeanFieldRecord record;
    record.lattice_parameter = 10.0;
    record.two_pi_over_lattice_parameter = 2.0 * 3.141592653589793 / 10.0;
    record.cell_volume = 1000.0;
    record.lattice_vectors[0] = Vector3D(1.0, 0.0, 0.0);
    record.lattice_vectors[1] = Vector3D(0.0, 1.0, 0.0);
    record.lattice_vectors[2] = Vector3D(0.0, 0.0, 1.0);
    record.reciprocal_lattice_vectors = record.lattice_vectors;
    record.atom_positions_cartesian = {Vector3D(0.0, 0.0, 0.0)};
    record.atomic_numbers = {14};
    record.fft_grid_dimensions = {4, 4, 4};
    record.charge_density_miller_indices = {
        IntegerVector3D(0, 0, 0), IntegerVector3D(1, 0, 0), IntegerVector3D(-1, 0, 0)};
    record.charge_density_energy_cutoff = 40.0;
    record.number_of_spin_components = number_of_spin_components;
    return record;
}

/// The last Fortran record of a file: a field file's data block.
std::vector<Complex> last_record_values(const std::string& path) {
    std::ifstream input_file(path, std::ios::binary);
    std::vector<char> payload;
    int32_t length = 0;
    while (input_file.read(reinterpret_cast<char*>(&length), 4)) {
        payload.resize(length);
        input_file.read(payload.data(), length);
        input_file.read(reinterpret_cast<char*>(&length), 4);
    }
    std::vector<Complex> values(payload.size() / sizeof(Complex));
    std::memcpy(values.data(), payload.data(), payload.size());
    return values;
}

std::string title_of(const std::string& path) {
    std::ifstream input_file(path, std::ios::binary);
    input_file.seekg(4);
    std::string title(32, ' ');
    input_file.read(title.data(), 32);
    return title.substr(0, title.find_last_not_of(' ') + 1);
}

} // namespace

TEST(BerkeleyGWExchangeCorrelationPotentialFile, WritesThePotentialUnscaled) {
    MeanFieldRecord record = synthetic_record(1);
    record.exchange_correlation_potential_g = {-0.6, 0.01, 0.01};
    std::string path = test_support::make_temporary_path("_vxc");
    BerkeleyGWExchangeCorrelationPotentialFile(record).write_to_file(path);
    EXPECT_EQ(title_of(path), "VXC-Complex");
    EXPECT_EQ(last_record_values(path), record.exchange_correlation_potential_g);
}

TEST(BerkeleyGWExchangeCorrelationPotentialFile, RefusesASpinPolarizedRecord) {
    MeanFieldRecord record = synthetic_record(2);
    record.exchange_correlation_potential_g = {-0.6, 0.01, 0.01};
    EXPECT_THROW(BerkeleyGWExchangeCorrelationPotentialFile(record).write_to_file(
                     test_support::make_temporary_path("_vxc")),
                 std::runtime_error);
}
