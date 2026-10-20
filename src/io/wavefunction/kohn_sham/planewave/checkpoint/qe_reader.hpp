#pragma once

/**
 * QEPlanewaveKohnShamWavefunctionReader — read one `wfc<k>.dat` back.
 *
 * The counterpart of the writer in qe.hpp, and the only way to get a single
 * band's coefficients out of a finished run without rerunning it. The records
 * are the ones that writer emits: the k-point header, the dimensions, the
 * reciprocal lattice, the Miller indices, and then one record per band.
 *
 * The layout is Quantum ESPRESSO's, which is also what the Q3 writer emits:
 * a packed 44-byte k-point header (`ik, xk(3), ispin, gamma_only, scalef`,
 * with `xk` Cartesian in 1/bohr), reciprocal vectors in 1/bohr, and Miller
 * indices interleaved per G-vector, `mill(3, ngw)`. Files from Q3 builds that
 * predate that layout carried a 48-byte header with alignment padding and
 * component-blocked Miller indices; nothing later in such a file tells the two
 * index orders apart, so the header length is checked and an old file is
 * refused instead of being read into a plausible, wrong band.
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

class QEPlanewaveKohnShamWavefunctionReader {
  public:
    /// Read `path`, replacing whatever this reader held; throws on a malformed file.
    void read(const std::string& path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            throw std::runtime_error("Cannot read wavefunction file: " + path);
        path_ = path;

        const std::vector<char> header = read_record(stream);
        if (header.size() != kpoint_header_bytes) {
            throw_corrupt("its k-point header is " + std::to_string(header.size()) +
                          " bytes, not the packed " + std::to_string(kpoint_header_bytes) +
                          " (a file from a Q3 build that padded the header is not readable)");
        }
        std::memcpy(&kpoint_coordinates_.x, header.data() + 4, sizeof(double));
        std::memcpy(&kpoint_coordinates_.y, header.data() + 12, sizeof(double));
        std::memcpy(&kpoint_coordinates_.z, header.data() + 20, sizeof(double));
        std::memcpy(&is_gamma_only_flag_, header.data() + 32, sizeof(int32_t));

        const std::vector<char> dimensions = read_record(stream);
        if (dimensions.size() != 4 * sizeof(int32_t))
            throw_corrupt("its dimension record is not four integers");
        const auto* dimension_values = reinterpret_cast<const int32_t*>(dimensions.data());
        const int number_of_plane_waves = dimension_values[0];
        number_of_bands_ = dimension_values[3];
        if (number_of_plane_waves <= 0 || number_of_bands_ <= 0) {
            throw_corrupt("it declares " + std::to_string(number_of_plane_waves) +
                          " plane wave(s) and " + std::to_string(number_of_bands_) + " band(s)");
        }

        const std::vector<char> reciprocal = read_record(stream);
        if (reciprocal.size() != 9 * sizeof(double))
            throw_corrupt("its reciprocal-lattice record is not nine reals");
        const auto* vectors = reinterpret_cast<const double*>(reciprocal.data());
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                reciprocal_lattice_vectors_[row][column] = vectors[row * 3 + column];

        const std::vector<char> millers = read_record(stream);
        if (millers.size() != 3 * static_cast<std::size_t>(number_of_plane_waves) * sizeof(int32_t))
            throw_corrupt("its Miller-index record is the wrong length");
        const auto* miller_values = reinterpret_cast<const int32_t*>(millers.data());
        miller_indices_.resize(number_of_plane_waves);
        for (int index = 0; index < number_of_plane_waves; index++) {
            miller_indices_[index] = {miller_values[3 * index],
                                      miller_values[3 * index + 1],
                                      miller_values[3 * index + 2]};
        }

        band_coefficients_.assign(number_of_bands_, {});
        for (int band = 0; band < number_of_bands_; band++) {
            const std::vector<char> coefficients = read_record(stream);
            if (coefficients.size() !=
                static_cast<std::size_t>(number_of_plane_waves) * sizeof(Complex)) {
                throw_corrupt("the record for band " + std::to_string(band + 1) +
                              " is the wrong length");
            }
            const auto* values = reinterpret_cast<const Complex*>(coefficients.data());
            band_coefficients_[band].assign(values, values + number_of_plane_waves);
        }
    }

    [[nodiscard]] int number_of_bands() const {
        return number_of_bands_;
    }
    [[nodiscard]] int number_of_plane_waves() const {
        return static_cast<int>(miller_indices_.size());
    }
    [[nodiscard]] bool is_gamma_only() const {
        return is_gamma_only_flag_ != 0;
    }
    /// Cartesian coordinates, in 1/bohr, of the k-point this file belongs to.
    [[nodiscard]] const Vector3D& kpoint_coordinates() const {
        return kpoint_coordinates_;
    }
    [[nodiscard]] const Matrix3x3& reciprocal_lattice_vectors() const {
        return reciprocal_lattice_vectors_;
    }
    [[nodiscard]] const std::vector<IntegerVector3D>& miller_indices() const {
        return miller_indices_;
    }
    /// Coefficients of one band, parallel to `miller_indices()`.
    [[nodiscard]] const std::vector<Complex>& band_coefficients(int band_index) const {
        return band_coefficients_[band_index];
    }

  private:
    /// int32 + three doubles + two int32 + one double, packed as Fortran writes it.
    static constexpr std::size_t kpoint_header_bytes = 44;

    std::string path_;
    Vector3D kpoint_coordinates_;
    int32_t is_gamma_only_flag_ = 0;
    int number_of_bands_ = 0;
    Matrix3x3 reciprocal_lattice_vectors_{};
    std::vector<IntegerVector3D> miller_indices_;
    std::vector<std::vector<Complex>> band_coefficients_;

    [[noreturn]] void throw_corrupt(const std::string& reason) const {
        throw std::runtime_error("'" + path_ + "' is not a readable wavefunction file: " + reason +
                                 ".");
    }

    std::vector<char> read_record(std::ifstream& stream) const {
        int32_t prefix_length = 0;
        stream.read(reinterpret_cast<char*>(&prefix_length), sizeof(prefix_length));
        if (!stream || prefix_length < 0)
            throw_corrupt("it ends where a record marker was expected");
        std::vector<char> payload(static_cast<std::size_t>(prefix_length));
        stream.read(payload.data(), prefix_length);
        int32_t suffix_length = 0;
        stream.read(reinterpret_cast<char*>(&suffix_length), sizeof(suffix_length));
        if (!stream || suffix_length != prefix_length)
            throw_corrupt("a record's trailing length marker does not match its leading one");
        return payload;
    }
};

} // namespace q3
