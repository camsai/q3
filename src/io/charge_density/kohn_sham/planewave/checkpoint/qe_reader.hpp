#pragma once

/**
 * QEPlanewaveKohnShamChargeDensityReader — read a `charge-density.dat` back.
 *
 * The counterpart of the writer in qe.hpp. A converged density is the one
 * thing a plane-wave run saves that every downstream question needs, and until
 * now nothing could read it: a density Quantum ESPRESSO produced could not be
 * brought into Q3, and a density Q3 wrote could only be looked at by rerunning
 * the calculation that produced it.
 *
 * Both dialects of the file are read, and they differ in two places.
 *
 * **The reciprocal-lattice record.** Quantum ESPRESSO writes
 * `gamma_only, ngm, nspin`, then `b1, b2, b3`, then the Miller indices, then
 * one rho(G) record per spin channel (the format is described in its
 * `Modules/io_base.f90`; cited, not transcribed — see PROVENANCE.md). The
 * writer in qe.hpp omits the reciprocal-lattice record. Which dialect a file
 * is in is therefore decided by the length of the second record: nine doubles
 * is the cell, anything else is already the Miller indices.
 *
 * **The Miller index layout.** Quantum ESPRESSO's array is `mill(3, ngm)`, so
 * the record interleaves each G-vector's three components; Q3's writer emits
 * the transpose, `mill(ngm, 3)` — every first component, then every second,
 * then every third. There is no way to tell those apart from the record's
 * length, since both hold `3 * ngm` integers, so the dialect established by
 * the record above decides this too. That coupling is the reason the detection
 * is stated here rather than buried: a file with a reciprocal-lattice record
 * and Q3's index layout, or the reverse, would be read silently and wrongly.
 * Neither producer emits one.
 *
 * A gamma-only file holds the half sphere {G : G·ẑ ≥ 0 by the engine's
 * ordering}; the caller reconstructs -G by conjugation, which is what
 * `real_space_field_from_reciprocal()` does.
 */

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

class QEPlanewaveKohnShamChargeDensityReader {
  public:
    /// Read `path`, replacing whatever this reader held; throws on a malformed file.
    void read(const std::string& path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            throw std::runtime_error("Cannot read charge density file: " + path);
        path_ = path;

        std::vector<char> header = read_record(stream);
        if (header.size() != 3 * sizeof(int32_t))
            throw_corrupt("its first record is not the three-integer header");
        const auto* header_values = reinterpret_cast<const int32_t*>(header.data());
        is_gamma_only_ = header_values[0] != 0;
        const int number_of_g_vectors = header_values[1];
        number_of_spin_components_ = header_values[2];
        if (number_of_g_vectors <= 0 || number_of_spin_components_ <= 0 ||
            number_of_spin_components_ > 2) {
            throw_corrupt("its header declares " + std::to_string(number_of_g_vectors) +
                          " G-vector(s) and " + std::to_string(number_of_spin_components_) +
                          " spin component(s)");
        }

        std::vector<char> second = read_record(stream);
        has_reciprocal_lattice_vectors_ = second.size() == 9 * sizeof(double);
        if (has_reciprocal_lattice_vectors_) {
            const auto* vectors = reinterpret_cast<const double*>(second.data());
            for (int row = 0; row < 3; row++)
                for (int column = 0; column < 3; column++)
                    reciprocal_lattice_vectors_[row][column] = vectors[row * 3 + column];
            second = read_record(stream);
        }

        if (second.size() != 3 * static_cast<std::size_t>(number_of_g_vectors) * sizeof(int32_t))
            throw_corrupt("its Miller-index record is the wrong length");
        const auto* miller_values = reinterpret_cast<const int32_t*>(second.data());
        miller_indices_.resize(number_of_g_vectors);
        for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
            if (has_reciprocal_lattice_vectors_) {
                miller_indices_[g_index] = {miller_values[3 * g_index],
                                            miller_values[3 * g_index + 1],
                                            miller_values[3 * g_index + 2]};
            } else {
                miller_indices_[g_index] = {miller_values[g_index],
                                            miller_values[number_of_g_vectors + g_index],
                                            miller_values[2 * number_of_g_vectors + g_index]};
            }
        }

        charge_density_g_.assign(number_of_spin_components_, {});
        for (int spin = 0; spin < number_of_spin_components_; spin++) {
            std::vector<char> density = read_record(stream);
            if (density.size() != static_cast<std::size_t>(number_of_g_vectors) * sizeof(Complex))
                throw_corrupt("its rho(G) record for spin channel " + std::to_string(spin + 1) +
                              " is the wrong length");
            const auto* values = reinterpret_cast<const Complex*>(density.data());
            charge_density_g_[spin].assign(values, values + number_of_g_vectors);
        }
    }

    [[nodiscard]] bool is_gamma_only() const {
        return is_gamma_only_;
    }
    [[nodiscard]] int number_of_spin_components() const {
        return number_of_spin_components_;
    }
    [[nodiscard]] int number_of_g_vectors() const {
        return static_cast<int>(miller_indices_.size());
    }
    [[nodiscard]] const std::vector<IntegerVector3D>& miller_indices() const {
        return miller_indices_;
    }
    /// rho(G) per spin channel, each parallel to `miller_indices()`.
    [[nodiscard]] const std::vector<std::vector<Complex>>& charge_density_g() const {
        return charge_density_g_;
    }
    /// True only for a Quantum ESPRESSO file, which carries the cell; see the header.
    [[nodiscard]] bool has_reciprocal_lattice_vectors() const {
        return has_reciprocal_lattice_vectors_;
    }
    /// Rows b1, b2, b3 in 1/bohr; meaningful only when the flag above is set.
    [[nodiscard]] const Matrix3x3& reciprocal_lattice_vectors() const {
        return reciprocal_lattice_vectors_;
    }

  private:
    std::string path_;
    bool is_gamma_only_ = false;
    int number_of_spin_components_ = 0;
    bool has_reciprocal_lattice_vectors_ = false;
    Matrix3x3 reciprocal_lattice_vectors_{};
    std::vector<IntegerVector3D> miller_indices_;
    std::vector<std::vector<Complex>> charge_density_g_;

    [[noreturn]] void throw_corrupt(const std::string& reason) const {
        throw std::runtime_error("'" + path_ +
                                 "' is not a readable charge-density file: " + reason + ".");
    }

    /**
     * One Fortran unformatted sequential record, whose payload length the
     * caller does not know: the leading marker is what says which dialect the
     * file is in, so it is read rather than asserted.
     */
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
