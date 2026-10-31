#pragma once

/**
 * BerkeleyGWSelfConsistentPotentialFile — writes BerkeleyGW's `VSC` file: the
 * local Kohn-Sham potential, V_loc + V_H + V_xc, from which ParaBands builds
 * the local part of the Hamiltonian it diagonalizes.
 *
 * Layout: BerkeleyGWFieldFile with the title "VSC-Complex" -- ParaBands reads
 * it with BerkeleyGW's RHO/VXC routines (`MeanField/ParaBands/parabands.f90`),
 * one field per spin, Rydberg, unscaled. ParaBands adds V_sc(G − G') to each
 * matrix element (`ham_build_column_loc` in `MeanField/ParaBands/hamiltonian.f90`),
 * so the convention needed is the plane-wave matrix element of the potential:
 * with Q3's transform, whose G = 0 component is the cell average, that is the
 * record's `self_consistent_potential_g` as it stands.
 */

#include <fstream>
#include <stdexcept>
#include <string>

#include "io/logger.hpp"
#include "io/mean_field_export/berkeleygw/field.hpp"
#include "io/mean_field_record/init.hpp"

namespace q3 {

/// Writes `VSC` from a mean-field record; see the file comment.
class BerkeleyGWSelfConsistentPotentialFile {
  public:
    explicit BerkeleyGWSelfConsistentPotentialFile(const MeanFieldRecord& record)
        : record_(record) {}
    /// See BerkeleyGWChargeDensityFile: this holds a reference.
    explicit BerkeleyGWSelfConsistentPotentialFile(const MeanFieldRecord&&) = delete;

    /// Writes the file, or throws -- before creating it -- when the record carries no potential.
    void write_to_file(const std::string& file_path) const {
        if (record_.self_consistent_potential_g.empty()) {
            throw std::runtime_error(
                "the mean-field record carries no self-consistent potential (a record read "
                "back from mean-field.q3mf does not store it; export from the run itself)");
        }

        std::ofstream output_file(file_path, std::ios::binary);
        if (!output_file.is_open()) {
            throw std::runtime_error("cannot open " + file_path + " for writing");
        }
        BerkeleyGWFieldFile(record_, "VSC-Complex", record_.self_consistent_potential_g, 1.0)
            .write(output_file);

        log::info("     Wrote BerkeleyGW self-consistent potential file: {}", file_path);
    }

  private:
    const MeanFieldRecord& record_;
};

} // namespace q3
