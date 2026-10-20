#pragma once

/**
 * Wannier90Exporter — an InterpolationRecord's Wannier payload as the four
 * Wannier90-format files (`<prefix>.mmn`, `.amn`, `.eig`, `.nnkp`).
 *
 * The one place that knows both the record's units and the formats'. The
 * `.nnkp` lattice blocks are Angstrom and inverse Angstrom while the record
 * carries bohr; the `.eig` is eV while the record holds Rydberg. The record
 * stays in Q3's units, the format writer stays ignorant of Q3's conventions,
 * and the conversion lives here.
 *
 * A record imported from Wannier90 files carries neither a neighbour table
 * nor overlaps (see importer.hpp); exporting it writes the `.amn`
 * and `.eig` that can be written and says so, rather than a `.nnkp` with an
 * empty nnkpts block or an empty `.mmn` that a consumer would take for real.
 */

#include <filesystem>
#include <stdexcept>
#include <string>

#include "io/interpolation/init.hpp"
#include "io/interpolation/wannier90/writer.hpp"
#include "io/logger.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

class Wannier90Exporter : public InterpolationExporter {
  public:
    void export_to_directory(const InterpolationRecord& record,
                             const std::string& directory,
                             const std::string& prefix) const override {
        if (record.method != "wannier") {
            throw std::invalid_argument("Wannier90Exporter: the record was built by the '" +
                                        record.method +
                                        "' method and carries no Wannier interface matrices");
        }
        Matrix3x3 real_lattice_angstrom;
        Matrix3x3 reciprocal_lattice_inverse_angstrom;
        for (int row = 0; row < 3; row++) {
            real_lattice_angstrom[row] =
                record.real_lattice_bohr[row] * constants::BOHR_RADIUS_ANGSTROM;
            reciprocal_lattice_inverse_angstrom[row] =
                record.reciprocal_lattice_inverse_bohr[row] * constants::ANGSTROM_TO_BOHR;
        }
        std::filesystem::create_directories(directory);
        const std::string seedname = directory + "/" + prefix;
        const WannierInputs& wannier = record.wannier;
        Wannier90FormatWriter::write_projections(seedname + ".amn", wannier);
        Wannier90FormatWriter::write_eigenvalues(seedname + ".eig", wannier);
        if (wannier.neighbor_directions.empty() || wannier.overlap_matrices.empty()) {
            log::warn("wrote {}.amn and .eig only: this record carries no neighbour table or "
                      "overlaps (it was imported from projections and eigenvalues), so no .nnkp "
                      "or .mmn is written -- the files it was imported from are those",
                      seedname);
            return;
        }
        Wannier90FormatWriter::write_neighbor_list(seedname + ".nnkp",
                                                   wannier,
                                                   real_lattice_angstrom,
                                                   reciprocal_lattice_inverse_angstrom);
        Wannier90FormatWriter::write_overlaps(seedname + ".mmn", wannier);
        log::info("wrote {}.mmn, .amn, .eig and .nnkp", seedname);
    }

    [[nodiscard]] std::string name() const override {
        return "wannier90";
    }
};

} // namespace q3
