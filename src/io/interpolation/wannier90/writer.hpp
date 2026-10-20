#pragma once

/**
 * Wannier90FormatWriter — writes a WannierInputs out as the four
 * Wannier90 interface files (`.mmn`, `.amn`, `.eig`, `.nnkp`).
 *
 * This is the half that makes Q3's own overlaps and projections usable: the
 * interface kernels compute M_mn(k,b) and A_mn(k) in memory, the localization
 * engine reads them from disk, and without a writer the two never meet.
 * Emitting the Wannier90 formats rather than only a private one is deliberate
 * -- the same files can be handed to `wannier90.x` itself, which is how Q3's
 * matrices get checked against an independent implementation rather than only
 * against Q3's own reader.
 *
 * The reader in reader.hpp was written from the user guide
 * before this writer existed, and neither consults the other. A round trip
 * through both therefore tests something real: a shared misreading of the
 * format cannot cancel out between a writer and a reader derived from it,
 * because this pair was not derived that way.
 *
 * Conventions, all of them the file format's rather than Q3's:
 *
 * - Indices are written 1-based (Fortran origin).
 * - A `.mmn` block's num_bands^2 elements go out with the FIRST index m
 *   varying fastest, which is the order WannierInputs already stores
 *   them in, so blocks stream straight out with no transpose.
 * - `.eig` is in ELECTRON VOLTS; the record holds Rydberg, and this converts.
 * - `.nnkp` lattice blocks are Angstrom (real) and inverse Angstrom
 *   (reciprocal); the caller passes them in those units.
 *
 * The per-(k,b) neighbour identities the `.mmn` and `.nnkp` need are not
 * stored in the record -- it holds the b-vector shell, which is the same for
 * every k-point -- so they are recomputed here with
 * KPointNeighborShells::resolve_neighbor, the same exact-integer routine the
 * overlap kernel used when it built those blocks.
 */

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"

namespace q3 {

class Wannier90FormatWriter {
  public:
    /// eV per Rydberg; the record is Rydberg, `.eig` is eV.
    static constexpr Real electron_volts_per_rydberg = constants::RY_TO_EV;

    /**
     * Write all four files for `seedname`, e.g. `si` -> `si.mmn`, `si.amn`,
     * `si.eig`, `si.nnkp`.
     *
     * `real_lattice` rows are a1, a2, a3 in Angstrom and `reciprocal_lattice`
     * rows b1, b2, b3 in inverse Angstrom -- the units the `.nnkp` format
     * specifies, which are not Q3's internal atomic units, so the caller
     * converts rather than this guessing.
     */
    static void write_all(const std::string& seedname,
                          const WannierInputs& record,
                          const Matrix3x3& real_lattice,
                          const Matrix3x3& reciprocal_lattice) {
        write_overlaps(seedname + ".mmn", record);
        write_projections(seedname + ".amn", record);
        write_eigenvalues(seedname + ".eig", record);
        write_neighbor_list(seedname + ".nnkp", record, real_lattice, reciprocal_lattice);
    }

    static void write_overlaps(const std::string& file_path, const WannierInputs& record) {
        std::ofstream stream = open(file_path);
        const int number_of_kpoints = record.number_of_kpoints();
        const int number_of_directions = static_cast<int>(record.neighbor_directions.size());
        require_overlaps_present(record);

        stream << "Written by Q3\n";
        stream << " " << record.number_of_bands << " " << number_of_kpoints << " "
               << number_of_directions << "\n";

        std::vector<KPoint> kpoints = build_kpoints(record);
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            for (int direction_index = 0; direction_index < number_of_directions;
                 direction_index++) {
                ResolvedKPointNeighbor resolved = KPointNeighborShells::resolve_neighbor(
                    kpoint_index,
                    record.neighbor_directions[direction_index].grid_step,
                    record.kpoint_grid_dimensions,
                    kpoints);
                stream << " " << kpoint_index + 1 << " " << resolved.neighbor_kpoint_index + 1
                       << " " << resolved.periodic_image_shift.x << " "
                       << resolved.periodic_image_shift.y << " " << resolved.periodic_image_shift.z
                       << "\n";

                const std::vector<Complex>& block =
                    record.overlap_matrices[kpoint_index][direction_index];
                for (const Complex& value : block) {
                    write_complex(stream, value);
                }
            }
        }
    }

    static void write_projections(const std::string& file_path, const WannierInputs& record) {
        std::ofstream stream = open(file_path);
        const int number_of_kpoints = record.number_of_kpoints();
        if (static_cast<int>(record.projection_matrices.size()) != number_of_kpoints) {
            throw std::invalid_argument(
                "Wannier90FormatWriter::write_projections: the record holds no projections");
        }

        stream << "Written by Q3\n";
        stream << " " << record.number_of_bands << " " << number_of_kpoints << " "
               << record.number_of_trial_orbitals << "\n";

        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            const std::vector<Complex>& block = record.projection_matrices[kpoint_index];
            for (int orbital_index = 0; orbital_index < record.number_of_trial_orbitals;
                 orbital_index++) {
                for (int band_index = 0; band_index < record.number_of_bands; band_index++) {
                    stream << " " << band_index + 1 << " " << orbital_index + 1 << " "
                           << kpoint_index + 1;
                    const Complex& value =
                        block[static_cast<std::size_t>(orbital_index) * record.number_of_bands +
                              band_index];
                    write_complex(stream, value);
                }
            }
        }
    }

    static void write_eigenvalues(const std::string& file_path, const WannierInputs& record) {
        if (static_cast<int>(record.eigenvalues_rydberg.size()) != record.number_of_kpoints()) {
            throw std::invalid_argument(
                "Wannier90FormatWriter::write_eigenvalues: the record holds no eigenvalues");
        }
        std::ofstream stream = open(file_path);
        for (int kpoint_index = 0; kpoint_index < record.number_of_kpoints(); kpoint_index++) {
            const std::vector<Real>& row = record.eigenvalues_rydberg[kpoint_index];
            if (static_cast<int>(row.size()) < record.number_of_bands) {
                throw std::invalid_argument("Wannier90FormatWriter::write_eigenvalues: k-point " +
                                            std::to_string(kpoint_index) + " carries " +
                                            std::to_string(row.size()) +
                                            " eigenvalues, fewer than number_of_bands = " +
                                            std::to_string(record.number_of_bands));
            }
            for (int band_index = 0; band_index < record.number_of_bands; band_index++) {
                stream << " " << band_index + 1 << " " << kpoint_index + 1 << " " << std::scientific
                       << std::setprecision(12) << row[band_index] * electron_volts_per_rydberg
                       << "\n";
            }
        }
    }

    static void write_neighbor_list(const std::string& file_path,
                                    const WannierInputs& record,
                                    const Matrix3x3& real_lattice,
                                    const Matrix3x3& reciprocal_lattice) {
        std::ofstream stream = open(file_path);
        const int number_of_kpoints = record.number_of_kpoints();
        const int number_of_directions = static_cast<int>(record.neighbor_directions.size());

        stream << "File written by Q3\n\n";
        stream << "calc_only_A  :  F\n\n";

        write_lattice_block(stream, "real_lattice", real_lattice);
        write_lattice_block(stream, "recip_lattice", reciprocal_lattice);

        stream << "begin kpoints\n";
        stream << "  " << number_of_kpoints << "\n";
        for (const Vector3D& kpoint : record.kpoint_coordinates) {
            stream << "  " << std::fixed << std::setprecision(12) << kpoint.x << " " << kpoint.y
                   << " " << kpoint.z << "\n";
        }
        stream << "end kpoints\n\n";

        std::vector<KPoint> kpoints = build_kpoints(record);
        stream << "begin nnkpts\n";
        stream << "  " << number_of_directions << "\n";
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            for (int direction_index = 0; direction_index < number_of_directions;
                 direction_index++) {
                ResolvedKPointNeighbor resolved = KPointNeighborShells::resolve_neighbor(
                    kpoint_index,
                    record.neighbor_directions[direction_index].grid_step,
                    record.kpoint_grid_dimensions,
                    kpoints);
                stream << "  " << kpoint_index + 1 << " " << resolved.neighbor_kpoint_index + 1
                       << " " << resolved.periodic_image_shift.x << " "
                       << resolved.periodic_image_shift.y << " " << resolved.periodic_image_shift.z
                       << "\n";
            }
        }
        stream << "end nnkpts\n";
    }

  private:
    static std::ofstream open(const std::string& file_path) {
        std::ofstream stream(file_path);
        if (!stream) {
            throw std::runtime_error("Wannier90FormatWriter: cannot write " + file_path);
        }
        return stream;
    }

    static void write_complex(std::ofstream& stream, const Complex& value) {
        stream << " " << std::scientific << std::setprecision(12) << value.real() << " "
               << value.imag() << "\n";
    }

    static void
    write_lattice_block(std::ofstream& stream, const std::string& name, const Matrix3x3& lattice) {
        stream << "begin " << name << "\n";
        for (int row = 0; row < 3; row++) {
            stream << "  " << std::fixed << std::setprecision(12) << lattice[row].x << " "
                   << lattice[row].y << " " << lattice[row].z << "\n";
        }
        stream << "end " << name << "\n\n";
    }

    /**
     * resolve_neighbor works on a KPoint list; the record stores bare
     * coordinates. Weights are irrelevant to the neighbour arithmetic, which
     * is pure integer grid indexing plus an exact fractional comparison.
     */
    static std::vector<KPoint> build_kpoints(const WannierInputs& record) {
        std::vector<KPoint> kpoints(record.kpoint_coordinates.size());
        for (std::size_t index = 0; index < record.kpoint_coordinates.size(); index++) {
            kpoints[index].coordinates = record.kpoint_coordinates[index];
            kpoints[index].weight = 0.0;
        }
        return kpoints;
    }

    static void require_overlaps_present(const WannierInputs& record) {
        if (static_cast<int>(record.overlap_matrices.size()) != record.number_of_kpoints()) {
            throw std::invalid_argument(
                "Wannier90FormatWriter::write_overlaps: the record holds no overlaps");
        }
        for (const std::vector<std::vector<Complex>>& per_kpoint : record.overlap_matrices) {
            if (per_kpoint.size() != record.neighbor_directions.size()) {
                throw std::invalid_argument(
                    "Wannier90FormatWriter::write_overlaps: a k-point carries a different number "
                    "of neighbour blocks than there are neighbour directions");
            }
        }
    }
};

} // namespace q3
