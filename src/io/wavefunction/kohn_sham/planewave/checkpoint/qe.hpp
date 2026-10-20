#pragma once

/**
 * QEPlanewaveKohnShamWavefunctionCheckpoint — QE-format binary writer.
 *
 * Serializes converged Kohn-Sham wavefunctions to disk in the Fortran
 * unformatted record layout of Quantum ESPRESSO's wfcN.dat (non-spin-polarized
 * runs; see TODOs.md for LSDA). One file is written per k-point,
 * containing Miller indices and complex wavefunction coefficients for
 * all bands.
 *
 * Writes the k-points the state holds, which under k-point pools are only this
 * pool's. Each file is named and labelled by the k-point's place in the global
 * grid, so every pool writing its own share leaves the same set of files a
 * serial run does, with no gather and no two processes opening the same file.
 */

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "io/checkpoint/packed_record_buffer.hpp"
#include "io/checkpoint/qe.hpp"
#include "io/logger.hpp"
#include "io/wavefunction/kohn_sham/planewave/checkpoint/init.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class QEPlanewaveKohnShamWavefunctionCheckpoint : public PlanewaveKohnShamWavefunctionCheckpoint {
  public:
    using PlanewaveKohnShamWavefunctionCheckpoint::PlanewaveKohnShamWavefunctionCheckpoint;

    void write_to_file(const std::string& path) const override {
        if (!scf_state_.planewave_basis) {
            log::error("planewave_basis is null in NormConservingPlanewaveKohnShamSCFState during "
                       "wavefunction serialization");
            return;
        }
        if (!scf_state_.execution_policy) {
            log::error("execution_policy is null in NormConservingPlanewaveKohnShamSCFState during "
                       "wavefunction serialization");
            return;
        }
        const auto& planewave_basis = *scf_state_.planewave_basis;
        const auto& execution_policy = *scf_state_.execution_policy;
        const auto& g_vectors = planewave_basis.g_vectors();
        const auto& quantum_system = scf_state_.quantum_system();
        const auto& g_vector_indices_all = planewave_basis.kpoint_g_vector_indices();
        const auto& number_of_plane_waves_all = planewave_basis.kpoint_number_of_plane_waves();
        int number_of_bands = scf_state_.quantum_system().number_of_bands();
        int number_of_kpoints = kpoint_grid_.number_of_kpoints();

        // QE writes b1, b2, b3 in 1/Bohr (it passes tpiba*bg).
        const Matrix3x3 reciprocal_lattice_vectors = crystal_structure_.reciprocal_vectors_bohr();

        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            const int global_kpoint_index = execution_policy.global_kpoint_index(kpoint_index);

            std::string file_path =
                path + "/wfc" + std::to_string(global_kpoint_index + 1) + ".dat";
            // Buffer must be declared BEFORE the ofstream so it outlives the stream.
            std::vector<char> file_buffer(4 * 1024 * 1024);
            std::ofstream output_file;
            output_file.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
            output_file.open(file_path, std::ios::binary);
            if (!output_file.is_open()) {
                log::warn("Cannot open {} for writing", file_path);
                continue;
            }

            int number_of_plane_waves = number_of_plane_waves_all[kpoint_index];
            const auto& g_vector_indices = g_vector_indices_all[kpoint_index];
            const auto& kpoint = kpoint_grid_[kpoint_index];

            write_kpoint_header(output_file, global_kpoint_index, kpoint);
            write_dimensions(output_file, number_of_plane_waves, number_of_bands);
            write_reciprocal_lattice_vectors(output_file, reciprocal_lattice_vectors);
            write_miller_indices(output_file, g_vectors, g_vector_indices, number_of_plane_waves);
            write_band_coefficients(
                output_file, quantum_system, kpoint_index, number_of_bands, number_of_plane_waves);

            log::info("     Wrote wavefunction: {} ({} bands, {} PW)",
                      file_path,
                      number_of_bands,
                      number_of_plane_waves);
        }
    }

  private:
    QEBinaryRecordFormat record_format_;

    /**
     * Record 1: k-point header (index, coordinates, spin, gamma flag, scale factor).
     *
     * Quantum ESPRESSO writes this record as one Fortran list,
     * `ik, xk(3), ispin, gamma_only, scalef` (Modules/io_base.f90, write_wfc),
     * and Fortran unformatted I/O packs a list with no alignment padding:
     * int32 at byte 0, three doubles at 4, int32 at 28, a 4-byte LOGICAL at
     * 32 and a double at 36 -- 44 bytes. The fields are therefore appended
     * one by one into a byte buffer rather than dumped from a C struct, whose
     * alignment would put four padding bytes after the leading integer and
     * shift every later field. Writing field by field also means no byte of
     * the record can come from uninitialized memory.
     *
     * Units follow QE: xk is Cartesian in 1/Bohr (QE passes tpiba*xk).
     */
    void
    write_kpoint_header(std::ofstream& output_file, int kpoint_index, const KPoint& kpoint) const {
        const Vector3D kpoint_cartesian = kpoint_cartesian_inverse_bohr(kpoint);
        PackedRecordBuffer record;
        record.append(static_cast<int32_t>(kpoint_index + 1));
        record.append(static_cast<double>(kpoint_cartesian.x));
        record.append(static_cast<double>(kpoint_cartesian.y));
        record.append(static_cast<double>(kpoint_cartesian.z));
        record.append(static_cast<int32_t>(1)); // spin index
        record.append(static_cast<int32_t>(scf_state_.planewave_basis->is_gamma_only() ? 1 : 0));
        record.append(1.0); // scale factor
        record_format_.write_record(
            output_file, record.data(), static_cast<int32_t>(record.size()));
    }

    /**
     * Fractional k-point coordinates (in units of the reciprocal basis) to
     * Cartesian 1/Bohr, the convention of QE's on-disk xk.
     */
    [[nodiscard]] Vector3D kpoint_cartesian_inverse_bohr(const KPoint& kpoint) const {
        const Matrix3x3 reciprocal_vectors_bohr = crystal_structure_.reciprocal_vectors_bohr();
        Vector3D cartesian;
        for (int dimension = 0; dimension < 3; dimension++) {
            cartesian[dimension] = kpoint.coordinates.x * reciprocal_vectors_bohr[0][dimension] +
                                   kpoint.coordinates.y * reciprocal_vectors_bohr[1][dimension] +
                                   kpoint.coordinates.z * reciprocal_vectors_bohr[2][dimension];
        }
        return cartesian;
    }

    /**
     * Record 2: dimensions (number of plane waves, max index, polarizations, bands).
     */
    void write_dimensions(std::ofstream& output_file,
                          int number_of_plane_waves,
                          int number_of_bands) const {
        struct DimensionsRecord {
            int32_t number_of_plane_waves_wavefunction;
            int32_t maximum_wavefunction_index;
            int32_t number_of_polarizations;
            int32_t number_of_bands;
        };
        /**
         * Four same-sized members leave this one with no padding today, so
         * the fill is not repairing anything. It is here so that adding a
         * member of a different width -- which would introduce padding
         * exactly as it did in the k-point header above -- cannot silently
         * start writing uninitialized bytes to disk again.
         */
        DimensionsRecord record;
        std::memset(&record, 0, sizeof(record));
        record.number_of_plane_waves_wavefunction = number_of_plane_waves;
        record.maximum_wavefunction_index = number_of_plane_waves;
        record.number_of_polarizations = 1;
        record.number_of_bands = number_of_bands;
        record_format_.write_record(output_file, &record, sizeof(record));
    }

    /**
     * Record 3: reciprocal lattice vectors b1, b2, b3 (3x3 doubles, 1/Bohr).
     */
    void write_reciprocal_lattice_vectors(std::ofstream& output_file,
                                          const Matrix3x3& reciprocal_lattice_vectors) const {
        double vectors_array[9];
        vectors_array[0] = reciprocal_lattice_vectors[0].x;
        vectors_array[1] = reciprocal_lattice_vectors[0].y;
        vectors_array[2] = reciprocal_lattice_vectors[0].z;
        vectors_array[3] = reciprocal_lattice_vectors[1].x;
        vectors_array[4] = reciprocal_lattice_vectors[1].y;
        vectors_array[5] = reciprocal_lattice_vectors[1].z;
        vectors_array[6] = reciprocal_lattice_vectors[2].x;
        vectors_array[7] = reciprocal_lattice_vectors[2].y;
        vectors_array[8] = reciprocal_lattice_vectors[2].z;
        record_format_.write_record(output_file, vectors_array, sizeof(vectors_array));
    }

    /**
     * Record 4: Miller indices for all G-vectors at this k-point.
     *
     * QE writes `mill(1:3, 1:igwx)`: Fortran column-major with the component
     * index first, so the three components of each G-vector are contiguous --
     * h0 k0 l0 h1 k1 l1 ... (interleaved, not all-h then all-k then all-l).
     */
    void write_miller_indices(std::ofstream& output_file,
                              const std::vector<GVector>& g_vectors,
                              const std::vector<int>& g_vector_indices,
                              int number_of_plane_waves) const {
        std::vector<int32_t> miller_indices(3 * number_of_plane_waves);
        for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
             plane_wave_index++) {
            int g_index = g_vector_indices[plane_wave_index];
            miller_indices[3 * plane_wave_index + 0] = g_vectors[g_index].miller.x;
            miller_indices[3 * plane_wave_index + 1] = g_vectors[g_index].miller.y;
            miller_indices[3 * plane_wave_index + 2] = g_vectors[g_index].miller.z;
        }
        record_format_.write_record(output_file,
                                    miller_indices.data(),
                                    static_cast<int32_t>(miller_indices.size() * sizeof(int32_t)));
    }

    /**
     * Records 5..5+number_of_bands-1: complex wavefunction coefficients per band.
     */
    void write_band_coefficients(std::ofstream& output_file,
                                 const PlanewaveKohnShamQuantumSystem& quantum_system,
                                 int kpoint_index,
                                 int number_of_bands,
                                 int number_of_plane_waves) const {
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            const Complex* band_data =
                quantum_system.wavefunction_band_data(kpoint_index, band_index);
            record_format_.write_record(
                output_file,
                band_data,
                static_cast<int32_t>(number_of_plane_waves * sizeof(Complex)));
        }
    }
};

} // namespace q3
