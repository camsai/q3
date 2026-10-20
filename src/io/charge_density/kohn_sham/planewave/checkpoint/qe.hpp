#pragma once

/**
 * QEPlanewaveKohnShamChargeDensityCheckpoint — QE-format binary writer.
 *
 * Serializes the converged reciprocal-space charge density ρ(G) to disk in the
 * Fortran unformatted record layout of Quantum ESPRESSO's charge-density.dat
 * (Modules/io_base.f90, write_rhog): header, reciprocal vectors, Miller
 * indices, then ρ(G). The full G sphere is written, so gamma_only is false.
 */

#include <fstream>
#include <string>
#include <vector>

#include "io/charge_density/kohn_sham/planewave/checkpoint/init.hpp"
#include "io/checkpoint/qe.hpp"
#include "io/logger.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class QEPlanewaveKohnShamChargeDensityCheckpoint : public PlanewaveKohnShamChargeDensityCheckpoint {
  public:
    using PlanewaveKohnShamChargeDensityCheckpoint::PlanewaveKohnShamChargeDensityCheckpoint;

    void write_to_file(const std::string& filename) const override {
        /**
         * Buffer must be declared BEFORE the ofstream so it outlives the stream.
         * The stream's destructor flushes to the buffer, so the buffer must still be valid.
         */
        std::vector<char> file_buffer(4 * 1024 * 1024);
        std::ofstream output_file;
        output_file.rdbuf()->pubsetbuf(file_buffer.data(), file_buffer.size());
        output_file.open(filename, std::ios::binary);
        if (!output_file.is_open()) {
            log::warn("Cannot open {} for writing", filename);
            return;
        }

        if (!scf_state_.planewave_basis) {
            log::error("planewave_basis is null in NormConservingPlanewaveKohnShamSCFState during "
                       "charge density serialization");
            return;
        }
        if (!scf_state_.crystal_structure) {
            log::error("crystal_structure is null during charge density serialization");
            return;
        }
        const auto& planewave_basis = *scf_state_.planewave_basis;
        const auto& charge_density_g = scf_state_.charge_density_g;
        int number_of_g_vectors = planewave_basis.number_of_g_vectors();

        QEBinaryRecordFormat record_format;

        // Record 1: gamma_only, number_of_g_vectors, number_of_spin_components
        struct {
            int32_t gamma_only = 0;
            int32_t number_of_g_vectors_header;
            int32_t number_of_spin_components = 1;
        } header;
        header.number_of_g_vectors_header = number_of_g_vectors;
        record_format.write_record(output_file, &header, sizeof(header));

        // Record 2: b1, b2, b3 in 1/Bohr
        const Matrix3x3 reciprocal_vectors_bohr =
            scf_state_.crystal_structure->reciprocal_vectors_bohr();
        double reciprocal_vectors_array[9];
        for (int vector_index = 0; vector_index < 3; vector_index++) {
            for (int dimension = 0; dimension < 3; dimension++) {
                reciprocal_vectors_array[3 * vector_index + dimension] =
                    reciprocal_vectors_bohr[vector_index][dimension];
            }
        }
        record_format.write_record(
            output_file, reciprocal_vectors_array, sizeof(reciprocal_vectors_array));

        /**
         * Record 3: Miller indices — QE's mill_g(1:3, 1:ngm_g) is column-major
         * with the component first, so components are interleaved per G-vector.
         */
        std::vector<int32_t> miller_indices(3 * number_of_g_vectors);
        const auto& g_vectors = planewave_basis.g_vectors();
        for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
            miller_indices[3 * g_index + 0] = g_vectors[g_index].miller.x;
            miller_indices[3 * g_index + 1] = g_vectors[g_index].miller.y;
            miller_indices[3 * g_index + 2] = g_vectors[g_index].miller.z;
        }
        record_format.write_record(output_file,
                                   miller_indices.data(),
                                   static_cast<int32_t>(miller_indices.size() * sizeof(int32_t)));

        // Record 4: charge_density_g(G) — complex*16 array
        record_format.write_record(output_file,
                                   charge_density_g.data(),
                                   static_cast<int32_t>(number_of_g_vectors * sizeof(Complex)));

        log::info("     Wrote charge density: {} ({} G-vectors)", filename, number_of_g_vectors);
    }
};

} // namespace q3
