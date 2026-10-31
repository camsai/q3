#pragma once

/**
 * BerkeleyGWKleinmanBylanderProjectorFile — writes BerkeleyGW's `VKB` file: the
 * nonlocal pseudopotential at every k-point, from which ParaBands builds the
 * nonlocal part of the Hamiltonian it diagonalizes,
 *
 *     V_NL(k+G, k+G') = Σ_atom Σ_ij <k+G|β_i> D_ij <β_j|k+G'>.
 *
 * The BerkeleyGW specification does not cover this file. Its authority is the
 * reader BerkeleyGW 4.0 ships (BSD-licensed): `pp_read_header` and
 * `pp_read_beta` in `MeanField/ParaBands/pseudopot_vkb.f90`, with the
 * contraction above from `ham_build_column_nl_col` in
 * `MeanField/ParaBands/hamiltonian.f90`. Shapes in brackets, first index fastest:
 *
 *   records 1-8:  BerkeleyGWCommonHeader, projector-dimensions variant:
 *                 record 2 is ns ng ntran cell_symmetry nat ecutrho
 *                 nk nsp nkb nhm ngkmax ecutwfc; record 3 as WFN's
 *   record  9:    ngk[nk]
 *   record 10:    kw[nk]                   -- normalized to sum to 1, as WFN's
 *   record 11:    kpt[3,nk]                -- crystal coordinates
 *   record 12:    ityp[nat]                -- 1-based species of each atom
 *   record 13:    nh[nsp]                  -- projector channels per species
 *   record 14:    deeq[nhm,nhm,nat,ns]     -- D_ij in Rydberg, zero-padded to nhm
 *   then:         [nrecord=1][ng][gvec[3,ng]]   -- the charge-density G-set
 *   body:         per k-point, [nrecord][ngk][gvec_k], then per spin, per
 *                 projector channel, [nrecord][ngk][<k+G|β>[ngk]]
 *
 * Records 9-11 and the k-point G-vectors must match the WFN they go with:
 * ParaBands compares k-points, weights, plane-wave counts, kgrid and kshift and
 * stops on any difference. Both files write them through
 * BerkeleyGWRecordWriter, so they cannot drift apart.
 *
 * Order of the body's channels. deeq is indexed by atom in the crystal's own
 * order, but the reader takes the projectors grouped by species -- every atom
 * of species 1 in order, then species 2 -- and maps them back. Within an atom
 * the channels follow the record's projector source, the order D was written
 * in, so the pair is consistent whatever that order is.
 *
 * Units: <k+G|β> is the plane-wave matrix element of the projector in Q3's
 * normalization, Ω^(-1/2) β(|k+G|) Y_lm (−i)^l e^(−i(k+G)·τ) (projector.hpp),
 * which with D in Rydberg gives V_NL in Rydberg, as ParaBands' kinetic energy
 * and V_sc are.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/checkpoint/qe.hpp"
#include "io/logger.hpp"
#include "io/mean_field_export/berkeleygw/header.hpp"
#include "io/mean_field_export/berkeleygw/record_writer.hpp"
#include "io/mean_field_record/init.hpp"

namespace q3 {

/// Writes `VKB` from a mean-field record that carries projectors; see the file comment.
class BerkeleyGWKleinmanBylanderProjectorFile {
  public:
    explicit BerkeleyGWKleinmanBylanderProjectorFile(const MeanFieldRecord& record)
        : record_(record) {}
    /// See BerkeleyGWChargeDensityFile: this holds a reference.
    explicit BerkeleyGWKleinmanBylanderProjectorFile(const MeanFieldRecord&&) = delete;

    /// Writes the file, or throws -- before creating it -- when the record carries no projectors.
    void write_to_file(const std::string& file_path) const {
        check_record();

        std::ofstream output_file(file_path, std::ios::binary);
        if (!output_file.is_open()) {
            throw std::runtime_error("cannot open " + file_path + " for writing");
        }

        BerkeleyGWCommonHeader header(record_, "VKB-Complex", projector_dimensions());
        header.write(output_file);

        BerkeleyGWRecordWriter record_writer(record_);
        record_writer.write_plane_wave_counts(output_file);
        record_writer.write_normalized_kpoint_weights(output_file);
        record_writer.write_kpoint_coordinates(output_file);
        write_atom_species_record(output_file);
        write_projectors_per_species_record(output_file);
        write_nonlocal_coefficients_record(output_file);
        record_writer.write_charge_density_gvectors(output_file);
        write_body(output_file, record_writer);

        log::info("     Wrote BerkeleyGW Kleinman-Bylander projector file: {}", file_path);
    }

  private:
    const MeanFieldRecord& record_;
    QEBinaryRecordFormat record_format_;

    void check_record() const {
        if (!record_.projector_source) {
            throw std::runtime_error("the mean-field record carries no projectors; request them "
                                     "with control.mean_field_export_kleinman_bylander_projectors");
        }
        if (static_cast<int>(record_.atom_species_indices.size()) != record_.number_of_atoms() ||
            static_cast<int>(record_.nonlocal_coefficients.size()) !=
                record_.number_of_spin_components) {
            throw std::runtime_error("the mean-field record's nonlocal pseudopotential does not "
                                     "match its atoms and spin channels");
        }
    }

    /// {nsp, nkb, nhm}.
    [[nodiscard]] std::array<int32_t, 3> projector_dimensions() const {
        int32_t number_of_projectors = 0;
        for (int species_index : record_.atom_species_indices) {
            number_of_projectors += record_.projectors_per_species[species_index];
        }
        int32_t maximum_projectors_per_atom =
            record_.projectors_per_species.empty()
                ? 0
                : *std::max_element(record_.projectors_per_species.begin(),
                                    record_.projectors_per_species.end());
        return {static_cast<int32_t>(record_.projectors_per_species.size()),
                number_of_projectors,
                maximum_projectors_per_atom};
    }

    /// Record 12: ityp(nat), 1-based.
    void write_atom_species_record(std::ofstream& output_file) const {
        std::vector<int32_t> species(record_.atom_species_indices.size());
        for (size_t atom_index = 0; atom_index < species.size(); atom_index++) {
            species[atom_index] = record_.atom_species_indices[atom_index] + 1;
        }
        record_format_.write_record(
            output_file, species.data(), static_cast<int32_t>(species.size() * sizeof(int32_t)));
    }

    /// Record 13: nh(nsp).
    void write_projectors_per_species_record(std::ofstream& output_file) const {
        std::vector<int32_t> counts(record_.projectors_per_species.begin(),
                                    record_.projectors_per_species.end());
        record_format_.write_record(
            output_file, counts.data(), static_cast<int32_t>(counts.size() * sizeof(int32_t)));
    }

    /// Record 14: deeq(nhm,nhm,nat,ns), each atom's block zero-padded to nhm x nhm.
    void write_nonlocal_coefficients_record(std::ofstream& output_file) const {
        const size_t maximum_channels = static_cast<size_t>(projector_dimensions()[2]);
        const size_t number_of_atoms = static_cast<size_t>(record_.number_of_atoms());
        const size_t block_size = maximum_channels * maximum_channels;
        std::vector<double> coefficients(
            block_size * number_of_atoms * record_.number_of_spin_components, 0.0);
        for (int spin_index = 0; spin_index < record_.number_of_spin_components; spin_index++) {
            for (size_t atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                const auto& block = record_.nonlocal_coefficients[spin_index][atom_index];
                size_t number_of_channels = static_cast<size_t>(
                    record_.projectors_per_species[record_.atom_species_indices[atom_index]]);
                double* destination =
                    coefficients.data() + block_size * (atom_index + number_of_atoms * spin_index);
                for (size_t channel_j = 0; channel_j < number_of_channels; channel_j++) {
                    for (size_t channel_i = 0; channel_i < number_of_channels; channel_i++) {
                        destination[channel_i + maximum_channels * channel_j] =
                            block[channel_i + number_of_channels * channel_j];
                    }
                }
            }
        }
        record_format_.write_record(output_file,
                                    coefficients.data(),
                                    static_cast<int32_t>(coefficients.size() * sizeof(double)));
    }

    /**
     * Per k-point: the k-point's G-vectors, then every projector channel as its
     * own block, atoms grouped by species. The projectors do not depend on spin
     * (D carries any spin dependence), so each spin channel repeats them, as
     * the reader expects one set per spin.
     */
    void write_body(std::ofstream& output_file, const BerkeleyGWRecordWriter& record_writer) const {
        std::vector<int> first_projector_of_atom(record_.number_of_atoms() + 1, 0);
        for (int atom_index = 0; atom_index < record_.number_of_atoms(); atom_index++) {
            first_projector_of_atom[atom_index + 1] =
                first_projector_of_atom[atom_index] +
                record_.projectors_per_species[record_.atom_species_indices[atom_index]];
        }

        for (int kpoint_index = 0; kpoint_index < record_.number_of_kpoints(); kpoint_index++) {
            record_writer.write_kpoint_gvectors(output_file, kpoint_index);

            int32_t number_of_plane_waves = record_.plane_waves_per_kpoint[kpoint_index];
            const Matrix<Complex>& projectors = record_.projector_source->projectors(kpoint_index);
            if (static_cast<int>(projectors.rows()) != first_projector_of_atom.back() ||
                static_cast<int32_t>(projectors.cols()) != number_of_plane_waves) {
                throw std::runtime_error("the projector source returned a " +
                                         std::to_string(projectors.rows()) + " x " +
                                         std::to_string(projectors.cols()) + " matrix at k-point " +
                                         std::to_string(kpoint_index) + ", expected " +
                                         std::to_string(first_projector_of_atom.back()) + " x " +
                                         std::to_string(number_of_plane_waves));
            }

            for (int spin_index = 0; spin_index < record_.number_of_spin_components; spin_index++) {
                for (int species_index = 0;
                     species_index < static_cast<int>(record_.projectors_per_species.size());
                     species_index++) {
                    for (int atom_index = 0; atom_index < record_.number_of_atoms(); atom_index++) {
                        if (record_.atom_species_indices[atom_index] != species_index) {
                            continue;
                        }
                        for (int projector_index = first_projector_of_atom[atom_index];
                             projector_index < first_projector_of_atom[atom_index + 1];
                             projector_index++) {
                            record_writer.write_complex_block(output_file,
                                                              number_of_plane_waves,
                                                              projectors[projector_index],
                                                              number_of_plane_waves);
                        }
                    }
                }
            }
        }
    }
};

} // namespace q3
