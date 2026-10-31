#pragma once

/**
 * BerkeleyGWWavefunctionFile — writes BerkeleyGW's `WFN` mean-field file.
 *
 * Layout (the BerkeleyGW specification, as pw2bgw.x's `write_wfng` writes it
 * -- see header.hpp for both citations; shapes in brackets, first index fastest):
 *   records 1-8:  BerkeleyGWCommonHeader (include_kpoint_fields=true)
 *   record  9:    ngk[nk]                          -- plane waves per k-point
 *   record 10:    kw[nk]                           -- weights, normalized to sum to 1
 *   record 11:    kpt[3,nk]                        -- crystal (fractional) coordinates
 *   record 12:    ifmin[nk*ns]                     -- lowest occupied band (1-based)
 *   record 13:    ifmax[nk*ns]                     -- highest occupied band (1-based, 0 if none)
 *   record 14:    energies[nbands,nk*ns]           -- Rydberg
 *   record 15:    occupations[nbands,nk*ns]
 *   record 16-18: nrecord=1, ng, gvec[3,ng]         -- charge-density G-set
 *   body:         per k-point, gvec_k then per-band coefficients (see below)
 *
 * Always written as flavor 2 ("WFN-Complex"): Q3 stores wavefunction
 * coefficients as Complex regardless of gamma-only status, so there is no
 * Q3-side "real" storage to map onto pw2bgw.x's flavor-1 (real_or_complex=1)
 * path, which additionally reorders bands/k-points into time-reversal pairs
 * (pw2bgw.x's real_wfng subroutine) -- out of scope here.
 *
 * SCOPE: spin-unpolarized (ns=1) only. write_to_file() rejects ns>1 up
 * front with a clear error rather than writing a silently-wrong file.
 * The WFN body has two layouts, selected by the file's own flavor tag and
 * spin count (write_wfng chooses between them). For flavor 2 with ns=1 --
 * our case -- each band is one record of exactly ngk coefficients, which is what write_body()
 * below implements. For flavor 1, or for ns>1 with either flavor, each BAND
 * is instead a SINGLE record holding all ns spins' coefficients concatenated,
 * G index fastest and spin index slowest (write_wfng; this is the
 * specification's ng-values-per-spin data layout), not one record
 * per (k,spin,band) as write_body() produces. Implementing that interleaving
 * correctly, with no BerkeleyGW reference file available in this environment to
 * validate against, was judged too risky to guess at silently -- rejecting is
 * the honest choice (see also exchange_correlation.hpp's vxc.dat gap).
 */

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
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class BerkeleyGWWavefunctionFile {
  public:
    explicit BerkeleyGWWavefunctionFile(const MeanFieldRecord& record) : record_(record) {}
    /**
     * These writers keep a reference to the record, so binding one to a
     * temporary would dangle the moment the full expression ends. Deleting the
     * rvalue overload turns that into a compile error instead of a silent
     * read of freed memory -- callers must keep the record alive themselves,
     * which every real call site already does.
     */
    explicit BerkeleyGWWavefunctionFile(const MeanFieldRecord&&) = delete;

    void write_to_file(const std::string& file_path) const {
        if (record_.number_of_spin_components > 1) {
            throw std::runtime_error(
                "BerkeleyGWWavefunctionFile: spin-polarized (nspin>1) WFN export is not yet "
                "supported. pw2bgw writes each band as a single record with all spins' "
                "coefficients concatenated (G fastest, spin slower) rather than one record per "
                "(k,spin,band); implementing that interleaving without a BerkeleyGW/QE reference "
                "to validate against was judged too risky to guess at silently. Use "
                "number_of_spin_components: 1, or omit mean_field_export, for a spin-polarized "
                "run.");
        }

        std::ofstream output_file(file_path, std::ios::binary);
        if (!output_file.is_open()) {
            log::warn("BerkeleyGWWavefunctionFile: cannot open {} for writing", file_path);
            return;
        }

        BerkeleyGWCommonHeader header(record_, "WFN-Complex", /*include_kpoint_fields=*/true);
        header.write(output_file);

        BerkeleyGWRecordWriter record_writer(record_);
        record_writer.write_plane_wave_counts(output_file);         // record 9
        record_writer.write_normalized_kpoint_weights(output_file); // record 10
        record_writer.write_kpoint_coordinates(output_file);        // record 11
        write_occupied_band_bounds_records(output_file);
        write_eigenvalues_record(output_file);
        write_occupations_record(output_file);
        record_writer.write_charge_density_gvectors(output_file); // records 16-18
        write_body(output_file, record_writer);

        log::info("     Wrote BerkeleyGW wavefunction file: {}", file_path);
    }

  private:
    const MeanFieldRecord& record_;
    QEBinaryRecordFormat record_format_;

    /**
     * A band is "occupied" for ifmin/ifmax purposes when its occupation
     * exceeds this threshold -- the same 0.5 cutoff write_wfng applies to
     * its own per-band occupations, which are QE's band weights divided by
     * the k-point weight, i.e. a per-band occupation number on a
     * max-1-per-(k,spin)-state scale; Q3's own
     * occupation_by_kpoint_state_index_and_band_index() is on the identical
     * max-1 scale (the fixed-occupation path returns exactly 1.0 or 0.0; the
     * smeared path's total_electrons accumulation in
     * fermi_energy/planewave/init.hpp multiplies the smearing function's
     * [0,1] output by the k-point weight, which already carries the ×2
     * spin-degeneracy factor for the unpolarized case) -- so the same
     * absolute 0.5 cutoff applies unchanged.
     */
    static constexpr Real occupied_band_occupation_threshold = 0.5;

    /// Records 12-13: ifmin[nk*ns], ifmax[nk*ns], both 1-based (write_wfng records 12-13).
    void write_occupied_band_bounds_records(std::ofstream& output_file) const {
        int number_of_kpoints = record_.number_of_kpoints();
        int number_of_spin_components = record_.number_of_spin_components;
        int number_of_bands = record_.number_of_bands;
        int total_kpoint_states = number_of_kpoints * number_of_spin_components;

        std::vector<int32_t> lowest_occupied_band(total_kpoint_states, 0);
        std::vector<int32_t> highest_occupied_band(total_kpoint_states, 0);

        for (int state_index = 0; state_index < total_kpoint_states; state_index++) {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                Real occupation = record_.occupations[state_index][band_index];
                if (occupation > occupied_band_occupation_threshold) {
                    if (lowest_occupied_band[state_index] == 0) {
                        lowest_occupied_band[state_index] = band_index + 1;
                    }
                    highest_occupied_band[state_index] = band_index + 1;
                }
            }
        }

        record_format_.write_record(
            output_file,
            lowest_occupied_band.data(),
            static_cast<int32_t>(lowest_occupied_band.size() * sizeof(int32_t)));
        record_format_.write_record(
            output_file,
            highest_occupied_band.data(),
            static_cast<int32_t>(highest_occupied_band.size() * sizeof(int32_t)));
    }

    /**
     * Record 14: energies[nbands,nk*ns], Rydberg, as the BerkeleyGW
     * specification requires. write_wfng writes QE's eigenvalues with NO
     * unit conversion (unlike vxc.dat's explicit conversion to eV) -- QE's
     * eigenvalues are already Rydberg internally, matching Q3's own convention (AGENTS.md § Units:
     * "use the same units as Quantum ESPRESSO"; results.json's "Ry" units), so
     * eigenvalue() is copied as-is. Band is the fast (first) Fortran index, so for each
     * k-point-state all nbands values are written contiguously before moving to the next state --
     * same spin-major state ordering as records 12-13 (see write_occupied_band_bounds_records).
     */
    void write_eigenvalues_record(std::ofstream& output_file) const {
        write_band_major_matrix_record(output_file, [this](int state_index, int band_index) {
            return record_.eigenvalues_rydberg[state_index][band_index];
        });
    }

    /// Record 15: occupations(nbands,nk*ns), same shape as record 14.
    void write_occupations_record(std::ofstream& output_file) const {
        write_band_major_matrix_record(output_file, [this](int state_index, int band_index) {
            return record_.occupations[state_index][band_index];
        });
    }

    template <typename ValueFunction>
    void write_band_major_matrix_record(std::ofstream& output_file, ValueFunction value_of) const {
        int number_of_kpoints = record_.number_of_kpoints();
        int number_of_spin_components = record_.number_of_spin_components;
        int number_of_bands = record_.number_of_bands;
        int total_kpoint_states = number_of_kpoints * number_of_spin_components;

        std::vector<double> values(static_cast<size_t>(number_of_bands) * total_kpoint_states);
        for (int state_index = 0; state_index < total_kpoint_states; state_index++) {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                values[static_cast<size_t>(state_index) * number_of_bands + band_index] =
                    value_of(state_index, band_index);
            }
        }
        record_format_.write_record(
            output_file, values.data(), static_cast<int32_t>(values.size() * sizeof(double)));
    }

    /**
     * Body (the BerkeleyGW specification's per-k-point G-vector list, then
     * the coefficients of each band, as write_wfng writes them -- the
     * flavor-2, ns=1 layout, the only one
     * implemented here; see the class comment's SCOPE note): per k-point, in
     * physical-k-point order (0..nk-1), one [nrecord][ngk][gvec_k] triple,
     * then per band one [nrecord][ngk][coefficients] triple. write_to_file()
     * has already rejected ns>1 before this is ever called, so
     * kpoint_state_index == kpoint_index unconditionally (spin_index is
     * always 0).
     */
    void write_body(std::ofstream& output_file, const BerkeleyGWRecordWriter& record_writer) const {
        int number_of_kpoints = record_.number_of_kpoints();
        int number_of_bands = record_.number_of_bands;

        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            record_writer.write_kpoint_gvectors(output_file, kpoint_index);

            int32_t number_of_plane_waves = record_.plane_waves_per_kpoint[kpoint_index];
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                int kpoint_state_index = kpoint_index; // spin_index is always 0 here (ns=1)
                record_writer.write_complex_block(
                    output_file,
                    number_of_plane_waves,
                    record_.band_source->coefficients(kpoint_state_index, band_index),
                    number_of_plane_waves);
            }
        }
    }
};

} // namespace q3
