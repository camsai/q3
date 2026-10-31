#pragma once

/**
 * BerkeleyGWFieldFile — one scalar field on the charge-density G-set, per spin:
 * the layout BerkeleyGW's RHO, VXC and VSC files share.
 *
 * Layout (the BerkeleyGW specification's RHO and VXC files; BerkeleyGW 4.0's
 * ParaBands reads VSC with the same routines, accepting the title "VSC" or the
 * legacy "VXC" -- `MeanField/ParaBands/parabands.f90`):
 *   records 1-8: BerkeleyGWCommonHeader (include_kpoint_fields=false)
 *   body:        [nrecord=1][ng][gvec[3,ng]] then [nrecord=1][ng][field[ng,ns]]
 *
 * The field is written one spin channel after another, G fastest, each value
 * multiplied by `scale`: the cell volume for RHO, whose G = 0 component the
 * specification makes the electron count; 1 for the potentials, whose G = 0
 * component is their cell average, in Rydberg.
 */

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "io/mean_field_export/berkeleygw/header.hpp"
#include "io/mean_field_export/berkeleygw/record_writer.hpp"
#include "io/mean_field_record/init.hpp"

namespace q3 {

/// One per-spin field on the charge-density G-set, in RHO's layout; see the file comment.
class BerkeleyGWFieldFile {
  public:
    /**
     * @param title            "RHO-Complex", "VXC-Complex" or "VSC-Complex".
     * @param values_per_spin  one array per spin channel, each parallel to the
     *                         record's charge-density G-set; there must be
     *                         exactly `number_of_spin_components` of them.
     */
    BerkeleyGWFieldFile(const MeanFieldRecord& record,
                        std::string title,
                        const std::vector<std::vector<Complex>>& values_per_spin,
                        Real scale)
        : record_(record)
        , title_(std::move(title))
        , values_per_spin_(values_per_spin)
        , scale_(scale) {}
    /// See the file writers' own deleted rvalue constructors: this holds references.
    BerkeleyGWFieldFile(const MeanFieldRecord&&,
                        std::string,
                        const std::vector<std::vector<Complex>>&,
                        Real) = delete;

    /// Writes header, G-set and field to an open stream; throws if the field is not on the G-set.
    void write(std::ofstream& output_file) const {
        int number_of_g_vectors = record_.number_of_g_vectors();
        int number_of_spin_components = record_.number_of_spin_components;
        if (static_cast<int>(values_per_spin_.size()) != number_of_spin_components) {
            throw std::runtime_error(
                title_ + ": the field has " + std::to_string(values_per_spin_.size()) +
                " spin channel(s), the record " + std::to_string(number_of_spin_components));
        }
        for (const auto& values : values_per_spin_) {
            if (static_cast<int>(values.size()) != number_of_g_vectors) {
                throw std::runtime_error(title_ + ": the field has " +
                                         std::to_string(values.size()) +
                                         " G-vector components, the charge-density G-set " +
                                         std::to_string(number_of_g_vectors));
            }
        }

        BerkeleyGWCommonHeader header(record_, title_, /*include_kpoint_fields=*/false);
        header.write(output_file);

        BerkeleyGWRecordWriter record_writer(record_);
        record_writer.write_charge_density_gvectors(output_file);

        std::vector<Complex> scaled_values;
        scaled_values.reserve(static_cast<size_t>(number_of_g_vectors) * number_of_spin_components);
        for (const auto& values : values_per_spin_) {
            for (const Complex& value : values) {
                scaled_values.push_back(value * scale_);
            }
        }
        record_writer.write_complex_block(
            output_file, number_of_g_vectors, scaled_values.data(), scaled_values.size());
    }

  private:
    const MeanFieldRecord& record_;
    std::string title_;
    const std::vector<std::vector<Complex>>& values_per_spin_;
    Real scale_;
};

} // namespace q3
