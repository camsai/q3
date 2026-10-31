#pragma once

/**
 * BerkeleyGWRecordWriter — the records BerkeleyGW's mean-field files share
 * after their header.
 *
 * Every file carries the charge-density G-set as one [nrecord=1][ng][gvec]
 * group; the k-point-dependent files (WFN, VKB) carry the same per-k-point
 * counts, weights and coordinates after the header, and the same
 * [nrecord][ngk][gvec_k] group before each k-point's data. BerkeleyGW compares
 * these between files -- ParaBands refuses a VKB whose k-points, weights or
 * plane-wave counts differ from its WFN's -- so they are written by one class
 * rather than kept in step by hand. Layout and authority: see header.hpp.
 */

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/checkpoint/qe.hpp"
#include "io/mean_field_record/init.hpp"

namespace q3 {

/// Writes the records BerkeleyGW's files share; see the file comment.
class BerkeleyGWRecordWriter {
  public:
    explicit BerkeleyGWRecordWriter(const MeanFieldRecord& record) : record_(record) {}
    /// See the file writers' own deleted rvalue constructors: this holds a reference.
    explicit BerkeleyGWRecordWriter(const MeanFieldRecord&&) = delete;

    /// [nrecord=1][ng][gvec(3,ng)] -- the charge-density G-set, in the record's order.
    void write_charge_density_gvectors(std::ofstream& output_file) const {
        std::vector<int32_t> miller_indices(3 * record_.charge_density_miller_indices.size());
        for (size_t g_index = 0; g_index < record_.charge_density_miller_indices.size();
             g_index++) {
            const auto& miller = record_.charge_density_miller_indices[g_index];
            miller_indices[3 * g_index + 0] = miller.x;
            miller_indices[3 * g_index + 1] = miller.y;
            miller_indices[3 * g_index + 2] = miller.z;
        }
        write_block(output_file,
                    static_cast<int32_t>(record_.charge_density_miller_indices.size()),
                    miller_indices.data(),
                    miller_indices.size() * sizeof(int32_t));
    }

    /// ngk(nk) -- plane waves per k-point.
    void write_plane_wave_counts(std::ofstream& output_file) const {
        std::vector<int32_t> counts(record_.plane_waves_per_kpoint.begin(),
                                    record_.plane_waves_per_kpoint.end());
        record_format_.write_record(
            output_file, counts.data(), static_cast<int32_t>(counts.size() * sizeof(int32_t)));
    }

    /**
     * kw(nk), normalized so sum(kw) = 1 as BerkeleyGW requires.
     *
     * The converter gets there by writing the k-point weight times the number
     * of spin components over 2, which leans on the invariant that the weights
     * sum to 2. Q3's generated grids share that invariant
     * (KPointsGrid::scale_weights()), but an explicit `control.kpoints_list`
     * reaches here from the input verbatim -- and a list written on
     * BerkeleyGW's own sum-to-1 convention, the natural choice for the NSCF
     * run BerkeleyGW is fed from, would land on disk as sum(kw) = 0.5. So the
     * weights are divided by their actual sum: identical whenever the
     * invariant holds, correct when it does not.
     */
    void write_normalized_kpoint_weights(std::ofstream& output_file) const {
        std::vector<double> weights(record_.kpoint_weights.begin(), record_.kpoint_weights.end());
        double weight_sum = 0.0;
        for (double weight : weights) {
            weight_sum += weight;
        }
        if (weight_sum <= 0.0) {
            throw std::runtime_error(
                "BerkeleyGW export: k-point weights sum to " + std::to_string(weight_sum) +
                ", so they cannot be normalized to BerkeleyGW's required sum of 1. "
                "Check control.kpoints_list.");
        }
        for (auto& weight : weights) {
            weight /= weight_sum;
        }
        record_format_.write_record(
            output_file, weights.data(), static_cast<int32_t>(weights.size() * sizeof(double)));
    }

    /**
     * kpt(3,nk), crystal coordinates. Q3 holds k-points in crystal coordinates
     * throughout (kpoint.hpp), so unlike the converter, which converts from
     * Cartesian, nothing is transformed here.
     */
    void write_kpoint_coordinates(std::ofstream& output_file) const {
        std::vector<double> coordinates(3 * record_.kpoint_coordinates.size());
        for (size_t kpoint_index = 0; kpoint_index < record_.kpoint_coordinates.size();
             kpoint_index++) {
            const Vector3D& point = record_.kpoint_coordinates[kpoint_index];
            coordinates[3 * kpoint_index + 0] = point.x;
            coordinates[3 * kpoint_index + 1] = point.y;
            coordinates[3 * kpoint_index + 2] = point.z;
        }
        record_format_.write_record(output_file,
                                    coordinates.data(),
                                    static_cast<int32_t>(coordinates.size() * sizeof(double)));
    }

    /// [nrecord=1][ngk][gvec_k(3,ngk)] -- the plane waves of one k-point, in the record's order.
    void write_kpoint_gvectors(std::ofstream& output_file, int kpoint_index) const {
        int number_of_plane_waves = record_.plane_waves_per_kpoint[kpoint_index];
        const auto& g_vector_indices = record_.kpoint_g_vector_indices[kpoint_index];
        std::vector<int32_t> miller_indices(3 * static_cast<size_t>(number_of_plane_waves));
        for (int plane_wave_index = 0; plane_wave_index < number_of_plane_waves;
             plane_wave_index++) {
            const auto& miller =
                record_.charge_density_miller_indices[g_vector_indices[plane_wave_index]];
            miller_indices[3 * plane_wave_index + 0] = miller.x;
            miller_indices[3 * plane_wave_index + 1] = miller.y;
            miller_indices[3 * plane_wave_index + 2] = miller.z;
        }
        write_block(output_file,
                    number_of_plane_waves,
                    miller_indices.data(),
                    miller_indices.size() * sizeof(int32_t));
    }

    /// [nrecord=1][count][values] -- one data block of `count` complex values per spin.
    void write_complex_block(std::ofstream& output_file,
                             int32_t count,
                             const Complex* values,
                             size_t number_of_values) const {
        write_block(output_file, count, values, number_of_values * sizeof(Complex));
    }

  private:
    const MeanFieldRecord& record_;
    QEBinaryRecordFormat record_format_;

    /**
     * Every G-vector or data block is preceded by the number of records it is
     * split into, and its length (the BerkeleyGW specification: "for ease of
     * writing files from a code parallelized over G-vectors"). This writer never
     * splits, so the first is always 1.
     */
    void write_block(std::ofstream& output_file,
                     int32_t count,
                     const void* payload,
                     size_t payload_bytes) const {
        int32_t number_of_records = 1;
        record_format_.write_record(output_file, &number_of_records, sizeof(number_of_records));
        record_format_.write_record(output_file, &count, sizeof(count));
        record_format_.write_record(output_file, payload, static_cast<int32_t>(payload_bytes));
    }
};

} // namespace q3
