#pragma once

/**
 * BerkeleyGWChargeDensityFile — writes BerkeleyGW's `RHO` mean-field file.
 *
 * Layout (the BerkeleyGW specification, as pw2bgw.x's `write_rhog` writes it
 * -- see header.hpp for both citations):
 *   records 1-8: BerkeleyGWCommonHeader (include_kpoint_fields=false)
 *   body:        [nrecord=1][ng][gvec[3,ng]] then [nrecord=1][ng][rho[ng,ns]]
 *
 * rho(G) is per-spin (the specification lists the data as ng values per
 * spin; write_rhog writes QE's G-space density one spin channel at a time --
 * for ns=2 the actual spin-up/spin-down density after QE's own
 * total/magnetization -> up/down conversion, not (total, magnetization)),
 * scaled by the cell volume (write_rhog), which makes the G=0 component the
 * number of electrons in the cell, as the specification requires. Q3's
 * charge_density_real is per-spin already (see NormConservingPlanewaveKohnShamQuantumSystem's
 * charge_density_real_[0]/[1] = 0.5*(total±magnetization) assembly in the
 * pseudopotential iterator), so each spin channel is independently
 * transformed to G-space via PlanewaveBasis::rho_r_to_g() rather than
 * relying on SCFState::charge_density_g (which is the single spin-summed
 * total, documented as such, and insufficient for ns=2).
 *
 * Unlike BerkeleyGWWavefunctionFile, RHO's on-disk layout does not depend on
 * nspin (write_rhog writes all ng x ns values as a single record regardless of
 * spin count -- there is no analog of WFN's layout-dependent per-band
 * interleaving here), so this writer supports spin-polarized (ns=2) runs
 * fully; only WFN rejects them.
 */

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "io/checkpoint/qe.hpp"
#include "io/logger.hpp"
#include "io/mean_field_export/berkeleygw/header.hpp"
#include "io/mean_field_record/init.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class BerkeleyGWChargeDensityFile {
  public:
    explicit BerkeleyGWChargeDensityFile(const MeanFieldRecord& record) : record_(record) {}
    /**
     * These writers keep a reference to the record, so binding one to a
     * temporary would dangle the moment the full expression ends. Deleting the
     * rvalue overload turns that into a compile error instead of a silent
     * read of freed memory -- callers must keep the record alive themselves,
     * which every real call site already does.
     */
    explicit BerkeleyGWChargeDensityFile(const MeanFieldRecord&&) = delete;

    void write_to_file(const std::string& file_path) const {

        std::ofstream output_file(file_path, std::ios::binary);
        if (!output_file.is_open()) {
            log::warn("BerkeleyGWChargeDensityFile: cannot open {} for writing", file_path);
            return;
        }

        BerkeleyGWCommonHeader header(record_, "RHO-Complex", /*include_kpoint_fields=*/false);
        header.write(output_file);

        write_gvectors_record(output_file);
        write_density_record(output_file);

        log::info("     Wrote BerkeleyGW charge density file: {}", file_path);
    }

  private:
    const MeanFieldRecord& record_;
    QEBinaryRecordFormat record_format_;

    /// [nrecord=1][ng][gvec(3,ng)] -- the same charge-density G-set as WFN records 16-18.
    void write_gvectors_record(std::ofstream& output_file) const {
        const auto& g_vectors = record_.charge_density_miller_indices;

        int32_t nrecord = 1;
        record_format_.write_record(output_file, &nrecord, sizeof(nrecord));
        int32_t number_of_g_vectors = static_cast<int32_t>(g_vectors.size());
        record_format_.write_record(output_file, &number_of_g_vectors, sizeof(number_of_g_vectors));

        std::vector<int32_t> miller_indices(3 * g_vectors.size());
        for (size_t g_index = 0; g_index < g_vectors.size(); g_index++) {
            miller_indices[3 * g_index + 0] = g_vectors[g_index].x;
            miller_indices[3 * g_index + 1] = g_vectors[g_index].y;
            miller_indices[3 * g_index + 2] = g_vectors[g_index].z;
        }
        record_format_.write_record(output_file,
                                    miller_indices.data(),
                                    static_cast<int32_t>(miller_indices.size() * sizeof(int32_t)));
    }

    /// [nrecord=1][ng][rho(ng,ns)] -- per-spin density in G-space, scaled by the cell volume.
    void write_density_record(std::ofstream& output_file) const {
        int number_of_g_vectors = record_.number_of_g_vectors();
        int number_of_spin_components = record_.number_of_spin_components;
        Real cell_volume = record_.cell_volume;

        int32_t nrecord = 1;
        record_format_.write_record(output_file, &nrecord, sizeof(nrecord));
        int32_t number_of_g_vectors_i32 = number_of_g_vectors;
        record_format_.write_record(
            output_file, &number_of_g_vectors_i32, sizeof(number_of_g_vectors_i32));

        std::vector<Complex> density_g_scaled(static_cast<size_t>(number_of_g_vectors) *
                                              number_of_spin_components);
        for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
            const std::vector<Complex>& density_g_for_spin = record_.charge_density_g[spin_index];
            for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
                density_g_scaled[static_cast<size_t>(spin_index) * number_of_g_vectors + g_index] =
                    density_g_for_spin[g_index] * cell_volume;
            }
        }

        record_format_.write_record(
            output_file,
            density_g_scaled.data(),
            static_cast<int32_t>(density_g_scaled.size() * sizeof(Complex)));
    }
};

} // namespace q3
