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

#include <fstream>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/mean_field_export/berkeleygw/field.hpp"
#include "io/mean_field_record/init.hpp"

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

        BerkeleyGWFieldFile(record_, "RHO-Complex", record_.charge_density_g, record_.cell_volume)
            .write(output_file);

        log::info("     Wrote BerkeleyGW charge density file: {}", file_path);
    }

  private:
    const MeanFieldRecord& record_;
};

} // namespace q3
