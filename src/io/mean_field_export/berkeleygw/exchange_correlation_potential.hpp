#pragma once

/**
 * BerkeleyGWExchangeCorrelationPotentialFile — writes BerkeleyGW's `VXC` file:
 * V_xc(G) itself, from which sigma computes its own <nk|V_xc|nk> when told
 * `dont_use_vxcdat` instead of reading them from `vxc.dat`.
 *
 * Layout: BerkeleyGWFieldFile with the title "VXC-Complex" -- the BerkeleyGW
 * specification's VXC file, in Rydberg, its G = 0 component the cell average.
 * That is the record's `exchange_correlation_potential_g` unscaled. sigma reads
 * it for exactly the bands it needs, which is what makes it the route for bands
 * the NSCF never had -- those parabands adds beyond them.
 *
 * SCOPE: spin-unpolarized only. The record carries one Vxc(G), the spin
 * average, and VXC is per spin; writing the average into both channels would
 * be wrong for any magnetic system, so a spin-polarized record is refused.
 */

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/mean_field_export/berkeleygw/field.hpp"
#include "io/mean_field_record/init.hpp"

namespace q3 {

/// Writes `VXC` from a mean-field record; see the file comment.
class BerkeleyGWExchangeCorrelationPotentialFile {
  public:
    explicit BerkeleyGWExchangeCorrelationPotentialFile(const MeanFieldRecord& record)
        : record_(record) {}
    /// See BerkeleyGWChargeDensityFile: this holds a reference.
    explicit BerkeleyGWExchangeCorrelationPotentialFile(const MeanFieldRecord&&) = delete;

    /// Writes the file, or throws -- before creating it -- when the record cannot supply it.
    void write_to_file(const std::string& file_path) const {
        if (record_.number_of_spin_components > 1) {
            throw std::runtime_error(
                "the record carries the spin-averaged Vxc(G) only, and VXC is per spin; "
                "spin-polarized VXC export is not supported");
        }
        if (record_.exchange_correlation_potential_g.empty()) {
            throw std::runtime_error("the mean-field record carries no exchange-correlation "
                                     "potential");
        }

        std::ofstream output_file(file_path, std::ios::binary);
        if (!output_file.is_open()) {
            throw std::runtime_error("cannot open " + file_path + " for writing");
        }
        const std::vector<std::vector<Complex>> values_per_spin = {
            record_.exchange_correlation_potential_g};
        BerkeleyGWFieldFile(record_, "VXC-Complex", values_per_spin, 1.0).write(output_file);

        log::info("     Wrote BerkeleyGW exchange-correlation potential file: {}", file_path);
    }

  private:
    const MeanFieldRecord& record_;
};

} // namespace q3
