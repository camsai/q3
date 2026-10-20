#pragma once

/**
 * BerkeleyGWExporter — orchestrates BerkeleyGW mean-field export.
 *
 * Writes WFN (wavefunctions), then RHO (charge density), then attempts
 * vxc.dat (exchange-correlation matrix elements). vxc.dat is written for
 * the supported case (spin-unpolarized, non-gamma-only -- see
 * exchange_correlation.hpp for the G-space convolution this uses and the
 * exact scope restrictions); outside that scope, or if the SCF state does
 * not carry a usable G-space Vxc, write_to_file() throws a clear, specific
 * error. That attempt is caught and logged here so an out-of-scope or
 * incomplete vxc.dat does not take down an otherwise-successful export.
 * WFN's own spin-polarized (nspin>1) rejection (see wavefunction.hpp) is
 * deliberately NOT caught here -- unlike vxc.dat, a spin-polarized run
 * silently getting RHO without WFN is not a useful partial artifact for a
 * BGW workflow, so that case fails the whole export loudly instead.
 */

#include <filesystem>
#include <string>
#include <system_error>

#include "io/logger.hpp"
#include "io/mean_field_export/berkeleygw/charge_density.hpp"
#include "io/mean_field_export/berkeleygw/exchange_correlation.hpp"
#include "io/mean_field_export/berkeleygw/wavefunction.hpp"
#include "io/mean_field_export/init.hpp"

namespace q3 {

class BerkeleyGWExporter : public MeanFieldExporter {
  public:
    void export_to_directory(const MeanFieldRecord& record,
                             const std::string& directory) const override {
        std::filesystem::create_directories(directory);

        BerkeleyGWWavefunctionFile wavefunction_file(record);
        write_or_remove(directory + "/WFN",
                        [&](const std::string& path) { wavefunction_file.write_to_file(path); });

        BerkeleyGWChargeDensityFile charge_density_file(record);
        write_or_remove(directory + "/RHO",
                        [&](const std::string& path) { charge_density_file.write_to_file(path); });

        BerkeleyGWExchangeCorrelationFile exchange_correlation_file(record);
        try {
            write_or_remove(directory + "/vxc.dat", [&](const std::string& path) {
                exchange_correlation_file.write_to_file(path);
            });
        } catch (const std::runtime_error& error) {
            log::warn("     vxc.dat not written: {}", error.what());
        }
    }

    [[nodiscard]] std::string name() const override {
        return "berkeleygw";
    }

  private:
    /**
     * Write one file, and remove it if writing fails part-way.
     *
     * A file is opened before its header is built, so a refusal there used to
     * leave a truncated WFN on disk. A later BerkeleyGW step, or a person,
     * could take it for the export's output. No file is better than one that
     * stops after its first record.
     */
    template <typename Writer>
    static void write_or_remove(const std::string& path, const Writer& write) {
        try {
            write(path);
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            throw;
        }
    }
};

} // namespace q3
