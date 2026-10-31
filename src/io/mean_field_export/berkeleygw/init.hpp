#pragma once

/**
 * BerkeleyGWExporter — orchestrates BerkeleyGW mean-field export.
 *
 * Writes WFN (wavefunctions), then RHO (charge density), then attempts
 * vxc.dat (exchange-correlation matrix elements), VXC (the exchange-correlation
 * potential) and VSC (the local Kohn-Sham potential), and -- when the record
 * carries projectors, which only
 * `control.mean_field_export_kleinman_bylander_projectors` asks for -- VKB (the
 * nonlocal pseudopotential). VSC and VKB are what ParaBands needs to extend WFN
 * with more bands; VXC is what sigma reads under `dont_use_vxcdat`. vxc.dat is written for
 * the supported case (spin-unpolarized, non-gamma-only -- see
 * exchange_correlation.hpp for the G-space convolution this uses and the
 * exact scope restrictions); outside that scope, or if the SCF state does
 * not carry a usable G-space Vxc, write_to_file() throws a clear, specific
 * error. That attempt is caught and logged here so an out-of-scope or
 * incomplete vxc.dat does not take down an otherwise-successful export. VXC,
 * VSC and VKB are attempted the same way, for the same reason.
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
#include "io/mean_field_export/berkeleygw/exchange_correlation_potential.hpp"
#include "io/mean_field_export/berkeleygw/kleinman_bylander_projectors.hpp"
#include "io/mean_field_export/berkeleygw/self_consistent_potential.hpp"
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

        BerkeleyGWExchangeCorrelationPotentialFile exchange_correlation_potential_file(record);
        write_or_warn(directory + "/VXC", "VXC", [&](const std::string& path) {
            exchange_correlation_potential_file.write_to_file(path);
        });

        BerkeleyGWSelfConsistentPotentialFile self_consistent_potential_file(record);
        write_or_warn(directory + "/VSC", "VSC", [&](const std::string& path) {
            self_consistent_potential_file.write_to_file(path);
        });

        if (record.projector_source) {
            BerkeleyGWKleinmanBylanderProjectorFile projector_file(record);
            write_or_warn(directory + "/VKB", "VKB", [&](const std::string& path) {
                projector_file.write_to_file(path);
            });
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

    /// write_or_remove(), with a refusal logged instead of ending the export.
    template <typename Writer>
    static void
    write_or_warn(const std::string& path, const std::string& name, const Writer& write) {
        try {
            write_or_remove(path, write);
        } catch (const std::runtime_error& error) {
            log::warn("     {} not written: {}", name, error.what());
        }
    }
};

} // namespace q3
