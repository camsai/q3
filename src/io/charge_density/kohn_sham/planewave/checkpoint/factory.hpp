#pragma once

/**
 * Factory for planewave Kohn-Sham charge density checkpoint writers.
 *
 * Supported formats: "qe" (Fortran unformatted, default).
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "io/charge_density/kohn_sham/planewave/checkpoint/init.hpp"
#include "io/charge_density/kohn_sham/planewave/checkpoint/qe.hpp"

namespace q3 {

inline std::unique_ptr<PlanewaveKohnShamChargeDensityCheckpoint>
create_charge_density_checkpoint(const NormConservingPlanewaveKohnShamSCFState& scf_state,
                                 const std::string& format = "qe") {
    if (format == "qe") {
        return std::make_unique<QEPlanewaveKohnShamChargeDensityCheckpoint>(scf_state);
    }
    throw std::runtime_error("Unknown charge density checkpoint format: " + format);
}

} // namespace q3
