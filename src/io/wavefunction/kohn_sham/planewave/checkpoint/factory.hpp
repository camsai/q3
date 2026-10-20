#pragma once

/**
 * Factory for planewave Kohn-Sham wavefunction checkpoint writers.
 *
 * Supported formats: "qe" (Fortran unformatted, default).
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "io/wavefunction/kohn_sham/planewave/checkpoint/init.hpp"
#include "io/wavefunction/kohn_sham/planewave/checkpoint/qe.hpp"

namespace q3 {

inline std::unique_ptr<PlanewaveKohnShamWavefunctionCheckpoint>
create_wavefunction_checkpoint(const NormConservingPlanewaveKohnShamSCFState& scf_state,
                               const std::string& format = "qe") {
    if (format == "qe") {
        return std::make_unique<QEPlanewaveKohnShamWavefunctionCheckpoint>(scf_state);
    }
    throw std::runtime_error("Unknown wavefunction checkpoint format: " + format);
}

} // namespace q3
