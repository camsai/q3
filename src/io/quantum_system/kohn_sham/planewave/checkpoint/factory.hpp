#pragma once

/**
 * Factory for planewave Kohn-Sham quantum system checkpoint managers.
 *
 * Supported formats: "q3" (Q3 native binary, default).
 */

#include <memory>
#include <stdexcept>
#include <string>

#include "io/quantum_system/kohn_sham/planewave/checkpoint/init.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"

namespace q3 {

inline std::unique_ptr<KohnShamQuantumSystemCheckpoint>
create_quantum_system_checkpoint(const std::string& format = "q3") {
    if (format == "q3") {
        return std::make_unique<Q3PlanewaveKohnShamQuantumSystemCheckpoint>();
    }
    throw std::runtime_error("Unknown quantum system checkpoint format: " + format);
}

} // namespace q3
