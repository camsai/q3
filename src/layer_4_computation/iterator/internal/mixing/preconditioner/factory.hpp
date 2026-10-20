#pragma once

/**
 * PreconditionerFactory — creates SCF residual preconditioners by name.
 *
 * Supported names:
 *   - "none"      identity; no preconditioning.
 *   - "kerker"    diagonal G²/(G²+q0²) filter (average Thomas-Fermi screening).
 *   - "local-tf"  Thomas-Fermi screening from the local density, solved by
 *                 conjugate gradients.
 *
 * Mirrors MixerFactory: an unknown name throws rather than silently selecting a
 * default, so a typo fails loudly instead of turning preconditioning off.
 */

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/preconditioner/init.hpp"
#include "layer_4_computation/iterator/internal/mixing/preconditioner/kerker.hpp"
#include "layer_4_computation/iterator/internal/mixing/preconditioner/local_thomas_fermi.hpp"
#include "layer_4_computation/iterator/internal/mixing/preconditioner/none.hpp"

namespace q3 {

/// Everything any preconditioner needs from the surrounding representation.
struct PreconditionerEnvironment {
    std::vector<Real> g_squared; ///< |G|² per G-vector, in bohr⁻².
    Real q0_squared = 1.0;       ///< Screening wavevector squared / vacuum floor, in bohr⁻².
    int first_nonzero_g_index = 0;
    std::function<void(const std::vector<Real>&, std::vector<Complex>&)> to_reciprocal_space;
    std::function<void(const std::vector<Complex>&, std::vector<Real>&)> to_real_space;
    /// Sums real-space integrals over the ranks sharing this pool's grid.
    std::function<void(std::vector<Real>&)> global_reduce;
};

namespace PreconditionerFactory {

/// The names create() accepts, for validation and error messages.
inline bool is_supported(const std::string& name) {
    return name == "none" || name == "kerker" || name == "local-tf";
}

/// Create a residual preconditioner by name.
inline std::unique_ptr<Preconditioner> create(const std::string& name,
                                              const PreconditionerEnvironment& environment) {
    if (name == "none") {
        return std::make_unique<NoPreconditioner>();
    }
    if (name == "kerker") {
        return std::make_unique<KerkerPreconditioner>(
            environment.g_squared, environment.q0_squared, environment.first_nonzero_g_index);
    }
    if (name == "local-tf") {
        return std::make_unique<LocalThomasFermiPreconditioner>(environment.g_squared,
                                                                environment.q0_squared,
                                                                environment.first_nonzero_g_index,
                                                                environment.to_reciprocal_space,
                                                                environment.to_real_space,
                                                                environment.global_reduce);
    }
    throw std::runtime_error("Unknown mixing preconditioner: '" + name +
                             "'. Supported values: 'none', 'kerker', 'local-tf' "
                             "(or 'default' to choose by smearing).");
}

} // namespace PreconditionerFactory
} // namespace q3
