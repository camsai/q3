#pragma once

/**
 * create_beta_function_table — pick the β(q) table implementation for a
 * pseudopotential set.
 *
 * The PAW table is selected when any species actually carries reciprocal-space
 * projectors — a capability test, not a format test, so a UPF-PAW dataset
 * (which supplies projectors radially) correctly gets the base table. This
 * factory is the sanctioned place for flavor dispatch; the base table header
 * stays free of PAW includes.
 */

#include <memory>

#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/paw.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

inline std::unique_ptr<BetaFunctionTable>
create_beta_function_table(const PseudopotentialSet& pseudopotential_set) {
    for (int index = 0; index < pseudopotential_set.size(); index++) {
        const auto* paw = dynamic_cast<const PawPseudopotential*>(&pseudopotential_set[index]);
        if (paw != nullptr && paw->has_reciprocal_projectors()) {
            return std::make_unique<PawBetaFunctionTable>();
        }
    }
    return std::make_unique<BetaFunctionTable>();
}

} // namespace q3
