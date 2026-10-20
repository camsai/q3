#pragma once

/**
 * create_subspace_selection — the settings decide the strategy, here and
 * nowhere else. Mirrors create_overlap_operator(): the dispatch is on what
 * was asked for, and the builder holds only the interface.
 */

#include <memory>

#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/isolated_manifold_selection.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/projectability_disentanglement_selection.hpp"

namespace q3 {

[[nodiscard]] inline std::unique_ptr<SubspaceSelection>
create_subspace_selection(const WannierConstructionSettings& settings) {
    if (settings.disentangles()) {
        return std::make_unique<ProjectabilityDisentanglementSelection>(*settings.disentanglement);
    }
    return std::make_unique<IsolatedManifoldSelection>();
}

} // namespace q3
