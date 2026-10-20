#pragma once

/**
 * PlanewavePseudopotentialKohnShamSCFIteratorFactory — dispatch NC vs ultrasoft.
 *
 * Inspects the pseudopotential set to decide which concrete iterator to
 * instantiate. The order matters: PAW is checked first because a PAW dataset
 * also answers yes to `has_augmentation_charges()` — it is an ultrasoft
 * dataset plus one-center data — so an augmentation-first test would route it
 * to the ultrasoft iterator and silently drop every one-center term.
 *
 * Config-based creation path for integration tests. Production code should
 * use create_kohn_sham_iterator(InputData) from the KS factory.
 */

#include <memory>

#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/paw/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

/**
 * Create a planewave pseudopotential iterator from a pre-built config.
 *
 * The caller is responsible for the lifetimes of crystal_structure,
 * kpoint_grid, and pseudopotential_set in the config.
 */
inline std::unique_ptr<PlanewavePseudopotentialKohnShamSCFIterator>
create_planewave_pseudopotential_iterator(KohnShamSCFIteratorConfig config) {
    const auto* pseudopotential_set = config.pseudopotential_set;
    if (pseudopotential_set->has_pseudopotential_type(
            PseudopotentialType::projector_augmented_wave)) {
        return std::make_unique<PawPlanewavePseudopotentialKohnShamSCFIterator>(std::move(config));
    }
    if (pseudopotential_set->has_augmentation_charges()) {
        return std::make_unique<UltrasoftPlanewavePseudopotentialKohnShamSCFIterator>(
            std::move(config));
    }
    return std::make_unique<PlanewavePseudopotentialKohnShamSCFIterator>(std::move(config));
}

} // namespace q3
