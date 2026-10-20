#pragma once

/**
 * NormConservingPlanewaveBasisFactory — creates NC or US pseudopotential basis.
 */

#include <memory>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

struct NormConservingPlanewaveBasisFactory {
    static std::unique_ptr<NormConservingPlanewaveBasis>
    create(const Crystal& crystal_structure,
           const PlanewaveBasis& planewave_basis,
           const PseudopotentialSet& pseudopotential_set) {
        // Capability, not type — see PseudopotentialSet::has_augmentation_charges().
        bool needs_augmentation = pseudopotential_set.has_augmentation_charges();

        std::unique_ptr<NormConservingPlanewaveBasis> basis;
        if (needs_augmentation) {
            basis = std::make_unique<UltrasoftPlanewaveBasis>();
        } else {
            basis = std::make_unique<NormConservingPlanewaveBasis>();
        }
        basis->init(crystal_structure, planewave_basis, pseudopotential_set);
        return basis;
    }
};

} // namespace q3
