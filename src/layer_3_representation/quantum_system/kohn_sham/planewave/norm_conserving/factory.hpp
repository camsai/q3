#pragma once

/**
 * Factory to create the correct NormConservingPlanewaveKohnShamQuantumSystem subclass
 * based on the types of pseudopotential_set provided (NC or USPP).
 */

#include <memory>
#include <vector>

#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"

namespace q3 {

class Crystal;
class PlanewaveBasis;

/// Creates the appropriate pseudopotential quantum system snapshot (NC or USPP).
struct NormConservingPlanewaveKohnShamQuantumSystemFactory {
    static std::unique_ptr<NormConservingPlanewaveKohnShamQuantumSystem>
    create(const Crystal& crystal_structure,
           const PlanewaveBasis& planewave_basis,
           const PseudopotentialSet& pseudopotential_set) {

        /**
         * Capability, not type: a PAW dataset carries the same augmentation
         * apparatus and needs the same quantum system, so a `== ultrasoft`
         * test would route it to the norm-conserving one and drop its
         * augmentation charges without a word.
         */
        bool needs_augmentation = pseudopotential_set.has_augmentation_charges();

        std::unique_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
        if (needs_augmentation) {
            quantum_system = std::make_unique<UltrasoftPlanewaveKohnShamQuantumSystem>();
        } else {
            quantum_system = std::make_unique<NormConservingPlanewaveKohnShamQuantumSystem>();
        }

        quantum_system->init(crystal_structure, planewave_basis, pseudopotential_set);
        return quantum_system;
    }

    /**
     * Create a shared quantum system snapshot for ownership sharing between
     * the iterator and the SCF state (avoids redundant PP basis initialization).
     */
    static std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem>
    create_shared(const Crystal& crystal_structure,
                  const PlanewaveBasis& planewave_basis,
                  const PseudopotentialSet& pseudopotential_set) {
        return std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem>(
            create(crystal_structure, planewave_basis, pseudopotential_set).release());
    }
};

} // namespace q3
