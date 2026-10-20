#pragma once

/**
 * create_overlap_operator — selects the overlap operator for the eigenproblem.
 *
 * Norm-conserving pseudopotential_set get the identity overlap (standard eigenproblem);
 * ultrasoft pseudopotential_set get the augmentation overlap (generalized eigenproblem).
 */

#include <memory>

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

inline std::unique_ptr<OverlapOperator>
create_overlap_operator(const Crystal& crystal_structure,
                        const NormConservingPlanewaveKohnShamQuantumSystem& quantum_system,
                        const Matrix<Complex>& kleinman_bylander_potential,
                        const execution::ExecutionPolicy& execution_policy) {
    if (quantum_system.has_ultrasoft()) {
        return std::make_unique<UltrasoftPseudopotentialOverlapOperator>(
            crystal_structure, quantum_system, kleinman_bylander_potential, execution_policy);
    }
    return std::make_unique<PseudopotentialOverlapOperator>(
        crystal_structure, quantum_system, kleinman_bylander_potential, execution_policy);
}

} // namespace q3
