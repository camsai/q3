#pragma once

/**
 * NormConservingPlanewaveKohnShamQuantumSystem — plane-wave pseudopotential quantum
 * system.
 *
 * Inherits electronic state data from PlanewaveKohnShamQuantumSystem.
 * All static pseudopotential data (local potential, beta projectors, atomic
 * charge density, NLCC core charge, projector mapping tables) is owned by the
 * NormConservingPlanewaveBasis, accessible via pseudopotential_basis().
 *
 * This class retains only:
 * - The d_matrix_element() virtual dispatch (NC uses bare D, US overrides with effective D)
 * - A pseudopotential_basis() accessor to the single source of truth
 *
 * Consumers should access all static PP data via pseudopotential_basis() directly.
 */

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/factory.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"

namespace q3 {

// Forward declarations
class Crystal;

/**
 * Norm-conserving planewave Kohn-Sham quantum system.
 *
 * Owns a NormConservingPlanewaveBasis (single source of truth for all static PP data).
 * Provides virtual d_matrix_element() dispatch (NC vs US).
 * All other PP data should be accessed via pseudopotential_basis().
 */
class NormConservingPlanewaveKohnShamQuantumSystem : public PlanewaveKohnShamQuantumSystem {
  public:
    NormConservingPlanewaveKohnShamQuantumSystem() = default;
    ~NormConservingPlanewaveKohnShamQuantumSystem() override = default;

    /**
     * Initialize by creating and populating the internal PP basis.
     */
    virtual void init(const Crystal& crystal_structure,
                      const PlanewaveBasis& planewave_basis,
                      const PseudopotentialSet& pseudopotential_set) {
        basis_ = &planewave_basis;
        norm_conserving_basis_ = NormConservingPlanewaveBasisFactory::create(
            crystal_structure, planewave_basis, pseudopotential_set);
    }

    // ─── NormConservingPlanewaveBasis access ────────────────────────

    /**
     * Access the underlying pseudopotential plane-wave basis (single source of truth).
     * All static PP data (local potential, projectors, beta functions, core charge,
     * atomic charge density, etc.) should be accessed through this.
     */
    [[nodiscard]] const NormConservingPlanewaveBasis& pseudopotential_basis() const {
        return *norm_conserving_basis_;
    }

    // ─── D-matrix virtual dispatch ──────────────────────────────────

    /**
     * Get D-matrix element for a pair of expanded projector channels.
     * For norm-conserving PPs, dispatches to the bare D-matrix (dion) when
     * the (l,m) indices match. For ultrasoft PPs, the ultrasoft subclass
     * overrides this to return the effective D-matrix element.
     */
    /**
     * D_ij for one projector pair, in one spin channel.
     *
     * Norm-conserving ignores `spin`: its D_ion comes straight from the dataset
     * and has no v_eff dependence, so both channels share it exactly. Ultrasoft
     * and PAW override this, where D = D_ion + ∫v_eff·Q_ij does depend on spin.
     */
    [[nodiscard]] virtual Real d_matrix_element(int atom_index,
                                                int species_type,
                                                int projector_index_i,
                                                int projector_index_j,
                                                int spin = 0) const {
        (void)spin;
        const auto& basis = *norm_conserving_basis_;
        int number_of_channels = basis.number_of_projector_channels(species_type);
        if (projector_index_i < number_of_channels && projector_index_j < number_of_channels) {
            int beta_a = basis.projector_to_beta_index(species_type, projector_index_i);
            int beta_b = basis.projector_to_beta_index(species_type, projector_index_j);
            if (basis.projector_combined_lm_index(species_type, projector_index_i) ==
                basis.projector_combined_lm_index(species_type, projector_index_j))
                return basis.bare_d_matrix_element(species_type, beta_a, beta_b);
        }
        return 0.0;
    }

    /**
     * Whether this quantum system uses ultrasoft pseudopotentials.
     * Consumers can use this to check if augmentation is needed.
     */
    [[nodiscard]] virtual bool has_ultrasoft() const {
        return false;
    }

    /**
     * Refuses, naming `purpose`, unless H ψ = E ψ is the whole eigenproblem.
     *
     * For a consumer that hands the Hamiltonian to a code with no notion of an
     * overlap operator, such as the BerkeleyGW `VKB` export. A norm-conserving
     * dataset leaves S = 1, so this accepts; the ultrasoft system, which PAW
     * runs use as well, overrides it to refuse.
     */
    virtual void require_identity_overlap(const std::string& purpose) const {
        (void)purpose;
    }

  protected:
    /// NC pseudopotential basis: local potential, beta projectors, core charge, atomic densities.
    std::unique_ptr<NormConservingPlanewaveBasis> norm_conserving_basis_;
};

} // namespace q3
