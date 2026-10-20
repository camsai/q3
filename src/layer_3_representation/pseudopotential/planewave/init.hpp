#pragma once

/**
 * PlanewavePseudopotential — planewave representation of a pseudopotential.
 *
 * Wraps a basis-independent Pseudopotential and provides access to the
 * underlying data.  In the future, this class will own precomputed G-space
 * tables (β(q), V_loc(G), ρ_at(G), ρ_core(G)) that are currently
 * computed by the quantum system.
 *
 * This is tier 3 in the 4-tier pseudopotential hierarchy:
 *   Tier 1: Pseudopotential       (data from UPF, basis-independent)
 *   Tier 2: UltrasoftPseudopotential (adds US-specific data)
 *   Tier 3: PlanewavePseudopotential (planewave representation) ← this
 *   Tier 4: UltrasoftPlanewavePseudopotential (US + planewave)
 */

#include <memory>

#include "layer_3_representation/pseudopotential/init.hpp"

namespace q3 {

class PlanewavePseudopotential {
  public:
    explicit PlanewavePseudopotential(std::unique_ptr<Pseudopotential> pseudopotential)
        : pseudopotential_(std::move(pseudopotential)) {}

    virtual ~PlanewavePseudopotential() = default;

    /**
     * Access the underlying basis-independent pseudopotential data.
     */
    [[nodiscard]] const Pseudopotential& data() const {
        return *pseudopotential_;
    }

    /**
     * Return the type classification of the underlying pseudopotential.
     */
    [[nodiscard]] PseudopotentialType pseudopotential_type_enum() const {
        return pseudopotential_->pseudopotential_type_enum();
    }

    /**
     * Whether this PP has augmentation charges.
     */
    [[nodiscard]] bool has_augmentation_charges() const {
        return pseudopotential_->has_augmentation_charges();
    }

  protected:
    std::unique_ptr<Pseudopotential> pseudopotential_;
};

} // namespace q3
