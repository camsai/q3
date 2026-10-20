#pragma once

/**
 * UltrasoftPlanewavePseudopotential — tier 4 in the pseudopotential hierarchy.
 *
 * Extends PlanewavePseudopotential with ultrasoft-specific planewave
 * infrastructure:
 * - Augmentation charge Q_ij(q,L) interpolation tables
 * - Spherical harmonics table for G-vectors
 * - Gaunt coefficient table for L-resolved augmentation
 *
 * These members are currently initialized and owned by the ultrasoft
 * quantum system; ownership will be migrated here in a future step.
 * For now this class establishes the canonical type location.
 *
 * Hierarchy:
 *   Tier 1: Pseudopotential       (UPF data, basis-independent)
 *   Tier 2: UltrasoftPseudopotential (adds US-specific data)
 *   Tier 3: PlanewavePseudopotential (wraps Pseudopotential)
 *   Tier 4: UltrasoftPlanewavePseudopotential ← this
 */

#include <memory>

#include "layer_3_representation/pseudopotential/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

class UltrasoftPlanewavePseudopotential : public PlanewavePseudopotential {
  public:
    explicit UltrasoftPlanewavePseudopotential(
        std::unique_ptr<UltrasoftPseudopotential> ultrasoft_pseudopotential)
        : PlanewavePseudopotential(std::move(ultrasoft_pseudopotential)) {}

    ~UltrasoftPlanewavePseudopotential() override = default;

    /**
     * Access the underlying ultrasoft pseudopotential data.
     */
    [[nodiscard]] const UltrasoftPseudopotential& ultrasoft_data() const {
        return static_cast<const UltrasoftPseudopotential&>(*pseudopotential_);
    }
};

} // namespace q3
