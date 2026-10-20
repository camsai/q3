#pragma once

/**
 * UltrasoftPseudopotential — ultrasoft pseudopotential subclass.
 *
 * Extends the base Pseudopotential with ultrasoft-specific fields:
 * augmentation functions Q_ij(r), Q_ij^L(r), augmentation overlap
 * integrals, inner cutoff radii, and augmentation coefficients.
 *
 * These fields are populated by the UPF readers when reading an
 * ultrasoft pseudopotential file.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"

namespace q3 {

/// Ultrasoft pseudopotential — extends base with augmentation data.
class UltrasoftPseudopotential : public Pseudopotential {
  public:
    ~UltrasoftPseudopotential() override = default;

    [[nodiscard]] std::unique_ptr<Pseudopotential> clone() const override {
        return std::make_unique<UltrasoftPseudopotential>(*this);
    }

    [[nodiscard]] PseudopotentialType pseudopotential_type_enum() const override {
        return PseudopotentialType::ultrasoft;
    }

    [[nodiscard]] bool has_augmentation_charges() const override {
        return true;
    }

    // ---- Ultrasoft-specific fields ----

    std::vector<std::vector<Real>> augmentation_function; // Q_ij(r) total
    std::vector<std::vector<std::vector<Real>>>
        augmentation_function_by_angular_momentum; // Q_ij^L(r)
    int number_of_q_angular_momenta{0};
    std::vector<std::vector<Real>> augmentation_overlap_integrals; // ∫ Q_ij dr
    int number_of_q_coefficients{0};
    std::vector<Real> inner_cutoff_radii;
    std::vector<std::vector<std::vector<Real>>> augmentation_coefficients;
};

} // namespace q3
