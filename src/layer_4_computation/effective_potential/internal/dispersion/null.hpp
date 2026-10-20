#pragma once

/**
 * NullDispersionCalculator — the "no dispersion correction" scheme.
 *
 * Null Object implementation of DispersionCalculator: contributes exactly
 * zero energy, zero forces, and zero stress. Selected by
 * create_dispersion_calculator("none", ...) so that orchestrators (SCF
 * energy hookup, force and stress assembly) can call the dispersion
 * interface unconditionally instead of branching on the scheme name.
 */

#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/init.hpp"

namespace q3 {

class NullDispersionCalculator : public DispersionCalculator {
  public:
    explicit NullDispersionCalculator(const Crystal& crystal_structure)
        : number_of_atoms_(crystal_structure.number_of_atoms()) {}

    [[nodiscard]] Real compute_energy() const override {
        return 0.0;
    }

    [[nodiscard]] DispersionGradients compute_gradients() const override {
        return {std::vector<Vector3D>(static_cast<std::size_t>(number_of_atoms_), {0.0, 0.0, 0.0}),
                Matrix3x3{}};
    }

    [[nodiscard]] std::string name() const override {
        return "none";
    }

  private:
    int number_of_atoms_;
};

} // namespace q3
