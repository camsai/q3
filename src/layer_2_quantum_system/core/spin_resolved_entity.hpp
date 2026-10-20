#pragma once

/**
 * SpinResolvedEntity — base class for quantum entities with spin-resolved data.
 *
 * Provides common spin component count and spin-polarization query for
 * charge density, effective potential, and other spin-resolved quantities.
 * The number of spin components is set at construction time by subclasses.
 *
 * Convention:
 *   number_of_spin_components = 1  →  non-spin-polarized (LDA/GGA)
 *   number_of_spin_components = 2  →  spin-polarized (LSDA/spin-GGA)
 */

namespace q3 {

class SpinResolvedEntity {
  public:
    virtual ~SpinResolvedEntity() = default;

    [[nodiscard]] int number_of_spin_components() const {
        return number_of_spin_components_;
    }
    [[nodiscard]] bool is_spin_polarized() const {
        return number_of_spin_components_ >= 2;
    }

  protected:
    int number_of_spin_components_ = 1;
};

} // namespace q3
