#pragma once

/**
 * How a PAW dataset's one-center terms are referenced.
 *
 * PAW is exact, but the split between what a dataset's `D_ion` already carries
 * and what the SCF adds on top is a *convention*, and the two codes Q3 is
 * validated against made different choices. Reading a dataset produced for one
 * convention and computing with the other is wrong by a per-element constant —
 * large enough to dominate any energy difference of interest — while every
 * observable that does not depend on the reference (eigenvalues, Ewald,
 * forces) agrees perfectly, which is exactly what makes the error hard to see.
 *
 * This class is the seam. The base is QE's convention, which is also the
 * absolute one: the one-center energy is E¹ − Ẽ¹ computed outright, `PP_DIJ`
 * already contains the kinetic and core-reference pieces, and no correction is
 * applied. Everything it does is nothing, deliberately — a UPF-PAW dataset
 * needs no adjustment, and that path is validated against QE to under
 * 10 meV/atom on Si, Cu and C.
 *
 * A convention for datasets generated differently overrides it and is
 * registered under its own name (registry.hpp). The factory picks the one the
 * dataset's own declared `one_center_convention` names, never from the file
 * format, so a dataset that arrives by some other route still gets the
 * treatment its numbers were generated for.
 */

#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/paw/one_center/angular_grid.hpp"
#include "layer_4_computation/paw/one_center/densities.hpp"
#include "layer_4_computation/paw/one_center/energy.hpp"

namespace q3 {
namespace paw_one_center {

/// Everything the engine needs to rebuild a dataset's own reference state.
struct ReferenceStateInputs {
    const PawPseudopotential* pseudopotential{nullptr};
    const ChannelExpansion* expansion{nullptr};
    const GauntCoefficientTable* gaunt{nullptr};
    const std::vector<AngularPoint>* angular_grid{nullptr};
    const std::vector<std::vector<Real>>* harmonics{nullptr};
    const std::vector<std::vector<std::array<Real, 3>>>* harmonic_gradients{nullptr};
    const XCFunctional* functional{nullptr};
};

/**
 * The absolute (QE / UPF-PAW) convention: compute E¹ − Ẽ¹ and add it.
 *
 * Every method is a no-op. Kept as a real class rather than a null pointer so
 * that the SCF driver has one code path, and so every other convention is
 * expressed as a difference from a stated baseline rather than as a branch.
 */
class PawOneCenterConvention {
  public:
    virtual ~PawOneCenterConvention() = default;

    [[nodiscard]] virtual std::string name() const {
        return "absolute";
    }

    /// Extra terms this convention wants in the one-center energy. None here.
    [[nodiscard]] virtual OneCenterOptions options() const {
        return {};
    }

    /// Build any per-species reference state. Called once, at setup.
    virtual void prepare(const ReferenceStateInputs&) {}

    /**
     * Correction added to the one-center energy for the current occupancies.
     *
     * Zero here. A referencing convention returns −Σ_ij ρ_ij D^ref_ij, which
     * is what makes D remain the exact derivative of the energy after the D
     * correction below is applied.
     */
    [[nodiscard]] virtual Real energy_correction(const OnSiteOccupancies&) const {
        return 0.0;
    }

    /// Correction applied to D_ij in place. None here.
    virtual void correct_d_matrix(std::vector<Real>&) const {}

    /**
     * Absolute one-center Hxc energy at the dataset's atomic reference state.
     *
     * This is a radial setup diagnostic, not an atomic total energy a dataset
     * may state for output.
     */
    [[nodiscard]] virtual Real reference_one_center_energy() const {
        return 0.0;
    }

    /**
     * One-center double counting at the atomic reference state.
     *
     * For an atomic-referenced convention this is
     * `E_Hxc[rho_ref] - rho_ref D_Hxc[rho_ref]`, including the AE-minus-PS and
     * core-only conventions.
     */
    [[nodiscard]] virtual Real reference_double_counting_energy() const {
        return 0.0;
    }
};

} // namespace paw_one_center
} // namespace q3
