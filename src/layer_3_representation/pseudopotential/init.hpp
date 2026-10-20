#pragma once

/**
 * Pseudopotential base class.
 *
 * Base for all pseudopotential types (norm-conserving, ultrasoft).
 * Contains common fields read from UPF files. Ultrasoft-specific fields
 * live in UltrasoftPseudopotential (ultrasoft/init.hpp).
 */

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/errors.hpp"
#include "layer_3_representation/pseudopotential/internal/charge_atomic.hpp"
#include "layer_3_representation/pseudopotential/internal/charge_core.hpp"
#include "layer_3_representation/pseudopotential/internal/potential_local.hpp"
#include "layer_3_representation/pseudopotential/internal/projector.hpp"
#include "layer_3_representation/pseudopotential/internal/wavefunction_atomic.hpp"

namespace q3 {

/// Pseudopotential type classification.
enum class PseudopotentialType {
    norm_conserving,
    ultrasoft,
    // Recognized for explicit rejection at input time; no PAW implementation exists yet.
    projector_augmented_wave,
};

/// Base class for all pseudopotential types.
class Pseudopotential {
  public:
    virtual ~Pseudopotential() = default;

    /// Clone this pseudopotential (deep copy preserving derived type).
    [[nodiscard]] virtual std::unique_ptr<Pseudopotential> clone() const {
        return std::make_unique<Pseudopotential>(*this);
    }

    /**
     * Return the type classification of this pseudopotential.
     * Overridden by UltrasoftPseudopotential to return ultrasoft. No base-class
     * instance can ever hold PAW data: both UPF readers throw
     * UnsupportedPseudopotentialTypeError before constructing/populating a
     * Pseudopotential from a PAW file, so projector_augmented_wave is reachable
     * only once a future PawPseudopotential subclass overrides this method.
     */
    [[nodiscard]] virtual PseudopotentialType pseudopotential_type_enum() const {
        return PseudopotentialType::norm_conserving;
    }

    /**
     * Whether this PP has augmentation charges.
     * Overridden by UltrasoftPseudopotential.
     */
    [[nodiscard]] virtual bool has_augmentation_charges() const {
        return false;
    }

    // ---- Common fields ----

    std::string element;
    std::string pseudopotential_type; // "NC", "US", "PAW" (PAW is rejected at read time)
    std::string xc_functional;
    Real valence_charge{};          // valence charge
    int maximum_angular_momentum{}; // max angular momentum
    int mesh_size{};
    int number_of_projectors{};    // number of beta projectors
    int number_of_wavefunctions{}; // number of atomic wavefunctions
    bool has_nonlinear_core_correction{false};

    std::vector<Real> radial_grid;
    std::vector<Real> radial_grid_derivative;
    PotentialLocal potential_local; // local potential V_loc(r) (Ry)

    std::vector<Projector> projectors;
    std::vector<std::vector<Real>> d_ion_matrix;          // D matrix (Ry)
    std::vector<WavefunctionAtomic> wavefunctions_atomic; // atomic wavefunctions
    ChargeAtomic charge_atomic;                           // atomic charge density ρ_at(r)
    ChargeCore charge_core;                               // core charge density (NLCC)
};

} // namespace q3
