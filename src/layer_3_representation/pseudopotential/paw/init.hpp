#pragma once

/**
 * PawPseudopotential — projector-augmented-wave pseudopotential subclass.
 *
 * Extends UltrasoftPseudopotential rather than Pseudopotential directly: a PAW
 * dataset carries the same augmentation apparatus an ultrasoft one does (UPF
 * PAW files declare is_ultrasoft="T" and tabulate PP_Q / PP_QIJL identically),
 * so the augmentation storage and the machinery built on it are shared rather
 * than duplicated. What PAW adds on top is the one-center data that makes the
 * transformation exact: the all-electron and pseudo partial waves, the
 * all-electron core density and local potential, and the multipole moments of
 * the compensation charge.
 *
 * Reading a PAW dataset is supported; running an SCF with one is not yet.
 * load_pseudopotentials() rejects PAW at the point pseudopotentials enter a
 * calculation, so a PAW file can never be silently treated as NC or US by the
 * downstream factories (which dispatch on pseudopotential_type_enum()).
 */

#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/internal/charge_core.hpp"
#include "layer_3_representation/pseudopotential/internal/partial_wave.hpp"
#include "layer_3_representation/pseudopotential/internal/potential_local.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

/// PAW pseudopotential — extends ultrasoft with one-center (on-site) data.
class PawPseudopotential : public UltrasoftPseudopotential {
  public:
    /**
     * Name of the one-center convention this dataset's D_ion and one-center
     * data are expressed in, resolved through the convention registry
     * (layer_4_computation/paw/one_center/convention/registry.hpp).
     *
     * Not a detail of the file format but a statement about what the numbers
     * in it *mean*, and conventions are not interchangeable. The default,
     * "absolute", is QE's convention, which UPF-PAW datasets follow: `PP_DIJ`
     * already contains the kinetic and core-reference pieces, and the SCF adds
     * an absolute one-center Hxc energy on top. A format whose D_ion is
     * instead referenced to the dataset's own atomic configuration (the
     * linearized frozen-core formulation of Kresse & Joubert, PRB 59, 1758
     * (1999)) has its reader record the name of the convention that
     * re-references it, registered alongside that reader.
     *
     * Pairing one dataset's numbers with another convention's formulation is
     * silently wrong by a per-element constant, which is why the reader
     * records it on the dataset rather than it being inferred at the point of
     * use.
     */
    std::string one_center_convention{"absolute"};

    ~PawPseudopotential() override = default;

    [[nodiscard]] std::unique_ptr<Pseudopotential> clone() const override {
        return std::make_unique<PawPseudopotential>(*this);
    }

    [[nodiscard]] PseudopotentialType pseudopotential_type_enum() const override {
        return PseudopotentialType::projector_augmented_wave;
    }

    [[nodiscard]] bool has_augmentation_charges() const override {
        return true;
    }

    // ---- One-center basis ----

    /// All-electron partial waves φ_i(r), one per projector channel.
    std::vector<PartialWave> partial_waves_all_electron;
    /// Pseudo partial waves φ̃_i(r), one per projector channel.
    std::vector<PartialWave> partial_waves_pseudo;
    /// Reference occupation of each projector channel in the generating atom.
    std::vector<Real> channel_occupations;

    // ---- One-center densities and potential ----

    /// All-electron core density (distinct from the pseudized NLCC core density).
    ChargeCore charge_core_all_electron;
    /// All-electron local potential (Ry); diverges as -2Z/r toward the origin.
    PotentialLocal potential_local_all_electron;

    // ---- Compensation-charge moments ----

    /**
     * Multipole moments of the augmentation channels, indexed
     * [angular_momentum][projector_i][projector_j]:
     *
     *     multipole_moments[L][i][j] = ∫ (φ_i φ_j − φ̃_i φ̃_j) r^L dr
     *
     * with r-weighted partial waves. Entries forbidden by the angular-momentum
     * triangle condition (|l_i − l_j| ≤ L ≤ l_i + l_j with l_i + l_j + L even)
     * are stored as zero. The L = 0 block equals augmentation_overlap_integrals.
     */
    std::vector<std::vector<std::vector<Real>>> multipole_moments;

    // ---- Augmentation metadata ----

    /// Shape function used to pseudize the augmentation charges (e.g. "BESSEL").
    std::string augmentation_shape;
    /// Radius beyond which the augmentation charges vanish (bohr).
    Real augmentation_cutoff_radius{};
    /// Grid index corresponding to augmentation_cutoff_radius.
    int augmentation_cutoff_radius_index{};
    /// Highest angular momentum carried by the augmentation channels.
    int maximum_angular_momentum_augmentation{};
    /// Number of angular-momentum channels, 2·l_max + 1.
    int number_of_augmentation_angular_momenta{};

    /// One-center core energy, when the dataset provides it.
    Real core_energy{};
    bool has_core_energy{false};

    // ---- Projectors supplied in reciprocal space ----

    /**
     * Some dataset formats tabulate the projectors in reciprocal space rather
     * than radially. Q3 also applies projectors in reciprocal space
     * (BetaFunctionTable tabulates β(q) and the engine uses that), so such a
     * dataset is best consumed directly instead of being transformed to r and
     * straight back again.
     *
     * The values are β(q) in the convention of Q3's own forward transform,
     * β(q) = 4π ∫ β(r) j_l(qr) r² dr, so they need no rescaling. The reader
     * that fills this field records how that agreement was established.
     *
     * Empty when the dataset gives projectors radially (UPF), in which case
     * `projectors` carries r·β(r) and the transform is used as before.
     *
     * Tabulated on a uniform grid q_i = i · reciprocal_grid_spacing, one entry
     * per projector channel.
     */
    std::vector<std::vector<Real>> projectors_reciprocal;
    Real reciprocal_grid_spacing{};

    [[nodiscard]] bool has_reciprocal_projectors() const {
        return !projectors_reciprocal.empty() && reciprocal_grid_spacing > 0.0;
    }

    /**
     * The atomic valence density ρ_at(q), when the dataset gives it
     * reciprocally rather than radially.
     *
     * Same situation as the projectors, and the same resolution. Q3 wants this
     * quantity in q-space too — the starting density is built by superposing
     * ρ_at(G) — so the table is consumed as given instead of being inverted to
     * r and transformed straight back.
     *
     * The convention matches Q3's radial path, ρ_at(q) = ∫ ρ_at(r) j_0(qr) dr,
     * whose q = 0 value is the valence charge, so no rescaling.
     *
     * Reconstructing it from the partial waves is not an alternative: a
     * dataset's reference occupancies need not be the neutral atom's, and then
     * Σ_i f_i |φ̃_i|² does not integrate to the valence charge.
     *
     * Empty when the dataset gives the density radially (UPF), in which case
     * `charge_atomic` carries it and the transform is used as before.
     *
     * Tabulated on a uniform grid q_i = i · reciprocal_charge_grid_spacing,
     * which need not be the projector grid.
     */
    std::vector<Real> charge_atomic_reciprocal;
    Real reciprocal_charge_grid_spacing{};

    [[nodiscard]] bool has_reciprocal_atomic_charge() const {
        return !charge_atomic_reciprocal.empty() && reciprocal_charge_grid_spacing > 0.0;
    }

    /**
     * The local pseudopotential V_loc(q), when the dataset gives it
     * reciprocally.
     *
     * This is *not* the same object as the radial `potential_local`. A format
     * may describe the potential inside the augmentation sphere radially and,
     * separately, tabulate the pseudized local potential the plane waves see;
     * using the radial one in its place is simply the wrong potential, not a
     * coarser version of the right one.
     *
     * Stored in the dataset's own convention, which the table itself makes
     * checkable: the Coulomb divergence is already subtracted, so the entry
     * at q = 0 is finite and equals the α_Z constant. Q3's own V_loc(q) keeps
     * that divergence, so the consumer subtracts 4πZe²/(Ωq²) back on — see
     * calculate_local_potential_at_q.
     *
     * Units are energy × volume, so V(G) is the tabulated value at |G| divided
     * by Ω; stored in Ry·bohr³.
     *
     * Empty for UPF datasets, which give the potential radially.
     */
    std::vector<Real> potential_local_reciprocal;
    Real reciprocal_potential_grid_spacing{};

    [[nodiscard]] bool has_reciprocal_local_potential() const {
        return !potential_local_reciprocal.empty() && reciprocal_potential_grid_spacing > 0.0;
    }

    /**
     * The pseudized partial core density ñ_c(q), when the dataset gives it
     * reciprocally.
     *
     * The third quantity a format may tabulate in q-space for the plane-wave
     * grid, and the third that must not be taken from radial data instead: a
     * radial pseudized core density kept for the one-center terms can be a
     * different object on a truncated grid, carrying a different charge.
     *
     * Getting it wrong is not a constant shift. The grid-side XC is evaluated
     * at n + ñ_c and is nonlinear in it, so an error here changes the energy's
     * dependence on volume — which is exactly how it shows up: as a
     * plausible-looking but wrong equation of state.
     *
     * Same convention as the radial path produces, 4π∫ñ_c(r)r²j₀(qr)dr, whose
     * q = 0 value is the partial core charge.
     *
     * Empty for UPF datasets.
     */
    std::vector<Real> charge_core_reciprocal;
    Real reciprocal_core_grid_spacing{};

    [[nodiscard]] bool has_reciprocal_core_charge() const {
        return !charge_core_reciprocal.empty() && reciprocal_core_grid_spacing > 0.0;
    }
};

} // namespace q3
