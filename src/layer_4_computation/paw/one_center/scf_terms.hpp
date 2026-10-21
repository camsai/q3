#pragma once

/**
 * The one-center engine's SCF-facing driver.
 *
 * Everything below this file works on one atom and one occupancy matrix. This
 * is what an SCF loop actually needs: given the projector overlap coefficients
 * the ultrasoft pathway already accumulates each step, produce
 *
 *   * Σ_a (E¹_a − Ẽ¹_a), the one-center contribution to the total energy, and
 *   * (D¹ − D̃¹)_Hxc per atom, added to the D the ultrasoft pathway built.
 *
 * Both come from the same call, because they come from the same potential: the
 * D matrix integrates against the branch potentials the energy computed, which
 * is what makes D ≡ ∂E/∂ρ_ij hold at the level the SCF sees rather than only
 * inside the module's own gate.
 *
 * **The projector overlap coefficients are ρ_ij.** The ultrasoft iterator
 * accumulates Σ_nk f_nk ⟨ψ|p̃_i⟩⟨p̃_j|ψ⟩ per atom over expanded (lm-resolved)
 * channels, and that array *is* the PAW on-site occupancy matrix — not
 * something convertible into it. What has to be true for that to be usable is
 * that both sides expand
 * radial channels into lm in the same order; they do, both walking channels in
 * file order and lm in `harmonic_order_sequence`, and
 * `channel_expansion_matches_projectors` states it as a checkable
 * predicate rather than a comment, because a silent mismatch here produces a
 * plausible, converged, wrong answer.
 *
 * Per-species work — the angular grid, the harmonics, the Gaunt table, the
 * channel expansion — is built once at construction. Only the per-atom radial
 * work repeats each step, which is why the one-center cost stays negligible
 * next to the plane-wave side.
 */

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/paw/one_center/angular_grid.hpp"
#include "layer_4_computation/paw/one_center/convention/factory.hpp"
#include "layer_4_computation/paw/one_center/d_matrix.hpp"
#include "layer_4_computation/paw/one_center/densities.hpp"
#include "layer_4_computation/paw/one_center/energy.hpp"
#include "layer_4_computation/paw/one_center/gradient.hpp"

namespace q3 {
namespace paw_one_center {

/**
 * Whether a dataset's one-center channel expansion agrees with the order the
 * plane-wave side accumulates projector overlaps in.
 *
 * The two are built independently — `expand_channels` from the partial waves,
 * the projector mapping table from the β projectors — so their agreement is a
 * property of the dataset, not of the code. A PAW file that paired them
 * differently would make every ρ_ij element land in the wrong place while
 * still producing a symmetric, plausible matrix.
 */
inline bool channel_expansion_matches_projectors(const PawPseudopotential& pseudopotential) {
    if (static_cast<int>(pseudopotential.projectors.size()) != pseudopotential.number_of_projectors)
        return false;
    for (int channel = 0; channel < pseudopotential.number_of_projectors; channel++) {
        if (pseudopotential.projectors[channel].angular_momentum !=
            pseudopotential.partial_waves_pseudo[channel].angular_momentum)
            return false;
    }
    return true;
}

/// Everything reusable across SCF steps for one species.
struct SpeciesOneCenterContext {
    const PawPseudopotential* pseudopotential{nullptr};
    ChannelExpansion expansion;
    GauntCoefficientTable gaunt;
    std::vector<AngularPoint> angular_grid;
    std::vector<std::vector<Real>> harmonics;
    std::vector<std::vector<std::array<Real, 3>>> harmonic_gradients;
    bool gradient_corrected{false};

    /**
     * How this dataset's one-center numbers are referenced. Owned per species
     * because it caches that species' reference state.
     */
    std::shared_ptr<PawOneCenterConvention> convention;

    [[nodiscard]] int number_of_channels() const {
        return expansion.size();
    }
};

/**
 * One-center energy and D correction for every atom, refreshed per SCF step.
 *
 * Constructed once per calculation; `update()` is the per-step call. Species
 * that are not PAW are simply absent from the context list and contribute
 * nothing, so a mixed PAW/ultrasoft cell needs no special case.
 */
class PawOneCenterTerms {
  public:
    /**
     * @param angular_order  drives the Gauss(θ)×uniform(φ) grid size; must be
     *                       at least the largest density L, and is set from the
     *                       dataset rather than fixed here
     */
    void build(const Crystal& crystal_structure,
               const PseudopotentialSet& pseudopotential_set,
               const XCFunctional& functional,
               int angular_order = 0) {
        functional_ = &functional;
        int number_of_species = pseudopotential_set.size();
        contexts_.assign(number_of_species, SpeciesOneCenterContext{});

        for (int species_index = 0; species_index < number_of_species; species_index++) {
            const auto* paw =
                dynamic_cast<const PawPseudopotential*>(&pseudopotential_set[species_index]);
            if (paw == nullptr)
                continue;
            if (!channel_expansion_matches_projectors(*paw))
                throw std::runtime_error(
                    "PAW dataset for " + paw->element +
                    " pairs its projectors and partial waves in different orders, so the "
                    "projector overlap coefficients cannot be read as on-site occupancies");

            auto& context = contexts_[species_index];
            context.pseudopotential = paw;
            context.expansion = expand_channels(*paw);

            int maximum_angular_momentum = 0;
            for (const auto& wave : paw->partial_waves_pseudo)
                maximum_angular_momentum =
                    std::max(maximum_angular_momentum, wave.angular_momentum);
            context.gaunt = generate_gaunt_coefficients(maximum_angular_momentum);

            int number_of_lm = density_lm_count(*paw);
            context.gradient_corrected = functional.requires_density_gradient();
            /**
             * The grid must integrate products of harmonics up to the density's
             * own largest L (2·l_max). GGA raises the effective angular order
             * by one because the density gradient couples neighboring angular
             * momenta. Without the increment, a d-channel density received
             * only 15 azimuthal points rather than 18 and its highest gradient
             * modes could alias. The dataset still drives the order unless the
             * caller explicitly asks for more.
             */
            int density_order = 2 * maximum_angular_momentum;
            int functional_order = density_order + (context.gradient_corrected ? 1 : 0);
            int order = std::max(angular_order, functional_order);
            context.angular_grid = build_angular_grid(order);
            context.harmonics = tabulate_harmonics(context.angular_grid, number_of_lm);

            if (context.gradient_corrected) {
                context.harmonic_gradients =
                    tabulate_harmonic_gradients(context.angular_grid, number_of_lm);
            }

            /**
             * The convention is prepared last because building its reference
             * state runs the full one-center engine once, which needs every
             * table above already in place.
             */
            context.convention = create_one_center_convention(*paw);
            ReferenceStateInputs inputs;
            inputs.pseudopotential = paw;
            inputs.expansion = &context.expansion;
            inputs.gaunt = &context.gaunt;
            inputs.angular_grid = &context.angular_grid;
            inputs.harmonics = &context.harmonics;
            inputs.harmonic_gradients =
                context.gradient_corrected ? &context.harmonic_gradients : nullptr;
            inputs.functional = &functional;
            context.convention->prepare(inputs);
            log::info("     PAW: {} one-center convention for {} (atomic Hxc {:.9f} Ry, "
                      "reference double counting {:.9f} Ry)",
                      context.convention->name(),
                      paw->element,
                      context.convention->reference_one_center_energy(),
                      context.convention->reference_double_counting_energy());
        }

        int number_of_atoms = crystal_structure.number_of_atoms();
        d_correction_by_spin_.assign(1, std::vector<std::vector<Real>>(number_of_atoms));
        energy_ = 0.0;
    }

    [[nodiscard]] bool is_active() const {
        for (const auto& context : contexts_) {
            if (context.pseudopotential != nullptr)
                return true;
        }
        return false;
    }

    /**
     * Recompute the one-center terms from the current occupancies.
     *
     * @param crystal_structure            atom → species map
     * @param projector_overlap_coefficients  ρ_ij: [atom][i * stride + j]
     * @param stride                       row length of that array, which is the
     *                                     *maximum* channel count over species,
     *                                     not this atom's own
     */
    void update(const Crystal& crystal_structure,
                const Matrix<Real>& projector_overlap_coefficients,
                int stride) {
        update_by_spin(crystal_structure, {&projector_overlap_coefficients}, stride);
    }

    /**
     * The spin-polarized form: one occupancy array per channel.
     *
     * Exchange-correlation is nonlinear in the spin density, so an atom's
     * on-site XC has to see ρ↑ and ρ↓ separately rather than their sum. Passing
     * a single array reproduces the unpolarized result exactly, which is what
     * `update` above does.
     *
     * @param projector_overlap_coefficients_by_spin  ρ_ij per channel, each
     *                                                [atom][i * channels + j]
     * @param row_capacity  allocated row length, used only to bound the read
     */
    void
    update_by_spin(const Crystal& crystal_structure,
                   const std::vector<const Matrix<Real>*>& projector_overlap_coefficients_by_spin,
                   int row_capacity) {
        int number_of_spins = static_cast<int>(projector_overlap_coefficients_by_spin.size());
        energy_ = 0.0;
        branch_totals_ = BranchTotals{};
        int number_of_atoms = crystal_structure.number_of_atoms();
        d_correction_by_spin_.assign(number_of_spins,
                                     std::vector<std::vector<Real>>(number_of_atoms));

        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            int species_index = crystal_structure.species_type_index(atom_index);
            const auto& context = contexts_[species_index];
            if (context.pseudopotential == nullptr)
                continue;

            int number_of_channels = context.number_of_channels();
            /**
             * The row stride is this **atom's own** channel count, not the
             * maximum over species.
             *
             * Rows are *allocated* max_channels² wide so a mixed cell's array
             * is rectangular, but the accumulation writes at
             * `i * channels(atom) + j` — and `add_augmentation_charges` reads it back
             * the same way. Using the allocated width here instead would shear
             * every row belonging to a species narrower than the widest one:
             * the one-center energy and D come out wrong for a mixed-species
             * PAW cell (SiC, NiO, any oxide) and the run still converges.
             */
            if (number_of_channels > row_capacity)
                throw std::runtime_error("PAW one-center channel count exceeds the allocated "
                                         "projector overlap row width");

            auto occupancies = OnSiteOccupancies::zeros(number_of_channels, number_of_spins);
            for (int spin = 0; spin < number_of_spins; spin++) {
                const auto& coefficients_for_spin = *projector_overlap_coefficients_by_spin[spin];
                for (int first = 0; first < number_of_channels; first++)
                    for (int second = 0; second < number_of_channels; second++)
                        occupancies.at(spin, first, second) =
                            coefficients_for_spin[atom_index][first * number_of_channels + second];
            }

            const auto& paw = *context.pseudopotential;
            if (number_of_spins == 1) {
                auto densities = build_one_center_densities(
                    paw, context.expansion, occupancies, context.gaunt, 0);
                auto energy = compute_one_center_energy_either_mode(
                    paw,
                    densities,
                    context.angular_grid,
                    context.harmonics,
                    *functional_,
                    context.gradient_corrected ? &context.harmonic_gradients : nullptr,
                    context.convention->options());
                accumulate_atom(context, paw, occupancies, energy, atom_index, 0);
                continue;
            }

            if (number_of_spins > 2)
                throw std::runtime_error(
                    "PAW one-center terms support at most two spin channels; " +
                    std::to_string(number_of_spins) + " were given");

            std::vector<OneCenterDensities> densities_by_spin;
            densities_by_spin.reserve(number_of_spins);
            for (int spin = 0; spin < number_of_spins; spin++)
                densities_by_spin.push_back(build_one_center_densities(
                    paw, context.expansion, occupancies, context.gaunt, spin));
            auto energies = compute_one_center_energy_spin(
                paw,
                densities_by_spin,
                context.angular_grid,
                context.harmonics,
                *functional_,
                context.gradient_corrected ? &context.harmonic_gradients : nullptr,
                context.convention->options());
            for (int spin = 0; spin < number_of_spins; spin++)
                accumulate_atom(context, paw, occupancies, energies[spin], atom_index, spin);
        }
    }

  private:
    /**
     * Fold one atom's (one channel's) result into the running totals.
     *
     * The convention's energy correction and D correction are applied per
     * channel because the convention linearizes per channel too: each channel's
     * D is referenced against the same atomic state, and the energy correction
     * is a function of that channel's occupancies.
     */
    void accumulate_atom(const SpeciesOneCenterContext& context,
                         const PawPseudopotential& paw,
                         const OnSiteOccupancies& occupancies,
                         const OneCenterEnergy& energy,
                         int atom_index,
                         int spin) {
        energy_ += energy.total();
        if (spin == 0)
            energy_ += context.convention->energy_correction(occupancies);
        branch_totals_.all_electron_hartree += energy.all_electron.hartree_energy;
        branch_totals_.pseudo_hartree += energy.pseudo.hartree_energy;
        branch_totals_.all_electron_xc += energy.all_electron.xc_energy;
        branch_totals_.pseudo_xc += energy.pseudo.xc_energy;
        d_correction_by_spin_[spin][atom_index] =
            compute_one_center_d_matrix(paw, context.expansion, context.gaunt, energy);
        context.convention->correct_d_matrix(d_correction_by_spin_[spin][atom_index]);
    }

  public:
    /**
     * The four pieces the one-center energy is the difference of, summed over
     * atoms.
     *
     * The second rung of the failure ladder: it separates Hartree from XC and
     * all-electron from pseudo, which between them localize a discrepancy to a
     * single step of the engine.
     *
     * Quantum ESPRESSO prints four quantities under the same names, but its
     * pseudo-branch lines already carry the minus sign of E¹ − Ẽ¹: its four
     * **add** to its printed one-center total, where these four are unsigned
     * branch energies and have to be differenced. Differencing its printed
     * values instead — which is what gave −3.407 Ry against a printed
     * −12.6126 Ry on diamond — is the wrong combination, not evidence of a
     * different decomposition. Its component print is also per-process rather
     * than reduced, so under MPI the four can be partial sums even when the
     * total is not.
     */
    struct BranchTotals {
        Real all_electron_hartree{0.0};
        Real pseudo_hartree{0.0};
        Real all_electron_xc{0.0};
        Real pseudo_xc{0.0};
    };

    [[nodiscard]] const BranchTotals& branch_totals() const {
        return branch_totals_;
    }

    /// Σ_a (E¹_a − Ẽ¹_a), in Rydberg.
    [[nodiscard]] Real energy() const {
        return energy_;
    }

    /**
     * (D¹ − D̃¹)_Hxc for one atom in one spin channel, as a dense (nh × nh)
     * matrix; empty if the species is not PAW.
     *
     * A channel beyond what was computed falls back to channel 0, so an
     * unpolarized `update()` followed by a spin-polarized read is consistent
     * rather than out of bounds.
     */
    [[nodiscard]] const std::vector<Real>& d_correction(int atom_index, int spin = 0) const {
        int channel = spin < static_cast<int>(d_correction_by_spin_.size()) ? spin : 0;
        return d_correction_by_spin_[channel][atom_index];
    }

    [[nodiscard]] int number_of_channels(int species_index) const {
        return contexts_[species_index].number_of_channels();
    }

  private:
    std::vector<SpeciesOneCenterContext> contexts_;
    /// [spin][atom] — one entry for an unpolarized run, two for a polarized one.
    std::vector<std::vector<std::vector<Real>>> d_correction_by_spin_;
    const XCFunctional* functional_{nullptr};
    Real energy_{0.0};
    BranchTotals branch_totals_;
};

} // namespace paw_one_center
} // namespace q3
