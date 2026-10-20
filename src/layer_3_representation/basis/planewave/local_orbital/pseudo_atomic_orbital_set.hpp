#pragma once

/**
 * PlanewavePseudoAtomicOrbitalSet — the pseudopotential's own atomic
 * wavefunctions (UPF PP_CHI / PP_PSWFC), rendered in plane waves.
 *
 * Every orbital the pseudopotential file carries, on every atom of that
 * species, in the enumeration order atomic_orbital_builder.hpp defines: the
 * same set the PDOS / Loewdin population analysis projects onto, and the
 * default initial guess of a Wannier construction. The radial part is
 * tabulated in q once at construction; each k-point's rendering then costs
 * an interpolation per G-vector plus structure factors.
 *
 * `trial_shells` narrows it to the shells it names per species, as 0-based
 * indices in the order the file lists them; a species it does not name keeps
 * all of them. The whole set is still rendered and the named rows kept, so the
 * orbitals are exactly the corresponding rows of the full set.
 *
 * All atom-centred by construction. For a material whose Wannier functions
 * are not -- silicon's valence functions sit on the bonds -- this set spans
 * the wrong subspace and the explicit-centre PlanewaveHydrogenicOrbitalSet
 * is the right choice; see that class.
 */

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_3_representation/basis/planewave/local_orbital/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

class PlanewavePseudoAtomicOrbitalSet : public PlanewaveLocalOrbitalSet {
  public:
    PlanewavePseudoAtomicOrbitalSet(
        const Crystal& crystal_structure,
        const PseudopotentialSet& pseudopotential_set,
        const std::map<std::string, std::vector<int>>& trial_shells = {})
        : pseudopotential_set_(pseudopotential_set) {
        const std::vector<std::vector<bool>> kept =
            kept_shells(crystal_structure, pseudopotential_set, trial_shells);
        /// One label per rendered orbital, in the order build_atomic_orbitals_g_space fills them.
        const std::vector<AtomicOrbitalLabel> every_label =
            enumerate_atomic_orbitals(crystal_structure, pseudopotential_set);
        for (std::size_t row = 0; row < every_label.size(); row++) {
            const AtomicOrbitalLabel& label = every_label[row];
            if (kept[static_cast<std::size_t>(label.species_type_index)]
                    [static_cast<std::size_t>(label.shell_index)]) {
                kept_rows_.push_back(static_cast<int>(row));
                orbital_labels_.push_back(label);
            }
        }
        number_of_rendered_orbitals_ = static_cast<int>(every_label.size());
        centers_crystal_.reserve(orbital_labels_.size());
        for (const AtomicOrbitalLabel& label : orbital_labels_) {
            centers_crystal_.push_back(crystal_structure.fractional_position(label.atom_index));
        }
        atomic_wavefunction_table_ = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure.cell_volume(), q_spacing_, number_of_q_points_);
    }

    [[nodiscard]] int number_of_orbitals() const override {
        return static_cast<int>(orbital_labels_.size());
    }

    [[nodiscard]] const std::vector<AtomicOrbitalLabel>& orbital_labels() const {
        return orbital_labels_;
    }

    void build(int kpoint_index,
               int number_of_plane_waves,
               const std::vector<int>& g_vector_indices,
               const Crystal& crystal_structure,
               const KPointsGrid& kpoint_grid,
               const PlanewaveBasis& planewave_basis,
               std::vector<std::vector<Complex>>& orbitals) const override {
        const bool every_shell = number_of_orbitals() == number_of_rendered_orbitals_;
        std::vector<std::vector<Complex>> rendered;
        build_atomic_orbitals_g_space(kpoint_index,
                                      number_of_plane_waves,
                                      g_vector_indices,
                                      crystal_structure,
                                      kpoint_grid,
                                      planewave_basis,
                                      pseudopotential_set_,
                                      atomic_wavefunction_table_,
                                      q_spacing_,
                                      number_of_q_points_,
                                      every_shell ? orbitals : rendered);
        if (every_shell) {
            return;
        }
        orbitals.resize(kept_rows_.size());
        for (std::size_t orbital = 0; orbital < kept_rows_.size(); orbital++) {
            orbitals[orbital] = std::move(rendered[kept_rows_[orbital]]);
        }
    }

    /// The atom each orbital is centred on; atom-centred by construction.
    [[nodiscard]] std::vector<Vector3D> centers_crystal() const override {
        return centers_crystal_;
    }

    [[nodiscard]] std::string description() const override {
        std::string text =
            std::to_string(number_of_orbitals()) +
            " pseudo-atomic orbitals from the pseudopotentials' atomic wavefunctions";
        if (number_of_orbitals() != number_of_rendered_orbitals_) {
            text += ", " + std::to_string(number_of_rendered_orbitals_ - number_of_orbitals()) +
                    " left out by control.interpolation.trial_shells";
        }
        return text;
    }

  private:
    /**
     * kept[type][shell]: whether that shell of that species is a trial orbital.
     *
     * Refuses a species the material does not have, a shell index past the
     * file's list, and a shell the reader marked not to be used -- a negative
     * occupation, or a zero-norm radial function, which the reader records as
     * one -- each a selection that cannot mean what it says.
     */
    [[nodiscard]] static std::vector<std::vector<bool>>
    kept_shells(const Crystal& crystal_structure,
                const PseudopotentialSet& pseudopotential_set,
                const std::map<std::string, std::vector<int>>& trial_shells) {
        std::vector<std::vector<bool>> kept(pseudopotential_set.size());
        std::map<std::string, int> type_of_species;
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            type_of_species.emplace(crystal_structure.element(atom_index),
                                    crystal_structure.species_type_index(atom_index));
        }
        for (std::size_t type_index = 0; type_index < kept.size(); type_index++) {
            kept[type_index].assign(pseudopotential_set[type_index].wavefunctions_atomic.size(),
                                    true);
        }
        for (const auto& [species, shells] : trial_shells) {
            const auto found = type_of_species.find(species);
            if (found == type_of_species.end()) {
                throw std::invalid_argument("control.interpolation.trial_shells names " + species +
                                            ", which is not a species of this material.");
            }
            const auto& wavefunctions = pseudopotential_set[found->second].wavefunctions_atomic;
            std::vector<bool> named(wavefunctions.size(), false);
            for (const int shell : shells) {
                if (shell < 0 || shell >= static_cast<int>(wavefunctions.size())) {
                    throw std::invalid_argument(
                        "control.interpolation.trial_shells." + species + " lists shell " +
                        std::to_string(shell) + ", but its pseudopotential lists " +
                        std::to_string(wavefunctions.size()) + " atomic wavefunctions.");
                }
                if (wavefunctions[shell].occupations < 0.0) {
                    throw std::invalid_argument("control.interpolation.trial_shells." + species +
                                                " lists shell " + std::to_string(shell) +
                                                ", which its pseudopotential marks not to be used "
                                                "as an orbital (a negative "
                                                "occupation, or a radial function of zero norm).");
                }
                named[shell] = true;
            }
            kept[found->second] = named;
        }
        return kept;
    }

    const PseudopotentialSet& pseudopotential_set_;
    std::vector<AtomicOrbitalLabel> orbital_labels_;
    /// The rows of the full rendering this set keeps, in order.
    std::vector<int> kept_rows_;
    /// How many orbitals the full rendering holds, trial shells or not.
    int number_of_rendered_orbitals_ = 0;
    std::vector<std::vector<std::vector<Real>>> atomic_wavefunction_table_;
    std::vector<Vector3D> centers_crystal_;
    Real q_spacing_ = 0.01;
    int number_of_q_points_ = 900;
};

} // namespace q3
