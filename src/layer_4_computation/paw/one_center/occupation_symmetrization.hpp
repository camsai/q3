#pragma once

/**
 * Symmetrizing the PAW on-site occupations ρ_ij.
 *
 * ρ_ij (becsum) is accumulated as a weighted sum over the *irreducible* wedge,
 *
 *     ρ^a_ij = Σ_{k∈IBZ} w_k Σ_v f_kv Re[⟨ψ_kv|β^a_i⟩⟨β^a_j|ψ_kv⟩],
 *
 * and a weighted wedge sum reproduces the full-zone sum only for quantities the
 * point group leaves alone. ρ_ij is not one: each projector channel carries an
 * angular index (β_i is a radial function times Y_{l_i m_i}), and every
 * operation mixes those indices and permutes equivalent atoms. Summing the
 * wedge and stopping leaves an array that is not symmetry-adapted — its trace
 * is right, its structure is not.
 *
 * The restoration is the group average, with the same pairing
 * `SymmetryAnalyzer::symmetrize_vector` uses for forces and for the same
 * reason — it is the one combination that leaves an already-symmetric array
 * unchanged:
 *
 *     ρ^a  ←  (1/N_R) Σ_R  D(R)ᵀ · ρ^{R(a)} · D(R)
 *
 * with R(a) = `atom_map[a]` the atom `a` is carried *to*, and D(R) the
 * real-harmonic rotation of the channel's l. Read the other way round, it is
 * the statement the crystal makes: ρ^{R(a)} = D(R) ρ^a D(R)ᵀ. The radial index
 * of a channel is untouched — a rotation does not mix β's of different radial
 * shape — so the average acts only inside each (β_i, β_j) block, on the
 * (2l_i+1)×(2l_j+1) angular indices.
 *
 * **Why ultrasoft never needed this.** The same array builds the augmentation
 * charge, and that reaches the grid, where the density symmetrizer repairs it.
 * PAW's one-centre Hartree and XC terms read ρ_ij directly and never pass
 * through a grid, so nothing downstream repairs them. Measured at 30 Ry,
 * 2×2×2 Γ-centred (3 irreducible k-points against the full 8): diamond PAW
 * split its Γ valence maximum 13.7274 / 13.7427 / 13.7465 eV where the full
 * mesh gives 13.7393 three times, and sat 0.19 mRy above it; silicon PAW 0.25
 * mRy; the same silicon cell with ultrasoft or norm-conserving datasets agreed
 * with its full mesh to 1e-7 Ry.
 *
 * **Time reversal needs no separate step here**, unlike the DFT+U occupation
 * matrix. becsum is accumulated as `Re[conj(⟨β_i|ψ⟩)⟨β_j|ψ⟩]`, which is
 * already the average over the ±k pair the wedge folded together — the real
 * part *is* the Θ projection. Quantum ESPRESSO's collinear becsum is real by
 * the same construction.
 *
 * **Spin.** Each channel is averaged over the crystal's spatial operations
 * separately, exactly as `PlanewaveChargeDensitySymmetrizer` treats each spin
 * component of the density; the assumption that the chemical cell's operations
 * are symmetries of each spin channel is the one Q3 already makes on the grid,
 * not a new one taken here.
 */

#include <algorithm>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/symmetry/operation.hpp"
#include "layer_4_computation/paw/one_center/harmonic_rotation.hpp"

namespace q3::paw_one_center {

/**
 * What each becsum channel of one species means angularly.
 *
 * The expanded channels run over (radial β, m) in `harmonic_order_sequence`
 * order, so a channel's angular partners are the ones sharing its β — a
 * contiguous run of 2l+1 entries. `manifold_base` is where that run starts and
 * `manifold_position` is the offset into it, which is also the row and column
 * index of D^l.
 */
struct ChannelAngularLayout {
    int number_of_channels{};
    std::vector<int> angular_momentum;
    std::vector<int> manifold_base;
    std::vector<int> manifold_position;
};

/**
 * Derive the layout from the per-channel radial index and l that the projector
 * mapping table already publishes, rather than re-deriving the expansion rule.
 */
[[nodiscard]] inline ChannelAngularLayout
build_channel_angular_layout(const std::vector<int>& beta_index_per_channel,
                             const std::vector<int>& angular_momentum_per_channel) {
    ChannelAngularLayout layout;
    layout.number_of_channels = static_cast<int>(beta_index_per_channel.size());
    layout.angular_momentum = angular_momentum_per_channel;
    layout.manifold_base.resize(beta_index_per_channel.size());
    layout.manifold_position.resize(beta_index_per_channel.size());

    for (size_t channel = 0; channel < beta_index_per_channel.size(); channel++) {
        size_t base = channel;
        while (base > 0 && beta_index_per_channel[base - 1] == beta_index_per_channel[channel])
            base--;
        layout.manifold_base[channel] = static_cast<int>(base);
        layout.manifold_position[channel] = static_cast<int>(channel - base);
    }
    return layout;
}

/**
 * The point-group average of ρ_ij, with D^l(R) built once and reused.
 *
 * Built once per run (the operations and the datasets do not change during an
 * SCF) and applied to every becsum array each iteration. `is_active()` is
 * false when there is nothing to do — one operation, or no channels — so the
 * caller does not need a second condition of its own.
 */
class OnSiteOccupationSymmetrizer {
  public:
    /**
     * @param crystal_structure   supplies the lattice for R_cart and the atom→species map
     * @param symmetry_operations the crystal's operations, `atom_map` filled in
     * @param layout_by_species   one entry per species type, indexed as `species_type_index`
     */
    void build(const Crystal& crystal_structure,
               const std::vector<SymmetryOperation>& symmetry_operations,
               std::vector<ChannelAngularLayout> layout_by_species) {
        layout_by_species_ = std::move(layout_by_species);
        symmetry_operations_ = symmetry_operations;
        rotations_by_operation_.clear();
        rotation_slot_by_species_.clear();
        angular_momenta_.clear();

        if (symmetry_operations_.size() <= 1)
            return;

        for (const auto& layout : layout_by_species_)
            for (int momentum : layout.angular_momentum)
                if (std::find(angular_momenta_.begin(), angular_momenta_.end(), momentum) ==
                    angular_momenta_.end())
                    angular_momenta_.push_back(momentum);
        if (angular_momenta_.empty())
            return;

        rotation_slot_by_species_.assign(layout_by_species_.size(), {});
        for (size_t species = 0; species < layout_by_species_.size(); species++)
            for (int momentum : layout_by_species_[species].angular_momentum)
                rotation_slot_by_species_[species].push_back(static_cast<int>(
                    std::find(angular_momenta_.begin(), angular_momenta_.end(), momentum) -
                    angular_momenta_.begin()));

        /**
         * One D^l per (operation, l): two channels sharing an l share the
         * matrix, and the angular quadrature that builds it is the expensive
         * part.
         */
        rotations_by_operation_.resize(symmetry_operations_.size());
        for (size_t operation = 0; operation < symmetry_operations_.size(); operation++) {
            Matrix3x3 cartesian =
                cartesian_rotation_of(crystal_structure, symmetry_operations_[operation]);
            rotations_by_operation_[operation].resize(angular_momenta_.size());
            for (size_t slot = 0; slot < angular_momenta_.size(); slot++)
                rotations_by_operation_[operation][slot] =
                    build_real_harmonic_rotation_matrix(angular_momenta_[slot], cartesian);
        }
    }

    [[nodiscard]] bool is_active() const {
        return !rotations_by_operation_.empty();
    }

    [[nodiscard]] int number_of_symmetry_operations() const {
        return static_cast<int>(symmetry_operations_.size());
    }

    /**
     * Replace ρ_ij by its group average, in place.
     *
     * `occupations[atom][i * nh_atom + j]`, with `nh_atom` the atom's *own*
     * channel count — the indexing stride the ultrasoft accumulation writes
     * with, which is not the allocated row length in a mixed-species cell.
     * An atom whose image under some operation could not be identified (an
     * `atom_map` entry left unresolved) contributes nothing for that
     * operation, and the average is taken over the operations that did
     * contribute: averaging against an array that was never accumulated would
     * be worse than averaging over fewer operations.
     */
    void symmetrize(const Crystal& crystal_structure, Matrix<Real>& occupations) const {
        if (!is_active())
            return;
        int number_of_atoms = crystal_structure.number_of_atoms();
        Matrix<Real> original = occupations;

        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            int species_index = crystal_structure.species_type_index(atom_index);
            if (species_index < 0 || species_index >= static_cast<int>(layout_by_species_.size()))
                continue;
            const auto& layout = layout_by_species_[species_index];
            const auto& rotation_slots = rotation_slot_by_species_[species_index];
            int channels = layout.number_of_channels;
            if (channels == 0)
                continue;

            std::vector<Real> averaged(static_cast<size_t>(channels) * channels, 0.0);
            int contributing_operations = 0;

            for (size_t operation = 0; operation < symmetry_operations_.size(); operation++) {
                const auto& atom_map = symmetry_operations_[operation].atom_map;
                if (atom_index >= static_cast<int>(atom_map.size()))
                    continue;
                int image_atom = atom_map[atom_index];
                if (image_atom < 0 || image_atom >= number_of_atoms)
                    continue;
                if (crystal_structure.species_type_index(image_atom) != species_index)
                    continue;

                const Real* source = original[image_atom];
                const auto& rotations = rotations_by_operation_[operation];

                for (int first = 0; first < channels; first++) {
                    const auto& rotation_first =
                        rotations[static_cast<size_t>(rotation_slots[first])];
                    int size_first = 2 * layout.angular_momentum[first] + 1;
                    int base_first = layout.manifold_base[first];
                    int position_first = layout.manifold_position[first];

                    for (int second = 0; second < channels; second++) {
                        const auto& rotation_second =
                            rotations[static_cast<size_t>(rotation_slots[second])];
                        int size_second = 2 * layout.angular_momentum[second] + 1;
                        int base_second = layout.manifold_base[second];
                        int position_second = layout.manifold_position[second];

                        /**
                         * (D ᵀ ρ D)_{m m'} = Σ_pq D_{p m} ρ_{pq} D_{q m'}, with
                         * p and q running over the partners of `first` and
                         * `second` — the channels sharing their radial index.
                         */
                        Real accumulated = 0.0;
                        for (int p = 0; p < size_first; p++) {
                            Real left = rotation_first[static_cast<size_t>(p) * size_first +
                                                       position_first];
                            if (left == 0.0)
                                continue;
                            const Real* row = source +
                                              static_cast<size_t>(base_first + p) * channels +
                                              base_second;
                            for (int q = 0; q < size_second; q++) {
                                Real right = rotation_second[static_cast<size_t>(q) * size_second +
                                                             position_second];
                                if (right == 0.0)
                                    continue;
                                accumulated += left * right * row[q];
                            }
                        }
                        averaged[static_cast<size_t>(first) * channels + second] += accumulated;
                    }
                }
                contributing_operations++;
            }

            if (contributing_operations == 0)
                continue;
            Real normalization = 1.0 / static_cast<Real>(contributing_operations);
            Real* destination = occupations[atom_index];
            for (int index = 0; index < channels * channels; index++)
                destination[index] = averaged[static_cast<size_t>(index)] * normalization;
        }
    }

  private:
    std::vector<SymmetryOperation> symmetry_operations_;
    std::vector<ChannelAngularLayout> layout_by_species_;
    /// The distinct l values across every species, in the order D^l was built for them.
    std::vector<int> angular_momenta_;
    /// rotation_slot_by_species_[species][channel] — which of those l that channel carries.
    std::vector<std::vector<int>> rotation_slot_by_species_;
    /// rotations_by_operation_[operation][slot] — D^l for `angular_momenta_[slot]`.
    std::vector<std::vector<std::vector<Real>>> rotations_by_operation_;
};

} // namespace q3::paw_one_center
