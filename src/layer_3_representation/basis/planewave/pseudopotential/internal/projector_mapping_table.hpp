#pragma once

/**
 * ProjectorMappingTable — Kleinman-Bylander projector channel mapping tables.
 *
 * Expands the radial beta projectors into (l,m)-resolved projector channels
 * and provides the reverse mapping from channel index back to radial index,
 * angular momentum, magnetic quantum number, and combined (l,m) index.
 */

#include <vector>

#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

class ProjectorMappingTable {
  public:
    ProjectorMappingTable() = default;

    /**
     * Build the projector expansion tables for all species and compute total projector count.
     * @param crystal_structure       Crystal (needed for atom→species mapping)
     * @param pseudopotential_set     The pseudopotentials
     * @param number_of_species_types Number of distinct species
     */
    void build(const Crystal& crystal_structure,
               const PseudopotentialSet& pseudopotential_set,
               int number_of_species_types) {
        total_number_of_projectors_ = 0;
        channels_per_type_.resize(number_of_species_types);
        to_beta_index_.resize(number_of_species_types);
        angular_momentum_.resize(number_of_species_types);
        magnetic_quantum_number_.resize(number_of_species_types);
        combined_lm_index_.resize(number_of_species_types);

        for (int type_index = 0; type_index < number_of_species_types; type_index++) {
            int channel_count = 0;
            for (int projector_index = 0;
                 projector_index <
                 static_cast<int>(pseudopotential_set[type_index].projectors.size());
                 projector_index++) {
                int angular_momentum_value =
                    pseudopotential_set[type_index].projectors[projector_index].angular_momentum;
                for (int magnetic_projection : harmonic_order_sequence(angular_momentum_value)) {
                    to_beta_index_[type_index].push_back(projector_index);
                    angular_momentum_[type_index].push_back(angular_momentum_value);
                    magnetic_quantum_number_[type_index].push_back(magnetic_projection);
                    combined_lm_index_[type_index].push_back(harmonic_combined_degree_order_index(
                        angular_momentum_value, magnetic_projection));
                    channel_count++;
                }
            }
            channels_per_type_[type_index] = channel_count;
        }

        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            int species_type = crystal_structure.species_type_index(atom_index);
            total_number_of_projectors_ += channels_per_type_[species_type];
        }
    }

    // ─── Accessors ─────────────────────────────────────────────────

    /**
     * Total Kleinman-Bylander projectors across all atoms.
     */
    [[nodiscard]] int total_number_of_projectors() const {
        return total_number_of_projectors_;
    }

    /**
     * Number of expanded projector channels for a species (sum of 2l+1).
     */
    [[nodiscard]] int number_of_channels(int species_type) const {
        return channels_per_type_[species_type];
    }

    /**
     * Map an expanded channel back to its radial beta projector index.
     */
    [[nodiscard]] int to_beta_index(int species_type, int channel_index) const {
        return to_beta_index_[species_type][channel_index];
    }

    /**
     * Angular momentum l of an expanded channel.
     */
    [[nodiscard]] int angular_momentum(int species_type, int channel_index) const {
        return angular_momentum_[species_type][channel_index];
    }

    /**
     * Magnetic quantum number m of an expanded channel.
     */
    [[nodiscard]] int magnetic_quantum_number(int species_type, int channel_index) const {
        return magnetic_quantum_number_[species_type][channel_index];
    }

    /**
     * Combined (l,m) index for an expanded channel (QE convention: l² + offset).
     */
    [[nodiscard]] int combined_lm_index(int species_type, int channel_index) const {
        return combined_lm_index_[species_type][channel_index];
    }

    /**
     * Raw access to the channels-per-type table (needed by USPP newd).
     */
    [[nodiscard]] const std::vector<int>& channels_per_type_table() const {
        return channels_per_type_;
    }

    /**
     * Raw access to the to_beta_index table (needed by USPP newd).
     */
    [[nodiscard]] const std::vector<std::vector<int>>& to_beta_index_table() const {
        return to_beta_index_;
    }

    /**
     * Raw access to the combined_lm_index table (needed by USPP newd).
     */
    [[nodiscard]] const std::vector<std::vector<int>>& combined_lm_index_table() const {
        return combined_lm_index_;
    }

    /**
     * Raw access to the angular_momentum table (needed by USPP newd).
     */
    [[nodiscard]] const std::vector<std::vector<int>>& angular_momentum_table() const {
        return angular_momentum_;
    }

  private:
    int total_number_of_projectors_{};
    std::vector<int> channels_per_type_;
    std::vector<std::vector<int>> to_beta_index_;
    std::vector<std::vector<int>> angular_momentum_;
    std::vector<std::vector<int>> magnetic_quantum_number_;
    std::vector<std::vector<int>> combined_lm_index_;
};

} // namespace q3
