#pragma once

/**
 * Shared builders from parsed InputData to layer-1 structures.
 *
 * Single source of truth for crystal construction and global k-point grid
 * setup (explicit list or generation, IBZ reduction, LSDA spin-channel weight
 * scaling). Both production entry points — the iterator factory and the
 * q3_core bindings — must build through these helpers so that input semantics
 * can never drift between them.
 */

#include <utility>
#include <vector>

#include "io/input/data.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"

namespace q3 {

/**
 * Per-atom symmetry partition implied by per-atom starting moments.
 *
 * Atoms of one species seeded with different moments (a same-species
 * antiferromagnet) must not be interchanged by a symmetry operation, or the
 * k-point reduction and the density symmetrization would average the two
 * sublattices back together. Returns one class per atom -- distinct
 * (species, moment) pairs numbered in order of first appearance -- or an
 * EMPTY vector when every species is uniform, which leaves the crystal on its
 * default species partition (and the run bit-identical to per-species input).
 */
inline std::vector<int> site_symmetry_classes_from_input(const InputData& input) {
    const auto& moments = input.atomic_starting_magnetization;
    if (input.number_of_spin_components != 2 || moments.empty() ||
        moments.size() != input.atoms.size())
        return {};
    std::vector<int> classes(input.atoms.size(), -1);
    std::vector<std::pair<int, Real>> class_keys;
    bool splits_a_species = false;
    for (size_t atom = 0; atom < input.atoms.size(); ++atom) {
        int species = input.atoms[atom].species_index;
        int found = -1;
        for (size_t key = 0; key < class_keys.size(); ++key) {
            if (class_keys[key].first != species)
                continue;
            if (class_keys[key].second == moments[atom]) {
                found = static_cast<int>(key);
                break;
            }
            splits_a_species = true;
        }
        if (found < 0) {
            found = static_cast<int>(class_keys.size());
            class_keys.emplace_back(species, moments[atom]);
        }
        classes[atom] = found;
    }
    return splits_a_species ? classes : std::vector<int>{};
}

/**
 * Build the Crystal described by the input: explicit lattice vectors ("free"
 * lattice) when present, Bravais-index construction otherwise.
 *
 * @param species  Species list to install (callers pass the pseudopotential-
 *                 patched copy, which is why this is a parameter rather than
 *                 read from input.species).
 */
inline Crystal build_crystal_from_input(const InputData& input,
                                        const std::vector<AtomSpecies>& species) {
    Crystal crystal = input.has_explicit_lattice_vectors
                          ? Crystal::from_lattice_vectors(
                                input.cell_dimension_1, input.lattice_vectors, species, input.atoms)
                          : Crystal::from_input_data(input.bravais_lattice_index,
                                                     input.cell_dimension_1,
                                                     species,
                                                     input.atoms,
                                                     input.cell_dimension_2,
                                                     input.cell_dimension_3,
                                                     input.cell_dimension_4,
                                                     input.cell_dimension_5,
                                                     input.cell_dimension_6);
    crystal.set_site_symmetry_classes(site_symmetry_classes_from_input(input));
    return crystal;
}

/**
 * Build the GLOBAL k-point grid for a calculation:
 *
 *   1. explicit kpoints_list when given (NSCF paths), else Monkhorst-Pack
 *      generation from the input grid dimensions and shifts;
 *   2. IBZ reduction (spglib or internal) unless symmetry is disabled;
 *   3. LSDA weight scaling — generated weights embed the spin degeneracy of 2,
 *      and with two explicit spin channels each k-point state carries half of
 *      that weight.
 *
 * Pool distribution (the per-rank subset) remains the caller's concern.
 */
inline KPointsGrid build_global_kpoint_grid_from_input(const InputData& input,
                                                       const Crystal& crystal) {
    KPointsGrid global_kpoint_grid;
    if (!input.kpoints_list.empty()) {
        for (const auto& [coordinates, weight] : input.kpoints_list) {
            KPoint kpoint;
            kpoint.coordinates = coordinates;
            kpoint.weight = weight;
            global_kpoint_grid.add_kpoint(kpoint);
        }
    } else {
        global_kpoint_grid.generate(input.kpoint_grid_dimension_1,
                                    input.kpoint_grid_dimension_2,
                                    input.kpoint_grid_dimension_3,
                                    input.kpoint_grid_shift_1,
                                    input.kpoint_grid_shift_2,
                                    input.kpoint_grid_shift_3);

        /**
         * Gated on the mesh having more than one point, not on its first
         * dimension having more than one: a 1x4x4 mesh -- a slab or a wire,
         * sampled only across the two periodic directions -- has sixteen
         * points and an irreducible wedge like any other, and the old test
         * left every one of them in the sum. Q3 then solved up to a whole
         * point group's worth of duplicate k-points, correctly and slowly.
         * A single point (1x1x1, shifted or not) is its own wedge, so the
         * reduction has nothing to do and is skipped rather than called.
         */
        const int number_of_mesh_points = input.kpoint_grid_dimension_1 *
                                          input.kpoint_grid_dimension_2 *
                                          input.kpoint_grid_dimension_3;
        if (!input.disable_symmetry && number_of_mesh_points > 1) {
            if (input.ibz_reduction_method == "spglib") {
                global_kpoint_grid = SymmetryAnalyzer::reduce_kpoints(crystal, global_kpoint_grid);
            } else {
                global_kpoint_grid =
                    SymmetryAnalyzer::reduce_kpoints_internal(crystal, global_kpoint_grid);
            }
        }
    }

    if (input.number_of_spin_components == 2) {
        global_kpoint_grid.scale_weights(0.5);
    }

    return global_kpoint_grid;
}

} // namespace q3
