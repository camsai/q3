#pragma once

/**
 * Factory for PlanewaveLocalOrbitalSet implementations.
 *
 * Resolves the one decision an input can make about local orbitals -- were
 * explicit centres given, or not -- to a concrete set, so no consumer
 * carries that branch. Empty `explicit_centers_crystal` selects the
 * pseudopotentials' own atomic wavefunctions -- all of them, or the shells
 * `trial_shells` names per species; a non-empty list places one hydrogenic
 * s orbital of decay length `decay_length_bohr` at each centre.
 *
 * Mirrors create_isolated_system_correction_calculator(): adding a source
 * of orbitals is one new class implementing PlanewaveLocalOrbitalSet plus
 * one case here.
 */

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/hydrogenic_orbital_set.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/pseudo_atomic_orbital_set.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

[[nodiscard]] inline std::unique_ptr<PlanewaveLocalOrbitalSet> create_planewave_local_orbital_set(
    const Crystal& crystal_structure,
    const PseudopotentialSet& pseudopotential_set,
    const std::vector<Vector3D>& explicit_centers_crystal,
    Real decay_length_bohr,
    const std::map<std::string, std::vector<int>>& trial_shells = {}) {
    if (explicit_centers_crystal.empty()) {
        return std::make_unique<PlanewavePseudoAtomicOrbitalSet>(
            crystal_structure, pseudopotential_set, trial_shells);
    }
    std::vector<HydrogenicOrbital> orbitals;
    orbitals.reserve(explicit_centers_crystal.size());
    for (const Vector3D& center : explicit_centers_crystal) {
        orbitals.push_back({center, decay_length_bohr});
    }
    return std::make_unique<PlanewaveHydrogenicOrbitalSet>(std::move(orbitals));
}

} // namespace q3
