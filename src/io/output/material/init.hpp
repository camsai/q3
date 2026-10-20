#pragma once

/**
 * The structure a run used, as an ESSE material, for results.json.
 *
 * The cell is the one the engine built -- any Bravais index, explicit vectors
 * or a structure substituted with `--material` -- in its own Cartesian frame,
 * so a reader gets back exactly the geometry the energy, forces and stress
 * belong to. That is what lets q3.workflows start a relaxation from any input
 * the engine reads, a pw.x deck with any ibrav included, without a reader of
 * its own: it runs the input once and continues from this material.
 */

#include <cmath>

#include <nlohmann/json.hpp>

#include "io/input/data.hpp"
#include "io/input/system_builder.hpp"
#include "layer_0_core/types/constants.hpp"

namespace q3 {

/**
 * The ESSE material of *input*: lattice (vectors in angstrom, with ESSE's lengths
 * and angles), and basis (species labels, crystal coordinates, and
 * `constraints` -- per site, `true` where a Cartesian component may move, as
 * mat3ra-made writes them).
 */
[[nodiscard]] inline nlohmann::json build_material_json(const InputData& input) {
    const Matrix3x3 vectors_bohr =
        build_crystal_from_input(input, input.species).lattice_vectors_bohr();
    const char* row_keys[3] = {"a", "b", "c"};
    nlohmann::json vectors = {{"units", "angstrom"}};
    Real lengths_angstrom[3];
    for (int row = 0; row < 3; ++row) {
        const Vector3D vector_angstrom = vectors_bohr[row] * constants::BOHR_RADIUS_ANGSTROM;
        vectors[row_keys[row]] = {vector_angstrom[0], vector_angstrom[1], vector_angstrom[2]};
        lengths_angstrom[row] = vector_angstrom.norm();
    }
    auto angle_degrees = [&](int first, int second) {
        const Real cosine = vectors_bohr[first].dot(vectors_bohr[second]) /
                            (vectors_bohr[first].norm() * vectors_bohr[second].norm());
        return std::acos(cosine) * 180.0 / constants::pi;
    };

    nlohmann::json basis;
    basis["units"] = "crystal";
    basis["elements"] = nlohmann::json::array();
    basis["coordinates"] = nlohmann::json::array();
    basis["constraints"] = nlohmann::json::array();
    for (size_t atom_index = 0; atom_index < input.atoms.size(); ++atom_index) {
        const Atom& atom = input.atoms[atom_index];
        const int site_id = static_cast<int>(atom_index) + 1;
        basis["elements"].push_back(
            {{"id", site_id}, {"value", input.species[atom.species_index].symbol}});
        basis["coordinates"].push_back(
            {{"id", site_id}, {"value", {atom.position[0], atom.position[1], atom.position[2]}}});
        basis["constraints"].push_back({{"id", site_id},
                                        {"value",
                                         {atom.movable_components[0],
                                          atom.movable_components[1],
                                          atom.movable_components[2]}}});
    }

    return {{"name", input.prefix},
            {"metadata", nlohmann::json::object()},
            {"lattice",
             {{"a", lengths_angstrom[0]},
              {"b", lengths_angstrom[1]},
              {"c", lengths_angstrom[2]},
              {"alpha", angle_degrees(1, 2)},
              {"beta", angle_degrees(0, 2)},
              {"gamma", angle_degrees(0, 1)},
              {"units", {{"length", "angstrom"}, {"angle", "degree"}}},
              {"vectors", vectors}}},
            {"basis", basis}};
}

} // namespace q3
