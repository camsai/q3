#pragma once

/**
 * SerialStructureFactorCalculator — the phase sums of StructureFactorCalculator on the host.
 *
 * Each phase is taken once per atom and G-vector, from positions converted to Cartesian
 * once, and serves every set of weights at that pair.
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/structure_factor_calculator.hpp"

namespace q3 {

/// Forms the phases on the host; see the file comment.
class SerialStructureFactorCalculator : public StructureFactorCalculator {
  public:
    /// `crystal_structure` and `planewave_basis` must outlive this object.
    SerialStructureFactorCalculator(const Crystal& crystal_structure,
                                    const PlanewaveBasis& planewave_basis)
        : planewave_basis_(planewave_basis)
        , atomic_positions_(cartesian_positions(crystal_structure)) {}

    /// See StructureFactorCalculator::structure_factors(); formed on the host.
    [[nodiscard]] std::vector<std::vector<Complex>>
    structure_factors(const std::vector<int>& g_vector_indices,
                      const std::vector<std::vector<Real>>& atom_weights) const override {
        const std::size_t number_of_atoms = atomic_positions_.size();
        const std::size_t number_of_g_vectors = g_vector_indices.size();
        for (const auto& weights : atom_weights) {
            if (weights.size() != number_of_atoms) {
                throw std::invalid_argument(
                    "SerialStructureFactorCalculator: a set of atom weights must have one "
                    "entry per atom");
            }
        }
        std::vector<std::vector<Complex>> result(
            atom_weights.size(), std::vector<Complex>(number_of_g_vectors, Complex(0.0, 0.0)));
        const auto& g_vectors = planewave_basis_.g_vectors();
        for (std::size_t entry = 0; entry < number_of_g_vectors; entry++) {
            const Vector3D& g_cartesian =
                g_vectors[static_cast<std::size_t>(g_vector_indices[entry])].g_cart;
            for (std::size_t atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                const Complex phase_factor = compute_phase_factor(
                    constants::two_pi * g_cartesian.dot(atomic_positions_[atom_index]));
                for (std::size_t set = 0; set < atom_weights.size(); set++) {
                    const Real weight = atom_weights[set][atom_index];
                    if (weight != 0.0) {
                        result[set][entry] += weight * phase_factor;
                    }
                }
            }
        }
        return result;
    }

    /// See StructureFactorCalculator::gradient_phase_sums(); formed on the host.
    [[nodiscard]] std::vector<Vector3D>
    gradient_phase_sums(const std::vector<int>& g_vector_indices,
                        const std::vector<std::vector<Complex>>& phase_weights,
                        const std::vector<int>& set_of_atom) const override {
        const std::size_t number_of_atoms = atomic_positions_.size();
        if (set_of_atom.size() != number_of_atoms) {
            throw std::invalid_argument(
                "SerialStructureFactorCalculator: set_of_atom must have one entry per atom");
        }
        for (const auto& weights : phase_weights) {
            if (weights.size() != g_vector_indices.size()) {
                throw std::invalid_argument(
                    "SerialStructureFactorCalculator: a set of phase weights must have one "
                    "entry per G-vector");
            }
        }
        std::vector<Vector3D> result(number_of_atoms, Vector3D{0.0, 0.0, 0.0});
        const auto& g_vectors = planewave_basis_.g_vectors();
        for (std::size_t entry = 0; entry < g_vector_indices.size(); entry++) {
            const Vector3D& g_cartesian =
                g_vectors[static_cast<std::size_t>(g_vector_indices[entry])].g_cart;
            for (std::size_t atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                const int set = set_of_atom[atom_index];
                if (set < 0) {
                    continue;
                }
                const Complex weighted_phase =
                    phase_weights[static_cast<std::size_t>(set)][entry] *
                    compute_phase_factor_conjugate(constants::two_pi *
                                                   g_cartesian.dot(atomic_positions_[atom_index]));
                for (int direction = 0; direction < 3; direction++) {
                    result[atom_index][direction] += g_cartesian[direction] * weighted_phase.imag();
                }
            }
        }
        return result;
    }

  private:
    [[nodiscard]] static std::vector<Vector3D>
    cartesian_positions(const Crystal& crystal_structure) {
        std::vector<Vector3D> positions(
            static_cast<std::size_t>(crystal_structure.number_of_atoms()));
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            positions[static_cast<std::size_t>(atom_index)] =
                crystal_structure.convert_from_crystal_to_cartesian(atom_index);
        }
        return positions;
    }

    const PlanewaveBasis& planewave_basis_;
    std::vector<Vector3D> atomic_positions_;
};

} // namespace q3
