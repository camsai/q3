#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_3_representation/basis/planewave/structure_factor_calculator_serial.hpp"
#include "utils/two_species_projector_system.hpp"

using namespace q3;
using test_support::TwoSpeciesProjectorSystem;

namespace {

/// The phase e^{sign·iG·τ} of one atom at one G-vector, from its definition.
Complex
phase(const TwoSpeciesProjectorSystem& system, int g_vector_index, int atom_index, Real sign) {
    const Vector3D position =
        system.crystal_structure.convert_from_crystal_to_cartesian(atom_index);
    const Real angle =
        constants::two_pi * system.planewave_basis.g_vectors()[g_vector_index].g_cart.dot(position);
    return {std::cos(angle), sign * std::sin(angle)};
}

std::vector<int> some_g_vectors(const TwoSpeciesProjectorSystem& system) {
    std::vector<int> indices;
    for (int g_vector_index = 0; g_vector_index < system.planewave_basis.number_of_g_vectors();
         g_vector_index += 7) {
        indices.push_back(g_vector_index);
    }
    return indices;
}

} // namespace

TEST(SerialStructureFactorCalculator, EachSetIsItsWeightedSumOfPhases) {
    TwoSpeciesProjectorSystem system;
    SerialStructureFactorCalculator calculator(system.crystal_structure, system.planewave_basis);
    const std::vector<int> indices = some_g_vectors(system);
    const std::vector<std::vector<Real>> weights = {{1.0, 0.0, 1.0, 0.0}, {0.5, -2.0, 3.0, 1.5}};

    const auto structure_factors = calculator.structure_factors(indices, weights);

    ASSERT_EQ(structure_factors.size(), weights.size());
    for (std::size_t set = 0; set < weights.size(); set++) {
        ASSERT_EQ(structure_factors[set].size(), indices.size());
        for (std::size_t entry = 0; entry < indices.size(); entry++) {
            Complex expected(0.0, 0.0);
            for (int atom_index = 0; atom_index < 4; atom_index++) {
                expected +=
                    weights[set][atom_index] * phase(system, indices[entry], atom_index, -1.0);
            }
            EXPECT_NEAR(structure_factors[set][entry].real(), expected.real(), 1e-12);
            EXPECT_NEAR(structure_factors[set][entry].imag(), expected.imag(), 1e-12);
        }
    }
}

TEST(SerialStructureFactorCalculator, AtGammaTheStructureFactorIsTheSumOfTheWeights) {
    TwoSpeciesProjectorSystem system;
    SerialStructureFactorCalculator calculator(system.crystal_structure, system.planewave_basis);
    ASSERT_EQ(system.planewave_basis.first_nonzero_g_index(), 1);

    const auto structure_factors = calculator.structure_factors({0}, {{0.5, -2.0, 3.0, 1.5}});

    EXPECT_DOUBLE_EQ(structure_factors[0][0].real(), 3.0);
    EXPECT_DOUBLE_EQ(structure_factors[0][0].imag(), 0.0);
}

TEST(SerialStructureFactorCalculator, GradientSumsFollowTheirDefinitionAndSkipAtomsWithoutASet) {
    TwoSpeciesProjectorSystem system;
    SerialStructureFactorCalculator calculator(system.crystal_structure, system.planewave_basis);
    const std::vector<int> indices = some_g_vectors(system);
    std::vector<std::vector<Complex>> phase_weights(2, std::vector<Complex>(indices.size()));
    for (std::size_t entry = 0; entry < indices.size(); entry++) {
        phase_weights[0][entry] = Complex(1.0 / (1.0 + entry), 0.3);
        phase_weights[1][entry] = Complex(-0.2, 2.0 / (2.0 + entry));
    }
    const std::vector<int> set_of_atom = {1, -1, 0, 1};

    const std::vector<Vector3D> sums =
        calculator.gradient_phase_sums(indices, phase_weights, set_of_atom);

    ASSERT_EQ(sums.size(), 4U);
    for (int atom_index = 0; atom_index < 4; atom_index++) {
        Vector3D expected{0.0, 0.0, 0.0};
        if (set_of_atom[atom_index] >= 0) {
            for (std::size_t entry = 0; entry < indices.size(); entry++) {
                const Real imaginary = (phase_weights[set_of_atom[atom_index]][entry] *
                                        phase(system, indices[entry], atom_index, +1.0))
                                           .imag();
                for (int direction = 0; direction < 3; direction++) {
                    expected[direction] +=
                        system.planewave_basis.g_vectors()[indices[entry]].g_cart[direction] *
                        imaginary;
                }
            }
        }
        for (int direction = 0; direction < 3; direction++) {
            EXPECT_NEAR(sums[atom_index][direction], expected[direction], 1e-12);
        }
    }
    EXPECT_EQ(sums[1][0], 0.0);
}

TEST(SerialStructureFactorCalculator, RefusesWeightsOfTheWrongLength) {
    TwoSpeciesProjectorSystem system;
    SerialStructureFactorCalculator calculator(system.crystal_structure, system.planewave_basis);
    EXPECT_THROW((void)calculator.structure_factors({0, 1}, {{1.0, 1.0}}), std::invalid_argument);
    EXPECT_THROW((void)calculator.gradient_phase_sums({0, 1}, {{Complex(1.0, 0.0)}}, {0, 0, 0, 0}),
                 std::invalid_argument);
    EXPECT_THROW((void)calculator.gradient_phase_sums({0}, {{Complex(1.0, 0.0)}}, {0, 0}),
                 std::invalid_argument);
}
