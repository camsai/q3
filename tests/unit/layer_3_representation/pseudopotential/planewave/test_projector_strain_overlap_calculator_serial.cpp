#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator_serial.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder_serial.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_strain_overlap_calculator_serial.hpp"
#include "utils/one_kpoint_quantum_system.hpp"
#include "utils/projector_strain_test_inputs.hpp"
#include "utils/two_species_projector_system.hpp"

using namespace q3;
using test_support::OneKpointQuantumSystem;
using test_support::ProjectorStrainTestInputs;
using test_support::TwoSpeciesProjectorSystem;

namespace {

constexpr int number_of_bands = 5;
constexpr int number_of_kinds = ProjectorStrainOverlapCalculator::number_of_kinds;
// Species A carries 4 projectors per atom and species B 18; the atoms are A, B, A, B.
constexpr int projectors_of_all_atoms = 4 + 18 + 4 + 18;

/// The two-species cell, bands over its k-point, and a host strain overlap calculator.
struct HostStrainCase {
    TwoSpeciesProjectorSystem system;
    ProjectorStrainTestInputs strain_inputs{system.crystal_structure,
                                            system.planewave_basis,
                                            system.kpoint,
                                            system.g_vector_indices,
                                            /*maximum_angular_momentum=*/2};
    KPointsGrid kpoint_grid;
    OneKpointQuantumSystem quantum_system;
    SerialBandProjectionCalculator band_projections;
    SerialProjectorStrainOverlapCalculator strain_overlaps{system.crystal_structure,
                                                           system.planewave_basis,
                                                           system.pseudopotential_basis(),
                                                           band_projections};
    std::vector<int> band_indices = {0, 1, 2, 3, 4};

    HostStrainCase() : quantum_system(make_grid(kpoint_grid), number_of_bands) {
        quantum_system.allocate_electronic_states({system.number_of_plane_waves()});
        quantum_system.fill_bands_with_a_pattern();
    }

    static const KPointsGrid& make_grid(KPointsGrid& grid) {
        grid.generate(1, 1, 1, 0, 0, 0);
        return grid;
    }

    [[nodiscard]] std::vector<Complex> block(int first_atom, int end_atom, int projectors) {
        std::vector<Complex> overlaps(static_cast<std::size_t>(number_of_kinds) * projectors *
                                      band_indices.size());
        strain_overlaps.compute_block(
            first_atom, end_atom, projectors, quantum_system, 0, band_indices, overlaps.data());
        return overlaps;
    }
};

} // namespace

TEST(SerialProjectorStrainOverlapCalculator, TheFirstRowsAreTheProjectorOverlaps) {
    HostStrainCase strain_case;
    strain_case.strain_overlaps.begin_kpoint(strain_case.strain_inputs.inputs(),
                                             projectors_of_all_atoms);

    const std::vector<Complex> overlaps = strain_case.block(0, 4, projectors_of_all_atoms);

    KleinmanBylanderProjectorSet projector_set;
    SerialProjectorSetBuilder(strain_case.system.crystal_structure,
                              strain_case.system.planewave_basis,
                              strain_case.system.pseudopotential_basis())
        .build(strain_case.system.kpoint,
               strain_case.system.g_vector_indices,
               strain_case.system.number_of_plane_waves(),
               projector_set);
    std::vector<Complex> projector_overlaps(static_cast<std::size_t>(projectors_of_all_atoms) *
                                            number_of_bands);
    strain_case.band_projections.project(projector_set.matrix(),
                                         projectors_of_all_atoms,
                                         strain_case.system.number_of_plane_waves(),
                                         strain_case.quantum_system,
                                         0,
                                         strain_case.band_indices,
                                         projector_overlaps.data());
    const std::size_t block_rows =
        static_cast<std::size_t>(number_of_kinds) * projectors_of_all_atoms;
    for (int band = 0; band < number_of_bands; ++band) {
        for (int projector = 0; projector < projectors_of_all_atoms; ++projector) {
            const Complex from_strain_block = overlaps[projector + band * block_rows];
            const Complex from_projector_set =
                projector_overlaps[projector + band * projectors_of_all_atoms];
            EXPECT_NEAR(from_strain_block.real(), from_projector_set.real(), 1.0e-12);
            EXPECT_NEAR(from_strain_block.imag(), from_projector_set.imag(), 1.0e-12);
        }
    }
}

TEST(SerialProjectorStrainOverlapCalculator, AtomsTakenInSmallerBlocksGiveTheSameOverlaps) {
    HostStrainCase strain_case;
    strain_case.strain_overlaps.begin_kpoint(strain_case.strain_inputs.inputs(),
                                             projectors_of_all_atoms);
    const std::vector<Complex> all_atoms = strain_case.block(0, 4, projectors_of_all_atoms);
    // Atoms 1 and 2: species B then species A, 18 + 4 projectors, starting at projector 4.
    const std::vector<Complex> middle_atoms = strain_case.block(1, 3, 22);

    const std::size_t all_rows =
        static_cast<std::size_t>(number_of_kinds) * projectors_of_all_atoms;
    const std::size_t middle_rows = static_cast<std::size_t>(number_of_kinds) * 22;
    for (int band = 0; band < number_of_bands; ++band) {
        for (int kind = 0; kind < number_of_kinds; ++kind) {
            for (int projector = 0; projector < 22; ++projector) {
                // To round-off only: a product of another shape sums in another order.
                const Complex in_small_block =
                    middle_atoms[kind * 22 + projector + band * middle_rows];
                const Complex in_whole_cell =
                    all_atoms[kind * projectors_of_all_atoms + 4 + projector + band * all_rows];
                EXPECT_NEAR(in_small_block.real(), in_whole_cell.real(), 1.0e-12);
                EXPECT_NEAR(in_small_block.imag(), in_whole_cell.imag(), 1.0e-12);
            }
        }
    }
}

TEST(SerialProjectorStrainOverlapCalculator, TheStrainDerivativeRowsAreNotZero) {
    HostStrainCase strain_case;
    strain_case.strain_overlaps.begin_kpoint(strain_case.strain_inputs.inputs(),
                                             projectors_of_all_atoms);

    const std::vector<Complex> overlaps = strain_case.block(0, 4, projectors_of_all_atoms);

    for (int kind = 1; kind < number_of_kinds; ++kind) {
        Real largest = 0.0;
        for (int projector = 0; projector < projectors_of_all_atoms; ++projector) {
            largest =
                std::max(largest, std::abs(overlaps[kind * projectors_of_all_atoms + projector]));
        }
        EXPECT_GT(largest, 1.0e-6) << "strain component " << kind - 1;
    }
}

TEST(SerialProjectorStrainOverlapCalculator, ValuesAtAPlaneWaveAreTheColumnsValuesThere) {
    HostStrainCase strain_case;
    strain_case.strain_overlaps.begin_kpoint(strain_case.strain_inputs.inputs(),
                                             projectors_of_all_atoms);
    (void)strain_case.block(0, 4, projectors_of_all_atoms);
    const int plane_wave = 3;

    std::vector<Complex> values;
    strain_case.strain_overlaps.block_values_at_plane_wave(plane_wave, values);

    ASSERT_EQ(values.size(), static_cast<std::size_t>(number_of_kinds) * projectors_of_all_atoms);
    KleinmanBylanderProjectorSet projector_set;
    SerialProjectorSetBuilder(strain_case.system.crystal_structure,
                              strain_case.system.planewave_basis,
                              strain_case.system.pseudopotential_basis())
        .build(strain_case.system.kpoint,
               strain_case.system.g_vector_indices,
               strain_case.system.number_of_plane_waves(),
               projector_set);
    for (int projector = 0; projector < projectors_of_all_atoms; ++projector) {
        const Complex expected = projector_set.projector(projector)[plane_wave];
        EXPECT_NEAR(values[projector].real(), expected.real(), 1.0e-14);
        EXPECT_NEAR(values[projector].imag(), expected.imag(), 1.0e-14);
    }
}
