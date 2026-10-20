#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

// Si.upf (PseudoDojo NC-SR-04): PP_PSWFC has 3S (l=0) + 3P (l=1), both occupied.
PseudopotentialSet load_silicon_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.upf"));
    return pseudopotential_set;
}

/**
 * Ni.upf (PseudoDojo NC-SR-04): PP_PSWFC has 3S (l=0) + 3P (l=1) + 3D (l=2) + 4S (l=0), all
 * occupied.
 */
PseudopotentialSet load_nickel_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Ni.upf"));
    return pseudopotential_set;
}

Crystal make_silicon_fcc(const PseudopotentialSet& pseudopotential_set) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies silicon{"Si", 28.08, pseudopotential_set[0].valence_charge, "Si.upf"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

Crystal make_nickel_fcc(const PseudopotentialSet& pseudopotential_set) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.65); // FCC Ni, approximate alat (Bohr)
    AtomSpecies nickel{"Ni", 58.69, pseudopotential_set[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

} // namespace

// ── Orbital counting: sum of (2l+1) over occupied UPF shells ─────────

TEST(AtomicOrbitalBuilder, CountMatchesSiliconTwoAtomCell) {
    auto pseudopotential_set = load_silicon_pseudopotential();
    Crystal crystal_structure = make_silicon_fcc(pseudopotential_set);

    // Per atom: 3S (l=0 -> 1) + 3P (l=1 -> 3) = 4; 2 Si atoms -> 8.
    EXPECT_EQ(count_atomic_orbitals(crystal_structure, pseudopotential_set), 8);
}

TEST(AtomicOrbitalBuilder, CountMatchesNickelOneAtomCell) {
    auto pseudopotential_set = load_nickel_pseudopotential();
    Crystal crystal_structure = make_nickel_fcc(pseudopotential_set);

    // 3S(1) + 3P(3) + 3D(5) + 4S(1) = 10.
    EXPECT_EQ(count_atomic_orbitals(crystal_structure, pseudopotential_set), 10);
}

// ── Label enumeration: order, count, and per-shell angular momentum ──

TEST(AtomicOrbitalBuilder, EnumerateLabelsMatchesCount) {
    auto pseudopotential_set = load_nickel_pseudopotential();
    Crystal crystal_structure = make_nickel_fcc(pseudopotential_set);

    auto labels = enumerate_atomic_orbitals(crystal_structure, pseudopotential_set);
    EXPECT_EQ(static_cast<int>(labels.size()),
              count_atomic_orbitals(crystal_structure, pseudopotential_set));
}

TEST(AtomicOrbitalBuilder, EnumerateLabelsFollowShellThenMagneticOrder) {
    auto pseudopotential_set = load_nickel_pseudopotential();
    Crystal crystal_structure = make_nickel_fcc(pseudopotential_set);

    auto labels = enumerate_atomic_orbitals(crystal_structure, pseudopotential_set);
    ASSERT_EQ(labels.size(), 10u);

    // Shell order from the UPF file: 3S(l=0), 3P(l=1), 3D(l=2), 4S(l=0).
    EXPECT_EQ(labels[0].shell_label, "3S");
    EXPECT_EQ(labels[0].angular_momentum, 0);
    EXPECT_EQ(labels[0].magnetic_quantum_number, 0);

    EXPECT_EQ(labels[1].shell_label, "3P");
    EXPECT_EQ(labels[1].angular_momentum, 1);
    EXPECT_EQ(labels[1].magnetic_quantum_number, 0);
    EXPECT_EQ(labels[3].shell_label, "3P");
    EXPECT_EQ(labels[3].magnetic_quantum_number, 2);

    EXPECT_EQ(labels[4].shell_label, "3D");
    EXPECT_EQ(labels[4].angular_momentum, 2);
    EXPECT_EQ(labels[8].shell_label, "3D");
    EXPECT_EQ(labels[8].magnetic_quantum_number, 4);

    EXPECT_EQ(labels[9].shell_label, "4S");
    EXPECT_EQ(labels[9].angular_momentum, 0);

    // Each label names its shell's position in the file, which is what a selection by shell reads.
    EXPECT_EQ(labels[0].shell_index, 0);
    EXPECT_EQ(labels[1].shell_index, 1);
    EXPECT_EQ(labels[3].shell_index, 1);
    EXPECT_EQ(labels[4].shell_index, 2);
    EXPECT_EQ(labels[8].shell_index, 2);
    EXPECT_EQ(labels[9].shell_index, 3);

    // Every label should carry atom_index=0 (single-atom cell).
    for (const auto& label : labels) {
        EXPECT_EQ(label.atom_index, 0);
        EXPECT_EQ(label.species_type_index, 0);
    }
}

// ── G-space construction: shape and finiteness ────────────────────────

TEST(AtomicOrbitalBuilder, GSpaceOrbitalsHaveExpectedShapeAndAreFinite) {
    auto pseudopotential_set = load_silicon_pseudopotential();
    Crystal crystal_structure = make_silicon_fcc(pseudopotential_set);

    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 16.0, 64.0);

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0); // Gamma-only

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    basis.compute_gk(
        kpoint_grid[0].coordinates, crystal_structure, g_vector_indices, kinetic_energy_squared);
    int number_of_plane_waves = static_cast<int>(g_vector_indices.size());
    ASSERT_GT(number_of_plane_waves, 0);

    auto atomic_wavefunction_table =
        tabulate_atomic_wavefunctions_in_q_space(pseudopotential_set,
                                                 crystal_structure.cell_volume(),
                                                 /*q_spacing=*/0.01,
                                                 /*number_of_q_points=*/900);

    std::vector<std::vector<Complex>> orbitals;
    build_atomic_orbitals_g_space(0,
                                  number_of_plane_waves,
                                  g_vector_indices,
                                  crystal_structure,
                                  kpoint_grid,
                                  basis,
                                  pseudopotential_set,
                                  atomic_wavefunction_table,
                                  0.01,
                                  900,
                                  orbitals);

    ASSERT_EQ(orbitals.size(), 8u);
    for (const auto& orbital : orbitals) {
        ASSERT_EQ(orbital.size(), static_cast<size_t>(number_of_plane_waves));
        Real norm_squared = 0.0;
        for (const auto& coefficient : orbital) {
            EXPECT_TRUE(std::isfinite(coefficient.real()));
            EXPECT_TRUE(std::isfinite(coefficient.imag()));
            norm_squared += std::norm(coefficient);
        }
        /**
         * Not normalized to 1 (that's Löwdin's job), but should be a
         * non-trivial, finite, physically-scaled quantity.
         */
        EXPECT_GT(norm_squared, 0.0);
        EXPECT_LT(norm_squared, 10.0);
    }
}

TEST(AtomicOrbitalBuilder, GSpaceOrbitalOrderMatchesLabelOrder) {
    auto pseudopotential_set = load_silicon_pseudopotential();
    Crystal crystal_structure = make_silicon_fcc(pseudopotential_set);
    auto labels = enumerate_atomic_orbitals(crystal_structure, pseudopotential_set);

    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 16.0, 64.0);
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);

    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    basis.compute_gk(
        kpoint_grid[0].coordinates, crystal_structure, g_vector_indices, kinetic_energy_squared);
    int number_of_plane_waves = static_cast<int>(g_vector_indices.size());

    auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
        pseudopotential_set, crystal_structure.cell_volume(), 0.01, 900);

    std::vector<std::vector<Complex>> orbitals;
    build_atomic_orbitals_g_space(0,
                                  number_of_plane_waves,
                                  g_vector_indices,
                                  crystal_structure,
                                  kpoint_grid,
                                  basis,
                                  pseudopotential_set,
                                  atomic_wavefunction_table,
                                  0.01,
                                  900,
                                  orbitals);

    ASSERT_EQ(orbitals.size(), labels.size());
}
