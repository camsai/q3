#include <gtest/gtest.h>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/direct.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

using namespace q3;

static execution::PlanewaveSerialExecutionPolicy serial_policy;

// Helper to create a PseudopotentialSet from a single unique_ptr
static PseudopotentialSet make_single_set(std::unique_ptr<Pseudopotential> pp) {
    PseudopotentialSet set;
    set.push_back(std::move(pp));
    return set;
}

// ── Read US-PP file ─────────────────────────────────────────────────
TEST(USPP, ReadUPF) {
    auto pp =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");

    EXPECT_EQ(pp->element, "Si");
    EXPECT_EQ(pp->pseudopotential_type, "US");
    EXPECT_EQ(pp->pseudopotential_type_enum(), PseudopotentialType::ultrasoft);
    EXPECT_NEAR(pp->valence_charge, 4.0, 1e-6);
    EXPECT_EQ(pp->maximum_angular_momentum, 2);
    EXPECT_EQ(pp->number_of_projectors, 4);
    EXPECT_EQ(pp->mesh_size, 1141);
}

TEST(USPP, QFunction) {
    auto pp =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/us/Si_PBE_USPP.UPF");

    int nb = pp->number_of_projectors;
    const auto& us_pp = static_cast<const UltrasoftPseudopotential&>(*pp);

    // Q functions should exist for all beta pairs
    EXPECT_EQ((int)us_pp.augmentation_function.size(), nb * nb);

    // qqq matrix should exist
    EXPECT_EQ((int)us_pp.augmentation_overlap_integrals.size(), nb);
    EXPECT_EQ((int)us_pp.augmentation_overlap_integrals[0].size(), nb);

    // qqq should be symmetric
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < nb; j++) {
            EXPECT_NEAR(us_pp.augmentation_overlap_integrals[i][j],
                        us_pp.augmentation_overlap_integrals[j][i],
                        1e-14)
                << "qqq[" << i << "][" << j << "] not symmetric";
        }
    }

    // Q functions should have mesh_size points
    for (int i = 0; i < nb; i++) {
        int ij = i * nb + i; // diagonal
        EXPECT_EQ((int)us_pp.augmentation_function[ij].size(), pp->mesh_size);
    }
}

TEST(USPP, SMatrixIdentityForNC) {
    // For NC-PP, S matrix should be identity
    auto pp =
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF");

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048);
    AtomSpecies si{"Si", 28.08, 4.0, ""};
    crystal_structure.add_species(si);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();

    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal_structure, 16.0, 64.0);

    NormConservingPlanewaveKohnShamQuantumSystem pseudopotential_quantum_system;
    pseudopotential_quantum_system.init(
        crystal_structure, planewave_basis, make_single_set(std::move(pp)));

    EXPECT_FALSE(pseudopotential_quantum_system.has_ultrasoft());
}

TEST(USPP, DirectEigenvaluesCheck) {
    auto pp = UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Cu.upf");

    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.843194);
    AtomSpecies cu{"Cu", 63.546, pp->valence_charge, "Cu.upf"};
    crystal_structure.add_species(cu);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();

    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy, crystal_structure, 45.0, 180.0);

    NormConservingPlanewaveKohnShamQuantumSystem pseudopotential_quantum_system;
    pseudopotential_quantum_system.init(
        crystal_structure, planewave_basis, make_single_set(std::move(pp)));

    Vector3D k_cryst{0.25, 0.25, 0.25};
    std::vector<int> g_vector_indices;
    std::vector<Real> kinetic_energy_squared;
    planewave_basis.compute_gk(
        k_cryst, crystal_structure, g_vector_indices, kinetic_energy_squared);

    int number_of_plane_waves = g_vector_indices.size();
    int number_of_bands = 14;

    std::vector<Real> real_space_potential(planewave_basis.fft_rho().total_grid_points(), 0.0);

    PlanewaveDirectDiagonalizer diag;

    KPointsGrid kpoint_grid;
    KPoint test_kpoint;
    test_kpoint.coordinates = k_cryst;
    test_kpoint.weight = 1.0;
    kpoint_grid.add_kpoint(test_kpoint);

    PlanewaveKohnShamQuantumSystem quantum_system(
        kpoint_grid, planewave_basis, number_of_bands, 1, 0.0);
    std::vector<int> pw_counts = {number_of_plane_waves};
    quantum_system.allocate_electronic_states(pw_counts);

    diag.solve(0,
               0,
               number_of_plane_waves,
               number_of_bands,
               g_vector_indices,
               kinetic_energy_squared,
               real_space_potential,
               crystal_structure,
               kpoint_grid,
               planewave_basis,
               pseudopotential_quantum_system,
               quantum_system);

    // All eigenvalues must be finite (not NaN or Inf)
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        EXPECT_TRUE(std::isfinite(quantum_system.eigenvalue(0, band_index)))
            << "Eigenvalue for band " << band_index << " is not finite";
    }

    // Eigenvalues must be sorted in ascending order
    for (int band_index = 1; band_index < number_of_bands; band_index++) {
        EXPECT_LE(quantum_system.eigenvalue(0, band_index - 1),
                  quantum_system.eigenvalue(0, band_index))
            << "Eigenvalues not sorted: band " << band_index - 1 << " > band " << band_index;
    }

    /**
     * First eigenvalue should be physically reasonable for T + V_NL (no local potential).
     * Without the local potential, the lowest eigenvalue may be positive.
     * Check it is within a reasonable range (e.g., below 10 Ry).
     */
    EXPECT_LT(quantum_system.eigenvalue(0, 0), 10.0)
        << "First eigenvalue should be within a reasonable range";

    /**
     * Number of eigenvalues should match the requested number of bands
     * (verified implicitly by accessing band indices 0..number_of_bands-1 above,
     *  but also check the last band is accessible and finite)
     */
    EXPECT_TRUE(std::isfinite(quantum_system.eigenvalue(0, number_of_bands - 1)))
        << "Last eigenvalue (band " << number_of_bands - 1 << ") should be finite";
}
