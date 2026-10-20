#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "io/input/data.hpp"
#include "io/input/parser/yaml.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"
#include "layer_4_computation/flavors/factory.hpp"
#include "layer_4_computation/flavors/result.hpp"

using namespace q3;

/**
 * ── A non-self-consistent pass on a PAW dataset ─────────────────────────
 *
 * A non-self-consistent run diagonalizes the Hamiltonian its SCF converged
 * to, at k-points of its own. On a PAW dataset part of that Hamiltonian is
 * D's one-centre term, built from the on-site occupations ρ_ij, and the pass
 * used to diagonalize before ever building it: the checkpoint carried no
 * ρ_ij, and the term is filled only when a density is constructed. Every
 * eigenvalue came out wrong -- copper's 3s by 103 eV -- while nothing
 * compared a PAW pass with the SCF it continues.
 *
 * So this does. Γ is the first point of the Γ-centred mesh the SCF ran on,
 * which makes the SCF's own eigenvalues there the answer the pass must
 * reproduce; two runs on one checkpoint, not either against a reference.
 */

namespace {

/// Diamond on the shipped PAW dataset, at cutoffs and a mesh that run in seconds.
InputData diamond_paw_input(const std::string& output_directory) {
    YAMLInputParser parser;
    InputData input =
        parser.parse(std::string(EXAMPLES_DIR) + "/total_energy--insulator--diamond/paw.yml");
    input.wavefunction_energy_cutoff = 30.0;
    input.charge_density_energy_cutoff = 240.0;
    input.kpoint_grid_dimension_1 = 2;
    input.kpoint_grid_dimension_2 = 2;
    input.kpoint_grid_dimension_3 = 2;
    input.kpoint_grid_shift_1 = 0;
    input.kpoint_grid_shift_2 = 0;
    input.kpoint_grid_shift_3 = 0;
    /**
     * Converged far enough that the SCF's eigenvalues are those of the state its checkpoint
     * holds: at 1e-10 the two sat 3e-6 Ry apart, which is the SCF, not the pass.
     */
    input.convergence_threshold = 1.0e-12;
    input.number_of_bands = 8;
    input.output_directory = output_directory;
    input.prefix = "diamond_paw";
    return input;
}

} // namespace

TEST(NonSelfConsistentFieldIntegration, APawPassReproducesTheBandsOfTheSelfConsistentRun) {
    const std::string directory =
        (std::filesystem::temp_directory_path() / "q3-paw-non-self-consistent").string();
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const InputData scf_input = diamond_paw_input(directory);
    const FlavorResult scf = create_flavor("self_consistent_field")->run(scf_input);
    ASSERT_TRUE(scf.converged);

    InputData nscf_input = scf_input;
    nscf_input.calculation = "non_self_consistent_field";
    nscf_input.kpoints_list = {{Vector3D{0.0, 0.0, 0.0}, 1.0}};
    const FlavorResult nscf = create_flavor("non_self_consistent_field")->run(nscf_input);

    const auto& scf_gamma = scf.eigenvalues_in_ry.at(0).at(0);
    const auto& nscf_gamma = nscf.eigenvalues_in_ry.at(0).at(0);
    ASSERT_EQ(nscf_gamma.size(), scf_gamma.size());
    /**
     * Every band but the highest, which a single pass converges least well
     * whatever the Hamiltonian. Measured at 3.4e-7 Ry; without the one-centre
     * term the same comparison misses by 0.067 Ry.
     */
    for (size_t band = 0; band + 1 < scf_gamma.size(); band++) {
        EXPECT_NEAR(nscf_gamma[band], scf_gamma[band], 1.0e-6) << "band " << band + 1;
    }

    /**
     * A checkpoint without the occupations -- what a binary before version 3
     * wrote -- cannot give the pass its one-centre term, and is refused rather
     * than diagonalized without it.
     */
    const std::string checkpoint_path = directory + "/diamond_paw.save/effective-potential.q3cp";
    Q3PlanewaveKohnShamQuantumSystemCheckpoint checkpoint;
    KohnShamQuantumSystemSnapshot snapshot;
    ASSERT_TRUE(checkpoint.load(checkpoint_path, snapshot));
    ASSERT_FALSE(snapshot.one_center_occupations.empty())
        << "an SCF on a PAW dataset stores its on-site occupations";
    snapshot.one_center_occupations.clear();
    ASSERT_TRUE(checkpoint.save(checkpoint_path, snapshot));
    try {
        create_flavor("non_self_consistent_field")->run(nscf_input);
        FAIL() << "a PAW pass without the SCF's on-site occupations must be refused";
    } catch (const std::runtime_error& refusal) {
        EXPECT_NE(std::string(refusal.what()).find("on-site occupations"), std::string::npos)
            << refusal.what();
    }
    std::filesystem::remove_all(directory);
}

/**
 * ── A pass over several k-points keeps its tolerance ──────────────────────
 *
 * A non-self-consistent run is one turn of the SCF loop at the tight floor,
 * and that turn ends with the SCF's first-iteration rule: re-diagonalize when
 * the density residual is below what the first pass could have left. The rule
 * took the first pass to have run at the loose SCF start, 1e-2, so its
 * "tighter" threshold, 0.1 dr2 / N, was 7.8e-4 Ry here -- far looser than the
 * floor the pass had run at. Whether it fired depended only on the residual
 * against the atomic-superposition density, which two k-points weighted alike
 * put under the bar, and a single k-point did not. Seeding each k-point from
 * the one before, the loose repeat then left the second k-point's occupied
 * bands up to 4.4e-5 Ry from a pass over that k-point alone.
 *
 * Both passes here diagonalize the same checkpointed potential, so they must
 * agree to the tolerance: 2.2e-14 Ry measured.
 */
TEST(NonSelfConsistentFieldIntegration, APassOverSeveralKPointsKeepsItsTolerance) {
    const std::string directory =
        (std::filesystem::temp_directory_path() / "q3-silicon-non-self-consistent").string();
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    YAMLInputParser parser;
    InputData scf_input =
        parser.parse(std::string(EXAMPLES_DIR) + "/total_energy--semiconductor--silicon/nc.yml");
    scf_input.wavefunction_energy_cutoff = 30.0;
    scf_input.charge_density_energy_cutoff = 120.0;
    scf_input.convergence_threshold = 1.0e-10;
    scf_input.use_deterministic_fft_planning = true;
    scf_input.output_directory = directory;
    scf_input.prefix = "silicon_nc";
    const FlavorResult scf = create_flavor("self_consistent_field")->run(scf_input);
    ASSERT_TRUE(scf.converged);

    // The two irreducible points of the shifted 2x2x2 mesh the SCF ran on.
    const Vector3D first_point{0.25, 0.25, 0.25};
    const Vector3D second_point{0.25, 0.25, -0.25};
    InputData nscf_input = scf_input;
    nscf_input.calculation = "non_self_consistent_field";
    nscf_input.kpoints_list = {{first_point, 1.0}, {second_point, 1.0}};
    const FlavorResult both = create_flavor("non_self_consistent_field")->run(nscf_input);
    nscf_input.kpoints_list = {{second_point, 1.0}};
    const FlavorResult alone = create_flavor("non_self_consistent_field")->run(nscf_input);

    const auto& second_in_both = both.eigenvalues_in_ry.at(1).at(0);
    const auto& second_alone = alone.eigenvalues_in_ry.at(0).at(0);
    ASSERT_EQ(second_in_both.size(), second_alone.size());
    // Silicon's eight valence electrons fill four bands.
    const size_t number_of_occupied_bands = 4;
    for (size_t band = 0; band < number_of_occupied_bands; band++) {
        EXPECT_NEAR(second_in_both[band], second_alone[band], 1.0e-10) << "band " << band + 1;
    }
    std::filesystem::remove_all(directory);
}
