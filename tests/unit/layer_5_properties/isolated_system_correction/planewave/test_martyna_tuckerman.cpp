#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/atom.hpp"
#include "layer_1_structure/crystal/atom_species.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/factory.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/makov_payne.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/martyna_tuckerman.hpp"

using namespace q3;

namespace {

static execution::PlanewaveSerialExecutionPolicy serial_policy;

/// Valence charge given to the single synthetic atom in every test below.
constexpr Real VALENCE_CHARGE = 4.0;

/**
 * Simple-cubic, single-atom-at-the-origin crystal (bravais_lattice_index 1, so both
 * MakovPayneCorrectionCalculator and MartynaTuckermanCorrectionCalculator accept it --
 * needed for the monopole-limit comparison test below).
 */
Crystal build_single_atom_simple_cubic_crystal(Real lattice_parameter_bohr) {
    std::vector<AtomSpecies> species = {AtomSpecies{"X", 1.0, VALENCE_CHARGE, ""}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}};
    return Crystal::from_input_data(1, lattice_parameter_bohr, species, atoms);
}

PlanewaveBasis build_planewave_basis(const Crystal& crystal_structure,
                                     Real wavefunction_energy_cutoff_ry) {
    PlanewaveBasis planewave_basis;
    planewave_basis.generate(serial_policy,
                             crystal_structure,
                             wavefunction_energy_cutoff_ry,
                             4.0 * wavefunction_energy_cutoff_ry);
    return planewave_basis;
}

/**
 * Build a single-spin-channel electron number density: an isotropic Gaussian of width
 * `sigma_in_bohr` centered at `center_alat` (Cartesian, alat units), minimum-image
 * wrapped exactly like MartynaTuckermanCorrectionCalculator's own grid traversal, and
 * normalized so that integrating it over the cell gives exactly `target_charge`.
 */
std::vector<Real> build_isotropic_gaussian_charge_density(const Crystal& crystal_structure,
                                                          const FFTGrid& density_grid,
                                                          const Vector3D& center_alat,
                                                          Real sigma_in_bohr,
                                                          Real target_charge) {
    int grid_dimension_1 = density_grid.grid_dimension_1();
    int grid_dimension_2 = density_grid.grid_dimension_2();
    int grid_dimension_3 = density_grid.grid_dimension_3();
    int total_grid_points = density_grid.total_grid_points();
    Real alat = crystal_structure.lattice_parameter();

    Matrix3x3 lattice_vectors_transpose = crystal_structure.lattice_vectors().transpose();
    Matrix3x3 lattice_inverse_transpose = crystal_structure.lattice_vectors().inverse().transpose();

    std::vector<Real> unnormalized_density(total_grid_points);
    Real unnormalized_sum = 0.0;
    for (int flat_index = 0; flat_index < total_grid_points; flat_index++) {
        int index_3 = flat_index % grid_dimension_3;
        int index_2 = (flat_index / grid_dimension_3) % grid_dimension_2;
        int index_1 = flat_index / (grid_dimension_3 * grid_dimension_2);
        Vector3D fractional_grid_point{static_cast<Real>(index_1) / grid_dimension_1,
                                       static_cast<Real>(index_2) / grid_dimension_2,
                                       static_cast<Real>(index_3) / grid_dimension_3};
        Vector3D grid_point_alat = lattice_vectors_transpose * fractional_grid_point;

        Vector3D displacement = grid_point_alat - center_alat;
        Vector3D fractional_displacement = lattice_inverse_transpose * displacement;
        fractional_displacement.x -= std::round(fractional_displacement.x);
        fractional_displacement.y -= std::round(fractional_displacement.y);
        fractional_displacement.z -= std::round(fractional_displacement.z);
        Vector3D wrapped_displacement = lattice_vectors_transpose * fractional_displacement;
        Real distance_in_bohr = wrapped_displacement.norm() * alat;

        Real value = std::exp(-(distance_in_bohr * distance_in_bohr) /
                              (2.0 * sigma_in_bohr * sigma_in_bohr));
        unnormalized_density[flat_index] = value;
        unnormalized_sum += value;
    }

    Real grid_volume_element = crystal_structure.cell_volume() / total_grid_points;
    Real scale = target_charge / (unnormalized_sum * grid_volume_element);
    for (Real& value : unnormalized_density) {
        value *= scale;
    }
    return unnormalized_density;
}

} // namespace

// ── Neutral, symmetric limit ────────────────────────────────────────────

/**
 * A single ion and an isotropic Gaussian electron cloud of the SAME integrated charge,
 * centered at the SAME point: net charge, dipole, AND (by the classical
 * concentric-spherical-shell/Gauss's-law argument) the electrostatic interaction with
 * every periodic image are all zero to excellent approximation -- unlike Makov-Payne's
 * multipole series (which only vanishes to dipole order here; see this test file's
 * "matches_makov_payne_first_order" test docstring for the quadrupole-order term MP
 * would still report), the Martyna-Tuckerman kernel-difference sum has no natural
 * "order" to truncate at, so this checks the FULL sum comes out negligibly small
 * directly, not merely its leading orders.
 */
TEST(MartynaTuckermanCorrectionCalculator, NeutralConcentricDensityGivesNegligibleCorrection) {
    Real lattice_parameter_bohr = 20.0;
    Crystal crystal_structure = build_single_atom_simple_cubic_crystal(lattice_parameter_bohr);
    PlanewaveBasis planewave_basis = build_planewave_basis(crystal_structure, 8.0);

    std::vector<Real> electron_density = build_isotropic_gaussian_charge_density(
        crystal_structure, planewave_basis.fft_rho(), Vector3D{0.0, 0.0, 0.0}, 1.0, VALENCE_CHARGE);

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;
    scf_state.planewave_basis = &planewave_basis;
    scf_state.charge_density_real = {electron_density};

    auto result = MartynaTuckermanCorrectionCalculator{}.compute(scf_state);

    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.method, "martyna-tuckerman");
    ASSERT_EQ(result.terms_in_ry.size(), 1u);
    EXPECT_EQ(result.terms_in_ry[0].first, "energy");
    EXPECT_NEAR(result.total_in_ry(), 0.0, 1.0e-5)
        << "a neutral, concentrically-symmetric charge distribution should have an "
           "(almost) exactly vanishing Martyna-Tuckerman correction";
}

// ── Monopole (large-cell) limit vs. Makov-Payne ─────────────────────────

/**
 * For a net-charged, compact density, the Martyna-Tuckerman correction's exact
 * reciprocal-space sum and Makov-Payne's first-order (Madelung monopole) term
 * describe the SAME leading-order physics -- the periodic self-interaction energy of
 * a point-like net charge with its own periodic images -- so their ratio should
 * approach 1 as the cell grows and the compact charge looks increasingly point-like
 * relative to the cell (higher multipole content contributes relatively less at
 * larger separation). Verified independently in a standalone numpy prototype
 * (not checked in -- see the Day-2 session notes) before writing this test, so the
 * bounds below are measured/expected behavior, not a guess.
 */
TEST(MartynaTuckermanCorrectionCalculator, ApproachesMakovPayneFirstOrderTermAsCellGrows) {
    std::vector<Real> lattice_parameters_bohr = {14.0, 20.0, 30.0};
    Real net_charge = 1.0;
    Real sigma_in_bohr = 1.0;

    std::vector<Real> ratios;
    for (Real lattice_parameter_bohr : lattice_parameters_bohr) {
        Crystal crystal_structure = build_single_atom_simple_cubic_crystal(lattice_parameter_bohr);
        PlanewaveBasis planewave_basis = build_planewave_basis(crystal_structure, 8.0);

        std::vector<Real> electron_density =
            build_isotropic_gaussian_charge_density(crystal_structure,
                                                    planewave_basis.fft_rho(),
                                                    Vector3D{0.0, 0.0, 0.0},
                                                    sigma_in_bohr,
                                                    VALENCE_CHARGE - net_charge);

        NormConservingPlanewaveKohnShamSCFState scf_state;
        scf_state.crystal_structure = &crystal_structure;
        scf_state.planewave_basis = &planewave_basis;
        scf_state.charge_density_real = {electron_density};

        auto martyna_tuckerman_result = MartynaTuckermanCorrectionCalculator{}.compute(scf_state);
        auto makov_payne_result = MakovPayneCorrectionCalculator{}.compute(scf_state);

        Real martyna_tuckerman_energy = martyna_tuckerman_result.total_in_ry();
        Real makov_payne_first_order = makov_payne_result.terms_in_ry[0].second;
        ASSERT_EQ(makov_payne_result.terms_in_ry[0].first, "first_order");
        ASSERT_GT(makov_payne_first_order, 0.0);

        Real ratio = martyna_tuckerman_energy / makov_payne_first_order;
        ratios.push_back(ratio);

        /**
         * Same sign, same order of magnitude at every size tested -- catches a sign or
         * gross scaling bug immediately, independent of the tighter trend check below.
         */
        EXPECT_GT(martyna_tuckerman_energy, 0.0)
            << "at lattice_parameter_bohr = " << lattice_parameter_bohr;
        EXPECT_NEAR(ratio, 1.0, 0.15) << "at lattice_parameter_bohr = " << lattice_parameter_bohr;
    }

    // The ratio must trend monotonically toward 1 as the cell grows.
    ASSERT_EQ(ratios.size(), 3u);
    Real deviation_smallest_cell = std::abs(ratios[0] - 1.0);
    Real deviation_middle_cell = std::abs(ratios[1] - 1.0);
    Real deviation_largest_cell = std::abs(ratios[2] - 1.0);
    EXPECT_GT(deviation_smallest_cell, deviation_middle_cell)
        << "ratio should approach 1 monotonically as the cell grows (smallest vs. middle cell)";
    EXPECT_GT(deviation_middle_cell, deviation_largest_cell)
        << "ratio should approach 1 monotonically as the cell grows (middle vs. largest cell)";
}

// ── Factory / interface conformance ─────────────────────────────────────

TEST(MartynaTuckermanCorrectionCalculator, NameMatchesFactoryKey) {
    MartynaTuckermanCorrectionCalculator calculator;
    EXPECT_EQ(calculator.name(), "martyna-tuckerman");
}
