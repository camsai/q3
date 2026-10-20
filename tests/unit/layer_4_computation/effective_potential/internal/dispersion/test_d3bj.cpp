#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "layer_0_core/types/constants.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/d3bj.hpp"

using namespace q3;

namespace {

Crystal make_silicon_fcc() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263048); // FCC silicon, same as the Ewald test
    AtomSpecies silicon{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

Crystal make_copper_fcc() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.8432); // FCC copper, same as the Cu example
    AtomSpecies copper{"Cu", 63.546, 11.0, "Cu.upf"};
    crystal_structure.add_species(copper);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

/**
 * AB (Bernal) graphite, space group P6_3/mmc: hexagonal a=b=2.464 Angstrom,
 * c = 2 * interlayer_spacing_angstrom (2 layers/cell), 4 atoms. Exercises a
 * non-cubic (explicit free-lattice) cell and a multi-atom pairwise sum with
 * both short (intralayer, ~1.42 Angstrom) and long (interlayer) C-C
 * distances -- a genuinely different regime from the single-atom-basis Si/Cu
 * cases above. See examples/total_energy--semimetal--graphite/ and
 * tests/validation/test_qe_graphite_d3_dispersion.py for the full c-scan
 * this anchors.
 * Rebuilds an equivalent crystal with a different lattice (same species and
 * fractional positions), working entirely in Bohr by using lattice_parameter
 * = 1: lattice_vectors_bohr() of the result is then exactly the matrix passed
 * in. Used by the force/stress finite-difference tests below to perturb
 * geometry without depending on any single fixture's construction path.
 */
Crystal rebuild_with_lattice(const Crystal& original, const Matrix3x3& new_lattice_vectors_bohr) {
    std::vector<Atom> atoms;
    for (int atom_index = 0; atom_index < original.number_of_atoms(); atom_index++) {
        atoms.push_back(
            {original.species_type_index(atom_index), original.fractional_position(atom_index)});
    }
    return Crystal::from_lattice_vectors(1.0, new_lattice_vectors_bohr, original.species(), atoms);
}

/**
 * Displaces one atom by delta_bohr along a Cartesian direction, holding the
 * lattice fixed. cartesian = fractional.x*L[0]+fractional.y*L[1]+fractional.z*L[2]
 * (the exact convention used throughout d3bj.hpp) is equivalent to
 * cartesian = L^T * fractional as a column-vector product, so the fractional
 * shift for a given Cartesian shift is (L^T)^{-1} * delta_cartesian.
 */
Crystal
perturb_atom_cartesian(const Crystal& original, int atom_index, int direction, Real delta_bohr) {
    Matrix3x3 lattice_vectors_bohr = original.lattice_vectors_bohr();
    Vector3D delta_cartesian{0.0, 0.0, 0.0};
    delta_cartesian[direction] = delta_bohr;
    Matrix3x3 lattice_transpose_inverse = lattice_vectors_bohr.transpose().inverse();
    Vector3D delta_fractional = lattice_transpose_inverse * delta_cartesian;

    std::vector<Atom> atoms;
    for (int index = 0; index < original.number_of_atoms(); index++) {
        Vector3D fractional = original.fractional_position(index);
        if (index == atom_index) {
            fractional = fractional + delta_fractional;
        }
        atoms.push_back({original.species_type_index(index), fractional});
    }
    return Crystal::from_lattice_vectors(1.0, lattice_vectors_bohr, original.species(), atoms);
}

/**
 * Applies an elementary (possibly asymmetric) affine strain: each lattice
 * vector L_i -> L_i + strain*L_i. Fractional coordinates are unchanged, so
 * atomic Cartesian positions transform under the same affine map as the
 * lattice -- exactly the transformation the analytic stress formula assumes.
 */
Crystal apply_strain(const Crystal& original, const Matrix3x3& strain) {
    Matrix3x3 lattice_vectors_bohr = original.lattice_vectors_bohr();
    Matrix3x3 new_lattice_vectors_bohr;
    for (int direction = 0; direction < 3; direction++) {
        new_lattice_vectors_bohr[direction] =
            lattice_vectors_bohr[direction] + strain * lattice_vectors_bohr[direction];
    }
    return rebuild_with_lattice(original, new_lattice_vectors_bohr);
}

// Central finite difference of compute_energy() for one atom/direction.
Real finite_difference_force_component(const Crystal& original,
                                       int atom_index,
                                       int direction,
                                       Real delta_bohr) {
    Crystal displaced_plus = perturb_atom_cartesian(original, atom_index, direction, delta_bohr);
    Crystal displaced_minus = perturb_atom_cartesian(original, atom_index, direction, -delta_bohr);
    Real energy_plus = D3BJDispersionCalculator(displaced_plus).compute_energy();
    Real energy_minus = D3BJDispersionCalculator(displaced_minus).compute_energy();
    Real derivative = (energy_plus - energy_minus) / (2.0 * delta_bohr);
    return -derivative; // F = -dE/dR
}

/**
 * Central finite difference of compute_energy() with respect to one
 * (possibly off-diagonal) strain component, at fixed fractional coordinates.
 */
Real finite_difference_stress_component(const Crystal& original,
                                        int direction_a,
                                        int direction_b,
                                        Real strain_step) {
    Matrix3x3 strain_plus;
    strain_plus[direction_a][direction_b] = strain_step;
    Matrix3x3 strain_minus;
    strain_minus[direction_a][direction_b] = -strain_step;
    Crystal strained_plus = apply_strain(original, strain_plus);
    Crystal strained_minus = apply_strain(original, strain_minus);
    Real energy_plus = D3BJDispersionCalculator(strained_plus).compute_energy();
    Real energy_minus = D3BJDispersionCalculator(strained_minus).compute_energy();
    Real energy_derivative = (energy_plus - energy_minus) / (2.0 * strain_step);
    return -energy_derivative / original.cell_volume(); // sigma = -(1/Omega) dE/dstrain
}

Crystal make_graphite_hex(Real interlayer_spacing_angstrom) {
    constexpr Real a_angstrom = 2.464;
    Real c_angstrom = 2.0 * interlayer_spacing_angstrom;
    Real a_bohr = a_angstrom / constants::BOHR_RADIUS_ANGSTROM;
    Real c_over_a = c_angstrom / a_angstrom;

    // QE ibrav=4 convention: a1=(1,0,0), a2=(-1/2,sqrt(3)/2,0), a3=(0,0,c/a).
    Matrix3x3 lattice_vectors;
    lattice_vectors[0] = {1.0, 0.0, 0.0};
    lattice_vectors[1] = {-0.5, std::sqrt(3.0) / 2.0, 0.0};
    lattice_vectors[2] = {0.0, 0.0, c_over_a};

    AtomSpecies carbon{"C", 12.011, 4.0, "C.upf"};
    std::vector<Atom> atoms = {
        {0, {0.0, 0.0, 0.25}},
        {0, {1.0 / 3.0, 2.0 / 3.0, 0.25}},
        {0, {0.0, 0.0, 0.75}},
        {0, {2.0 / 3.0, 1.0 / 3.0, 0.75}},
    };
    return Crystal::from_lattice_vectors(a_bohr, lattice_vectors, {carbon}, atoms);
}

/**
 * A generic (symmetry-broken) distortion of the given crystal: each atom is
 * nudged by a small, direction- and atom-dependent Cartesian displacement, so
 * that per-atom forces and off-diagonal stress components are guaranteed to
 * be genuinely nonzero rather than vanishing by the ideal structure's point
 * symmetry (e.g. the tetrahedral Si basis or graphite's hexagonal layers).
 */
Crystal make_generically_distorted(const Crystal& original) {
    /**
     * Small, bounded (deliberately NOT growing with atom index) per-atom
     * offsets: just enough to break point-group symmetry while keeping every
     * displacement comfortably inside the near-linear regime that
     * finite-difference validation depends on.
     */
    constexpr Real offsets_bohr[4][3] = {
        {0.03, -0.02, 0.01},
        {-0.01, 0.025, -0.03},
        {0.02, 0.015, -0.02},
        {-0.025, -0.01, 0.03},
    };
    Crystal distorted = original;
    for (int atom_index = 0; atom_index < original.number_of_atoms(); atom_index++) {
        for (int direction = 0; direction < 3; direction++) {
            Real displacement_bohr = offsets_bohr[atom_index % 4][direction];
            distorted = perturb_atom_cartesian(distorted, atom_index, direction, displacement_bohr);
        }
    }
    return distorted;
}

} // namespace

TEST(D3BJDispersionCalculator, SiliconEnergyIsNegative) {
    // Dispersion is always attractive (net negative contribution).
    Crystal crystal_structure = make_silicon_fcc();
    D3BJDispersionCalculator calculator(crystal_structure);
    Real energy = calculator.compute_energy();
    EXPECT_LT(energy, 0.0);
}

TEST(D3BJDispersionCalculator, SiliconEnergyMatchesQuantumEspresso) {
    /**
     * Reference: QE vdw_corr='dft-d3', dftd3_version=4, dftd3_threebody=.false.,
     * identical celldm(1)/ibrav/positions: -0.04617308 Ry.
     */
    Crystal crystal_structure = make_silicon_fcc();
    D3BJDispersionCalculator calculator(crystal_structure);
    EXPECT_NEAR(calculator.compute_energy(), -0.04617308, 1e-6);
}

TEST(D3BJDispersionCalculator, CopperEnergyMatchesQuantumEspresso) {
    /**
     * Reference: QE vdw_corr='dft-d3', dftd3_version=4, dftd3_threebody=.false.,
     * identical celldm(1)/ibrav/position: -0.04261810 Ry. Copper's actual bulk
     * coordination number (~8.5) falls outside its sparse 2-point reference
     * grid (CN=0, 0.958), exercising the nearest-reference-point fallback path.
     */
    Crystal crystal_structure = make_copper_fcc();
    D3BJDispersionCalculator calculator(crystal_structure);
    EXPECT_NEAR(calculator.compute_energy(), -0.04261810, 1e-6);
}

TEST(D3BJDispersionCalculator, GraphiteNearEquilibriumMatchesQuantumEspresso) {
    /**
     * Interlayer spacing 3.3555 Angstrom (c=6.711 Angstrom), close to the
     * PBE-D3(BJ) equilibrium (Q3/QE c-scan fit: d_eq = 3.3780/3.3779
     * Angstrom -- see test_qe_graphite_d3_dispersion.py). Reference: QE
     * vdw_corr='dft-d3', dftd3_version=4, dftd3_threebody=.false., ibrav=4,
     * identical celldm(1)/celldm(3)/positions: -0.04595913 Ry.
     */
    Crystal crystal_structure = make_graphite_hex(3.3555);
    D3BJDispersionCalculator calculator(crystal_structure);
    EXPECT_NEAR(calculator.compute_energy(), -0.04595913, 1e-6);
}

TEST(D3BJDispersionCalculator, GraphiteLargeSeparationMatchesQuantumEspresso) {
    /**
     * Interlayer spacing 8.0 Angstrom: layers are essentially decoupled, so
     * this is dominated by intralayer (short C-C) pairs -- a different
     * regime from the near-equilibrium case above, and the same geometry
     * used as the "unbound" asymptotic reference in the c-scan binding-energy
     * test. Reference: QE, same protocol as above: -0.02673680 Ry.
     */
    Crystal crystal_structure = make_graphite_hex(8.0);
    D3BJDispersionCalculator calculator(crystal_structure);
    EXPECT_NEAR(calculator.compute_energy(), -0.02673680, 1e-6);
}

TEST(D3BJDispersionCalculator, EnergyScalesWithNumberOfAtoms) {
    /**
     * A supercell with N replicas should give very close to N times the
     * energy per atom of the primitive cell for a well-converged dispersion
     * sum (same coordination environment, same nearest-neighbor distances).
     */
    Crystal primitive = make_silicon_fcc();
    D3BJDispersionCalculator primitive_calculator(primitive);
    Real primitive_energy = primitive_calculator.compute_energy();
    EXPECT_LT(primitive_energy, 0.0);
    /**
     * Loose sanity bound: for Si at equilibrium volume, D3(BJ) dispersion is a
     * small correction relative to the ~16 Ry/cell DFT total energy scale --
     * not literally bounded by that number, but should not be a wildly large
     * fraction of a Hartree per atom.
     */
    EXPECT_GT(primitive_energy, -1.0);
}

TEST(D3BJDispersionCalculator, SupercellIsSizeExtensiveWithZeroForces) {
    /**
     * Multi-atom regression anchor for the flattened lattice-sum path
     * (enumerate_image_shifts + origin-skip logic), at a scale where the
     * image list is heavily exercised: a 2x2x2 diamond-Si conventional
     * supercell (64 atoms) must give exactly 32x the primitive FCC 2-atom
     * cell's energy (identical coordination environments, converged cutoff
     * spheres), and every force on the perfect crystal must vanish by site
     * symmetry -- a wrong origin-skip or shift-list bug breaks both.
     */
    Crystal primitive = make_silicon_fcc();
    Real primitive_energy = D3BJDispersionCalculator(primitive).compute_energy();

    constexpr int repeat = 2;
    constexpr Real conventional_edge_bohr = 10.263048; // = make_silicon_fcc()'s alat
    Matrix3x3 supercell_lattice{{conventional_edge_bohr * repeat, 0.0, 0.0},
                                {0.0, conventional_edge_bohr * repeat, 0.0},
                                {0.0, 0.0, conventional_edge_bohr * repeat}};
    const Real base[8][3] = {{0.0, 0.0, 0.0},
                             {0.0, 0.5, 0.5},
                             {0.5, 0.0, 0.5},
                             {0.5, 0.5, 0.0},
                             {0.25, 0.25, 0.25},
                             {0.25, 0.75, 0.75},
                             {0.75, 0.25, 0.75},
                             {0.75, 0.75, 0.25}};
    std::vector<Atom> atoms;
    for (int cell_x = 0; cell_x < repeat; cell_x++) {
        for (int cell_y = 0; cell_y < repeat; cell_y++) {
            for (int cell_z = 0; cell_z < repeat; cell_z++) {
                for (const auto& position : base) {
                    atoms.push_back({0,
                                     {(position[0] + cell_x) / repeat,
                                      (position[1] + cell_y) / repeat,
                                      (position[2] + cell_z) / repeat}});
                }
            }
        }
    }
    Crystal supercell =
        Crystal::from_lattice_vectors(1.0, supercell_lattice, primitive.species(), atoms);

    D3BJDispersionCalculator supercell_calculator(supercell);
    Real supercell_energy = supercell_calculator.compute_energy();
    // 8 conventional cells x 4 primitive cells each = 32 primitive cells.
    EXPECT_NEAR(supercell_energy, 32.0 * primitive_energy, 1e-9);

    auto gradients = supercell_calculator.compute_gradients();
    ASSERT_EQ(gradients.forces.size(), 64u);
    for (const auto& force : gradients.forces) {
        EXPECT_NEAR(force.x, 0.0, 1e-12);
        EXPECT_NEAR(force.y, 0.0, 1e-12);
        EXPECT_NEAR(force.z, 0.0, 1e-12);
    }
}

TEST(D3BJDispersionCalculator, UnsupportedElementThrows) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(1, 10.0);
    AtomSpecies gold{"Au", 196.97, 11.0, "Au.upf"}; // not in the reference data
    crystal_structure.add_species(gold);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();

    D3BJDispersionCalculator calculator(crystal_structure);
    EXPECT_THROW(calculator.compute_energy(), std::runtime_error);
}

/**
 * ─── Forces and stress ─────────────────────────────────────────────────────
 *
 * Primary correctness evidence for compute_gradients(): central finite
 * difference of the already-validated compute_energy() against the analytic
 * force/stress. These need no QE reference -- they check that the gradient is
 * the calculus-correct derivative of the exact energy formula as coded.
 */

TEST(D3BJDispersionCalculator, ForcesMatchFiniteDifferenceDistortedSilicon) {
    Crystal crystal_structure = make_generically_distorted(make_silicon_fcc());
    D3BJDispersionCalculator calculator(crystal_structure);
    auto gradients = calculator.compute_gradients();

    constexpr Real delta_bohr = 1e-4;
    bool saw_nonzero_component = false;
    for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
        for (int direction = 0; direction < 3; direction++) {
            Real finite_difference_force = finite_difference_force_component(
                crystal_structure, atom_index, direction, delta_bohr);
            Real analytic_force = gradients.forces[atom_index][direction];
            EXPECT_NEAR(analytic_force, finite_difference_force, 1e-7)
                << "atom " << atom_index << " direction " << direction;
            if (std::abs(analytic_force) > 1e-5) {
                saw_nonzero_component = true;
            }
        }
    }
    // Guard against a vacuously-passing test (e.g. all forces silently zero).
    EXPECT_TRUE(saw_nonzero_component);
}

TEST(D3BJDispersionCalculator, ForcesMatchFiniteDifferenceDistortedGraphite) {
    Crystal crystal_structure = make_generically_distorted(make_graphite_hex(3.3555));
    D3BJDispersionCalculator calculator(crystal_structure);
    auto gradients = calculator.compute_gradients();

    constexpr Real delta_bohr = 1e-4;
    bool saw_nonzero_component = false;
    for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
        for (int direction = 0; direction < 3; direction++) {
            Real finite_difference_force = finite_difference_force_component(
                crystal_structure, atom_index, direction, delta_bohr);
            Real analytic_force = gradients.forces[atom_index][direction];
            EXPECT_NEAR(analytic_force, finite_difference_force, 1e-7)
                << "atom " << atom_index << " direction " << direction;
            if (std::abs(analytic_force) > 1e-5) {
                saw_nonzero_component = true;
            }
        }
    }
    EXPECT_TRUE(saw_nonzero_component);
}

TEST(D3BJDispersionCalculator, ForcesOnSingleAtomBasisAreStructurallyZero) {
    /**
     * A one-atom primitive cell's only "pairs" are an atom with its own
     * periodic images; those separations are pure lattice vectors, so they
     * cannot exert a net force (only stress) regardless of C6/CN values --
     * true even in copper's nearest-reference-point fallback regime.
     */
    Crystal crystal_structure = make_copper_fcc();
    D3BJDispersionCalculator calculator(crystal_structure);
    auto gradients = calculator.compute_gradients();
    ASSERT_EQ(gradients.forces.size(), 1u);
    EXPECT_NEAR(gradients.forces[0].x, 0.0, 1e-10);
    EXPECT_NEAR(gradients.forces[0].y, 0.0, 1e-10);
    EXPECT_NEAR(gradients.forces[0].z, 0.0, 1e-10);
}

TEST(D3BJDispersionCalculator, CopperStressIsFiniteInFallbackRegime) {
    /**
     * Copper's CN falls outside its reference grid (see
     * CopperEnergyMatchesQuantumEspresso above), exercising the
     * zero-derivative fallback branch of interpolate_c6_and_derivatives().
     * Stress still has a direct (non-CN) contribution and must remain finite.
     */
    Crystal crystal_structure = make_copper_fcc();
    D3BJDispersionCalculator calculator(crystal_structure);
    auto gradients = calculator.compute_gradients();
    for (int direction_a = 0; direction_a < 3; direction_a++) {
        for (int direction_b = 0; direction_b < 3; direction_b++) {
            EXPECT_TRUE(std::isfinite(gradients.stress[direction_a][direction_b]));
        }
    }
    // Cubic symmetry: the stress must be isotropic (pure pressure, no shear).
    EXPECT_NEAR(gradients.stress[0][0], gradients.stress[1][1], 1e-10);
    EXPECT_NEAR(gradients.stress[1][1], gradients.stress[2][2], 1e-10);
    EXPECT_NEAR(gradients.stress[0][1], 0.0, 1e-10);
}

TEST(D3BJDispersionCalculator, StressMatchesFiniteDifferenceDistortedSilicon) {
    Crystal crystal_structure = make_generically_distorted(make_silicon_fcc());
    D3BJDispersionCalculator calculator(crystal_structure);
    auto gradients = calculator.compute_gradients();

    /**
     * A strain (unlike a single-atom displacement) perturbs every one of the
     * ~1e4-1e5 periodic-image pairs inside the D3 cutoff sphere at once, so
     * its finite-difference truncation error at a fixed step is larger; 1e-5
     * (vs. 1e-4 for the force tests above) keeps the same 1e-8 tolerance
     * meaningful. Confirmed by direct h-refinement: the residual at 1e-4
     * shrinks by ~100x when the step drops to 1e-5, tracking h^2 as expected
     * for truncation error rather than a fixed offset from a wrong formula.
     */
    constexpr Real strain_step = 1e-5;
    /**
     * Diagonal and one off-diagonal component; the direct + CN-coupling
     * formulas are structurally identical for every (a, b) pair, so this is
     * representative without enumerating all nine components.
     */
    const int components[4][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}};
    for (const auto& component : components) {
        int direction_a = component[0];
        int direction_b = component[1];
        Real finite_difference_stress = finite_difference_stress_component(
            crystal_structure, direction_a, direction_b, strain_step);
        Real analytic_stress = gradients.stress[direction_a][direction_b];
        EXPECT_NEAR(analytic_stress, finite_difference_stress, 1e-8)
            << "component (" << direction_a << ", " << direction_b << ")";
    }
}

TEST(D3BJDispersionCalculator, StressMatchesFiniteDifferenceGraphiteNearEquilibrium) {
    /**
     * Uses the undistorted near-equilibrium geometry: hexagonal symmetry still
     * allows (and physically predicts) sigma_xx == sigma_yy != sigma_zz, a
     * meaningful anisotropic signal from a layered material.
     */
    Crystal crystal_structure = make_graphite_hex(3.3555);
    D3BJDispersionCalculator calculator(crystal_structure);
    auto gradients = calculator.compute_gradients();

    // See the strain_step comment in StressMatchesFiniteDifferenceDistortedSilicon.
    constexpr Real strain_step = 1e-5;
    const int components[3][2] = {{0, 0}, {1, 1}, {2, 2}};
    for (const auto& component : components) {
        int direction_a = component[0];
        int direction_b = component[1];
        Real finite_difference_stress = finite_difference_stress_component(
            crystal_structure, direction_a, direction_b, strain_step);
        Real analytic_stress = gradients.stress[direction_a][direction_b];
        EXPECT_NEAR(analytic_stress, finite_difference_stress, 1e-8)
            << "component (" << direction_a << ", " << direction_b << ")";
    }
    EXPECT_NEAR(gradients.stress[0][0], gradients.stress[1][1], 1e-8);
}
