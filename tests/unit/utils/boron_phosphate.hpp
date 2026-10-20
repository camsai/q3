#pragma once

/**
 * BPO4 (Materials Project mp-3589) with its atoms written outside the home cell.
 *
 * The positions are the angstrom rows of the deck in camsai/q3#18, which the QE reader then
 * took as crystal coordinates: fractional coordinates as large as 3.8 in a body-centred
 * tetragonal cell. As a structure it is unphysical -- two O-O pairs sit 1.08 Å apart -- and
 * that is what makes it a sharp fixture for real-space lattice sums. At the report's cutoffs
 * the Ewald splitting parameter comes out at 2.7 and its real-space cutoff at 2.43 bohr, so
 * the only real-space terms are those close pairs, and the nearest image of one of them is
 * a translation of (2, -3, 0) away from the positions as written. A sum that loops over
 * images around the raw difference of two positions misses it.
 *
 * Shared because the Ewald energy, force and stress tests all need exactly this cell, and a
 * digit that drifted in one copy would decouple that file's numbers from the others'.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"

namespace q3::test_support {

/**
 * The Ewald energy of this structure at these cutoffs: -79.866771929 Ry, whichever cell each
 * atom is written in. An independent Ewald sum converged in its splitting parameter gives
 * -79.866771891 Ry, inside the 1e-7 Ry G-space truncation bound the parameter is chosen for.
 */
inline constexpr Real boron_phosphate_ewald_energy_rydberg = -79.866771929;

/// B, P, O, O, O, O at the crystal coordinates the misread deck produced.
inline std::vector<Atom> boron_phosphate_atoms_as_written() {
    // Species indices follow boron_phosphate_crystal(): 0 = B, 1 = O, 2 = P.
    return {
        {0, {0.0000001025, 2.2027939847, 1.6759652700}},
        {2, {0.0000000000, 0.0000000000, 0.0000000000}},
        {1, {-1.0624485731, 1.6057891884, 2.5013468678}},
        {1, {0.5970048232, 1.1403365814, 0.8505836715}},
        {1, {1.0624573122, -1.6057849961, 2.5013468646}},
        {1, {3.8085792206, 3.2652426136, 0.8505836745}},
    };
}

/// The same atoms with every crystal coordinate wrapped into [0, 1).
inline std::vector<Atom> wrapped_into_home_cell(std::vector<Atom> atoms) {
    for (Atom& atom : atoms)
        for (int axis = 0; axis < 3; axis++)
            atom.position[axis] -= std::floor(atom.position[axis]);
    return atoms;
}

/// The same atoms, each moved by a different whole lattice vector.
inline std::vector<Atom> shifted_by_lattice_vectors(std::vector<Atom> atoms) {
    const std::vector<Vector3D> translations = {
        {-3.0, 1.0, 2.0},
        {4.0, -2.0, 0.0},
        {1.0, 5.0, -4.0},
        {-2.0, -3.0, 3.0},
        {0.0, 2.0, -5.0},
        {5.0, 0.0, 1.0},
    };
    for (size_t atom = 0; atom < atoms.size(); atom++)
        atoms[atom].position = atoms[atom].position + translations[atom % translations.size()];
    return atoms;
}

/// The mp-3589 cell, given in angstrom as in the report, holding `atoms`.
inline Crystal boron_phosphate_crystal(const std::vector<Atom>& atoms) {
    const Real rows_angstrom[3][3] = {
        {4.4055882500, -0.0000000040, 0.0000000010},
        {0.0000002830, 4.4055881420, 0.0000000030},
        {-2.2027943440, -2.2027942410, 3.3519305350},
    };
    const Real first_row_length_angstrom = std::sqrt(rows_angstrom[0][0] * rows_angstrom[0][0] +
                                                     rows_angstrom[0][1] * rows_angstrom[0][1] +
                                                     rows_angstrom[0][2] * rows_angstrom[0][2]);
    const Real lattice_parameter = first_row_length_angstrom * constants::ANGSTROM_TO_BOHR;

    Matrix3x3 lattice_vectors;
    for (int row = 0; row < 3; row++)
        lattice_vectors[row] =
            Vector3D(rows_angstrom[row][0], rows_angstrom[row][1], rows_angstrom[row][2]) *
            (1.0 / first_row_length_angstrom);

    // SG15 valence charges, the ones the report's pseudopotentials carry.
    const std::vector<AtomSpecies> species = {
        {"B", 10.81, 3.0, "B.upf"},
        {"O", 15.999, 6.0, "O.upf"},
        {"P", 30.97376, 5.0, "P.upf"},
    };
    return Crystal::from_lattice_vectors(lattice_parameter, lattice_vectors, species, atoms);
}

/**
 * G vectors for the report's cutoffs, ecutwfc = 60 Ry and ecutrho = 240 Ry, which are what
 * select the splitting parameter 2.7. Estimated FFT plans: the Ewald sums never transform.
 */
inline void generate_boron_phosphate_basis(PlanewaveBasis& basis, const Crystal& crystal) {
    static execution::PlanewaveSerialExecutionPolicy serial_policy;
    basis.generate(serial_policy, crystal, 60.0, 240.0, false, false);
}

} // namespace q3::test_support
