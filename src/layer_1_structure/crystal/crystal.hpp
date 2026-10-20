#pragma once

/**
 * Crystal: crystal structure container with lattice vectors, atomic species,
 * atom positions, and derived reciprocal-space quantities.
 *
 * Holds the Bravais lattice, computes reciprocal lattice vectors and cell volume,
 * and provides coordinate transformations between fractional and Cartesian frames.
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/atom.hpp"
#include "layer_1_structure/crystal/atom_species.hpp"
#include "layer_1_structure/crystal/lattice.hpp"
#include "layer_1_structure/crystal/periodic_table.hpp"

namespace q3 {

/**
 * Crystal structure container.
 *
 * Holds lattice vectors, atomic species, atom positions, and derived
 * quantities (reciprocal lattice, cell volume, etc.).
 */
class Crystal {
  public:
    Crystal() = default;

    /**
     * Construct a fully initialized crystal from raw input parameters.
     *
     * Builds the lattice from Bravais index, populates species and atoms,
     * and computes all derived quantities (reciprocal lattice, volume, etc.).
     */
    static Crystal from_input_data(int bravais_lattice_index,
                                   Real cell_dimension_1,
                                   const std::vector<AtomSpecies>& species,
                                   const std::vector<Atom>& atoms,
                                   Real cell_dimension_2 = 0,
                                   Real cell_dimension_3 = 0,
                                   Real cell_dimension_4 = 0,
                                   Real cell_dimension_5 = 0,
                                   Real cell_dimension_6 = 0) {
        Crystal crystal;
        crystal.set_from_lattice_type(bravais_lattice_index,
                                      cell_dimension_1,
                                      cell_dimension_2,
                                      cell_dimension_3,
                                      cell_dimension_4,
                                      cell_dimension_5,
                                      cell_dimension_6);
        crystal.bravais_lattice_index_ = bravais_lattice_index;
        for (const auto& species_entry : species) {
            crystal.add_species(species_entry);
        }
        for (const auto& atom_entry : atoms) {
            crystal.add_atom(atom_entry);
        }
        crystal.compute_derived_quantities();
        return crystal;
    }

    /**
     * Construct a fully initialized crystal from explicit lattice vectors
     * ("free" lattice, Bravais index 0).
     *
     * @param lattice_parameter  alat in bohr.
     * @param lattice_vectors    rows a1,a2,a3 in units of alat.
     */
    static Crystal from_lattice_vectors(Real lattice_parameter,
                                        const Matrix3x3& lattice_vectors,
                                        const std::vector<AtomSpecies>& species,
                                        const std::vector<Atom>& atoms) {
        Crystal crystal;
        crystal.lattice_parameter_ = lattice_parameter;
        crystal.set_lattice(lattice_vectors);
        for (const auto& species_entry : species) {
            crystal.add_species(species_entry);
        }
        for (const auto& atom_entry : atoms) {
            crystal.add_atom(atom_entry);
        }
        crystal.compute_derived_quantities();
        return crystal;
    }

    /**
     * Build lattice from Bravais index convention.
     * Supported bravais_lattice_index values: 0, 1 (SC), 2 (FCC), 3/-3 (BCC), 4 (hex), 5/-5 (trig
     * R), 6 (tet), 7 (bct), 8 (ort), 9 (bco), 10 (fco), 11 (bort), 12/-12 (mono), 14 (tri).
     */
    void set_from_lattice_type(int bravais_lattice_index, Real cell_dimension_1) {
        set_from_lattice_type(bravais_lattice_index, cell_dimension_1, 0, 0, 0, 0, 0);
    }
    void set_from_lattice_type(int bravais_lattice_index,
                               Real cell_dimension_1,
                               Real cell_dimension_2,
                               Real cell_dimension_3,
                               Real cell_dimension_4,
                               Real cell_dimension_5,
                               Real cell_dimension_6) {
        lattice_parameter_ = cell_dimension_1;
        lattice_vectors_ = get_lattice_vectors_from_bravais_index(bravais_lattice_index,
                                                                  cell_dimension_1,
                                                                  cell_dimension_2,
                                                                  cell_dimension_3,
                                                                  cell_dimension_4,
                                                                  cell_dimension_5,
                                                                  cell_dimension_6);
    }

    /**
     * Set lattice vectors directly (in units of alat).
     */
    void set_lattice(const Matrix3x3& lattice_vectors) {
        lattice_vectors_ = lattice_vectors;
    }

    /**
     * Add an atomic species.
     */
    void add_species(const AtomSpecies& species) {
        species_.push_back(species);
    }

    /**
     * Add an atom (position in crystal coords). Any installed site symmetry
     * partition no longer covers every atom, so it is reset to the default.
     */
    void add_atom(const Atom& atom) {
        atoms_.push_back(atom);
        site_symmetry_classes_.clear();
    }

    /**
     * Update the positions of all atoms (in crystal/fractional coordinates).
     */
    void set_atomic_positions(const std::vector<Vector3D>& positions) {
        for (std::size_t atom_index = 0;
             atom_index < positions.size() && atom_index < atoms_.size();
             ++atom_index) {
            atoms_[atom_index].position = positions[atom_index];
        }
    }

    /**
     * Compute derived quantities from lattice vectors and alat:
     * reciprocal lattice vectors, cell volume, two_pi_over_lattice_parameter,
     * two_pi_over_lattice_parameter_squared. Must be called after setting lattice vectors and alat.
     */
    void compute_derived_quantities() {
        using namespace constants;

        two_pi_over_lattice_parameter_ = two_pi / lattice_parameter_;
        two_pi_over_lattice_parameter_squared_ =
            two_pi_over_lattice_parameter_ * two_pi_over_lattice_parameter_;

        // Cell volume = |a1 · (a2 × a3)| * alat^3

        // lattice_vectors_ is in units of alat, so its determinant gives V/alat^3.
        Real volume_fraction = lattice_vectors_.determinant();
        cell_volume_ = std::abs(volume_fraction) * lattice_parameter_ * lattice_parameter_ *
                       lattice_parameter_;

        // Reciprocal lattice: bg = (at^{-1})^T

        // where at has rows = lattice vectors.

        // This gives bg such that bg[i] · at[j] = delta_ij.
        Matrix3x3 lattice_inverse = lattice_vectors_.inverse();
        reciprocal_lattice_vectors_ = lattice_inverse.transpose();
    }

    // --- Accessors ---
    [[nodiscard]] Real lattice_parameter() const {
        return lattice_parameter_;
    }
    /**
     * QE ibrav convention this crystal was built from (0 for "free"/explicit lattice
     * vectors, i.e. Crystal::from_lattice_vectors() -- see set_from_lattice_type()'s
     * docstring for the full ibrav -> Bravais-type mapping). Consumers that need a
     * specific lattice symmetry (e.g. isolated_system_correction/'s Makov-Payne
     * Madelung-constant lookup, cubic-only) must check this explicitly.
     */
    [[nodiscard]] int bravais_lattice_index() const {
        return bravais_lattice_index_;
    }
    [[nodiscard]] Real cell_volume() const {
        return cell_volume_;
    }
    [[nodiscard]] Real two_pi_over_lattice_parameter() const {
        return two_pi_over_lattice_parameter_;
    }
    [[nodiscard]] Real two_pi_over_lattice_parameter_squared() const {
        return two_pi_over_lattice_parameter_squared_;
    }
    [[nodiscard]] const Matrix3x3& lattice_vectors() const {
        return lattice_vectors_;
    }
    [[nodiscard]] const Matrix3x3& reciprocal_lattice_vectors() const {
        return reciprocal_lattice_vectors_;
    }

    [[nodiscard]] int number_of_atoms() const {
        return static_cast<int>(atoms_.size());
    }
    [[nodiscard]] int number_of_species() const {
        return static_cast<int>(species_.size());
    }
    [[nodiscard]] const std::vector<Atom>& atoms() const {
        return atoms_;
    }
    [[nodiscard]] const std::vector<AtomSpecies>& species() const {
        return species_;
    }

    /**
     * Cartesian position of atom i (in units of alat).
     */
    [[nodiscard]] Vector3D convert_from_crystal_to_cartesian(int atom_index) const {
        /**
         * Convert from crystal coords to Cartesian (in units of alat)
         * r_cart = tau[0]*a1 + tau[1]*a2 + tau[2]*a3
         */
        const Vector3D& fractional = atoms_[atom_index].position;
        Vector3D cartesian_position;
        for (int dimension = 0; dimension < 3; dimension++) {
            cartesian_position[dimension] = fractional.x * lattice_vectors_[0][dimension] +
                                            fractional.y * lattice_vectors_[1][dimension] +
                                            fractional.z * lattice_vectors_[2][dimension];
        }
        return cartesian_position;
    }

    /**
     * The separation r_a − r_b of two atoms (Cartesian, units of alat), moved by a lattice
     * vector so that each of its crystal coordinates lies in [−1/2, 1/2].
     *
     * Positions are kept as given, and crystal coordinates outside [0, 1) are legal input, so
     * the plain difference of two positions can lie several cells away. A real-space lattice
     * sum that visits the images within a distance R by running ±ceil(R / h_k) cells along
     * each axis around this separation -- h_k the spacing of the lattice planes parallel to
     * the other two vectors -- reaches all of them: an image's crystal coordinate along k is
     * at most R / h_k in magnitude, and this separation's own is at most 1/2. Around the plain
     * difference the same loop can miss even the nearest image; the Ewald sums did, for an
     * O-O pair 1.08 Å apart in camsai/q3#18.
     */
    [[nodiscard]] Vector3D home_cell_separation(int atom_index_a, int atom_index_b) const {
        Vector3D fractional = atoms_[atom_index_a].position - atoms_[atom_index_b].position;
        for (int axis = 0; axis < 3; axis++)
            fractional[axis] -= std::round(fractional[axis]);
        Vector3D cartesian_separation;
        for (int dimension = 0; dimension < 3; dimension++) {
            cartesian_separation[dimension] = fractional.x * lattice_vectors_[0][dimension] +
                                              fractional.y * lattice_vectors_[1][dimension] +
                                              fractional.z * lattice_vectors_[2][dimension];
        }
        return cartesian_separation;
    }

    /**
     * Fractional (crystal) position of atom i.
     */
    [[nodiscard]] Vector3D fractional_position(int atom_index) const {
        return atoms_[atom_index].position;
    }

    [[nodiscard]] int species_type_index(int atom_index) const {
        return atoms_[atom_index].species_index;
    }

    /**
     * Symmetry class of atom i: two atoms may be mapped onto each other by a
     * symmetry operation only if their classes agree. By default the class is
     * the species index. A finer partition (set_site_symmetry_classes) lets a
     * caller distinguish atoms of one species that a higher layer treats as
     * inequivalent, e.g. sites seeded with opposite starting moments.
     */
    [[nodiscard]] int site_symmetry_class(int atom_index) const {
        return site_symmetry_classes_.empty() ? atoms_[atom_index].species_index
                                              : site_symmetry_classes_[atom_index];
    }

    /**
     * Install a per-atom symmetry partition (one non-negative integer per atom;
     * equal integers = interchangeable sites). It must refine the species
     * partition: atoms of different species never share a class. An empty
     * vector restores the default (class = species index).
     */
    void set_site_symmetry_classes(const std::vector<int>& classes) {
        if (!classes.empty()) {
            if (classes.size() != atoms_.size())
                throw std::runtime_error("site symmetry classes: expected " +
                                         std::to_string(atoms_.size()) + " entries, got " +
                                         std::to_string(classes.size()));
            for (size_t first = 0; first < atoms_.size(); ++first) {
                if (classes[first] < 0)
                    throw std::runtime_error("site symmetry classes must be non-negative");
                for (size_t second = first + 1; second < atoms_.size(); ++second)
                    if (classes[first] == classes[second] &&
                        atoms_[first].species_index != atoms_[second].species_index)
                        throw std::runtime_error(
                            "site symmetry classes must not merge atoms of different species");
            }
        }
        site_symmetry_classes_ = classes;
    }

    [[nodiscard]] bool has_site_symmetry_classes() const {
        return !site_symmetry_classes_.empty();
    }
    [[nodiscard]] Real valence_charge(int species_index) const {
        return species_[species_index].valence_charge;
    }

    /**
     * Lattice vectors in Bohr (Cartesian).
     */
    [[nodiscard]] Matrix3x3 lattice_vectors_bohr() const {
        return lattice_vectors_ * lattice_parameter_;
    }

    /**
     * Reciprocal lattice vectors in 1/Bohr.
     */
    [[nodiscard]] Matrix3x3 reciprocal_vectors_bohr() const {
        return reciprocal_lattice_vectors_ * two_pi_over_lattice_parameter_;
    }

    /**
     * Atomic number (Z) for atom i. Inferred from element symbol.
     */
    [[nodiscard]] int atomic_number(int atom_index) const {
        const std::string& symbol = species_[atoms_[atom_index].species_index].symbol;
        return PeriodicTable::atomic_number_by_element_symbol(symbol);
    }

    /**
     * Get element symbol for atom i.
     */
    [[nodiscard]] const std::string& element(int atom_index) const {
        return species_[atoms_[atom_index].species_index].symbol;
    }

  private:
    Real lattice_parameter_ = 0;
    int bravais_lattice_index_ = 0;
    Real cell_volume_ = 0;
    Real two_pi_over_lattice_parameter_ = 0;
    Real two_pi_over_lattice_parameter_squared_ = 0;

    Matrix3x3 lattice_vectors_;            // lattice vectors in units of alat (rows = a1,a2,a3)
    Matrix3x3 reciprocal_lattice_vectors_; // reciprocal lattice vectors in units of 2π/alat

    std::vector<AtomSpecies> species_;
    std::vector<Atom> atoms_;
    std::vector<int> site_symmetry_classes_; // empty = species index
};

} // namespace q3
