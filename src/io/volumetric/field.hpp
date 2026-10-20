#pragma once

/**
 * VolumetricField — one scalar field sampled on a run's real-space grid,
 * together with the cell and the atoms it belongs to.
 *
 * The seam between "where the numbers came from" and "which file format they
 * are written as". Every source in `io/` that can put a field on the dense FFT
 * grid — the charge density, the effective potential, a band's probability
 * density — fills one of these, and every writer under `io/volumetric/` reads
 * one. A new format is then a writer and nothing else, and a new field is a
 * filler and nothing else.
 *
 * **Grid ordering.** `values` is indexed `(i1 * n2 + i2) * n3 + i3`, which is
 * how the engine stores every real-space array (both
 * `FFTGrid::g_vector_to_fft_grid_index` and `FFTGrid::global_real_space_index`
 * lay a grid out this way): the **third** lattice direction varies fastest.
 * Point `(i1, i2, i3)` sits at `r = i1/n1 · a1 + i2/n2 · a2 + i3/n3 · a3`, the
 * grid is periodic, and the boundary point is stored once — a format that
 * wants it duplicated (XSF does) wraps the index itself.
 *
 * **Units.** Lengths are bohr; the field's own unit is whatever `unit` names,
 * which is electrons/bohr^3 for a density and Rydberg for a potential. Formats
 * wanting Ångström, or a volume-weighted density (some want both), convert
 * inside their own writer, so exactly one place per format knows its
 * conventions.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/periodic_table.hpp"

namespace q3 {

struct VolumetricField {
    /// Grid point counts along a1, a2, a3.
    std::array<int, 3> grid_dimensions = {0, 0, 0};
    /// Rows are a1, a2, a3, Cartesian, in bohr.
    Matrix3x3 lattice_vectors_bohr{};

    /// Element symbols, one per species.
    std::vector<std::string> species_symbols;
    /// Valence charge of each species, parallel to `species_symbols`.
    std::vector<Real> species_valence_charges;
    /// Species of each atom, an index into `species_symbols`.
    std::vector<int> atom_species_indices;
    /// Fractional (crystal) coordinates, parallel to `atom_species_indices`.
    std::vector<Vector3D> atom_positions_crystal;

    /// The samples themselves, `number_of_grid_points()` of them.
    std::vector<Real> values;
    /// Free text for a format's comment line, e.g. "charge density".
    std::string description;
    /// The unit `values` is in, e.g. "e/bohr^3".
    std::string unit;
    /**
     * Whether `integral()` is a particle count — true for the charge density
     * and for a band's probability density, false for a potential.
     *
     * Formats disagree about what they hold, and the disagreement is silent:
     * one format stores the density multiplied by the cell volume, so that
     * summing it and dividing by the point count gives electrons, while a
     * potential in the same file shape is stored unscaled. A writer that
     * cannot honour the distinction refuses the field rather than applying the
     * wrong one.
     */
    bool is_a_density = false;

    [[nodiscard]] int number_of_grid_points() const {
        return grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2];
    }

    [[nodiscard]] int number_of_atoms() const {
        return static_cast<int>(atom_species_indices.size());
    }

    [[nodiscard]] Real cell_volume_bohr3() const {
        return std::abs(lattice_vectors_bohr.determinant());
    }

    /// The sample at grid point (i1, i2, i3), third index fastest.
    [[nodiscard]] Real value(int index_1, int index_2, int index_3) const {
        const std::size_t flat_index =
            (static_cast<std::size_t>(index_1) * grid_dimensions[1] + index_2) *
                grid_dimensions[2] +
            index_3;
        return values[flat_index];
    }

    /**
     * The field integrated over the cell by the midpoint rule the grid is a
     * sample of: `Omega / N * sum(values)`.
     *
     * For a charge density this is the electron count, which is the one number
     * that says whether a written file is right — a transposed grid or a
     * missed unit changes it, and a viewer's isosurface does not.
     */
    [[nodiscard]] Real integral() const {
        Real sum = 0.0;
        for (Real sample : values)
            sum += sample;
        return sum * cell_volume_bohr3() / static_cast<Real>(number_of_grid_points());
    }

    /// Cartesian position of one atom, in bohr.
    [[nodiscard]] Vector3D cartesian_position_bohr(int atom_index) const {
        const Vector3D& fractional = atom_positions_crystal[atom_index];
        Vector3D position;
        for (int dimension = 0; dimension < 3; dimension++) {
            position[dimension] = fractional.x * lattice_vectors_bohr[0][dimension] +
                                  fractional.y * lattice_vectors_bohr[1][dimension] +
                                  fractional.z * lattice_vectors_bohr[2][dimension];
        }
        return position;
    }

    /// Atomic number of one atom, from its element symbol.
    [[nodiscard]] int atomic_number(int atom_index) const {
        return PeriodicTable::atomic_number_by_element_symbol(
            species_symbols[atom_species_indices[atom_index]]);
    }

    /// How many atoms each species has, in species order, for formats that group atoms so.
    [[nodiscard]] std::vector<int> atoms_per_species() const {
        std::vector<int> counts(species_symbols.size(), 0);
        for (int species_index : atom_species_indices) {
            if (species_index >= 0 && species_index < static_cast<int>(counts.size()))
                counts[species_index]++;
        }
        return counts;
    }
};

} // namespace q3
