#pragma once

/**
 * Crystal symmetry via spglib.
 *
 * Provides spacegroup detection, symmetry operations, and IBZ k-point reduction.
 */

#include <cmath>
#include <spglib.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/operation.hpp"

namespace q3 {

namespace detail {
// Convert Crystal to spglib cell format
inline void crystal_to_spglib(const Crystal& crystal,
                              double lattice[3][3],
                              std::vector<double>& positions,
                              std::vector<int>& types) {
    Real lattice_parameter = crystal.lattice_parameter();
    const Matrix3x3& direct_lattice_vectors = crystal.lattice_vectors();

    // spglib uses column-major lattice[cartesian_coordinate][basis_vector_index]

    // Our direct_lattice_vectors are stored as rows:

    // direct_lattice_vectors[basis_vector_index][cartesian_coordinate] Therefore, we transpose the

    // matrix to pass it to spglib.
    for (int coordinate_index = 0; coordinate_index < 3; coordinate_index++) {
        for (int basis_index = 0; basis_index < 3; basis_index++) {
            lattice[coordinate_index][basis_index] =
                direct_lattice_vectors[basis_index][coordinate_index] *
                lattice_parameter; // in Bohr
        }
    }

    int number_of_atoms = crystal.number_of_atoms();
    positions.resize(3 * number_of_atoms);
    types.resize(number_of_atoms);
    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        Vector3D fractional_position = crystal.fractional_position(atom_index);
        positions[3 * atom_index] = fractional_position.x;
        positions[3 * atom_index + 1] = fractional_position.y;
        positions[3 * atom_index + 2] = fractional_position.z;
        types[atom_index] = crystal.site_symmetry_class(atom_index) + 1; // spglib: 1-based
    }
}
} // namespace detail

/// Symmetry analysis and IBZ k-point reduction using spglib.
class SymmetryAnalyzer {
  public:
    /**
     * Analyze symmetry of a crystal structure.
     * Returns the international spacegroup symbol.
     */
    static std::string
    find_spacegroup(const Crystal& crystal,
                    Real symmetry_precision = math::tolerances::default_symmetry_precision) {
        double lattice[3][3];
        std::vector<double> positions;
        std::vector<int> types;
        detail::crystal_to_spglib(crystal, lattice, positions, types);

        SpglibDataset* dataset = spg_get_dataset(lattice,
                                                 reinterpret_cast<double(*)[3]>(positions.data()),
                                                 types.data(),
                                                 crystal.number_of_atoms(),
                                                 symmetry_precision);

        if (!dataset) {
            log::warn("spglib: failed to determine spacegroup");
            return "Unknown";
        }

        std::string symbol(dataset->international_symbol);
        int spacegroup_number = dataset->spacegroup_number;
        int number_of_operations = dataset->n_operations;

        spg_free_dataset(dataset);

        log::info("     Spacegroup: {} (#{}, {} symmetry ops)",
                  symbol,
                  spacegroup_number,
                  number_of_operations);
        return symbol;
    }

    /// Get the number of symmetry operations.
    static int number_of_symmetry_operations(
        const Crystal& crystal,
        Real symmetry_precision = math::tolerances::default_symmetry_precision) {
        double lattice[3][3];
        std::vector<double> positions;
        std::vector<int> types;
        detail::crystal_to_spglib(crystal, lattice, positions, types);

        SpglibDataset* dataset = spg_get_dataset(lattice,
                                                 reinterpret_cast<double(*)[3]>(positions.data()),
                                                 types.data(),
                                                 crystal.number_of_atoms(),
                                                 symmetry_precision);

        if (!dataset)
            return 1;
        int number_of_operations = dataset->n_operations;
        spg_free_dataset(dataset);
        return number_of_operations;
    }

    /**
     * Reduce a uniform k-grid to the irreducible Brillouin zone.
     * Returns a KPointsGrid with reduced k-points and correct weights.
     */
    static KPointsGrid
    reduce_kpoints(const Crystal& crystal,
                   const KPointsGrid& input_grid,
                   Real symmetry_precision = math::tolerances::default_symmetry_precision) {
        const auto& dimensions = input_grid.grid_dimensions();
        const auto& shifts = input_grid.grid_shifts();
        int grid_dimension_x = dimensions[0];
        int grid_dimension_y = dimensions[1];
        int grid_dimension_z = dimensions[2];
        int shift_x = shifts[0];
        int shift_y = shifts[1];
        int shift_z = shifts[2];

        double lattice[3][3];
        std::vector<double> positions;
        std::vector<int> types;
        detail::crystal_to_spglib(crystal, lattice, positions, types);

        int mesh[3] = {grid_dimension_x, grid_dimension_y, grid_dimension_z};
        int is_shift[3] = {shift_x, shift_y, shift_z};
        int is_time_reversal = 1;

        int total_grid_points = grid_dimension_x * grid_dimension_y * grid_dimension_z;
        std::vector<int> mapping(total_grid_points);
        std::vector<int> grid_address_flat(total_grid_points * 3);
        auto* grid_address = reinterpret_cast<int(*)[3]>(grid_address_flat.data());

        int number_of_operations = number_of_symmetry_operations(crystal, symmetry_precision);

        int number_of_irreducible_kpoints =
            spg_get_ir_reciprocal_mesh(grid_address,
                                       mapping.data(),
                                       mesh,
                                       is_shift,
                                       is_time_reversal,
                                       lattice,
                                       reinterpret_cast<double(*)[3]>(positions.data()),
                                       types.data(),
                                       crystal.number_of_atoms(),
                                       symmetry_precision);

        if (number_of_irreducible_kpoints <= 0) {
            log::warn("spglib: reciprocal mesh generation failed, using full grid");
            KPointsGrid fallback_grid;
            fallback_grid.generate(
                grid_dimension_x, grid_dimension_y, grid_dimension_z, shift_x, shift_y, shift_z);
            return fallback_grid;
        }

        // Collect irreducible k-points and their weights
        std::vector<int> irreducible_indices;
        std::vector<int> weights;
        for (int grid_index = 0; grid_index < total_grid_points; grid_index++) {
            if (mapping[grid_index] == grid_index) {
                irreducible_indices.push_back(grid_index);
                weights.push_back(0);
            }
        }

        for (int grid_index = 0; grid_index < total_grid_points; grid_index++) {
            int representative = mapping[grid_index];
            for (size_t irreducible_index = 0; irreducible_index < irreducible_indices.size();
                 irreducible_index++) {
                if (irreducible_indices[irreducible_index] == representative) {
                    weights[irreducible_index]++;
                    break;
                }
            }
        }

        // Build reduced KPointsGrid
        KPointsGrid kpoints;
        kpoints.set_grid_metadata(dimensions, shifts);
        for (int irreducible_index = 0; irreducible_index < number_of_irreducible_kpoints;
             irreducible_index++) {
            int grid_index = irreducible_indices[irreducible_index];
            KPoint kpoint_data;
            kpoint_data.coordinates.x =
                static_cast<Real>(grid_address[grid_index][0] + is_shift[0] / 2.0) / mesh[0];
            kpoint_data.coordinates.y =
                static_cast<Real>(grid_address[grid_index][1] + is_shift[1] / 2.0) / mesh[1];
            kpoint_data.coordinates.z =
                static_cast<Real>(grid_address[grid_index][2] + is_shift[2] / 2.0) / mesh[2];

            // Fold back to first BZ
            for (int coordinate_index = 0; coordinate_index < 3; coordinate_index++) {
                kpoint_data.coordinates[coordinate_index] -=
                    std::floor(kpoint_data.coordinates[coordinate_index] + 0.5 +
                               math::tolerances::zero_vector_magnitude_threshold);
            }

            kpoint_data.weight = static_cast<Real>(weights[irreducible_index]) / total_grid_points *
                                 2.0; // factor 2 for spin
            kpoints.add_kpoint(kpoint_data);
        }

        log::info("     IBZ reduction: {} -> {} k-points ({} symmetry ops)",
                  total_grid_points,
                  number_of_irreducible_kpoints,
                  number_of_operations);
        kpoints.pretty_print();

        return kpoints;
    }

    /**
     *
     * Reduction rule: iterate k-points in order, and for each unvisited k-point, apply
     * all symmetry operations (+ time reversal) to find and mark equivalent k-points
     * with higher indices. The representative for each equivalence class is the
     * lowest-index member. Note this reduces in a single pass with the crystal's own
     * point group; pw.x instead reduces with the Bravais lattice group and restores the
     * k-points the crystal group does not relate in a second pass, so the two are not
     * the same rule even though the inner loop is.
     *
     * Unlike reduce_kpoints() which delegates to spg_get_ir_reciprocal_mesh(), this method
     * uses Q3's own symmetry operations (from spglib's spg_get_dataset, via
     * get_symmetry_operations()) and performs the equivalence checking internally, ensuring
     * identical IBZ representatives.
     *
     * Uses the actual (atomic-basis-aware) crystal point group, not the Bravais lattice's
     * holohedral group: for any crystal whose basis reduces the symmetry below the lattice's
     * (e.g. a finite-difference-displaced structure, or any multi-atom basis at a
     * less-than-maximal-symmetry arrangement), the lattice's holohedral rotations include
     * operations that are not actual symmetries of the decorated crystal -- treating k-points
     * related only by one of those extra operations as equivalent silently corrupts the
     * k-point sum (see TODOs.md's "IBZ k-point vector reconstruction" writeup). Reuses
     * reciprocal_rotation (= (R^{-1})^T), the same convention already used for G-vector
     * transformation in the density symmetrizer, since k-vectors are reciprocal-space
     * quantities, not real-space positions -- crystallographic rotation matrices in a
     * (generally oblique) fractional basis are not self-inverse-transpose.
     */
    static KPointsGrid reduce_kpoints_internal(
        const Crystal& crystal,
        const KPointsGrid& input_grid,
        bool use_time_reversal = true,
        Real symmetry_precision = math::tolerances::default_symmetry_precision) {
        const auto& dimensions = input_grid.grid_dimensions();
        const auto& shifts = input_grid.grid_shifts();
        int grid_dimension_1 = dimensions[0];
        int grid_dimension_2 = dimensions[1];
        int grid_dimension_3 = dimensions[2];
        int shift_1 = shifts[0];
        int shift_2 = shifts[1];
        int shift_3 = shifts[2];

        int total_grid_points = grid_dimension_1 * grid_dimension_2 * grid_dimension_3;

        /**
         * The crystal's actual (basis-aware) point group -- NOT
         * compute_bravais_lattice_rotations(), which only looks at lattice vectors and ignores
         * whether the atomic decoration actually respects each candidate rotation.
         */
        auto symmetry_operations = get_symmetry_operations(crystal, symmetry_precision);
        int number_of_operations = static_cast<int>(symmetry_operations.size());

        /**
         * The mesh is the one the caller already generated, read rather than
         * rebuilt. Reconstructing it here meant a second construction that had
         * to agree with KPointsGrid::generate() on ordering, on the shift
         * convention and on zone folding, with nothing checking that it did --
         * and the two did not fold identically: this one left coordinates in
         * [0,1) and folded only on the way out, while the grid folds as it
         * generates. Reading the list removes the question.
         *
         * The numbering is internal either way. It decides only which member of
         * each equivalence class becomes the representative; the spglib path
         * (reduce_kpoints, selected by ibz_reduction_method) numbers the same
         * mesh in the opposite order and travels through the same KPointsGrid,
         * results.json, checkpoint and BerkeleyGW writers.
         */
        const std::vector<KPoint>& mesh_kpoints = input_grid.kpoints();
        if (static_cast<int>(mesh_kpoints.size()) != total_grid_points) {
            throw std::runtime_error("reduce_kpoints_internal: the input grid holds " +
                                     std::to_string(mesh_kpoints.size()) +
                                     " k-points but its dimensions imply " +
                                     std::to_string(total_grid_points) +
                                     ". Call KPointsGrid::generate() before reducing.");
        }

        const auto crystal_component = [&mesh_kpoints](int flat_index, int dimension) -> double {
            const Vector3D& coordinates = mesh_kpoints[flat_index].coordinates;
            return dimension == 0 ? coordinates.x
                                  : (dimension == 1 ? coordinates.y : coordinates.z);
        };

        /**
         * has_representative[i] records whether k-point i has already been
         * folded onto an earlier one. Asking that question directly, rather
         * than encoding it as "the map points at itself", means reading the
         * bookkeeping never requires knowing the convention first.
         */
        std::vector<bool> has_representative(total_grid_points, false);
        std::vector<double> weight_accumulator(total_grid_points, 0.0);

        constexpr double grid_tolerance = 1.0e-5;

        for (int kpoint_index = 0; kpoint_index < total_grid_points; kpoint_index++) {
            if (has_representative[kpoint_index]) {
                continue; // already folded onto a lower-index representative
            }
            weight_accumulator[kpoint_index] = 1.0;

            for (int operation_index = 0; operation_index < number_of_operations;
                 operation_index++) {
                /**
                 * k-vectors are reciprocal-space quantities: use reciprocal_rotation
                 * (= (R^{-1})^T), not rotation (the real-space/position convention) --
                 * crystallographic rotation matrices are not self-inverse-transpose in a
                 * generally oblique fractional basis (e.g. FCC's rhombohedral primitive cell).
                 */
                const auto& rotation = symmetry_operations[operation_index].reciprocal_rotation;

                // Apply rotation in crystal coordinates: k' = R_reciprocal * k
                double rotated_crystal[3];
                for (int dimension = 0; dimension < 3; dimension++) {
                    rotated_crystal[dimension] =
                        rotation[dimension][0] * crystal_component(kpoint_index, 0) +
                        rotation[dimension][1] * crystal_component(kpoint_index, 1) +
                        rotation[dimension][2] * crystal_component(kpoint_index, 2);
                    // Fold back to [0, 1) via nint
                    rotated_crystal[dimension] -= std::round(rotated_crystal[dimension]);
                }

                // Find which grid point this maps to
                double fractional_index_1 = rotated_crystal[0] * grid_dimension_1 - 0.5 * shift_1;
                double fractional_index_2 = rotated_crystal[1] * grid_dimension_2 - 0.5 * shift_2;
                double fractional_index_3 = rotated_crystal[2] * grid_dimension_3 - 0.5 * shift_3;

                bool is_on_grid =
                    std::abs(fractional_index_1 - std::round(fractional_index_1)) <=
                        grid_tolerance &&
                    std::abs(fractional_index_2 - std::round(fractional_index_2)) <=
                        grid_tolerance &&
                    std::abs(fractional_index_3 - std::round(fractional_index_3)) <= grid_tolerance;

                if (is_on_grid) {
                    int mapped_index_1 =
                        ((static_cast<int>(std::round(fractional_index_1 + 2 * grid_dimension_1))) %
                         grid_dimension_1);
                    int mapped_index_2 =
                        ((static_cast<int>(std::round(fractional_index_2 + 2 * grid_dimension_2))) %
                         grid_dimension_2);
                    int mapped_index_3 =
                        ((static_cast<int>(std::round(fractional_index_3 + 2 * grid_dimension_3))) %
                         grid_dimension_3);
                    int mapped_flat_index = mapped_index_3 + mapped_index_2 * grid_dimension_3 +
                                            mapped_index_1 * grid_dimension_2 * grid_dimension_3;

                    if (mapped_flat_index > kpoint_index &&
                        !has_representative[mapped_flat_index]) {
                        has_representative[mapped_flat_index] = true;
                        weight_accumulator[kpoint_index] += 1.0;
                    }
                }

                // Also check -k' (time reversal)
                if (use_time_reversal) {
                    double negated_fractional_index_1 =
                        -rotated_crystal[0] * grid_dimension_1 - 0.5 * shift_1;
                    double negated_fractional_index_2 =
                        -rotated_crystal[1] * grid_dimension_2 - 0.5 * shift_2;
                    double negated_fractional_index_3 =
                        -rotated_crystal[2] * grid_dimension_3 - 0.5 * shift_3;

                    bool is_negated_on_grid =
                        std::abs(negated_fractional_index_1 -
                                 std::round(negated_fractional_index_1)) <= grid_tolerance &&
                        std::abs(negated_fractional_index_2 -
                                 std::round(negated_fractional_index_2)) <= grid_tolerance &&
                        std::abs(negated_fractional_index_3 -
                                 std::round(negated_fractional_index_3)) <= grid_tolerance;

                    if (is_negated_on_grid) {
                        int negated_mapped_index_1 =
                            ((static_cast<int>(
                                 std::round(negated_fractional_index_1 + 2 * grid_dimension_1))) %
                             grid_dimension_1);
                        int negated_mapped_index_2 =
                            ((static_cast<int>(
                                 std::round(negated_fractional_index_2 + 2 * grid_dimension_2))) %
                             grid_dimension_2);
                        int negated_mapped_index_3 =
                            ((static_cast<int>(
                                 std::round(negated_fractional_index_3 + 2 * grid_dimension_3))) %
                             grid_dimension_3);
                        int negated_flat_index =
                            negated_mapped_index_3 + negated_mapped_index_2 * grid_dimension_3 +
                            negated_mapped_index_1 * grid_dimension_2 * grid_dimension_3;

                        if (negated_flat_index > kpoint_index &&
                            !has_representative[negated_flat_index]) {
                            has_representative[negated_flat_index] = true;
                            weight_accumulator[kpoint_index] += 1.0;
                        }
                    }
                }
            }
        }

        // Collect irreducible k-points and normalize weights
        double total_weight = 0.0;
        int number_of_irreducible_kpoints = 0;
        KPointsGrid kpoints;

        for (int grid_index = 0; grid_index < total_grid_points; grid_index++) {
            if (!has_representative[grid_index]) {
                number_of_irreducible_kpoints++;
                total_weight += weight_accumulator[grid_index];

                KPoint kpoint_data;
                /**
                 * Already in the first BZ: KPointsGrid::generate() folds to
                 * [-0.5, 0.5) as it builds. Folding again here is not a no-op --
                 * std::round(-0.5) is -1, so a coordinate sitting exactly on the
                 * zone boundary would come back as +0.5 and name the opposite
                 * face. The fold belonged to the mesh this function used to
                 * rebuild for itself, and it left with it.
                 */
                kpoint_data.coordinates = mesh_kpoints[grid_index].coordinates;
                kpoint_data.weight = weight_accumulator[grid_index];
                kpoints.add_kpoint(kpoint_data);
            }
        }

        /**
         * Build final KPointsGrid with normalized weights
         * w_k = count_k / total * 2 (factor 2 for spin degeneracy)
         */
        KPointsGrid normalized_kpoints;
        normalized_kpoints.set_grid_metadata(dimensions, shifts);
        for (int kpoint_index = 0; kpoint_index < kpoints.number_of_kpoints(); kpoint_index++) {
            KPoint kpoint_data;
            kpoint_data.coordinates = kpoints[kpoint_index].coordinates;
            kpoint_data.weight = kpoints[kpoint_index].weight / total_weight * 2.0;
            normalized_kpoints.add_kpoint(kpoint_data);
        }

        log::info("     IBZ reduction (internal): {} -> {} k-points ({} symmetry ops{})",
                  total_grid_points,
                  number_of_irreducible_kpoints,
                  number_of_operations,
                  use_time_reversal ? " + time reversal" : "");
        normalized_kpoints.pretty_print();

        return normalized_kpoints;
    }

    /**
     * Get all symmetry operations (rotation matrices and fractional translations).
     * Rotation matrices are in crystal (fractional) coordinates.
     * Fractional translations are in crystal coordinates.
     * Includes atom_map for each operation: atom_map[na] = image of atom na.
     */
    static std::vector<SymmetryOperation> get_symmetry_operations(
        const Crystal& crystal,
        Real symmetry_precision = math::tolerances::default_symmetry_precision) {
        double lattice[3][3];
        std::vector<double> positions;
        std::vector<int> types;
        detail::crystal_to_spglib(crystal, lattice, positions, types);

        SpglibDataset* dataset = spg_get_dataset(lattice,
                                                 reinterpret_cast<double(*)[3]>(positions.data()),
                                                 types.data(),
                                                 crystal.number_of_atoms(),
                                                 symmetry_precision);

        std::vector<SymmetryOperation> operations;
        if (!dataset) {
            log::warn("spglib: failed to get symmetry operations");
            // Return identity only
            SymmetryOperation identity;
            identity.rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
            identity.reciprocal_rotation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
            identity.fractional_translation = {0.0, 0.0, 0.0};
            operations.push_back(identity);
            return operations;
        }

        operations.resize(dataset->n_operations);
        for (int operation_index = 0; operation_index < dataset->n_operations; operation_index++) {
            auto& operation = operations[operation_index];
            // spglib rotation[isym][i][j]: rotated_cryst[i] = sum_j rotation[isym][i][j] * cryst[j]
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    operation.rotation[row][column] =
                        dataset->rotations[operation_index][row][column];
                }
            }
            operation.fractional_translation.x = dataset->translations[operation_index][0];
            operation.fractional_translation.y = dataset->translations[operation_index][1];
            operation.fractional_translation.z = dataset->translations[operation_index][2];

            // Compute reciprocal_rotation = (R^{-1})^T for G-vector transformation.

            // For integer rotation matrices with det(R) = ±1:

            //    R^{-1} = adj(R) / det(R)

            //    (R^{-1})^T = adj(R)^T / det(R) = cofactor(R) / det(R)

            // But adj(R)^T = cofactor matrix (since adj = cofactor^T).

            // So (R^{-1})^T = cofactor(R) / det(R).

            // Equivalently: (R^{-1})^T_{ij} = adj(R)_{ji} / det(R)
            const auto& rotation_matrix = operation.rotation;

            // Compute determinant
            int determinant =
                rotation_matrix[0][0] * (rotation_matrix[1][1] * rotation_matrix[2][2] -
                                         rotation_matrix[1][2] * rotation_matrix[2][1]) -
                rotation_matrix[0][1] * (rotation_matrix[1][0] * rotation_matrix[2][2] -
                                         rotation_matrix[1][2] * rotation_matrix[2][0]) +
                rotation_matrix[0][2] * (rotation_matrix[1][0] * rotation_matrix[2][1] -
                                         rotation_matrix[1][1] * rotation_matrix[2][0]);

            // Compute cofactor matrix elements

            // cofactor[i][j] = (-1)^(i+j) * minor(i,j)
            int cofactor_matrix[3][3];
            cofactor_matrix[0][0] = rotation_matrix[1][1] * rotation_matrix[2][2] -
                                    rotation_matrix[1][2] * rotation_matrix[2][1];
            cofactor_matrix[0][1] = -(rotation_matrix[1][0] * rotation_matrix[2][2] -
                                      rotation_matrix[1][2] * rotation_matrix[2][0]);
            cofactor_matrix[0][2] = rotation_matrix[1][0] * rotation_matrix[2][1] -
                                    rotation_matrix[1][1] * rotation_matrix[2][0];
            cofactor_matrix[1][0] = -(rotation_matrix[0][1] * rotation_matrix[2][2] -
                                      rotation_matrix[0][2] * rotation_matrix[2][1]);
            cofactor_matrix[1][1] = rotation_matrix[0][0] * rotation_matrix[2][2] -
                                    rotation_matrix[0][2] * rotation_matrix[2][0];
            cofactor_matrix[1][2] = -(rotation_matrix[0][0] * rotation_matrix[2][1] -
                                      rotation_matrix[0][1] * rotation_matrix[2][0]);
            cofactor_matrix[2][0] = rotation_matrix[0][1] * rotation_matrix[1][2] -
                                    rotation_matrix[0][2] * rotation_matrix[1][1];
            cofactor_matrix[2][1] = -(rotation_matrix[0][0] * rotation_matrix[1][2] -
                                      rotation_matrix[0][2] * rotation_matrix[1][0]);
            cofactor_matrix[2][2] = rotation_matrix[0][0] * rotation_matrix[1][1] -
                                    rotation_matrix[0][1] * rotation_matrix[1][0];

            // reciprocal_rotation = (R^{-1})^T = cofactor(R) / det(R)

            // (R^{-1})^T_{ij} = cofactor_{ij} / det
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    operation.reciprocal_rotation[row][column] =
                        cofactor_matrix[row][column] / determinant;
                }
            }

            // Build atom mapping table: atom_map[na] = nb such that

            // R * tau(na) + ft ≡ tau(nb) (mod lattice vector).
            int number_of_atoms = crystal.number_of_atoms();
            operation.atom_map.resize(number_of_atoms, -1);
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                Vector3D tau_cryst = crystal.fractional_position(atom_index);
                int element_type = crystal.site_symmetry_class(atom_index);

                // Apply rotation + fractional translation in crystal coordinates
                Vector3D rotated_position;
                rotated_position.x =
                    rotation_matrix[0][0] * tau_cryst.x + rotation_matrix[0][1] * tau_cryst.y +
                    rotation_matrix[0][2] * tau_cryst.z + operation.fractional_translation.x;
                rotated_position.y =
                    rotation_matrix[1][0] * tau_cryst.x + rotation_matrix[1][1] * tau_cryst.y +
                    rotation_matrix[1][2] * tau_cryst.z + operation.fractional_translation.y;
                rotated_position.z =
                    rotation_matrix[2][0] * tau_cryst.x + rotation_matrix[2][1] * tau_cryst.y +
                    rotation_matrix[2][2] * tau_cryst.z + operation.fractional_translation.z;

                // Find which atom matches (modulo lattice vector, same symmetry class)
                for (int target_atom = 0; target_atom < number_of_atoms; target_atom++) {
                    if (crystal.site_symmetry_class(target_atom) != element_type)
                        continue;
                    Vector3D target_cryst = crystal.fractional_position(target_atom);
                    Vector3D difference = rotated_position - target_cryst;
                    // Reduce to [-0.5, 0.5)
                    difference.x -= std::round(difference.x);
                    difference.y -= std::round(difference.y);
                    difference.z -= std::round(difference.z);
                    if (std::abs(difference.x) < symmetry_precision &&
                        std::abs(difference.y) < symmetry_precision &&
                        std::abs(difference.z) < symmetry_precision) {
                        operation.atom_map[atom_index] = target_atom;
                        break;
                    }
                }
            }
        }

        spg_free_dataset(dataset);
        return operations;
    }

    /**
     * Symmetrize a vector quantity f(3, nat) such as forces.
     * Convert to crystal coordinates, average over the symmetry operations (using the
     * rotation matrices and the atom mapping), convert back to Cartesian.
     * The average is the group-average (Reynolds) projector onto the subspace of fields
     * invariant under the crystal's point group: an already-symmetric field is an exact
     * fixed point, and any component the group forbids is projected to zero.
     */
    static void symmetrize_vector(const Crystal& crystal,
                                  const std::vector<SymmetryOperation>& symmetry_operations,
                                  std::vector<Vector3D>& vector_field) {
        int number_of_symmetry_operations = static_cast<int>(symmetry_operations.size());
        if (number_of_symmetry_operations <= 1)
            return;

        int number_of_atoms = crystal.number_of_atoms();
        const Matrix3x3& direct_lattice_vectors = crystal.lattice_vectors();
        const Matrix3x3& reciprocal_lattice_vectors = crystal.reciprocal_lattice_vectors();

        // Step 1: convert vector_field from Cartesian to crystal coordinates.

        // A covariant vector is projected onto the direct lattice vectors:

        // work[ipol] = F · a_ipol = Σ_j F[j] * A[ipol][j].

        // In Q3: at[ipol][j] = direct_lattice_vectors[ipol][j]
        std::vector<Vector3D> work(number_of_atoms);
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            for (int dimension = 0; dimension < 3; dimension++) {
                work[atom_index][dimension] =
                    vector_field[atom_index][0] * direct_lattice_vectors[dimension][0] +
                    vector_field[atom_index][1] * direct_lattice_vectors[dimension][1] +
                    vector_field[atom_index][2] * direct_lattice_vectors[dimension][2];
            }
        }

        // Step 2: symmetrize in crystal coordinates.

        /**
         * In crystal coordinates the symmetrized field is
         *   result[atom][i] = Σ_op Σ_j rotation[j][i] * work[atom_map[atom]][j] / n_ops,
         * i.e. the stored `.rotation` TRANSPOSED, applied to the IMAGE atom's components.
         *
         * Why transposed, and why paired with the image. `work` holds the covariant
         * components w_i = F · a_i. Since an operation maps the lattice onto itself as
         * M a_i = Σ_j R[j][i] a_j (with R the stored fractional-coordinate rotation for
         * positions, x' = R x), the covariant components of the ROTATED field are
         * (R^{-1})^T w. The symmetry-restoring average is the group average
         *   F_sym(a) = (1/n_ops) Σ_op M_op^{-1} F(op(a)),
         * whose covariant form needs the transform of the INVERSE operation -- which is
         * R^T -- applied to the components at the image atom op(a) = atom_map[atom].
         * That combination, and only that combination, leaves an already-symmetric field
         * unchanged.
         *
         * Note R^T is NOT (R^{-1})^T here: an integer crystallographic rotation in a
         * fractional basis is not orthogonal, so the two differ (the same point this
         * class makes for k-vectors in reduce_kpoints_internal). Using
         * `reciprocal_rotation = (rotation^{-1})^T` with the image atom would not be a
         * fixed point of an already-symmetric field.
         */
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            vector_field[atom_index] = {0, 0, 0};
        }

        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            for (int operation_index = 0; operation_index < number_of_symmetry_operations;
                 operation_index++) {
                const auto& rotation = symmetry_operations[operation_index].rotation;
                int mapped_atom = symmetry_operations[operation_index].atom_map[atom_index];

                for (int dimension = 0; dimension < 3; dimension++) {
                    vector_field[atom_index][dimension] +=
                        rotation[0][dimension] * work[mapped_atom][0] +
                        rotation[1][dimension] * work[mapped_atom][1] +
                        rotation[2][dimension] * work[mapped_atom][2];
                }
            }
        }

        // Divide by number of symmetry operations
        Real inverse_number_of_operations = 1.0 / number_of_symmetry_operations;
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            for (int dimension = 0; dimension < 3; dimension++) {
                work[atom_index][dimension] =
                    vector_field[atom_index][dimension] * inverse_number_of_operations;
            }
        }

        // Step 3: convert back to Cartesian coordinates.

        // Expand the crystal components back on the reciprocal lattice vectors:

        // F'[dim] = Σ_j work[j] * B[j][dim] = (B^T × work)[dim],

        // where B has rows = b_i (reciprocal_lattice_vectors). This is the inverse of Step 1.
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            for (int dimension = 0; dimension < 3; dimension++) {
                vector_field[atom_index][dimension] =
                    work[atom_index][0] * reciprocal_lattice_vectors[0][dimension] +
                    work[atom_index][1] * reciprocal_lattice_vectors[1][dimension] +
                    work[atom_index][2] * reciprocal_lattice_vectors[2][dimension];
            }
        }
    }

    /**
     * Symmetrize a rank-2 tensor (3×3 matrix) such as the stress tensor.
     * Convert to crystal coordinates, average over the point group, convert back:
     *   T'[i][j] = (1/n_ops) Σ_op Σ_{k,l} rotation[k][i] * rotation[l][j] * T[k][l],
     * i.e. the same group-average projector symmetrize_vector applies, with both tensor
     * indices carried instead of one. With no atom index there is no image lookup, and
     * because the sum runs over the whole group it is unchanged by replacing each
     * operation with its inverse.
     */
    static void symmetrize_matrix(const Crystal& crystal,
                                  const std::vector<SymmetryOperation>& symmetry_operations,
                                  Matrix3x3& tensor) {
        int number_of_symmetry_operations = static_cast<int>(symmetry_operations.size());
        if (number_of_symmetry_operations <= 1)
            return;

        const Matrix3x3& direct_lattice = crystal.lattice_vectors();
        const Matrix3x3& reciprocal_lattice = crystal.reciprocal_lattice_vectors();

        // Step 1: Cartesian → crystal coordinates

        // Project both tensor indices onto the direct lattice vectors:

        // work(i,j) = Σ_{k,l} T(k,l) * A[i][k] * A[j][l], where A[i][k] is the k-th

        // Cartesian component of the i-th lattice vector -- i.e. Q3's row-major at[i][k],

        // the transpose of the column-major `at(k,i)` layout Quantum ESPRESSO uses.
        Matrix3x3 crystal_tensor;
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                Real sum = 0.0;
                for (int contracted_row = 0; contracted_row < 3; contracted_row++) {
                    for (int contracted_column = 0; contracted_column < 3; contracted_column++) {
                        sum += tensor[contracted_row][contracted_column] *
                               direct_lattice[row][contracted_row] *
                               direct_lattice[column][contracted_column];
                    }
                }
                crystal_tensor[row][column] = sum;
            }
        }

        // Step 2: symmetrize in crystal coordinates

        // Both tensor indices are covariant in crystal coordinates, so each operation

        // contributes R^T * T * R, i.e. Σ_{k,l} R[k][i] * R[l][j] * T[k][l] -- R^T being

        // the covariant transform of the inverse operation, as in symmetrize_vector.

        // Summed over the whole group the distinction is immaterial here, since the

        // group contains the inverse of every operation and no atom index is paired in.
        Matrix3x3 symmetrized;
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                symmetrized[row][column] = 0.0;
            }
        }

        for (int operation_index = 0; operation_index < number_of_symmetry_operations;
             operation_index++) {
            const auto& rotation = symmetry_operations[operation_index].rotation;
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    for (int contracted_row = 0; contracted_row < 3; contracted_row++) {
                        for (int contracted_column = 0; contracted_column < 3;
                             contracted_column++) {
                            symmetrized[row][column] +=
                                rotation[contracted_row][row] *
                                rotation[contracted_column][column] *
                                crystal_tensor[contracted_row][contracted_column];
                        }
                    }
                }
            }
        }

        Real inverse_number_of_operations = 1.0 / number_of_symmetry_operations;
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                crystal_tensor[row][column] =
                    symmetrized[row][column] * inverse_number_of_operations;
            }
        }

        // Step 3: crystal → Cartesian coordinates

        // Expand both tensor indices back on the reciprocal lattice vectors:

        // T(i,j) = Σ_{k,l} work(k,l) * B[k][i] * B[l][j], where B[k][i] is the i-th

        // Cartesian component of the k-th reciprocal vector -- i.e. Q3's row-major

        // bg[k][i], the transpose of the `bg(i,k)` layout Quantum ESPRESSO uses.
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                Real sum = 0.0;
                for (int contracted_row = 0; contracted_row < 3; contracted_row++) {
                    for (int contracted_column = 0; contracted_column < 3; contracted_column++) {
                        sum += crystal_tensor[contracted_row][contracted_column] *
                               reciprocal_lattice[contracted_row][row] *
                               reciprocal_lattice[contracted_column][column];
                    }
                }
                tensor[row][column] = sum;
            }
        }
    }
};

} // namespace q3
