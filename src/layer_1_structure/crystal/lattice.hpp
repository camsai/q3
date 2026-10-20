#pragma once

/**
 * Lattice vector generation from Bravais lattice index.
 *
 * Provides get_lattice_vectors_from_bravais_index, which converts a standard
 * Bravais lattice index and cell dimension parameters into an explicit 3x3
 * matrix of lattice vectors in units of the lattice parameter (alat).
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/matrix_3x3.hpp"
#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

/**
 * Build lattice vectors from cell parameters a, b, c (any consistent length
 * unit) and angles alpha, beta, gamma (degrees).
 *
 * Standard triclinic construction: a1 along x, a2 in the xy plane
 * (a2 = b(cos γ, sin γ, 0)), a3 completing the cell. Rows of the returned
 * matrix are a1, a2, a3 in the same unit as the input lengths.
 *
 * Throws std::runtime_error for degenerate angle combinations (sin γ ≈ 0 or a
 * zero-volume cell).
 */
inline Matrix3x3 get_lattice_vectors_from_cell_parameters(Real length_a,
                                                          Real length_b,
                                                          Real length_c,
                                                          Real alpha_degrees,
                                                          Real beta_degrees,
                                                          Real gamma_degrees) {
    Real cos_alpha = std::cos(alpha_degrees * constants::degrees_to_radians);
    Real cos_beta = std::cos(beta_degrees * constants::degrees_to_radians);
    Real cos_gamma = std::cos(gamma_degrees * constants::degrees_to_radians);
    Real sin_gamma = std::sin(gamma_degrees * constants::degrees_to_radians);
    if (std::abs(sin_gamma) < 1e-12)
        throw std::runtime_error("Invalid lattice angle gamma: " + std::to_string(gamma_degrees));

    Real c_direction_x = cos_beta;
    Real c_direction_y = (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    Real c_direction_z_squared =
        1.0 - c_direction_x * c_direction_x - c_direction_y * c_direction_y;
    if (c_direction_z_squared <= 0.0)
        throw std::runtime_error("Invalid lattice angles: cell has no volume");

    Matrix3x3 lattice_vectors;
    lattice_vectors[0] = {length_a, 0.0, 0.0};
    lattice_vectors[1] = {length_b * cos_gamma, length_b * sin_gamma, 0.0};
    lattice_vectors[2] = {length_c * c_direction_x,
                          length_c * c_direction_y,
                          length_c * std::sqrt(c_direction_z_squared)};
    return lattice_vectors;
}

/**
 * Compute lattice vectors from a Bravais lattice index and cell dimension parameters.
 *
 * The index encoding, and the primitive vectors each index stands for, are the `ibrav`
 * convention published in Quantum ESPRESSO's pw.x input documentation (Doc/INPUT_PW, the
 * `ibrav` variable). They are reproduced because they are what a QE input file *means*:
 * positions given in crystal units are coordinates in exactly these vectors, so a
 * different -- equally valid -- primitive cell for the same Bravais lattice would read
 * the same file as a different structure. There is no wider standard that fixes them;
 * other codes number and orient their cells differently. The one-line comment on each
 * `case` below restates that documentation's entry for its index, in the dimensionless
 * form this function returns.
 *
 * The returned matrix has rows a1, a2, a3 in units of alat (cell_dimension_1).
 *
 * Supported bravais_lattice_index values: 0, 1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, 10, 11, 12,
 * -12, 14. The convention also defines 13, -13, -9 and 91; Q3 does not build those cells
 * and refuses them rather than substituting a neighbouring one.
 *
 * Throws std::runtime_error for unsupported bravais_lattice_index values.
 */
inline Matrix3x3 get_lattice_vectors_from_bravais_index(int bravais_lattice_index,
                                                        Real cell_dimension_1,
                                                        Real cell_dimension_2 = 0,
                                                        Real cell_dimension_3 = 0,
                                                        Real cell_dimension_4 = 0,
                                                        Real cell_dimension_5 = 0,
                                                        Real cell_dimension_6 = 0) {
    Matrix3x3 lattice_vectors;
    Real ratio_c_over_a = cell_dimension_3;
    Real ratio_b_over_a = cell_dimension_2;

    switch (bravais_lattice_index) {
        case 0:
            /**
             * Free lattice: caller must set lattice vectors separately.
             * Initialize to identity as placeholder.
             */
            lattice_vectors[0] = {1.0, 0.0, 0.0};
            lattice_vectors[1] = {0.0, 1.0, 0.0};
            lattice_vectors[2] = {0.0, 0.0, 1.0};
            break;

        case 1:
            // Simple cubic: a1=a(1,0,0), a2=a(0,1,0), a3=a(0,0,1)
            lattice_vectors[0] = {1.0, 0.0, 0.0};
            lattice_vectors[1] = {0.0, 1.0, 0.0};
            lattice_vectors[2] = {0.0, 0.0, 1.0};
            break;

        case 2:
            // FCC: a1 = a/2 (-1,0,1), a2 = a/2 (0,1,1), a3 = a/2 (-1,1,0)
            lattice_vectors[0] = {-0.5, 0.0, 0.5};
            lattice_vectors[1] = {0.0, 0.5, 0.5};
            lattice_vectors[2] = {-0.5, 0.5, 0.0};
            break;

        case 3:
            // BCC: a1 = a/2 (1,1,1), a2 = a/2 (-1,1,1), a3 = a/2 (-1,-1,1)
            lattice_vectors[0] = {0.5, 0.5, 0.5};
            lattice_vectors[1] = {-0.5, 0.5, 0.5};
            lattice_vectors[2] = {-0.5, -0.5, 0.5};
            break;

        case -3:
            // BCC alt: a1 = a/2 (-1,1,1), a2 = a/2 (1,-1,1), a3 = a/2 (1,1,-1)
            lattice_vectors[0] = {-0.5, 0.5, 0.5};
            lattice_vectors[1] = {0.5, -0.5, 0.5};
            lattice_vectors[2] = {0.5, 0.5, -0.5};
            break;

        case 4: {
            // Hexagonal: a1 = a(1,0,0), a2 = a(-1/2,√3/2,0), a3 = a(0,0,c/a)
            Real half_sqrt3 = std::sqrt(3.0) / 2.0;
            lattice_vectors[0] = {1.0, 0.0, 0.0};
            lattice_vectors[1] = {-0.5, half_sqrt3, 0.0};
            lattice_vectors[2] = {0.0, 0.0, ratio_c_over_a};
            break;
        }

        case 5: {
            // Trigonal R (3-fold axis c): cell_dimension_4=cos(alpha)
            Real cos_alpha = cell_dimension_4;
            Real vector_component_x = std::sqrt((1.0 - cos_alpha) / 2.0);
            Real vector_component_y = std::sqrt((1.0 - cos_alpha) / 6.0);
            Real vector_component_z = std::sqrt((1.0 + 2.0 * cos_alpha) / 3.0);
            lattice_vectors[0] = {vector_component_x, -vector_component_y, vector_component_z};
            lattice_vectors[1] = {0.0, 2.0 * vector_component_y, vector_component_z};
            lattice_vectors[2] = {-vector_component_x, -vector_component_y, vector_component_z};
            break;
        }

        case -5: {
            // Trigonal R (3-fold axis <111>): cell_dimension_4=cos(alpha)
            Real cos_alpha = cell_dimension_4;
            Real vector_component_y = std::sqrt((1.0 - cos_alpha) / 6.0);
            Real vector_component_z = std::sqrt((1.0 + 2.0 * cos_alpha) / 3.0);
            Real vector_component_u =
                vector_component_z - 2.0 * std::sqrt(2.0) * vector_component_y;
            Real vector_component_v = vector_component_z + std::sqrt(2.0) * vector_component_y;
            lattice_vectors[0] = {vector_component_u / std::sqrt(3.0),
                                  vector_component_v / std::sqrt(3.0),
                                  vector_component_v / std::sqrt(3.0)};
            lattice_vectors[1] = {vector_component_v / std::sqrt(3.0),
                                  vector_component_u / std::sqrt(3.0),
                                  vector_component_v / std::sqrt(3.0)};
            lattice_vectors[2] = {vector_component_v / std::sqrt(3.0),
                                  vector_component_v / std::sqrt(3.0),
                                  vector_component_u / std::sqrt(3.0)};
            break;
        }

        case 6:
            // Tetragonal P: a1 = a(1,0,0), a2 = a(0,1,0), a3 = a(0,0,c/a)
            lattice_vectors[0] = {1.0, 0.0, 0.0};
            lattice_vectors[1] = {0.0, 1.0, 0.0};
            lattice_vectors[2] = {0.0, 0.0, ratio_c_over_a};
            break;

        case 7:
            // Body-centered tetragonal: a1=a/2(1,-1,c/a), a2=a/2(1,1,c/a), a3=a/2(-1,-1,c/a)
            lattice_vectors[0] = {0.5, -0.5, 0.5 * ratio_c_over_a};
            lattice_vectors[1] = {0.5, 0.5, 0.5 * ratio_c_over_a};
            lattice_vectors[2] = {-0.5, -0.5, 0.5 * ratio_c_over_a};
            break;

        case 8:
            // Orthorhombic P: a1=a(1,0,0), a2=a(0,b/a,0), a3=a(0,0,c/a)
            lattice_vectors[0] = {1.0, 0.0, 0.0};
            lattice_vectors[1] = {0.0, ratio_b_over_a, 0.0};
            lattice_vectors[2] = {0.0, 0.0, ratio_c_over_a};
            break;

        case 9:
            /**
             * Base-centered orthorhombic (bco):
             * a1=a/2(1,b/a,0), a2=a/2(-1,b/a,0), a3=a(0,0,c/a)
             */
            lattice_vectors[0] = {0.5, 0.5 * ratio_b_over_a, 0.0};
            lattice_vectors[1] = {-0.5, 0.5 * ratio_b_over_a, 0.0};
            lattice_vectors[2] = {0.0, 0.0, ratio_c_over_a};
            break;

        case 10:
            /**
             * Face-centered orthorhombic:
             * a1=a/2(1,0,c/a), a2=a/2(1,b/a,0), a3=a/2(0,b/a,c/a)
             */
            lattice_vectors[0] = {0.5, 0.0, 0.5 * ratio_c_over_a};
            lattice_vectors[1] = {0.5, 0.5 * ratio_b_over_a, 0.0};
            lattice_vectors[2] = {0.0, 0.5 * ratio_b_over_a, 0.5 * ratio_c_over_a};
            break;

        case 11:
            /**
             * Body-centered orthorhombic:
             * a1=a/2(1,b/a,c/a), a2=a/2(-1,b/a,c/a), a3=a/2(-1,-b/a,c/a)
             */
            lattice_vectors[0] = {0.5, 0.5 * ratio_b_over_a, 0.5 * ratio_c_over_a};
            lattice_vectors[1] = {-0.5, 0.5 * ratio_b_over_a, 0.5 * ratio_c_over_a};
            lattice_vectors[2] = {-0.5, -0.5 * ratio_b_over_a, 0.5 * ratio_c_over_a};
            break;

        case 12: {
            /**
             * Monoclinic P (unique axis c): cell_dimension_4=cos(ab)
             * a1=a(1,0,0), a2=a*b/a(cos(ab),sin(ab),0), a3=a(0,0,c/a)
             */
            Real cos_ab = cell_dimension_4;
            Real sin_ab = std::sqrt(1.0 - cos_ab * cos_ab);
            lattice_vectors[0] = {1.0, 0.0, 0.0};
            lattice_vectors[1] = {ratio_b_over_a * cos_ab, ratio_b_over_a * sin_ab, 0.0};
            lattice_vectors[2] = {0.0, 0.0, ratio_c_over_a};
            break;
        }

        case -12: {
            // Monoclinic P (unique axis b): cell_dimension_5=cos(ac)
            Real cos_ac = cell_dimension_5;
            Real sin_ac = std::sqrt(1.0 - cos_ac * cos_ac);
            lattice_vectors[0] = {1.0, 0.0, 0.0};
            lattice_vectors[1] = {0.0, ratio_b_over_a, 0.0};
            lattice_vectors[2] = {ratio_c_over_a * cos_ac, 0.0, ratio_c_over_a * sin_ac};
            break;
        }

        case 14: {
            /**
             * Triclinic: cell_dimension_4=cos(bc), cell_dimension_5=cos(ac),
             * cell_dimension_6=cos(ab)
             */
            Real cos_bc = cell_dimension_4;
            Real cos_ac = cell_dimension_5;
            Real cos_ab = cell_dimension_6;
            Real sin_ab = std::sqrt(1.0 - cos_ab * cos_ab);
            lattice_vectors[0] = {1.0, 0.0, 0.0};
            lattice_vectors[1] = {ratio_b_over_a * cos_ab, ratio_b_over_a * sin_ab, 0.0};
            Real vector_component_c_x = ratio_c_over_a * cos_ac;
            Real vector_component_c_y = ratio_c_over_a * (cos_bc - cos_ac * cos_ab) / sin_ab;
            Real vector_component_c_z =
                ratio_c_over_a *
                std::sqrt(1.0 + 2.0 * cos_bc * cos_ac * cos_ab - cos_bc * cos_bc - cos_ac * cos_ac -
                          cos_ab * cos_ab) /
                sin_ab;
            lattice_vectors[2] = {vector_component_c_x, vector_component_c_y, vector_component_c_z};
            break;
        }

        default:
            throw std::runtime_error(
                "Unsupported bravais_lattice_index=" + std::to_string(bravais_lattice_index) +
                ". Supported: 0 (free), 1 (SC), 2 (FCC), 3/-3 (BCC), 4 (hex), 5/-5 (trig R), "
                "6 (tet), 7 (bct), 8 (ort), 9 (bco), 10 (fco), 11 (bort), "
                "12/-12 (mono), 14 (tri)");
    }

    return lattice_vectors;
}

} // namespace q3
