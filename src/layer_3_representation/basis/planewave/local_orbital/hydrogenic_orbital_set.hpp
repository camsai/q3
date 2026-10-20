#pragma once

/**
 * PlanewaveHydrogenicOrbitalSet — s-type orbitals at arbitrary centres,
 * rendered in the plane-wave basis at one k-point.
 *
 * The pseudo-atomic orbitals a pseudopotential ships are all centred on
 * atoms, which is the wrong shape for a material whose Wannier functions
 * are not. Silicon is the standard example: its four valence Wannier
 * functions sit on the Si-Si BONDS, not on either atom, and no subset of
 * atom-centred orbitals spans that manifold well. Being able to place
 * orbitals freely is what makes an isolated-manifold Wannierization
 * possible at all without a minimiser.
 *
 * A hydrogenic 1s of decay a,
 *
 *   g(r) = (pi a^3)^(-1/2) exp(-|r - tau| / a)
 *
 * has an analytic Fourier transform,
 *
 *   g(q) = (pi a^3)^(-1/2) * 8 pi a^3 / (1 + (q a)^2)^2 * exp(-i q.tau)
 *
 * so this needs no radial table and no interpolation -- unlike the
 * pseudo-atomic set, which reads a tabulated chi(q). That also means there
 * is no interpolation error here to trade off against anything.
 *
 * Only l = 0 is implemented. An s orbital is spherically symmetric, so its
 * angular factor is the constant Y_00 and the i^l phase is 1; both drop out
 * of the Loewdin orthonormalization that follows anyway. Higher l would need
 * the real spherical harmonics and the i^l convention the atomic-orbital
 * builder already uses, and is deliberately left until something needs it.
 *
 * The overall normalization is likewise irrelevant downstream: consumers
 * orthonormalize these in the S metric before projecting, which divides it
 * out. It is kept correct anyway so that a raw projection is interpretable
 * on its own.
 */

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/init.hpp"

namespace q3 {

/// One s-type orbital: where it sits and how fast it decays.
struct [[nodiscard]] HydrogenicOrbital {
    /// Centre in crystal (fractional) coordinates.
    Vector3D center_crystal;
    /// Decay length in bohr; the Wannier90 convention's `zona` is 1/a.
    Real decay_length_bohr = 1.0;
};

class PlanewaveHydrogenicOrbitalSet : public PlanewaveLocalOrbitalSet {
  public:
    explicit PlanewaveHydrogenicOrbitalSet(std::vector<HydrogenicOrbital> orbitals)
        : orbitals_(std::move(orbitals)) {
        for (const HydrogenicOrbital& orbital : orbitals_) {
            if (orbital.decay_length_bohr <= 0.0) {
                throw std::invalid_argument(
                    "PlanewaveHydrogenicOrbitalSet: decay_length_bohr must be positive, got " +
                    std::to_string(orbital.decay_length_bohr));
            }
        }
    }

    [[nodiscard]] int number_of_orbitals() const override {
        return static_cast<int>(orbitals_.size());
    }

    [[nodiscard]] const std::vector<HydrogenicOrbital>& orbitals() const {
        return orbitals_;
    }

    /**
     * Follows the same conventions as build_atomic_orbitals_g_space: the
     * structure factor is e^{-i 2*pi (k+G).tau} with both the k-point and
     * the centre in alat-relative Cartesian units, and |k+G| carries the
     * 2*pi/alat factor to become a physical wavevector.
     */
    void build(int kpoint_index,
               int number_of_plane_waves,
               const std::vector<int>& g_vector_indices,
               const Crystal& crystal_structure,
               const KPointsGrid& kpoint_grid,
               const PlanewaveBasis& planewave_basis,
               std::vector<std::vector<Complex>>& orbitals) const override {
        const Real two_pi_over_lattice_parameter =
            crystal_structure.two_pi_over_lattice_parameter();
        const Vector3D kpoint_crystal = kpoint_grid[kpoint_index].coordinates;
        const Matrix3x3& reciprocal_lattice = crystal_structure.reciprocal_lattice_vectors();

        Vector3D kpoint_cartesian;
        for (int dimension = 0; dimension < 3; dimension++) {
            kpoint_cartesian[dimension] = kpoint_crystal.x * reciprocal_lattice[0][dimension] +
                                          kpoint_crystal.y * reciprocal_lattice[1][dimension] +
                                          kpoint_crystal.z * reciprocal_lattice[2][dimension];
        }

        orbitals.assign(orbitals_.size(),
                        std::vector<Complex>(number_of_plane_waves, Complex(0.0, 0.0)));

        for (std::size_t orbital_index = 0; orbital_index < orbitals_.size(); orbital_index++) {
            const HydrogenicOrbital& orbital = orbitals_[orbital_index];
            /**
             * The centre arrives in crystal coordinates; carry it to the
             * alat-relative Cartesian frame the structure factor below uses,
             * exactly as Crystal::convert_from_crystal_to_cartesian does for
             * an atom.
             */
            const Matrix3x3& lattice_vectors = crystal_structure.lattice_vectors();
            Vector3D center_cartesian;
            for (int dimension = 0; dimension < 3; dimension++) {
                center_cartesian[dimension] =
                    orbital.center_crystal.x * lattice_vectors[0][dimension] +
                    orbital.center_crystal.y * lattice_vectors[1][dimension] +
                    orbital.center_crystal.z * lattice_vectors[2][dimension];
            }

            const Real kpoint_phase_argument =
                constants::two_pi *
                (kpoint_cartesian.x * center_cartesian.x + kpoint_cartesian.y * center_cartesian.y +
                 kpoint_cartesian.z * center_cartesian.z);
            const Complex kpoint_phase = compute_phase_factor(kpoint_phase_argument);

            const Real decay = orbital.decay_length_bohr;
            /**
             * g(q) = 8 sqrt(pi a^3) / (1 + a^2 q^2)^2 is the transform of the
             * unit-normalized 1s orbital; the 1/sqrt(Omega) carries it to the
             * plane-wave coefficient convention the Kleinman-Bylander
             * projectors and the pseudo-atomic orbitals use.
             */
            const Real normalisation = 8.0 * constants::pi * decay * decay * decay /
                                       std::sqrt(constants::pi * decay * decay * decay) /
                                       std::sqrt(crystal_structure.cell_volume());

            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                const GVector& g_vector = planewave_basis.g_vectors()[g_vector_indices[g_index]];
                const Real component_x = kpoint_cartesian.x + g_vector.g_cart.x;
                const Real component_y = kpoint_cartesian.y + g_vector.g_cart.y;
                const Real component_z = kpoint_cartesian.z + g_vector.g_cart.z;
                const Real magnitude =
                    std::sqrt(component_x * component_x + component_y * component_y +
                              component_z * component_z) *
                    two_pi_over_lattice_parameter;

                const Real denominator = 1.0 + magnitude * magnitude * decay * decay;
                const Real radial = normalisation / (denominator * denominator);

                const Real structure_phase_argument =
                    constants::two_pi * (g_vector.g_cart.x * center_cartesian.x +
                                         g_vector.g_cart.y * center_cartesian.y +
                                         g_vector.g_cart.z * center_cartesian.z);
                orbitals[orbital_index][g_index] =
                    kpoint_phase * compute_phase_factor(structure_phase_argument) * radial;
            }
        }
    }

    /// Exactly the centres the manifest named, in the order it named them.
    [[nodiscard]] std::vector<Vector3D> centers_crystal() const override {
        std::vector<Vector3D> centers;
        centers.reserve(orbitals_.size());
        for (const HydrogenicOrbital& orbital : orbitals_) {
            centers.push_back(orbital.center_crystal);
        }
        return centers;
    }

    [[nodiscard]] std::string description() const override {
        std::ostringstream text;
        text << number_of_orbitals() << " hydrogenic s orbitals at explicit centres (decay "
             << (orbitals_.empty() ? 0.0 : orbitals_.front().decay_length_bohr) << " bohr)";
        return text.str();
    }

  private:
    std::vector<HydrogenicOrbital> orbitals_;
};

} // namespace q3
