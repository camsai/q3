#pragma once

/**
 * ProjectorStrainInputs — what the strain derivatives of one k-point's
 * projectors are built from, apart from the atoms.
 *
 * The k-point, its plane waves, and the functions of k+G that every atom's
 * projectors share: |k+G|, the real spherical harmonics and their Cartesian
 * derivatives, the components of k+G, and the weights (k+G)_a (k+G)_b / |k+G|
 * of the six strain components. Whoever computes the stress computes these
 * once per k-point and lends them, through this record, to whatever forms the
 * projector overlaps: nothing here is owned, and everything pointed at must
 * outlive the use of the record.
 */

#include <array>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/// The atom-independent inputs of a k-point's projector strain derivatives; see the file comment.
struct ProjectorStrainInputs {
    /// The k-point in Cartesian coordinates, in units of 2π/a.
    Vector3D kpoint_cartesian;
    /// The k-point's plane waves, as indices into the basis's G-vectors.
    const std::vector<int>* plane_wave_indices = nullptr;
    /// How many of them there are.
    int number_of_plane_waves = 0;
    /// |k+G| in inverse bohr, per plane wave.
    const std::vector<Real>* magnitude_bohr = nullptr;
    /// Real spherical harmonics Y_lm(k+G), indexed [plane wave][lm].
    const std::vector<std::vector<Real>>* spherical_harmonics = nullptr;
    /// Their derivatives along each Cartesian direction, in (2π/a)⁻¹: [direction][plane wave][lm].
    const std::vector<std::vector<std::vector<Real>>>* spherical_harmonic_derivatives = nullptr;
    /// The Cartesian components of k+G in inverse bohr, per plane wave.
    std::array<const std::vector<Real>*, 3> kpoint_plus_g_components{nullptr, nullptr, nullptr};
    /// (k+G)_a (k+G)_b / |k+G| for the six strain components, per plane wave.
    const std::array<std::vector<Real>, 6>* radial_strain_weights = nullptr;
};

} // namespace q3
