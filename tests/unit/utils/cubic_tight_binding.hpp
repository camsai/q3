#pragma once

/**
 * The one-band simple-cubic tight-binding model every interpolator test uses.
 *
 * e(k) = onsite + 2 t [cos 2pi kx + cos 2pi ky + cos 2pi kz] is the analytic
 * band of a cubic lattice with nearest-neighbour hopping t. Sampling it on a
 * mesh and transforming to R recovers exactly those hoppings, so the
 * interpolated band equals the analytic one everywhere, on the mesh and
 * between its points -- which makes it the one fixture where an interpolation
 * error is unambiguously a bug rather than physics.
 *
 * Shared rather than copied because six test files need the same model, and a
 * sign or 2-pi convention that drifts in one copy silently decouples that
 * file's expectations from everyone else's.
 */

#include <cmath>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/local_orbital.hpp"

namespace q3::test_support {

inline Matrix3x3 simple_cubic_lattice(Real lattice_parameter) {
    Matrix3x3 lattice;
    lattice[0] = Vector3D(lattice_parameter, 0.0, 0.0);
    lattice[1] = Vector3D(0.0, lattice_parameter, 0.0);
    lattice[2] = Vector3D(0.0, 0.0, lattice_parameter);
    return lattice;
}

inline Real analytic_cubic_band(const Vector3D& kpoint, Real onsite, Real hopping) {
    return onsite +
           2.0 * hopping *
               (std::cos(constants::two_pi * kpoint.x) + std::cos(constants::two_pi * kpoint.y) +
                std::cos(constants::two_pi * kpoint.z));
}

/// The Gamma-centred mesh, first index slowest, as KPointsGrid enumerates it.
inline std::vector<Vector3D> gamma_centered_mesh(int mesh) {
    std::vector<Vector3D> kpoints;
    kpoints.reserve(static_cast<std::size_t>(mesh) * mesh * mesh);
    for (int x = 0; x < mesh; x++) {
        for (int y = 0; y < mesh; y++) {
            for (int z = 0; z < mesh; z++) {
                kpoints.emplace_back(static_cast<Real>(x) / mesh,
                                     static_cast<Real>(y) / mesh,
                                     static_cast<Real>(z) / mesh);
            }
        }
    }
    return kpoints;
}

/**
 * The analytic band sampled on `mesh`^3 and taken through the Wannier method
 * with an identity gauge, which is what a one-band model's gauge always is.
 */
inline LocalOrbitalQuantumStateInterpolator build_cubic_interpolator(
    int mesh, Real onsite, Real hopping, const std::string& description = "test") {
    WannierInputs inputs;
    inputs.kpoint_grid_dimensions = {mesh, mesh, mesh};
    inputs.number_of_bands = 1;
    inputs.number_of_trial_orbitals = 1;
    inputs.kpoint_coordinates = gamma_centered_mesh(mesh);
    for (const Vector3D& kpoint : inputs.kpoint_coordinates) {
        inputs.eigenvalues_rydberg.push_back({analytic_cubic_band(kpoint, onsite, hopping)});
        inputs.projection_matrices.push_back({Complex(1.0, 0.0)});
    }
    return LocalOrbitalQuantumStateInterpolator(
        WannierHamiltonianBuilder().build(inputs, simple_cubic_lattice(4.0)).hamiltonian,
        description);
}

} // namespace q3::test_support
