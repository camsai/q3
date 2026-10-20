#pragma once

/**
 * InterpolatedBandStructureCalculator — eigenvalues along an arbitrary list
 * of k-points, from a QuantumStateInterpolator.
 *
 * Interpolation when the model came from a coarse-mesh construction (Wannier
 * functions, a Fourier fit): exact at the mesh points, an interpolation
 * everywhere else, with an error set by the method -- how fast H(R) decays,
 * how many star functions were fitted. Not interpolation at all when the
 * model came from real-space integrals in an atomic-orbital basis; then every
 * k-point is as good as every other. The calculator is the same in both
 * cases, which is why it is written against the interpolator interface and
 * not against any one method's model.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/init.hpp"

namespace q3 {

/// Eigenvalues along a k-path, one row per k-point.
struct [[nodiscard]] InterpolatedBandStructure {
    std::vector<Vector3D> kpoint_coordinates;
    /// [path_index][band_index], Rydberg, ascending within each row.
    std::vector<std::vector<Real>> eigenvalues_rydberg;
};

class InterpolatedBandStructureCalculator {
  public:
    explicit InterpolatedBandStructureCalculator(const QuantumStateInterpolator& interpolator)
        : interpolator_(interpolator) {}

    /// Eigenvalues at each supplied k-point, in crystal coordinates.
    [[nodiscard]] InterpolatedBandStructure
    compute(const std::vector<Vector3D>& kpoint_coordinates) const {
        InterpolatedBandStructure bands;
        bands.kpoint_coordinates = kpoint_coordinates;
        bands.eigenvalues_rydberg.reserve(kpoint_coordinates.size());
        for (const Vector3D& kpoint : kpoint_coordinates) {
            bands.eigenvalues_rydberg.push_back(interpolator_.eigenvalues_at(kpoint));
        }
        return bands;
    }

  private:
    const QuantumStateInterpolator& interpolator_;
};

} // namespace q3
