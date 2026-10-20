#pragma once

/**
 * QuantumStateInterpolator — eigenvalues at any k-point, from a model built out of a
 * finished calculation.
 *
 * This is the umbrella over the interpolation METHODS. What differs between
 * them is what got interpolated. Wannier functions interpolate a Hamiltonian
 * in a localized basis: the k -> R transform of the coarse-mesh Bloch
 * Hamiltonian in the projection gauge, evaluated back at any k by a Bloch
 * sum. A Fourier fit interpolates the eigenvalues themselves, band by band,
 * with symmetry-adapted plane waves. An optimal-basis construction
 * interpolates the Hamiltonian in a compressed delocalized basis. Each has
 * its own failure mode -- a gauge that needs localizing, crossings smoothed
 * by index, a basis that grows with the band count -- and the umbrella must
 * not hide that, which is why the method stays visible in the input and in
 * the saved record.
 *
 * What they share is exactly this interface: hand over a k-point, get the
 * eigenvalues. The band-structure and density-of-states calculators in
 * layer 5 are written against it, so a new method reaches every consumer by
 * implementing one class and one line in the factory beside this file.
 *
 * k-points are crystal (fractional) coordinates, as everywhere else in Q3.
 * Implementations may keep scratch state (a LAPACK workspace, say) behind a
 * const interface; none is required to be safe to share between threads,
 * so a threaded consumer gives each thread its own instance.
 */

#include <map>
#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

class QuantumStateInterpolator {
  public:
    virtual ~QuantumStateInterpolator() = default;

    /// Number of eigenvalues returned per k-point.
    [[nodiscard]] virtual int number_of_bands() const = 0;

    /// Ascending eigenvalues in Rydberg at one k-point in crystal coordinates.
    [[nodiscard]] virtual std::vector<Real> eigenvalues_at(const Vector3D& kpoint) const = 0;

    /// One line for logs: the method and the size of the model.
    [[nodiscard]] virtual std::string description() const = 0;

    /**
     * Method-specific numbers worth reporting beside the results -- a
     * model's size, a weight identity it checked -- keyed by a name fit for a
     * JSON report. Empty by default; a method with nothing to add adds
     * nothing.
     */
    [[nodiscard]] virtual std::map<std::string, Real> diagnostics() const {
        return {};
    }
};

} // namespace q3
