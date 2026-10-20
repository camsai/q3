#pragma once

/**
 * PlanewaveLocalOrbitalSet — a set of localized orbitals RENDERED in the
 * plane-wave basis at one k-point.
 *
 * A local orbital is an abstract object: a centre, an angular character, a
 * radial form. This interface is what it looks like from inside the
 * plane-wave representation -- a vector of coefficients over the k-point's
 * G-vectors, for each orbital in the set -- which is the form every
 * plane-wave-side consumer needs: projections <psi_nk|g>, overlaps between
 * orbitals, the initial guess of a Wannier construction.
 *
 * Implementations differ only in where the orbitals come from:
 * PlanewavePseudoAtomicOrbitalSet reads them from the pseudopotential's
 * tabulated atomic wavefunctions, PlanewaveHydrogenicOrbitalSet places
 * analytic s orbitals at arbitrary centres. A consumer written against this
 * interface serves both, and any future source -- Gaussians, numerical
 * atomic orbitals from a basis-set file -- without change. That is the
 * point of the seam: the CHOICE of orbitals is a strategy, made once by the
 * factory beside this file, not a branch repeated in every consumer.
 */

#include <string>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class PlanewaveLocalOrbitalSet {
  public:
    virtual ~PlanewaveLocalOrbitalSet() = default;

    [[nodiscard]] virtual int number_of_orbitals() const = 0;

    /**
     * Render every orbital at one k-point: on return
     * `orbitals[orbital_index][g_index]` holds the coefficient of the
     * `g_vector_indices[g_index]`-th plane wave, so the result is stored
     * against exactly the G-vector list the wavefunctions at that k-point
     * use and can be contracted with them directly.
     *
     * Normalisation is left to the implementation; consumers that need
     * orthonormal orbitals in a given metric orthonormalize them.
     */
    virtual void build(int kpoint_index,
                       int number_of_plane_waves,
                       const std::vector<int>& g_vector_indices,
                       const Crystal& crystal_structure,
                       const KPointsGrid& kpoint_grid,
                       const PlanewaveBasis& planewave_basis,
                       std::vector<std::vector<Complex>>& orbitals) const = 0;

    /// One line for logs, naming the source and the count.
    [[nodiscard]] virtual std::string description() const = 0;

    /**
     * Where each orbital sits, in crystal (fractional) coordinates, in orbital
     * order. A localiser needs these: the branch of Im ln M_nn is fixed by
     * choosing, for each diagonal overlap, the value nearest the one a known
     * centre implies, and the trial orbital's own position is the one thing
     * known about the Wannier function before anything has been minimised.
     * Deriving a guide from the projection gauge's Berry-phase centres instead
     * was tried and fails on exactly the structures that need guiding -- when
     * the branches are inconsistent across k, those centres are already wrong.
     */
    [[nodiscard]] virtual std::vector<Vector3D> centers_crystal() const = 0;
};

} // namespace q3
