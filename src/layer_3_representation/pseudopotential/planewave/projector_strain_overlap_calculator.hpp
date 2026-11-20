#pragma once

/**
 * ProjectorStrainOverlapCalculator — the overlaps of bands with the
 * Kleinman-Bylander projectors and with their six strain derivatives.
 *
 * What the nonlocal stress contracts with the D matrix. For a block of
 * consecutive atoms it yields, per band, the overlap <β|ψ> of every projector
 * of those atoms and the overlap of each projector's derivative along each
 * strain component. How the overlaps are formed is the implementation's: the
 * host one builds the projectors and their derivatives as columns and
 * multiplies them into the bands, and a device one never builds the columns at
 * all. The stress is the same code either way.
 *
 * The strain derivative of a projector along (a, b) has a radial part,
 * dβ/dq (k+G)_a (k+G)_b / |k+G| Y_lm, and an angular part,
 * β ∂Y_lm/∂(k+G)_a (k+G)_b; the two are summed into one derivative per
 * component.
 *
 * Layout of the overlaps, column-major, one column per band: with P the number
 * of projectors of the block, rows [0, P) are <β|ψ> in the block's projector
 * order (atom by atom, each atom's beta functions in order, each with its 2l+1
 * components), and rows [(s+1)·P, (s+2)·P) are the derivatives along strain
 * component s.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_strain_inputs.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"

namespace q3 {

/// Overlaps of bands with projectors and their strain derivatives; see the file comment.
class ProjectorStrainOverlapCalculator {
  public:
    /// The projector itself and its derivative along each of the six strain components.
    static constexpr int number_of_kinds = 7;
    /// First Cartesian direction of each strain component, in the order xx, xy, xz, yy, yz, zz.
    static constexpr int strain_direction_a[6] = {0, 0, 0, 1, 1, 2};
    /// Second Cartesian direction of each strain component, in the same order.
    static constexpr int strain_direction_b[6] = {0, 1, 2, 1, 2, 2};

    virtual ~ProjectorStrainOverlapCalculator() = default;

    /**
     * Take the k-point that the blocks computed next belong to. What `inputs` points at must
     * stay as it is until the last of those blocks has been computed and read.
     *
     * @param inputs                        The k-point's atom-independent inputs.
     * @param maximum_projectors_per_block  The largest number of projectors a block will hold.
     */
    virtual void begin_kpoint(const ProjectorStrainInputs& inputs,
                              int maximum_projectors_per_block) = 0;

    /**
     * The overlaps of the chosen bands with the projectors of the atoms [first_atom, end_atom)
     * and with their strain derivatives, in the layout of the file comment:
     * number_of_kinds × number_of_projectors rows by band_indices.size() columns.
     *
     * @param number_of_projectors  The projectors of those atoms together.
     * @param band_indices          Band indices of the k-point state, in ascending order.
     */
    virtual void compute_block(int first_atom,
                               int end_atom,
                               int number_of_projectors,
                               const PlanewaveKohnShamQuantumSystem& quantum_system,
                               int kpoint_state_index,
                               const std::vector<int>& band_indices,
                               Complex* overlaps) = 0;

    /**
     * The value at one plane wave of each projector and derivative of the block computed
     * last, in the row order of its overlaps. The gamma-only stress needs the G = 0 values to
     * take out the term its half-sphere sum counts twice.
     */
    virtual void block_values_at_plane_wave(int plane_wave_index,
                                            std::vector<Complex>& values) const = 0;
};

} // namespace q3
