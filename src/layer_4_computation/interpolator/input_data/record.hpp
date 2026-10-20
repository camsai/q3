#pragma once

/**
 * InterpolationRecord — what an NSCF run leaves behind for the interpolation
 * step, as plain data: which method built it, the lattice it lives on, and
 * the method's own payload.
 *
 * Mirrors MeanFieldRecord (io/mean_field_record/init.hpp): purely
 * descriptive, so the same record can come from a live run or be read back
 * from `<prefix>.save/interpolation.q3ip`, and the same interpolator code
 * consumes it either way. The lattice is carried here rather than looked up
 * from the crystal because the evaluation step has no crystal -- it has this
 * file -- and both the Wigner-Seitz construction and the Wannier90 export
 * need the real-space vectors.
 *
 * One payload member per method, populated only for the method named. A
 * variant would express the exclusivity in the type; with one method that is
 * ceremony, and the factory beside this file dispatches on `method` in one
 * place, which is where a second payload will be added alongside its member.
 */

#include <string>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"

namespace q3 {

struct [[nodiscard]] InterpolationRecord {
    /// The method that built the payload: "wannier".
    std::string method;

    /// Rows a1, a2, a3, Cartesian, in bohr.
    Matrix3x3 real_lattice_bohr;
    /// Rows b1, b2, b3, Cartesian, in inverse bohr, with the 2 pi included.
    Matrix3x3 reciprocal_lattice_inverse_bohr;

    /**
     * The run's Fermi energy, so an evaluation can report bands relative to
     * it without knowing which payload carries it. Zero when the record was
     * imported from files that do not state one.
     */
    Real fermi_energy_rydberg = 0.0;

    /// Payload for method == "wannier": interface matrices on the full k-grid.
    WannierInputs wannier;
};

} // namespace q3
