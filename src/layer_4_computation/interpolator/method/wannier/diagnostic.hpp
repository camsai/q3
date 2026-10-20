#pragma once

/**
 * WannierDiagnostic — the failures of the Wannier construction that a
 * driver can act on, as codes beside the prose.
 *
 * The retry ladder in q3.workflows.wannier decides what to try next from
 * WHY an evaluation failed: a disentanglement that ran out of iterations
 * wants a looser tolerance, a window that froze too many bands wants a
 * lower one, a record that cannot be localised wants no retry at all.
 * Reading that out of the error prose would tie a routing decision to the
 * wording of a sentence, so the code is the contract and the prose is for
 * the reader.
 *
 * Every enumerator is spelled exactly as the utility prints it on the
 * `diagnostic:` line after `Error:`, and exactly as the Python side's enum
 * names it; a test on each side reads this file to keep the two equal.
 */

#include <stdexcept>
#include <string>

namespace q3 {

enum class WannierDiagnostic {
    /// The subspace iteration exhausted its budget; the error carries the spreads it reached.
    disentanglement_unconverged,
    /// The record lacks the overlaps or the projectabilities that disentanglement reads.
    disentanglement_unsupported,
    /// A k-point freezes more bands than there are Wannier functions.
    frozen_window_overfull,
    /// A k-point keeps fewer bands than there are Wannier functions.
    kept_bands_underfull,
    /// The trial orbitals do not span the chosen manifold at some k-point.
    gauge_singular,
    /// A diagonal overlap vanished, so a Berry phase -- a centre -- is undefined.
    gauge_collapsed,
    /// The gauge-invariant spread came out negative: the overlaps are not of orthonormal states.
    overlaps_not_orthonormal,
    /// The record lacks the overlaps that localisation minimises.
    localization_unsupported,
    /// The k-mesh has an axis of two points, on which k+b and k-b coincide.
    two_point_axis,
};

/// The enumerator's own spelling, which is what the `diagnostic:` line prints.
[[nodiscard]] inline std::string wannier_diagnostic_name(WannierDiagnostic diagnostic) {
    switch (diagnostic) {
        case WannierDiagnostic::disentanglement_unconverged:
            return "disentanglement_unconverged";
        case WannierDiagnostic::disentanglement_unsupported:
            return "disentanglement_unsupported";
        case WannierDiagnostic::frozen_window_overfull:
            return "frozen_window_overfull";
        case WannierDiagnostic::kept_bands_underfull:
            return "kept_bands_underfull";
        case WannierDiagnostic::gauge_singular:
            return "gauge_singular";
        case WannierDiagnostic::gauge_collapsed:
            return "gauge_collapsed";
        case WannierDiagnostic::overlaps_not_orthonormal:
            return "overlaps_not_orthonormal";
        case WannierDiagnostic::localization_unsupported:
            return "localization_unsupported";
        case WannierDiagnostic::two_point_axis:
            return "two_point_axis";
    }
    throw std::logic_error("wannier_diagnostic_name: an enumerator without a name");
}

} // namespace q3
