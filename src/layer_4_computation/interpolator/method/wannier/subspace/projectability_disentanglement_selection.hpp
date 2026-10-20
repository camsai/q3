#pragma once

/**
 * ProjectabilityDisentanglementSelection — the N states chosen out of a
 * wider window: classify every band at every k-point by its projectability
 * and its energy (BandClassifier), iterate the subspace that is smoothest
 * across the mesh (SubspaceDisentangler), and fix the projection gauge
 * inside it.
 *
 * The frame handed on is V(k) U(k): the iteration's subspace times the
 * projection gauge of A(k) restricted to it. The initial subspace of the
 * iteration is the orthonormalised column space of the restricted A(k) --
 * which is exactly what the projection gauge computes, so the same kernel
 * serves both the guess and the gauge.
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic.hpp"
#include "layer_4_computation/interpolator/method/wannier/diagnostic_error.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification_summary.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classifier.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_excluded_from_freeze.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/disentanglement_result.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/projection_gauge.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/subspace_disentangler.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"

namespace q3 {

class ProjectabilityDisentanglementSelection : public SubspaceSelection {
  public:
    /// Inverted thresholds are refused here, by the classifier, before any record is read.
    explicit ProjectabilityDisentanglementSelection(ProjectabilityDisentanglementSettings settings)
        : settings_(settings), classifier_(settings.thresholds) {}

    [[nodiscard]] WannierSubspace choose(const WannierConstructionInputs& inputs) const override {
        require_supported(inputs);
        const WannierInputs& record = inputs.record();

        WannierSubspace subspace;
        subspace.classification = classifier_.classify(record);
        log_classification(subspace.classification.summary);

        const std::vector<DenseMatrix<Complex>> projections =
            inputs.projections_restricted_to(subspace.classification);
        const DisentanglementResult result =
            SubspaceDisentangler(inputs.stencil(),
                                 subspace.classification.kpoints,
                                 inputs.number_of_wannier_functions(),
                                 settings_.iteration)
                .run(inputs.overlaps(), ProjectionGauge().compute_all(projections));
        subspace.frames = frames_of(result.subspaces, projections, subspace.classification, record);

        subspace.report.performed = true;
        subspace.report.classification = subspace.classification.summary;
        subspace.report.gauge_invariant_spread_bohr_squared =
            result.gauge_invariant_spread_bohr_squared;
        subspace.report.iterations = result.iterations;
        return subspace;
    }

  private:
    /**
     * Disentanglement reads the overlaps and the projectabilities; a record without them is
     * refused.
     */
    static void require_supported(const WannierConstructionInputs& inputs) {
        if (!inputs.has_overlaps()) {
            throw WannierDiagnosticError(
                WannierDiagnostic::disentanglement_unsupported,
                "ProjectabilityDisentanglementSelection: disentanglement needs the neighbour "
                "overlaps M_mn(k,b), and this record carries none. A record imported from another "
                "code's projections and eigenvalues alone cannot be disentangled; recapture it "
                "with Q3.");
        }
        if (!inputs.record().has_projectabilities()) {
            throw WannierDiagnosticError(
                WannierDiagnostic::disentanglement_unsupported,
                "ProjectabilityDisentanglementSelection: disentanglement classifies bands by "
                "their projectability p_nk = sum_m |A_mn(k)|^2, and this record carries none. "
                "Like the overlaps above this is a capability the record lacks rather than a "
                "value to choose; recapture it with Q3.");
        }
    }

    /// What the classifier decided, and -- never silently -- what it kept out of the freeze.
    static void log_classification(const BandClassificationSummary& summary) {
        log::info("disentanglement: {} ({} band{} crossing the Fermi level); frozen window "
                  "E_F + {:.3f} eV{}",
                  summary.metallic ? "a metal" : "an insulator",
                  summary.number_of_bands_crossing_fermi,
                  summary.number_of_bands_crossing_fermi == 1 ? "" : "s",
                  summary.frozen_window_above_fermi_rydberg * constants::RY_TO_EV,
                  summary.frozen_window_automatic ? " chosen automatically" : " as given");
        for (const BandExcludedFromFreeze& band : summary.bands_excluded_from_freeze) {
            log::warn("disentanglement: band {} lies inside the frozen window at {} k-point{} "
                      "where its projectability is below the minimum, so it was excluded from "
                      "the freeze there rather than frozen -- the trial orbitals cannot "
                      "represent it, and freezing it would make the gauge singular. The model "
                      "has no state of that character at those k-points.",
                      band.band_index + 1,
                      band.number_of_kpoints,
                      band.number_of_kpoints == 1 ? "" : "s");
        }
    }

    /// V(k) U(k): each chosen subspace times the projection gauge of A(k) inside it.
    [[nodiscard]] static std::vector<DenseMatrix<Complex>>
    frames_of(const std::vector<DenseMatrix<Complex>>& chosen_subspaces,
              const std::vector<DenseMatrix<Complex>>& projections,
              const BandClassification& classification,
              const WannierInputs& record) {
        std::vector<DenseMatrix<Complex>> frames;
        frames.reserve(chosen_subspaces.size());
        for (std::size_t kpoint_index = 0; kpoint_index < chosen_subspaces.size(); kpoint_index++) {
            const DenseMatrix<Complex>& chosen = chosen_subspaces[kpoint_index];
            const DenseMatrix<Complex> reduced = multiply(MatrixOperation::ConjugateTranspose,
                                                          chosen,
                                                          MatrixOperation::None,
                                                          projections[kpoint_index]);
            frames.push_back(multiply(MatrixOperation::None,
                                      chosen,
                                      MatrixOperation::None,
                                      gauge_inside(reduced,
                                                   static_cast<int>(kpoint_index),
                                                   classification.kpoints[kpoint_index],
                                                   record)));
        }
        return frames;
    }

    /**
     * The projection gauge of A(k) inside the chosen subspace.
     *
     * A rank-deficient A inside the chosen subspace has one cause that
     * dominates every other, and the gauge cannot see it: a band frozen by
     * the ENERGY rule despite having no projectability puts a direction in
     * the subspace that the trial orbitals cannot reach. Freezing such a
     * state is deliberate -- it is what keeps a metal's Fermi surface in the
     * model -- so it is not refused during classification. But when it does
     * go singular, the k-point and the offending band are what the user
     * needs, and by the time the SVD speaks they are gone.
     */
    [[nodiscard]] static DenseMatrix<Complex> gauge_inside(const DenseMatrix<Complex>& reduced,
                                                           int kpoint_index,
                                                           const KPointBandClassification& entry,
                                                           const WannierInputs& record) {
        try {
            return ProjectionGauge().compute(reduced);
        } catch (const WannierDiagnosticError& error) {
            throw WannierDiagnosticError(
                error.diagnostic(),
                std::string(error.what()) + " This is k-point " + std::to_string(kpoint_index) +
                    " of the disentangled window, where " + std::to_string(entry.number_frozen()) +
                    " of " + std::to_string(entry.number_kept()) + " kept bands are frozen" +
                    likely_cause(kpoint_index, entry, record));
        }
    }

    /**
     * The rest of the sentence above: the least projectable frozen band and
     * what to do about it -- or, where nothing is frozen, that the energy
     * rule is not the cause. Naming a band there would name one that was
     * never frozen, at a projectability nobody measured.
     */
    [[nodiscard]] static std::string likely_cause(int kpoint_index,
                                                  const KPointBandClassification& entry,
                                                  const WannierInputs& record) {
        if (entry.frozen_positions.empty()) {
            return ". No band is frozen there, so the energy rule is not the cause: the subspace "
                   "the iteration chose holds a direction the trial orbitals cannot reach. Raise "
                   "control.interpolation.projectability_minimum so bands they cannot represent "
                   "are not kept, or give the trial orbitals a shell that can represent them.";
        }
        const std::vector<Real>& projectabilities =
            record.projectabilities[static_cast<std::size_t>(kpoint_index)];
        int weakest_band = entry.kept_bands[static_cast<std::size_t>(entry.frozen_positions[0])];
        for (const int position : entry.frozen_positions) {
            const int band = entry.kept_bands[static_cast<std::size_t>(position)];
            if (projectabilities[static_cast<std::size_t>(band)] <
                projectabilities[static_cast<std::size_t>(weakest_band)]) {
                weakest_band = band;
            }
        }
        return " and the least projectable of them is band " + std::to_string(weakest_band + 1) +
               " at p = " +
               std::to_string(projectabilities[static_cast<std::size_t>(weakest_band)]) +
               ". A frozen band with no projectability is a direction the trial orbitals cannot "
               "reach, and it is the usual cause: lower "
               "control.interpolation.frozen_window_above_fermi_ev so the energy rule stops "
               "claiming it, or give the trial orbitals a shell that can represent it.";
    }

    ProjectabilityDisentanglementSettings settings_;
    BandClassifier classifier_;
};

} // namespace q3
