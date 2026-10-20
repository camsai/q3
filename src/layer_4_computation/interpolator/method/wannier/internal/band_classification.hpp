#pragma once

/**
 * BandClassification — what the classifier decided: at each k-point, which
 * bands are kept exactly, which compete for the remaining room, and which are
 * thrown away; and the verdict on the system as a whole.
 *
 * Plain data. The rule that produces it is BandClassifier, beside this file;
 * the subspace iteration and the overlaps' restriction read it.
 */

#include <cstddef>
#include <vector>

#include "layer_4_computation/interpolator/method/wannier/internal/band_classification_summary.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_role.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/kpoint_band_classification.hpp"

namespace q3 {

/// Every k-point's verdict, and the summary of the whole.
struct [[nodiscard]] BandClassification {
    std::vector<KPointBandClassification> kpoints;
    BandClassificationSummary summary;

    /**
     * The classification of a window that keeps everything and freezes
     * nothing: an isolated manifold, where the bands ARE the states. It lets
     * the gauge, the composition and the localisation run as one code path
     * whether the subspace was chosen or given.
     */
    [[nodiscard]] static BandClassification every_band_kept(int number_of_kpoints,
                                                            int number_of_bands) {
        KPointBandClassification entry;
        entry.roles.assign(static_cast<std::size_t>(number_of_bands), BandRole::Candidate);
        for (int band = 0; band < number_of_bands; band++) {
            entry.kept_bands.push_back(band);
        }
        BandClassification classification;
        classification.kpoints.assign(static_cast<std::size_t>(number_of_kpoints), entry);
        return classification;
    }
};

} // namespace q3
