#pragma once

/**
 * BandExcludedFromFreeze — a band the energy rule wanted frozen and the
 * singular-gauge rule kept out of the freeze, and at how many k-points.
 *
 * It is a fact about the dataset -- the trial orbitals have none of the
 * character the state needs -- and the one thing about a model that should
 * never be silent, so it travels from the classifier to the log and the
 * results file under names rather than as a pair whose order a reader has to
 * look up.
 */

namespace q3 {

struct [[nodiscard]] BandExcludedFromFreeze {
    /// Zero-based, within the captured window.
    int band_index = 0;
    int number_of_kpoints = 0;
};

} // namespace q3
