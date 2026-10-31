#pragma once

/**
 * The band edges either side of the Fermi energy, and the gap between them,
 * read off a run's eigenvalues_at_kpoints record.
 *
 * Both are found from the eigenvalues and the Fermi energy alone, so they are
 * worked out once from the records a flavor listed rather than by each
 * flavor: add_band_edge_records() adds ESSE's band_edge_valence_max,
 * band_edge_conduction_min and band_gaps records beside them.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "generated/esse/properties/non_scalar/band_gaps.hpp"
#include "generated/esse/properties/non_scalar/eigenvalues_at_kpoints.hpp"
#include "generated/esse/properties/scalar/band_edge_conduction_min.hpp"
#include "generated/esse/properties/scalar/band_edge_valence_max.hpp"
#include "generated/esse/properties/scalar/fermi_energy.hpp"
#include "io/output/results.hpp"
#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * ESSE's band_edge_valence_max for `eigenvalues`: the highest level at or below
 * `fermi_energy_in_ry`, at the k-point of the last state holding it when the
 * record lists k-points; nothing when no level lies there.
 *
 * The divider is the Fermi energy itself, not occupation: in an insulator it
 * sits in the gap, and this is the valence-band maximum; in a metal it lies
 * inside a band, and with band_edge_conduction_min() it brackets the local
 * level spacing.
 */
[[nodiscard]] inline std::optional<generated::BandEdgeValenceMax>
band_edge_valence_max(const generated::EigenvaluesAtKpoints& eigenvalues, Real fermi_energy_in_ry) {
    std::optional<generated::BandEdgeValenceMax> edge;
    for (const auto& kpoint : eigenvalues.values) {
        for (const auto& channel : kpoint.eigenvalues) {
            for (const Real energy_in_ry : channel.energies) {
                if (energy_in_ry <= fermi_energy_in_ry && (!edge || energy_in_ry >= edge->value)) {
                    edge = generated::BandEdgeValenceMax{
                        generated::BandEdgeValenceMaxNameEnum::BAND_EDGE_VALENCE_MAX,
                        kpoint.kpoint,
                        std::nullopt,
                        generated::BandEdgeValenceMaxUnitsEnum::RY,
                        energy_in_ry};
                }
            }
        }
    }
    return edge;
}

/// ESSE's band_edge_conduction_min: the lowest level above the Fermi energy, as above.
[[nodiscard]] inline std::optional<generated::BandEdgeConductionMin>
band_edge_conduction_min(const generated::EigenvaluesAtKpoints& eigenvalues,
                         Real fermi_energy_in_ry) {
    std::optional<generated::BandEdgeConductionMin> edge;
    for (const auto& kpoint : eigenvalues.values) {
        for (const auto& channel : kpoint.eigenvalues) {
            for (const Real energy_in_ry : channel.energies) {
                if (energy_in_ry > fermi_energy_in_ry && (!edge || energy_in_ry <= edge->value)) {
                    edge = generated::BandEdgeConductionMin{
                        generated::BandEdgeConductionMinNameEnum::BAND_EDGE_CONDUCTION_MIN,
                        kpoint.kpoint,
                        std::nullopt,
                        generated::BandEdgeConductionMinUnitsEnum::RY,
                        energy_in_ry};
                }
            }
        }
    }
    return edge;
}

/**
 * Whether any level's occupation lies strictly between empty and full, by more
 * than the tail a smeared insulator leaves on its band edges: the run's own
 * statement that it found a metal.
 */
[[nodiscard]] inline bool
is_fractionally_occupied(const generated::EigenvaluesAtKpoints& eigenvalues) {
    constexpr Real occupation_tolerance = 1.0e-6;
    for (const auto& kpoint : eigenvalues.values) {
        for (const auto& channel : kpoint.eigenvalues) {
            for (const Real occupation : channel.occupations.value_or(std::vector<double>{})) {
                if (occupation > occupation_tolerance && occupation < 1.0 - occupation_tolerance) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// One ESSE band gap between two levels, with the k-points they lie at where known.
[[nodiscard]] inline generated::BandGapsBandGap
band_gap(generated::BandGapsBandGapTypeEnum type,
         Real valence_in_ry,
         Real conduction_in_ry,
         const std::optional<std::vector<double>>& valence_kpoint,
         const std::optional<std::vector<double>>& conduction_kpoint) {
    generated::BandGapsBandGap gap;
    gap.type = type;
    gap.units = generated::BandGapsBandGapUnitsEnum::RY;
    gap.value = conduction_in_ry - valence_in_ry;
    gap.eigenvalue_valence = valence_in_ry;
    gap.eigenvalue_conduction = conduction_in_ry;
    gap.kpoint_valence = valence_kpoint;
    gap.kpoint_conduction = conduction_kpoint;
    return gap;
}

/**
 * ESSE's band_gaps for `eigenvalues` about `fermi_energy_in_ry`: the "indirect"
 * entry is the fundamental gap between the two band edges, wherever they sit,
 * and the "direct" entry is the smallest vertical gap at any single k-point.
 *
 * Nothing where there is no gap: both edges have to exist, and no level may
 * be fractionally occupied. In a smeared metal the edges are the local level
 * spacing at E_F (0.018 Ry on the parity matrix's copper), which a band_gaps
 * record would state as the material's gap.
 */
[[nodiscard]] inline std::optional<generated::BandGaps>
band_gaps(const generated::EigenvaluesAtKpoints& eigenvalues, Real fermi_energy_in_ry) {
    const auto valence = band_edge_valence_max(eigenvalues, fermi_energy_in_ry);
    const auto conduction = band_edge_conduction_min(eigenvalues, fermi_energy_in_ry);
    if (!valence || !conduction || is_fractionally_occupied(eigenvalues)) {
        return std::nullopt;
    }

    std::optional<generated::BandGapsBandGap> direct;
    for (const auto& kpoint : eigenvalues.values) {
        Real kpoint_valence_in_ry = -std::numeric_limits<Real>::infinity();
        Real kpoint_conduction_in_ry = std::numeric_limits<Real>::infinity();
        for (const auto& channel : kpoint.eigenvalues) {
            for (const Real energy_in_ry : channel.energies) {
                if (energy_in_ry <= fermi_energy_in_ry) {
                    kpoint_valence_in_ry = std::max(kpoint_valence_in_ry, energy_in_ry);
                } else {
                    kpoint_conduction_in_ry = std::min(kpoint_conduction_in_ry, energy_in_ry);
                }
            }
        }
        const bool has_both_edges =
            std::isfinite(kpoint_valence_in_ry) && std::isfinite(kpoint_conduction_in_ry);
        if (has_both_edges &&
            (!direct || kpoint_conduction_in_ry - kpoint_valence_in_ry < direct->value)) {
            direct = band_gap(generated::BandGapsBandGapTypeEnum::DIRECT,
                              kpoint_valence_in_ry,
                              kpoint_conduction_in_ry,
                              kpoint.kpoint,
                              kpoint.kpoint);
        }
    }

    generated::BandGaps record;
    record.name = generated::BandGapsNameEnum::BAND_GAPS;
    if (direct) {
        record.values.push_back(*direct);
    }
    record.values.push_back(band_gap(generated::BandGapsBandGapTypeEnum::INDIRECT,
                                     valence->value,
                                     conduction->value,
                                     valence->kpoint,
                                     conduction->kpoint));
    return record;
}

/**
 * Add the band edges and the gap between them to `results`, worked out from
 * its eigenvalues_at_kpoints and fermi_energy records: each edge where such a
 * level exists, and band_gaps where there is a gap. Adds nothing when either
 * record is missing.
 */
inline void add_band_edge_records(generated::Results& results) {
    const auto eigenvalues = find_record<generated::EigenvaluesAtKpoints>(results);
    const auto fermi_energy = find_record<generated::FermiEnergy>(results);
    if (!eigenvalues || !fermi_energy) {
        return;
    }
    if (const auto valence = band_edge_valence_max(*eigenvalues, fermi_energy->value)) {
        set_record(results, *valence);
    }
    if (const auto conduction = band_edge_conduction_min(*eigenvalues, fermi_energy->value)) {
        set_record(results, *conduction);
    }
    if (const auto gaps = band_gaps(*eigenvalues, fermi_energy->value)) {
        set_record(results, *gaps);
    }
}

} // namespace q3
