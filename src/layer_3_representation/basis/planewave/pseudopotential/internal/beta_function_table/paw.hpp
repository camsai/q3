#pragma once

/**
 * PawBetaFunctionTable — β(q) tabulation for sets containing PAW datasets
 * that supply their projectors in reciprocal space.
 *
 * Such a dataset carries no radial projector at all, so the base table's
 * Bessel transform would integrate an empty array and tabulate zeros — the
 * nonlocal term silently vanishing. This subclass resamples the dataset's own
 * β(q) table instead, which is also more faithful than any radial round trip:
 * the tabulated range ends far above every physical cutoff, while inverting a
 * 100-point table truncates.
 *
 * The override is per projector and falls back to the base implementation for
 * any species that supplies projectors radially, so mixed sets — UPF species
 * alongside reciprocally tabulated ones — tabulate each species the way its dataset
 * intends. This file is the one place plane-wave tabulation code is allowed
 * to know the PAW type; everything shared stays flavor-blind (see the base
 * class header).
 */

#include <vector>

#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/reciprocal_projector_resampling.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"

namespace q3 {

class PawBetaFunctionTable : public BetaFunctionTable {
  protected:
    void tabulate_projector(const Pseudopotential& pseudopotential,
                            int projector_index,
                            const std::vector<Real>& q_point_values,
                            std::vector<Real>& destination) const override {
        const auto* paw = dynamic_cast<const PawPseudopotential*>(&pseudopotential);
        if (paw != nullptr && paw->has_reciprocal_projectors() &&
            projector_index < static_cast<int>(paw->projectors_reciprocal.size())) {
            resample_uniform_reciprocal_table(paw->projectors_reciprocal[projector_index],
                                              paw->reciprocal_grid_spacing,
                                              paw->projectors[projector_index].angular_momentum,
                                              q_point_values,
                                              destination);
            return;
        }
        BetaFunctionTable::tabulate_projector(
            pseudopotential, projector_index, q_point_values, destination);
    }
};

} // namespace q3
