#pragma once

/**
 * WannierConstructionInputs — a record's inputs, checked once and resolved
 * once for one construction.
 *
 * The strategies under subspace/ and gauge/ read the same things: the inputs
 * as the record stores them, the k-mesh stencil built from the grid and the
 * neighbour directions, and the projections and the overlaps as matrices --
 * whole, restricted to the bands a classification keeps, or rotated into a
 * gauge. Resolving them here is what lets a strategy take one argument;
 * refusing a malformed record here, before any strategy runs, is what lets
 * every strategy assume a well-formed one.
 *
 * The stencil and the overlaps are optional because a record imported from
 * another code's projections and eigenvalues alone carries neither -- and
 * the projection gauge on an isolated manifold needs neither. Whatever reads
 * a phase asks spread_is_defined() first.
 *
 * A view over the record it was built from, alive for one build() call; it
 * copies nothing but the projections and the overlaps, which it holds as
 * matrices.
 */

#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_stencil.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/bloch_overlaps.hpp"

namespace q3 {

class WannierConstructionInputs {
  public:
    explicit WannierConstructionInputs(const WannierInputs& record) : record_(record) {
        const int number_of_kpoints = record.number_of_kpoints();
        if (number_of_kpoints == 0) {
            throw std::invalid_argument("WannierConstructionInputs: the inputs hold no k-points");
        }
        if (static_cast<int>(record.eigenvalues_rydberg.size()) != number_of_kpoints ||
            static_cast<int>(record.projection_matrices.size()) != number_of_kpoints) {
            throw std::invalid_argument(
                "WannierConstructionInputs: the eigenvalues and projections do not cover the "
                "inputs' " +
                std::to_string(number_of_kpoints) + " k-points");
        }
        for (int axis = 0; axis < 3; axis++) {
            if (record.kpoint_grid_dimensions[axis] <= 0) {
                throw std::invalid_argument(
                    "WannierConstructionInputs: the inputs do not state the k-point grid their "
                    "matrices were built on");
            }
        }
        /**
         * The model is a Fourier sum over the Wigner-Seitz supercell of the
         * grid, and that transform is exact only over the whole grid: a
         * record holding some other number of k-points is not a smaller
         * problem, it is a wrong one.
         */
        const int grid_points = record.kpoint_grid_dimensions[0] *
                                record.kpoint_grid_dimensions[1] * record.kpoint_grid_dimensions[2];
        if (number_of_kpoints != grid_points) {
            throw std::invalid_argument(
                "WannierConstructionInputs: the inputs hold " + std::to_string(number_of_kpoints) +
                " k-points and state a " + std::to_string(record.kpoint_grid_dimensions[0]) + "x" +
                std::to_string(record.kpoint_grid_dimensions[1]) + "x" +
                std::to_string(record.kpoint_grid_dimensions[2]) + " grid, which has " +
                std::to_string(grid_points));
        }
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            if (static_cast<int>(record.eigenvalues_rydberg[kpoint_index].size()) <
                record.number_of_bands) {
                throw std::invalid_argument("WannierConstructionInputs: k-point " +
                                            std::to_string(kpoint_index) +
                                            " carries fewer eigenvalues than number_of_bands");
            }
            const std::size_t projection_elements =
                static_cast<std::size_t>(record.number_of_bands) *
                static_cast<std::size_t>(record.number_of_trial_orbitals);
            if (record.projection_matrices[kpoint_index].size() != projection_elements) {
                throw std::invalid_argument(
                    "WannierConstructionInputs: the projection at k-point " +
                    std::to_string(kpoint_index) + " holds " +
                    std::to_string(record.projection_matrices[kpoint_index].size()) +
                    " elements, not number_of_bands x number_of_trial_orbitals = " +
                    std::to_string(projection_elements));
            }
            projections_.emplace_back(record.number_of_bands,
                                      record.number_of_trial_orbitals,
                                      record.projection_matrices[kpoint_index]);
        }
        if (!record.neighbor_directions.empty()) {
            stencil_ =
                KPointStencil::from_grid(record.kpoint_grid_dimensions, record.neighbor_directions);
        }
        if (record.has_overlaps()) {
            if (!stencil_) {
                throw std::invalid_argument(
                    "WannierConstructionInputs: the record carries neighbour overlaps but no "
                    "neighbour directions, so nothing says which k-point each overlap reaches");
            }
            overlaps_.emplace(
                record.overlap_matrices, record.number_of_bands, record.number_of_bands);
        }
    }

    /// The inputs as the record stores them.
    [[nodiscard]] const WannierInputs& record() const {
        return record_;
    }
    [[nodiscard]] int number_of_kpoints() const {
        return record_.number_of_kpoints();
    }
    [[nodiscard]] int number_of_wannier_functions() const {
        return record_.number_of_trial_orbitals;
    }

    /// Whether the record names its neighbour directions, so k + b is resolved.
    [[nodiscard]] bool has_stencil() const {
        return stencil_.has_value();
    }
    [[nodiscard]] const KPointStencil& stencil() const {
        if (!stencil_) {
            throw std::logic_error("WannierConstructionInputs: the record carries no neighbour "
                                   "directions, so it has no stencil");
        }
        return *stencil_;
    }
    /// Whether the record carries M_mn(k,b); implies has_stencil().
    [[nodiscard]] bool has_overlaps() const {
        return overlaps_.has_value();
    }
    [[nodiscard]] const BlochOverlaps& overlaps() const {
        if (!overlaps_) {
            throw std::logic_error(
                "WannierConstructionInputs: the record carries no neighbour overlaps");
        }
        return *overlaps_;
    }

    /// A(k) restricted to the bands `classification` keeps at each k-point: kept x N, band as row.
    [[nodiscard]] std::vector<DenseMatrix<Complex>>
    projections_restricted_to(const BandClassification& classification) const {
        std::vector<int> every_orbital(static_cast<std::size_t>(number_of_wannier_functions()));
        std::iota(every_orbital.begin(), every_orbital.end(), 0);
        std::vector<DenseMatrix<Complex>> restricted;
        restricted.reserve(projections_.size());
        for (std::size_t kpoint_index = 0; kpoint_index < projections_.size(); kpoint_index++) {
            restricted.push_back(projections_[kpoint_index].submatrix(
                classification.kpoints[kpoint_index].kept_bands, every_orbital));
        }
        return restricted;
    }

    /**
     * M_mn(k,b) in a gauge: restricted to the bands `classification` keeps
     * and rotated by `frames`, C(k)^dagger M(k,b) C(k+b). What a spread is
     * evaluated on; defined only where has_overlaps().
     */
    [[nodiscard]] BlochOverlaps
    overlaps_in_gauge(const BandClassification& classification,
                      const std::vector<DenseMatrix<Complex>>& frames) const {
        return overlaps()
            .restricted_to(classification.kpoints, stencil())
            .transformed_by(frames, frames, stencil());
    }

    /// The direction of a two-point axis, if any; KPointStencil::two_point_axis_direction says why.
    [[nodiscard]] std::optional<int> two_point_axis_direction() const {
        return stencil_ ? stencil_->two_point_axis_direction() : std::nullopt;
    }

    /**
     * Whether the spread is defined at all: it needs the overlaps, and a
     * mesh without a two-point axis. The subspace iteration is untouched by
     * the two-point rule -- it never reads a phase -- which is why a
     * two-point axis is usable without localisation and not with it.
     */
    [[nodiscard]] bool spread_is_defined() const {
        return overlaps_.has_value() && !two_point_axis_direction().has_value();
    }

  private:
    const WannierInputs& record_;
    std::optional<KPointStencil> stencil_;
    std::optional<BlochOverlaps> overlaps_;
    /// A(k) as the record gives it, number_of_bands x number_of_trial_orbitals, one per k-point.
    std::vector<DenseMatrix<Complex>> projections_;
};

} // namespace q3
