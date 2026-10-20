#pragma once

/**
 * PlanewaveChargeDensity — planewave representation of charge density.
 *
 * Stores spin-resolved charge density on real-space FFT grids with
 * reciprocal-space representation capability. This is a data container;
 * computation is done by ChargeDensityCalculator subclasses.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_2_quantum_system/derived/charge_density.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/mixins/ff_transformable.hpp"

namespace q3 {

/// Type aliases for charge density storage
using SpinResolvedRealSpaceGrid = std::vector<std::vector<Real>>;
using ReciprocalSpaceCoefficients = std::vector<Complex>;

/// Planewave charge density — stores ρ(r) on FFT grid.
class PlanewaveChargeDensity
    : public ChargeDensity,
      public FFTransformable<SpinResolvedRealSpaceGrid, ReciprocalSpaceCoefficients> {
  public:
    PlanewaveChargeDensity() = default;
    explicit PlanewaveChargeDensity(const PlanewaveBasis& basis, int number_of_spin_components)
        : basis_(&basis) {
        number_of_spin_components_ = number_of_spin_components;
        int grid_points = basis.fft_rho().total_grid_points();
        real_space_data_.resize(number_of_spin_components);
        for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
            real_space_data_[spin_index].assign(grid_points, 0.0);
        }
        reciprocal_space_data_.resize(basis.number_of_g_vectors(), Complex(0.0, 0.0));
    }

    // ─── ChargeDensity interface ────────────────────────────────
    [[nodiscard]] Real get_total_charge() const override {
        if (!basis_ || cell_volume_ < math::tolerances::cell_volume_threshold)
            return 0.0;
        int grid_points = basis_->fft_rho().total_grid_points();
        Real charge = 0.0;
        for (int spin_index = 0; spin_index < number_of_spin_components_; spin_index++) {
            for (int grid_point_index = 0; grid_point_index < grid_points; grid_point_index++) {
                charge += real_space_data_[spin_index][grid_point_index];
            }
        }
        return charge * cell_volume_ / grid_points;
    }
    void reset() override {
        int grid_points = basis_ ? basis_->fft_rho().total_grid_points() : 0;
        for (int spin_index = 0; spin_index < number_of_spin_components_; spin_index++) {
            real_space_data_[spin_index].assign(grid_points, 0.0);
        }
        if (!reciprocal_space_data_.empty()) {
            std::fill(
                reciprocal_space_data_.begin(), reciprocal_space_data_.end(), Complex(0.0, 0.0));
        }
    }

    // ─── Spin-resolved access ───────────────────────────────────
    std::vector<Real>& real_space_by_spin_index(int spin_index) {
        return real_space_data_[spin_index];
    }
    [[nodiscard]] const std::vector<Real>& real_space_by_spin_index(int spin_index) const {
        return real_space_data_[spin_index];
    }

    // ─── FFT transforms ────────────────────────────────────────
    void to_reciprocal_space() override {
        // FFT total density to G-space
        int grid_points = basis_->fft_rho().total_grid_points();
        std::vector<Real> total_density(grid_points, 0.0);
        for (int spin_index = 0; spin_index < number_of_spin_components_; spin_index++) {
            for (int grid_point_index = 0; grid_point_index < grid_points; grid_point_index++) {
                total_density[grid_point_index] += real_space_data_[spin_index][grid_point_index];
            }
        }
        basis_->rho_r_to_g(total_density, reciprocal_space_data_);
    }
    void to_real_space() override {
        // FFT total density from G-space to R-space (only for single spin)
        if (number_of_spin_components_ == 1) {
            basis_->rho_g_to_r(reciprocal_space_data_, real_space_data_[0]);
        }
    }

    // ─── Band-by-band accumulation ─────────────────────────────
    void add_band_contribution(int spin_index,
                               Real weight_times_occupation,
                               const Complex* psi_real_space,
                               int real_space_grid_size,
                               Real inverse_cell_volume) {
        for (int grid_point_index = 0; grid_point_index < real_space_grid_size;
             grid_point_index++) {
            real_space_data_[spin_index][grid_point_index] +=
                weight_times_occupation * std::norm(psi_real_space[grid_point_index]) *
                inverse_cell_volume;
        }
    }

    // ─── Smooth copy management ────────────────────────────────
    void save_smooth_copy() {
        smooth_real_space_data_ = real_space_data_;
    }
    [[nodiscard]] const SpinResolvedRealSpaceGrid& smooth_real_space_data() const {
        return smooth_real_space_data_;
    }
    SpinResolvedRealSpaceGrid& smooth_real_space_data() {
        return smooth_real_space_data_;
    }

    // ─── MPI ───────────────────────────────────────────────────

    // ─── Basis access ──────────────────────────────────────────
    [[nodiscard]] const PlanewaveBasis& basis() const {
        return *basis_;
    }
    [[nodiscard]] int grid_size() const {
        return basis_ ? basis_->fft_rho().total_grid_points() : 0;
    }
    [[nodiscard]] Real cell_volume() const {
        return cell_volume_;
    }
    void set_cell_volume(Real volume) {
        cell_volume_ = volume;
    }

  private:
    const PlanewaveBasis* basis_ = nullptr;
    Real cell_volume_ = 0.0;
    SpinResolvedRealSpaceGrid smooth_real_space_data_;
};

} // namespace q3
