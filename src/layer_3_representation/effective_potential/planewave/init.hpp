#pragma once

/**
 * PlanewaveEffectivePotential — planewave representation of the effective potential.
 *
 * Stores the total effective potential in real and reciprocal space, plus
 * XC-specific reciprocal-space data and optional kinetic potential (meta-GGA).
 *
 * This is the canonical Layer 3 representation of the effective KS potential.
 * The Layer 4 calculator populates this class rather than using a raw struct.
 */

#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_2_quantum_system/derived/effective_potential.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/mixins/ff_transformable.hpp"

namespace q3 {

/// Type aliases for potential storage
using SpinResolvedPotentialGrid = std::vector<std::vector<Real>>;
using PotentialReciprocalCoefficients = std::vector<Complex>;

/// Planewave effective potential — stores V(r) on FFT grid.
class PlanewaveEffectivePotential
    : public EffectivePotential,
      public FFTransformable<SpinResolvedPotentialGrid, PotentialReciprocalCoefficients> {
  public:
    PlanewaveEffectivePotential() = default;
    explicit PlanewaveEffectivePotential(const PlanewaveBasis& basis, int number_of_spin_components)
        : basis_(&basis) {
        number_of_spin_components_ = number_of_spin_components;
        int grid_points = basis.fft_rho().total_grid_points();
        real_space_data_.resize(number_of_spin_components);
        for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
            real_space_data_[spin_index].assign(grid_points, 0.0);
        }
        int number_of_g_vectors = basis.number_of_g_vectors();
        reciprocal_space_data_.resize(number_of_g_vectors, Complex(0.0, 0.0));
        exchange_correlation_reciprocal_space_data_.resize(number_of_g_vectors, Complex(0.0, 0.0));
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
        // FFT total potential to G-space
        if (number_of_spin_components_ == 1) {
            basis_->rho_r_to_g(real_space_data_[0], reciprocal_space_data_);
        } else {
            // For spin-polarized, FFT total (sum of spin channels)
            int grid_points = basis_->fft_rho().total_grid_points();
            std::vector<Real> total_potential(grid_points, 0.0);
            for (int spin_index = 0; spin_index < number_of_spin_components_; spin_index++) {
                for (int grid_point_index = 0; grid_point_index < grid_points; grid_point_index++) {
                    total_potential[grid_point_index] +=
                        real_space_data_[spin_index][grid_point_index];
                }
            }
            basis_->rho_r_to_g(total_potential, reciprocal_space_data_);
        }
    }
    void to_real_space() override {
        if (number_of_spin_components_ == 1) {
            basis_->rho_g_to_r(reciprocal_space_data_, real_space_data_[0]);
        }
    }

    // ─── XC reciprocal space ───────────────────────────────────
    std::vector<Complex>& exchange_correlation_reciprocal_space_data() {
        return exchange_correlation_reciprocal_space_data_;
    }
    [[nodiscard]] const std::vector<Complex>& exchange_correlation_reciprocal_space_data() const {
        return exchange_correlation_reciprocal_space_data_;
    }

    // ─── Kinetic potential (meta-GGA) ──────────────────────────
    std::vector<std::vector<Real>>& kinetic_potential_data() {
        return kinetic_potential_data_;
    }
    [[nodiscard]] const std::vector<std::vector<Real>>& kinetic_potential_data() const {
        return kinetic_potential_data_;
    }
    void initialize_kinetic_potential() {
        int grid_points = basis_ ? basis_->fft_rho().total_grid_points() : 0;
        kinetic_potential_data_.resize(number_of_spin_components_);
        for (int spin_index = 0; spin_index < number_of_spin_components_; spin_index++) {
            kinetic_potential_data_[spin_index].assign(grid_points, 0.0);
        }
    }

    // ─── Energy results (populated by calculator) ──────────────
    Real hartree_energy = 0.0;
    Real exchange_correlation_energy = 0.0;
    Real exchange_correlation_potential_energy = 0.0;

    // ─── Charge density G-space (set by calculator) ────────────
    std::vector<Complex>& charge_density_reciprocal() {
        return charge_density_reciprocal_;
    }
    [[nodiscard]] const std::vector<Complex>& charge_density_reciprocal() const {
        return charge_density_reciprocal_;
    }

    // ─── Basis access ──────────────────────────────────────────
    [[nodiscard]] const PlanewaveBasis& basis() const {
        return *basis_;
    }
    [[nodiscard]] int grid_size() const {
        return basis_ ? basis_->fft_rho().total_grid_points() : 0;
    }

    // ─── Logging ───────────────────────────────────────────────
    void log_summary() const {
        log::info("  Effective potential summary:");
        log::info("    grid points            = {}", grid_size());
        log::info("    spin components        = {}", number_of_spin_components_);
        log::info("    Hartree energy         = {:.9f} Ry", hartree_energy);
        log::info("    XC energy              = {:.9f} Ry", exchange_correlation_energy);
    }

  private:
    const PlanewaveBasis* basis_ = nullptr;
    std::vector<Complex> exchange_correlation_reciprocal_space_data_;
    std::vector<std::vector<Real>> kinetic_potential_data_;
    std::vector<Complex> charge_density_reciprocal_;
};

} // namespace q3
