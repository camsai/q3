#pragma once

/**
 * LibXC-based exchange-correlation functional wrapper.
 *
 * Wraps the LibXC library to evaluate XC energy densities and potentials
 * for LDA and GGA functionals. Supports both unpolarized (number_of_spin_components=1) and
 * spin-polarized (number_of_spin_components=2) calculations.
 *
 * Density thresholds are matched to QE's behavior (1e-10) to ensure
 * numerical agreement in energy comparisons.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <xc.h>

#include "layer_0_core/execution/threading.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"

namespace q3 {

/**
 * LibXC-based functional. Wraps supported functionals from the LibXC library.
 * Supports LDA and GGA families. Specified by exchange + correlation IDs.
 * Can operate in both unpolarized (number_of_spin_components=1) and spin-polarized
 * (number_of_spin_components=2) modes.
 */
class XC_LibXC : public XCFunctional {
  public:
    /**
     * Construct from LibXC functional IDs.
     * number_of_spin_components: 1=unpolarized, 2=spin-polarized (LSDA/SGGA)
     */
    XC_LibXC(int exchange_id,
             int correlation_id,
             int number_of_spin_components = 1,
             std::string functional_identifier = "") {
        exchange_id_ = exchange_id;
        correlation_id_ = correlation_id;
        number_of_spin_components_ = number_of_spin_components;

        if (functional_identifier.empty()) {
            functional_identifier =
                "libxc:" + std::to_string(exchange_id) + ":" + std::to_string(correlation_id);
        }

        int libxc_spin_polarization =
            (number_of_spin_components == 2) ? XC_POLARIZED : XC_UNPOLARIZED;

        // Initialize exchange functional
        if (xc_func_init(&func_x_, exchange_id, libxc_spin_polarization) != 0) {
            throw std::runtime_error("LibXC: failed to initialize exchange functional id=" +
                                     std::to_string(exchange_id));
        }

        // Initialize correlation functional
        if (xc_func_init(&func_c_, correlation_id, libxc_spin_polarization) != 0) {
            xc_func_end(&func_x_);
            throw std::runtime_error("LibXC: failed to initialize correlation functional id=" +
                                     std::to_string(correlation_id));
        }

        int exchange_family = func_x_.info->family;
        int correlation_family = func_c_.info->family;

        try {
            ensure_supported_family(
                functional_identifier, "exchange", exchange_id, func_x_, exchange_family);
            ensure_supported_family(
                functional_identifier, "correlation", correlation_id, func_c_, correlation_family);
        } catch (...) {
            xc_func_end(&func_c_);
            xc_func_end(&func_x_);
            throw;
        }

        exchange_requires_gradient_ = exchange_family == XC_FAMILY_GGA;
        correlation_requires_gradient_ = correlation_family == XC_FAMILY_GGA;

        // Set LibXC internal density thresholds to match QE's behavior.

        // QE sets dens_threshold = 1e-10 ("small") for correlation

        // and dens_threshold = grho_threshold_gga = 1e-10 for exchange.
        constexpr double libxc_density_threshold = 1e-10;
        xc_func_set_dens_threshold(&func_x_, libxc_density_threshold);
        xc_func_set_dens_threshold(&func_c_, libxc_density_threshold);

        functional_name_ = std::string(func_x_.info->name) + "+" + std::string(func_c_.info->name);
    }

    ~XC_LibXC() override {
        xc_func_end(&func_x_);
        xc_func_end(&func_c_);
    }

    [[nodiscard]] bool requires_density_gradient() const override {
        return exchange_requires_gradient_ || correlation_requires_gradient_;
    }

    [[nodiscard]] int number_of_spin_components() const override {
        return number_of_spin_components_;
    }

    [[nodiscard]] std::string name() const override {
        return functional_name_;
    }

    // ─── Unpolarized LDA ────────────────────────────────────────────────
    void compute(const std::vector<Real>& density,
                 std::vector<Real>& xc_potential,
                 std::vector<Real>& xc_energy_density) const override {
        int number_of_grid_points = static_cast<int>(density.size());
        xc_potential.resize(number_of_grid_points);
        xc_energy_density.resize(number_of_grid_points);

        std::vector<Real> exchange_energies(number_of_grid_points);
        std::vector<Real> correlation_energies(number_of_grid_points);
        std::vector<Real> exchange_potentials(number_of_grid_points);
        std::vector<Real> correlation_potentials(number_of_grid_points);

        std::vector<Real> absolute_density(number_of_grid_points);

        // Cast-away const to pass to LibXC functions which take non-const xc_func_type*
        auto* non_const_exchange = const_cast<xc_func_type*>(&func_x_);
        auto* non_const_correlation = const_cast<xc_func_type*>(&func_c_);

        for_each_grid_slice(number_of_grid_points, [&](int offset, int count) {
            for (int grid_point_index = offset; grid_point_index < offset + count;
                 grid_point_index++) {
                absolute_density[grid_point_index] = std::abs(density[grid_point_index]);
            }
            xc_lda_exc_vxc(non_const_exchange,
                           count,
                           absolute_density.data() + offset,
                           exchange_energies.data() + offset,
                           exchange_potentials.data() + offset);
            xc_lda_exc_vxc(non_const_correlation,
                           count,
                           absolute_density.data() + offset,
                           correlation_energies.data() + offset,
                           correlation_potentials.data() + offset);
            for (int grid_point_index = offset; grid_point_index < offset + count;
                 grid_point_index++) {
                xc_energy_density[grid_point_index] =
                    constants::electron_charge_squared_rydberg *
                    (exchange_energies[grid_point_index] + correlation_energies[grid_point_index]);
                xc_potential[grid_point_index] = constants::electron_charge_squared_rydberg *
                                                 (exchange_potentials[grid_point_index] +
                                                  correlation_potentials[grid_point_index]);
            }
        });

        /**
         * The finiteness check is a serial pass over the finished arrays
         * rather than part of the slice body, because an exception that
         * escapes a parallel region terminates the process. Evaluating every
         * point before reporting changes nothing a caller can observe: the
         * evaluation has no side effects, so the point reported and the
         * numbers reported with it are the same either way.
         */
        for (int grid_point_index = 0; grid_point_index < number_of_grid_points;
             grid_point_index++) {
            if (!all_finite(
                    {xc_energy_density[grid_point_index], xc_potential[grid_point_index]})) {
                throw_non_finite_output("unpolarized LDA",
                                        grid_point_index,
                                        {{"rho", absolute_density[grid_point_index]}});
            }
        }
    }

    // ─── Spin-polarized LDA (LSDA) ─────────────────────────────────────
    void compute_spin(const std::vector<Real>& density_up,
                      const std::vector<Real>& density_down,
                      std::vector<Real>& xc_potential_up,
                      std::vector<Real>& xc_potential_down,
                      std::vector<Real>& xc_energy_density) const override {
        int number_of_grid_points = static_cast<int>(density_up.size());
        xc_potential_up.resize(number_of_grid_points);
        xc_potential_down.resize(number_of_grid_points);
        xc_energy_density.resize(number_of_grid_points);

        std::vector<Real> density_array(2 * number_of_grid_points);
        std::vector<Real> exchange_energies(number_of_grid_points);
        std::vector<Real> correlation_energies(number_of_grid_points);
        std::vector<Real> exchange_potentials(2 * number_of_grid_points);
        std::vector<Real> correlation_potentials(2 * number_of_grid_points);

        auto* non_const_exchange = const_cast<xc_func_type*>(&func_x_);
        auto* non_const_correlation = const_cast<xc_func_type*>(&func_c_);

        /**
         * The spin arrays are interleaved two per grid point, so a slice that
         * starts at grid point `offset` starts at element `2 * offset` in
         * them. Slices stay whole grid points, so a pair is never split.
         */
        for_each_grid_slice(number_of_grid_points, [&](int offset, int count) {
            for (int grid_point_index = offset; grid_point_index < offset + count;
                 grid_point_index++) {
                density_array[2 * grid_point_index] = std::max(
                    density_up[grid_point_index], math::tolerances::minimum_electron_density);
                density_array[2 * grid_point_index + 1] = std::max(
                    density_down[grid_point_index], math::tolerances::minimum_electron_density);
            }
            xc_lda_exc_vxc(non_const_exchange,
                           count,
                           density_array.data() + 2 * offset,
                           exchange_energies.data() + offset,
                           exchange_potentials.data() + 2 * offset);
            xc_lda_exc_vxc(non_const_correlation,
                           count,
                           density_array.data() + 2 * offset,
                           correlation_energies.data() + offset,
                           correlation_potentials.data() + 2 * offset);
            for (int grid_point_index = offset; grid_point_index < offset + count;
                 grid_point_index++) {
                xc_energy_density[grid_point_index] =
                    constants::electron_charge_squared_rydberg *
                    (exchange_energies[grid_point_index] + correlation_energies[grid_point_index]);
                xc_potential_up[grid_point_index] = constants::electron_charge_squared_rydberg *
                                                    (exchange_potentials[2 * grid_point_index] +
                                                     correlation_potentials[2 * grid_point_index]);
                xc_potential_down[grid_point_index] =
                    constants::electron_charge_squared_rydberg *
                    (exchange_potentials[2 * grid_point_index + 1] +
                     correlation_potentials[2 * grid_point_index + 1]);
            }
        });

        /// Checked serially, for the reason given in `compute()`.
        for (int grid_point_index = 0; grid_point_index < number_of_grid_points;
             grid_point_index++) {
            if (!all_finite({xc_energy_density[grid_point_index],
                             xc_potential_up[grid_point_index],
                             xc_potential_down[grid_point_index]})) {
                throw_non_finite_output("spin-polarized LDA",
                                        grid_point_index,
                                        {{"rho_up", density_array[2 * grid_point_index]},
                                         {"rho_down", density_array[2 * grid_point_index + 1]}});
            }
        }
    }

    // ─── Unpolarized GGA ────────────────────────────────────────────────
    void compute_gradient_corrected(const std::vector<Real>& density,
                                    const std::vector<Real>& gradient_density_squared,
                                    std::vector<Real>& xc_potential,
                                    std::vector<Real>& gradient_correction_density,
                                    std::vector<Real>& xc_energy_density) const override {
        int number_of_grid_points = static_cast<int>(density.size());
        xc_potential.resize(number_of_grid_points);
        gradient_correction_density.resize(number_of_grid_points);
        xc_energy_density.resize(number_of_grid_points);

        constexpr Real rho_threshold_lda = 1e-10;
        constexpr Real rho_threshold_gga = 1e-6;
        constexpr Real gradient_threshold_gga = 1e-10;

        ensure_work_buffers_size(number_of_grid_points);
        auto& exchange_energies = work_exchange_energies_;
        auto& correlation_energies = work_correlation_energies_;
        auto& exchange_potentials = work_exchange_potentials_;
        auto& correlation_potentials = work_correlation_potentials_;
        auto& vsigma_exchange = work_vsigma_exchange_;
        auto& vsigma_correlation = work_vsigma_correlation_;
        auto& absolute_density = work_absolute_density_;
        auto& density_gradient_squared_clamped = work_density_gradient_squared_clamped_;

        auto* non_const_exchange = const_cast<xc_func_type*>(&func_x_);
        auto* non_const_correlation = const_cast<xc_func_type*>(&func_c_);

        for_each_grid_slice(number_of_grid_points, [&](int offset, int count) {
            for (int grid_point_index = offset; grid_point_index < offset + count;
                 grid_point_index++) {
                absolute_density[grid_point_index] = std::abs(density[grid_point_index]);
                density_gradient_squared_clamped[grid_point_index] =
                    std::max(gradient_density_squared[grid_point_index], 0.0);
            }

            if (exchange_requires_gradient_) {
                xc_gga_exc_vxc(non_const_exchange,
                               count,
                               absolute_density.data() + offset,
                               density_gradient_squared_clamped.data() + offset,
                               exchange_energies.data() + offset,
                               exchange_potentials.data() + offset,
                               vsigma_exchange.data() + offset);
            } else {
                xc_lda_exc_vxc(non_const_exchange,
                               count,
                               absolute_density.data() + offset,
                               exchange_energies.data() + offset,
                               exchange_potentials.data() + offset);
                std::fill_n(vsigma_exchange.begin() + offset, count, 0.0);
            }

            if (correlation_requires_gradient_) {
                xc_gga_exc_vxc(non_const_correlation,
                               count,
                               absolute_density.data() + offset,
                               density_gradient_squared_clamped.data() + offset,
                               correlation_energies.data() + offset,
                               correlation_potentials.data() + offset,
                               vsigma_correlation.data() + offset);
            } else {
                xc_lda_exc_vxc(non_const_correlation,
                               count,
                               absolute_density.data() + offset,
                               correlation_energies.data() + offset,
                               correlation_potentials.data() + offset);
                std::fill_n(vsigma_correlation.begin() + offset, count, 0.0);
            }
        });

        for (int grid_point_index = 0; grid_point_index < number_of_grid_points;
             grid_point_index++) {
            if (absolute_density[grid_point_index] <= rho_threshold_lda) {
                xc_energy_density[grid_point_index] = 0.0;
                xc_potential[grid_point_index] = 0.0;
                gradient_correction_density[grid_point_index] = 0.0;
                continue;
            }

            xc_energy_density[grid_point_index] =
                constants::electron_charge_squared_rydberg *
                (exchange_energies[grid_point_index] + correlation_energies[grid_point_index]);
            xc_potential[grid_point_index] =
                constants::electron_charge_squared_rydberg *
                (exchange_potentials[grid_point_index] + correlation_potentials[grid_point_index]);

            if (absolute_density[grid_point_index] <= rho_threshold_gga ||
                std::abs(density_gradient_squared_clamped[grid_point_index]) <=
                    gradient_threshold_gga) {
                gradient_correction_density[grid_point_index] = 0.0;
            } else {
                gradient_correction_density[grid_point_index] =
                    constants::electron_charge_squared_rydberg *
                    (vsigma_exchange[grid_point_index] + vsigma_correlation[grid_point_index]);
            }
            if (!all_finite({xc_energy_density[grid_point_index],
                             xc_potential[grid_point_index],
                             gradient_correction_density[grid_point_index]})) {
                throw_non_finite_output(
                    "unpolarized GGA",
                    grid_point_index,
                    {{"rho", absolute_density[grid_point_index]},
                     {"sigma", density_gradient_squared_clamped[grid_point_index]}});
            }
        }
    }

    // ─── Spin-polarized GGA ─────────────────────────────────────────────
    void
    compute_gradient_corrected_spin(const std::vector<Real>& density_up,
                                    const std::vector<Real>& density_down,
                                    const std::vector<Real>& gradient_density_squared_up_up,
                                    const std::vector<Real>& gradient_density_squared_up_down,
                                    const std::vector<Real>& gradient_density_squared_down_down,
                                    std::vector<Real>& xc_potential_up,
                                    std::vector<Real>& xc_potential_down,
                                    std::vector<Real>& gradient_correction_up_up,
                                    std::vector<Real>& gradient_correction_up_down,
                                    std::vector<Real>& gradient_correction_down_down,
                                    std::vector<Real>& xc_energy_density) const override {
        int total_grid_points = static_cast<int>(density_up.size());
        xc_potential_up.resize(total_grid_points);
        xc_potential_down.resize(total_grid_points);
        gradient_correction_up_up.resize(total_grid_points);
        gradient_correction_up_down.resize(total_grid_points);
        gradient_correction_down_down.resize(total_grid_points);
        xc_energy_density.resize(total_grid_points);

        constexpr Real rho_threshold_lda = 1e-10;
        constexpr Real rho_threshold_gga = 1e-6;
        constexpr Real gradient_threshold_gga = 1e-10;

        std::vector<Real> density_array(2 * total_grid_points);
        std::vector<Real> gradient_squared_array(3 * total_grid_points);
        for (int grid_point_index = 0; grid_point_index < total_grid_points; grid_point_index++) {
            density_array[2 * grid_point_index] = std::abs(density_up[grid_point_index]);
            density_array[2 * grid_point_index + 1] = std::abs(density_down[grid_point_index]);

            gradient_squared_array[3 * grid_point_index] =
                std::max(gradient_density_squared_up_up[grid_point_index], 0.0);
            gradient_squared_array[3 * grid_point_index + 1] =
                gradient_density_squared_up_down[grid_point_index];
            gradient_squared_array[3 * grid_point_index + 2] =
                std::max(gradient_density_squared_down_down[grid_point_index], 0.0);
        }

        std::vector<Real> exchange_energies(total_grid_points);
        std::vector<Real> correlation_energies(total_grid_points);
        std::vector<Real> exchange_density_potential(2 * total_grid_points, 0.0);
        std::vector<Real> correlation_density_potential(2 * total_grid_points, 0.0);
        std::vector<Real> exchange_gradient_potential(3 * total_grid_points, 0.0);
        std::vector<Real> correlation_gradient_potential(3 * total_grid_points, 0.0);

        auto* non_const_exchange = const_cast<xc_func_type*>(&func_x_);
        auto* non_const_correlation = const_cast<xc_func_type*>(&func_c_);

        if (exchange_requires_gradient_) {
            xc_gga_exc_vxc(non_const_exchange,
                           total_grid_points,
                           density_array.data(),
                           gradient_squared_array.data(),
                           exchange_energies.data(),
                           exchange_density_potential.data(),
                           exchange_gradient_potential.data());
        } else {
            xc_lda_exc_vxc(non_const_exchange,
                           total_grid_points,
                           density_array.data(),
                           exchange_energies.data(),
                           exchange_density_potential.data());
        }

        if (correlation_requires_gradient_) {
            xc_gga_exc_vxc(non_const_correlation,
                           total_grid_points,
                           density_array.data(),
                           gradient_squared_array.data(),
                           correlation_energies.data(),
                           correlation_density_potential.data(),
                           correlation_gradient_potential.data());
        } else {
            xc_lda_exc_vxc(non_const_correlation,
                           total_grid_points,
                           density_array.data(),
                           correlation_energies.data(),
                           correlation_density_potential.data());
        }

        for (int grid_point_index = 0; grid_point_index < total_grid_points; grid_point_index++) {
            Real density_up = density_array[2 * grid_point_index];
            Real density_down = density_array[2 * grid_point_index + 1];
            Real gradient_squared_up = std::abs(gradient_squared_array[3 * grid_point_index]);
            Real gradient_squared_down = std::abs(gradient_squared_array[3 * grid_point_index + 2]);

            if (density_up <= rho_threshold_lda || density_down <= rho_threshold_lda) {
                xc_energy_density[grid_point_index] = 0.0;
                xc_potential_up[grid_point_index] = xc_potential_down[grid_point_index] = 0.0;
                gradient_correction_up_up[grid_point_index] =
                    gradient_correction_up_down[grid_point_index] =
                        gradient_correction_down_down[grid_point_index] = 0.0;
                continue;
            }

            xc_energy_density[grid_point_index] =
                constants::electron_charge_squared_rydberg *
                (exchange_energies[grid_point_index] + correlation_energies[grid_point_index]);
            xc_potential_up[grid_point_index] =
                constants::electron_charge_squared_rydberg *
                (exchange_density_potential[2 * grid_point_index] +
                 correlation_density_potential[2 * grid_point_index]);
            xc_potential_down[grid_point_index] =
                constants::electron_charge_squared_rydberg *
                (exchange_density_potential[2 * grid_point_index + 1] +
                 correlation_density_potential[2 * grid_point_index + 1]);

            if (density_up <= rho_threshold_gga || density_down <= rho_threshold_gga ||
                gradient_squared_up <= gradient_threshold_gga ||
                gradient_squared_down <= gradient_threshold_gga) {
                gradient_correction_up_up[grid_point_index] =
                    gradient_correction_up_down[grid_point_index] =
                        gradient_correction_down_down[grid_point_index] = 0.0;
            } else {
                gradient_correction_up_up[grid_point_index] =
                    constants::electron_charge_squared_rydberg *
                    (exchange_gradient_potential[3 * grid_point_index] +
                     correlation_gradient_potential[3 * grid_point_index]);
                gradient_correction_up_down[grid_point_index] =
                    constants::electron_charge_squared_rydberg *
                    (exchange_gradient_potential[3 * grid_point_index + 1] +
                     correlation_gradient_potential[3 * grid_point_index + 1]);
                gradient_correction_down_down[grid_point_index] =
                    constants::electron_charge_squared_rydberg *
                    (exchange_gradient_potential[3 * grid_point_index + 2] +
                     correlation_gradient_potential[3 * grid_point_index + 2]);
            }

            // Safety clamps
            auto clamp_potential_value = [](Real& value, Real limit) {
                if (std::abs(value) > limit) {
                    value = (value > 0) ? limit : -limit;
                }
            };
            clamp_potential_value(xc_potential_up[grid_point_index], 100.0);
            clamp_potential_value(xc_potential_down[grid_point_index], 100.0);
            clamp_potential_value(gradient_correction_up_up[grid_point_index], 100.0);
            clamp_potential_value(gradient_correction_up_down[grid_point_index], 100.0);
            clamp_potential_value(gradient_correction_down_down[grid_point_index], 100.0);

            if (!all_finite({xc_energy_density[grid_point_index],
                             xc_potential_up[grid_point_index],
                             xc_potential_down[grid_point_index],
                             gradient_correction_up_up[grid_point_index],
                             gradient_correction_up_down[grid_point_index],
                             gradient_correction_down_down[grid_point_index]})) {
                throw_non_finite_output(
                    "spin-polarized GGA",
                    grid_point_index,
                    {{"rho_up", density_up},
                     {"rho_down", density_down},
                     {"sigma_up_up", gradient_squared_array[3 * grid_point_index]},
                     {"sigma_up_down", gradient_squared_array[3 * grid_point_index + 1]},
                     {"sigma_down_down", gradient_squared_array[3 * grid_point_index + 2]}});
            }
        }
    }

  private:
    /**
     * The shortest slice of grid worth handing to a thread.
     *
     * Forking a team costs a few microseconds; a LibXC evaluation costs tens
     * of nanoseconds a point. Below roughly this many points the team is the
     * more expensive half of the work, so `for_each_grid_slice` runs the whole
     * grid inline instead.
     */
    static constexpr int minimum_grid_points_per_thread = 4096;

    /**
     * Run `evaluate_slice(offset, count)` over contiguous slices that together
     * cover [0, number_of_grid_points), one slice per thread.
     *
     * Every functional evaluated through here is pointwise: a grid point's
     * outputs depend on that point's inputs and on nothing else. Splitting the
     * arrays therefore reproduces what one whole-grid call would have written,
     * to the bit and not merely to a tolerance, whatever the slices are -- so
     * these results do not move with the thread count. Nothing is summed
     * across slices, so there is no summation order to preserve.
     *
     * LibXC's evaluators read the functional and write only the output arrays
     * they are handed, which is what lets several threads drive one
     * `xc_func_type` over disjoint slices.
     *
     * `evaluate_slice` must not throw: an exception that escapes a parallel
     * region terminates the program. Callers do their finiteness checks in a
     * serial pass afterwards.
     */
    template <typename SliceEvaluator>
    static void for_each_grid_slice(int number_of_grid_points, SliceEvaluator evaluate_slice) {
        [[maybe_unused]] int slice_threads = execution::threading::thread_count_for(
            number_of_grid_points, minimum_grid_points_per_thread);
#pragma omp parallel num_threads(slice_threads) if (slice_threads > 1)
        {
            long long first_point = 0;
            long long last_point = number_of_grid_points;
            execution::threading::slice_bounds(number_of_grid_points,
                                               execution::threading::team_size(),
                                               execution::threading::thread_index(),
                                               first_point,
                                               last_point);
            if (last_point > first_point) {
                evaluate_slice(static_cast<int>(first_point),
                               static_cast<int>(last_point - first_point));
            }
        }
    }

    [[nodiscard]] static bool all_finite(std::initializer_list<Real> values) {
        for (Real value : values) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Refuse a value LibXC returned non-finite, at a point Q3 is about to use.
     *
     * Every caller checks the values it is about to store, after its density
     * floor, GGA screening and safety clamps, so a point Q3 zeroes or bounds
     * cannot trip this; what reaches it goes into the potential or the energy.
     * Without it a NaN travelled silently -- the clamps compare magnitudes,
     * which a NaN never exceeds -- and surfaced a step later as a failed
     * subspace eigensolve and NaN energies. LibXC 7.1.0 to 7.1.2 do exactly
     * that for spin-polarized PBE correlation wherever the total density
     * gradient vanishes while the spin gradients do not, as between the
     * opposite moments of an antiferromagnet. The build takes a system LibXC
     * only from the series Q3 is tested against, so this is what stops an
     * untested one when it is forced, and only then does the message point at
     * the version. It is formatted only here, so the per-point cost is one
     * isfinite per value.
     */
    [[noreturn]] void
    throw_non_finite_output(const char* evaluation,
                            int grid_point_index,
                            std::initializer_list<std::pair<const char*, Real>> inputs) const {
        std::string input_description;
        for (const auto& [name, value] : inputs) {
            char buffer[64];
            std::snprintf(buffer,
                          sizeof(buffer),
                          "%s%s=%.17g",
                          input_description.empty() ? "" : ", ",
                          name,
                          value);
            input_description += buffer;
        }
        std::string message = std::string("LibXC ") + xc_version_string() +
                              " returned a non-finite value in " + evaluation + " evaluation of '" +
                              functional_name_ + "' at grid point " +
                              std::to_string(grid_point_index) + " (" + input_description + ").";

        // The series Q3_TESTED_LIBXC_SERIES in CMakeLists.txt accepts from the system.
        int major_version = 0;
        int minor_version = 0;
        int micro_version = 0;
        xc_version(&major_version, &minor_version, &micro_version);
        const bool tested_series = (major_version == 5 && minor_version == 2) ||
                                   (major_version == 7 && minor_version == 0);
        if (!tested_series) {
            message += " Q3 is tested against LibXC 5.2 and 7.0; 7.1.0 to 7.1.2 return NaN for "
                       "the gradient derivatives of spin-polarized PBE correlation where the "
                       "total density gradient vanishes. Without Q3_ALLOW_UNTESTED_SYSTEM_LIBXC "
                       "the build uses a tested LibXC, building 7.0.0 from source if it has "
                       "to; see KNOWN-ISSUES.md.";
        }
        throw std::runtime_error(message);
    }

    [[nodiscard]] static std::string family_name(int family) {
        switch (family) {
            case XC_FAMILY_LDA:
                return "LDA";
            case XC_FAMILY_GGA:
                return "GGA";
            case XC_FAMILY_MGGA:
                return "MGGA";
            case XC_FAMILY_LCA:
                return "LCA";
            case XC_FAMILY_OEP:
                return "OEP";
            case XC_FAMILY_HYB_GGA:
                return "HYB_GGA";
            case XC_FAMILY_HYB_MGGA:
                return "HYB_MGGA";
            case XC_FAMILY_HYB_LDA:
                return "HYB_LDA";
            default:
                return "UNKNOWN(" + std::to_string(family) + ")";
        }
    }

    [[nodiscard]] static std::string missing_capability(int family) {
        switch (family) {
            case XC_FAMILY_MGGA:
                return "kinetic-energy-density and density-Laplacian inputs";
            case XC_FAMILY_HYB_LDA:
            case XC_FAMILY_HYB_GGA:
                return "nonlocal exact-exchange operator";
            case XC_FAMILY_HYB_MGGA:
                return "nonlocal exact-exchange operator and kinetic-energy-density inputs";
            default:
                return "family-specific evaluation path";
        }
    }

    static void ensure_supported_family(const std::string& functional_identifier,
                                        const std::string& component_name,
                                        int functional_identifier_number,
                                        const xc_func_type& functional,
                                        int family) {
        if (family == XC_FAMILY_LDA || family == XC_FAMILY_GGA) {
            return;
        }

        throw std::runtime_error(
            "Unsupported LibXC functional identifier '" + functional_identifier + "': " +
            component_name + " component id=" + std::to_string(functional_identifier_number) +
            " ('" + std::string(functional.info->name) + "') has family " + family_name(family) +
            "; Q3 does not implement the required " + missing_capability(family) +
            ". Q3 currently supports only complete LDA and GGA LibXC evaluation.");
    }

    xc_func_type func_x_;
    xc_func_type func_c_;
    int exchange_id_;
    int correlation_id_;
    bool exchange_requires_gradient_;
    bool correlation_requires_gradient_;
    int number_of_spin_components_;
    std::string functional_name_;

    /**
     * Cached work buffers to avoid per-call heap allocation in GGA methods.
     * These are mutable because compute methods are const.
     */
    mutable std::vector<Real> work_exchange_energies_;
    mutable std::vector<Real> work_correlation_energies_;
    mutable std::vector<Real> work_exchange_potentials_;
    mutable std::vector<Real> work_correlation_potentials_;
    mutable std::vector<Real> work_vsigma_exchange_;
    mutable std::vector<Real> work_vsigma_correlation_;
    mutable std::vector<Real> work_absolute_density_;
    mutable std::vector<Real> work_density_gradient_squared_clamped_;

    /**
     * Ensure all work buffers are at least number_of_grid_points in size.
     * Only reallocates if the current size is insufficient.
     */
    void ensure_work_buffers_size(int number_of_grid_points) const {
        auto ensure = [](std::vector<Real>& buffer, int size) {
            if (static_cast<int>(buffer.size()) < size) {
                buffer.resize(size);
            }
        };
        ensure(work_exchange_energies_, number_of_grid_points);
        ensure(work_correlation_energies_, number_of_grid_points);
        ensure(work_exchange_potentials_, number_of_grid_points);
        ensure(work_correlation_potentials_, number_of_grid_points);
        ensure(work_vsigma_exchange_, number_of_grid_points);
        ensure(work_vsigma_correlation_, number_of_grid_points);
        ensure(work_absolute_density_, number_of_grid_points);
        ensure(work_density_gradient_squared_clamped_, number_of_grid_points);
    }
};

} // namespace q3
