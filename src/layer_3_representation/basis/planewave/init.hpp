#pragma once

#include <algorithm>

#include <cmath>

/**
 * PlanewaveBasis — facade over domain-specific plane-wave basis components.
 *
 * Sub-domains:
 *   fft_pair_       — charge-density and wavefunction FFT grids
 *   g_vectors_      — charge-density G-vector sphere
 *   plane_wave_table_ — per-k-point active plane waves (initialized separately)
 *   transformers    — stateless G ↔ R conversion (WavefunctionTransformer,
 * ChargeDensityTransformer)
 */

#include <vector>

#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/pair.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/kpoint/plane_wave_selector.hpp"
#include "layer_3_representation/basis/planewave/internal/kpoint/plane_wave_table.hpp"
#include "layer_3_representation/basis/planewave/internal/transform/charge_density_transformer.hpp"
#include "layer_3_representation/basis/planewave/internal/transform/wavefunction_transformer.hpp"

namespace q3 {

/**
 * Energy cutoffs that determine the plane-wave basis size.
 */
struct PlanewaveBasisConfig {
    /// Wavefunction energy cutoff in Rydberg.
    Real energy_cutoff_wavefunction = 0.0;
    /// Charge density energy cutoff in Rydberg (typically 4× wavefunction cutoff for NC PP).
    Real energy_cutoff_charge_density = 0.0;
};

class PlanewaveBasis {
  public:
    PlanewaveBasis() = default;

    explicit PlanewaveBasis(PlanewaveBasisConfig config)
        : energy_cutoff_wavefunction_(config.energy_cutoff_wavefunction)
        , energy_cutoff_charge_density_(config.energy_cutoff_charge_density) {
        if (energy_cutoff_charge_density_ <= 0.0) {
            energy_cutoff_charge_density_ = 4.0 * energy_cutoff_wavefunction_;
        }
    }

    /**
     * Generate G-vectors and FFT grids for given cutoffs.
     */
    void generate(const FFTGridFactory& fft_grid_factory,
                  const Crystal& crystal_structure,
                  Real wavefunction_energy_cutoff,
                  Real charge_density_energy_cutoff,
                  bool is_gamma_only = false,
                  bool use_measured_fft_planning = true,
                  bool use_exact_wavefunction_grid = false) {
        is_gamma_only_ = is_gamma_only;
        energy_cutoff_wavefunction_ = wavefunction_energy_cutoff;
        energy_cutoff_charge_density_ = charge_density_energy_cutoff;
        if (energy_cutoff_charge_density_ <= 0.0) {
            energy_cutoff_charge_density_ = 4.0 * energy_cutoff_wavefunction_;
        }
        fft_pair_.initialize(fft_grid_factory,
                             crystal_structure,
                             energy_cutoff_charge_density_,
                             energy_cutoff_wavefunction_,
                             use_measured_fft_planning,
                             use_exact_wavefunction_grid);
        /**
         * Charge density G-vectors must remain replicated on all band ranks.
         * Only wavefunction coefficients (per k-point) are distributed via FFT
         * slab decomposition — see PlaneWaveSelector::select().
         */
        g_vectors_.generate(crystal_structure,
                            energy_cutoff_charge_density_,
                            fft_pair_.fft_rho().grid_dimension_1(),
                            fft_pair_.fft_rho().grid_dimension_2(),
                            fft_pair_.fft_rho().grid_dimension_3(),
                            is_gamma_only_,
                            0,
                            1,
                            -1,
                            -1);
        maximum_plane_waves_per_kpoint_ = g_vectors_.estimate_maximum_plane_waves_per_kpoint(
            energy_cutoff_wavefunction_, crystal_structure.two_pi_over_lattice_parameter_squared());

        /**
         * Tell the density grid which sticks carry G-vectors, so a stick
         * layout can balance them and skip the empty ones. A grid that does
         * not distribute by stick ignores this.
         */
        fft_pair_.fft_rho().assign_sticks(
            stick_weights_for_grid(fft_pair_.fft_rho(), [](const GVector&) { return true; }));
    }

    /**
     * Count the G-vectors on each stick of a grid, by wrapped first and second
     * Miller index, for those the predicate accepts and that fit in the grid.
     */
    template <typename Predicate>
    [[nodiscard]] std::vector<int> stick_weights_for_grid(const FFTGrid& grid,
                                                          Predicate accept) const {
        int dimension_1 = grid.grid_dimension_1();
        int dimension_2 = grid.grid_dimension_2();
        int dimension_3 = grid.grid_dimension_3();
        std::vector<int> weights(static_cast<std::size_t>(dimension_1) * dimension_2, 0);
        for (const auto& g_vector : g_vectors_.g_vectors()) {
            int miller_1 = static_cast<int>(std::lround(g_vector.miller.x));
            int miller_2 = static_cast<int>(std::lround(g_vector.miller.y));
            int miller_3 = static_cast<int>(std::lround(g_vector.miller.z));
            if (std::abs(miller_1) > dimension_1 / 2 || std::abs(miller_2) > dimension_2 / 2 ||
                std::abs(miller_3) > dimension_3 / 2 || !accept(g_vector)) {
                continue;
            }
            int wrapped_1 = miller_1 >= 0 ? miller_1 : miller_1 + dimension_1;
            int wrapped_2 = miller_2 >= 0 ? miller_2 : miller_2 + dimension_2;
            weights[static_cast<std::size_t>(wrapped_1) * dimension_2 + wrapped_2] += 1;
        }
        /**
         * Every stick is owned, balanced by how much of the sphere it carries.
         * A stick outside the sphere could be dropped, but two consumers here
         * still read the whole grid box -- the
         * dense-to-smooth potential copy and the smooth-to-dense density
         * transfer -- and truncating would change their answer against a
         * serial run. Transporting the empty sticks costs the all-to-all the
         * same volume the slab transpose moved; skipping them is a later
         * optimisation that wants those two loops made sphere-only first.
         */
        for (auto& weight : weights) {
            weight = std::max(weight, 1);
        }
        return weights;
    }

    // ─── Charge-density G-vectors ──────────────────────────────────

    [[nodiscard]] int number_of_g_vectors() const {
        return g_vectors_.number_of_g_vectors();
    }
    [[nodiscard]] const std::vector<GVector>& g_vectors() const {
        return g_vectors_.g_vectors();
    }
    [[nodiscard]] int first_nonzero_g_index() const {
        return g_vectors_.first_nonzero_g_index();
    }
    [[nodiscard]] const std::vector<Real>& g_vector_magnitudes() const {
        return g_vectors_.g_vector_magnitudes();
    }
    [[nodiscard]] bool is_gamma_only() const {
        return is_gamma_only_;
    }

    /**
     * Whether this rank's plane-wave list begins with G = 0.
     *
     * The gamma trick sums 2 Re over the stored half sphere and removes the
     * origin, which the half sphere holds once, from the double count. When
     * reciprocal space is distributed by stick exactly one rank holds the
     * origin, and only that rank may remove it. Serial and replicated layouts
     * always hold it, so the default
     * is true; initialize_kpoint_data() sets it from the list actually kept.
     * Meaningful only when is_gamma_only().
     */
    [[nodiscard]] bool gamma_origin_is_local() const {
        return gamma_origin_is_local_;
    }

    // ─── Per-k-point plane waves ───────────────────────────────────

    void compute_gk(const Vector3D& k_coordinates,
                    const Crystal& crystal_structure,
                    std::vector<int>& g_vector_indices,
                    std::vector<Real>& kinetic_energy_squared,
                    int band_size = 1) const {
        /**
         * G-vector distribution: when band_size > 1, pass the MPI FFT grid so
         * that PlaneWaveSelector only keeps G-vectors in this rank's slab.
         */
        const FFTGrid* fft_wave_ptr = (band_size > 1) ? &fft_pair_.fft_wave() : nullptr;
        PlaneWaveSelector::select(g_vectors_,
                                  k_coordinates,
                                  crystal_structure,
                                  energy_cutoff_wavefunction_,
                                  g_vector_indices,
                                  kinetic_energy_squared,
                                  fft_wave_ptr,
                                  band_size);
    }

    [[nodiscard]] int maximum_plane_waves_per_kpoint() const {
        return maximum_plane_waves_per_kpoint_;
    }

    void initialize_kpoint_data(const Crystal& crystal_structure,
                                const KPointsGrid& kpoint_grid,
                                int band_size = 1) {
        /**
         * Under a shared pool the wavefunction grid must know its sticks before
         * the selector asks it who owns what. A stick counts if any G-vector on
         * it can fall inside the wavefunction cutoff at some k-point, which is
         * |G| within sqrt(E_cut) plus the largest |k| on the grid -- a superset
         * that costs a few empty sticks and never drops a plane wave.
         */
        if (band_size > 1) {
            const auto& reciprocal = crystal_structure.reciprocal_lattice_vectors();
            Real largest_k = 0.0;
            for (int index = 0; index < kpoint_grid.number_of_kpoints(); ++index) {
                const auto& k = kpoint_grid[index].coordinates;
                Vector3D k_cartesian;
                for (int dimension = 0; dimension < 3; ++dimension) {
                    k_cartesian[dimension] = k.x * reciprocal[0][dimension] +
                                             k.y * reciprocal[1][dimension] +
                                             k.z * reciprocal[2][dimension];
                }
                largest_k = std::max(largest_k, std::sqrt(k_cartesian.norm2()));
            }
            Real cutoff_radius =
                std::sqrt(energy_cutoff_wavefunction_ /
                          crystal_structure.two_pi_over_lattice_parameter_squared());
            Real radius_squared = (cutoff_radius + largest_k) * (cutoff_radius + largest_k) *
                                  (1.0 + math::tolerances::energy_cutoff_rounding_tolerance);
            fft_pair_.fft_wave().assign_sticks(
                stick_weights_for_grid(fft_pair_.fft_wave(), [radius_squared](const GVector& g) {
                    return g.g_cart.norm2() <= radius_squared;
                }));
        }

        const FFTGrid* fft_wave_ptr = (band_size > 1) ? &fft_pair_.fft_wave() : nullptr;
        plane_wave_table_.initialize(g_vectors_,
                                     crystal_structure,
                                     kpoint_grid,
                                     energy_cutoff_wavefunction_,
                                     fft_wave_ptr,
                                     band_size);

        /**
         * Gamma-only means a single k-point, so one list decides: the origin is
         * local when the first entry -- the list is sorted by |k+G|^2, and at
         * Gamma nothing else has zero kinetic energy -- is G = 0.
         */
        gamma_origin_is_local_ = true;
        if (is_gamma_only_ && kpoint_grid.number_of_kpoints() > 0) {
            const auto& indices = plane_wave_table_.g_vector_indices_by_kpoint_index(0);
            gamma_origin_is_local_ =
                !indices.empty() &&
                g_vectors_.g_vectors()[static_cast<std::size_t>(indices[0])].magnitude_squared <
                    math::tolerances::numerical_zero_threshold;
        }
    }

    [[nodiscard]] const std::vector<int>& g_vector_indices_by_kpoint_index(int kpoint_index) const {
        return plane_wave_table_.g_vector_indices_by_kpoint_index(kpoint_index);
    }

    [[nodiscard]] int number_of_plane_waves_by_kpoint_index(int kpoint_index) const {
        return plane_wave_table_.number_of_plane_waves_by_kpoint_index(kpoint_index);
    }

    [[nodiscard]] const std::vector<Real>&
    kinetic_energy_squared_by_kpoint_index(int kpoint_index) const {
        return plane_wave_table_.kinetic_energy_squared_by_kpoint_index(kpoint_index);
    }

    [[nodiscard]] const std::vector<std::vector<int>>& kpoint_g_vector_indices() const {
        return plane_wave_table_.kpoint_g_vector_indices();
    }

    [[nodiscard]] const std::vector<int>& kpoint_number_of_plane_waves() const {
        return plane_wave_table_.kpoint_number_of_plane_waves();
    }

    [[nodiscard]] const std::vector<std::vector<Real>>& kpoint_kinetic_energy_squared() const {
        return plane_wave_table_.kpoint_kinetic_energy_squared();
    }

    // ─── FFT grids ─────────────────────────────────────────────────

    [[nodiscard]] FFTGrid& fft_rho() {
        return fft_pair_.fft_rho();
    }
    [[nodiscard]] const FFTGrid& fft_rho() const {
        return fft_pair_.fft_rho();
    }
    [[nodiscard]] FFTGrid& fft_wave() {
        return fft_pair_.fft_wave();
    }
    [[nodiscard]] const FFTGrid& fft_wave() const {
        return fft_pair_.fft_wave();
    }
    [[nodiscard]] const SerialFFTGrid& fft_wave_serial() const {
        return fft_pair_.fft_wave_serial();
    }

    // ─── G ↔ R transforms ──────────────────────────────────────────

    void g_to_r(const std::vector<Complex>& coefficients_g_space,
                const std::vector<int>& g_vector_indices,
                int number_of_plane_waves,
                std::vector<Complex>& coefficients_real_space) const {
        WavefunctionTransformer::g_to_r(g_vectors_,
                                        fft_pair_.fft_wave(),
                                        coefficients_g_space,
                                        g_vector_indices,
                                        number_of_plane_waves,
                                        coefficients_real_space);
    }

    void r_to_g(const std::vector<Complex>& coefficients_real_space,
                const std::vector<int>& g_vector_indices,
                int number_of_plane_waves,
                std::vector<Complex>& coefficients_g_space) const {
        WavefunctionTransformer::r_to_g(g_vectors_,
                                        fft_pair_.fft_wave(),
                                        coefficients_real_space,
                                        g_vector_indices,
                                        number_of_plane_waves,
                                        coefficients_g_space);
    }

    void rho_g_to_r(const std::vector<Complex>& charge_density_g_space,
                    std::vector<Real>& charge_density_real_space) const {
        ChargeDensityTransformer::rho_g_to_r(
            g_vectors_, fft_pair_.fft_rho(), charge_density_g_space, charge_density_real_space);
    }

    void rho_r_to_g(const std::vector<Real>& charge_density_real_space,
                    std::vector<Complex>& charge_density_g_space) const {
        ChargeDensityTransformer::rho_r_to_g(
            g_vectors_, fft_pair_.fft_rho(), charge_density_real_space, charge_density_g_space);
    }

    // ─── Cutoffs ───────────────────────────────────────────────────

    [[nodiscard]] Real wavefunction_energy_cutoff() const {
        return energy_cutoff_wavefunction_;
    }
    [[nodiscard]] Real charge_density_energy_cutoff() const {
        return energy_cutoff_charge_density_;
    }

    // ─── Sub-component access ────────────────────────────────────────

    [[nodiscard]] const GVectorsGrid& g_vectors_grid() const {
        return g_vectors_;
    }
    [[nodiscard]] const PlanewaveFFTPair& fft_pair() const {
        return fft_pair_;
    }
    [[nodiscard]] const PlaneWaveTable& plane_wave_table() const {
        return plane_wave_table_;
    }

  private:
    Real energy_cutoff_wavefunction_{};
    Real energy_cutoff_charge_density_{};
    PlanewaveFFTPair fft_pair_;
    GVectorsGrid g_vectors_;
    PlaneWaveTable plane_wave_table_;
    int maximum_plane_waves_per_kpoint_{};
    bool is_gamma_only_{false};
    bool gamma_origin_is_local_{true};
};

} // namespace q3
