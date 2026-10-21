#pragma once

/**
 * UltrasoftPlanewavePseudopotentialKohnShamSCFIterator — ultrasoft plane-wave PSP SCF (layer 4).
 *
 * Extends the norm-conserving pseudopotential iterator with the ultrasoft-only
 * machinery: the projector overlap coefficients, the augmentation charges
 * added to the smooth density, the effective D-matrix used by the Hamiltonian, and
 * the smooth (pre-augmentation) charge density that is mixed and snapshotted.
 */

#include <algorithm>
#include <chrono>
#include <vector>

#include "layer_0_core/math/linear_algebra/blas.hpp"

#include "layer_3_representation/charge_density/planewave/pseudopotential/init.hpp"
#include "layer_3_representation/effective_potential/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"

namespace q3 {
using namespace q3::blas;

class UltrasoftPlanewavePseudopotentialKohnShamSCFIterator
    : public PlanewavePseudopotentialKohnShamSCFIterator {
  public:
    explicit UltrasoftPlanewavePseudopotentialKohnShamSCFIterator(KohnShamSCFIteratorConfig config)
        : PlanewavePseudopotentialKohnShamSCFIterator(std::move(config)) {}
    ~UltrasoftPlanewavePseudopotentialKohnShamSCFIterator() override = default;

    [[nodiscard]] const Matrix<Real>& projector_overlap_coefficients() const override {
        return projector_overlap_coefficients_;
    }

    NormConservingPlanewaveKohnShamSCFState result() const override {
        NormConservingPlanewaveKohnShamSCFState data =
            PlanewavePseudopotentialKohnShamSCFIterator::result();
        /**
         * Copy the converged effective D-matrix into the snapshot's owned storage.
         * Without this, d_matrix_element() falls back to bare_d_matrix_element() which
         * uses channel indices (0..number_of_channels-1) on the radial d_ion_matrix
         * (size nbeta × nbeta), causing an out-of-bounds access → SIGSEGV for USPP.
         */
        auto& ultrasoft_snapshot =
            static_cast<UltrasoftPlanewaveKohnShamQuantumSystem&>(*data.quantum_system_snapshot_);
        ultrasoft_snapshot.set_nonlocal_effective_potential(nonlocal_effective_potential_);
        data.projector_overlap_coefficients = projector_overlap_coefficients_;
        data.projector_overlap_coefficients_by_spin = projector_overlap_coefficients_by_spin_;
        return data;
    }

  protected:
    Matrix<Real> projector_overlap_coefficients_;

    /**
     * The same coefficients per spin channel, un-summed.
     *
     * `projector_overlap_coefficients_` above is accumulated over
     * `kpoint_index_state`, which folds spin in, so by the time it is read the
     * two channels are indistinguishable. That is what the augmentation charge
     * on the grid wants — it is a total density — but the PAW one-center terms
     * need ρ↑ and ρ↓ apart, because exchange-correlation is nonlinear in the
     * spin density. Kept as a parallel array rather than reshaping the primary
     * one so every existing ultrasoft consumer is untouched.
     */
    std::vector<Matrix<Real>> projector_overlap_coefficients_by_spin_;
    UltrasoftPlanewaveEffectivePotential nonlocal_effective_potential_;

    /**
     * Smooth (pre-augmentation) charge density used by ultrasoft pseudopotential_set,
     * plus its converged snapshot restored at self-consistency.
     */
    std::vector<std::vector<Real>> charge_density_real_smooth_;
    std::vector<std::vector<Real>> convergence_output_density_smooth_;

    /**
     * Cached G-space augmented density from construct_density_from_states.
     * Used by symmetrize_density to avoid a redundant R→G FFT.
     */
    std::vector<std::vector<Complex>> cached_augmented_density_g_;

    /**
     * Access the ultrasoft quantum system (downcast).
     */
    [[nodiscard]] const UltrasoftPlanewaveKohnShamQuantumSystem& ultrasoft_quantum_system() const {
        return static_cast<const UltrasoftPlanewaveKohnShamQuantumSystem&>(
            *quantum_system_snapshot_);
    }

    [[nodiscard]] UltrasoftPlanewaveKohnShamQuantumSystem& mutable_ultrasoft_quantum_system() {
        return static_cast<UltrasoftPlanewaveKohnShamQuantumSystem&>(*quantum_system_snapshot_);
    }

    /**
     * The nonlocal D matrix is built per spin channel.
     *
     * D_ij = D_ion + ∫v_eff·Q_ij, and for a spin-polarized system v_eff differs
     * between channels, so D does too. It used to be built once from the spin
     * average and shared: exact for a non-magnetic cell, and for a magnetic one
     * it collapsed the moment — nickel with a PAW dataset converged to 1e-4
     * Bohr magnetons against a PBE moment near 0.64, which the norm-conserving
     * path reproduces at 0.605. Norm-conserving was never affected because its
     * D_ion is a constant with no v_eff dependence, which is why Q3's LSDA
     * validation — all of it norm-conserving — passed and said nothing about
     * this.
     *
     * `nonlocal_potentials_by_spin()` returns one entry per channel, or a
     * single averaged entry when the per-spin copies are unavailable, so the
     * unpolarized path is bit-for-bit what it was.
     */

    /**
     * The potential each spin channel's D should be built from.
     *
     * Falls back to a single entry holding the spin average whenever the
     * per-spin copies are absent, which keeps every non-spin-polarized path and
     * every path that predates them behaving exactly as before.
     */
    [[nodiscard]] std::vector<std::vector<Complex>> nonlocal_potentials_by_spin() const {
        if (!total_potential_g_by_spin_.empty())
            return total_potential_g_by_spin_;
        return {total_potential_g_};
    }

    void prepare_hamiltonian() override {
        const auto& ultrasoft_system = ultrasoft_quantum_system();
        nonlocal_effective_potential_.compute_by_spin(crystal_structure(),
                                                      planewave_basis_,
                                                      ultrasoft_system.ultrasoft_basis(),
                                                      nonlocal_potentials_by_spin());

        // Wire the D-matrix into the quantum system for d_matrix_element() dispatch
        mutable_ultrasoft_quantum_system().set_nonlocal_effective_potential(
            &nonlocal_effective_potential_);
    }

    void construct_density_from_states() override {
        auto timer_us_start = std::chrono::high_resolution_clock::now();
        PlanewavePseudopotentialKohnShamSCFIterator::construct_density_from_states();
        auto timer_us_base = std::chrono::high_resolution_clock::now();
        snapshot_smooth_density();
        auto timer_us_snap = std::chrono::high_resolution_clock::now();
        accumulate_projector_overlaps();
        auto timer_us_projector_overlaps = std::chrono::high_resolution_clock::now();
        add_augmentation_to_charge_density();
        auto timer_us_aug = std::chrono::high_resolution_clock::now();
        warn_if_total_charge_deviates();

        log::debug("     US density sub-timing:");
        log::debug(
            "       base_accumulate:   {:.1f} ms",
            std::chrono::duration<double, std::milli>(timer_us_base - timer_us_start).count());
        log::debug(
            "       snapshot_smooth:   {:.1f} ms",
            std::chrono::duration<double, std::milli>(timer_us_snap - timer_us_base).count());
        log::debug(
            "       projector_overlaps:{:.1f} ms",
            std::chrono::duration<double, std::milli>(timer_us_projector_overlaps - timer_us_snap)
                .count());
        log::debug(
            "       augmentation:      {:.1f} ms",
            std::chrono::duration<double, std::milli>(timer_us_aug - timer_us_projector_overlaps)
                .count());
    }

    void symmetrize_density() override {
        /**
         * If augmentation cached the G-space density, use the G-space symmetrizer
         * to avoid a redundant R→G FFT. This saves ~100ms per iteration.
         */
        if (!cached_augmented_density_g_.empty()) {
            density_symmetrizer_.symmetrize_from_g_space(planewave_basis_,
                                                         cached_augmented_density_g_,
                                                         charge_density_real_,
                                                         number_of_spin_components());
            cached_augmented_density_g_.clear();
            return;
        }
        // Fallback to base class R-space symmetrization
        PlanewavePseudopotentialKohnShamSCFIterator::symmetrize_density();
    }

    std::vector<std::vector<Real>> input_density_for_mixing() const override {
        return charge_density_real_;
    }

    void synchronize_smooth_density_from_mixed() override {
        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            charge_density_real_smooth_[spin_index] = charge_density_real_[spin_index];
        }
    }

    void on_charge_density_initialized() override {
        reset_projector_overlap_coefficients();
        charge_density_real_smooth_.resize(number_of_spin_components());
        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            charge_density_real_smooth_[spin_index] = charge_density_real_[spin_index];
        }
    }

    void snapshot_output_density() override {
        PlanewavePseudopotentialKohnShamSCFIterator::snapshot_output_density();
        convergence_output_density_smooth_ = charge_density_real_smooth_;
    }

    void restore_density_at_convergence() override {
        PlanewavePseudopotentialKohnShamSCFIterator::restore_density_at_convergence();
        charge_density_real_smooth_ = convergence_output_density_smooth_;
    }

    // ─── Ultrasoft density construction ──────────────────────────────
    /// Allocate and zero the per-atom projector overlap matrix when augmentation is required.
    void reset_projector_overlap_coefficients() {
        const auto& pseudopotential_snapshot = *quantum_system_snapshot_;
        int number_of_atoms = crystal_structure().number_of_atoms();
        int maximum_projector_channels = 0;
        for (int species_index = 0; species_index < crystal_structure().number_of_species();
             ++species_index) {
            maximum_projector_channels = std::max(
                maximum_projector_channels,
                pseudopotential_snapshot.pseudopotential_basis().number_of_projector_channels(
                    species_index));
        }
        projector_overlap_coefficients_.assign(
            number_of_atoms, maximum_projector_channels * maximum_projector_channels, 0.0);
        projector_overlap_coefficients_by_spin_.assign(number_of_spin_components(), {});
        for (auto& per_spin : projector_overlap_coefficients_by_spin_)
            per_spin.assign(
                number_of_atoms, maximum_projector_channels * maximum_projector_channels, 0.0);
    }

    void snapshot_smooth_density() {
        charge_density_real_smooth_.resize(number_of_spin_components());
        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            charge_density_real_smooth_[spin_index] = charge_density_real_[spin_index];
        }
    }

    /**
     * Accumulate the per-atom projector overlap coefficients from the bands.
     *
     * Uses BLAS zgemm to compute all <beta|psi> overlaps in a single call
     * instead of per-projector-per-band zdotc calls.
     */
    /**
     * Virtual so PAW can damp the result before it is used.
     *
     * The coefficients feed two things — the augmentation charge on the grid and, for
     * PAW, the on-site D matrix — and the second closes a loop that the
     * density mixer cannot reach: ρ_ij is recomputed from fresh ψ every step,
     * so its feedback gain is 1 no matter how small the mixing beta. That is
     * harmless while the on-site terms are small, and not harmless for a
     * dataset with a stiff channel, which is why PAW overrides this.
     */
    virtual void accumulate_projector_overlaps() {
        const auto& pseudopotential_snapshot = *quantum_system_snapshot_;
        reset_projector_overlap_coefficients();

        int number_of_kpoints = kpoint_grid().number_of_kpoints();
        int number_of_kpoint_states = number_of_kpoints * number_of_spin_components();
        int total_projectors =
            pseudopotential_snapshot.pseudopotential_basis().total_number_of_projectors();

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             ++kpoint_index_state) {
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            Real kpoint_weight = kpoint_grid()[kpoint_index].weight;
            int plane_waves_at_kpoint = number_of_plane_waves(kpoint_index);

            // Use cached KB projectors (computed once in solve_single_particle_states)
            const auto& kleinman_bylander_projectors =
                cached_kleinman_bylander_projectors_[kpoint_index];

            // Gather occupied band data and weights
            std::vector<int> occupied_band_indices;
            std::vector<Real> occupied_band_weights;
            for (int band_index = 0; band_index < number_of_bands_; ++band_index) {
                Real occupation = quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                    kpoint_index_state, band_index);
                if (occupation < math::tolerances::occupation_threshold) {
                    continue;
                }
                occupied_band_indices.push_back(band_index);
                occupied_band_weights.push_back(kpoint_weight * occupation);
            }
            int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());
            if (number_of_occupied_bands == 0) {
                continue;
            }

            // Gather wavefunctions into a contiguous matrix: psi(pw, occupied_bands)
            std::vector<Complex> psi_matrix(static_cast<size_t>(plane_waves_at_kpoint) *
                                            number_of_occupied_bands);
            for (int occupied_index = 0; occupied_index < number_of_occupied_bands;
                 ++occupied_index) {
                const Complex* band_data = quantum_system_.wavefunction_band_data(
                    kpoint_index_state, occupied_band_indices[occupied_index]);
                std::copy(band_data,
                          band_data + plane_waves_at_kpoint,
                          psi_matrix.data() +
                              static_cast<size_t>(occupied_index) * plane_waves_at_kpoint);
            }

            // Compute projector_overlaps(total_projectors, occupied_bands) = beta^H × psi
            std::vector<Complex> projector_overlaps(static_cast<size_t>(total_projectors) *
                                                        number_of_occupied_bands,
                                                    Complex(0.0, 0.0));

            if (planewave_basis_.is_gamma_only()) {
                int double_number_of_plane_waves = 2 * plane_waves_at_kpoint;
                double alpha_two = 2.0;
                double double_beta_zero = 0.0;

                std::vector<double> real_projector_overlaps(
                    static_cast<size_t>(total_projectors) * number_of_occupied_bands, 0.0);
                const double* real_kleinman_bylander_projectors =
                    reinterpret_cast<const double*>(kleinman_bylander_projectors.data());
                const double* real_psi_matrix = reinterpret_cast<const double*>(psi_matrix.data());

                dgemm_("T",
                       "N",
                       &total_projectors,
                       &number_of_occupied_bands,
                       &double_number_of_plane_waves,
                       &alpha_two,
                       real_kleinman_bylander_projectors,
                       &double_number_of_plane_waves,
                       real_psi_matrix,
                       &double_number_of_plane_waves,
                       &double_beta_zero,
                       real_projector_overlaps.data(),
                       &total_projectors);

                /**
                 * G=0 correction. The 2 Re sum above counts the origin twice, so
                 * the rank whose plane-wave list begins with G = 0 removes one
                 * copy. The density list cannot answer that: it is complete on
                 * every rank, so asking it whether G = 0 comes first says yes on
                 * every rank of a shared pool, and each rank without the origin
                 * subtracted its first stored plane wave instead.
                 */
                bool has_zero_g_vector = planewave_basis_.gamma_origin_is_local();
                if (has_zero_g_vector) {
                    for (int occupied_index = 0; occupied_index < number_of_occupied_bands;
                         ++occupied_index) {
                        Complex psi_g0 =
                            psi_matrix[static_cast<size_t>(occupied_index) * plane_waves_at_kpoint];
                        for (int projector_index = 0; projector_index < total_projectors;
                             ++projector_index) {
                            Complex kleinman_bylander_g0 =
                                kleinman_bylander_projectors[projector_index][0];
                            size_t index = static_cast<size_t>(projector_index) +
                                           static_cast<size_t>(occupied_index) * total_projectors;
                            real_projector_overlaps[index] -=
                                (std::conj(kleinman_bylander_g0) * psi_g0).real();
                        }
                    }
                }

                for (size_t index = 0; index < real_projector_overlaps.size(); ++index) {
                    projector_overlaps[index] = Complex(real_projector_overlaps[index], 0.0);
                }
            } else {
                Complex alpha_blas(1.0, 0.0);
                Complex beta_blas(0.0, 0.0);
                auto* cast_alpha = reinterpret_cast<const std::complex<double>*>(&alpha_blas);
                auto* cast_beta_coeff = reinterpret_cast<const std::complex<double>*>(&beta_blas);
                auto* cast_kb = reinterpret_cast<const std::complex<double>*>(
                    kleinman_bylander_projectors.data());
                auto* cast_psi = reinterpret_cast<const std::complex<double>*>(psi_matrix.data());
                auto* cast_projector_overlaps =
                    reinterpret_cast<std::complex<double>*>(projector_overlaps.data());

                zgemm_("C",
                       "N",
                       &total_projectors,
                       &number_of_occupied_bands,
                       &plane_waves_at_kpoint,
                       cast_alpha,
                       cast_kb,
                       &plane_waves_at_kpoint,
                       cast_psi,
                       &plane_waves_at_kpoint,
                       cast_beta_coeff,
                       cast_projector_overlaps,
                       &total_projectors);
            }

            /**
             * Band-parallel ranks own disjoint G-vector slabs, so BLAS has
             * produced only each rank's partial <beta|psi>. Complete those
             * amplitudes before forming rho_ij = <psi|beta_i><beta_j|psi>.
             * Reducing rho_ij afterward is not equivalent because it omits
             * cross-rank products. The later reduce_vector remains the
             * separate across-pool sum over distinct k-points.
             */
            if (execution_policy_ != nullptr) {
                execution_policy_->reduce_within_group(projector_overlaps.data(),
                                                       static_cast<int>(projector_overlaps.size()));
            }

            // Accumulate the projector overlap coefficients from projector_overlaps
            int current_projector_index = 0;
            for (int atom_index = 0; atom_index < crystal_structure().number_of_atoms();
                 ++atom_index) {
                int species_type = crystal_structure().species_type_index(atom_index);
                int number_of_projectors_on_atom =
                    pseudopotential_snapshot.pseudopotential_basis().number_of_projector_channels(
                        species_type);

                for (int projector_a_index = 0; projector_a_index < number_of_projectors_on_atom;
                     ++projector_a_index) {
                    for (int projector_b_index = 0;
                         projector_b_index < number_of_projectors_on_atom;
                         ++projector_b_index) {
                        Real sum = 0.0;
                        int global_a = current_projector_index + projector_a_index;
                        int global_b = current_projector_index + projector_b_index;
                        for (int occupied_index = 0; occupied_index < number_of_occupied_bands;
                             ++occupied_index) {
                            /**
                             * projector_overlaps is column-major (total_projectors ×
                             * occupied_bands)
                             */
                            Complex projector_overlaps_a =
                                projector_overlaps[static_cast<size_t>(global_a) +
                                                   static_cast<size_t>(occupied_index) *
                                                       total_projectors];
                            Complex projector_overlaps_b =
                                projector_overlaps[static_cast<size_t>(global_b) +
                                                   static_cast<size_t>(occupied_index) *
                                                       total_projectors];
                            sum +=
                                occupied_band_weights[occupied_index] *
                                std::real(std::conj(projector_overlaps_a) * projector_overlaps_b);
                        }
                        int coefficient_index =
                            projector_a_index * number_of_projectors_on_atom + projector_b_index;
                        projector_overlap_coefficients_[atom_index][coefficient_index] += sum;
                        projector_overlap_coefficients_by_spin_[kpoint_index_state /
                                                                number_of_kpoints][atom_index]
                                                               [coefficient_index] += sum;
                    }
                }
                current_projector_index += number_of_projectors_on_atom;
            }
        }

        // Reduce projector overlap coefficients across all ranks
        if (execution_policy_ != nullptr) {
            execution_policy_->reduce_vector(projector_overlap_coefficients_.raw());
            for (auto& per_spin : projector_overlap_coefficients_by_spin_)
                execution_policy_->reduce_vector(per_spin.raw());
        }
    }

    /**
     * Add the compensation charge n̂ to the smooth density, per spin channel.
     *
     * The augmentation charge is built from ρ_ij, and ρ_ij is spin-resolved, so
     * each channel gets its own n̂ from its own occupancies. That matters far
     * more than it sounds: in a 3d metal almost the entire moment lives inside
     * the augmentation spheres, so how n̂ is divided between the channels *is*
     * the magnetization.
     *
     * This used to add one total n̂ and split it by the ratio of the smooth
     * densities, which forces the on-site polarization to match the
     * interstitial polarization — small, and nearly unpolarized. Nickel came
     * out at 1e-4 Bohr magnetons against a PBE moment near 0.6.
     */
    void add_augmentation_to_charge_density() {
        const auto& ultrasoft_system = ultrasoft_quantum_system();

        // Reuse cached structure factors from the D-matrix computation
        const auto* structure_factor_cache =
            nonlocal_effective_potential_.cached_structure_factors().empty()
                ? nullptr
                : &nonlocal_effective_potential_.cached_structure_factors();

        int number_of_spins = number_of_spin_components();
        cached_augmented_density_g_.resize(number_of_spins);
        for (int spin_index = 0; spin_index < number_of_spins; ++spin_index) {
            std::vector<Complex> density_g;
            planewave_basis_.rho_r_to_g(charge_density_real_[spin_index], density_g);

            /**
             * `projector_overlap_coefficients_by_spin_` is empty only on paths
             * that never accumulated it; falling back to the spin-summed array
             * then reproduces the old single-channel behaviour rather than
             * adding nothing.
             */
            const Matrix<Real>& occupancies =
                spin_index < static_cast<int>(projector_overlap_coefficients_by_spin_.size())
                    ? projector_overlap_coefficients_by_spin_[spin_index]
                    : projector_overlap_coefficients_;

            ultrasoft_system.ultrasoft_basis().add_augmentation_charges(
                crystal_structure(), occupancies, density_g, structure_factor_cache);

            planewave_basis_.rho_g_to_r(density_g, charge_density_real_[spin_index]);
            cached_augmented_density_g_[spin_index] = std::move(density_g);
        }
    }
};

} // namespace q3
