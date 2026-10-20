#pragma once

/**
 * KohnShamSCFIterator — Kohn-Sham DFT cycle.
 *
 * Owns charge density, effective potential, XC functional, mixer, occupations,
 * and the fixed Kohn-Sham iteration order. Configuration is provided via
 * KohnShamSCFIteratorConfig which extends SCFIteratorConfig.
 */

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/quantum_system/kohn_sham/init.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/factory.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"
#include "layer_4_computation/iterator/init.hpp"
#include "layer_4_computation/iterator/internal/mixing/factory.hpp"
#include "layer_4_computation/iterator/internal/mixing/init.hpp"
#include "layer_4_computation/iterator/internal/mixing/preconditioner/factory.hpp"
#include "layer_4_computation/iterator/internal/smearing/factory.hpp"
#include "layer_4_computation/iterator/internal/smearing/init.hpp"
#include "layer_4_computation/state/internal/energies/kohn_sham/init.hpp"

namespace q3 {

struct KohnShamSCFIteratorConfig : SCFIteratorConfig {
    // === Kohn-Sham level ===
    std::string exchange_correlation_functional = "pz";
    std::string mixing_mode = "broyden";
    Real mixing_beta = 0.7;
    int mixing_history_size = 8;
    std::string mixing_preconditioning = "default";
    Real mixing_q0 = 1.0;
    std::string smearing = "none";
    Real smearing_width = 0.0;
    int methfessel_paxton_order = 1;            // used only by Methfessel-Paxton smearing
    std::string dispersion_correction = "none"; // "none" or "d3bj"
    Real total_charge = 0.0;                    // electrons removed (QE tot_charge convention)
    std::vector<HubbardSpecification> hubbard;  // DFT+U (Dudarev); empty = no correction
    /// First-iteration occupation-eigenvalue overrides; empty = no seeding.
    std::vector<HubbardStartingEigenvalue> hubbard_starting_eigenvalues;
    /// Iterations for which the seeded occupation matrix is held; 0 = first iteration only.
    int hubbard_fixed_occupation_iterations = 0;
    /**
     * "atomic" (raw pseudo-atomic orbitals) or "ortho-atomic" (Lowdin-orthogonalized
     * over the full system-wide orbital set) -- see InputData::hubbard_projector_type.
     */
    std::string hubbard_projector_type = "atomic";

    // === Planewave level ===
    const KPointsGrid* kpoint_grid = nullptr;
    Real wavefunction_energy_cutoff = 30.0;  // Ry
    Real charge_density_energy_cutoff = 0.0; // Ry (0 → auto = 4 × wfc cutoff)
    int number_of_bands = 0;                 // 0 → auto
    std::string diagonalization_method = "davidson";
    int davidson_basis_expansion_factor = 2;
    /**
     * false (default) = FFTW_MEASURE (fastest, but per-process-timed and so
     * not bit-reproducible run to run); true = FFTW_ESTIMATE (deterministic).
     * See InputData::use_deterministic_fft_planning and TODOs.md §2.3.
     */
    bool use_deterministic_fft_planning = false;
    /**
     * false (default) = the faster, approximate 2 x ecutwfc wavefunction FFT grid;
     * true = the exact grid (4 x ecutwfc, as pw.x sizes it). See
     * InputData::use_exact_wavefunction_grid.
     */
    bool use_exact_wavefunction_grid = false;
    /**
     * true = use no symmetry at all. The k-point mesh is left unreduced by
     * build_global_kpoint_grid_from_input, and the iterator then leaves the density
     * unsymmetrized too, so a full-zone run is the raw-mesh answer (QE's nosym + noinv)
     * rather than the symmetrized-mesh one. See InputData::disable_symmetry.
     */
    bool disable_symmetry = false;

    // === Pseudopotential level ===
    const PseudopotentialSet* pseudopotential_set = nullptr;
    std::string calculation = "self_consistent_field";
    /**
     * Initial magnetization per species (parallel to the crystal's species list,
     * values in [-1, 1]; may be shorter — missing entries mean 0; empty = unset).
     */
    std::vector<Real> starting_magnetization;
    /**
     * Initial magnetization per atom (parallel to the crystal's atoms, values in
     * [-1, 1]); empty = unset. When set it takes precedence over the per-species
     * `starting_magnetization`.
     */
    std::vector<Real> atomic_starting_magnetization;
    std::vector<std::pair<std::string, Vector3D>> band_path;
    int band_path_points_per_segment = 20;
    std::string nonlocal_stress_derivative_method = "analytical";

    // === Implementation selectors ===
    std::string basis_type = "planewave";
    std::string representation_type = "pseudopotential";

    /**
     * === Parallelism (optional) ===
     * null means serial; set to an active PlanewaveExecutionPolicy for pool-based k-point
     * parallelism.
     */
    const execution::ExecutionPolicy* execution_policy = nullptr;
};

class KohnShamSCFIterator : public SCFIterator {
  public:
    explicit KohnShamSCFIterator(KohnShamSCFIteratorConfig config)
        : SCFIterator(config)
        , config_(std::move(config))
        , exchange_correlation_functional_(config_.exchange_correlation_functional)
        , mixing_mode_(config_.mixing_mode)
        , mixing_beta_(config_.mixing_beta)
        , mixing_history_size_(config_.mixing_history_size)
        , mixing_preconditioning_(config_.mixing_preconditioning)
        , mixing_q0_(config_.mixing_q0)
        , smearing_(config_.smearing)
        , smearing_width_(config_.smearing_width)
        , methfessel_paxton_order_(config_.methfessel_paxton_order) {
        if (mixing_preconditioning_ == "default") {
            if (smearing_ != "none") {
                mixing_preconditioning_ = "kerker";
            } else {
                mixing_preconditioning_ = "none";
            }
        }
        /**
         * Validate here, where "default" has just been resolved and before any
         * representation exists to build a preconditioner from. An unrecognized
         * value used to fall through to a `== "kerker"` test further down and
         * silently disable preconditioning: `preconditioning: kerkar` cost a
         * slower run and produced no diagnostic to explain it.
         */
        if (!PreconditionerFactory::is_supported(mixing_preconditioning_)) {
            throw std::runtime_error("Unknown mixing preconditioner: '" + mixing_preconditioning_ +
                                     "'. Supported values: 'default', 'none', 'kerker', "
                                     "'local-tf'.");
        }
    }

    ~KohnShamSCFIterator() override = default;

    // Stored configuration — must be declared before fields that reference it in the ctor.
    KohnShamSCFIteratorConfig config_;

    // Configuration fields — initialized from config_ in the constructor.
    std::string exchange_correlation_functional_;
    std::string mixing_mode_;
    Real mixing_beta_;
    int mixing_history_size_;
    std::string mixing_preconditioning_;
    Real mixing_q0_;
    std::string smearing_;
    Real smearing_width_;
    int methfessel_paxton_order_;

    /// One row per SCF iteration, in order; the source for per-iteration companion outputs.
    struct ScfIterationRecord {
        int iteration;
        Real total_energy_in_ry;
        Real energy_change_in_ry;
        Real density_residual;
    };

    [[nodiscard]] const std::vector<ScfIterationRecord>& scf_history() const {
        return scf_history_;
    }

    virtual const KohnShamEnergies& energies() const = 0;
    virtual KohnShamEnergies& energies_mutable() = 0;

    [[nodiscard]] Real total_energy() const override {
        return energies().total();
    }
    [[nodiscard]] const XCFunctional& exchange_correlation() const {
        return *exchange_correlation_;
    }

    [[nodiscard]] const std::vector<Complex>& charge_density_g() const {
        return charge_density_g_;
    }
    [[nodiscard]] const std::vector<Real>& charge_density_real() const {
        return charge_density_real_[0];
    }
    [[nodiscard]] const std::vector<Real>& charge_density_for_spin(int spin_index) const {
        return charge_density_real_[spin_index];
    }
    [[nodiscard]] const std::vector<Real>& real_space_potential() const {
        return real_space_potential_[0];
    }
    [[nodiscard]] const std::vector<Real>& potential_for_spin(int spin_index) const {
        return real_space_potential_[spin_index];
    }
    [[nodiscard]] const std::vector<Complex>& total_potential_g() const {
        return total_potential_g_;
    }
    [[nodiscard]] const std::vector<Complex>& exchange_correlation_potential_g() const {
        return exchange_correlation_potential_g_;
    }

    /**
     * Access the crystal structure owned or referenced by this iterator.
     */
    [[nodiscard]] Crystal& crystal_structure() const {
        return *crystal_structure_;
    }

    /**
     * Load a pre-computed effective potential directly into real_space_potential_.
     *
     * Used by the NSCF flavor to inject V_eff loaded from a v2 checkpoint
     * instead of computing it from scratch via compute_effective_potential_from_density().
     * Must be called after iterator construction but before run().
     */
    void load_effective_potential(std::vector<std::vector<Real>> effective_potential) {
        real_space_potential_ = std::move(effective_potential);
        has_preloaded_effective_potential_ = true;
    }

    void load_charge_density(std::vector<std::vector<Real>> charge_density) {
        charge_density_real_ = std::move(charge_density);
        has_preloaded_charge_density_ = true;
        preloaded_charge_density_is_a_starting_guess_ = false;
    }

    /**
     * Offer a density to start from, as a previous geometry's saved one is
     * offered (startingpot = 'file'). Unlike load_charge_density(), which asks
     * for exactly that density, an offer this run's grid cannot take is
     * declined, and the SCF starts from overlapping atoms instead; one it can
     * take is rescaled to this system's electron count, since the cell may
     * have changed since it was saved.
     */
    void offer_starting_charge_density(std::vector<std::vector<Real>> charge_density) {
        charge_density_real_ = std::move(charge_density);
        has_preloaded_charge_density_ = true;
        preloaded_charge_density_is_a_starting_guess_ = true;
    }

  protected:
    [[nodiscard]] const KohnShamSCFIteratorConfig& config() const {
        return config_;
    }

    std::vector<Complex> charge_density_g_;
    std::vector<std::vector<Real>> charge_density_real_;
    std::vector<std::vector<Real>> real_space_potential_;
    std::vector<Complex> total_potential_g_;

    /**
     * The same potential per spin channel, un-averaged.
     *
     * `total_potential_g_` above is the spin average, which the nonlocal D was
     * historically built from. That is exact for a non-magnetic cell and wrong
     * for a magnetic one, because D_ij = D_ion + ∫v_eff·Q_ij inherits v_eff's
     * spin dependence. Empty on paths that never filled it, in which case
     * consumers fall back to the average.
     */
    std::vector<std::vector<Complex>> total_potential_g_by_spin_;
    std::vector<Complex> exchange_correlation_potential_g_;

    /**
     * True when V_eff has been injected via load_effective_potential().
     * When set, update_effective_potential() must not overwrite real_space_potential_.
     */
    bool has_preloaded_effective_potential_{false};
    bool has_preloaded_charge_density_{false};
    /// Set by offer_starting_charge_density(): a density that need not fit.
    bool preloaded_charge_density_is_a_starting_guess_{false};

    std::unique_ptr<XCFunctional> exchange_correlation_;
    std::unique_ptr<Mixer> mixer_;
    std::unique_ptr<SmearingScheme> smearing_scheme_;

    // KS-level iteration state
    Real density_residual_{0.0};
    Real total_energy_previous_{0.0};
    std::vector<ScfIterationRecord> scf_history_;
    bool first_iteration_rediag_done_{false};
    Real diagonalizer_tolerance_{0.0};

    /**
     * The eigenvalue threshold the diagonalizer runs at: the schedule's value,
     * held at the floor below which the solvers resolve no further change.
     */
    [[nodiscard]] Real diagonalization_threshold() const {
        return std::max(diagonalizer_tolerance_, math::tolerances::minimum_eigenvalue_tolerance);
    }

    void initialize() override {
        total_energy_previous_ = 0.0;
        density_residual_ = 1.0;

        exchange_correlation_ = XCFunctionalFactory::create(exchange_correlation_functional_,
                                                            number_of_spin_components());
        mixer_ = MixerFactory::create(mixing_mode_, mixing_beta_, mixing_history_size_);
        smearing_scheme_ = create_smearing(smearing_, smearing_width_, methfessel_paxton_order_);

        auto timer_start = std::chrono::high_resolution_clock::now();
        initialize_representation();
        auto timer_repr = std::chrono::high_resolution_clock::now();
        initiate_mixer();
        initialize_charge_density();
        auto timer_charge = std::chrono::high_resolution_clock::now();
        initialize_single_particle_states();
        auto timer_wfc = std::chrono::high_resolution_clock::now();

        /**
         * Keep the atomic superposition density as the input for the first SCF iteration,
         * matching QE's ordering — potential from the atomic superposition, then wavefunction
         * initialization from that potential, then the SCF loop. The effective potential
         * from the atomic density was already computed during initialize_single_particle_states().
         * Do NOT reconstruct density from wavefunctions here — doing so creates a nearly
         * self-consistent wfc-based density that gives the Broyden mixer a near-zero first
         * residual, corrupting its history and causing ~30 extra iterations.
         */

        log::debug("     Init timing breakdown:");
        log::debug("       representation:      {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_repr - timer_start).count());
        log::debug("       charge_density+mixer: {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_charge - timer_repr).count());
        log::debug("       wfc_init:            {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_wfc - timer_charge).count());

        SCFIterator::initialize(); // prints SCF banner
    }

    void perform_iteration() override {
        auto timer_iter_start = std::chrono::high_resolution_clock::now();
        auto mixing_input_density = input_density_for_mixing();

        // 1. Solve single-particle states using the current potential
        prepare_hamiltonian();
        auto timer_prep_end = std::chrono::high_resolution_clock::now();
        set_diagonalization_tolerance(iteration_);
        solve_single_particle_states();
        auto timer_solve_end = std::chrono::high_resolution_clock::now();
        ProfilingCounters::instance().log_resident_checkpoint_debug("solve_states");
        update_occupations();
        auto timer_occupations_end = std::chrono::high_resolution_clock::now();

        // 2. Construct new density from wavefunctions
        construct_density_from_states();
        auto timer_density_end = std::chrono::high_resolution_clock::now();
        symmetrize_density();
        auto timer_symm_end = std::chrono::high_resolution_clock::now();

        /**
         * 3. Compute double-counting correction using the CURRENT potential
         *    (V_Hxc built from previous iteration's mixed density) and the NEW output density.
         *    This is the band-energy double-counting term as QE defines it:
         *    deband = -Σ ρ_out(r) · V_Hxc(r)
         */
        energies_mutable().double_counting_correction = compute_double_counting_correction();
        compute_energy_components();

        // 4. Compute residual and check convergence
        density_residual_ = compute_density_residual(mixing_input_density);

        /**
         * 4a. First-iteration re-diagonalization (matching QE's SCF loop):
         * If the density residual is smaller than the estimated diagonalization error,
         * the eigenvalue threshold was too loose — re-diag with a tighter threshold.
         * This is critical for avoiding the false fixed-point trap in small systems.
         *
         * The error estimate is the threshold the diagonalization actually ran
         * at, times the electron count, as in QE's electrons_scf
         * (PW/src/electrons.f90), and only a diagonalization that ran looser
         * than the residual now asks for is repeated. A self-consistent run's
         * first diagonalization runs at the loose initial threshold, so this is
         * the 1e-2 rule it always was. A non-self-consistent pass has already
         * run at the floor: there the "tighter" threshold was 0.1 dr2 / N,
         * 7.8e-4 Ry for silicon's two irreducible k-points, and the repeat
         * loosened the pass it was meant to sharpen -- and, seeding each
         * k-point from the one before, left the second up to 4.4e-5 Ry off in
         * its occupied bands.
         */
        if (iteration_ == 1 && !first_iteration_rediag_done_) {
            const Real threshold_used = diagonalization_threshold();
            Real diagonalization_error_estimate =
                threshold_used * std::max(1.0, number_of_electrons());
            Real tighter_threshold =
                std::max(0.1 * density_residual_ / std::max(1.0, number_of_electrons()),
                         math::tolerances::minimum_eigenvalue_tolerance);
            if (density_residual_ < diagonalization_error_estimate &&
                tighter_threshold < threshold_used) {
                log::info("     Re-diagonalizing with tighter threshold "
                          "(dr2={:.2e} < tr2_min={:.2e})",
                          density_residual_,
                          diagonalization_error_estimate);
                first_iteration_rediag_done_ = true;

                /**
                 * Re-compute with tighter tolerance using the SAME potential
                 * (matching QE, which restarts the SCF step without mixing
                 * before the re-diagonalization)
                 */
                set_diagonalizer_tolerance(tighter_threshold);

                prepare_hamiltonian();
                solve_single_particle_states();
                update_occupations();

                // Recompute density/energy with the improved wavefunctions
                construct_density_from_states();
                symmetrize_density();
                energies_mutable().double_counting_correction =
                    compute_double_counting_correction();
                compute_energy_components();
                density_residual_ = compute_density_residual(mixing_input_density);

                // Reset mixer history — the coarse first-iteration data is misleading
                mixer_->reset();
            }
        }
        auto timer_residual_end = std::chrono::high_resolution_clock::now();

        auto log_iteration_timing = [&](auto timer_mix_end, auto timer_potential_end) {
            log::debug("     Iteration timing breakdown:");
            log::debug("       prepare_hamiltonian: {:.1f} ms",
                       std::chrono::duration<double, std::milli>(timer_prep_end - timer_iter_start)
                           .count());
            log::debug("       solve_states:        {:.1f} ms",
                       std::chrono::duration<double, std::milli>(timer_solve_end - timer_prep_end)
                           .count());
            log::debug(
                "       update_occupations:  {:.1f} ms",
                std::chrono::duration<double, std::milli>(timer_occupations_end - timer_solve_end)
                    .count());
            log::debug(
                "       construct_density:   {:.1f} ms",
                std::chrono::duration<double, std::milli>(timer_density_end - timer_occupations_end)
                    .count());
            log::debug("       symmetrize_density:  {:.1f} ms",
                       std::chrono::duration<double, std::milli>(timer_symm_end - timer_density_end)
                           .count());
            log::debug(
                "       deband+residual:     {:.1f} ms",
                std::chrono::duration<double, std::milli>(timer_residual_end - timer_symm_end)
                    .count());
            log::debug("       mix+sync:            {:.1f} ms",
                       std::chrono::duration<double, std::milli>(timer_mix_end - timer_residual_end)
                           .count());
            log::debug(
                "       update_potential:    {:.1f} ms",
                std::chrono::duration<double, std::milli>(timer_potential_end - timer_mix_end)
                    .count());
            log::debug(
                "       total_iteration:     {:.1f} ms",
                std::chrono::duration<double, std::milli>(timer_potential_end - timer_iter_start)
                    .count());
            ProfilingCounters::instance().release_unused_heap("SCF iteration");
            ProfilingCounters::instance().log_resident_checkpoint_debug("SCF iteration");
        };

        snapshot_output_density();

        if (is_iteration_converged()) {
            log_iteration_timing(timer_residual_end, timer_residual_end);
            return; // Don't mix — will be handled by finalize()
        }

        // 5. Mix density to get next input density
        mix_density(mixing_input_density);

        synchronize_smooth_density_from_mixed();
        auto timer_mix_end = std::chrono::high_resolution_clock::now();

        /**
         * 6. Update effective potential from the MIXED density (single call per iteration).
         *    This produces V_Hxc for the next iteration's Hamiltonian AND provides
         *    E_Hartree and E_XC evaluated at the mixed density.
         */
        update_effective_potential();
        auto timer_pot_end = std::chrono::high_resolution_clock::now();

        /**
         * 7. Compute delta_escf: first-order energy correction for using mixed
         *    density rather than output density in E_Hartree and E_XC.
         *    delta_escf = -Σ_r (ρ_mixed(r) - ρ_out(r)) · V_Hxc^mixed(r)
         *    (matches QE's SCF-correction term of the same name)
         */
        Real delta_escf = compute_delta_escf_correction();

        /**
         * The total energy reported is:
         *   E = E_band + deband + E_Hartree(mixed) + E_XC(mixed)
         *       + E_Ewald + E_smearing(-TS) + delta_escf
         * which gives a variational estimate using the output density. The -TS
         * term is added by PlanewaveKohnShamEnergies::total(), not here; it is
         * zero unless smearing is configured. Every term above is present in
         * total(), so this line and that function must be kept in step.
         */
        energies_mutable().double_counting_correction += delta_escf;

        log_iteration_timing(timer_mix_end, timer_pot_end);
    }

    bool is_iteration_converged() const override {
        return iteration_ > 1 && density_residual_ < convergence_threshold_;
    }

    void finalize() override {
        restore_density_at_convergence();
        update_effective_potential();
        compute_energy_components();
    }

    void report_iteration() override {
        if (iteration_ == 1) {
            report_first_iteration_details();
            scf_history_.push_back({iteration_, energies().total(), 0.0, density_residual_});
        } else {
            Real delta_energy = energies().total() - total_energy_previous_;
            log_iteration_energy(delta_energy);
        }
        total_energy_previous_ = energies().total();
    }

    /// Single source of truth for the per-iteration energy/convergence log line.
    void log_iteration_energy(Real delta_energy) {
        log::info(
            "     iteration # {:3d}     Etot = {:14.9f} Ry   delta(E) = {:.2e}   dr2 = {:.2e}",
            iteration_,
            energies().total(),
            delta_energy,
            density_residual_);
        /**
         * Recorded here rather than at the call sites so the history and the
         * printed line can never disagree — a per-iteration output that says
         * something different from stdout is worse than none.
         */
        scf_history_.push_back({iteration_, energies().total(), delta_energy, density_residual_});
    }

    void report_convergence() override {
        Real delta_energy = energies().total() - total_energy_previous_;
        log_iteration_energy(delta_energy);

        log::info("");
        log::info("     convergence achieved after {} iterations", iteration_);

        report_convergence_details();
    }

    void report_non_convergence() override {
        /**
         * A non-self-consistent pass is one diagonalization by design, so
         * stopping after it is not a failure to converge. The warning made
         * every NSCF log read as though the run had failed.
         */
        if (config_.calculation == "non_self_consistent_field") {
            log::info("     non-self-consistent pass complete");
            return;
        }
        log::warn("");
        log::warn("     WARNING: SCF did not converge after {} iterations", maximum_iterations_);
        log::info("     total energy       = {:14.9f}", energies().total());
        log::info("");
    }

    virtual void initialize_representation() = 0;
    virtual void initialize_charge_density() = 0;
    virtual void initialize_single_particle_states() = 0;
    virtual void initiate_mixer() = 0;

    virtual std::vector<std::vector<Real>> input_density_for_mixing() const = 0;
    virtual void update_effective_potential() = 0;
    virtual void prepare_hamiltonian() {}
    virtual void set_diagonalization_tolerance(int iteration) = 0;
    virtual void set_diagonalizer_tolerance(Real tolerance) = 0;
    virtual void solve_single_particle_states() = 0;
    virtual void update_occupations() = 0;
    virtual void construct_density_from_states() = 0;
    virtual void symmetrize_density() = 0;
    virtual Real compute_density_residual(
        const std::vector<std::vector<Real>>& input_density_for_mixing) const = 0;
    virtual void mix_density(const std::vector<std::vector<Real>>& input_density_for_mixing) = 0;

    /**
     * Hook for representations that keep an auxiliary density (e.g. the smooth
     * pre-augmentation density of ultrasoft pseudopotential_set). No-op by default.
     */
    virtual void synchronize_smooth_density_from_mixed() {}
    virtual void snapshot_output_density() = 0;
    virtual void restore_density_at_convergence() = 0;

    virtual Real compute_double_counting_correction() const = 0;

    /**
     * Compute the first-order energy correction for evaluating E_Hartree
     * and E_XC at the mixed density instead of the output density.
     *
     * delta_escf = -Σ_r (ρ_mixed(r) - ρ_out(r)) · V_Hxc^mixed(r) · (Ω/N)
     *
     * Matches QE's SCF-correction term of the same name. Must be called AFTER
     * update_effective_potential() has been applied to the mixed density
     * and AFTER snapshot_output_density() has saved the output density.
     */
    virtual Real compute_delta_escf_correction() const = 0;

    /**
     * Compute sub-components (e.g. band energy, entropy) needed for total().
     * Default no-op; override in representations that compute eagerly.
     */
    virtual void compute_energy_components() {}

    virtual void report_first_iteration_details() = 0;

    /**
     * Representation-specific tail of the convergence report (eigenvalues,
     * magnetization, final energy breakdown). Generic banner lives in report_convergence().
     */
    virtual void report_convergence_details() = 0;

    virtual int number_of_bands() const = 0;
    virtual Real number_of_electrons() const = 0;
    virtual int number_of_kpoint_states() const = 0;
};

} // namespace q3
