#pragma once

/**
 * PlanewaveKohnShamSCFIterator — abstract plane-wave Kohn-Sham SCF layer.
 *
 * Owns the plane-wave basis, per-k-point G-vector mappings, wavefunctions,
 * eigenvalues, the eigensolver, and FFT-based density/potential operations.
 * Implements all SCF hooks that are purely plane-wave:
 *   representation setup, mixing, occupations, density accumulation,
 *   symmetrization, double-counting correction, energy components,
 *   convergence reporting.
 *
 * Does NOT own pseudopotential-specific data (KB projectors, local ionic
 * potential, atomic superposition, core charge).  Those are deferred to
 * concrete subclasses such as PlanewavePseudopotentialKohnShamSCFIterator.
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_2_quantum_system/derived/magnetization.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_state/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/charge_density/planewave/init.hpp"
#include "layer_4_computation/charge_density/planewave/internal/accumulator.hpp"
#include "layer_4_computation/diagonalizer/planewave/factory.hpp"
#include "layer_4_computation/diagonalizer/planewave/init.hpp"
#include "layer_4_computation/effective_potential/internal/electrostatic/ewald_summation.hpp"
#include "layer_4_computation/effective_potential/planewave/init.hpp"
#include "layer_4_computation/fermi_energy/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/internal/energy_calculator.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/internal/reporter.hpp"
#include "layer_4_computation/state/internal/energies/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

class PlanewaveKohnShamSCFIterator : public KohnShamSCFIterator {
  public:
    explicit PlanewaveKohnShamSCFIterator(KohnShamSCFIteratorConfig config)
        : KohnShamSCFIterator(std::move(config)) {
        if (config_.execution_policy != nullptr) {
            execution_policy_ = const_cast<execution::PlanewaveExecutionPolicy*>(
                dynamic_cast<const execution::PlanewaveExecutionPolicy*>(config_.execution_policy));
        }
    }
    ~PlanewaveKohnShamSCFIterator() override = default;

    /**
     * Assemble and return a snapshot of the converged SCF state.
     *
     * Returns a self-contained state carrying all converged data needed by
     * post-SCF consumers (forces, stress, output writer, binary checkpoints).
     */
    virtual NormConservingPlanewaveKohnShamSCFState result() const = 0;

    /// Transfer ownership of a KPointsGrid into this iterator.
    void set_owned_kpoint_grid(KPointsGrid kpoint_grid) {
        owned_kpoint_grid_ = std::move(kpoint_grid);
        kpoint_grid_ = &owned_kpoint_grid_.value();
    }

    /**
     * Transfer ownership of an execution policy into this iterator.
     *
     * After this call the iterator holds the policy for its lifetime and
     * uses it for pool-level collective operations (charge-density reduction,
     * scalar reduction).
     */
    void set_owned_execution_policy(std::unique_ptr<execution::PlanewaveExecutionPolicy> policy) {
        owned_execution_policy_ = std::move(policy);
        execution_policy_ = owned_execution_policy_.get();
    }

    /**
     * Inject an externally-constructed diagonalizer.
     *
     * When set before run(), the iterator skips its internal
     * create_planewave_diagonalizer() call and uses this one instead.
     * Ownership is transferred to the iterator.
     */
    void set_diagonalizer(std::unique_ptr<PlanewaveDiagonalizer> diagonalizer) {
        diagonalizer_ = std::move(diagonalizer);
    }

    /**
     * Release what only the SCF needed, once it has converged: the solver's
     * working basis and the mixing history. The forces and stress that follow
     * work from the wavefunctions and density in the quantum system, and they
     * allocate arrays of their own -- on the 54-atom silicon cell at two ranks
     * the forces stage set the run's peak 0.3 GiB above the SCF's, because
     * the SCF's working set was still alive under it. A subclass adds the
     * caches it owns. A later run() rebuilds everything on demand.
     */
    virtual void release_scf_workspace() {
        if (diagonalizer_) {
            diagonalizer_->release_workspace();
        }
        if (mixer_) {
            mixer_->reset();
        }
        ProfilingCounters::instance().release_unused_heap("SCF workspace release");
    }

    [[nodiscard]] const execution::PlanewaveExecutionPolicy& execution_policy() const {
        if (execution_policy_ == nullptr) {
            throw std::runtime_error("PlanewaveKohnShamSCFIterator::execution_policy(): null.");
        }
        return *execution_policy_;
    }

    /**
     * Set the preloaded quantum system snapshot.
     */
    void load_quantum_system(const PlanewaveKohnShamQuantumSystem& system) {
        quantum_system_.load_quantum_system(system);
    }

    /**
     * Preload single-particle states from the cached quantum system snapshot.
     */
    void preload_single_particle_states() {
        if (quantum_system_.preloaded_quantum_system_ != nullptr) {
            log::debug("Planewave preload: Copying states from preloaded quantum system.");
            quantum_system_.copy_quantum_states_from(*quantum_system_.preloaded_quantum_system_);
            update_effective_potential();
            prepare_hamiltonian();
        } else {
            log::debug("Planewave preload: preloaded_quantum_system_ is null.");
        }
    }

    /**
     * Whether the loaded density has this run's spin components and this
     * rank's share of its charge-density grid. Every rank of a run reads the
     * same saved file, so the answer is the same on every rank: a whole grid
     * fits ranks that each hold one, and no rank of a pool split into slabs.
     */
    [[nodiscard]] bool loaded_charge_density_fits_grid() const {
        const auto grid_points =
            static_cast<std::size_t>(planewave_basis_.fft_rho().local_grid_points());
        if (static_cast<int>(charge_density_real_.size()) != number_of_spin_components()) {
            return false;
        }
        for (const auto& spin_density : charge_density_real_) {
            if (spin_density.size() != grid_points) {
                return false;
            }
        }
        return true;
    }

    /// Rescale the loaded density so it integrates to this system's electron count.
    void scale_loaded_charge_density_to_electron_count() {
        Real integrated_electrons = 0.0;
        for (const auto& spin_density : charge_density_real_) {
            for (const Real value : spin_density) {
                integrated_electrons += value;
            }
        }
        integrated_electrons *=
            crystal_structure().cell_volume() / planewave_basis_.fft_rho().total_grid_points();
        if (integrated_electrons <= 0.0) {
            return;
        }
        const Real scale = number_of_electrons() / integrated_electrons;
        for (auto& spin_density : charge_density_real_) {
            for (Real& value : spin_density) {
                value *= scale;
            }
        }
    }

    /**
     * Preload charge density from the cached charge_density_real_.
     */
    void preload_charge_density() {
        log::debug("Planewave preload: Transforming real charge density to G-space.");
        int grid_points = planewave_basis_.fft_rho().total_grid_points();
        std::vector<Real> total_charge_density_real(grid_points, 0.0);
        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            for (int grid_index = 0; grid_index < grid_points; ++grid_index) {
                total_charge_density_real[grid_index] +=
                    charge_density_real_[spin_index][grid_index];
            }
        }
        planewave_basis_.rho_r_to_g(total_charge_density_real, charge_density_g_);
    }

    // ─── Planewave accessors ─────────────────────────────────────────

    /**
     * Access the k-point grid.
     */
    [[nodiscard]] const KPointsGrid& kpoint_grid() const {
        return *kpoint_grid_;
    }

    /**
     * Access the plane-wave basis set.
     */
    [[nodiscard]] const PlanewaveBasis& planewave_basis() const {
        return planewave_basis_;
    }

    /**
     * Access the generic Kohn-Sham quantum system.
     */
    [[nodiscard]] const PlanewaveKohnShamQuantumSystem& quantum_system() const {
        return quantum_system_;
    }

    /**
     * Access a single wavefunction coefficient Ψ_{k,n}(G).
     */
    [[nodiscard]] Complex wavefunction(int kpoint_index, int band_index, int g_vector_index) const {
        return quantum_system_.wavefunction(kpoint_index, band_index, g_vector_index);
    }

    /**
     * Access eigenvalues as a Matrix (for compatibility).
     */
    [[nodiscard]] Matrix<Real> eigenvalues() const {
        return quantum_system_.eigenvalues_matrix();
    }

    /**
     * The eigenvalues of every k-point state in the run, in global order (Ry).
     *
     * eigenvalues() holds only this pool's k-point states. The rows here are
     * spin-major over the global grid, as a serial run's are. Collective under
     * MPI: every rank must call it.
     */
    [[nodiscard]] Matrix<Real> global_eigenvalues() const {
        const Matrix<Real> local_eigenvalues = eigenvalues();
        const int number_of_bands = static_cast<int>(local_eigenvalues.cols());
        std::vector<Real> global_data =
            execution_policy_->assemble_global_kpoint_state_data(local_eigenvalues.raw(),
                                                                 kpoint_grid().number_of_kpoints(),
                                                                 number_of_spin_components(),
                                                                 number_of_bands);
        Matrix<Real> eigenvalues_in_global_order(global_data.size() / number_of_bands,
                                                 number_of_bands);
        eigenvalues_in_global_order.raw() = std::move(global_data);
        return eigenvalues_in_global_order;
    }

    /**
     * Access a single eigenvalue ε_{k,n} (Ry).
     */
    [[nodiscard]] Real eigenvalue(int kpoint_index, int band_index) const {
        return quantum_system_.eigenvalue(kpoint_index, band_index);
    }

    /**
     * Access the occupation number f_{k,n} for a given k-point state and band.
     */
    [[nodiscard]] Real occupation(int kpoint_index, int band_index) const {
        return quantum_system_.occupation_by_kpoint_state_index_and_band_index(kpoint_index,
                                                                               band_index);
    }

    /**
     * G-vector index mapping for all k-points: g_vector_indices[k][i] → global G-vector index.
     */
    [[nodiscard]] const std::vector<std::vector<int>>& g_vector_indices() const {
        return planewave_basis_.kpoint_g_vector_indices();
    }

    /**
     * Kinetic energies |k+G|² for all k-points.
     */
    [[nodiscard]] const std::vector<std::vector<Real>>& kinetic_energy_squared() const {
        return planewave_basis_.kpoint_kinetic_energy_squared();
    }

    /**
     * Number of plane waves per k-point (full array).
     */
    [[nodiscard]] const std::vector<int>& number_of_plane_waves() const {
        return planewave_basis_.kpoint_number_of_plane_waves();
    }

    /**
     * Number of plane waves for a specific k-point.
     */
    [[nodiscard]] int number_of_plane_waves(int kpoint_index) const {
        return planewave_basis_.number_of_plane_waves_by_kpoint_index(kpoint_index);
    }

    /**
     * G-vector indices for a specific k-point.
     */
    [[nodiscard]] const std::vector<int>& g_vector_indices(int kpoint_index) const {
        return planewave_basis_.g_vector_indices_by_kpoint_index(kpoint_index);
    }

    /**
     * Total number of bands (user-specified or auto-determined).
     */
    [[nodiscard]] int number_of_bands() const override {
        return number_of_bands_;
    }

    /**
     * Total number of valence electrons from all atoms.
     */
    [[nodiscard]] Real number_of_electrons() const override {
        return number_of_electrons_;
    }

    /**
     * Total number of k-point states (k-points × spin components).
     */
    [[nodiscard]] int number_of_kpoint_states() const override {
        return kpoint_grid().number_of_kpoints() * number_of_spin_components();
    }

    /**
     * Number of fully occupied bands per spin channel.
     */
    [[nodiscard]] int number_of_occupied_bands_per_spin() const {
        return quantum_system_.number_of_occupied_bands_per_spin();
    }

    /**
     * Spin degeneracy factor (2.0 for non-spin-polarized, 1.0 for LSDA).
     */
    [[nodiscard]] Real spin_degeneracy() const {
        return quantum_system_.spin_degeneracy();
    }

    /**
     * ∫(ρ↑−ρ↓) and ∫|ρ↑−ρ↓| for this run, in Bohr magnetons.
     *
     * Present only for a spin-polarized run that reached its convergence
     * report, which is where they are integrated; empty elsewhere, so a
     * consumer can tell "not a magnetic calculation" from "a moment that came
     * out at zero" without asking which kind of run this was.
     */
    [[nodiscard]] const std::optional<Magnetization>& magnetization() const {
        return magnetization_;
    }

  protected:
    /// Set by report_convergence_details(); see magnetization() above.
    std::optional<Magnetization> magnetization_;

    // ─── Configuration read from KohnShamSCFIteratorConfig ──────────
    const KPointsGrid* kpoint_grid_ = nullptr;
    Real wavefunction_energy_cutoff_ = 30.0;
    Real charge_density_energy_cutoff_ = 0.0;
    std::string diagonalization_method_ = "davidson";

    /**
     * Execution policy for pool-level collective operations.
     * Set via set_owned_execution_policy() before run().
     */
    execution::PlanewaveExecutionPolicy* execution_policy_ = nullptr;

    /**
     * Optional owned execution policy — transfers lifetime from the factory.
     *
     * IMPORTANT: This member MUST be declared before planewave_basis_ and all
     * other MPI-using members. C++ destroys members in reverse declaration order,
     * so members declared AFTER this one are destroyed BEFORE it. The owned policy
     * calls MPI_Finalize() in its destructor; planewave_basis_ contains MPIFFTGrid
     * which calls fftw_destroy_plan (which internally frees MPI communicators) in
     * its destructor. If planewave_basis_ were destroyed after MPI_Finalize, OpenMPI
     * would report a fatal MPI_Comm_free-after-MPI_Finalize violation.
     */
    std::unique_ptr<execution::PlanewaveExecutionPolicy> owned_execution_policy_;

    // ─── Planewave fields ────────────────────────────────────────────
    PlanewaveBasis planewave_basis_;
    PlanewaveKohnShamQuantumSystem quantum_system_;

    Real number_of_electrons_{0.0};
    int number_of_bands_{0};

    std::unique_ptr<PlanewaveDiagonalizer> diagonalizer_;
    PlanewaveFermiEnergyCalculator fermi_energy_calculator_;
    PlanewaveChargeDensitySymmetrizer density_symmetrizer_;
    /// Accumulates real-space charge density from wavefunctions each SCF iteration.
    std::unique_ptr<PlanewaveChargeDensityAccumulator> charge_density_accumulator_;
    /// Computes band energy and smearing entropy from the occupied Kohn-Sham states.
    std::unique_ptr<PlanewaveKohnShamEnergyCalculator> energy_calculator_;

    /// Converged output density snapshot, restored once self-consistency is reached.
    std::vector<std::vector<Real>> convergence_output_density_;
    /// Cached real-space ionic local potential (constant during SCF, avoids repeated FFTs).
    std::vector<Real> cached_local_ionic_potential_real_;
    /**
     * Number of smooth G-vectors for density residual (matching QE's smooth-grid count).
     * For dual-grid US/PAW calculations, this excludes hard G-vectors above 4*ecutwfc.
     */
    int number_of_smooth_g_vectors_{0};

  private:
    /**
     * Optional owned KPointsGrid — used when the factory constructs the iterator
     * from InputData (the iterator owns the grid rather than borrowing it).
     */
    std::optional<KPointsGrid> owned_kpoint_grid_;

  protected:
    /**
     * When set (by set_diagonalizer_tolerance_for_nscf), overrides the initial
     * loose tolerance on the first call to set_diagonalization_tolerance().
     * Stored as a pending value because the diagonalizer_ is not yet constructed
     * when the NSCF flavor calls set_diagonalizer_tolerance_for_nscf() pre-run.
     */
    std::optional<Real> nscf_override_tolerance_;

    /**
     * The NSCF flavor's per-k-point Davidson cap, pending like the tolerance
     * above and applied at the same point.
     */
    std::optional<int> nscf_override_maximum_iterations_;

    /**
     * Set once the NSCF cap is applied: a single pass keeps whatever a solve
     * leaves unconverged, so each such solve is counted and reported.
     */
    bool report_unconverged_bands_ = false;

    // ─── PW-level energy helpers ─────────────────────────────────────

    [[nodiscard]] const PlanewaveKohnShamEnergies& planewave_energies() const {
        return static_cast<const PlanewaveKohnShamEnergies&>(energies());
    }
    PlanewaveKohnShamEnergies& planewave_energies_mutable() {
        return static_cast<PlanewaveKohnShamEnergies&>(energies_mutable());
    }

    // ─── KohnShamSCFIterator hooks ────────────────────────────────────

    void initialize_representation() override {
        // Read planewave config — skip fields already set by owned-data setters.
        if (config().kpoint_grid != nullptr && !owned_kpoint_grid_.has_value()) {
            kpoint_grid_ = config().kpoint_grid;
        }

        wavefunction_energy_cutoff_ = config().wavefunction_energy_cutoff;
        charge_density_energy_cutoff_ = config().charge_density_energy_cutoff;
        number_of_bands_ = config().number_of_bands;
        diagonalization_method_ = config().diagonalization_method;

        crystal_structure().compute_derived_quantities();
        Real effective_charge_density_cutoff = effective_charge_density_energy_cutoff();

        /**
         * Detect gamma-only: single k-point at Γ = (0,0,0) with no shift.
         * This enables half-sphere G-vector storage for ~2× speedup.
         *
         * Under a shared pool the half sphere is rebuilt inside every transform
         * from each G and its negative, so the FFT layout has to keep the two
         * on one rank. The stick grid does; the slab grid it replaced did not,
         * which is what used to force this off under MPI. The grid is asked
         * once it exists, below, rather than assumed here.
         */
        bool use_gamma_only = false;
        {
            const auto& grid_dims = kpoint_grid().grid_dimensions();
            const auto& grid_shifts = kpoint_grid().grid_shifts();
            if (grid_dims[0] == 1 && grid_dims[1] == 1 && grid_dims[2] == 1 &&
                grid_shifts[0] == 0 && grid_shifts[1] == 0 && grid_shifts[2] == 0) {
                use_gamma_only = true;
                log::info("     Gamma-only optimization enabled (half-sphere G-vectors)");
            }
        }
        if (execution_policy_ == nullptr) {
            owned_execution_policy_ = std::make_unique<execution::PlanewaveSerialExecutionPolicy>();
            execution_policy_ = owned_execution_policy_.get();
        }
        planewave_basis_.generate(*execution_policy_,
                                  crystal_structure(),
                                  wavefunction_energy_cutoff_,
                                  effective_charge_density_cutoff,
                                  use_gamma_only,
                                  !config_.use_deterministic_fft_planning,
                                  config_.use_exact_wavefunction_grid);
        if (use_gamma_only && !planewave_basis_.fft_wave().co_locates_mirrors()) {
            log::info("     Gamma-only disabled: this FFT layout separates a G-vector from its "
                      "negative, so the half sphere cannot be rebuilt locally");
            use_gamma_only = false;
            planewave_basis_.generate(*execution_policy_,
                                      crystal_structure(),
                                      wavefunction_energy_cutoff_,
                                      effective_charge_density_cutoff,
                                      use_gamma_only,
                                      !config_.use_deterministic_fft_planning,
                                      config_.use_exact_wavefunction_grid);
        }

        number_of_electrons_ = compute_number_of_electrons();
        number_of_bands_ = resolve_number_of_bands();

        if (!diagonalizer_) {
            diagonalizer_ =
                (execution_policy_ != nullptr)
                    ? create_planewave_diagonalizer(diagonalization_method_, *execution_policy_)
                    : create_planewave_diagonalizer(diagonalization_method_);
        }
        diagonalizer_->set_basis_expansion_factor(config().davidson_basis_expansion_factor);
        const auto* saved_preloaded = quantum_system_.preloaded_quantum_system_;

        /**
         * K-points are already distributed by the factory (set_owned_kpoint_grid)
         * and protected by the owned_kpoint_grid_.has_value() guard above.
         */

        quantum_system_ = PlanewaveKohnShamQuantumSystem(kpoint_grid(),
                                                         planewave_basis_,
                                                         number_of_bands_,
                                                         number_of_spin_components(),
                                                         number_of_electrons_);

        if (saved_preloaded != nullptr) {
            quantum_system_.load_quantum_system(*saved_preloaded);
        }

        fermi_energy_calculator_ = PlanewaveFermiEnergyCalculator(
            quantum_system_, smearing_scheme_.get(), execution_policy_);

        build_kpoint_plane_wave_data();
        allocate_electronic_states();
        ProfilingCounters::instance().log_resident_checkpoint("basis + wavefunctions");
        charge_density_accumulator_ = std::make_unique<PlanewaveChargeDensityAccumulator>(
            planewave_basis_, kpoint_grid(), quantum_system_);
        energy_calculator_ = std::make_unique<PlanewaveKohnShamEnergyCalculator>(
            kpoint_grid(), quantum_system_, smearing_scheme_.get());

        planewave_energies_mutable().ewald_energy =
            compute_ewald(crystal_structure(), planewave_basis_);

        if (!config_.disable_symmetry) {
            density_symmetrizer_.build_shells(crystal_structure(), planewave_basis_);
        }

        log_representation_summary();
    }

    void initiate_mixer() override {
        MixerEnvironment environment;
        environment.to_reciprocal_space = [this](const std::vector<Real>& density_real,
                                                 std::vector<Complex>& density_g) {
            planewave_basis_.rho_r_to_g(density_real, density_g);
        };
        environment.to_real_space = [this](const std::vector<Complex>& density_g,
                                           std::vector<Real>& density_real) {
            planewave_basis_.rho_g_to_r(density_g, density_real);
        };

        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        Real two_pi_over_lattice_parameter_squared =
            crystal_structure().two_pi_over_lattice_parameter_squared();
        /**
         * Same metric definitions as the SCF dr2 (see hartree_metric_prefactor /
         * magnetization_metric_weight) so mixer-internal norms and the SCF
         * convergence check agree.
         */
        Real charge_prefactor = hartree_metric_prefactor();
        environment.magnetization_metric_weight = magnetization_metric_weight();
        environment.first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();
        environment.is_gamma_only = planewave_basis_.is_gamma_only();
        environment.global_reduce = [this](std::vector<Real>& values) {
            if (execution_policy_ != nullptr) {
                execution_policy_->reduce_within_pool_vector(values);
            }
        };

        environment.charge_metric_weights.assign(number_of_g_vectors, 0.0);
        for (int g_index = environment.first_nonzero_g_index; g_index < number_of_g_vectors;
             ++g_index) {
            Real g_squared = planewave_basis_.g_vectors()[g_index].magnitude_squared;
            environment.charge_metric_weights[g_index] = charge_prefactor / g_squared;
        }

        /**
         * The residual preconditioner. |G|² is carried in bohr⁻² -- the stored
         * magnitude is in units of (2π/a)² -- because every preconditioner
         * compares it against a screening wavevector in those units.
         */
        PreconditionerEnvironment preconditioner_environment;
        preconditioner_environment.g_squared.assign(number_of_g_vectors, 0.0);
        for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
            preconditioner_environment.g_squared[g_index] =
                planewave_basis_.g_vectors()[g_index].magnitude_squared *
                two_pi_over_lattice_parameter_squared;
        }
        preconditioner_environment.q0_squared = mixing_q0_ * mixing_q0_;
        preconditioner_environment.first_nonzero_g_index = environment.first_nonzero_g_index;
        preconditioner_environment.to_reciprocal_space = environment.to_reciprocal_space;
        preconditioner_environment.to_real_space = environment.to_real_space;
        preconditioner_environment.global_reduce = environment.global_reduce;
        environment.preconditioner =
            PreconditionerFactory::create(mixing_preconditioning_, preconditioner_environment);

        /**
         * Compute the number of smooth G-vectors for dual-grid mixing (QE convention).
         * The smooth cutoff is 4*ecutwfc (the norm-conserving default density cutoff).
         * G-vectors above this cutoff get simple linear mixing instead of Broyden,
         * keeping noisy high-frequency augmentation artifacts out of the Broyden subspace.
         */
        Real smooth_cutoff = 4.0 * wavefunction_energy_cutoff_;
        Real dense_cutoff = effective_charge_density_energy_cutoff();
        if (dense_cutoff > smooth_cutoff + 1.0e-6) {
            /**
             * Count G-vectors within the smooth cutoff
             * G-vectors are sorted by magnitude, so we can stop when we exceed the cutoff
             */
            Real smooth_g_squared_max = smooth_cutoff / two_pi_over_lattice_parameter_squared;
            int smooth_count = 0;
            for (int g_index = 0; g_index < number_of_g_vectors; ++g_index) {
                if (planewave_basis_.g_vectors()[g_index].magnitude_squared <=
                    smooth_g_squared_max) {
                    smooth_count = g_index + 1;
                } else {
                    break; // G-vectors are sorted by magnitude
                }
            }
            environment.number_of_smooth_g_vectors = smooth_count;
            number_of_smooth_g_vectors_ = smooth_count;
            /**
             * Zero metric weights for hard G-vectors so the Gram matrix and projections
             * naturally use only smooth G-vectors, matching QE's density dot product,
             * which is truncated at the smooth G-vector count.
             */
            for (int g_index = smooth_count; g_index < number_of_g_vectors; ++g_index) {
                environment.charge_metric_weights[g_index] = 0.0;
            }
            log::info("     Dual-grid mixing: {} smooth / {} total G-vectors "
                      "(smooth cutoff = {:.0f} Ry, dense cutoff = {:.0f} Ry)",
                      smooth_count,
                      number_of_g_vectors,
                      smooth_cutoff,
                      dense_cutoff);
        } else {
            // Single grid: all G-vectors are smooth
            environment.number_of_smooth_g_vectors = number_of_g_vectors;
            number_of_smooth_g_vectors_ = number_of_g_vectors;
        }

        mixer_->initialize(environment);
    }

    std::vector<std::vector<Real>> input_density_for_mixing() const override {
        return charge_density_real_;
    }

    void update_effective_potential() override {
        /**
         * Skip recomputation when V_eff was injected via load_effective_potential().
         * This path is taken by the NSCF flavor which loads V_eff from a v2 checkpoint.
         */
        if (has_preloaded_effective_potential_) {
            /**
             * Compute total_potential_g_ by FFT-ing the spin-averaged potential.
             * This is required by prepare_hamiltonian() (critical for ultrasoft, harmless for NC).
             */
            int number_of_grid_points = real_space_potential_.empty()
                                            ? 0
                                            : static_cast<int>(real_space_potential_[0].size());
            if (number_of_grid_points > 0) {
                std::vector<Real> spin_averaged_potential(number_of_grid_points, 0.0);
                int number_of_spin_comp = static_cast<int>(real_space_potential_.size());
                for (const auto& spin_potential : real_space_potential_) {
                    for (int grid_index = 0; grid_index < number_of_grid_points; ++grid_index) {
                        spin_averaged_potential[grid_index] +=
                            spin_potential[grid_index] / number_of_spin_comp;
                    }
                }
                planewave_basis_.rho_r_to_g(spin_averaged_potential, total_potential_g_);
                /**
                 * Per-spin copies as well: a checkpoint stores only the average,
                 * but `real_space_potential_` was loaded per channel, so the
                 * nonlocal D can still be built per channel rather than from
                 * the average it would otherwise fall back to.
                 */
                total_potential_g_by_spin_.assign(number_of_spin_comp, {});
                for (int spin_index = 0; spin_index < number_of_spin_comp; ++spin_index)
                    planewave_basis_.rho_r_to_g(real_space_potential_[spin_index],
                                                total_potential_g_by_spin_[spin_index]);
            }
            /**
             * Initialize the cached ionic local potential so that
             * compute_double_counting_correction() does not crash when it is called after
             * solve_single_particle_states(). In the SCF path this is populated lazily inside
             * compute_effective_potential_from_density(), but that function is skipped for NSCF,
             * leaving the cache empty.
             */
            if (cached_local_ionic_potential_real_.empty()) {
                const auto& local_ionic_potential = local_ionic_potential_g();
                if (!local_ionic_potential.empty()) {
                    planewave_basis_.rho_g_to_r(local_ionic_potential,
                                                cached_local_ionic_potential_real_);
                }
            }
            return;
        }
        compute_effective_potential_from_density();
    }

    void set_diagonalization_tolerance(int iteration) override {
        if (nscf_override_maximum_iterations_.has_value()) {
            diagonalizer_->set_maximum_iterations(nscf_override_maximum_iterations_.value());
            nscf_override_maximum_iterations_.reset();
            report_unconverged_bands_ = true;
        }
        if (nscf_override_tolerance_.has_value()) {
            /**
             * NSCF mode: use the tight override tolerance set by the NSCF flavor
             * instead of the adaptive SCF scheme. The override was requested before
             * run() (when diagonalizer_ was still null), so it is applied here on
             * the first call, once the diagonalizer has been constructed.
             */
            diagonalizer_tolerance_ = nscf_override_tolerance_.value();
            nscf_override_tolerance_.reset();
        } else if (iteration <= 1) {
            // Iteration 1: use initial loose threshold (matching QE's SCF setup)
            diagonalizer_tolerance_ = math::tolerances::initial_eigenvalue_tolerance;
        } else {
            /**
             * Iteration 2+: tighten the eigenvalue threshold to stay an order of
             * magnitude below the self-consistency error, per electron-state:
             *   tolerance <- min(tolerance, 0.1 * residual / max(1, electrons))
             * The min keeps the schedule monotone, so a transient rise in the
             * residual cannot loosen a threshold that has already tightened; the
             * max(1, ...) guard covers a cell with fewer than one electron.
             * The 0.1 margin and the 1e-13 floor applied below are heuristics --
             * the same pair Quantum ESPRESSO uses -- and neither is observable in
             * the converged result, which is fixed by convergence_threshold on
             * the residual, not by this tolerance. Iteration 1's loose 1e-2 start
             * is of the same kind: the atomic-superposition potential is far from
             * self-consistent, so precision there is wasted.
             */
            diagonalizer_tolerance_ =
                std::min(diagonalizer_tolerance_,
                         0.1 * density_residual_ / std::max(1.0, number_of_electrons_));
        }
        diagonalizer_->set_tolerance(diagonalization_threshold());
    }

    void set_diagonalizer_tolerance(Real tolerance) override {
        diagonalizer_tolerance_ = tolerance;
        diagonalizer_->set_tolerance(tolerance);
    }

    void update_occupations() override {
        fermi_energy_calculator_.calculate_occupations();
    }

    void construct_density_from_states() override {
        charge_density_accumulator_->accumulate(number_of_bands_,
                                                number_of_spin_components(),
                                                1.0 / crystal_structure().cell_volume(),
                                                charge_density_real_);
        if (execution_policy_ != nullptr) {
            execution_policy_->reduce_charge_density(charge_density_real_);
        }
    }

    void symmetrize_density() override {
        density_symmetrizer_.symmetrize(
            planewave_basis_, charge_density_real_, number_of_spin_components());
    }

    Real compute_density_residual(
        const std::vector<std::vector<Real>>& input_density_for_mixing) const override {
        return compute_residual_between(input_density_for_mixing, charge_density_real_);
    }

  protected:
    /**
     * Hartree-metric prefactor e² · 4π / tpiba² · V/2. The per-G charge-channel
     * metric weight is this divided by the dimensionless (miller-basis)
     * |G|². Single source for both the mixer environment and the SCF dr2.
     */
    Real hartree_metric_prefactor() const {
        return constants::electron_charge_squared_rydberg * constants::four_pi /
               crystal_structure().two_pi_over_lattice_parameter_squared() *
               crystal_structure().cell_volume() * 0.5;
    }

    /**
     * Flat metric weight for the magnetization channel: e² · 4π / (2π)² · V/2
     * (the constant prefactor of the spin-polarized term in QE's density dot
     * product). Single source for both the mixer environment and the SCF dr2.
     */
    Real magnetization_metric_weight() const {
        return constants::electron_charge_squared_rydberg * constants::four_pi /
               (constants::two_pi * constants::two_pi) * crystal_structure().cell_volume() * 0.5;
    }

    /**
     * Compute the Hartree-metric density residual between two charge densities,
     * synchronized across k-point pools.
     *
     * The raw residual (see compute_local_residual_between below) is computed
     * independently on every rank from that rank's own rho_r_to_g transforms.
     * FFTW plans are timing-selected per rank (FFTW_MEASURE), so bit-identical
     * input densities yield bit-DIFFERENT residuals on different pool ranks --
     * measured on a 4-pool Ni LSDA+smearing run: last-bit (~1e-15 relative)
     * divergence already at iteration 1, amplified through the nonlinear
     * mixing-history feedback to factors of 2-30x near convergence (the
     * residual tends to zero while the per-rank noise it is measured against
     * does not). This residual gates three branches that
     * change the inter-pool collective call sequence: the SCF loop exit
     * (is_iteration_converged), the skip-mixing early return, and the
     * first-iteration re-diagonalization. If ranks decide those branches
     * differently, their Fermi-bisection / charge-density-Allreduce /
     * force-stress collectives pair up out of order across pools:
     * MPI_ERR_TRUNCATE when the mismatched calls have different message
     * sizes, silently corrupted totals when they happen to match. Reducing
     * with MPI_MAX makes every rank branch on the same value and is
     * conservative -- the SCF stops only when the slowest pool's estimate has
     * converged. (Within a pool no synchronization is needed: band ranks
     * compute the residual from pool-collectively-transformed, replicated
     * G-space data, so they already agree bitwise.)
     */
    Real compute_residual_between(const std::vector<std::vector<Real>>& density_in,
                                  const std::vector<std::vector<Real>>& density_out) const {
        Real density_residual = compute_local_residual_between(density_in, density_out);
        if (execution_policy_ != nullptr && execution_policy_->number_of_pools() > 1) {
            execution_policy_->reduce_scalar_max(density_residual);
        }
        return density_residual;
    }

    /**
     * This rank's own (unsynchronized) Hartree-metric residual between two
     * charge densities.
     *
     * Uses only smooth G-vectors (up to number_of_smooth_g_vectors_), matching QE, which
     * measures dr2 as the metric dot product of (ρ_out − ρ_in) over the smooth G-vector
     * count only. Subclasses (e.g. ultrasoft) may call this with smooth-only
     * densities to avoid augmentation contamination.
     * Never branch on this value directly under k-point-pool MPI -- go through
     * compute_residual_between so all pools agree (see its docstring).
     */
    Real compute_local_residual_between(const std::vector<std::vector<Real>>& density_in,
                                        const std::vector<std::vector<Real>>& density_out) const {
        int residual_g_count = (number_of_smooth_g_vectors_ > 0)
                                   ? number_of_smooth_g_vectors_
                                   : planewave_basis_.number_of_g_vectors();
        int first_nonzero_g_index = planewave_basis_.first_nonzero_g_index();

        Real density_residual = 0.0;
        Real gamma_factor = planewave_basis_.is_gamma_only() ? 2.0 : 1.0;

        if (number_of_spin_components() == 1) {
            std::vector<Complex> input_density_g;
            std::vector<Complex> output_density_g;
            planewave_basis_.rho_r_to_g(density_in[0], input_density_g);
            planewave_basis_.rho_r_to_g(density_out[0], output_density_g);
            for (int g_vector_index = first_nonzero_g_index; g_vector_index < residual_g_count;
                 ++g_vector_index) {
                Complex delta = output_density_g[g_vector_index] - input_density_g[g_vector_index];
                Real g_squared = planewave_basis_.g_vectors()[g_vector_index].magnitude_squared;
                density_residual += std::norm(delta) / g_squared * gamma_factor;
            }
            return density_residual * hartree_metric_prefactor();
        }

        /**
         * LSDA: measure the total density Δ(ρ↑+ρ↓) with the Hartree metric and
         * the magnetization Δ(ρ↑−ρ↓) with a constant metric that INCLUDES G=0
         * (QE's density-dot-product convention). A per-channel Hartree metric
         * would miss a uniform electron transfer between the spin channels — it
         * lives purely in the magnetization's G=0 component — and report
         * convergence while the moment is still relaxing.
         */
        std::vector<Complex> input_up_g, input_down_g, output_up_g, output_down_g;
        planewave_basis_.rho_r_to_g(density_in[0], input_up_g);
        planewave_basis_.rho_r_to_g(density_in[1], input_down_g);
        planewave_basis_.rho_r_to_g(density_out[0], output_up_g);
        planewave_basis_.rho_r_to_g(density_out[1], output_down_g);

        Real magnetization_residual = 0.0;
        for (int g_vector_index = 0; g_vector_index < residual_g_count; ++g_vector_index) {
            Complex delta_up = output_up_g[g_vector_index] - input_up_g[g_vector_index];
            Complex delta_down = output_down_g[g_vector_index] - input_down_g[g_vector_index];
            Complex delta_total = delta_up + delta_down;
            Complex delta_magnetization = delta_up - delta_down;
            // G=0 is stored once even on gamma-only half-sphere grids.
            Real g_factor = (g_vector_index < first_nonzero_g_index) ? 1.0 : gamma_factor;
            if (g_vector_index >= first_nonzero_g_index) {
                Real g_squared = planewave_basis_.g_vectors()[g_vector_index].magnitude_squared;
                density_residual += std::norm(delta_total) / g_squared * g_factor;
            }
            magnetization_residual += std::norm(delta_magnetization) * g_factor;
        }

        return density_residual * hartree_metric_prefactor() +
               magnetization_residual * magnetization_metric_weight();
    }

    void mix_density(const std::vector<std::vector<Real>>& input_density_for_mixing) override {
        mixer_->mix(input_density_for_mixing, charge_density_real_, number_of_spin_components());
    }

    void snapshot_output_density() override {
        convergence_output_density_ = charge_density_real_;
    }

    void restore_density_at_convergence() override {
        charge_density_real_ = convergence_output_density_;
    }

    Real compute_double_counting_correction() const override {
        /**
         * Each rank integrates only its local real-space slabs (FFTW-MPI slab
         * decomposition). With 1-rank-per-pool the integral is complete; with
         * multiple ranks per pool it must be summed across the band communicator.
         */
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        int total_grid_points = planewave_basis_.fft_rho().total_grid_points();
        Real volume_per_grid_point = crystal_structure().cell_volume() / total_grid_points;
        Real double_counting_correction = 0.0;

        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                double_counting_correction -= charge_density_real_[spin_index][grid_index] *
                                              (real_space_potential_[spin_index][grid_index] -
                                               cached_local_ionic_potential_real_[grid_index]);
            }
        }
        double_counting_correction *= volume_per_grid_point;
        if (execution_policy_ != nullptr) {
            execution_policy_->reduce_within_pool_scalar(double_counting_correction);
        }
        return double_counting_correction;
    }

    Real compute_delta_escf_correction() const override {
        /**
         * Same slab decomposition as compute_double_counting_correction: partial
         * real-space integral over local_grid_points, requires band-communicator
         * reduction when multiple ranks share the pool.
         */
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        int total_grid_points = planewave_basis_.fft_rho().total_grid_points();
        Real volume_per_grid_point = crystal_structure().cell_volume() / total_grid_points;
        Real delta_escf = 0.0;

        /**
         * After mixing, charge_density_real_ = rho_mixed
         * convergence_output_density_ = rho_out (saved by snapshot_output_density())
         * real_space_potential_ = V_Hxc from rho_mixed (set by update_effective_potential())
         */
        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                Real density_difference = charge_density_real_[spin_index][grid_index] -
                                          convergence_output_density_[spin_index][grid_index];
                delta_escf -= density_difference * (real_space_potential_[spin_index][grid_index] -
                                                    cached_local_ionic_potential_real_[grid_index]);
            }
        }
        delta_escf *= volume_per_grid_point;
        if (execution_policy_ != nullptr) {
            execution_policy_->reduce_within_pool_scalar(delta_escf);
        }
        return delta_escf;
    }

    void compute_energy_components() override {
        energies_mutable().band_energy =
            energy_calculator_->compute_band_energy(number_of_bands_, number_of_spin_components());
        planewave_energies_mutable().smearing_entropy_contribution =
            energy_calculator_->compute_smearing_entropy(number_of_bands_,
                                                         number_of_spin_components());
        if (execution_policy_ != nullptr) {
            execution_policy_->reduce_scalar(energies_mutable().band_energy);
            execution_policy_->reduce_scalar(
                planewave_energies_mutable().smearing_entropy_contribution);
        }
    }

    void report_first_iteration_details() override {
        PlanewaveKohnShamSCFReporter::report_first_iteration(energies().total(),
                                                             planewave_energies());
    }

    void report_convergence_details() override {
        PlanewaveKohnShamSCFConvergenceSummary summary;
        summary.number_of_spin_components = number_of_spin_components();
        summary.number_of_bands = number_of_bands_;
        if (execution_policy_ != nullptr) {
            summary.number_of_kpoint_pools = execution_policy_->number_of_pools();
        }
        summary.energies = &planewave_energies();
        if (summary.number_of_spin_components == 2) {
            Real total_magnetization_sum = 0.0;
            Real absolute_magnetization_sum = 0.0;
            int local_grid_points = static_cast<int>(charge_density_real_[0].size());
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                Real spin_density =
                    charge_density_real_[0][grid_index] - charge_density_real_[1][grid_index];
                total_magnetization_sum += spin_density;
                absolute_magnetization_sum += std::abs(spin_density);
            }
            if (execution_policy_ != nullptr) {
                execution_policy_->reduce_within_pool_scalar(total_magnetization_sum);
                execution_policy_->reduce_within_pool_scalar(absolute_magnetization_sum);
            }
            Real volume_element =
                crystal_structure().cell_volume() / planewave_basis_.fft_rho().total_grid_points();
            summary.total_magnetization = total_magnetization_sum * volume_element;
            summary.absolute_magnetization = absolute_magnetization_sum * volume_element;
            /**
             * Kept here, where they are integrated for the report, rather than
             * re-integrated for the results file: the density this reads is
             * released with the rest of the SCF workspace before the flavor
             * assembles its result, and a second integration somewhere else
             * would be a second chance for the two to disagree -- the same
             * reason the SCF history is recorded at the line that prints it.
             */
            magnetization_ =
                Magnetization{summary.total_magnetization, summary.absolute_magnetization};
        }
        /**
         * Report every irreducible k-point, not just this rank's pool share.
         *
         * kpoint_grid() is the DISTRIBUTED grid, so with pools it holds only
         * this pool's k-points -- a 60-k-point calculation on 8 pools used to
         * report 8, restart its labels at k_0 in each pool, and give no way to
         * tell which of the 60 they were. The same fixed-width pack-and-gather
         * the SCF flavor already uses to build results.json (see
         * flavors/self_consistent_field.hpp) puts them back in global order,
         * and it costs one Allreduce at the end of a converged run.
         *
         * Layout per k-point state: number_of_bands_ eigenvalues followed by
         * the three k-point coordinates. Weights are not needed for the report.
         * assemble_global_kpoint_state_data() is spin-major over the global
         * grid and returns the whole array on every rank; the serial policy
         * hands the input straight back, so this path is not MPI-only.
         */
        const int local_number_of_kpoints = kpoint_grid().number_of_kpoints();
        const int number_of_values_per_state = number_of_bands_ + 3;
        const int local_number_of_kpoint_states =
            local_number_of_kpoints * summary.number_of_spin_components;
        std::vector<Real> local_state_data(static_cast<std::size_t>(local_number_of_kpoint_states) *
                                               number_of_values_per_state,
                                           0.0);
        for (int kpoint_index_state = 0; kpoint_index_state < local_number_of_kpoint_states;
             ++kpoint_index_state) {
            const int local_physical_kpoint_index =
                (local_number_of_kpoints > 0) ? (kpoint_index_state % local_number_of_kpoints) : 0;
            const std::size_t state_offset =
                static_cast<std::size_t>(kpoint_index_state) * number_of_values_per_state;
            for (int band_index = 0; band_index < number_of_bands_; ++band_index) {
                local_state_data[state_offset + static_cast<std::size_t>(band_index)] =
                    quantum_system_.eigenvalue(kpoint_index_state, band_index);
            }
            const auto& kpoint = kpoint_grid()[local_physical_kpoint_index];
            local_state_data[state_offset + static_cast<std::size_t>(number_of_bands_)] =
                kpoint.coordinates.x;
            local_state_data[state_offset + static_cast<std::size_t>(number_of_bands_) + 1] =
                kpoint.coordinates.y;
            local_state_data[state_offset + static_cast<std::size_t>(number_of_bands_) + 2] =
                kpoint.coordinates.z;
        }

        std::vector<Real> global_state_data = local_state_data;
        if (execution_policy_ != nullptr) {
            global_state_data = execution_policy_->assemble_global_kpoint_state_data(
                local_state_data,
                local_number_of_kpoints,
                summary.number_of_spin_components,
                number_of_values_per_state);
        }

        const int global_number_of_kpoint_states =
            static_cast<int>(global_state_data.size()) / number_of_values_per_state;
        summary.number_of_kpoints =
            (summary.number_of_spin_components > 0)
                ? global_number_of_kpoint_states / summary.number_of_spin_components
                : global_number_of_kpoint_states;
        summary.eigenvalues.resize(static_cast<std::size_t>(global_number_of_kpoint_states));
        summary.kpoints.assign(static_cast<std::size_t>(summary.number_of_kpoints), KPoint{});
        for (int kpoint_index_state = 0; kpoint_index_state < global_number_of_kpoint_states;
             ++kpoint_index_state) {
            const std::size_t state_offset =
                static_cast<std::size_t>(kpoint_index_state) * number_of_values_per_state;
            summary.eigenvalues[static_cast<std::size_t>(kpoint_index_state)].assign(
                global_state_data.begin() + static_cast<std::ptrdiff_t>(state_offset),
                global_state_data.begin() +
                    static_cast<std::ptrdiff_t>(state_offset + number_of_bands_));
            const int global_physical_kpoint_index =
                (summary.number_of_kpoints > 0) ? (kpoint_index_state % summary.number_of_kpoints)
                                                : 0;
            summary.kpoints[static_cast<std::size_t>(global_physical_kpoint_index)].coordinates = {
                global_state_data[state_offset + static_cast<std::size_t>(number_of_bands_)],
                global_state_data[state_offset + static_cast<std::size_t>(number_of_bands_) + 1],
                global_state_data[state_offset + static_cast<std::size_t>(number_of_bands_) + 2]};
        }
        PlanewaveKohnShamSCFReporter::report_convergence_summary(summary);

        /**
         * The counters as the SCF leaves them, which is not what the entry
         * point prints at exit: forces and stress add transforms of their own
         * afterwards, so the two figures differ, and that difference is the
         * reason to have this one at all.
         *
         * The Davidson timing table is deliberately not dumped here. Nothing
         * between this line and the entry point's closing dump touches the
         * accumulator, so the two were byte-identical -- the same table twice
         * in one run.
         */
        log::debug("     SCF profiling summary:");
        log::debug("       FFT calls total:      {}",
                   ProfilingCounters::instance().fft_call_count.load());
        log::debug("       Davidson iterations:  {}",
                   ProfilingCounters::instance().davidson_iteration_count.load());
        log::debug("       H|psi> applications:  {}",
                   ProfilingCounters::instance().hamiltonian_application_count.load());
    }

    // ─── Abstract hooks for pseudopotential subclasses ───────────────

    /**
     * Compute the effective potential from the current charge density.
     * Must set charge_density_g_, real_space_potential_, total_potential_g_,
     * exchange_correlation_potential_g_, and energy components (hartree, xc).
     */
    virtual void compute_effective_potential_from_density() = 0;

    /**
     * Return the local ionic potential in reciprocal space.
     */
    [[nodiscard]] virtual const std::vector<Complex>& local_ionic_potential_g() const = 0;

    /**
     * Return the core charge density in reciprocal space.
     */
    [[nodiscard]] virtual const std::vector<Complex>& core_charge_density_g() const = 0;

    /**
     * Hook invoked once the initial atomic-superposition charge density is ready.
     *
     * Default no-op.  The ultrasoft subclass overrides it to initialise the smooth
     * density and projector overlap storage.
     */
    virtual void on_charge_density_initialized() {}

    // ─── Representation setup helpers ────────────────────────────────
    Real effective_charge_density_energy_cutoff() const {
        return charge_density_energy_cutoff_ > 0 ? charge_density_energy_cutoff_
                                                 : 4.0 * wavefunction_energy_cutoff_;
    }

    Real compute_number_of_electrons() const {
        Real count = 0.0;
        for (int atom_index = 0; atom_index < crystal_structure().number_of_atoms(); ++atom_index) {
            count += crystal_structure().valence_charge(
                crystal_structure().species_type_index(atom_index));
        }
        /**
         * total_charge follows QE's tot_charge convention: positive = electrons
         * removed (cationic). The Ewald ion-ion sum and the Hartree G=0 skip are
         * both keyed only to the ionic charge (see ewald_summation.hpp), so no
         * separate compensating-background term is needed here -- only the
         * electron count itself changes.
         */
        return count - config().total_charge;
    }

    /**
     * Bands to solve for, when the input did not say.
     *
     * Insulator: the occupied count plus four empty bands, which cost little and
     * give the eigensolver room to converge the top of the valence manifold.
     * Metal: 20% more than the occupied count, but never fewer than four extra —
     * a partly filled band must be resolved well above the Fermi level or the
     * smeared occupations are truncated and the total energy is wrong.
     *
     * The 20% and the four-band floor are heuristics, and they are the same pair
     * Quantum ESPRESSO uses, deliberately: for a metal the band count changes the
     * total energy through the fractional occupations of the extra bands, so
     * reproducing an upstream energy at the same input requires reproducing its
     * band count. That equivalence is what the three LSDA cases in
     * tests/validation/test_qe_energy_comparison.py check — they pass no band
     * count to either code. The insulating branch deliberately differs (the
     * upstream default adds no margin there); empty bands cannot move a
     * fixed-occupation total, so nothing observable turns on it.
     */
    int resolve_number_of_bands() const {
        int number_of_occupied_bands = static_cast<int>(std::round(number_of_electrons_ / 2.0));
        if (number_of_bands_ > 0) {
            if (number_of_bands_ <= number_of_occupied_bands) {
                log::warn("number_of_bands ({}) equals the number of occupied states ({}). "
                          "This may cause SCF convergence issues. Consider adding extra empty "
                          "bands (e.g., number_of_bands: {}).",
                          number_of_bands_,
                          number_of_occupied_bands,
                          number_of_occupied_bands + 4);
            }
            return number_of_bands_;
        }
        bool metallic =
            (smearing_ != "none" && smearing_width_ > math::tolerances::smearing_width_threshold);
        if (metallic) {
            return std::max(static_cast<int>(std::round(1.2 * number_of_electrons_ / 2.0)),
                            number_of_occupied_bands + 4);
        }
        return number_of_occupied_bands + 4;
    }

    /**
     * Whether band distribution was abandoned during setup because some rank's
     * FFT slab contained no plane waves. Reported in the setup summary, since it
     * silently changes the parallel behaviour of the whole run.
     */
    bool replicated_plane_wave_fallback_active_ = false;

    void build_kpoint_plane_wave_data() {
        int band_size = (execution_policy_ != nullptr) ? execution_policy_->band_size() : 1;
        planewave_basis_.initialize_kpoint_data(crystal_structure(), kpoint_grid(), band_size);

        /**
         * Safety: if band parallelism gave any rank 0 plane waves at any
         * k-point, fall back to replicated mode for ALL ranks in the pool.
         * This prevents zero-dimension BLAS crashes and asymmetric MPI reductions.
         */
        if (band_size > 1 && execution_policy_ != nullptr) {
            int local_minimum_plane_waves =
                planewave_basis_.kpoint_number_of_plane_waves().empty()
                    ? 1
                    : *std::min_element(planewave_basis_.kpoint_number_of_plane_waves().begin(),
                                        planewave_basis_.kpoint_number_of_plane_waves().end());
            int global_minimum_plane_waves = local_minimum_plane_waves;
            execution_policy_->reduce_band_group_scalar_min(global_minimum_plane_waves);
            if (global_minimum_plane_waves == 0) {
                log::warn("Band parallelism: some ranks received 0 plane waves. "
                          "Falling back to replicated G-vectors.");
                execution_policy_->disable_band_distribution();

                /**
                 * The FFT grids were built during generate(), while the group was
                 * still distributed, so they are MPI grids whose transforms are
                 * collective across the band group. Disabling band distribution
                 * turns every group reduction into a no-op and leaves each rank
                 * holding a replicated copy of the coefficients -- at which point
                 * ranks legitimately take different paths through Davidson while
                 * those collective transforms still expect every one of them. The
                 * first divergence blocks inside FFTW-MPI's transpose.
                 *
                 * Rebuild the basis now that group_size() is 1, so the factory
                 * hands back serial grids each rank drives alone. This repeats the
                 * FFT planning generate() already did, which is the price of
                 * discovering the empty slab only after the grids exist.
                 */
                planewave_basis_.generate(*execution_policy_,
                                          crystal_structure(),
                                          wavefunction_energy_cutoff_,
                                          effective_charge_density_energy_cutoff(),
                                          planewave_basis_.is_gamma_only(),
                                          !config_.use_deterministic_fft_planning,
                                          config_.use_exact_wavefunction_grid);
                planewave_basis_.initialize_kpoint_data(crystal_structure(), kpoint_grid(), 1);
                replicated_plane_wave_fallback_active_ = true;
            }
        }
    }

    void allocate_electronic_states() {
        quantum_system_.allocate_electronic_states(planewave_basis_.kpoint_number_of_plane_waves());
    }

    void log_representation_summary() const {
        const auto& number_of_plane_waves_vector = planewave_basis_.kpoint_number_of_plane_waves();
        PlanewaveKohnShamSCFSetupSummary summary;
        summary.lattice_parameter = crystal_structure().lattice_parameter();
        summary.cell_volume = crystal_structure().cell_volume();
        summary.number_of_atoms = crystal_structure().number_of_atoms();
        summary.wavefunction_energy_cutoff = wavefunction_energy_cutoff_;
        summary.charge_density_energy_cutoff = effective_charge_density_energy_cutoff();
        summary.fft_rho_grid_dimension_1 = planewave_basis_.fft_rho().grid_dimension_1();
        summary.fft_rho_grid_dimension_2 = planewave_basis_.fft_rho().grid_dimension_2();
        summary.fft_rho_grid_dimension_3 = planewave_basis_.fft_rho().grid_dimension_3();
        summary.fft_wave_grid_dimension_1 = planewave_basis_.fft_wave().grid_dimension_1();
        summary.fft_wave_grid_dimension_2 = planewave_basis_.fft_wave().grid_dimension_2();
        summary.fft_wave_grid_dimension_3 = planewave_basis_.fft_wave().grid_dimension_3();
        summary.number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        /**
         * The irreducible count, not this pool's share. kpoint_grid() is the
         * distributed grid, so with 8 pools a 60-k-point calculation reported
         * 8 here -- next to an "IBZ reduction ... -> 60 k-points" line two
         * rows above, which made one of the two look wrong.
         *
         * Derived from the same assembly the convergence report uses, with a
         * one-value-per-state payload: its return size is the global state
         * count, so this costs an Allreduce of a few dozen doubles once per
         * run. The serial policy returns the input unchanged, which gives the
         * local count -- and in serial the local count IS the global one.
         */
        summary.number_of_kpoints = kpoint_grid().number_of_kpoints();
        if (execution_policy_ != nullptr) {
            const int local_kpoint_states = summary.number_of_kpoints * number_of_spin_components();
            const std::vector<Real> kpoint_state_probe(
                static_cast<std::size_t>(local_kpoint_states), 0.0);
            const std::size_t global_kpoint_states =
                execution_policy_
                    ->assemble_global_kpoint_state_data(kpoint_state_probe,
                                                        summary.number_of_kpoints,
                                                        number_of_spin_components(),
                                                        1)
                    .size();
            if (number_of_spin_components() > 0) {
                summary.number_of_kpoints =
                    static_cast<int>(global_kpoint_states) / number_of_spin_components();
            }
        }
        summary.max_plane_waves_per_kpoint =
            number_of_plane_waves_vector.empty()
                ? 0
                : *std::max_element(number_of_plane_waves_vector.begin(),
                                    number_of_plane_waves_vector.end());
        /**
         * Each pool holds only its local k-points, so the local max may be less
         * than the global max. Reduce across pools to get the true maximum.
         */
        if (execution_policy_ != nullptr) {
            execution_policy_->reduce_scalar_max(summary.max_plane_waves_per_kpoint);
        }
        summary.number_of_spin_components = number_of_spin_components();
        summary.number_of_electrons = number_of_electrons_;
        summary.number_of_bands = number_of_bands_;
        summary.xc_functional_name = exchange_correlation_->name();
        summary.diagonalizer_name = diagonalizer_->name();
        summary.davidson_basis_expansion_factor =
            diagonalization_method_ == "davidson" ? config().davidson_basis_expansion_factor : 0;
        summary.mixer_name = mixer_->name();
        summary.mixing_beta = mixing_beta_;
        summary.ewald_energy = planewave_energies().ewald_energy;
        summary.density_symmetrization_active = density_symmetrizer_.is_initialized();
        summary.number_of_shells = density_symmetrizer_.number_of_shells();
        summary.number_of_symmetry_operations =
            density_symmetrizer_.number_of_symmetry_operations();
        if (execution_policy_ != nullptr) {
            summary.number_of_processes = execution_policy_->number_of_processes();
            summary.number_of_kpoint_pools = execution_policy_->number_of_pools();
            /**
             * Ranks per pool, not band_size(). The two agree for the
             * reciprocal-space split, but band_size() is deliberately 1 once
             * bands are distributed -- reporting it there would claim one rank
             * per pool on a run using four.
             */
            summary.band_group_size =
                summary.number_of_kpoint_pools > 0
                    ? summary.number_of_processes / summary.number_of_kpoint_pools
                    : summary.number_of_processes;
            summary.bands_are_distributed = execution_policy_->band_distribution_size() > 1;
            summary.replicated_plane_wave_fallback_active = replicated_plane_wave_fallback_active_;
        }
        PlanewaveKohnShamSCFReporter::report_setup_summary(summary);
    }

    void warn_if_total_charge_deviates() const {
        /**
         * Each rank holds the real-space planes of the density it owns: all of
         * them unless reciprocal space is split within the pool, in which case
         * the local sum is a fraction of the charge -- about half on two ranks,
         * which is what this used to report as a deviation. The sum is completed
         * within the group before it is compared, so every rank must call this.
         */
        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        int total_grid_points = planewave_basis_.fft_rho().total_grid_points();
        Real total_charge = 0.0;
        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
                total_charge += charge_density_real_[spin_index][grid_index];
            }
        }
        execution_policy_->reduce_within_group(&total_charge, 1);
        total_charge *= crystal_structure().cell_volume() / total_grid_points;
        if (execution_policy_->is_root() && std::abs(total_charge - number_of_electrons_) > 0.01) {
            log::warn("     Total charge {:.4f} deviates from target {} by >{:.1f}e",
                      total_charge,
                      number_of_electrons_,
                      std::abs(total_charge - number_of_electrons_));
        }
    }
};

} // namespace q3
