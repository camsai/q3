#pragma once

/**
 * PlanewavePseudopotentialKohnShamSCFIterator — concrete pseudopotential SCF (layers 3–4).
 *
 * Extends the abstract PlanewaveKohnShamSCFIterator with the pseudopotential
 * ionic world: the local ionic potential, partial-core charge, KB projectors,
 * and the initial charge density / wavefunction setup from atomic superpositions.
 *
 * This class implements the norm-conserving pseudopotential case.  Ultrasoft-only
 * machinery (smooth pre-augmentation density, projector overlap coefficients,
 * augmentation charges, the effective D-matrix) lives in
 * UltrasoftPlanewavePseudopotentialKohnShamSCFIterator, leaving room for a
 * future PAW sibling.
 */

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "io/charge_density/kohn_sham/planewave/checkpoint/factory.hpp"
#include "io/logger.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"
#include "io/wavefunction/kohn_sham/planewave/checkpoint/factory.hpp"

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/lowdin_orthogonalization.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/wavefunction_atomic.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/factory.hpp"
#include "layer_3_representation/wavefunction/planewave/norm_conserving/initializer.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/planewave_potential_utils.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/factory.hpp"
#include "layer_4_computation/effective_potential/internal/dispersion/factory.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/occupation_mixer.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/seeding.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/symmetrization.hpp"
#include "layer_4_computation/effective_potential/planewave/pseudopotential/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/state/internal/energies/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

class PlanewavePseudopotentialKohnShamSCFIterator : public PlanewaveKohnShamSCFIterator {
  public:
    explicit PlanewavePseudopotentialKohnShamSCFIterator(KohnShamSCFIteratorConfig config)
        : PlanewaveKohnShamSCFIterator(std::move(config)) {}
    ~PlanewavePseudopotentialKohnShamSCFIterator() override = default;

    /// Transfer ownership of a PseudopotentialSet into this iterator.
    void set_owned_pseudopotential_set(PseudopotentialSet pseudopotential_set) {
        owned_pseudopotential_set_ = std::move(pseudopotential_set);
        pseudopotential_set_ = &owned_pseudopotential_set_.value();
    }

    [[nodiscard]] const NormConservingPlanewaveKohnShamEnergies& energies() const override {
        return energies_;
    }
    NormConservingPlanewaveKohnShamEnergies& energies_mutable() override {
        return energies_;
    }

    /**
     * Assemble and return a snapshot of the converged SCF state.
     *
     * The ultrasoft subclass overrides this to inject the effective D-matrix into the snapshot.
     */
    NormConservingPlanewaveKohnShamSCFState result() const override {
        NormConservingPlanewaveKohnShamSCFState data;
        data.scf_iterations = iteration();
        data.converged = is_converged();
        data.energies_data_ = energies_;
        /**
         * Share the existing quantum system snapshot instead of creating a new one.
         * The PP basis (beta tables, local potential, core charge) is expensive to
         * initialize, and we only need a read-only view for forces/stress/IO.
         */
        data.quantum_system_snapshot_ = quantum_system_snapshot_;
        data.quantum_system_snapshot_->copy_quantum_states_from(quantum_system_);
        data.charge_density_g = charge_density_g_;
        data.charge_density_real = charge_density_real_;
        data.total_potential_g = total_potential_g_;
        data.total_potential_g_by_spin = total_potential_g_by_spin_;
        data.exchange_correlation_potential_g = exchange_correlation_potential_g();
        data.real_space_potential = real_space_potential_;
        data.planewave_basis = &planewave_basis_;
        data.exchange_correlation = &exchange_correlation();
        data.crystal_structure = &crystal_structure();
        data.kpoint_grid = &kpoint_grid();
        data.execution_policy = execution_policy_;
        data.hubbard_manifolds = hubbard_manifolds_;
        data.hubbard_d_matrix_per_spin = hubbard_d_matrix_per_spin_;
        data.hubbard_projector_type = config().hubbard_projector_type;
        return data;
    }

    /**
     * Write binary checkpoint files (charge density, wavefunctions, and effective potential).
     *
     * Writes:
     *   - charge-density.dat — QE-compatible Fortran binary charge density
     *   - wfc*.dat           — QE-compatible Fortran binary wavefunctions per k-point
     *   - effective-potential.q3cp — Q3-native v3 binary with V_eff in real space and,
     *                                for PAW, the on-site occupations ρ_ij
     *
     * The effective-potential file enables NSCF calculations to load V_eff directly
     * instead of reconstructing it from charge density + input parameters, and to
     * build a PAW dataset's one-centre term of D from the converged ρ_ij.
     *
     * Under k-point pools of one rank each, every rank writes the wfc*.dat files
     * of its own k-points, and the root writes the rest. Those it can write
     * alone: the density, the potential built from it and the PAW ρ_ij are
     * reduced across pools, so every rank already holds them whole. Only the
     * eigenvalues are per pool, and are gathered first. The result is the same
     * directory a serial run writes, which a serial NSCF restores.
     *
     * Collective: every rank must call this.
     */
    void write_checkpoints(const std::string& output_directory,
                           const std::string& prefix) const override {
        if (!execution_policy_->each_process_owns_whole_kpoints()) {
            if (execution_policy_->is_root()) {
                log::warn("Checkpoint output is not supported when ranks share a k-point pool: "
                          "they split each k-point's plane-wave coefficients and the density "
                          "grid between them, and the checkpoint writers have no gather for "
                          "either. No checkpoint directory was created; results output is "
                          "unaffected. Give every rank its own pool, as the automatic MPI "
                          "topology does, to write one.");
            }
            return;
        }
        auto scf_state = result();
        const Matrix<Real> eigenvalues_of_every_kpoint = global_eigenvalues();
        std::string save_directory = output_directory + "/" + prefix + ".save";
        /**
         * Every rank creates the directory it writes into rather than waiting
         * on the root. Ranks may race to create it, so success is judged by the
         * directory existing afterwards, not by which call made it.
         */
        std::error_code directory_error;
        std::filesystem::create_directories(save_directory, directory_error);
        if (!std::filesystem::is_directory(save_directory)) {
            throw std::runtime_error("Cannot create the checkpoint directory '" + save_directory +
                                     "': " + directory_error.message());
        }

        auto wavefunctions_checkpoint = create_wavefunction_checkpoint(scf_state);
        wavefunctions_checkpoint->write_to_file(save_directory);

        if (!execution_policy_->is_root()) {
            return;
        }
        if (execution_policy_->number_of_pools() > 1) {
            log::info("     Each of the {} k-point pools wrote the wavefunctions of its own "
                      "k-points, {} in all",
                      execution_policy_->number_of_pools(),
                      eigenvalues_of_every_kpoint.rows() / number_of_spin_components());
        }

        auto charge_density_checkpoint = create_charge_density_checkpoint(scf_state);
        charge_density_checkpoint->write_to_file(save_directory + "/charge-density.dat");

        /**
         * Write the effective potential (V_Hxc + V_loc) to the Q3-native
         * checkpoint, along with how the run that produced it ended.
         *
         * The header fields are not decoration. A checkpoint travels away from
         * the run that wrote it -- to an NSCF job, to a restart, to a third
         * party verifying a published calculation -- and a reader has no other
         * way to learn whether the density it holds converged, or what energy
         * it converged to. Leaving them zero makes the file assert that it came
         * from an unconverged run at zero energy, which is worse than saying
         * nothing.
         */
        KohnShamQuantumSystemSnapshot effective_potential_snapshot;
        effective_potential_snapshot.iteration = iteration();
        effective_potential_snapshot.number_of_spin_components = number_of_spin_components();
        effective_potential_snapshot.number_of_kpoint_states =
            static_cast<int>(eigenvalues_of_every_kpoint.rows());
        effective_potential_snapshot.number_of_bands =
            static_cast<int>(eigenvalues_of_every_kpoint.cols());
        effective_potential_snapshot.total_energy = total_energy();
        effective_potential_snapshot.converged = is_converged();
        effective_potential_snapshot.eigenvalues = eigenvalues_of_every_kpoint;
        effective_potential_snapshot.charge_density_real = charge_density_real_;
        effective_potential_snapshot.effective_potential_real = real_space_potential_;
        effective_potential_snapshot.one_center_occupations = one_center_occupations();

        Q3PlanewaveKohnShamQuantumSystemCheckpoint effective_potential_checkpoint;
        effective_potential_checkpoint.save(save_directory + "/effective-potential.q3cp",
                                            effective_potential_snapshot,
                                            /*save_wavefunctions=*/false);
    }

    // ─── Pseudopotential accessors ───────────────────────────────────

    /**
     * Tr[n] per Hubbard manifold, per spin, from the converged occupation
     * matrices -- the diagnostic the DFT+U line prints, in full precision and
     * indexed the same way as the manifolds the SCF state carries.
     *
     * Empty when the run has no Hubbard manifolds.
     */
    [[nodiscard]] const std::vector<std::vector<Real>>& hubbard_occupation_traces() const {
        return hubbard_occupation_traces_;
    }

    /**
     * Access the pseudopotential set by const reference.
     */
    [[nodiscard]] const PseudopotentialSet& pseudopotential_set() const {
        return *pseudopotential_set_;
    }

    /**
     * Access the pseudopotential quantum-system snapshot built during initialization.
     */
    [[nodiscard]] const NormConservingPlanewaveKohnShamQuantumSystem&
    norm_conserving_quantum_system() const {
        return *quantum_system_snapshot_;
    }

    /**
     * Give the snapshot the electronic state the iterator currently holds.
     *
     * The snapshot is the only object carrying the pseudopotential data --
     * beta tables, augmentation, the ultrasoft basis -- because the factory
     * built it as the right subclass. `quantum_system_` is a by-value member
     * of the base type and carries the wavefunctions and eigenvalues. Anything
     * needing both at once (a Wannier capture on an ultrasoft or PAW dataset)
     * has to bring them together first, which `result()` also does when it
     * assembles the state to hand back.
     *
     * Copies electronic state only; the pseudopotential data on the snapshot
     * is untouched.
     */
    void refresh_quantum_system_snapshot() {
        quantum_system_snapshot_->copy_quantum_states_from(quantum_system_);
    }

    /**
     * Overlap coefficients ⟨β_i|ψ_nk⟩ at convergence.
     *
     * Returns an empty matrix for norm-conserving calculations.
     * The ultrasoft iterator overrides this to return the actual becsum.
     */
    [[nodiscard]] virtual const Matrix<Real>& projector_overlap_coefficients() const {
        static const Matrix<Real> empty_matrix{};
        return empty_matrix;
    }

    /**
     * The on-site occupations ρ_ij per spin channel, which the checkpoint
     * stores: what D's one-centre term is built from. Empty wherever D has no
     * such term; the PAW iterator overrides this.
     */
    [[nodiscard]] virtual std::vector<Matrix<Real>> one_center_occupations() const {
        return {};
    }

    /**
     * Whether a non-self-consistent pass must be handed those occupations
     * before it diagonalizes -- true exactly where D carries a one-centre term.
     */
    [[nodiscard]] virtual bool requires_one_center_occupations() const {
        return false;
    }

    /**
     * Hand over ρ_ij from a checkpoint, so D's one-centre term is built from the
     * converged occupations before the first Hamiltonian and stays so for the
     * whole run. Call after construction and before run(), like
     * load_effective_potential(). Nothing to do where D has no such term.
     */
    virtual void load_one_center_occupations(
        [[maybe_unused]] const std::vector<Matrix<Real>>& occupations_by_spin) {}

    /**
     * Set the diagonalizer tolerance directly, bypassing the adaptive SCF scheme.
     *
     * Intended for use by the NSCF flavor, which runs only a single diagonalization
     * pass. The SCF adaptive scheme always starts at the loose initial threshold
     * (1e-2 Ry) on iteration 1, which is far too loose for a fixed-potential NSCF
     * calculation and causes Davidson to converge to wrong conduction-band eigenvalues
     * at special high-symmetry k-points.
     *
     * This method may be called before run() when the diagonalizer_ is not yet
     * constructed. The requested tolerance is stored as a pending override and
     * applied on the first call to set_diagonalization_tolerance() inside run().
     */
    void set_diagonalizer_tolerance_for_nscf(Real tolerance) {
        nscf_override_tolerance_ = tolerance;
    }

    /**
     * Allow a fixed-potential pass enough Davidson iterations to converge every band.
     *
     * Pending, like the tolerance above, until the diagonalizer exists. Also
     * turns on the report of any k-point whose solve still ends with bands
     * unconverged, since a single pass has no later iteration to finish them.
     */
    void set_diagonalizer_maximum_iterations_for_nscf(int maximum_iterations) {
        nscf_override_maximum_iterations_ = maximum_iterations;
    }

    /**
     * Vxc(G) of a given density -- the exchange-correlation potential (core
     * charge included) that update_effective_potential() would leave in
     * exchange_correlation_potential_g() had the SCF stopped at this density.
     *
     * For a fixed-potential pass, which loads V_eff from the checkpoint and so
     * never evaluates XC itself: the mean-field export's vxc.dat needs Vxc of
     * the converged SCF density, as pw2bgw takes it from QE's saved density.
     * Touches none of the iterator's own potentials or energies.
     */
    [[nodiscard]] std::vector<Complex>
    exchange_correlation_potential_g_of(const std::vector<std::vector<Real>>& charge_density_real) {
        return effective_potential_calculator()
            .compute(charge_density_real, number_of_spin_components(), local_ionic_potential_g())
            .exchange_correlation_potential_g;
    }

  protected:
    // ─── Pseudopotential config ──────────────────────────────────────
    const PseudopotentialSet* pseudopotential_set_ = nullptr;
    std::string calculation_ = "self_consistent_field";
    // Per-species initial magnetization (see KohnShamSCFIteratorConfig).
    std::vector<Real> starting_magnetization_;
    std::vector<Real> atomic_starting_magnetization_; // per atom; empty = per species
    std::vector<std::pair<std::string, Vector3D>> band_path_;
    int band_path_points_per_segment_ = 20;

    // ─── Pseudopotential fields ──────────────────────────────────────
    NormConservingPlanewaveKohnShamEnergies energies_;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system_snapshot_;

  private:
    /**
     * Optional owned PseudopotentialSet — used when the factory constructs the
     * iterator from InputData (the iterator owns the set rather than borrowing it).
     */
    std::optional<PseudopotentialSet> owned_pseudopotential_set_;
    /// Set once the geometry-dependent D3(BJ) dispersion energy has been computed.
    bool dispersion_energy_computed_ = false;
    /// Persistent effective potential calculator (caches real-space core charge density).
    std::unique_ptr<NormConservingPlanewaveEffectivePotentialCalculator>
        effective_potential_calculator_;

  public:
    /**
     * The projector cache is the SCF's too: the forces and stress build their
     * own. The applicator holds pointers into it, so its view is cleared first
     * and the storage freed after; the cache refills on demand.
     */
    void release_scf_workspace() override {
        PlanewaveKohnShamSCFIterator::release_scf_workspace();
        if (diagonalizer_) {
            for (int kpoint_index = 0;
                 kpoint_index < static_cast<int>(cached_kleinman_bylander_projectors_.size());
                 ++kpoint_index) {
                diagonalizer_->set_cached_projectors(kpoint_index, nullptr);
            }
        }
        std::vector<q3::Matrix<Complex>>().swap(cached_kleinman_bylander_projectors_);
    }

  protected:
    /// Cached KB projectors per k-point (reused by becsum accumulation).
    std::vector<q3::Matrix<Complex>> cached_kleinman_bylander_projectors_;

    /// DFT+U (Hubbard): resolved atom/shell manifolds; empty means no correction is active.
    std::vector<HubbardManifold> hubbard_manifolds_;
    /**
     * Tr[n] for each manifold and spin channel, as of the last SCF step that
     * built the occupation matrices. Indexed [manifold][spin], parallel to
     * hubbard_manifolds_; empty when no correction is active.
     */
    std::vector<std::vector<Real>> hubbard_occupation_traces_;
    /**
     * Cached Hubbard atomic-orbital projectors per k-point, all manifolds' orbitals
     * concatenated in hubbard_manifolds_ order (geometry-only, set once at setup).
     */
    std::vector<q3::Matrix<Complex>> cached_hubbard_projectors_;
    /**
     * The cell's symmetry operations, built once.
     *
     * Only DFT+U needs them at this level, and only per SCF step, so they are
     * cached lazily rather than computed in `initialize()` where every run
     * would pay for them.
     */
    [[nodiscard]] const std::vector<SymmetryOperation>& cached_symmetry_operations() const {
        if (!symmetry_operations_cached_) {
            cached_symmetry_operations_ =
                SymmetryAnalyzer::get_symmetry_operations(crystal_structure());
            symmetry_operations_cached_ = true;
        }
        return cached_symmetry_operations_;
    }
    mutable std::vector<SymmetryOperation> cached_symmetry_operations_;
    mutable bool symmetry_operations_cached_{false};

    /// Σ_σ Tr[V_U^σ n^σ]; see `hubbard_potential_energy()`.
    Real hubbard_potential_energy_{0.0};
    /**
     * Dense block-diagonal Hubbard D-matrix per spin channel (column-major, refreshed
     * once per SCF iteration in construct_density_from_states()).
     */
    std::vector<std::vector<Complex>> hubbard_d_matrix_per_spin_;
    int total_hubbard_orbitals_ = 0;
    /**
     * ‖n_fresh − n_damped‖² — how far the freshly accumulated occupation
     * matrix is from the damped one driving V_U, *not* the step actually
     * taken (which is this scaled by the damping factor). The distinction
     * matters because this is the quantity gating SCF exit. Same
     * squared-magnitude convention `density_residual_` uses. Updated by
     * `accumulate_hubbard_occupation_and_potential()`; consulted by
     * `is_iteration_converged()`. A large sentinel (never a false "converged")
     * until the damping cache has been through at least one real blend.
     */
    Real hubbard_occupation_residual_{std::numeric_limits<Real>::max()};
    /// Owns the damping/Anderson state; see HubbardOccupationMixer.
    HubbardOccupationMixer hubbard_occupation_mixer_;
    /**
     * Iterations of occupation seeding still owed. Starts at
     * 1 + `hubbard_fixed_occupation_iterations`: the first application is the
     * starting guess, any further ones hold that guess in place while the
     * wavefunctions adapt to it.
     */
    int hubbard_seed_iterations_remaining_ = 0;

    // ─── KohnShamSCFIterator hooks ────────────────────────────────────

    void initialize_representation() override {
        /**
         * Read pseudopotential config before calling parent (which reads PW config).
         * Skip fields already set by owned-data setters (InputData factory path).
         */
        if (config().pseudopotential_set != nullptr) {
            pseudopotential_set_ = config().pseudopotential_set;
        }
        calculation_ = config().calculation;
        starting_magnetization_ = config().starting_magnetization;
        atomic_starting_magnetization_ = config().atomic_starting_magnetization;
        bool any_nonzero_magnetization = false;
        for (const auto* magnetizations :
             {&starting_magnetization_, &atomic_starting_magnetization_})
            for (Real magnetization : *magnetizations) {
                if (magnetization < -1.0 || magnetization > 1.0)
                    throw std::runtime_error(
                        "starting_magnetization values must be in [-1, 1], got " +
                        std::to_string(magnetization));
                if (magnetization != 0.0)
                    any_nonzero_magnetization = true;
            }
        /**
         * Fixed occupations fill N/2 electrons per spin channel, so the TOTAL
         * moment is pinned to zero regardless of the starting magnetization
         * (QE rejects this input combination outright; see
         * number_of_occupied_bands_per_spin). Warn rather than silently
         * discarding the user's requested moment.
         */
        if (config().number_of_spin_components == 2 && any_nonzero_magnetization &&
            config().smearing == "none") {
            log::warn("LSDA with fixed occupations pins the total moment to zero; the "
                      "requested starting_magnetization can only seed a local (e.g. AFM) "
                      "spin arrangement. Use smearing to allow a net moment.");
        }
        band_path_ = config().band_path;
        band_path_points_per_segment_ = config().band_path_points_per_segment;

        PlanewaveKohnShamSCFIterator::initialize_representation();
    }

    void initialize_charge_density() override {
        create_norm_conserving_quantum_system_snapshot();
        if (has_preloaded_charge_density_ && preloaded_charge_density_is_a_starting_guess_) {
            if (loaded_charge_density_fits_grid()) {
                scale_loaded_charge_density_to_electron_count();
            } else {
                log::info("     The saved starting density does not fit this run's grid; "
                          "starting from overlapping atoms instead");
                has_preloaded_charge_density_ = false;
            }
        }
        if (has_preloaded_charge_density_) {
            preload_charge_density();
            on_charge_density_initialized();
        } else {
            initialize_charge_density_from_atomic_superposition();
        }
    }

    /**
     * Resolve DFT+U (Hubbard) manifolds and cache their atomic-orbital projectors, once,
     * for the whole SCF run (geometry-only, like KB projectors). No-op if config().hubbard
     * is empty. The one remaining scope gate is the Davidson diagonalizer.
     *
     * hubbard_projector_type selects "atomic" (raw pseudo-atomic orbitals, the default) or
     * "ortho-atomic" (the full system-wide atomic-orbital set Lowdin-orthogonalized jointly
     * before the manifold's columns are selected out -- see the call site below).
     * HubbardForceCalculator carries the ortho-atomic gradient for norm-conserving datasets,
     * including the term from every other orbital's own position-dependence through the joint
     * S^-1/2 transform.
     */
    void setup_hubbard() {
        if (config().hubbard.empty()) {
            return;
        }
        if (diagonalization_method_ != "davidson") {
            throw std::runtime_error(
                "DFT+U (hubbard) requires diagonalization: davidson (the direct diagonalizer "
                "does not go through the Hamiltonian applicator this feature hooks into)");
        }
        if (config().hubbard_projector_type != "atomic" &&
            config().hubbard_projector_type != "ortho-atomic") {
            throw std::runtime_error("DFT+U: unrecognized hubbard_projector_type '" +
                                     config().hubbard_projector_type +
                                     "' (expected 'atomic' or 'ortho-atomic')");
        }

        auto orbital_labels = enumerate_atomic_orbitals(
            crystal_structure(),
            quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set());
        hubbard_manifolds_ =
            resolve_hubbard_manifolds(crystal_structure(), orbital_labels, config().hubbard);

        /**
         * Occupation seeding directives are validated here, where the
         * manifolds are known, so a typo'd species or an impossible spin dies
         * at setup with a named cause instead of surfacing as a silently
         * unseeded run converging into the wrong minimum -- the exact failure
         * mode seeding exists to prevent.
         */
        for (const auto& directive : config().hubbard_starting_eigenvalues) {
            if (directive.spin_index >= number_of_spin_components())
                throw std::runtime_error(
                    "hubbard starting eigenvalue for species '" + directive.species_symbol +
                    "' addresses spin channel " + std::to_string(directive.spin_index + 1) +
                    " but the calculation has " + std::to_string(number_of_spin_components()) +
                    " spin component(s)");
            bool species_has_manifold =
                std::any_of(hubbard_manifolds_.begin(),
                            hubbard_manifolds_.end(),
                            [&](const HubbardManifold& manifold) {
                                return crystal_structure().element(manifold.atom_index) ==
                                       directive.species_symbol;
                            });
            if (!species_has_manifold)
                throw std::runtime_error("hubbard starting eigenvalue names species '" +
                                         directive.species_symbol +
                                         "', which has no Hubbard manifold");
        }
        hubbard_seed_iterations_remaining_ =
            config().hubbard_starting_eigenvalues.empty()
                ? 0
                : 1 + std::max(0, config().hubbard_fixed_occupation_iterations);

        total_hubbard_orbitals_ = 0;
        for (const auto& manifold : hubbard_manifolds_) {
            total_hubbard_orbitals_ += static_cast<int>(manifold.orbital_indices.size());
        }

        hubbard_d_matrix_per_spin_.assign(
            number_of_spin_components(),
            std::vector<Complex>(static_cast<size_t>(total_hubbard_orbitals_) *
                                     total_hubbard_orbitals_,
                                 Complex(0.0, 0.0)));

        constexpr Real q_spacing = 0.01;
        constexpr int number_of_q_points = 900;
        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set(),
            crystal_structure().cell_volume(),
            q_spacing,
            number_of_q_points);

        int number_of_kpoints = kpoint_grid().number_of_kpoints();
        cached_hubbard_projectors_.resize(number_of_kpoints);
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
            std::vector<std::vector<Complex>> all_atomic_orbitals;
            build_atomic_orbitals_g_space(
                kpoint_index,
                number_of_plane_waves(kpoint_index),
                g_vector_indices(kpoint_index),
                crystal_structure(),
                kpoint_grid(),
                planewave_basis_,
                quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set(),
                atomic_wavefunction_table,
                q_spacing,
                number_of_q_points,
                all_atomic_orbitals);

            int plane_waves_at_kpoint = number_of_plane_waves(kpoint_index);
            int number_of_atomic_orbitals = static_cast<int>(all_atomic_orbitals.size());
            std::unique_ptr<OverlapOperator> overlap_operator;
            q3::Matrix<Complex> kleinman_bylander_projectors;
            if (quantum_system_snapshot_->has_ultrasoft()) {
                const auto& pseudopotential_basis =
                    quantum_system_snapshot_->pseudopotential_basis();
                compute_kleinman_bylander_projectors(
                    kpoint_grid()[kpoint_index].coordinates,
                    crystal_structure(),
                    planewave_basis_,
                    pseudopotential_basis.pseudopotential_set(),
                    pseudopotential_basis.beta_function_table(),
                    g_vector_indices(kpoint_index),
                    plane_waves_at_kpoint,
                    pseudopotential_basis.total_number_of_projectors(),
                    kleinman_bylander_projectors);
                overlap_operator = create_overlap_operator(crystal_structure(),
                                                           *quantum_system_snapshot_,
                                                           kleinman_bylander_projectors,
                                                           *execution_policy_);
            }

            /**
             * ortho-atomic: Lowdin-orthogonalize the FULL system-wide atomic-orbital set
             * (all atoms, all shells) jointly -- matching QE's convention, which
             * orthogonalizes the system-wide atomic-orbital set first -- before
             * selecting out the Hubbard manifold's columns below. This must operate on the
             * whole set, not a per-manifold subset: a manifold-only Lowdin transform would
             * be a materially different (non-standard) definition.
             */
            if (config().hubbard_projector_type == "ortho-atomic") {
                std::vector<Complex> packed_atomic_orbitals(
                    static_cast<size_t>(plane_waves_at_kpoint) * number_of_atomic_orbitals);
                for (int orbital_index = 0; orbital_index < number_of_atomic_orbitals;
                     orbital_index++) {
                    std::copy(all_atomic_orbitals[static_cast<size_t>(orbital_index)].begin(),
                              all_atomic_orbitals[static_cast<size_t>(orbital_index)].end(),
                              packed_atomic_orbitals.begin() +
                                  static_cast<size_t>(orbital_index) * plane_waves_at_kpoint);
                }

                /**
                 * Ultrasoft and PAW ortho-atomic projectors use the generalized
                 * <phi_i|S|phi_j> metric. Apply the same production overlap operator used
                 * by Davidson so its beta-projector ordering, q matrix, Gamma completion,
                 * and band-group reduction cannot drift from the eigenproblem definition.
                 */
                std::vector<Complex> overlap_applied_atomic_orbitals;
                const Complex* metric_right_orbitals = packed_atomic_orbitals.data();
                if (overlap_operator != nullptr) {
                    overlap_applied_atomic_orbitals.resize(packed_atomic_orbitals.size());
                    overlap_operator->apply_batch(plane_waves_at_kpoint,
                                                  number_of_atomic_orbitals,
                                                  packed_atomic_orbitals.data(),
                                                  overlap_applied_atomic_orbitals.data());
                    metric_right_orbitals = overlap_applied_atomic_orbitals.data();
                }

                std::vector<Complex> atomic_orbital_overlap;
                compute_hubbard_projector_overlaps(plane_waves_at_kpoint,
                                                   number_of_atomic_orbitals,
                                                   number_of_atomic_orbitals,
                                                   packed_atomic_orbitals.data(),
                                                   metric_right_orbitals,
                                                   atomic_orbital_overlap);
                /**
                 * The Lowdin metric is itself a physical G-space overlap. Gamma-only storage
                 * contains one member of each +/-G pair, so diagonalizing the raw half-sphere
                 * sum would define a different projector from the explicit full-sphere run.
                 * Complete the metric before the band-group reduction and eigendecomposition,
                 * using the same overlap rule as the Hubbard occupation path below.
                 */
                if (planewave_basis_.is_gamma_only()) {
                    convert_gamma_only_projector_overlaps_to_full_sphere(
                        plane_waves_at_kpoint,
                        number_of_atomic_orbitals,
                        number_of_atomic_orbitals,
                        packed_atomic_orbitals.data(),
                        metric_right_orbitals,
                        find_g_zero_plane_wave_index(planewave_basis_,
                                                     g_vector_indices(kpoint_index),
                                                     plane_waves_at_kpoint),
                        /*band_functions_carry_g_component=*/false,
                        atomic_orbital_overlap);
                }
                execution_policy_->reduce_within_group(
                    atomic_orbital_overlap.data(), static_cast<int>(atomic_orbital_overlap.size()));
                lowdin_orthogonalize_orbitals(all_atomic_orbitals,
                                              std::move(atomic_orbital_overlap));
            }

            cached_hubbard_projectors_[kpoint_index].resize(total_hubbard_orbitals_,
                                                            number_of_plane_waves(kpoint_index));
            int row_index = 0;
            for (const auto& manifold : hubbard_manifolds_) {
                for (int orbital_index : manifold.orbital_indices) {
                    for (int g_index = 0; g_index < number_of_plane_waves(kpoint_index);
                         ++g_index) {
                        cached_hubbard_projectors_[kpoint_index][row_index][g_index] =
                            all_atomic_orbitals[static_cast<size_t>(orbital_index)]
                                               [static_cast<size_t>(g_index)];
                    }
                    row_index++;
                }
            }
            /**
             * For ultrasoft and PAW the Hubbard occupations are defined with
             * the all-electron inner product, <phi_m|S|psi>, not <phi_m|psi>:
             * the pseudo orbital misses the on-site charge the augmentation
             * carries, which for a 3d manifold is most of it.
             *
             * Rather than thread S through the occupation matrix, the
             * Hamiltonian term and the projector overlaps separately, S is
             * applied once here to the cached projectors. S is Hermitian, so
             * <S phi_m|psi> = <phi_m|S|psi>, and every consumer downstream —
             * the occupation matrix, the applicator's |phi> V <phi| term —
             * becomes correct with no further change.
             */
            if (overlap_operator != nullptr) {
                std::vector<Complex> overlap_applied_hubbard_projectors(
                    static_cast<size_t>(plane_waves_at_kpoint) * total_hubbard_orbitals_);
                overlap_operator->apply_batch(plane_waves_at_kpoint,
                                              total_hubbard_orbitals_,
                                              cached_hubbard_projectors_[kpoint_index].data(),
                                              overlap_applied_hubbard_projectors.data());
                std::copy(overlap_applied_hubbard_projectors.begin(),
                          overlap_applied_hubbard_projectors.end(),
                          cached_hubbard_projectors_[kpoint_index].data());
            }

            diagonalizer_->set_cached_hubbard_projectors(kpoint_index,
                                                         &cached_hubbard_projectors_[kpoint_index]);
        }

        log::info("     DFT+U (Hubbard): {} manifold(s), {} total orbital(s), {} projectors",
                  hubbard_manifolds_.size(),
                  total_hubbard_orbitals_,
                  config().hubbard_projector_type);
    }

    void initialize_single_particle_states() override {
        setup_hubbard();
        if (quantum_system_.preloaded_quantum_system_ != nullptr) {
            preload_single_particle_states();

            int number_of_kpoints = kpoint_grid().number_of_kpoints();
            cached_kleinman_bylander_projectors_.resize(number_of_kpoints);
            for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
                compute_kleinman_bylander_projectors(
                    kpoint_grid()[kpoint_index].coordinates,
                    crystal_structure(),
                    planewave_basis_,
                    quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set(),
                    quantum_system_snapshot_->pseudopotential_basis().beta_function_table(),
                    g_vector_indices(kpoint_index),
                    number_of_plane_waves(kpoint_index),
                    quantum_system_snapshot_->pseudopotential_basis().total_number_of_projectors(),
                    cached_kleinman_bylander_projectors_[kpoint_index]);
                diagonalizer_->set_cached_projectors(
                    kpoint_index, &cached_kleinman_bylander_projectors_[kpoint_index]);
            }
            return;
        }

        auto timer_0 = std::chrono::high_resolution_clock::now();
        NormConservingPlanewaveWavefunctionInitializer atomic_wavefunction_initializer;
        atomic_wavefunction_initializer.initialize_interpolation_table(
            quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set(),
            crystal_structure().cell_volume());
        int number_of_atomic_wavefunctions = count_atomic_orbitals(
            crystal_structure(),
            quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set());
        log_pseudopotential_summary(number_of_atomic_wavefunctions);
        report_memory_estimate(number_of_atomic_wavefunctions);

        auto timer_1 = std::chrono::high_resolution_clock::now();
        update_effective_potential();
        prepare_hamiltonian();
        auto timer_2 = std::chrono::high_resolution_clock::now();
        ProfilingCounters::instance().log_resident_checkpoint("potential + hamiltonian");

        int number_of_kpoints = kpoint_grid().number_of_kpoints();

        // Pre-compute KB projectors for each k-point (needed for initial density construction)
        cached_kleinman_bylander_projectors_.resize(number_of_kpoints);
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
            compute_kleinman_bylander_projectors(
                kpoint_grid()[kpoint_index].coordinates,
                crystal_structure(),
                planewave_basis_,
                quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set(),
                quantum_system_snapshot_->pseudopotential_basis().beta_function_table(),
                g_vector_indices(kpoint_index),
                number_of_plane_waves(kpoint_index),
                quantum_system_snapshot_->pseudopotential_basis().total_number_of_projectors(),
                cached_kleinman_bylander_projectors_[kpoint_index]);
            // Wire the cached projectors to the diagonalizer to avoid recomputation
            diagonalizer_->set_cached_projectors(
                kpoint_index, &cached_kleinman_bylander_projectors_[kpoint_index]);
        }
        auto timer_3 = std::chrono::high_resolution_clock::now();
        ProfilingCounters::instance().log_resident_checkpoint("projector caching");

        /**
         * The starting-guess Hamiltonian multiplies by the local potential on the
         * wavefunction grid, pre-scaled by 1/N the way the Hamiltonian
         * application holds it: restricted from the density grid through G space,
         * completed within the pool, transformed back. The initializer took that
         * conversion out of its own hands when the grids were first distributed,
         * and this call went on passing the density-grid potential unconverted --
         * read on the wrong grid, N times too large and different on every
         * topology. Wherever there were more atomic orbitals than bands, the bands
         * kept were chosen with it.
         */
        int total_smooth_points = planewave_basis_.fft_wave().total_grid_points();
        std::vector<std::vector<Real>> starting_guess_potential(
            static_cast<std::size_t>(number_of_spin_components()));
        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            auto smooth_potential_g = compute_smooth_grid_potential_from_dense(
                real_space_potential_[spin_index], planewave_basis_);
            execution_policy_->reduce_group_distributed_result(smooth_potential_g.data(),
                                                               total_smooth_points);
            auto& potential = starting_guess_potential[static_cast<std::size_t>(spin_index)];
            potential = inverse_fft_smooth_potential(smooth_potential_g, planewave_basis_);
            Real inverse_smooth_points = 1.0 / total_smooth_points;
            for (auto& value : potential) {
                value *= inverse_smooth_points;
            }
        }

        int number_of_kpoint_states = number_of_kpoints * number_of_spin_components();
        double atomic_wfc_ms = 0.0;
        double rotate_ms = 0.0;
        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             ++kpoint_index_state) {
            int spin_index = kpoint_index_state / number_of_kpoints;
            int kpoint_index = kpoint_index_state % number_of_kpoints;

            auto timer_awfc_start = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<Complex>> atomic_wavefunctions;
            atomic_wavefunction_initializer.compute_atomic_wavefunctions(
                kpoint_index,
                number_of_plane_waves(kpoint_index),
                number_of_bands_,
                g_vector_indices(kpoint_index),
                crystal_structure(),
                kpoint_grid(),
                planewave_basis_,
                quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set(),
                atomic_wavefunctions);
            auto timer_awfc_end = std::chrono::high_resolution_clock::now();
            atomic_wfc_ms +=
                std::chrono::duration<double, std::milli>(timer_awfc_end - timer_awfc_start)
                    .count();
            ProfilingCounters::instance().log_resident_checkpoint_debug("atomic orbitals");

            atomic_wavefunction_initializer.rotate_wavefunctions(
                kpoint_index,
                kpoint_index_state,
                number_of_plane_waves(kpoint_index),
                number_of_bands_,
                g_vector_indices(kpoint_index),
                kinetic_energy_squared()[kpoint_index],
                starting_guess_potential[static_cast<std::size_t>(spin_index)],
                crystal_structure(),
                kpoint_grid(),
                planewave_basis_,
                *execution_policy_,
                *quantum_system_snapshot_,
                atomic_wavefunctions,
                quantum_system_,
                cached_kleinman_bylander_projectors_[kpoint_index]);
            auto timer_rotate_end = std::chrono::high_resolution_clock::now();
            rotate_ms +=
                std::chrono::duration<double, std::milli>(timer_rotate_end - timer_awfc_end)
                    .count();
            ProfilingCounters::instance().log_resident_checkpoint_debug("orbital rotation");
        }
        auto timer_4 = std::chrono::high_resolution_clock::now();
        ProfilingCounters::instance().release_unused_heap("wavefunction init");
        ProfilingCounters::instance().log_resident_checkpoint("wavefunction init");

        log::debug("     wfc_init breakdown:");
        log::debug("       interp_table+count: {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_1 - timer_0).count());
        log::debug("       effective_potential: {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_2 - timer_1).count());
        log::debug("       kb_projectors:      {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_3 - timer_2).count());
        log::debug("       atomic_wfc_gen:     {:.1f} ms", atomic_wfc_ms);
        log::debug("       rotate_wfc:         {:.1f} ms", rotate_ms);
        log::debug("       atomic_wfc+rotate:  {:.1f} ms",
                   std::chrono::duration<double, std::milli>(timer_4 - timer_3).count());
    }

    void solve_single_particle_states() override {
        diagonalizer_->set_first_iteration(iteration_ == 1);

        int number_of_kpoints = kpoint_grid().number_of_kpoints();
        int number_of_kpoint_states = number_of_kpoints * number_of_spin_components();

        // Wire or pre-compute KB projectors for each k-point (reused by becsum accumulation)
        if (cached_kleinman_bylander_projectors_.empty()) {
            cached_kleinman_bylander_projectors_.resize(number_of_kpoints);
            for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
                compute_kleinman_bylander_projectors(
                    kpoint_grid()[kpoint_index].coordinates,
                    crystal_structure(),
                    planewave_basis_,
                    quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set(),
                    quantum_system_snapshot_->pseudopotential_basis().beta_function_table(),
                    g_vector_indices(kpoint_index),
                    number_of_plane_waves(kpoint_index),
                    quantum_system_snapshot_->pseudopotential_basis().total_number_of_projectors(),
                    cached_kleinman_bylander_projectors_[kpoint_index]);
                diagonalizer_->set_cached_projectors(
                    kpoint_index, &cached_kleinman_bylander_projectors_[kpoint_index]);
            }
        } else {
            for (int kpoint_index = 0; kpoint_index < number_of_kpoints; ++kpoint_index) {
                diagonalizer_->set_cached_projectors(
                    kpoint_index, &cached_kleinman_bylander_projectors_[kpoint_index]);
            }
        }

        /**
         * ── Forward pass: solve k-points 0, 1, 2, ..., N-1 ──
         * Each k-point's converged wavefunctions seed the next k-point.
         * At band-path segment boundaries (high-symmetry points where the path
         * direction changes), propagation is skipped to avoid biasing the Davidson
         * solver with wavefunctions from the wrong symmetry direction.
         */
        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             ++kpoint_index_state) {
            int spin_index = kpoint_index_state / number_of_kpoints;
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            // Invalidate cache when the spin channel (and thus potential) changes
            if (kpoint_index == 0) {
                diagonalizer_->invalidate_smooth_potential_cache();
                if (!hubbard_manifolds_.empty()) {
                    diagonalizer_->set_hubbard_d_matrix(&hubbard_d_matrix_per_spin_[spin_index]);
                }
            }
            diagonalizer_->solve(kpoint_index_state,
                                 kpoint_index,
                                 number_of_plane_waves(kpoint_index),
                                 number_of_bands_,
                                 g_vector_indices(kpoint_index),
                                 kinetic_energy_squared()[kpoint_index],
                                 real_space_potential_[spin_index],
                                 crystal_structure(),
                                 kpoint_grid(),
                                 planewave_basis_,
                                 *quantum_system_snapshot_,
                                 quantum_system_);
            report_unconverged_solve(kpoint_index_state);

            /**
             * Propagate converged wavefunctions to the next k-point, unless we
             * are at a segment boundary where the path direction changes.
             *
             * HARD RULE: Propagation by direct coefficient copying is only valid if
             * the G-vector bases are identical. In Q3, k-points have different G-spheres
             * and sortings. Direct propagation destroys the good starting guess from
             * the previous SCF iteration. Disable for SCF.
             */
            bool is_scf = (calculation_ == "scf" || calculation_ == "self_consistent_field");
            if (!is_scf && !is_at_band_path_segment_boundary(kpoint_index)) {
                propagate_wavefunctions_to_next_kpoint(kpoint_index_state, number_of_kpoint_states);
            }
        }

        /**
         * ── Backward fix: re-solve edge k-points using neighbor seeds ──
         * This is only relevant for band-path calculations to fix symmetry-breaking
         * at segment boundaries. Skip for SCF.
         */
        if (calculation_ != "scf" && calculation_ != "self_consistent_field") {
            fix_edge_kpoints_with_backward_propagation(number_of_kpoints, number_of_kpoint_states);
        }
    }

    void construct_density_from_states() override {
        PlanewaveKohnShamSCFIterator::construct_density_from_states();
        accumulate_hubbard_occupation_and_potential();
    }

    /**
     * The base check alone lets a DFT+U run exit while the (now damped, see
     * `previous_hubbard_occupation_matrices_`) occupation matrix is still
     * visibly moving iteration to iteration: the density can be
     * self-consistent while V_U, built from a still-relaxing n, is not.
     * `hubbard_occupation_residual_` is updated for *this* iteration inside
     * `construct_density_from_states()`, which `perform_iteration()` always
     * runs before this is checked (`SCFIterator::iterate()`), so it is never
     * stale by an iteration. The `hubbard_manifolds_.empty()` short-circuit
     * means non-Hubbard runs never consult the residual at all, so its
     * "unconverged" default (see the member doc) never matters for them.
     *
     * `occupation_matrices` going into the damping step is already pool- and
     * band-reduced (`accumulate_hubbard_occupation_and_potential`'s own MPI
     * reduction, upstream of the damping block), and `mixing_beta_` is the
     * same scalar on every rank, so `hubbard_occupation_residual_` is
     * deterministic pure arithmetic on already-rank-consistent inputs —
     * unlike `density_residual_`, it needs no `reduce_scalar_max` of its own
     * (see that residual's docstring for why *it* needs one: per-rank FFTW
     * plan noise, which nothing in the damping step re-introduces).
     */
    bool is_iteration_converged() const override {
        return PlanewaveKohnShamSCFIterator::is_iteration_converged() &&
               (hubbard_manifolds_.empty() || !hubbard_occupation_mixer_.damping_active() ||
                hubbard_occupation_residual_ < convergence_threshold_);
    }

    /**
     * Recompute the DFT+U occupation matrix from this iteration's converged wavefunctions,
     * then the Dudarev energy and D-matrix for each spin channel. Called once per SCF
     * iteration (after all k-points/spins are solved), mirroring
     * UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::accumulate_projector_overlaps()'s
     * k-point-state loop and pool-reduction pattern. No-op if hubbard_manifolds_ is empty.
     */
    void accumulate_hubbard_occupation_and_potential() {
        if (hubbard_manifolds_.empty()) {
            return;
        }

        int number_of_kpoints = kpoint_grid().number_of_kpoints();
        int number_of_kpoint_states = number_of_kpoints * number_of_spin_components();
        size_t number_of_manifolds = hubbard_manifolds_.size();

        std::vector<int> manifold_row_offsets(number_of_manifolds);
        int total_hubbard_orbitals = 0;
        {
            int offset = 0;
            for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                 ++manifold_index) {
                manifold_row_offsets[manifold_index] = offset;
                offset +=
                    static_cast<int>(hubbard_manifolds_[manifold_index].orbital_indices.size());
            }
            total_hubbard_orbitals = offset;
        }

        // occupation_matrices[manifold_index][spin_index] = row-major manifold_size^2.
        std::vector<std::vector<std::vector<Complex>>> occupation_matrices(number_of_manifolds);
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; ++manifold_index) {
            int manifold_size =
                static_cast<int>(hubbard_manifolds_[manifold_index].orbital_indices.size());
            occupation_matrices[manifold_index].assign(
                number_of_spin_components(),
                std::vector<Complex>(static_cast<size_t>(manifold_size) * manifold_size,
                                     Complex(0.0, 0.0)));
        }

        for (int kpoint_index_state = 0; kpoint_index_state < number_of_kpoint_states;
             ++kpoint_index_state) {
            int spin_index = kpoint_index_state / number_of_kpoints;
            int kpoint_index = kpoint_index_state % number_of_kpoints;
            Real kpoint_weight = kpoint_grid()[kpoint_index].weight;
            int plane_waves_at_kpoint = number_of_plane_waves(kpoint_index);
            const auto& hubbard_projectors = cached_hubbard_projectors_[kpoint_index];

            std::vector<Real> band_occupations(number_of_bands_);
            for (int band_index = 0; band_index < number_of_bands_; ++band_index) {
                band_occupations[band_index] =
                    quantum_system_.occupation_by_kpoint_state_index_and_band_index(
                        kpoint_index_state, band_index);
            }

            /**
             * All <phi_j|psi_b> for this k-point in one ZGEMM -- see
             * compute_hubbard_projector_overlaps()'s docstring for why this is batched over
             * every Hubbard orbital rather than looped per manifold. Only occupied bands are
             * contracted, as the scalar loop this replaced also skipped the rest.
             */
            std::vector<int> occupied_band_indices;
            for (int band_index = 0; band_index < number_of_bands_; ++band_index) {
                if (band_occupations[band_index] >= math::tolerances::occupation_threshold) {
                    occupied_band_indices.push_back(band_index);
                }
            }
            int number_of_occupied_bands = static_cast<int>(occupied_band_indices.size());

            std::vector<Complex> packed_projectors(static_cast<size_t>(plane_waves_at_kpoint) *
                                                   total_hubbard_orbitals);
            for (int orbital_index = 0; orbital_index < total_hubbard_orbitals; ++orbital_index) {
                const Complex* projector = hubbard_projectors[orbital_index];
                std::copy(projector,
                          projector + plane_waves_at_kpoint,
                          packed_projectors.begin() +
                              static_cast<size_t>(orbital_index) * plane_waves_at_kpoint);
            }

            std::vector<Complex> occupied_wavefunctions(static_cast<size_t>(plane_waves_at_kpoint) *
                                                        number_of_occupied_bands);
            for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                 ++occupied_band_index) {
                int band_index = occupied_band_indices[occupied_band_index];
                size_t offset = static_cast<size_t>(occupied_band_index) * plane_waves_at_kpoint;
                for (int g_index = 0; g_index < plane_waves_at_kpoint; ++g_index) {
                    occupied_wavefunctions[offset + g_index] =
                        quantum_system_.wavefunction(kpoint_index_state, band_index, g_index);
                }
            }

            std::vector<Complex> batched_overlaps;
            compute_hubbard_projector_overlaps(plane_waves_at_kpoint,
                                               total_hubbard_orbitals,
                                               number_of_occupied_bands,
                                               packed_projectors.data(),
                                               occupied_wavefunctions.data(),
                                               batched_overlaps);

            /**
             * Gamma-only sums over half the G sphere, so <phi_m|psi> comes out
             * at half its full-sphere value plus an Im-Im cross term. Completing
             * it here rather than inside the occupation kernel keeps n_mm' =
             * conj(O_m) O_m' exactly as written, and makes n real symmetric --
             * which in turn makes the Dudarev D matrix real, so the Hamiltonian
             * term it feeds preserves the half sphere's reality condition.
             */
            if (planewave_basis_.is_gamma_only()) {
                convert_gamma_only_projector_overlaps_to_full_sphere(
                    plane_waves_at_kpoint,
                    total_hubbard_orbitals,
                    number_of_occupied_bands,
                    packed_projectors.data(),
                    occupied_wavefunctions.data(),
                    find_g_zero_plane_wave_index(
                        planewave_basis_, g_vector_indices(kpoint_index), plane_waves_at_kpoint),
                    /*band_functions_carry_g_component=*/false,
                    batched_overlaps);
            }

            for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                 ++manifold_index) {
                int manifold_size =
                    static_cast<int>(hubbard_manifolds_[manifold_index].orbital_indices.size());
                int row_offset = manifold_row_offsets[manifold_index];

                /**
                 * Scatter this manifold's row block; unoccupied bands stay zero, exactly as
                 * the scalar loop's `continue` left them.
                 */
                Matrix<Complex> overlaps(static_cast<size_t>(number_of_bands_),
                                         static_cast<size_t>(manifold_size));
                for (int occupied_band_index = 0; occupied_band_index < number_of_occupied_bands;
                     ++occupied_band_index) {
                    size_t band_index =
                        static_cast<size_t>(occupied_band_indices[occupied_band_index]);
                    size_t column_offset =
                        static_cast<size_t>(occupied_band_index) * total_hubbard_orbitals;
                    for (int m = 0; m < manifold_size; ++m) {
                        overlaps(band_index, static_cast<size_t>(m)) =
                            batched_overlaps[column_offset + static_cast<size_t>(row_offset) +
                                             static_cast<size_t>(m)];
                    }
                }
                /**
                 * Under MPI band-parallelism (G-vector distribution within a k-point pool),
                 * each rank's g_index loop above only covers its LOCAL plane-wave subset, so
                 * overlaps here is only a partial <phi_m|psi> sum -- reduce across the
                 * band-parallel group before use, mirroring the applicator's identical
                 * <phi_m|psi> reduction for the H|psi> Hubbard term
                 * (hamiltonian_applicator/serial.hpp's reduce_within_group(hubbard_overlaps, ...)).
                 * A no-op when band_size == 1 (serial ExecutionPolicy's reduce_within_group()).
                 */
                execution_policy_->reduce_within_group(overlaps.data(),
                                                       number_of_bands_ * manifold_size);
                accumulate_hubbard_occupation_matrix(
                    number_of_bands_,
                    manifold_size,
                    overlaps,
                    band_occupations,
                    kpoint_weight,
                    occupation_matrices[manifold_index][spin_index]);
            }
        }

        /**
         * Each pool computed the occupation matrix only for its subset of k-points.
         * Flatten (Complex has no reduce_vector overload) -> reduce across pools -> unflatten.
         */
        std::vector<Real> flat_buffer;
        for (const auto& per_spin_matrices : occupation_matrices) {
            for (const auto& matrix : per_spin_matrices) {
                for (const auto& element : matrix) {
                    flat_buffer.push_back(element.real());
                    flat_buffer.push_back(element.imag());
                }
            }
        }
        execution_policy_->reduce_vector(flat_buffer);
        {
            size_t flat_index = 0;
            for (auto& per_spin_matrices : occupation_matrices) {
                for (auto& matrix : per_spin_matrices) {
                    for (auto& element : matrix) {
                        element = Complex(flat_buffer[flat_index], flat_buffer[flat_index + 1]);
                        flat_index += 2;
                    }
                }
            }
        }

        /**
         * Restore the symmetry the IBZ sum does not carry: time reversal, then
         * the spatial operations.
         *
         * `n` has angular indices, so a weighted irreducible-wedge sum does not
         * reproduce the full-zone result the way an invariant scalar does — the
         * trace comes out right and the structure does not. Averaging over the
         * group is what QE does too; without it silicon's 3p
         * manifold, which symmetry forces to be isotropic, came out anisotropic
         * enough to move E_U from 0.1918 to 0.1204 Ry.
         *
         * The wedge was reduced by the spatial operations *and* time reversal,
         * so both have to come back; the first call is the Θ half and is
         * described where it is defined. It is a separate call rather than
         * another operation inside the group loop because Θ is antiunitary — it
         * conjugates `n` instead of rotating its indices, and no real D(R) can
         * express that. The two commute (a real D preserves realness), so the
         * order is only a matter of which one gets to run on the smaller job.
         *
         * Must run after both reductions: the matrix has to be complete before
         * it is averaged.
         *
         * Unconditional, with no "was symmetry used?" flag to thread through:
         * a full-zone sum already produces an invariant matrix, and averaging
         * an invariant over the group returns it unchanged. Gamma-only is the
         * one exclusion for the spatial half — its half-sphere basis breaks the
         * point group by construction, the same reason the density symmetrizer
         * stands down there. Θ needs no such exclusion: a Γ-only `n` is real
         * already.
         */
        symmetrize_hubbard_occupation_matrices_under_time_reversal(occupation_matrices);
        if (!planewave_basis_.is_gamma_only()) {
            std::vector<HubbardManifoldLayout> layouts;
            layouts.reserve(number_of_manifolds);
            for (const auto& manifold : hubbard_manifolds_)
                layouts.push_back({manifold.atom_index,
                                   manifold.angular_momentum,
                                   static_cast<int>(manifold.orbital_indices.size())});
            symmetrize_hubbard_occupation_matrices(
                crystal_structure(), cached_symmetry_operations(), layouts, occupation_matrices);
        }

        /**
         * First iteration only, after the matrix is complete (reduced and
         * symmetrized) and before anything reads it: replace the named
         * eigenvalues. The damping cache bootstraps from these seeded
         * matrices below, and the first Hubbard potential is built from
         * them, which is the entire mechanism -- V_U starts in the wanted
         * basin and self-consistency takes over from there.
         *
         * The user value is per-spin ([0, 1]); the raw matrix carries the
         * k-weight spin factor, so scale by spin_degeneracy() on the way in
         * (a factor of 1 for a spin-polarized run).
         */
        if (hubbard_seed_iterations_remaining_ > 0) {
            bool seeding_already_reported =
                hubbard_seed_iterations_remaining_ <
                1 + std::max(0, config().hubbard_fixed_occupation_iterations);
            for (size_t manifold_index = 0; manifold_index < number_of_manifolds;
                 ++manifold_index) {
                const auto& manifold = hubbard_manifolds_[manifold_index];
                int manifold_size = static_cast<int>(manifold.orbital_indices.size());
                const std::string& atom_species = crystal_structure().element(manifold.atom_index);
                for (const auto& directive : config().hubbard_starting_eigenvalues) {
                    if (directive.species_symbol != atom_species)
                        continue;
                    if (directive.eigenvalue_index >= manifold_size)
                        throw std::runtime_error("hubbard starting eigenvalue " +
                                                 std::to_string(directive.eigenvalue_index + 1) +
                                                 " for species '" + directive.species_symbol +
                                                 "' exceeds the manifold size " +
                                                 std::to_string(manifold_size));
                    apply_hubbard_starting_eigenvalue(
                        occupation_matrices[manifold_index]
                                           [static_cast<size_t>(directive.spin_index)],
                        manifold_size,
                        directive.eigenvalue_index,
                        directive.target_occupation * spin_degeneracy());
                    if (!seeding_already_reported)
                        log::info("     DFT+U: seeding atom {} spin {} eigenvalue {} -> {:.4f}",
                                  manifold.atom_index + 1,
                                  directive.spin_index + 1,
                                  directive.eigenvalue_index + 1,
                                  directive.target_occupation);
                }
            }
            hubbard_seed_iterations_remaining_--;
        }

        /**
         * Tr[n] per manifold, per spin — the first thing to look at when a
         * DFT+U result looks wrong.
         *
         * It is the number of electrons the manifold's projectors actually see,
         * so it says immediately whether the projection is capturing the shell.
         * For ultrasoft and PAW that depends on the overlap operator: the
         * pseudo orbital alone misses the on-site charge the augmentation
         * carries. Measured on fcc nickel with a PAW dataset: Tr[n] = 8.83
         * with the overlap operator and 2.71 without it, so the bare pseudo
         * projection sees under a third of the d shell. Logged rather than left
         * to a debugger, because the failure it diagnoses is a converged,
         * plausible, wrong answer.
         */
        hubbard_occupation_traces_.assign(number_of_manifolds, {});
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; ++manifold_index) {
            int manifold_size =
                static_cast<int>(hubbard_manifolds_[manifold_index].orbital_indices.size());
            std::string traces;
            for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
                Real trace = 0.0;
                for (int m = 0; m < manifold_size; m++)
                    trace += occupation_matrices[manifold_index][spin_index]
                                                [static_cast<size_t>(m) * manifold_size + m]
                                                    .real();
                traces += fmt::format("{}{:.4f}", spin_index == 0 ? "" : " / ", trace);
                /**
                 * Kept at full precision alongside the four-decimal line, and
                 * recorded here rather than recomputed for the results file, so
                 * the two can never name different numbers -- the same reason
                 * the SCF history is recorded where its line is printed.
                 */
                hubbard_occupation_traces_[manifold_index].push_back(trace);
            }
            log::info("     DFT+U: atom {} Tr[n] = {}",
                      hubbard_manifolds_[manifold_index].atom_index + 1,
                      traces);
        }

        /**
         * Damp the occupation matrix before it drives V_U. The mixer decides
         * whether damping is warranted at all and whether to accelerate; see
         * `HubbardOccupationMixer` for why each of those is conditional.
         */
        bool damping_was_active = hubbard_occupation_mixer_.damping_active();
        const auto& damped_occupation_matrices = hubbard_occupation_mixer_.blend(
            occupation_matrices, mixing_beta_, convergence_threshold_);
        hubbard_occupation_residual_ = hubbard_occupation_mixer_.residual();
        if (!damping_was_active && hubbard_occupation_mixer_.damping_active())
            log::info("     DFT+U: occupation matrix is not converging monotonically; "
                      "damping it at mixing_beta from here");

        /**
         * Dudarev energy + per-spin D-matrix, assembled into the block-diagonal buffers
         * the applicator expects (column-major, one buffer per spin).
         *
         * Normalization: kpoint_grid() weights already carry the spin factor (they sum
         * to 2 for spin-unpolarized runs, matching the electron-counting convention
         * documented in fermi_energy/planewave/init.hpp), so the n_mm' accumulated above
         * has eigenvalues up to spin_degeneracy(), not the [0,1] the Dudarev formula
         * (E_U = (U/2) Tr[n(1-n)]) is defined for. Rescale n by 1/spin_degeneracy() before
         * the formula, then scale the resulting per-channel energy back up by
         * spin_degeneracy() -- physically, an unpolarized run has two identical,
         * degenerate spin channels each contributing this same per-spin energy. The
         * D-matrix is used as computed (already expressed per spin channel, matching how
         * it is applied to a single, spin-independent wavefunction solve at nspin=1).
         */
        Real spin_degeneracy_factor = spin_degeneracy();
        Real total_hubbard_energy = 0.0;
        Real total_hubbard_potential_energy = 0.0;
        for (auto& buffer : hubbard_d_matrix_per_spin_) {
            std::fill(buffer.begin(), buffer.end(), Complex(0.0, 0.0));
        }
        for (size_t manifold_index = 0; manifold_index < number_of_manifolds; ++manifold_index) {
            int manifold_size =
                static_cast<int>(hubbard_manifolds_[manifold_index].orbital_indices.size());
            int row_offset = manifold_row_offsets[manifold_index];
            Real u_rydberg = hubbard_manifolds_[manifold_index].u_rydberg;
            for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
                /**
                 * A scaled copy, not a reference: this is the mixer's retained
                 * damped state, carried into the *next* iteration, and it must
                 * survive this rescale untouched.
                 */
                auto raw_occupation_matrix = damped_occupation_matrices[manifold_index][spin_index];
                for (auto& element : raw_occupation_matrix) {
                    element /= spin_degeneracy_factor;
                }
                auto result = compute_hubbard_energy_and_potential(
                    manifold_size, u_rydberg, raw_occupation_matrix);
                total_hubbard_energy += result.energy * spin_degeneracy_factor;

                /**
                 * Tr[V_U n] for this manifold and channel, accumulated from the
                 * same matrices the energy used so the two cannot drift apart.
                 * `compute_double_counting_correction()` subtracts it; see the
                 * note there for why it is mandatory.
                 *
                 * V is indexed (m, m') and pairs with n_{m'm} — the transpose
                 * convention `compute_hubbard_energy_and_potential` already
                 * documents for the D matrix itself.
                 */
                Real manifold_potential_energy = 0.0;
                for (int m = 0; m < manifold_size; ++m) {
                    for (int m_prime = 0; m_prime < manifold_size; ++m_prime) {
                        manifold_potential_energy +=
                            (result.d_matrix[static_cast<size_t>(m * manifold_size + m_prime)] *
                             raw_occupation_matrix[static_cast<size_t>(m_prime * manifold_size +
                                                                       m)])
                                .real();
                    }
                }
                total_hubbard_potential_energy +=
                    manifold_potential_energy * spin_degeneracy_factor;

                auto& combined_buffer = hubbard_d_matrix_per_spin_[spin_index];
                for (int m = 0; m < manifold_size; ++m) {
                    for (int m_prime = 0; m_prime < manifold_size; ++m_prime) {
                        int global_row = row_offset + m;
                        int global_column = row_offset + m_prime;
                        // Column-major, matching the applicator's BLAS "N","N" GEMM convention.
                        combined_buffer[static_cast<size_t>(global_row) +
                                        static_cast<size_t>(global_column) *
                                            total_hubbard_orbitals_] =
                            result.d_matrix[static_cast<size_t>(m * manifold_size + m_prime)];
                    }
                }
            }
        }
        /**
         * A spin-unpolarized calculation double-counts each manifold's occupation across
         * the (non-existent) second spin channel; number_of_spin_components() == 1 already
         * accounts for this since the loop above only ran once per manifold in that case.
         */
        energies_mutable().hubbard_energy = total_hubbard_energy;
        hubbard_potential_energy_ = total_hubbard_potential_energy;
    }

    /**
     * The base correction, plus DFT+U's own.
     *
     * Structurally identical to what the PAW iterator does for its one-center
     * term: a potential that entered the band energy has to be removed from it
     * here, and the corresponding energy is added back separately. The two
     * additions compose — a PAW + DFT+U run subtracts both.
     */
    Real compute_double_counting_correction() const override {
        return PlanewaveKohnShamSCFIterator::compute_double_counting_correction() -
               hubbard_potential_energy();
    }

    /**
     * Σ_σ Tr[V_U^σ n^σ] — the DFT+U potential's own contribution to the band
     * energy, which the double-counting correction has to remove.
     *
     * The Hubbard potential is applied inside the Hamiltonian, so every
     * eigenvalue already contains it. Adding E_U on top without removing
     * Tr[V_U n] counts the on-site interaction twice, and because
     * Tr[V_U n] = U(½Tr n − Tr n²) is *negative* for a partially filled shell,
     * the error lowers the total energy — which is how this presented:
     * switching U on made Q3's energy fall by ~1 Ry where E_U is positive
     * definite and QE's rises. The PAW one-center term has exactly this
     * counterpart and always did (`compute_double_counting_correction()` in the PAW iterator);
     * DFT+U simply never grew one.
     */
    [[nodiscard]] Real hubbard_potential_energy() const {
        return hubbard_potential_energy_;
    }

    /**
     * Copy converged wavefunctions from the current k-point to the next k-point's
     * slot in quantum_system_, providing a physically motivated initial guess for
     * the Davidson solver.
     *
     * When the two k-points have different numbers of plane waves (due to the
     * |k+G|² cutoff sphere varying with k), the coefficients are truncated or
     * zero-padded. This is valid because G-vectors are sorted by kinetic energy,
     * so the leading coefficients correspond to the most important basis functions.
     */
    void propagate_wavefunctions_to_next_kpoint(int current_kpoint_state_index,
                                                int total_kpoint_states) {
        int next_kpoint_state_index = current_kpoint_state_index + 1;
        if (next_kpoint_state_index >= total_kpoint_states) {
            return;
        }

        int number_of_kpoints = kpoint_grid().number_of_kpoints();
        int current_kpoint_index = current_kpoint_state_index % number_of_kpoints;
        int next_kpoint_index = next_kpoint_state_index % number_of_kpoints;

        // Only propagate within the same spin channel (don't cross spin boundaries)
        int current_spin_index = current_kpoint_state_index / number_of_kpoints;
        int next_spin_index = next_kpoint_state_index / number_of_kpoints;
        if (current_spin_index != next_spin_index) {
            return;
        }

        int source_number_of_plane_waves = number_of_plane_waves(current_kpoint_index);
        int target_number_of_plane_waves = number_of_plane_waves(next_kpoint_index);
        int coefficients_to_copy =
            std::min(source_number_of_plane_waves, target_number_of_plane_waves);

        for (int band_index = 0; band_index < number_of_bands_; ++band_index) {
            const Complex* source_band_data =
                quantum_system_.wavefunction_band_data(current_kpoint_state_index, band_index);
            Complex* target_band_data =
                quantum_system_.wavefunction_band_data(next_kpoint_state_index, band_index);

            // Copy the shared leading coefficients
            std::copy(source_band_data, source_band_data + coefficients_to_copy, target_band_data);

            // Zero-pad if the target has more plane waves than the source
            if (target_number_of_plane_waves > source_number_of_plane_waves) {
                std::fill(target_band_data + coefficients_to_copy,
                          target_band_data + target_number_of_plane_waves,
                          Complex(0.0, 0.0));
            }

            // Also propagate the eigenvalue as a starting estimate
            quantum_system_.set_eigenvalue(
                next_kpoint_state_index,
                band_index,
                quantum_system_.eigenvalue(current_kpoint_state_index, band_index));
        }
    }

    /**
     * Copy converged wavefunctions from one k-point state to another (backward direction).
     *
     * Unlike propagate_wavefunctions_to_next_kpoint which only moves forward,
     * this copies from any source to any target state index. Used to seed
     * k[0] from k[1]'s converged wavefunctions after the forward pass.
     */
    void propagate_wavefunctions_backward(int source_kpoint_state_index,
                                          int target_kpoint_state_index) {
        int number_of_kpoints = kpoint_grid().number_of_kpoints();
        int source_kpoint_index = source_kpoint_state_index % number_of_kpoints;
        int target_kpoint_index = target_kpoint_state_index % number_of_kpoints;

        int source_number_of_plane_waves = number_of_plane_waves(source_kpoint_index);
        int target_number_of_plane_waves = number_of_plane_waves(target_kpoint_index);
        int coefficients_to_copy =
            std::min(source_number_of_plane_waves, target_number_of_plane_waves);

        for (int band_index = 0; band_index < number_of_bands_; ++band_index) {
            const Complex* source_band_data =
                quantum_system_.wavefunction_band_data(source_kpoint_state_index, band_index);
            Complex* target_band_data =
                quantum_system_.wavefunction_band_data(target_kpoint_state_index, band_index);

            std::copy(source_band_data, source_band_data + coefficients_to_copy, target_band_data);

            if (target_number_of_plane_waves > source_number_of_plane_waves) {
                std::fill(target_band_data + coefficients_to_copy,
                          target_band_data + target_number_of_plane_waves,
                          Complex(0.0, 0.0));
            }

            quantum_system_.set_eigenvalue(
                target_kpoint_state_index,
                band_index,
                quantum_system_.eigenvalue(source_kpoint_state_index, band_index));
        }
    }

    /**
     * Check whether a k-point index falls on a band-path segment boundary.
     *
     * Segment boundaries are the high-symmetry points where the path direction
     * changes (e.g., G, X, W, K in a G→X→W→K→G→L path). At these points,
     * the eigenstates may reorder between segments, so forward propagation of
     * wavefunctions across the boundary would bias the Davidson solver toward
     * eigenstates from the wrong symmetry direction.
     *
     * Returns false for SCF calculations (no band path defined) and for the
     * endpoint of the last segment (which has no following segment to pollute).
     */
    [[nodiscard]] bool is_at_band_path_segment_boundary(int kpoint_index) const {
        if (band_path_.empty() || band_path_points_per_segment_ <= 0) {
            return false;
        }
        int number_of_segments = static_cast<int>(band_path_.size()) - 1;
        if (number_of_segments <= 0) {
            return false;
        }
        /**
         * The last k-point (endpoint of the final segment) is not a boundary
         * because there is no following segment.
         */
        int last_kpoint_index = number_of_segments * band_path_points_per_segment_;
        if (kpoint_index >= last_kpoint_index) {
            return false;
        }
        // Boundaries are at multiples of points_per_segment: 0, PPS, 2*PPS, ...
        return (kpoint_index % band_path_points_per_segment_) == 0;
    }

    /**
     * Re-solve edge k-points that lacked proper propagation seeds.
     *
     * After the forward pass, two types of k-points may have suboptimal
     * eigenvalues for high conduction bands:
     *
     *  (a) k[0] — the very first k-point, solved with only atomic-orbital
     *      seeds because no predecessor exists.
     *
     *  (b) The first k-point of each new segment after a direction change —
     *      these were solved without propagation seeds because we intentionally
     *      skipped propagation across the segment boundary.
     *
     * For each such edge, we propagate backwards from a correctly converged
     * neighbor (2 steps into the segment) and re-solve. The neighbor at
     * offset +2 was itself seeded by propagation within the segment (from
     * offset +1) and so has correct eigenstates for this path direction.
     */
    void fix_edge_kpoints_with_backward_propagation(int number_of_kpoints,
                                                    int number_of_kpoint_states) {
        if (number_of_kpoints <= 2) {
            return;
        }

        /**
         * Collect all k-point indices that need re-solving.
         * Always include k[0]. If a band path is defined, also include
         * the first k-point of each segment after the first.
         */
        std::vector<int> edge_kpoint_indices;
        edge_kpoint_indices.push_back(0);

        if (!band_path_.empty() && band_path_points_per_segment_ > 0) {
            int number_of_segments = static_cast<int>(band_path_.size()) - 1;
            for (int segment_index = 1; segment_index < number_of_segments; ++segment_index) {
                int boundary_kpoint_index = segment_index * band_path_points_per_segment_;
                if (boundary_kpoint_index < number_of_kpoints) {
                    edge_kpoint_indices.push_back(boundary_kpoint_index);
                }
            }
        }

        /**
         * Number of k-points to re-solve at each edge (the boundary itself
         * plus this many into the segment). More points improves accuracy for
         * high conduction bands at the cost of extra diagonalizations.
         */
        const int backward_sweep_depth = 5;

        for (int spin_index = 0; spin_index < number_of_spin_components(); ++spin_index) {
            for (int edge_kpoint_index : edge_kpoint_indices) {
                /**
                 * Find the seed: the first k-point beyond the sweep region that
                 * was correctly solved during the forward pass.
                 */
                int sweep_end =
                    std::min(edge_kpoint_index + backward_sweep_depth, number_of_kpoints - 1);
                int seed_kpoint_index = std::min(sweep_end + 1, number_of_kpoints - 1);

                /**
                 * Sweep backwards from the seed toward the edge point.
                 * Each re-solved k-point then serves as the seed for the next.
                 */
                for (int target_kpoint_index = sweep_end; target_kpoint_index >= edge_kpoint_index;
                     --target_kpoint_index) {

                    int source_state_index = spin_index * number_of_kpoints + seed_kpoint_index;
                    int target_state_index = spin_index * number_of_kpoints + target_kpoint_index;

                    propagate_wavefunctions_backward(source_state_index, target_state_index);

                    diagonalizer_->solve(target_state_index,
                                         target_kpoint_index,
                                         number_of_plane_waves(target_kpoint_index),
                                         number_of_bands_,
                                         g_vector_indices(target_kpoint_index),
                                         kinetic_energy_squared()[target_kpoint_index],
                                         real_space_potential_[spin_index],
                                         crystal_structure(),
                                         kpoint_grid(),
                                         planewave_basis_,
                                         *quantum_system_snapshot_,
                                         quantum_system_);
                    report_unconverged_solve(target_state_index);

                    // The just-solved point becomes the seed for the next iteration
                    seed_kpoint_index = target_kpoint_index;
                }
            }
        }
    }

    /**
     * Warn when a non-self-consistent solve kept bands short of the tolerance.
     *
     * A self-consistent run finishes them on its next iteration, so only the
     * NSCF pass reports: its eigenvectors are the answer, and a model built on
     * them (a band structure, a Wannier capture) inherits the error silently.
     */
    void report_unconverged_solve(int kpoint_index_state) const {
        if (!report_unconverged_bands_) {
            return;
        }
        int unconverged = diagonalizer_->number_of_unconverged_bands();
        if (unconverged > 0) {
            log::warn("NSCF: k-point state {} kept {} of {} bands unconverged after the "
                      "Davidson iteration cap; its eigenvalues and eigenvectors are not "
                      "converged to the requested tolerance",
                      kpoint_index_state,
                      unconverged,
                      number_of_bands_);
        }
    }

    // ─── Ionic / external world ──────────────────────────────────────
    [[nodiscard]] const std::vector<Complex>& local_ionic_potential_g() const override {
        return quantum_system_snapshot_->pseudopotential_basis().total_local_potential_g();
    }
    [[nodiscard]] const std::vector<Complex>& core_charge_density_g() const override {
        return quantum_system_snapshot_->pseudopotential_basis().core_charge_density_reciprocal();
    }

    // ─── Effective potential ─────────────────────────────────────────
    void compute_effective_potential_from_density() override {
        if (planewave_energies().ewald_energy < math::tolerances::numerical_zero_threshold) {
            energies_.ewald_energy = compute_ewald(crystal_structure(), planewave_basis_);
        }
        /**
         * Dispersion depends only on the (fixed) geometry, so compute it once
         * per iterator lifetime. An explicit flag — not a zero-energy sentinel —
         * marks it done: a legitimately zero dispersion energy must not trigger
         * recomputation, and a stale nonzero value must not survive a rebuild.
         * The factory resolves the configured scheme ("none" contributes an
         * exact zero), so no per-scheme branching happens here.
         */
        if (!dispersion_energy_computed_) {
            energies_.van_der_waals_energy =
                create_dispersion_calculator(config().dispersion_correction, crystal_structure())
                    ->compute_energy();
            dispersion_energy_computed_ = true;
        }

        EffectivePotentialResult effective_potential = effective_potential_calculator().compute(
            charge_density_real_, number_of_spin_components(), local_ionic_potential_g());

        charge_density_g_ = std::move(effective_potential.charge_density_g);
        real_space_potential_ = std::move(effective_potential.real_space_potential);
        total_potential_g_ = std::move(effective_potential.total_potential_g);
        total_potential_g_by_spin_ = std::move(effective_potential.total_potential_g_by_spin);
        exchange_correlation_potential_g_ =
            std::move(effective_potential.exchange_correlation_potential_g);
        energies_.hartree_energy = effective_potential.hartree_energy;
        energies_.exchange_correlation_energy = effective_potential.exchange_correlation_energy;
        energies_.exchange_correlation_potential_energy =
            effective_potential.exchange_correlation_potential_energy;

        // Cache the real-space ionic local potential on first call (constant during SCF).
        if (cached_local_ionic_potential_real_.empty()) {
            planewave_basis_.rho_g_to_r(local_ionic_potential_g(),
                                        cached_local_ionic_potential_real_);
        }
    }

    /**
     * The effective potential calculator, created on first use and kept, since it
     * caches the real-space core charge density.
     */
    NormConservingPlanewaveEffectivePotentialCalculator& effective_potential_calculator() {
        if (!effective_potential_calculator_) {
            effective_potential_calculator_ =
                std::make_unique<NormConservingPlanewaveEffectivePotentialCalculator>(
                    planewave_basis_,
                    crystal_structure(),
                    *exchange_correlation_,
                    *execution_policy_,
                    quantum_system_snapshot_->pseudopotential_basis()
                        .core_charge_density_reciprocal());
        }
        return *effective_potential_calculator_;
    }

    // ─── Pseudopotential setup helpers ───────────────────────────────
    void create_norm_conserving_quantum_system_snapshot() {
        quantum_system_snapshot_ =
            NormConservingPlanewaveKohnShamQuantumSystemFactory::create_shared(
                crystal_structure(), planewave_basis_, pseudopotential_set());
    }

    /**
     * Estimate this rank's footprint from the dimensions setup has settled.
     *
     * Called once, from the same place the pseudopotential summary is printed,
     * because the projector count is the last of its inputs to exist.
     */
    void report_memory_estimate(int number_of_atomic_wavefunctions) const {
        PlanewaveMemoryEstimateInputs inputs;
        inputs.number_of_bands = number_of_bands_;
        const auto& plane_waves_per_kpoint = planewave_basis_.kpoint_number_of_plane_waves();
        inputs.max_plane_waves_per_kpoint =
            plane_waves_per_kpoint.empty()
                ? 0
                : *std::max_element(plane_waves_per_kpoint.begin(), plane_waves_per_kpoint.end());
        inputs.number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        inputs.number_of_projectors =
            quantum_system_snapshot_->pseudopotential_basis().total_number_of_projectors();
        inputs.number_of_atomic_orbitals = number_of_atomic_wavefunctions;
        inputs.number_of_atoms = crystal_structure().number_of_atoms();
        inputs.number_of_spin_components = number_of_spin_components();
        inputs.local_number_of_kpoint_states = number_of_kpoint_states();
        inputs.mixing_history_depth = mixing_history_size_;
        inputs.fft_rho_grid_points = planewave_basis_.fft_rho().total_grid_points();
        inputs.fft_wave_grid_points = planewave_basis_.fft_wave().total_grid_points();
        inputs.overlap_is_identity = !quantum_system_snapshot_->has_ultrasoft();
        inputs.diagonalizer_is_davidson = diagonalization_method_ == "davidson";
        inputs.davidson_basis_expansion_factor = config().davidson_basis_expansion_factor;
        PlanewaveKohnShamSCFReporter::report_memory_estimate(
            PlanewaveMemoryEstimate::compute(inputs));
    }

    void log_pseudopotential_summary(int number_of_atomic_wavefunctions) const {
        log::info("     number of KB projectors   = {}",
                  quantum_system_snapshot_->pseudopotential_basis().total_number_of_projectors());
        log::info("     Starting wavefunctions    = {} atomic wfcs",
                  number_of_atomic_wavefunctions);
    }

    /**
     * Per-species initial magnetization, normalized to the species count.
     *
     * An EMPTY config vector means the input gave no starting_magnetization at
     * all; every species then defaults to constants::default_starting_magnetization
     * so a spin-polarized run can break spin symmetry. Explicit values — including
     * explicit zeros — are honored as given.
     */
    std::vector<Real> effective_species_magnetizations() const {
        std::vector<Real> species_magnetization(crystal_structure().number_of_species(), 0.0);
        if (starting_magnetization_.empty()) {
            species_magnetization.assign(species_magnetization.size(),
                                         constants::default_starting_magnetization);
            return species_magnetization;
        }
        for (size_t species_index = 0; species_index < starting_magnetization_.size() &&
                                       species_index < species_magnetization.size();
             ++species_index) {
            species_magnetization[species_index] = starting_magnetization_[species_index];
        }
        return species_magnetization;
    }

    void initialize_charge_density_from_atomic_superposition() {
        int number_of_g_vectors = planewave_basis_.number_of_g_vectors();
        Real cell_volume = crystal_structure().cell_volume();
        bool spin_polarized = number_of_spin_components() == 2;

        std::vector<Real> species_magnetization;
        std::vector<Complex> magnetization_density_g;
        if (spin_polarized) {
            species_magnetization = effective_species_magnetizations();
            magnetization_density_g.assign(number_of_g_vectors, Complex(0.0, 0.0));
            if (!atomic_starting_magnetization_.empty() &&
                static_cast<int>(atomic_starting_magnetization_.size()) !=
                    crystal_structure().number_of_atoms())
                throw std::runtime_error("atomic_starting_magnetization lists " +
                                         std::to_string(atomic_starting_magnetization_.size()) +
                                         " values but the cell has " +
                                         std::to_string(crystal_structure().number_of_atoms()) +
                                         " atoms");
        }

        charge_density_g_.resize(number_of_g_vectors, Complex(0.0, 0.0));
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; ++g_vector_index) {
            for (int atom_index = 0; atom_index < crystal_structure().number_of_atoms();
                 ++atom_index) {
                int species_type = crystal_structure().species_type_index(atom_index);
                Vector3D position =
                    crystal_structure().convert_from_crystal_to_cartesian(atom_index);
                const auto& g_vector = planewave_basis_.g_vectors()[g_vector_index];
                Real phase = constants::two_pi * g_vector.g_cart.dot(position);
                Complex structure_factor = compute_phase_factor(phase);
                Complex atomic_contribution =
                    quantum_system_snapshot_->pseudopotential_basis().atomic_charge_density_g(
                        species_type)[g_vector_index] *
                    structure_factor / cell_volume;
                charge_density_g_[g_vector_index] += atomic_contribution;
                if (spin_polarized) {
                    /**
                     * Magnetization density m(G): each atom contributes its starting
                     * magnetization -- its own when given per atom, else its species'
                     * -- times its atomic charge density.
                     */
                    Real atom_magnetization = atomic_starting_magnetization_.empty()
                                                  ? species_magnetization[species_type]
                                                  : atomic_starting_magnetization_[atom_index];
                    magnetization_density_g[g_vector_index] +=
                        atom_magnetization * atomic_contribution;
                }
            }
        }

        Real total_starting_charge = charge_density_g_[0].real() * cell_volume;
        if (std::abs(total_starting_charge) > math::tolerances::zero_vector_magnitude_threshold) {
            Real scaling_factor = number_of_electrons_ / total_starting_charge;
            for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; ++g_vector_index) {
                charge_density_g_[g_vector_index] *= scaling_factor;
                if (spin_polarized)
                    magnetization_density_g[g_vector_index] *= scaling_factor;
            }
        }

        int local_grid_points = planewave_basis_.fft_rho().local_grid_points();
        charge_density_real_.resize(number_of_spin_components());

        if (!spin_polarized) {
            planewave_basis_.rho_g_to_r(charge_density_g_, charge_density_real_[0]);
            on_charge_density_initialized();
            return;
        }

        // ρ↑ = (ρ + m)/2, ρ↓ = (ρ − m)/2 pointwise in real space.
        std::vector<Real> total_charge_density_real;
        std::vector<Real> magnetization_density_real;
        planewave_basis_.rho_g_to_r(charge_density_g_, total_charge_density_real);
        planewave_basis_.rho_g_to_r(magnetization_density_g, magnetization_density_real);

        charge_density_real_[0].resize(local_grid_points);
        charge_density_real_[1].resize(local_grid_points);
        for (int grid_index = 0; grid_index < local_grid_points; ++grid_index) {
            charge_density_real_[0][grid_index] = 0.5 * (total_charge_density_real[grid_index] +
                                                         magnetization_density_real[grid_index]);
            charge_density_real_[1][grid_index] = 0.5 * (total_charge_density_real[grid_index] -
                                                         magnetization_density_real[grid_index]);
        }
        on_charge_density_initialized();
    }
};

} // namespace q3
