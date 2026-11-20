#pragma once

/**
 * PlanewaveDavidsonDiagonalizer — planewave Davidson iterative diagonalizer.
 *
 * Implements the block Davidson algorithm for finding the lowest eigenvalues
 * and eigenvectors of the Kohn-Sham Hamiltonian. The subspace is iteratively
 * expanded with preconditioned residuals, and collapsed back to the target
 * number of bands when it reaches the maximum allowed size (4×number_of_bands).
 *
 * Implements the iterative method of E. R. Davidson, J. Comput. Phys. 17, 87
 * (1975), in block form: the subspace is expanded for all bands together.
 *
 * The Hamiltonian application H|ψ⟩ is delegated to a
 * PlanewaveHamiltonianApplicator (serial or GPU).
 *
 * Subspace linear algebra (orthogonalization, subspace matrix construction,
 * diagonalization, Ritz vectors, collapse) is delegated to a
 * PlanewaveSubspaceManager (serial or GPU).
 *
 * This class retains only the iteration loop control, basis seeding,
 * and preconditioner application.
 */
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/execution/serial.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/profiling/counters.hpp"
#include "layer_0_core/profiling/stage_timer.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_4_computation/diagonalizer/planewave/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/factory.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/projector_set_provider.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/serial.hpp"

namespace q3 {

/**
 * Abstract workspace for Davidson diagonalizer.
 *
 * Holds state resident in the appropriate memory space (CPU or GPU)
 * to avoid memory copies during iterative steps.
 */
class PlanewaveDavidsonWorkspace {
  public:
    virtual ~PlanewaveDavidsonWorkspace() = default;

    /**
     * Size the workspace for this k-point.
     *
     * overlap_is_identity lets a norm-conserving run skip the S arrays
     * entirely; the accessors then return the basis and Ritz buffers, which
     * hold exactly the same values.
     */
    virtual void resize(int maximum_basis_size,
                        int number_of_plane_waves,
                        int number_of_bands,
                        bool overlap_is_identity) = 0;

    virtual Complex* basis_vectors() = 0;
    virtual const Complex* basis_vectors() const = 0;

    virtual Complex* hamiltonian_basis_vectors() = 0;
    virtual const Complex* hamiltonian_basis_vectors() const = 0;

    virtual Complex* overlap_basis_vectors() = 0;
    virtual const Complex* overlap_basis_vectors() const = 0;

    Real* local_eigenvalues() {
        return local_eigenvalues_host_.data();
    }
    const Real* local_eigenvalues() const {
        return local_eigenvalues_host_.data();
    }

    virtual Complex* subspace_hamiltonian() = 0;
    virtual const Complex* subspace_hamiltonian() const = 0;

    virtual Complex* subspace_overlap() = 0;
    virtual const Complex* subspace_overlap() const = 0;

    /**
     * One band-sized buffer the solver borrows for a rotation that cannot be
     * done in place -- the subspace collapse and the final extraction. It
     * replaced the Ritz vectors, H|Ritz>, S|Ritz> and residual arrays, which
     * the correction-vector build made unnecessary.
     */
    virtual Complex* band_scratch() = 0;
    virtual const Complex* band_scratch() const = 0;

    virtual void sync_to_host() = 0;

    /**
     * Publish the host eigenvalues to wherever the workspace's other arrays live. Nothing by
     * default: a host workspace has them there already.
     */
    virtual void sync_to_device() {}

  protected:
    std::vector<Real> local_eigenvalues_host_;
};

class SerialDavidsonWorkspace : public PlanewaveDavidsonWorkspace {
  public:
    void resize(int maximum_basis_size,
                int number_of_plane_waves,
                int number_of_bands,
                bool overlap_is_identity) override {
        std::size_t basis_elements = static_cast<std::size_t>(maximum_basis_size) *
                                     static_cast<std::size_t>(number_of_plane_waves);
        std::size_t band_elements = static_cast<std::size_t>(number_of_bands) *
                                    static_cast<std::size_t>(number_of_plane_waves);
        std::size_t current_size =
            basis_vectors_.empty()
                ? 0
                : basis_vectors_.size() / static_cast<std::size_t>(number_of_plane_waves);
        if (current_size >= static_cast<std::size_t>(maximum_basis_size) &&
            overlap_is_aliased_ == overlap_is_identity) {
            return;
        }

        /**
         * For norm-conserving runs the overlap operator is the identity, so
         * S|V> and S|psi> are bit-for-bit the basis and the Ritz vectors. The
         * two arrays are not allocated at all; the accessors hand back the
         * originals, and the subspace manager skips the writes that would
         * otherwise be applied to the same memory twice. That is five arrays
         * of number_of_bands x number_of_plane_waves not held.
         */
        overlap_is_aliased_ = overlap_is_identity;

        basis_vectors_.assign(basis_elements, Complex(0.0, 0.0));
        hamiltonian_basis_vectors_.assign(basis_elements, Complex(0.0, 0.0));
        subspace_hamiltonian_.assign(static_cast<std::size_t>(maximum_basis_size) *
                                         static_cast<std::size_t>(maximum_basis_size),
                                     Complex(0.0, 0.0));
        subspace_overlap_.assign(static_cast<std::size_t>(maximum_basis_size) *
                                     static_cast<std::size_t>(maximum_basis_size),
                                 Complex(0.0, 0.0));
        band_scratch_.assign(band_elements, Complex(0.0, 0.0));
        if (overlap_is_aliased_) {
            overlap_basis_vectors_.clear();
            overlap_basis_vectors_.shrink_to_fit();
        } else {
            overlap_basis_vectors_.assign(basis_elements, Complex(0.0, 0.0));
        }

        local_eigenvalues_host_.assign(maximum_basis_size, 0.0);
    }

    Complex* basis_vectors() override {
        return basis_vectors_.data();
    }
    const Complex* basis_vectors() const override {
        return basis_vectors_.data();
    }

    Complex* hamiltonian_basis_vectors() override {
        return hamiltonian_basis_vectors_.data();
    }
    const Complex* hamiltonian_basis_vectors() const override {
        return hamiltonian_basis_vectors_.data();
    }

    Complex* overlap_basis_vectors() override {
        return overlap_is_aliased_ ? basis_vectors_.data() : overlap_basis_vectors_.data();
    }
    const Complex* overlap_basis_vectors() const override {
        return overlap_is_aliased_ ? basis_vectors_.data() : overlap_basis_vectors_.data();
    }

    Complex* subspace_hamiltonian() override {
        return subspace_hamiltonian_.data();
    }
    const Complex* subspace_hamiltonian() const override {
        return subspace_hamiltonian_.data();
    }

    Complex* subspace_overlap() override {
        return subspace_overlap_.data();
    }
    const Complex* subspace_overlap() const override {
        return subspace_overlap_.data();
    }

    Complex* band_scratch() override {
        return band_scratch_.data();
    }
    const Complex* band_scratch() const override {
        return band_scratch_.data();
    }

    void sync_to_host() override {}

  private:
    std::vector<Complex> basis_vectors_;
    std::vector<Complex> hamiltonian_basis_vectors_;
    std::vector<Complex> overlap_basis_vectors_;
    std::vector<Complex> subspace_hamiltonian_;
    std::vector<Complex> subspace_overlap_;
    std::vector<Complex> band_scratch_;
    /// True when S is the identity and the overlap accessors alias the originals.
    bool overlap_is_aliased_ = false;
};

/**
 * Davidson iterative diagonalizer.
 * Finds only the lowest number_of_bands eigenvalues in O(number_of_plane_waves * number_of_bands)
 * per iteration, vs O(number_of_plane_waves³) for direct.
 */
class PlanewaveDavidsonDiagonalizer : public PlanewaveDiagonalizer {
  public:
    // Fallback serial execution policy — used as default when no policy is injected.
    inline static const execution::SerialExecutionPolicy default_serial_execution_policy{};

    /**
     * Construct the Davidson diagonalizer with an execution policy for band-group
     * vector operations (dot_product, norm). Defaults to a serial (BLAS-only) policy
     * if none is provided, preserving compatibility with existing call sites.
     *
     * Optionally accepts a PlanewaveHamiltonianApplicator and/or
     * PlanewaveSubspaceManager for delegation. If not provided, internal serial
     * implementations are created on first solve() call.
     */
    explicit PlanewaveDavidsonDiagonalizer(
        const execution::ExecutionPolicy& execution_policy = default_serial_execution_policy,
        PlanewaveHamiltonianApplicator* hamiltonian_applicator = nullptr,
        PlanewaveSubspaceManager* subspace_manager = nullptr)
        : execution_policy_{execution_policy}
        , injected_hamiltonian_applicator_{hamiltonian_applicator}
        , injected_subspace_manager_{subspace_manager} {
        // Create owned subspace manager if none injected.
        if (injected_subspace_manager_ == nullptr) {
            owned_subspace_manager_ =
                std::make_unique<SerialPlanewaveSubspaceManager>(execution_policy);
        }
    }

    /**
     * Set pre-computed KB projectors for a specific k-point.
     * No-op if the applicator has not been created yet (before first solve()).
     */
    void set_basis_expansion_factor(int factor) override {
        if (factor >= 2) {
            davidson_basis_expansion_factor = factor;
        }
    }

    /**
     * Drop the working basis, its Hamiltonian image and the subspace: the
     * largest thing the solver owns, and dead once the SCF has converged.
     */
    void set_maximum_iterations(int maximum_iterations) override {
        max_iterations_ = std::max(1, maximum_iterations);
    }

    [[nodiscard]] int number_of_unconverged_bands() const override {
        return number_of_unconverged_bands_;
    }

    void release_workspace() override {
        workspace_.reset();
    }

    void set_projector_set_store(KleinmanBylanderProjectorSetStore* projector_set_store) override {
        projector_set_provider_.set_store(projector_set_store);
        if (has_hamiltonian_applicator()) {
            get_hamiltonian_applicator().set_projector_set_store(projector_set_store);
        }
    }

    /**
     * Set pre-computed DFT+U (Hubbard) atomic-orbital projectors for a specific k-point.
     * No-op if the applicator has not been created yet (before first solve()).
     */
    void set_cached_hubbard_projectors(int kpoint_index,
                                       const q3::Matrix<Complex>* projectors) override {
        if (has_hamiltonian_applicator()) {
            get_hamiltonian_applicator().set_cached_hubbard_projectors(kpoint_index, projectors);
        }
        pending_cached_hubbard_projectors_[kpoint_index] = projectors;
        hubbard_term_was_set_ = hubbard_term_was_set_ || projectors != nullptr;
    }

    /**
     * Set the dense block-diagonal Hubbard D-matrix (refreshed once per SCF iteration).
     * No-op if the applicator has not been created yet (before first solve()).
     */
    void set_hubbard_d_matrix(const std::vector<Complex>* dense_d_matrix_column_major) override {
        if (has_hamiltonian_applicator()) {
            get_hamiltonian_applicator().set_hubbard_d_matrix(dense_d_matrix_column_major);
        }
        pending_hubbard_d_matrix_ = dense_d_matrix_column_major;
        hubbard_term_was_set_ = hubbard_term_was_set_ || dense_d_matrix_column_major != nullptr;
    }

    /**
     * Invalidate the cached smooth grid potential.
     * No-op if the applicator has not been created yet (nothing cached).
     */
    void invalidate_smooth_potential_cache() override {
        if (has_hamiltonian_applicator()) {
            get_hamiltonian_applicator().invalidate_smooth_potential_cache();
        }
    }

    void solve(int kpoint_index_state,
               int kpoint_index,
               int number_of_plane_waves,
               int number_of_bands,
               const std::vector<int>& g_vector_indices,
               const std::vector<Real>& kinetic_energy_squared,
               const std::vector<Real>& real_space_potential,
               const Crystal& crystal_structure,
               const KPointsGrid& kpoint_grid,
               const PlanewaveBasis& planewave_basis,
               const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
               PlanewaveKohnShamQuantumSystem& quantum_system) override {
        std::optional<StageTimer> prepare_timer(std::in_place, "davidson.prepare");
        // The starting rotation is over once a solve begins; its batch buffers go back.
        release_starting_rotation();
        // Ensure the applicator exists (lazy construction if not injected).
        ensure_hamiltonian_applicator_initialized(
            crystal_structure, planewave_basis, kpoint_grid, pseudopotential_quantum_system);

        auto& applicator = get_hamiltonian_applicator();
        auto& subspace = get_subspace_manager();

        /**
         * Which spin channel this call belongs to.
         *
         * The caller flattens (spin, k-point) into `kpoint_index_state`, so the
         * channel is the quotient. The applicator needs it because the nonlocal
         * D is per channel for ultrasoft and PAW; `real_space_potential` below
         * is already this channel's, and the two have to agree.
         */
        int spin_channel = kpoint_grid.number_of_kpoints() > 0
                               ? kpoint_index_state / kpoint_grid.number_of_kpoints()
                               : 0;
        applicator.set_spin_channel(spin_channel);

        // Update the smooth potential if it has been invalidated.
        if (!applicator.is_smooth_potential_cache_valid()) {
            StageTimer potential_timer("davidson.prepare.potential");
            applicator.update_potential(real_space_potential);
        }

        // Configure the applicator for this k-point.
        {
            StageTimer kpoint_timer("davidson.prepare.kpoint");
            applicator.prepare_kpoint(
                kpoint_index, number_of_plane_waves, g_vector_indices, kinetic_energy_squared);
        }

        // Cache gamma-only flag for the duration of this solve() call.
        bool is_gamma_only = applicator.is_gamma_only_current();
        /**
         * Under a reciprocal-space distribution one rank holds G = 0, and the
         * manager's gamma corrections must know whether this is that rank.
         */
        subspace.set_gamma_origin_is_local(applicator.gamma_origin_is_local_current());

        // The overlap operator selects the standard (NC, S=I) or generalized (US) eigenproblem.
        const KleinmanBylanderProjectorSet& kleinman_bylander_projector_set =
            projector_set_provider_.projector_set(
                kpoint_index,
                crystal_structure,
                kpoint_grid,
                planewave_basis,
                pseudopotential_quantum_system.pseudopotential_basis(),
                g_vector_indices,
                number_of_plane_waves);

        std::optional<StageTimer> overlap_timer(std::in_place, "davidson.prepare.overlap_operator");
        auto overlap_operator = create_overlap_operator(crystal_structure,
                                                        pseudopotential_quantum_system,
                                                        kleinman_bylander_projector_set,
                                                        execution_policy_);
        overlap_timer.reset();
        bool is_identity_overlap = overlap_operator->is_identity();

        int global_number_of_plane_waves =
            execution_policy_.global_number_of_elements(number_of_plane_waves);
        int maximum_basis_size =
            std::max(std::min(davidson_basis_expansion_factor * number_of_bands,
                              global_number_of_plane_waves),
                     number_of_bands);

        // Ensure workspace is initialized and sized.
        if (workspace_ == nullptr) {
            workspace_ = create_workspace();
        }
        workspace_->resize(
            maximum_basis_size, number_of_plane_waves, number_of_bands, is_identity_overlap);
        log_davidson_memory("after workspace resize", maximum_basis_size, number_of_plane_waves);

        (void)stage_time_point();
        prepare_timer.reset();
        // Phase 1: Seed basis from previous wavefunctions
        auto timer_seed_start = stage_time_point();
        int basis_size = seed_basis_from_wavefunctions(number_of_plane_waves,
                                                       number_of_bands,
                                                       quantum_system,
                                                       kpoint_index_state,
                                                       is_gamma_only,
                                                       overlap_operator.get(),
                                                       applicator,
                                                       subspace,
                                                       workspace_->basis_vectors(),
                                                       workspace_->hamiltonian_basis_vectors(),
                                                       workspace_->overlap_basis_vectors());
        auto timer_seed_end = stage_time_point();
        ProfilingCounters::instance().davidson_timing.seed_time_ms +=
            std::chrono::duration<double, std::milli>(timer_seed_end - timer_seed_start).count();

        // Phase 2: Fill remaining basis slots with kinetic-energy-sorted plane waves
        if (basis_size < number_of_bands) {
            basis_size = seed_basis_from_kinetic_energy(basis_size,
                                                        number_of_plane_waves,
                                                        number_of_bands,
                                                        kinetic_energy_squared,
                                                        is_gamma_only,
                                                        overlap_operator.get(),
                                                        applicator,
                                                        subspace,
                                                        workspace_->basis_vectors(),
                                                        workspace_->hamiltonian_basis_vectors(),
                                                        workspace_->overlap_basis_vectors());
        }

        execution_policy_.broadcast_across_bands(&basis_size, 1);
        synchronize_basis_across_bands(0,
                                       basis_size,
                                       number_of_plane_waves,
                                       workspace_->basis_vectors(),
                                       workspace_->hamiltonian_basis_vectors(),
                                       workspace_->overlap_basis_vectors());

        std::optional<StageTimer> initial_subspace_timer(std::in_place,
                                                         "davidson.initial_subspace");
        // Build initial subspace matrices for the seeded basis.
        subspace.update_subspace_matrices(0,
                                          basis_size,
                                          basis_size,
                                          maximum_basis_size,
                                          number_of_plane_waves,
                                          is_gamma_only,
                                          workspace_->basis_vectors(),
                                          workspace_->hamiltonian_basis_vectors(),
                                          workspace_->overlap_basis_vectors(),
                                          workspace_->subspace_hamiltonian(),
                                          workspace_->subspace_overlap());

        (void)stage_time_point();
        initial_subspace_timer.reset();
        // Hybrid lrot approach: use stored eigenvalues when available
        std::vector<Real> previous_eigenvalues(number_of_bands, 0.0);
        bool has_stored_eigenvalues = false;
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            Real stored = quantum_system.eigenvalue(kpoint_index_state, band_index);
            if (stored != 0.0) {
                has_stored_eigenvalues = true;
                break;
            }
        }
        if (has_stored_eigenvalues) {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                previous_eigenvalues[band_index] =
                    quantum_system.eigenvalue(kpoint_index_state, band_index);
            }
        } else {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                previous_eigenvalues[band_index] =
                    workspace_->subspace_hamiltonian()[band_index * maximum_basis_size + band_index]
                        .real();
            }
        }

        // Phase 3: Davidson iteration loop
        number_of_unconverged_bands_ = number_of_bands;
        for (int iteration = 0; iteration < max_iterations_; iteration++) {
            ProfilingCounters::instance().davidson_iteration_count++;
            auto timer_diag_start = stage_time_point();
            std::vector<Complex> eigenvector_matrix(maximum_basis_size * maximum_basis_size);
            int diagonalization_info =
                subspace.diagonalize_subspace(basis_size,
                                              maximum_basis_size,
                                              is_gamma_only,
                                              workspace_->subspace_hamiltonian(),
                                              workspace_->subspace_overlap(),
                                              workspace_->local_eigenvalues(),
                                              eigenvector_matrix.data());
            execution_policy_.broadcast_across_bands(&diagonalization_info, 1);
            synchronize_eigenpairs_across_bands(basis_size,
                                                maximum_basis_size,
                                                workspace_->local_eigenvalues(),
                                                eigenvector_matrix.data());
            auto timer_diag_end = stage_time_point();
            ProfilingCounters::instance().davidson_timing.diagonalize_time_ms +=
                std::chrono::duration<double, std::milli>(timer_diag_end - timer_diag_start)
                    .count();

            // Recovery: if zhegv failed, force collapse and retry
            if (diagonalization_info != 0 && basis_size > number_of_bands) {
                log::warn("    Davidson: zhegv failed (info={}), forcing subspace collapse",
                          diagonalization_info);
                auto timer_collapse_start = stage_time_point();
                basis_size = subspace.collapse_subspace(basis_size,
                                                        number_of_plane_waves,
                                                        number_of_bands,
                                                        maximum_basis_size,
                                                        is_gamma_only,
                                                        is_identity_overlap,
                                                        overlap_operator.get(),
                                                        /*subspace_hamiltonian=*/nullptr,
                                                        /*subspace_overlap=*/nullptr,
                                                        workspace_->basis_vectors(),
                                                        workspace_->hamiltonian_basis_vectors(),
                                                        workspace_->overlap_basis_vectors(),
                                                        workspace_->band_scratch());
                /**
                 * Only the size needs agreeing here. The collapse itself now
                 * forms each band range on one rank and allgathers, so the
                 * collapsed basis is already identical everywhere; republishing
                 * it would move three full arrays for nothing.
                 */
                execution_policy_.broadcast_across_bands(&basis_size, 1);
                std::memset(workspace_->subspace_hamiltonian(),
                            0,
                            maximum_basis_size * maximum_basis_size * sizeof(Complex));
                std::memset(workspace_->subspace_overlap(),
                            0,
                            maximum_basis_size * maximum_basis_size * sizeof(Complex));
                subspace.update_subspace_matrices(0,
                                                  basis_size,
                                                  basis_size,
                                                  maximum_basis_size,
                                                  number_of_plane_waves,
                                                  is_gamma_only,
                                                  workspace_->basis_vectors(),
                                                  workspace_->hamiltonian_basis_vectors(),
                                                  workspace_->overlap_basis_vectors(),
                                                  workspace_->subspace_hamiltonian(),
                                                  workspace_->subspace_overlap());
                auto timer_collapse_end = stage_time_point();
                ProfilingCounters::instance().davidson_timing.collapse_time_ms +=
                    std::chrono::duration<double, std::milli>(timer_collapse_end -
                                                              timer_collapse_start)
                        .count();
                log_davidson_memory("after collapse", maximum_basis_size, number_of_plane_waves);
                diagonalization_info =
                    subspace.diagonalize_subspace(basis_size,
                                                  maximum_basis_size,
                                                  is_gamma_only,
                                                  workspace_->subspace_hamiltonian(),
                                                  workspace_->subspace_overlap(),
                                                  workspace_->local_eigenvalues(),
                                                  eigenvector_matrix.data());
                execution_policy_.broadcast_across_bands(&diagonalization_info, 1);
                synchronize_eigenpairs_across_bands(basis_size,
                                                    maximum_basis_size,
                                                    workspace_->local_eigenvalues(),
                                                    eigenvector_matrix.data());
                if (diagonalization_info != 0) {
                    log::warn("    Davidson: zhegv still failed after collapse (info={}), "
                              "using previous eigenvalues",
                              diagonalization_info);
                    for (int band_index = 0; band_index < number_of_bands; band_index++) {
                        workspace_->local_eigenvalues()[band_index] =
                            previous_eigenvalues[band_index];
                    }
                }
            }

            bool converged = true;
            int new_vectors = 0;

            /**
             * No Ritz vectors are formed. Convergence is decided on the
             * eigenvalues, and the correction vectors below are built straight
             * from the subspace coefficients into the free basis columns.
             */
            auto timer_ritz_start = stage_time_point();
            std::vector<bool> band_converged(number_of_bands, true);
            auto timer_ritz_end = stage_time_point();
            ProfilingCounters::instance().davidson_timing.ritz_vector_time_ms +=
                std::chrono::duration<double, std::milli>(timer_ritz_end - timer_ritz_start)
                    .count();

            auto timer_residual_start = stage_time_point();
            bool skip_convergence_check = (iteration == 0);

            int number_of_occupied_bands = quantum_system.number_of_occupied_bands_per_spin();
            Real empty_band_tolerance =
                std::max(math::tolerances::empty_band_tolerance_relaxation_factor * tolerance_,
                         math::tolerances::empty_band_tolerance_floor);

            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                Real eigenvalue_change = std::abs(workspace_->local_eigenvalues()[band_index] -
                                                  previous_eigenvalues[band_index]);
                Real band_tolerance =
                    (band_index < number_of_occupied_bands) ? tolerance_ : empty_band_tolerance;
                bool this_band_converged =
                    (!skip_convergence_check && eigenvalue_change < band_tolerance);

                if (this_band_converged) {
                    continue;
                }

                converged = false;
                band_converged[band_index] = false;
            }

            int number_unconverged = 0;
            Real max_eigenvalue_change = 0.0;
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                if (!band_converged[band_index])
                    number_unconverged++;
                Real change = std::abs(workspace_->local_eigenvalues()[band_index] -
                                       previous_eigenvalues[band_index]);
                max_eigenvalue_change = std::max(max_eigenvalue_change, change);
            }
            number_of_unconverged_bands_ = number_unconverged;
            log::debug("       Davidson iter {}: basis={} unconv={}/{} tol={:.2e} max_de={:.4e}",
                       iteration,
                       basis_size,
                       number_unconverged,
                       number_of_bands,
                       tolerance_,
                       max_eigenvalue_change);

            if (converged) {
                auto timer_residual_end = stage_time_point();
                ProfilingCounters::instance().davidson_timing.residual_time_ms +=
                    std::chrono::duration<double, std::milli>(timer_residual_end -
                                                              timer_residual_start)
                        .count();
                (void)stage_time_point();
                StageTimer extract_timer("davidson.extract");
                subspace.extract_eigenvectors(basis_size,
                                              number_of_plane_waves,
                                              number_of_bands,
                                              kpoint_index_state,
                                              is_gamma_only,
                                              eigenvector_matrix.data(),
                                              workspace_->basis_vectors(),
                                              workspace_->local_eigenvalues(),
                                              workspace_->band_scratch(),
                                              quantum_system);
                return;
            }

            /**
             * Expand subspace with preconditioned residuals if not converged
             * 1. Compute residuals and apply preconditioner.
             */
            workspace_->sync_to_device();

            auto timer_residual_end = stage_time_point();
            ProfilingCounters::instance().davidson_timing.residual_time_ms +=
                std::chrono::duration<double, std::milli>(timer_residual_end - timer_residual_start)
                    .count();

            auto timer_ortho_start = stage_time_point();
            int start_new_vectors = new_vectors;
            workspace_->sync_to_host();
            /**
             * Build the preconditioned corrections directly into the columns
             * the basis is about to grow into, compacted over the unconverged
             * bands. Nothing is staged through a residual array on the way.
             */
            int number_of_candidates = subspace.build_correction_vectors(
                basis_size,
                number_of_plane_waves,
                number_of_bands,
                maximum_basis_size - (basis_size + new_vectors),
                is_gamma_only,
                eigenvector_matrix.data(),
                basis_size,
                workspace_->local_eigenvalues(),
                band_converged,
                kinetic_energy_squared.data(),
                workspace_->hamiltonian_basis_vectors(),
                workspace_->overlap_basis_vectors(),
                workspace_->basis_vectors() +
                    static_cast<std::size_t>(basis_size + new_vectors) * number_of_plane_waves);

            int accepted_candidates =
                subspace.orthogonalize_batch(basis_size + new_vectors,
                                             number_of_candidates,
                                             number_of_plane_waves,
                                             is_gamma_only,
                                             overlap_operator.get(),
                                             workspace_->basis_vectors(),
                                             workspace_->overlap_basis_vectors());

            execution_policy_.broadcast_across_bands(&accepted_candidates, 1);
            synchronize_basis_across_bands(basis_size + new_vectors,
                                           accepted_candidates,
                                           number_of_plane_waves,
                                           workspace_->basis_vectors(),
                                           nullptr,
                                           workspace_->overlap_basis_vectors());
            new_vectors += accepted_candidates;
            int added_vectors = new_vectors - start_new_vectors;
            auto timer_ortho_end = stage_time_point();
            ProfilingCounters::instance().davidson_timing.orthogonalize_time_ms +=
                std::chrono::duration<double, std::milli>(timer_ortho_end - timer_ortho_start)
                    .count();

            if (added_vectors > 0) {
                auto timer_hpsi_start = stage_time_point();
                applicator.apply_batch(
                    workspace_->basis_vectors() + basis_size * number_of_plane_waves,
                    workspace_->hamiltonian_basis_vectors() + basis_size * number_of_plane_waves,
                    added_vectors);
                auto timer_hpsi_end = stage_time_point();
                ProfilingCounters::instance().davidson_timing.hamiltonian_application_time_ms +=
                    std::chrono::duration<double, std::milli>(timer_hpsi_end - timer_hpsi_start)
                        .count();
            }

            int old_basis_size = basis_size;
            basis_size += new_vectors;

            if (new_vectors > 0) {
                auto timer_update_start = stage_time_point();
                subspace.update_subspace_matrices(old_basis_size,
                                                  new_vectors,
                                                  basis_size,
                                                  maximum_basis_size,
                                                  number_of_plane_waves,
                                                  is_gamma_only,
                                                  workspace_->basis_vectors(),
                                                  workspace_->hamiltonian_basis_vectors(),
                                                  workspace_->overlap_basis_vectors(),
                                                  workspace_->subspace_hamiltonian(),
                                                  workspace_->subspace_overlap());
                auto timer_update_end = stage_time_point();
                ProfilingCounters::instance().davidson_timing.subspace_update_time_ms +=
                    std::chrono::duration<double, std::milli>(timer_update_end - timer_update_start)
                        .count();
            }

            // Subspace collapse when basis is full
            if (basis_size >= maximum_basis_size) {
                auto timer_collapse_start = stage_time_point();
                basis_size = subspace.collapse_subspace(basis_size,
                                                        number_of_plane_waves,
                                                        number_of_bands,
                                                        maximum_basis_size,
                                                        is_gamma_only,
                                                        is_identity_overlap,
                                                        overlap_operator.get(),
                                                        workspace_->subspace_hamiltonian(),
                                                        workspace_->subspace_overlap(),
                                                        workspace_->basis_vectors(),
                                                        workspace_->hamiltonian_basis_vectors(),
                                                        workspace_->overlap_basis_vectors(),
                                                        workspace_->band_scratch());
                /**
                 * Only the size needs agreeing here. The collapse itself now
                 * forms each band range on one rank and allgathers, so the
                 * collapsed basis is already identical everywhere; republishing
                 * it would move three full arrays for nothing.
                 */
                execution_policy_.broadcast_across_bands(&basis_size, 1);
                std::memset(workspace_->subspace_hamiltonian(),
                            0,
                            maximum_basis_size * maximum_basis_size * sizeof(Complex));
                std::memset(workspace_->subspace_overlap(),
                            0,
                            maximum_basis_size * maximum_basis_size * sizeof(Complex));
                subspace.update_subspace_matrices(0,
                                                  basis_size,
                                                  basis_size,
                                                  maximum_basis_size,
                                                  number_of_plane_waves,
                                                  is_gamma_only,
                                                  workspace_->basis_vectors(),
                                                  workspace_->hamiltonian_basis_vectors(),
                                                  workspace_->overlap_basis_vectors(),
                                                  workspace_->subspace_hamiltonian(),
                                                  workspace_->subspace_overlap());
                auto timer_collapse_end = stage_time_point();
                ProfilingCounters::instance().davidson_timing.collapse_time_ms +=
                    std::chrono::duration<double, std::milli>(timer_collapse_end -
                                                              timer_collapse_start)
                        .count();
                log_davidson_memory("after collapse", maximum_basis_size, number_of_plane_waves);
            }

            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                previous_eigenvalues[band_index] = workspace_->local_eigenvalues()[band_index];
            }
            log_davidson_memory("at end of iteration", basis_size, number_of_plane_waves);
        }

        // Final extraction after max iterations exhausted
        std::vector<Complex> eigenvector_matrix(maximum_basis_size * maximum_basis_size);
        subspace.diagonalize_subspace(basis_size,
                                      maximum_basis_size,
                                      is_gamma_only,
                                      workspace_->subspace_hamiltonian(),
                                      workspace_->subspace_overlap(),
                                      workspace_->local_eigenvalues(),
                                      eigenvector_matrix.data());
        synchronize_eigenpairs_across_bands(basis_size,
                                            maximum_basis_size,
                                            workspace_->local_eigenvalues(),
                                            eigenvector_matrix.data());
        (void)stage_time_point();
        StageTimer extract_timer("davidson.extract");
        subspace.extract_eigenvectors(basis_size,
                                      number_of_plane_waves,
                                      number_of_bands,
                                      kpoint_index_state,
                                      is_gamma_only,
                                      eigenvector_matrix.data(),
                                      workspace_->basis_vectors(),
                                      workspace_->local_eigenvalues(),
                                      workspace_->band_scratch(),
                                      quantum_system);
    }

    [[nodiscard]] std::string name() const override {
        return "davidson";
    }

    void set_tolerance(Real tolerance) override {
        tolerance_ = tolerance;
    }

  protected:
    /**
     * Working basis size as a multiple of the band count, from
     * method.scf.davidson.basis_expansion_factor.
     *
     * Two arrays of this many vectors dominate the solver's memory. The
     * default is 2; 4 restarts less often at twice
     * the footprint, and was the default until the collapse stopped
     * rebuilding its subspace from the vectors, which was most of what a
     * restart cost. Measured on the 54-atom silicon cell at four ranks, 2
     * takes 14% more wall time than 4 for 12% less memory; on the 432-atom
     * deck the same switch removes 1.0 GB of replicated subspace per rank.
     */
    int davidson_basis_expansion_factor = 2;

    /**
     * Record the resident set at a point inside the solver, at debug level.
     *
     * The solver allocates in two steps that are far apart in time: the
     * workspace on the first solve, and the collapse scratch at the first
     * restart, several Davidson iterations later. A run sampled before the
     * first restart understates its own footprint, so both are probed.
     */
    void log_davidson_memory(const char* label,
                             int maximum_basis_size,
                             int number_of_plane_waves) const {
        ResidentMemorySample sample = ProfilingCounters::instance().record_resident_memory();
        if (!sample.available) {
            return;
        }
        log::debug("       Davidson memory {}: basis={} npw={} rss={} peak={}",
                   label,
                   maximum_basis_size,
                   number_of_plane_waves,
                   ResidentMemorySample::format_gibibytes(sample.current_bytes),
                   ResidentMemorySample::format_gibibytes(sample.peak_bytes));
    }

    // ─── What a subclass with other hardware replaces ─────────────────

    /// The arrays a solve works in. Host memory here.
    [[nodiscard]] virtual std::unique_ptr<PlanewaveDavidsonWorkspace> create_workspace() const {
        return std::make_unique<SerialDavidsonWorkspace>();
    }

    /// Copy host data into a workspace array.
    virtual void copy_into_workspace(Complex* destination, const Complex* source, int count) const {
        std::memcpy(destination, source, static_cast<std::size_t>(count) * sizeof(Complex));
    }

    /// Return once everything this solver has queued is done. Nothing is queued on the host.
    virtual void wait_for_queued_work() const {}

    /// Give back what was held for the starting rotation only. Nothing here.
    virtual void release_starting_rotation() {}

    /**
     * Copy every band of a k-point state into the workspace as the first candidates, and
     * return how many were copied.
     */
    virtual int copy_bands_into_workspace(const PlanewaveKohnShamQuantumSystem& quantum_system,
                                          int kpoint_index_state,
                                          int number_of_bands,
                                          int number_of_plane_waves,
                                          Complex* candidates) const {
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            copy_into_workspace(
                &candidates[static_cast<std::size_t>(band_index) * number_of_plane_waves],
                quantum_system.wavefunction_band_data(kpoint_index_state, band_index),
                number_of_plane_waves);
        }
        return number_of_bands;
    }

    /**
     * Create the Hamiltonian applicator this solver owns, and with it anything that has to
     * match it. The host applicator here; the subspace manager made at construction suits it.
     */
    virtual void create_owned_hamiltonian_applicator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const KPointsGrid& kpoint_grid,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system) {
        owned_hamiltonian_applicator_ =
            std::make_unique<SerialPlanewaveHamiltonianApplicator>(crystal_structure,
                                                                   planewave_basis,
                                                                   kpoint_grid,
                                                                   pseudopotential_quantum_system,
                                                                   execution_policy_);
    }

    /**
     * The time at a stage boundary, taken once the device has finished what
     * the stage before it queued.
     *
     * Device products return when they are queued, not when they are done, so
     * a clock read straight after one charges its cost to whichever later
     * stage first has to wait -- on a device run the subspace eigenproblem
     * appeared to cost what the products ahead of it did. Waiting here costs
     * nothing the solve would not have waited for anyway, and is a no-op
     * without a device.
     */
    [[nodiscard]] std::chrono::high_resolution_clock::time_point stage_time_point() const {
        wait_for_queued_work();
        return std::chrono::high_resolution_clock::now();
    }

    // ─── Applicator management ────────────────────────────────────

    bool has_hamiltonian_applicator() const {
        return injected_hamiltonian_applicator_ != nullptr || owned_hamiltonian_applicator_;
    }

    void ensure_hamiltonian_applicator_initialized(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        const KPointsGrid& kpoint_grid,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system) {
        if (injected_hamiltonian_applicator_ == nullptr && !owned_hamiltonian_applicator_) {
            create_owned_hamiltonian_applicator(
                crystal_structure, planewave_basis, kpoint_grid, pseudopotential_quantum_system);
            owned_hamiltonian_applicator_->set_projector_set_store(projector_set_provider_.store());
            for (auto& [kpoint_index, projectors] : pending_cached_hubbard_projectors_) {
                owned_hamiltonian_applicator_->set_cached_hubbard_projectors(kpoint_index,
                                                                             projectors);
            }
            pending_cached_hubbard_projectors_.clear();
            if (pending_hubbard_d_matrix_ != nullptr) {
                owned_hamiltonian_applicator_->set_hubbard_d_matrix(pending_hubbard_d_matrix_);
                pending_hubbard_d_matrix_ = nullptr;
            }
        }
    }

    PlanewaveHamiltonianApplicator& get_hamiltonian_applicator() {
        if (injected_hamiltonian_applicator_ != nullptr) {
            return *injected_hamiltonian_applicator_;
        }
        return *owned_hamiltonian_applicator_;
    }
    PlanewaveSubspaceManager& get_subspace_manager() {
        if (injected_subspace_manager_ != nullptr) {
            return *injected_subspace_manager_;
        }
        return *owned_subspace_manager_;
    }

    // ─── Basis initialization helpers ────────────────────────────────

    int seed_basis_from_wavefunctions(int number_of_plane_waves,
                                      int number_of_bands,
                                      const PlanewaveKohnShamQuantumSystem& quantum_system,
                                      int kpoint_index_state,
                                      bool is_gamma_only,
                                      OverlapOperator* overlap_operator,
                                      PlanewaveHamiltonianApplicator& applicator,
                                      PlanewaveSubspaceManager& subspace,
                                      Complex* basis_vectors,
                                      Complex* hamiltonian_basis_vectors,
                                      Complex* overlap_basis_vectors) const {
        int basis_size = 0;

        int number_of_pw_for_state =
            quantum_system.number_of_plane_waves_for_kpoint_state(kpoint_index_state);
        if (number_of_pw_for_state == number_of_plane_waves) {
            int number_of_candidates = copy_bands_into_workspace(
                quantum_system,
                kpoint_index_state,
                number_of_bands,
                number_of_plane_waves,
                &basis_vectors[static_cast<std::size_t>(basis_size) * number_of_plane_waves]);

            int accepted = subspace.orthogonalize_batch(basis_size,
                                                        number_of_candidates,
                                                        number_of_plane_waves,
                                                        is_gamma_only,
                                                        overlap_operator,
                                                        basis_vectors,
                                                        overlap_basis_vectors);
            execution_policy_.broadcast_across_bands(&accepted, 1);
            synchronize_basis_across_bands(basis_size,
                                           accepted,
                                           number_of_plane_waves,
                                           basis_vectors,
                                           nullptr,
                                           overlap_basis_vectors);
            basis_size += accepted;

            if (basis_size > 0) {
                applicator.apply_batch(basis_vectors, hamiltonian_basis_vectors, basis_size);
            }
        }

        return basis_size;
    }

    int seed_basis_from_kinetic_energy(int basis_size,
                                       int number_of_plane_waves,
                                       int number_of_bands,
                                       const std::vector<Real>& kinetic_energy_squared,
                                       bool is_gamma_only,
                                       OverlapOperator* overlap_operator,
                                       PlanewaveHamiltonianApplicator& applicator,
                                       PlanewaveSubspaceManager& subspace,
                                       Complex* basis_vectors,
                                       Complex* hamiltonian_basis_vectors,
                                       Complex* overlap_basis_vectors) const {
        std::vector<std::pair<Real, int>> kinetic_energy_with_index;
        for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
            kinetic_energy_with_index.push_back(
                {kinetic_energy_squared[g_vector_index], g_vector_index});
        }
        std::sort(kinetic_energy_with_index.begin(), kinetic_energy_with_index.end());

        int start_index = basis_size;
        int number_of_candidates = 0;
        for (int band_index = start_index; band_index < number_of_bands; band_index++) {
            std::fill(
                &basis_vectors[(basis_size + number_of_candidates) * number_of_plane_waves],
                &basis_vectors[(basis_size + number_of_candidates + 1) * number_of_plane_waves],
                Complex(0.0, 0.0));
            basis_vectors[(basis_size + number_of_candidates) * number_of_plane_waves +
                          kinetic_energy_with_index[band_index].second] = Complex(1.0, 0.0);
            number_of_candidates++;
        }

        int accepted = subspace.orthogonalize_batch(basis_size,
                                                    number_of_candidates,
                                                    number_of_plane_waves,
                                                    is_gamma_only,
                                                    overlap_operator,
                                                    basis_vectors,
                                                    overlap_basis_vectors);
        execution_policy_.broadcast_across_bands(&accepted, 1);
        synchronize_basis_across_bands(basis_size,
                                       accepted,
                                       number_of_plane_waves,
                                       basis_vectors,
                                       nullptr,
                                       overlap_basis_vectors);
        basis_size += accepted;

        int new_vectors = basis_size - start_index;
        if (new_vectors > 0) {
            applicator.apply_batch(&basis_vectors[start_index * number_of_plane_waves],
                                   &hamiltonian_basis_vectors[start_index * number_of_plane_waves],
                                   new_vectors);
        }

        return basis_size;
    }

    /**
     * ── Band-distribution synchronisation ─────────────────────────────────
     *
     * With bands distributed, every rank holds the whole Davidson basis and
     * runs the subspace algebra redundantly. Redundant is not identical:
     * vendor BLAS chooses its summation order from the machine state, so ranks
     * drift apart in the last bits, and Davidson turns that into control-flow
     * divergence -- orthogonalisation rejects candidates on a threshold, so
     * ranks stop agreeing on how many vectors the next batch holds, and the
     * allgather that divides H|psi> fails outright.
     *
     * The fix is the one Quantum ESPRESSO uses across its band groups: one
     * rank's result is authoritative, and it is republished at every point the
     * basis or a decision changes. Republishing only the newly accepted block
     * is not enough -- it was tried -- because those vectors are orthogonal to
     * the sender's basis, not to a receiver's slightly different one. The
     * basis therefore has to be rank 0's everywhere, at all times, which is
     * what the two helpers below and their call sites guarantee.
     */
    void synchronize_basis_across_bands(int first_vector,
                                        int number_of_vectors,
                                        int number_of_plane_waves,
                                        Complex* basis_vectors,
                                        Complex* hamiltonian_basis_vectors,
                                        Complex* overlap_basis_vectors) const {
        if (execution_policy_.band_distribution_size() <= 1 || number_of_vectors <= 0)
            return;
        std::size_t offset = static_cast<std::size_t>(first_vector) * number_of_plane_waves;
        int count = number_of_vectors * number_of_plane_waves;
        execution_policy_.broadcast_across_bands(basis_vectors + offset, count);
        if (hamiltonian_basis_vectors != nullptr)
            execution_policy_.broadcast_across_bands(hamiltonian_basis_vectors + offset, count);
        /**
         * Skip the overlap broadcast when it aliases the basis: the bytes have
         * just been sent, and sending them again would double the traffic of
         * the largest collective in the solver for no change in the result.
         */
        if (overlap_basis_vectors != nullptr && overlap_basis_vectors != basis_vectors)
            execution_policy_.broadcast_across_bands(overlap_basis_vectors + offset, count);
    }

    /// Eigenvalues drive every convergence decision; eigenvectors drive every Ritz vector.
    void synchronize_eigenpairs_across_bands(int basis_size,
                                             int maximum_basis_size,
                                             Real* eigenvalues,
                                             Complex* eigenvector_matrix) const {
        if (execution_policy_.band_distribution_size() <= 1)
            return;
        execution_policy_.broadcast_across_bands(eigenvalues, basis_size);
        execution_policy_.broadcast_across_bands(eigenvector_matrix,
                                                 maximum_basis_size * maximum_basis_size);
    }

    int max_iterations_ = 20;
    /// Bands the last solve() left short of the tolerance when it ran out of iterations.
    int number_of_unconverged_bands_ = 0;
    Real tolerance_ = math::tolerances::initial_eigenvalue_tolerance;

    // ─── Execution policy ──────────────────────────────────────────────

    const execution::ExecutionPolicy& execution_policy_;

    // ─── Hamiltonian applicator ────────────────────────────────────────

    PlanewaveHamiltonianApplicator* injected_hamiltonian_applicator_ = nullptr;
    std::unique_ptr<PlanewaveHamiltonianApplicator> owned_hamiltonian_applicator_;
    /// The iterator's projector sets when it handed them over, else a set built per solve.
    ProjectorSetProvider projector_set_provider_;
    std::map<int, const q3::Matrix<Complex>*> pending_cached_hubbard_projectors_;
    const std::vector<Complex>* pending_hubbard_d_matrix_ = nullptr;
    /// Whether a Hubbard projector set or D matrix has ever been handed to this solver.
    bool hubbard_term_was_set_ = false;

    // ─── Subspace manager ──────────────────────────────────────────────

    PlanewaveSubspaceManager* injected_subspace_manager_ = nullptr;
    std::unique_ptr<PlanewaveSubspaceManager> owned_subspace_manager_;
    std::unique_ptr<PlanewaveDavidsonWorkspace> workspace_;
};

} // namespace q3
