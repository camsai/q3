#pragma once

/**
 * GPUPlanewaveDavidsonDiagonalizer — the Davidson solver with its arrays and products on
 * a device.
 *
 * The iteration is PlanewaveDavidsonDiagonalizer's. This class replaces what depends on
 * where the arrays live: the workspace is device memory, the Hamiltonian applicator and
 * the subspace manager are the GPU ones, the bands are read from their device copy, a
 * stage is timed once the device has finished it, and the starting rotation is offered
 * the device. The applicator and the subspace manager are made on the first solve, when
 * the crystal and the basis are known.
 */

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/math/linear_algebra/gpu/lapack/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/davidson.hpp"
#include "layer_4_computation/diagonalizer/planewave/davidson_workspace_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_provider.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_provider_factory_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/factory.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/factory_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/subspace_manager/gpu.hpp"

namespace q3 {

/// The Davidson solver on a device.
class GPUPlanewaveDavidsonDiagonalizer : public PlanewaveDavidsonDiagonalizer {
  public:
    /**
     * @param execution_policy  Execution policy for band-group operations.
     * @param gpu_blas          GPU BLAS wrapper. Must outlive this object.
     * @param gpu_fft_grid      GPU FFT grid, or null for a backend without one. Must outlive
     *                          this object.
     * @param gpu_eigensolver   GPU eigensolver. Must outlive this object.
     * @param device_backend    GPU device backend (shared ownership).
     */
    GPUPlanewaveDavidsonDiagonalizer(
        const execution::ExecutionPolicy& execution_policy,
        execution::gpu::blas::GPUBLAS& gpu_blas,
        gpu::GPUFFTGrid* gpu_fft_grid,
        execution::gpu::lapack::GPUEigensolver& gpu_eigensolver,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend)
        : PlanewaveDavidsonDiagonalizer(execution_policy)
        , gpu_blas_{gpu_blas}
        , gpu_fft_grid_{gpu_fft_grid}
        , gpu_eigensolver_{gpu_eigensolver}
        , gpu_device_backend_{std::move(device_backend)} {}

    /**
     * Enable mixed-precision mode: the local potential FFT section of H|ψ⟩ runs in FP32
     * while kinetic energy and nonlocal projectors remain FP64. Call before the first solve.
     */
    void set_use_mixed_precision(bool use_mixed_precision) {
        use_mixed_precision_ = use_mixed_precision;
    }

    /**
     * Solve the subspace eigenproblem with the device eigensolver, where the default copies
     * the subspace matrices to the host for LAPACK. For a backend whose device eigensolver
     * has been measured to pay on matrices this small; call before the first solve.
     */
    void set_solves_subspace_on_device(bool solves_subspace_on_device) {
        solves_subspace_on_device_ = solves_subspace_on_device;
    }

    /**
     * Take the store, and give its sets' device copies a budget: half of the device's
     * memory, the other half being this solver's basis and workspace, the bands and the
     * transforms. With more k-points than fit, the sets used longest ago give up their
     * device copies and are uploaded or rebuilt when their k-point comes round again. A
     * backend that reports no total sets no limit, and a null store takes the store back.
     */
    void set_projector_set_store(KleinmanBylanderProjectorSetStore* projector_set_store) override {
        PlanewaveDavidsonDiagonalizer::set_projector_set_store(projector_set_store);
        if (projector_set_store != nullptr) {
            projector_set_store->set_device_memory_budget(
                gpu_device_backend_->total_memory_bytes() / projector_device_memory_budget_divisor);
        }
    }

    /**
     * Offer the device for the starting rotation: H, the subspace projections and the rotation.
     *
     * Who does it is decided once, at the first k-point, by
     * create_gpu_starting_subspace_solver_provider(): the device where its applicator is known
     * to reproduce the initializer's host Hamiltonian, the caller's host solver otherwise.
     */
    StartingSubspaceSolver& prepare_starting_subspace_solver(
        int kpoint_index_state,
        int kpoint_index,
        int number_of_plane_waves,
        const std::vector<int>& g_vector_indices,
        const std::vector<Real>& kinetic_energy_squared,
        const std::vector<Real>& real_space_potential,
        const Crystal& crystal_structure,
        const KPointsGrid& kpoint_grid,
        const PlanewaveBasis& planewave_basis,
        const NormConservingPlanewaveKohnShamQuantumSystem& pseudopotential_quantum_system,
        const KleinmanBylanderProjectorSet& projector_set,
        StartingSubspaceSolver& serial_solver) override {
        if (starting_subspace_solver_provider_ == nullptr) {
            StartingRotationContext context;
            context.has_hubbard_term = hubbard_term_was_set_;
            context.is_gamma_only = planewave_basis.is_gamma_only();
            context.band_group_size = execution_policy_.group_size();
            context.overlap_is_identity = create_overlap_operator(crystal_structure,
                                                                  pseudopotential_quantum_system,
                                                                  projector_set,
                                                                  execution_policy_)
                                              ->is_identity();
            auto applicator = [&]() -> PlanewaveHamiltonianApplicator& {
                ensure_hamiltonian_applicator_initialized(crystal_structure,
                                                          planewave_basis,
                                                          kpoint_grid,
                                                          pseudopotential_quantum_system);
                return get_hamiltonian_applicator();
            };
            starting_subspace_solver_provider_ = create_gpu_starting_subspace_solver_provider(
                context, applicator, gpu_blas_, gpu_device_backend_);
        }
        return starting_subspace_solver_provider_->prepare(kpoint_index_state,
                                                           kpoint_index,
                                                           number_of_plane_waves,
                                                           g_vector_indices,
                                                           kinetic_energy_squared,
                                                           real_space_potential,
                                                           kpoint_grid,
                                                           serial_solver);
    }

    /// A Hubbard term changes who may do the starting rotation; the next one decides again.
    void set_cached_hubbard_projectors(int kpoint_index,
                                       const q3::Matrix<Complex>* projectors) override {
        PlanewaveDavidsonDiagonalizer::set_cached_hubbard_projectors(kpoint_index, projectors);
        starting_subspace_solver_provider_.reset();
    }

    /// See set_cached_hubbard_projectors().
    void set_hubbard_d_matrix(const std::vector<Complex>* dense_d_matrix_column_major) override {
        PlanewaveDavidsonDiagonalizer::set_hubbard_d_matrix(dense_d_matrix_column_major);
        starting_subspace_solver_provider_.reset();
    }

  protected:
    [[nodiscard]] std::unique_ptr<PlanewaveDavidsonWorkspace> create_workspace() const override {
        return std::make_unique<GPUDavidsonWorkspace>(gpu_device_backend_);
    }

    /**
     * The workspace is device-resident managed memory, and an element-wise host write would
     * migrate it to the host page by page (and the next kernel back); the backend's copy
     * moves it in one transfer.
     */
    void
    copy_into_workspace(Complex* destination, const Complex* source, int count) const override {
        gpu_device_backend_->copy(
            destination, source, static_cast<std::size_t>(count) * sizeof(Complex));
    }

    /**
     * Device products return when they are queued, not when they are done, so a clock read
     * straight after one charges its cost to whichever later stage first has to wait.
     */
    void wait_for_queued_work() const override {
        gpu_device_backend_->synchronize();
    }

    /// The starting rotation is over once a solve begins; its batch buffers go back.
    void release_starting_rotation() override {
        starting_subspace_solver_provider_.reset();
    }

    /**
     * The bands are read where the last solve left them, on the device: one copy from the
     * k-point state's block into the workspace, both device memory. The block is uploaded
     * here only if the host has written it since.
     */
    int copy_bands_into_workspace(const PlanewaveKohnShamQuantumSystem& quantum_system,
                                  int kpoint_index_state,
                                  int number_of_bands,
                                  int number_of_plane_waves,
                                  Complex* candidates) const override {
        gpu_device_backend_->copy(
            candidates,
            quantum_system.device_band_block(kpoint_index_state, gpu_device_backend_),
            static_cast<std::size_t>(number_of_bands) * number_of_plane_waves * sizeof(Complex));
        return number_of_bands;
    }

    /// The GPU applicator, and the GPU subspace manager that works on the same device arrays.
    void create_owned_hamiltonian_applicator(const Crystal& crystal_structure,
                                             const PlanewaveBasis& planewave_basis,
                                             const KPointsGrid& kpoint_grid,
                                             const NormConservingPlanewaveKohnShamQuantumSystem&
                                                 pseudopotential_quantum_system) override {
        auto gpu_applicator =
            std::make_unique<GPUPlanewaveHamiltonianApplicator>(crystal_structure,
                                                                planewave_basis,
                                                                kpoint_grid,
                                                                pseudopotential_quantum_system,
                                                                gpu_blas_,
                                                                gpu_fft_grid_,
                                                                gpu_device_backend_);
        gpu_applicator->set_use_mixed_precision(use_mixed_precision_);
        owned_hamiltonian_applicator_ = std::move(gpu_applicator);

        auto gpu_subspace_manager = create_gpu_planewave_subspace_manager(
            planewave_basis.is_gamma_only(), gpu_blas_, execution_policy_, gpu_device_backend_);
        if (solves_subspace_on_device_) {
            gpu_subspace_manager->set_device_eigensolver(&gpu_eigensolver_);
        }
        owned_subspace_manager_ = std::move(gpu_subspace_manager);
        log::info("     GPU Davidson: using GPU applicator and subspace manager");
    }

  private:
    /// The projector sets may hold one part in this many of the device's memory.
    static constexpr std::size_t projector_device_memory_budget_divisor = 2;

    execution::gpu::blas::GPUBLAS& gpu_blas_;
    gpu::GPUFFTGrid* gpu_fft_grid_;
    execution::gpu::lapack::GPUEigensolver& gpu_eigensolver_;
    /// Whether the subspace manager is handed gpu_eigensolver_; see the setter.
    bool solves_subspace_on_device_ = false;
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> gpu_device_backend_;
    bool use_mixed_precision_ = false;
    /// Who does the starting rotation, chosen at its first k-point; see the method that uses it.
    std::unique_ptr<StartingSubspaceSolverProvider> starting_subspace_solver_provider_;
};

} // namespace q3
