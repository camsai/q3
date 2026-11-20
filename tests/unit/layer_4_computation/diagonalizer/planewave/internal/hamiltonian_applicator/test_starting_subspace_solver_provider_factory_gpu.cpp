#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/wavefunction/planewave/starting_subspace_solver.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_provider_factory_gpu.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"

using namespace q3;

namespace {

using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;

/// An applicator that records what the provider asked of it, and applies nothing.
class RecordingApplicator : public PlanewaveHamiltonianApplicator {
  public:
    void update_potential(const std::vector<Real>& /*real_space_potential*/) override {
        ++potential_updates;
        potential_is_valid = true;
    }
    void prepare_kpoint(int kpoint_index,
                        int number_of_plane_waves,
                        const std::vector<int>& /*g_vector_indices*/,
                        const std::vector<Real>& /*kinetic_energy_squared*/) override {
        prepared_kpoint_index = kpoint_index;
        prepared_number_of_plane_waves = number_of_plane_waves;
    }
    void apply_batch(const Complex* /*input_wavefunctions*/,
                     Complex* /*output_hamiltonian_wavefunctions*/,
                     int /*number_of_vectors*/) override {}
    void invalidate_smooth_potential_cache() override {
        potential_is_valid = false;
    }
    [[nodiscard]] bool is_smooth_potential_cache_valid() const override {
        return potential_is_valid;
    }

    int potential_updates = 0;
    bool potential_is_valid = false;
    int prepared_kpoint_index = -1;
    int prepared_number_of_plane_waves = -1;
};

/// A host solver that is only ever returned, never used.
class UnusedSolver : public StartingSubspaceSolver {
  public:
    void apply_hamiltonian(const Complex* /*vectors*/, int /*number_of_vectors*/) override {}
    void project(Complex* /*hamiltonian_matrix*/, Complex* /*overlap_matrix*/) override {}
    void rotate_into_bands(const Complex* /*coefficients*/,
                           int /*number_of_bands*/,
                           PlanewaveKohnShamQuantumSystem& /*quantum_system*/,
                           int /*kpoint_index_state*/) override {}
};

class StartingSubspaceSolverProviderFactoryTest : public ::testing::Test {
  protected:
    StartingSubspaceSolverProviderFactoryTest()
        : device_backend_{std::make_shared<CPUBackedGPUDeviceBackend>()}, blas_{device_backend_} {
        kpoint_grid_.generate(2, 1, 1, 0, 0, 0);
    }

    /// The provider for `context`, and whether it asked for the applicator.
    std::unique_ptr<StartingSubspaceSolverProvider> create(const StartingRotationContext& context,
                                                           bool& applicator_was_asked_for) {
        applicator_was_asked_for = false;
        auto applicator = [&]() -> PlanewaveHamiltonianApplicator& {
            applicator_was_asked_for = true;
            return applicator_;
        };
        return create_gpu_starting_subspace_solver_provider(
            context, applicator, blas_, device_backend_);
    }

    StartingSubspaceSolver& prepare(StartingSubspaceSolverProvider& provider,
                                    int kpoint_index_state,
                                    int number_of_plane_waves) {
        int kpoint_index = kpoint_index_state % kpoint_grid_.number_of_kpoints();
        return provider.prepare(kpoint_index_state,
                                kpoint_index,
                                number_of_plane_waves,
                                g_vector_indices_,
                                kinetic_energy_squared_,
                                potential_,
                                kpoint_grid_,
                                serial_solver_);
    }

    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend_;
    CPUBackedGPUBLAS blas_;
    KPointsGrid kpoint_grid_;
    RecordingApplicator applicator_;
    UnusedSolver serial_solver_;
    std::vector<int> g_vector_indices_;
    std::vector<Real> kinetic_energy_squared_;
    std::vector<Real> potential_;
};

} // namespace

TEST_F(StartingSubspaceSolverProviderFactoryTest, TheStandardProblemGoesToTheDevice) {
    bool applicator_was_asked_for = false;
    auto provider = create(StartingRotationContext{}, applicator_was_asked_for);

    EXPECT_TRUE(applicator_was_asked_for);
    StartingSubspaceSolver& solver = prepare(*provider, 0, 11);
    EXPECT_NE(&solver, &serial_solver_);
    EXPECT_NE(dynamic_cast<GPUStartingSubspaceSolver*>(&solver), nullptr);
}

TEST_F(StartingSubspaceSolverProviderFactoryTest, EveryOtherRunKeepsTheHostSolver) {
    StartingRotationContext with_hubbard;
    with_hubbard.has_hubbard_term = true;
    StartingRotationContext gamma_point;
    gamma_point.is_gamma_only = true;
    StartingRotationContext band_group;
    band_group.band_group_size = 2;
    StartingRotationContext general_overlap;
    general_overlap.overlap_is_identity = false;

    for (const StartingRotationContext& context :
         {with_hubbard, gamma_point, band_group, general_overlap}) {
        bool applicator_was_asked_for = true;
        auto provider = create(context, applicator_was_asked_for);

        EXPECT_FALSE(applicator_was_asked_for);
        EXPECT_EQ(&prepare(*provider, 0, 11), &serial_solver_);
        EXPECT_EQ(applicator_.prepared_kpoint_index, -1);
    }
}

TEST_F(StartingSubspaceSolverProviderFactoryTest, TheDeviceProviderPreparesEachKPoint) {
    bool applicator_was_asked_for = false;
    auto provider = create(StartingRotationContext{}, applicator_was_asked_for);

    StartingSubspaceSolver& first = prepare(*provider, 0, 11);
    EXPECT_EQ(applicator_.prepared_kpoint_index, 0);
    EXPECT_EQ(applicator_.prepared_number_of_plane_waves, 11);
    EXPECT_EQ(applicator_.spin_channel(), 0);
    EXPECT_EQ(applicator_.potential_updates, 1);

    // The same channel's next k-point reuses the potential and the solver.
    StartingSubspaceSolver& second = prepare(*provider, 1, 13);
    EXPECT_EQ(&second, &first);
    EXPECT_EQ(applicator_.prepared_kpoint_index, 1);
    EXPECT_EQ(applicator_.prepared_number_of_plane_waves, 13);
    EXPECT_EQ(applicator_.potential_updates, 1);

    // The second spin channel's first k-point state: the channel moves, its potential is set.
    prepare(*provider, 2, 11);
    EXPECT_EQ(applicator_.spin_channel(), 1);
    EXPECT_EQ(applicator_.prepared_kpoint_index, 0);
    EXPECT_EQ(applicator_.potential_updates, 2);
}
