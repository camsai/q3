#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/starting_subspace_solver_gpu.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/one_kpoint_quantum_system.hpp"

using namespace q3;

namespace {

using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::OneKpointQuantumSystem;

/// An applicator whose H is diagonal in the plane waves: H_gg = 1 + g/2.
class DiagonalApplicator : public PlanewaveHamiltonianApplicator {
  public:
    explicit DiagonalApplicator(int number_of_plane_waves)
        : number_of_plane_waves_(number_of_plane_waves) {}

    static Real diagonal(int g_index) {
        return 1.0 + 0.5 * g_index;
    }
    void update_potential(const std::vector<Real>&) override {}
    void prepare_kpoint(int, int, const std::vector<int>&, const std::vector<Real>&) override {}
    void apply_batch(const Complex* input_wavefunctions,
                     Complex* output_hamiltonian_wavefunctions,
                     int number_of_vectors) override {
        for (int vector = 0; vector < number_of_vectors; ++vector) {
            for (int g_index = 0; g_index < number_of_plane_waves_; ++g_index) {
                std::size_t element =
                    static_cast<std::size_t>(vector) * number_of_plane_waves_ + g_index;
                output_hamiltonian_wavefunctions[element] =
                    diagonal(g_index) * input_wavefunctions[element];
            }
        }
    }
    void invalidate_smooth_potential_cache() override {}

  private:
    int number_of_plane_waves_;
};

} // namespace

TEST(GPUStartingSubspaceSolver, ProjectsAndRotatesAsTheDirectSumsDo) {
    const int number_of_plane_waves = 37;
    const int number_of_vectors = 5;
    const int number_of_bands = 3;

    std::vector<Complex> vectors(static_cast<std::size_t>(number_of_plane_waves) *
                                 number_of_vectors);
    for (int vector = 0; vector < number_of_vectors; ++vector) {
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            vectors[static_cast<std::size_t>(vector) * number_of_plane_waves + g_index] =
                Complex(std::sin(0.3 * (g_index + 1) * (vector + 1)),
                        std::cos(0.7 * (g_index + 2) + 0.2 * vector));
        }
    }
    auto element = [&](int g_index, int vector) {
        return vectors[static_cast<std::size_t>(vector) * number_of_plane_waves + g_index];
    };

    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS blas(device_backend);
    DiagonalApplicator applicator(number_of_plane_waves);
    GPUStartingSubspaceSolver solver(applicator, blas, device_backend, number_of_plane_waves);

    solver.apply_hamiltonian(vectors.data(), number_of_vectors);
    std::vector<Complex> hamiltonian_matrix(number_of_vectors * number_of_vectors);
    std::vector<Complex> overlap_matrix(number_of_vectors * number_of_vectors);
    solver.project(hamiltonian_matrix.data(), overlap_matrix.data());

    for (int row = 0; row < number_of_vectors; ++row) {
        for (int column = 0; column < number_of_vectors; ++column) {
            Complex expected_hamiltonian(0.0, 0.0), expected_overlap(0.0, 0.0);
            for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                Complex product = std::conj(element(g_index, row)) * element(g_index, column);
                expected_overlap += product;
                expected_hamiltonian += DiagonalApplicator::diagonal(g_index) * product;
            }
            int index = row + column * number_of_vectors;
            EXPECT_NEAR(std::abs(hamiltonian_matrix[index] - expected_hamiltonian), 0.0, 1e-11);
            EXPECT_NEAR(std::abs(overlap_matrix[index] - expected_overlap), 0.0, 1e-11);
        }
    }

    std::vector<Complex> coefficients(static_cast<std::size_t>(number_of_vectors) *
                                      number_of_bands);
    for (int band = 0; band < number_of_bands; ++band) {
        for (int vector = 0; vector < number_of_vectors; ++vector) {
            coefficients[static_cast<std::size_t>(band) * number_of_vectors + vector] =
                Complex(0.1 * (vector + 1) - 0.05 * band, 0.03 * (band + 1) * (vector - 2));
        }
    }
    /**
     * The rotated vectors are the bands of a k-point state: the solver writes them into the
     * quantum system's band block on the device, and nothing crosses to the host for it.
     */
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);
    OneKpointQuantumSystem quantum_system(kpoint_grid, number_of_bands);
    quantum_system.allocate_electronic_states({number_of_plane_waves});
    const execution::gpu::device::DeviceTransferTotals before_rotation =
        device_backend->transfer_totals();

    solver.rotate_into_bands(coefficients.data(), number_of_bands, quantum_system, 0);

    const execution::gpu::device::DeviceTransferTotals rotation_traffic =
        device_backend->transfer_totals().since(before_rotation);
    EXPECT_EQ(rotation_traffic.device_to_host_bytes, 0u);
    EXPECT_EQ(rotation_traffic.host_to_device_bytes, coefficients.size() * sizeof(Complex));
    EXPECT_TRUE(
        device_backend->is_device_memory(quantum_system.device_band_block(0, device_backend)));
    EXPECT_EQ(device_backend->transfer_totals().since(before_rotation).host_to_device_bytes,
              coefficients.size() * sizeof(Complex));

    for (int band = 0; band < number_of_bands; ++band) {
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            Complex expected(0.0, 0.0);
            for (int vector = 0; vector < number_of_vectors; ++vector) {
                expected +=
                    element(g_index, vector) *
                    coefficients[static_cast<std::size_t>(band) * number_of_vectors + vector];
            }
            EXPECT_NEAR(
                std::abs(quantum_system.wavefunction(0, band, g_index) - expected), 0.0, 1e-12);
        }
    }
}

TEST(GPUStartingSubspaceSolver, RefusesABandBlockOfAnotherShape) {
    const int number_of_plane_waves = 7;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS blas(device_backend);
    DiagonalApplicator applicator(number_of_plane_waves);
    GPUStartingSubspaceSolver solver(applicator, blas, device_backend, number_of_plane_waves);
    std::vector<Complex> vectors(static_cast<std::size_t>(number_of_plane_waves) * 2,
                                 Complex(1.0, 0.0));
    solver.apply_hamiltonian(vectors.data(), 2);
    std::vector<Complex> coefficients(2 * 2, Complex(1.0, 0.0));

    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);
    OneKpointQuantumSystem quantum_system(kpoint_grid, 2);
    quantum_system.allocate_electronic_states({number_of_plane_waves + 1});

    EXPECT_THROW(solver.rotate_into_bands(coefficients.data(), 2, quantum_system, 0),
                 std::invalid_argument);
}
