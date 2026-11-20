#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "layer_3_representation/basis/planewave/structure_factor_calculator_serial.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/structure_factor_calculator_gpu.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/two_species_projector_system.hpp"

using namespace q3;
using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::TwoSpeciesProjectorSystem;

namespace {

/// The four-atom cell with the host calculator and a device one that forms `block` phases at once.
struct DevicePhaseSumCase {
    TwoSpeciesProjectorSystem system;
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas{device_backend};
    SerialStructureFactorCalculator host_calculator{system.crystal_structure,
                                                    system.planewave_basis};
    GPUStructureFactorCalculator device_calculator;
    std::vector<int> g_vector_indices;

    explicit DevicePhaseSumCase(std::size_t block)
        : device_calculator(
              system.crystal_structure, system.planewave_basis, device_backend, device_blas, block)
        , g_vector_indices(static_cast<std::size_t>(system.planewave_basis.number_of_g_vectors())) {
        std::iota(g_vector_indices.begin(), g_vector_indices.end(), 0);
    }

    [[nodiscard]] std::vector<std::vector<Complex>> phase_weights() const {
        std::vector<std::vector<Complex>> weights(2, std::vector<Complex>(g_vector_indices.size()));
        for (std::size_t entry = 0; entry < g_vector_indices.size(); entry++) {
            weights[0][entry] = Complex(1.0 / (1.0 + entry), 0.3);
            weights[1][entry] = Complex(-0.2, 2.0 / (2.0 + entry));
        }
        return weights;
    }

    void expect_same_structure_factors() const {
        const std::vector<std::vector<Real>> atom_weights = {{1.0, 0.0, 1.0, 0.0},
                                                             {0.5, -2.0, 3.0, 1.5}};
        const auto host = host_calculator.structure_factors(g_vector_indices, atom_weights);
        const auto device = device_calculator.structure_factors(g_vector_indices, atom_weights);
        ASSERT_EQ(device.size(), host.size());
        for (std::size_t set = 0; set < host.size(); set++) {
            ASSERT_EQ(device[set].size(), host[set].size());
            for (std::size_t entry = 0; entry < host[set].size(); entry++) {
                EXPECT_NEAR(device[set][entry].real(), host[set][entry].real(), 1e-12);
                EXPECT_NEAR(device[set][entry].imag(), host[set][entry].imag(), 1e-12);
            }
        }
    }

    void expect_same_gradient_sums() const {
        const std::vector<int> set_of_atom = {1, -1, 0, 1};
        const auto weights = phase_weights();
        const auto host =
            host_calculator.gradient_phase_sums(g_vector_indices, weights, set_of_atom);
        const auto device =
            device_calculator.gradient_phase_sums(g_vector_indices, weights, set_of_atom);
        ASSERT_EQ(device.size(), host.size());
        for (std::size_t atom_index = 0; atom_index < host.size(); atom_index++) {
            for (int direction = 0; direction < 3; direction++) {
                EXPECT_NEAR(device[atom_index][direction], host[atom_index][direction], 1e-10);
            }
        }
        EXPECT_EQ(device[1][0], 0.0);
    }
};

} // namespace

TEST(GPUStructureFactorCalculator, StructureFactorsAreTheHostOnes) {
    DevicePhaseSumCase phase_sum_case(GPUStructureFactorCalculator::default_phase_block_elements);
    phase_sum_case.expect_same_structure_factors();
}

TEST(GPUStructureFactorCalculator, GradientSumsAreTheHostOnes) {
    DevicePhaseSumCase phase_sum_case(GPUStructureFactorCalculator::default_phase_block_elements);
    phase_sum_case.expect_same_gradient_sums();
}

TEST(GPUStructureFactorCalculator, GVectorsTakenInBlocksGiveTheSameSums) {
    // 4 atoms and 40 phases per block: ten G-vectors at a time, the last block shorter.
    DevicePhaseSumCase phase_sum_case(40);
    ASSERT_GT(phase_sum_case.g_vector_indices.size(), 25U);
    phase_sum_case.expect_same_structure_factors();
    phase_sum_case.expect_same_gradient_sums();
}

TEST(GPUStructureFactorCalculator, RefusesANullBackendAndWeightsOfTheWrongLength) {
    TwoSpeciesProjectorSystem system;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas{device_backend};
    EXPECT_THROW(GPUStructureFactorCalculator(
                     system.crystal_structure, system.planewave_basis, nullptr, device_blas),
                 std::invalid_argument);
    GPUStructureFactorCalculator calculator(
        system.crystal_structure, system.planewave_basis, device_backend, device_blas);
    EXPECT_THROW((void)calculator.structure_factors({0, 1}, {{1.0, 1.0}}), std::invalid_argument);
    EXPECT_THROW((void)calculator.gradient_phase_sums({0}, {{Complex(1.0, 0.0)}}, {0, 0}),
                 std::invalid_argument);
}
