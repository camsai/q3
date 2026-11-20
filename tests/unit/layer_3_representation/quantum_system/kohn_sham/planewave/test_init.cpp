#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"

using namespace q3;
using execution::gpu::device::DeviceTransferTotals;
using test_support::CPUBackedGPUDeviceBackend;

namespace {

constexpr int bands_per_kpoint_state = 3;

/**
 * A quantum system on a 2x1x1 mesh, built without a basis: the band blocks
 * need only the number of plane waves at each k-point, here 4 and 5.
 */
class BandBlockQuantumSystem : public PlanewaveKohnShamQuantumSystem {
  public:
    BandBlockQuantumSystem(const KPointsGrid& kpoint_grid, int number_of_spin_components_value) {
        kpoint_grid_ = &kpoint_grid;
        number_of_bands_ = bands_per_kpoint_state;
        number_of_electrons_ = 2.0;
        this->number_of_spin_components = number_of_spin_components_value;
    }
};

struct BandBlockFixture {
    KPointsGrid kpoint_grid;
    std::unique_ptr<BandBlockQuantumSystem> quantum_system;
    std::vector<int> number_of_plane_waves_per_kpoint;

    explicit BandBlockFixture(int number_of_spin_components = 1) {
        kpoint_grid.generate(2, 1, 1, 0, 0, 0);
        number_of_plane_waves_per_kpoint.assign(kpoint_grid.number_of_kpoints(), 4);
        number_of_plane_waves_per_kpoint.back() = 5;
        quantum_system =
            std::make_unique<BandBlockQuantumSystem>(kpoint_grid, number_of_spin_components);
        quantum_system->allocate_electronic_states(number_of_plane_waves_per_kpoint);
    }

    /// Band n of k-point state s holds (s, n) + i·g at plane wave g.
    void fill_bands() {
        for (int state = 0; state < quantum_system->number_of_kpoint_states(); ++state) {
            int number_of_plane_waves =
                quantum_system->number_of_plane_waves_for_kpoint_state(state);
            for (int band = 0; band < bands_per_kpoint_state; ++band) {
                Complex* band_data = quantum_system->mutable_wavefunction_band_data(state, band);
                for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                    band_data[g_index] = Complex(10.0 * state + band, g_index);
                }
            }
        }
    }
};

} // namespace

// ─── Layout ─────────────────────────────────────────────────────────────────

TEST(PlanewaveKohnShamQuantumSystem, AllocatedBandBlocksAreZeroAndSizedPerKpoint) {
    BandBlockFixture fixture;
    const PlanewaveKohnShamQuantumSystem& quantum_system = *fixture.quantum_system;

    ASSERT_EQ(quantum_system.number_of_kpoint_states(), 2);
    EXPECT_EQ(quantum_system.number_of_plane_waves_for_kpoint_state(0), 4);
    EXPECT_EQ(quantum_system.number_of_plane_waves_for_kpoint_state(1), 5);
    for (int band = 0; band < bands_per_kpoint_state; ++band) {
        for (int g_index = 0; g_index < 5; ++g_index) {
            EXPECT_EQ(quantum_system.wavefunction(1, band, g_index), Complex(0.0, 0.0));
        }
        EXPECT_EQ(quantum_system.eigenvalue(1, band), 0.0);
    }
}

TEST(PlanewaveKohnShamQuantumSystem, ABandIsAColumnOfItsKpointStatesBlock) {
    BandBlockFixture fixture;
    fixture.fill_bands();
    const PlanewaveKohnShamQuantumSystem& quantum_system = *fixture.quantum_system;

    for (int state = 0; state < 2; ++state) {
        const int number_of_plane_waves =
            quantum_system.number_of_plane_waves_for_kpoint_state(state);
        const Complex* band_block = quantum_system.band_block(state);
        for (int band = 0; band < bands_per_kpoint_state; ++band) {
            EXPECT_EQ(quantum_system.wavefunction_band_data(state, band),
                      band_block + static_cast<std::size_t>(band) * number_of_plane_waves);
            for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
                const Complex expected(10.0 * state + band, g_index);
                EXPECT_EQ(band_block[band * number_of_plane_waves + g_index], expected);
                EXPECT_EQ(quantum_system.wavefunction(state, band, g_index), expected);
            }
        }
    }
}

TEST(PlanewaveKohnShamQuantumSystem, TheWritableBlockAndTheWritableBandAreTheSameStorage) {
    BandBlockFixture fixture;

    fixture.quantum_system->mutable_band_block(1)[2 * 5 + 3] = Complex(7.0, -7.0);

    EXPECT_EQ(fixture.quantum_system->wavefunction(1, 2, 3), Complex(7.0, -7.0));
    EXPECT_EQ(fixture.quantum_system->mutable_wavefunction_band_data(1, 2),
              fixture.quantum_system->mutable_band_block(1) + 2 * 5);
}

TEST(PlanewaveKohnShamQuantumSystem, SpinChannelsAreSeparateKpointStatesWithTheirOwnBlocks) {
    BandBlockFixture fixture(/*number_of_spin_components=*/2);
    fixture.fill_bands();
    const PlanewaveKohnShamQuantumSystem& quantum_system = *fixture.quantum_system;

    ASSERT_EQ(quantum_system.number_of_kpoint_states(), 4);
    EXPECT_EQ(quantum_system.number_of_plane_waves_for_kpoint_state(2), 4);
    EXPECT_EQ(quantum_system.number_of_plane_waves_for_kpoint_state(3), 5);
    EXPECT_EQ(quantum_system.wavefunction(3, 1, 4), Complex(31.0, 4.0));
    EXPECT_EQ(quantum_system.wavefunction(1, 1, 4), Complex(11.0, 4.0));
}

// ─── Snapshots ──────────────────────────────────────────────────────────────

TEST(PlanewaveKohnShamQuantumSystem, ACopiedSystemHasItsOwnBandBlocks) {
    BandBlockFixture fixture;
    fixture.fill_bands();
    fixture.quantum_system->set_eigenvalue(1, 2, -0.25);
    BandBlockQuantumSystem snapshot(fixture.kpoint_grid, 1);

    snapshot.copy_quantum_states_from(*fixture.quantum_system);
    fixture.quantum_system->mutable_wavefunction_band_data(1, 2)[4] = Complex(0.0, 0.0);

    EXPECT_EQ(snapshot.number_of_plane_waves_for_kpoint_state(1), 5);
    EXPECT_EQ(snapshot.wavefunction(1, 2, 4), Complex(12.0, 4.0));
    EXPECT_EQ(snapshot.eigenvalue(1, 2), -0.25);
    EXPECT_NE(snapshot.band_block(1), fixture.quantum_system->band_block(1));
}

// ─── Device copies ──────────────────────────────────────────────────────────

TEST(PlanewaveKohnShamQuantumSystem, ANullBackendIsRefusedForADeviceBlock) {
    BandBlockFixture fixture;

    EXPECT_THROW((void)fixture.quantum_system->device_band_block(0, nullptr),
                 std::invalid_argument);
}

TEST(PlanewaveKohnShamQuantumSystem, TheDeviceBlockIsUploadedOnceAndReadersDoNotMarkItStale) {
    BandBlockFixture fixture;
    fixture.fill_bands();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    const PlanewaveKohnShamQuantumSystem& quantum_system = *fixture.quantum_system;
    const std::size_t block_bytes = 5 * bands_per_kpoint_state * sizeof(Complex);

    const Complex* device_block = quantum_system.device_band_block(1, device_backend);
    (void)quantum_system.wavefunction_band_data(1, 0);
    (void)quantum_system.band_block(1);
    (void)quantum_system.wavefunction(1, 2, 4);
    const Complex* device_block_again = quantum_system.device_band_block(1, device_backend);

    EXPECT_TRUE(device_backend->is_device_memory(device_block));
    EXPECT_EQ(device_block, device_block_again);
    EXPECT_EQ(device_block[2 * 5 + 4], Complex(12.0, 4.0));
    const DeviceTransferTotals totals = device_backend->transfer_totals();
    EXPECT_EQ(totals.host_to_device_bytes, block_bytes);
    EXPECT_EQ(totals.host_to_device_copies, 1u);
    EXPECT_EQ(totals.device_to_host_copies, 0u);
}

TEST(PlanewaveKohnShamQuantumSystem, AHostWriterMakesTheDeviceBlockStale) {
    BandBlockFixture fixture;
    fixture.fill_bands();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    (void)fixture.quantum_system->device_band_block(0, device_backend);

    fixture.quantum_system->mutable_wavefunction_band_data(0, 1)[2] = Complex(-3.0, 0.5);
    const Complex* device_block = fixture.quantum_system->device_band_block(0, device_backend);

    EXPECT_EQ(device_block[1 * 4 + 2], Complex(-3.0, 0.5));
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 2u);
}

TEST(PlanewaveKohnShamQuantumSystem, ADeviceWriterIsSeenByTheNextHostReader) {
    BandBlockFixture fixture;
    fixture.fill_bands();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();

    fixture.quantum_system->mutable_device_band_block(1, device_backend)[0 * 5 + 1] =
        Complex(99.0, 0.0);

    const PlanewaveKohnShamQuantumSystem& quantum_system = *fixture.quantum_system;
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 0u);
    EXPECT_EQ(quantum_system.wavefunction(1, 0, 1), Complex(99.0, 0.0));
    EXPECT_EQ(quantum_system.wavefunction(1, 2, 4), Complex(12.0, 4.0));
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 1u);
    // The other k-point state's block was never asked for on the device.
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_bytes,
              5 * bands_per_kpoint_state * sizeof(Complex));
}

TEST(PlanewaveKohnShamQuantumSystem, AWholeBlockDeviceWriterUploadsNothing) {
    BandBlockFixture fixture;
    fixture.fill_bands();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    const int number_of_elements = 5 * bands_per_kpoint_state;

    Complex* device_block = fixture.quantum_system->overwrite_device_band_block(1, device_backend);
    for (int element = 0; element < number_of_elements; ++element) {
        device_block[element] = Complex(-1.0 * element, 0.0);
    }

    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 0u);
    // The device holds the bands now; the host sees them when it next asks.
    EXPECT_EQ(fixture.quantum_system->wavefunction(1, 2, 4), Complex(-14.0, 0.0));
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 1u);
}

TEST(PlanewaveKohnShamQuantumSystem, ACopiedSystemGetsTheDevicesValuesAndNoDeviceCopy) {
    BandBlockFixture fixture;
    fixture.fill_bands();
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    fixture.quantum_system->mutable_device_band_block(0, device_backend)[0] = Complex(5.0, 5.0);
    BandBlockQuantumSystem snapshot(fixture.kpoint_grid, 1);

    snapshot.copy_quantum_states_from(*fixture.quantum_system);
    const execution::gpu::device::DeviceTransferTotals after_copy =
        device_backend->transfer_totals();

    // The one block only the device held came down once; reading the snapshot moves nothing.
    EXPECT_EQ(after_copy.device_to_host_copies, 1u);
    EXPECT_EQ(snapshot.wavefunction(0, 0, 0), Complex(5.0, 5.0));
    EXPECT_EQ(snapshot.wavefunction(1, 2, 4), Complex(12.0, 4.0));
    EXPECT_EQ(device_backend->transfer_totals().since(after_copy).device_to_host_copies, 0u);
    EXPECT_EQ(device_backend->transfer_totals().since(after_copy).host_to_device_copies, 0u);
}
