#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/types/init.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"

using namespace q3;
using execution::gpu::device::DeviceTransferTotals;
using execution::gpu::device::GPUDeviceBuffer;
using test_support::CPUBackedGPUDeviceBackend;

namespace {

constexpr std::size_t number_of_elements = 8;
constexpr std::size_t buffer_bytes = number_of_elements * sizeof(Real);

} // namespace

TEST(GPUDeviceBackend, AFreshBackendHasMovedNothing) {
    CPUBackedGPUDeviceBackend device_backend;

    const DeviceTransferTotals totals = device_backend.transfer_totals();

    EXPECT_EQ(totals.host_to_device_bytes, 0u);
    EXPECT_EQ(totals.device_to_host_bytes, 0u);
    EXPECT_EQ(totals.host_to_device_copies, 0u);
    EXPECT_EQ(totals.device_to_host_copies, 0u);
}

TEST(GPUDeviceBackend, ACopyFromHostMemoryIntoABufferCountsAsHostToDevice) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    GPUDeviceBuffer<Real> device_buffer(number_of_elements, device_backend);
    std::vector<Real> host_array(number_of_elements, 1.5);

    device_backend->copy(device_buffer.data(), host_array.data(), buffer_bytes);

    const DeviceTransferTotals totals = device_backend->transfer_totals();
    EXPECT_EQ(totals.host_to_device_bytes, buffer_bytes);
    EXPECT_EQ(totals.host_to_device_copies, 1u);
    EXPECT_EQ(totals.device_to_host_bytes, 0u);
    EXPECT_EQ(device_buffer.data()[number_of_elements - 1], 1.5);
}

TEST(GPUDeviceBackend, ACopyFromABufferIntoHostMemoryCountsAsDeviceToHost) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    GPUDeviceBuffer<Real> device_buffer(number_of_elements, device_backend);
    device_buffer.data()[3] = -2.0;
    std::vector<Real> host_array(number_of_elements, 0.0);

    device_backend->copy(host_array.data(), device_buffer.data(), buffer_bytes);

    const DeviceTransferTotals totals = device_backend->transfer_totals();
    EXPECT_EQ(totals.device_to_host_bytes, buffer_bytes);
    EXPECT_EQ(totals.device_to_host_copies, 1u);
    EXPECT_EQ(totals.host_to_device_bytes, 0u);
    EXPECT_EQ(host_array[3], -2.0);
}

TEST(GPUDeviceBackend, CopiesThatStayOnOneSideAreNotCrossings) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    GPUDeviceBuffer<Real> first_buffer(number_of_elements, device_backend);
    GPUDeviceBuffer<Real> second_buffer(number_of_elements, device_backend);
    std::vector<Real> first_host_array(number_of_elements, 1.0);
    std::vector<Real> second_host_array(number_of_elements, 0.0);
    device_backend->fill_zero(first_buffer.data(), buffer_bytes);

    device_backend->copy(second_buffer.data(), first_buffer.data(), buffer_bytes);
    device_backend->copy(second_host_array.data(), first_host_array.data(), buffer_bytes);

    const DeviceTransferTotals totals = device_backend->transfer_totals();
    EXPECT_EQ(totals.host_to_device_copies, 0u);
    EXPECT_EQ(totals.device_to_host_copies, 0u);
    EXPECT_EQ(second_host_array[0], 1.0);
    EXPECT_EQ(second_buffer.data()[0], 0.0);
}

TEST(GPUDeviceBackend, APointerIntoTheMiddleOfABufferIsDeviceMemory) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    GPUDeviceBuffer<Real> device_buffer(number_of_elements, device_backend);
    std::vector<Real> host_array(2, 4.0);

    device_backend->copy(device_buffer.data() + 6, host_array.data(), 2 * sizeof(Real));

    EXPECT_TRUE(device_backend->is_device_memory(device_buffer.data()));
    EXPECT_TRUE(device_backend->is_device_memory(device_buffer.data() + number_of_elements - 1));
    EXPECT_FALSE(device_backend->is_device_memory(device_buffer.data() + number_of_elements));
    EXPECT_FALSE(device_backend->is_device_memory(host_array.data()));
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_bytes, 2 * sizeof(Real));
}

TEST(GPUDeviceBackend, ABuffersMemoryStopsBeingDeviceMemoryWhenTheBufferIsGone) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    const Real* released_pointer = nullptr;
    {
        GPUDeviceBuffer<Real> device_buffer(number_of_elements, device_backend);
        released_pointer = device_buffer.data();
        EXPECT_TRUE(device_backend->is_device_memory(released_pointer));
    }

    EXPECT_FALSE(device_backend->is_device_memory(released_pointer));
}

TEST(GPUDeviceBackend, AMovedBufferKeepsItsMemoryRegistered) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    GPUDeviceBuffer<Real> device_buffer(number_of_elements, device_backend);
    const Real* pointer = device_buffer.data();

    GPUDeviceBuffer<Real> moved_buffer(std::move(device_buffer));

    EXPECT_EQ(moved_buffer.data(), pointer);
    EXPECT_TRUE(device_backend->is_device_memory(pointer));
}

TEST(GPUDeviceBackend, TheDifferenceOfTwoReadingsIsTheTrafficBetweenThem) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    GPUDeviceBuffer<Real> device_buffer(number_of_elements, device_backend);
    std::vector<Real> host_array(number_of_elements, 1.0);
    device_backend->copy(device_buffer.data(), host_array.data(), buffer_bytes);
    const DeviceTransferTotals before = device_backend->transfer_totals();

    device_backend->copy(device_buffer.data(), host_array.data(), buffer_bytes);
    device_backend->copy(host_array.data(), device_buffer.data(), buffer_bytes / 2);

    const DeviceTransferTotals traffic = device_backend->transfer_totals().since(before);
    EXPECT_EQ(traffic.host_to_device_bytes, buffer_bytes);
    EXPECT_EQ(traffic.host_to_device_copies, 1u);
    EXPECT_EQ(traffic.device_to_host_bytes, buffer_bytes / 2);
    EXPECT_EQ(traffic.device_to_host_copies, 1u);
}
