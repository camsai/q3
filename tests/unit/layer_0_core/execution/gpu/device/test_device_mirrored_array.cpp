#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "layer_0_core/execution/gpu/device/device_mirrored_array.hpp"
#include "layer_0_core/types/init.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"

using namespace q3;
using execution::gpu::device::DeviceMirroredArray;
using execution::gpu::device::DeviceTransferTotals;
using test_support::CPUBackedGPUDeviceBackend;

namespace {

constexpr std::size_t number_of_elements = 6;
constexpr std::size_t array_bytes = number_of_elements * sizeof(Complex);

/// An array holding 1, 2, ... on the host, and a fresh host stand-in backend to mirror it on.
struct MirroredArrayFixture {
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    DeviceMirroredArray<Complex> array{number_of_elements};

    MirroredArrayFixture() {
        Complex* elements = array.mutable_host();
        for (std::size_t element = 0; element < number_of_elements; ++element) {
            elements[element] = Complex(static_cast<Real>(element + 1), 0.0);
        }
    }

    [[nodiscard]] DeviceTransferTotals totals() const {
        return device_backend->transfer_totals();
    }
};

} // namespace

// ─── Without a device: a plain host array ───────────────────────────────────

TEST(DeviceMirroredArray, ANewArrayIsAZeroedHostArray) {
    DeviceMirroredArray<Complex> array(number_of_elements);

    EXPECT_EQ(array.size(), number_of_elements);
    EXPECT_TRUE(array.host_is_current());
    EXPECT_FALSE(array.device_is_current());
    for (std::size_t element = 0; element < number_of_elements; ++element) {
        EXPECT_EQ(array.host()[element], Complex(0.0, 0.0));
    }
}

TEST(DeviceMirroredArray, HostWritesAreReadBack) {
    DeviceMirroredArray<Real> array(3);
    array.mutable_host()[1] = 4.5;

    const DeviceMirroredArray<Real>& read_only = array;
    EXPECT_EQ(read_only.host()[1], 4.5);
    EXPECT_EQ(read_only.host(), array.mutable_host());
}

TEST(DeviceMirroredArray, ANullBackendIsRefusedAndChangesNothing) {
    MirroredArrayFixture fixture;

    EXPECT_THROW((void)fixture.array.device(nullptr), std::invalid_argument);
    EXPECT_THROW((void)fixture.array.mutable_device(nullptr), std::invalid_argument);
    EXPECT_THROW((void)fixture.array.overwrite_device(nullptr), std::invalid_argument);
    EXPECT_TRUE(fixture.array.host_is_current());
    EXPECT_FALSE(fixture.array.device_is_current());
    EXPECT_EQ(fixture.array.host()[5], Complex(6.0, 0.0));
}

TEST(DeviceMirroredArray, ReleasingADeviceCopyThatWasNeverMadeDoesNothing) {
    DeviceMirroredArray<Real> array(2);
    array.mutable_host()[0] = 7.0;

    array.release_device();

    EXPECT_EQ(array.host()[0], 7.0);
    EXPECT_TRUE(array.host_is_current());
}

TEST(DeviceMirroredArray, DefaultConstructedIsEmpty) {
    DeviceMirroredArray<Complex> array;

    EXPECT_EQ(array.size(), 0u);
    EXPECT_TRUE(array.host_is_current());
    EXPECT_FALSE(array.device_is_current());
}

// ─── Which copy is current ──────────────────────────────────────────────────

TEST(DeviceMirroredArray, HostWritesAloneMoveNothing) {
    MirroredArrayFixture fixture;

    EXPECT_TRUE(fixture.array.host_is_current());
    EXPECT_FALSE(fixture.array.device_is_current());
    EXPECT_EQ(fixture.totals().host_to_device_bytes, 0u);
    EXPECT_EQ(fixture.totals().device_to_host_bytes, 0u);
}

TEST(DeviceMirroredArray, TheFirstDeviceViewUploadsOnceAndBothAreCurrent) {
    MirroredArrayFixture fixture;
    const DeviceMirroredArray<Complex>& read_only = fixture.array;

    const Complex* device_elements = read_only.device(fixture.device_backend);

    EXPECT_TRUE(read_only.host_is_current());
    EXPECT_TRUE(read_only.device_is_current());
    EXPECT_TRUE(fixture.device_backend->is_device_memory(device_elements));
    EXPECT_FALSE(fixture.device_backend->is_device_memory(read_only.host()));
    for (std::size_t element = 0; element < number_of_elements; ++element) {
        EXPECT_EQ(device_elements[element], read_only.host()[element]);
    }
    EXPECT_EQ(fixture.totals().host_to_device_bytes, array_bytes);
    EXPECT_EQ(fixture.totals().host_to_device_copies, 1u);
    EXPECT_EQ(fixture.totals().device_to_host_bytes, 0u);
}

TEST(DeviceMirroredArray, ReadingACurrentCopyAgainMovesNothing) {
    MirroredArrayFixture fixture;
    const Complex* first_device_view = fixture.array.device(fixture.device_backend);
    const DeviceTransferTotals after_upload = fixture.totals();

    const Complex* second_device_view = fixture.array.device(fixture.device_backend);
    (void)fixture.array.host();

    EXPECT_EQ(first_device_view, second_device_view);
    EXPECT_EQ(fixture.totals().since(after_upload).host_to_device_copies, 0u);
    EXPECT_EQ(fixture.totals().since(after_upload).device_to_host_copies, 0u);
}

TEST(DeviceMirroredArray, AHostWriteMakesTheDeviceCopyStaleUntilTheNextDeviceView) {
    MirroredArrayFixture fixture;
    (void)fixture.array.device(fixture.device_backend);

    fixture.array.mutable_host()[2] = Complex(-5.0, 1.0);

    EXPECT_TRUE(fixture.array.host_is_current());
    EXPECT_FALSE(fixture.array.device_is_current());
    EXPECT_EQ(fixture.totals().host_to_device_copies, 1u);

    const Complex* device_elements = fixture.array.device(fixture.device_backend);

    EXPECT_TRUE(fixture.array.device_is_current());
    EXPECT_EQ(device_elements[2], Complex(-5.0, 1.0));
    EXPECT_EQ(fixture.totals().host_to_device_copies, 2u);
    EXPECT_EQ(fixture.totals().host_to_device_bytes, 2 * array_bytes);
    EXPECT_EQ(fixture.totals().device_to_host_copies, 0u);
}

TEST(DeviceMirroredArray, ADeviceWriteMakesTheHostCopyStaleUntilTheNextHostView) {
    MirroredArrayFixture fixture;

    Complex* device_elements = fixture.array.mutable_device(fixture.device_backend);
    device_elements[4] = Complex(9.0, -2.0);

    EXPECT_FALSE(fixture.array.host_is_current());
    EXPECT_TRUE(fixture.array.device_is_current());
    EXPECT_EQ(fixture.totals().device_to_host_copies, 0u);

    const DeviceMirroredArray<Complex>& read_only = fixture.array;
    EXPECT_EQ(read_only.host()[4], Complex(9.0, -2.0));
    EXPECT_EQ(read_only.host()[0], Complex(1.0, 0.0));
    EXPECT_TRUE(read_only.host_is_current());
    EXPECT_TRUE(read_only.device_is_current());
    EXPECT_EQ(fixture.totals().device_to_host_copies, 1u);
    EXPECT_EQ(fixture.totals().device_to_host_bytes, array_bytes);
}

TEST(DeviceMirroredArray, AHostPointerKeptAcrossADeviceWriteIsStale) {
    MirroredArrayFixture fixture;
    const Complex* kept_host_view = fixture.array.host();

    fixture.array.mutable_device(fixture.device_backend)[0] = Complex(42.0, 0.0);

    /**
     * The contract of the accessors: the pointer taken before the device write
     * still shows what the host copy held then. Only asking for the host view
     * again brings the new value across.
     */
    EXPECT_EQ(kept_host_view[0], Complex(1.0, 0.0));
    EXPECT_EQ(fixture.array.host()[0], Complex(42.0, 0.0));
}

TEST(DeviceMirroredArray, AWritableHostViewOfAStaleHostCopyDownloadsFirst) {
    MirroredArrayFixture fixture;
    fixture.array.mutable_device(fixture.device_backend)[5] = Complex(0.0, 3.0);

    // A partial host write must not lose what only the device held.
    fixture.array.mutable_host()[0] = Complex(8.0, 0.0);

    EXPECT_EQ(fixture.totals().device_to_host_copies, 1u);
    EXPECT_FALSE(fixture.array.device_is_current());
    EXPECT_EQ(fixture.array.host()[0], Complex(8.0, 0.0));
    EXPECT_EQ(fixture.array.host()[5], Complex(0.0, 3.0));
}

TEST(DeviceMirroredArray, AWritableDeviceViewOfAStaleDeviceCopyUploadsFirst) {
    MirroredArrayFixture fixture;
    (void)fixture.array.device(fixture.device_backend);
    fixture.array.mutable_host()[1] = Complex(6.0, 6.0);

    // A partial device write must not lose what only the host held.
    fixture.array.mutable_device(fixture.device_backend)[0] = Complex(-1.0, 0.0);

    EXPECT_EQ(fixture.totals().host_to_device_copies, 2u);
    EXPECT_EQ(fixture.array.host()[0], Complex(-1.0, 0.0));
    EXPECT_EQ(fixture.array.host()[1], Complex(6.0, 6.0));
}

// ─── Overwriting the whole array on the device ──────────────────────────────

TEST(DeviceMirroredArray, AWholeArrayDeviceWriterUploadsNothing) {
    MirroredArrayFixture fixture;

    Complex* device_elements = fixture.array.overwrite_device(fixture.device_backend);
    for (std::size_t element = 0; element < number_of_elements; ++element) {
        device_elements[element] = Complex(0.0, static_cast<Real>(element));
    }

    EXPECT_TRUE(fixture.device_backend->is_device_memory(device_elements));
    EXPECT_FALSE(fixture.array.host_is_current());
    EXPECT_TRUE(fixture.array.device_is_current());
    EXPECT_EQ(fixture.totals().host_to_device_copies, 0u);
    EXPECT_EQ(fixture.totals().device_to_host_copies, 0u);
    // The host sees what was written when it next asks, in one download.
    EXPECT_EQ(fixture.array.host()[3], Complex(0.0, 3.0));
    EXPECT_EQ(fixture.totals().device_to_host_copies, 1u);
}

TEST(DeviceMirroredArray, OverwritingAStaleDeviceCopyReusesItsBufferWithoutATransfer) {
    MirroredArrayFixture fixture;
    const Complex* first_view = fixture.array.device(fixture.device_backend);
    fixture.array.mutable_host()[0] = Complex(3.0, 3.0);
    const DeviceTransferTotals before = fixture.totals();

    Complex* device_elements = fixture.array.overwrite_device(fixture.device_backend);

    EXPECT_EQ(device_elements, first_view);
    EXPECT_EQ(fixture.totals().since(before).host_to_device_copies, 0u);
    EXPECT_EQ(fixture.totals().since(before).device_to_host_copies, 0u);
}

TEST(DeviceMirroredArray, ReadingTheDeviceAfterAnOverwriteUploadsNothing) {
    MirroredArrayFixture fixture;
    Complex* device_elements = fixture.array.overwrite_device(fixture.device_backend);
    device_elements[0] = Complex(2.0, 0.0);

    const Complex* read_view = fixture.array.device(fixture.device_backend);

    EXPECT_EQ(read_view, device_elements);
    EXPECT_EQ(read_view[0], Complex(2.0, 0.0));
    EXPECT_EQ(fixture.totals().host_to_device_copies, 0u);
}

// ─── Release and restore ────────────────────────────────────────────────────

TEST(DeviceMirroredArray, ReleasingKeepsValuesThatOnlyTheDeviceHeld) {
    MirroredArrayFixture fixture;
    Complex* device_elements = fixture.array.mutable_device(fixture.device_backend);
    device_elements[3] = Complex(2.5, 2.5);

    fixture.array.release_device();

    EXPECT_FALSE(fixture.device_backend->is_device_memory(device_elements));
    EXPECT_TRUE(fixture.array.host_is_current());
    EXPECT_FALSE(fixture.array.device_is_current());
    EXPECT_EQ(fixture.totals().device_to_host_copies, 1u);
    EXPECT_EQ(fixture.array.host()[3], Complex(2.5, 2.5));
}

TEST(DeviceMirroredArray, ReleasingACopyTheHostAlreadyMatchesMovesNothing) {
    MirroredArrayFixture fixture;
    (void)fixture.array.device(fixture.device_backend);

    fixture.array.release_device();

    EXPECT_EQ(fixture.totals().device_to_host_copies, 0u);
    EXPECT_FALSE(fixture.array.device_is_current());
}

TEST(DeviceMirroredArray, ADeviceViewAfterReleaseUploadsAgain) {
    MirroredArrayFixture fixture;
    (void)fixture.array.device(fixture.device_backend);
    fixture.array.release_device();

    const Complex* device_elements = fixture.array.device(fixture.device_backend);

    EXPECT_TRUE(fixture.array.device_is_current());
    EXPECT_EQ(device_elements[5], Complex(6.0, 0.0));
    EXPECT_EQ(fixture.totals().host_to_device_copies, 2u);
}

// ─── Another device ─────────────────────────────────────────────────────────

TEST(DeviceMirroredArray, AskingOnAnotherBackendMovesTheCopyThereAndKeepsTheValues) {
    MirroredArrayFixture fixture;
    Complex* first_device_elements = fixture.array.mutable_device(fixture.device_backend);
    first_device_elements[0] = Complex(13.0, 0.0);
    auto other_backend = std::make_shared<CPUBackedGPUDeviceBackend>();

    const Complex* device_elements = fixture.array.device(other_backend);

    EXPECT_TRUE(other_backend->is_device_memory(device_elements));
    EXPECT_FALSE(fixture.device_backend->is_device_memory(first_device_elements));
    EXPECT_EQ(device_elements[0], Complex(13.0, 0.0));
    // Down from the first device, which alone held the value, and up to the second.
    EXPECT_EQ(fixture.totals().device_to_host_bytes, array_bytes);
    EXPECT_EQ(other_backend->transfer_totals().host_to_device_bytes, array_bytes);
}

TEST(DeviceMirroredArray, OverwritingOnAnotherBackendDownloadsNothingFromTheFirst) {
    MirroredArrayFixture fixture;
    (void)fixture.array.mutable_device(fixture.device_backend);
    auto other_backend = std::make_shared<CPUBackedGPUDeviceBackend>();

    Complex* device_elements = fixture.array.overwrite_device(other_backend);

    EXPECT_TRUE(other_backend->is_device_memory(device_elements));
    EXPECT_EQ(fixture.totals().device_to_host_copies, 0u);
    EXPECT_EQ(other_backend->transfer_totals().host_to_device_copies, 0u);
}

// ─── Copying, moving, resizing ──────────────────────────────────────────────

TEST(DeviceMirroredArray, ACopyHoldsTheValuesOnTheHostAndHasNoDeviceCopy) {
    MirroredArrayFixture fixture;
    fixture.array.mutable_device(fixture.device_backend)[0] = Complex(11.0, 0.0);

    DeviceMirroredArray<Complex> copy(fixture.array);

    EXPECT_TRUE(copy.host_is_current());
    EXPECT_FALSE(copy.device_is_current());
    EXPECT_EQ(copy.host()[0], Complex(11.0, 0.0));
    EXPECT_EQ(copy.host()[1], Complex(2.0, 0.0));

    copy.mutable_host()[1] = Complex(0.0, 0.0);
    EXPECT_EQ(fixture.array.host()[1], Complex(2.0, 0.0));
}

TEST(DeviceMirroredArray, CopyAssignmentDropsTheDestinationsDeviceCopy) {
    MirroredArrayFixture fixture;
    DeviceMirroredArray<Complex> destination(2);
    const Complex* destination_device_elements = destination.device(fixture.device_backend);

    destination = fixture.array;

    EXPECT_EQ(destination.size(), number_of_elements);
    EXPECT_TRUE(destination.host_is_current());
    EXPECT_FALSE(destination.device_is_current());
    EXPECT_FALSE(fixture.device_backend->is_device_memory(destination_device_elements));
    EXPECT_EQ(destination.host()[4], Complex(5.0, 0.0));
}

TEST(DeviceMirroredArray, AMoveCarriesBothCopiesWithoutATransfer) {
    MirroredArrayFixture fixture;
    const Complex* device_elements = fixture.array.mutable_device(fixture.device_backend);
    const DeviceTransferTotals before_move = fixture.totals();

    DeviceMirroredArray<Complex> moved(std::move(fixture.array));

    EXPECT_FALSE(moved.host_is_current());
    EXPECT_TRUE(moved.device_is_current());
    EXPECT_EQ(moved.device(fixture.device_backend), device_elements);
    EXPECT_EQ(fixture.totals().since(before_move).device_to_host_copies, 0u);
    EXPECT_EQ(fixture.totals().since(before_move).host_to_device_copies, 0u);
    EXPECT_EQ(moved.host()[2], Complex(3.0, 0.0));
}

TEST(DeviceMirroredArray, ResizingKeepsLeadingValuesAndDropsTheDeviceCopy) {
    MirroredArrayFixture fixture;
    fixture.array.mutable_device(fixture.device_backend)[1] = Complex(20.0, 0.0);

    fixture.array.resize(number_of_elements + 2);

    EXPECT_EQ(fixture.array.size(), number_of_elements + 2);
    EXPECT_TRUE(fixture.array.host_is_current());
    EXPECT_FALSE(fixture.array.device_is_current());
    EXPECT_EQ(fixture.array.host()[1], Complex(20.0, 0.0));
    EXPECT_EQ(fixture.array.host()[number_of_elements + 1], Complex(0.0, 0.0));
}

TEST(DeviceMirroredArray, AnEmptyArrayHasAnEmptyDeviceView) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    DeviceMirroredArray<Complex> array(0);

    EXPECT_EQ(array.device(device_backend), nullptr);
    EXPECT_TRUE(array.device_is_current());
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 0u);
}

// ─── Readers on several threads ─────────────────────────────────────────────

TEST(DeviceMirroredArray, ManyThreadsReadingAStaleHostCopyDownloadItOnce) {
    constexpr std::size_t large_number_of_elements = 1 << 16;
    constexpr int number_of_threads = 8;
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    DeviceMirroredArray<Real> array(large_number_of_elements);
    Real* device_elements = array.overwrite_device(device_backend);
    for (std::size_t element = 0; element < large_number_of_elements; ++element) {
        device_elements[element] = static_cast<Real>(element);
    }
    const DeviceMirroredArray<Real>& read_only = array;

    // What a parallel loop over bands does: every thread asks for the host view itself.
    std::vector<Real> last_element_seen(number_of_threads, -1.0);
    std::vector<std::thread> readers;
    for (int thread_index = 0; thread_index < number_of_threads; ++thread_index) {
        readers.emplace_back([&read_only, &last_element_seen, thread_index] {
            last_element_seen[static_cast<std::size_t>(thread_index)] =
                read_only.host()[large_number_of_elements - 1];
        });
    }
    for (std::thread& reader : readers) {
        reader.join();
    }

    for (Real seen : last_element_seen) {
        EXPECT_EQ(seen, static_cast<Real>(large_number_of_elements - 1));
    }
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 1u);
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_bytes,
              large_number_of_elements * sizeof(Real));
}

// ─── A host copy that exists only when something asks for it ────────────────

TEST(DeviceMirroredArray, AnArrayWrittenOnlyOnTheDeviceTakesNoHostMemory) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    DeviceMirroredArray<Complex> array;
    array.reset(number_of_elements);

    EXPECT_EQ(array.size(), number_of_elements);
    EXPECT_FALSE(array.host_is_allocated());
    EXPECT_FALSE(array.device_is_allocated());

    Complex* device_elements = array.overwrite_device(device_backend);
    for (std::size_t element = 0; element < number_of_elements; ++element) {
        device_elements[element] = Complex(static_cast<Real>(element), 1.0);
    }

    EXPECT_FALSE(array.host_is_allocated());
    EXPECT_TRUE(array.device_is_allocated());
    EXPECT_TRUE(array.device_is_current());
    EXPECT_EQ(device_backend->transfer_totals().host_to_device_copies, 0u);
    EXPECT_EQ(array.device(device_backend), device_elements);
    EXPECT_FALSE(array.host_is_allocated());
}

TEST(DeviceMirroredArray, TheHostCopyOfADeviceOnlyArrayIsMadeAndFilledWhenAskedFor) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    DeviceMirroredArray<Complex> array;
    array.reset(number_of_elements);
    Complex* device_elements = array.overwrite_device(device_backend);
    for (std::size_t element = 0; element < number_of_elements; ++element) {
        device_elements[element] = Complex(static_cast<Real>(element), 1.0);
    }

    const Complex* host_elements = array.host();

    EXPECT_TRUE(array.host_is_allocated());
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_copies, 1u);
    EXPECT_EQ(device_backend->transfer_totals().device_to_host_bytes, array_bytes);
    for (std::size_t element = 0; element < number_of_elements; ++element) {
        EXPECT_EQ(host_elements[element], Complex(static_cast<Real>(element), 1.0));
    }
}

TEST(DeviceMirroredArray, AResetArrayReadsAsZerosOnTheHost) {
    MirroredArrayFixture fixture;

    fixture.array.reset(4);

    EXPECT_EQ(fixture.array.size(), 4u);
    EXPECT_FALSE(fixture.array.host_is_allocated());
    const Complex* elements = fixture.array.host();
    for (std::size_t element = 0; element < 4; ++element) {
        EXPECT_EQ(elements[element], Complex(0.0, 0.0));
    }
    EXPECT_TRUE(fixture.array.host_is_allocated());
}

TEST(DeviceMirroredArray, AResetArrayReadsAsZerosOnTheDeviceWithoutAHostCopy) {
    MirroredArrayFixture fixture;
    fixture.array.reset(number_of_elements);

    const Complex* device_elements = fixture.array.device(fixture.device_backend);

    for (std::size_t element = 0; element < number_of_elements; ++element) {
        EXPECT_EQ(device_elements[element], Complex(0.0, 0.0));
    }
    EXPECT_FALSE(fixture.array.host_is_allocated());
    EXPECT_EQ(fixture.totals().host_to_device_copies, 0u);
}

TEST(DeviceMirroredArray, ResetBringsNothingDownFromTheDevice) {
    MirroredArrayFixture fixture;
    (void)fixture.array.mutable_device(fixture.device_backend);
    const DeviceTransferTotals before = fixture.totals();

    fixture.array.reset(number_of_elements);

    EXPECT_EQ(fixture.totals().since(before).device_to_host_copies, 0u);
    EXPECT_FALSE(fixture.array.device_is_allocated());
    EXPECT_FALSE(fixture.array.device_is_current());
}

TEST(DeviceMirroredArray, ResetToNothingIsAnEmptyArray) {
    MirroredArrayFixture fixture;
    (void)fixture.array.device(fixture.device_backend);

    fixture.array.reset(0);

    EXPECT_EQ(fixture.array.size(), 0u);
    EXPECT_TRUE(fixture.array.host_is_current());
    EXPECT_FALSE(fixture.array.device_is_allocated());
}

TEST(DeviceMirroredArray, AHostWriteAfterAResetStartsFromZeros) {
    MirroredArrayFixture fixture;
    fixture.array.reset(number_of_elements);

    Complex* elements = fixture.array.mutable_host();
    elements[2] = Complex(5.0, 0.0);

    EXPECT_EQ(fixture.array.host()[0], Complex(0.0, 0.0));
    EXPECT_EQ(fixture.array.host()[2], Complex(5.0, 0.0));
    EXPECT_EQ(fixture.array.device(fixture.device_backend)[2], Complex(5.0, 0.0));
}

TEST(DeviceMirroredArray, ResizingAnArrayThatHoldsNothingOnlyChangesItsSize) {
    DeviceMirroredArray<Complex> array;
    array.reset(number_of_elements);

    array.resize(2 * number_of_elements);

    EXPECT_EQ(array.size(), 2 * number_of_elements);
    EXPECT_FALSE(array.host_is_allocated());
    EXPECT_EQ(array.host()[2 * number_of_elements - 1], Complex(0.0, 0.0));
}

TEST(DeviceMirroredArray, ACopyOfADeviceOnlyArrayHoldsItsValuesOnTheHost) {
    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    DeviceMirroredArray<Complex> array;
    array.reset(number_of_elements);
    Complex* device_elements = array.overwrite_device(device_backend);
    for (std::size_t element = 0; element < number_of_elements; ++element) {
        device_elements[element] = Complex(2.0, static_cast<Real>(element));
    }

    DeviceMirroredArray<Complex> copy(array);

    EXPECT_EQ(copy.size(), number_of_elements);
    EXPECT_TRUE(copy.host_is_current());
    EXPECT_FALSE(copy.device_is_allocated());
    EXPECT_EQ(copy.host()[3], Complex(2.0, 3.0));
}
