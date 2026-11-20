#include <gtest/gtest.h>

#include <vector>

#include "layer_4_computation/diagonalizer/planewave/internal/overlap/overlap_block_gpu_copied.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"

using namespace q3;
using test_support::CPUBackedGPUDeviceBackend;

TEST(CopiedGPUOverlapBlock, FillingCopiesTheVectors) {
    CPUBackedGPUDeviceBackend device_backend;
    std::vector<Complex> vectors = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    std::vector<Complex> overlap_vectors(vectors.size(), Complex(9.0, 9.0));
    CopiedGPUOverlapBlock block(
        device_backend, vectors.size(), vectors.data(), overlap_vectors.data());

    block.fill_from_vectors();

    EXPECT_EQ(overlap_vectors, vectors);
}

TEST(CopiedGPUOverlapBlock, FollowingCopiesTheUpdatedVectorsAndDoesNotRunTheUpdate) {
    CPUBackedGPUDeviceBackend device_backend;
    std::vector<Complex> vectors = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<Complex> overlap_vectors(vectors.size(), Complex(9.0, 9.0));
    CopiedGPUOverlapBlock block(
        device_backend, vectors.size(), vectors.data(), overlap_vectors.data());

    vectors[1] = Complex(-7.0, 0.5);
    int updates_run = 0;
    block.follow([&](Complex*) { ++updates_run; });

    EXPECT_EQ(updates_run, 0);
    EXPECT_EQ(overlap_vectors, vectors);
}

TEST(CopiedGPUOverlapBlock, ABlockThatIsTheVectorsThemselvesIsLeftAlone) {
    CPUBackedGPUDeviceBackend device_backend;
    std::vector<Complex> vectors = {{1.0, 2.0}, {3.0, 4.0}};
    const std::vector<Complex> before = vectors;
    CopiedGPUOverlapBlock block(device_backend, vectors.size(), vectors.data(), vectors.data());

    block.fill_from_vectors();
    block.follow([](Complex*) {});

    EXPECT_EQ(vectors, before);
}
