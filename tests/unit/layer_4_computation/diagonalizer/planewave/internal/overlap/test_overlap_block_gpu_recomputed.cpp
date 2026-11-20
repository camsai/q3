#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "layer_4_computation/diagonalizer/planewave/internal/overlap/overlap_block_gpu_recomputed.hpp"

using namespace q3;

namespace {

/// S = 3 on a block of two elements.
void triple(const Complex* vectors, Complex* overlap_vectors) {
    for (std::size_t index = 0; index < 2; ++index) {
        overlap_vectors[index] = 3.0 * vectors[index];
    }
}

} // namespace

TEST(RecomputedGPUOverlapBlock, FillingAppliesTheOperator) {
    std::vector<Complex> vectors = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<Complex> overlap_vectors(2, Complex(9.0, 9.0));
    RecomputedGPUOverlapBlock block(triple, vectors.data(), overlap_vectors.data());

    block.fill_from_vectors();

    EXPECT_EQ(overlap_vectors[0], Complex(3.0, 6.0));
    EXPECT_EQ(overlap_vectors[1], Complex(9.0, 12.0));
}

TEST(RecomputedGPUOverlapBlock, FollowingRunsTheUpdateOnTheOverlapBlockOnly) {
    std::vector<Complex> vectors = {{1.0, 2.0}, {3.0, 4.0}};
    const std::vector<Complex> vectors_before = vectors;
    std::vector<Complex> overlap_vectors = {{3.0, 6.0}, {9.0, 12.0}};
    RecomputedGPUOverlapBlock block(triple, vectors.data(), overlap_vectors.data());

    Complex* updated_block = nullptr;
    block.follow([&](Complex* given) {
        updated_block = given;
        given[0] *= 2.0;
    });

    EXPECT_EQ(updated_block, overlap_vectors.data());
    EXPECT_EQ(overlap_vectors[0], Complex(6.0, 12.0));
    EXPECT_EQ(vectors, vectors_before);
}
