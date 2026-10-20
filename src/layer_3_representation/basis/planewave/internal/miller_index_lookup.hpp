#pragma once

/**
 * MillerIndexLookup — Miller (h,k,l) -> global charge-density G-vector index.
 *
 * Every plane-wave calculator that needs to find a G-vector by its integer
 * Miller triple (rather than iterate the sorted-by-magnitude list) needs
 * this reverse lookup; the Wannier overlap kernels are the first callers
 * with a hot inner loop over it (matching a k-point's plane-wave G-set
 * against its k+b neighbor's, after the periodic-image Miller shift). The
 * packing scheme generalizes the ad hoc packed-key map built inline in
 * layer_4_computation/charge_density/planewave/init.hpp -- same idea (pack
 * three small integers into one int64_t hash-map key), but the offset is
 * sized from the actual Miller range on construction instead of a fixed
 * assumed bound, so it cannot silently collide on a denser G-sphere.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vector.hpp"

namespace q3 {

class MillerIndexLookup {
  public:
    MillerIndexLookup() = default;

    /**
     * Build the lookup table from a charge-density G-vector list: index i's
     * Miller triple maps to i. `g_vectors` is typically
     * GVectorsGrid::g_vectors() -- the sphere every per-k-point plane-wave
     * set and every augmentation-function table is indexed into.
     */
    void build(const std::vector<GVector>& g_vectors) {
        index_by_key_.clear();
        index_by_key_.reserve(g_vectors.size());

        offset_ = 1;
        for (const GVector& g_vector : g_vectors) {
            offset_ = std::max({offset_,
                                std::abs(g_vector.miller.x) + 1,
                                std::abs(g_vector.miller.y) + 1,
                                std::abs(g_vector.miller.z) + 1});
        }

        for (int g_vector_index = 0; g_vector_index < static_cast<int>(g_vectors.size());
             g_vector_index++) {
            const IntegerVector3D& miller = g_vectors[g_vector_index].miller;
            index_by_key_[pack(miller.x, miller.y, miller.z)] = g_vector_index;
        }
    }

    /**
     * The global G-vector index for the given Miller triple, or -1 if it
     * falls outside the charge-density sphere this lookup was built from.
     * A miss is a legitimate outcome (a k+b plane wave can fall just
     * outside the sphere at the zone boundary) -- callers count misses
     * rather than treat them as errors.
     */
    [[nodiscard]] int find(const IntegerVector3D& miller) const {
        return find(miller.x, miller.y, miller.z);
    }

    [[nodiscard]] int find(int miller_h, int miller_k, int miller_l) const {
        if (std::abs(miller_h) >= offset_ || std::abs(miller_k) >= offset_ ||
            std::abs(miller_l) >= offset_) {
            return -1;
        }
        auto iterator = index_by_key_.find(pack(miller_h, miller_k, miller_l));
        return iterator == index_by_key_.end() ? -1 : iterator->second;
    }

    [[nodiscard]] int size() const {
        return static_cast<int>(index_by_key_.size());
    }

  private:
    [[nodiscard]] int64_t pack(int miller_h, int miller_k, int miller_l) const {
        const int64_t span = 2 * static_cast<int64_t>(offset_) + 1;
        return (static_cast<int64_t>(miller_h) + offset_) * span * span +
               (static_cast<int64_t>(miller_k) + offset_) * span +
               (static_cast<int64_t>(miller_l) + offset_);
    }

    std::unordered_map<int64_t, int> index_by_key_;
    int offset_ = 1;
};

} // namespace q3
