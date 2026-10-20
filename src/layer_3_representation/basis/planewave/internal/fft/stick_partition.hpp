#pragma once

/**
 * Assignment of FFT sticks to ranks.
 *
 * A stick is the set of grid points sharing the first two Miller indices and
 * running the whole third dimension. Distributing whole sticks, rather than
 * the whole planes FFTW-MPI hands out, buys three things the slab layout
 * cannot give:
 *
 *   - A stick and its mirror can be assigned together, which is what keeps the
 *     gamma-only reconstruction local. Under contiguous plane ranges they are
 *     almost never on the same rank: for a 108-plane grid over four ranks, 2
 *     planes of 108 have their mirror locally.
 *   - The split can be balanced. Sticks hold unequal numbers of G-vectors
 *     inside the cutoff sphere and the outer ones hold none, so dealing them
 *     longest-first evens out what each rank carries. The slab split was
 *     measured uneven by 3.8x across four ranks.
 *   - The rank count is capped by the number of sticks rather than the number
 *     of planes, which is larger by a factor of the second dimension.
 *
 * This class is only the assignment. It takes a weight per stick -- the caller
 * counts G-vectors inside its own cutoff -- and says who owns what, so it can
 * be reasoned about and tested without an FFT or an MPI communicator.
 */

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace q3 {

class FFTStickPartition {
  public:
    /// Sticks with no G-vectors inside the cutoff belong to nobody.
    static constexpr int unassigned = -1;

    /**
     * Deal sticks onto ranks, mirror pairs together, heaviest pair first.
     *
     * `stick_weights` is indexed by `wrapped_first * dimension_2 + wrapped_second`
     * and holds the number of G-vectors on that stick; zero means the stick is
     * outside the cutoff everywhere and is left unassigned.
     */
    void build(int dimension_1,
               int dimension_2,
               const std::vector<int>& stick_weights,
               int number_of_ranks) {
        dimension_1_ = dimension_1;
        dimension_2_ = dimension_2;
        number_of_ranks_ = std::max(number_of_ranks, 1);
        std::size_t stick_count = static_cast<std::size_t>(dimension_1) * dimension_2;
        owner_.assign(stick_count, unassigned);
        weight_per_rank_.assign(static_cast<std::size_t>(number_of_ranks_), 0);

        /**
         * Group sticks into mirror pairs before weighing them. A pair is
         * assigned as a unit, so that -G is always on the rank that holds G.
         * Sticks that are their own mirror -- the origin row, and the Nyquist
         * rows on even grids -- stand alone.
         */
        struct StickPair {
            int first_index;
            int second_index; // equal to first_index when self-mirrored
            long long weight;
        };
        std::vector<StickPair> pairs;
        pairs.reserve(stick_count);
        std::vector<bool> taken(stick_count, false);
        for (int first = 0; first < dimension_1; ++first) {
            for (int second = 0; second < dimension_2; ++second) {
                std::size_t index = stick_index(first, second);
                if (taken[index] || stick_weights[index] == 0) {
                    continue;
                }
                int mirror_first = (dimension_1 - first) % dimension_1;
                int mirror_second = (dimension_2 - second) % dimension_2;
                std::size_t mirror_index = stick_index(mirror_first, mirror_second);
                taken[index] = true;
                long long weight = stick_weights[index];
                if (mirror_index != index) {
                    taken[mirror_index] = true;
                    weight += stick_weights[mirror_index];
                }
                pairs.push_back({static_cast<int>(index), static_cast<int>(mirror_index), weight});
            }
        }

        /**
         * Longest-first onto the least-loaded rank. Ties break on the stick
         * index so the assignment is identical on every rank and between runs
         * -- ranks build this independently and must agree exactly.
         */
        std::sort(pairs.begin(), pairs.end(), [](const StickPair& left, const StickPair& right) {
            if (left.weight != right.weight) {
                return left.weight > right.weight;
            }
            return left.first_index < right.first_index;
        });
        for (const auto& pair : pairs) {
            int target = 0;
            for (int rank = 1; rank < number_of_ranks_; ++rank) {
                if (weight_per_rank_[static_cast<std::size_t>(rank)] <
                    weight_per_rank_[static_cast<std::size_t>(target)]) {
                    target = rank;
                }
            }
            owner_[static_cast<std::size_t>(pair.first_index)] = target;
            owner_[static_cast<std::size_t>(pair.second_index)] = target;
            weight_per_rank_[static_cast<std::size_t>(target)] += pair.weight;
        }
    }

    /// Rank owning the stick at wrapped indices, or `unassigned` if empty.
    [[nodiscard]] int owner_of(int wrapped_first, int wrapped_second) const {
        return owner_[stick_index(wrapped_first, wrapped_second)];
    }

    /// Rank owning the stick carrying this G-vector, by signed Miller indices.
    [[nodiscard]] int owner_of_miller(int miller_1, int miller_2) const {
        return owner_of(wrap(miller_1, dimension_1_), wrap(miller_2, dimension_2_));
    }

    [[nodiscard]] const std::vector<int>& owner() const {
        return owner_;
    }

    [[nodiscard]] const std::vector<long long>& weight_per_rank() const {
        return weight_per_rank_;
    }

    /// Heaviest rank's share divided by the lightest; 1.0 is a perfect split.
    [[nodiscard]] double imbalance() const {
        if (weight_per_rank_.empty()) {
            return 1.0;
        }
        auto bounds = std::minmax_element(weight_per_rank_.begin(), weight_per_rank_.end());
        if (*bounds.first <= 0) {
            return *bounds.second > 0 ? 0.0 : 1.0;
        }
        return static_cast<double>(*bounds.second) / static_cast<double>(*bounds.first);
    }

    [[nodiscard]] int number_of_ranks() const {
        return number_of_ranks_;
    }

    static int wrap(int miller, int dimension) {
        int wrapped = miller % dimension;
        return wrapped < 0 ? wrapped + dimension : wrapped;
    }

  private:
    [[nodiscard]] std::size_t stick_index(int wrapped_first, int wrapped_second) const {
        return static_cast<std::size_t>(wrapped_first) * dimension_2_ + wrapped_second;
    }

    int dimension_1_ = 0;
    int dimension_2_ = 0;
    int number_of_ranks_ = 1;
    std::vector<int> owner_;
    std::vector<long long> weight_per_rank_;
};

} // namespace q3
