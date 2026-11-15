#pragma once

/**
 * FFT grid distributed by sticks in reciprocal space and by planes in real
 * space, with the transpose between them owned here rather than by FFTW.
 *
 * A stick is the set of grid points sharing the first two indices and running
 * the whole third dimension. Reciprocal-space data lives as whole sticks,
 * assigned to ranks by FFTStickPartition -- mirror pairs together, balanced by
 * weight. Real-space data lives as whole planes perpendicular to the third
 * dimension, a contiguous range per rank. A transform is three steps:
 *
 *   inverse:  1D transforms along the third dimension on local sticks,
 *             an all-to-all that turns sticks into planes,
 *             2D transforms on local planes.
 *   forward:  the same three in reverse.
 *
 * Sticks that carry no G-vectors inside the cutoff are never transported.
 * They are zero on the way in and dropped on the way out, which is the
 * truncation every plane-wave code applies at the cutoff sphere.
 *
 * The two layouts are the reason this class exists. FFTW-MPI's slab layout
 * puts a plane and its mirror on different ranks, which is what forced the
 * gamma-only path off under MPI; here a stick and its mirror always share a
 * rank. It is also what lets the split balance: measured on a 72^3 grid, the
 * slab split was uneven by 12.9x across four ranks and left ranks empty at
 * eight, where this one is exactly balanced at every width tested.
 *
 * Consequences for callers, stated once:
 *   - input_buffer() is the stick layout, sized local_g_space_points();
 *     output_buffer() is the plane layout, sized local_grid_points(). They
 *     are different buffers of different sizes.
 *   - g_vector_to_fft_grid_index() indexes the stick layout, or -1 when the
 *     stick is not held here.
 *   - The pointer forms forward(data)/inverse(data) switch layout in place:
 *     the caller's buffer must hold the larger of the two counts, and after
 *     inverse() it is in plane layout, after forward() in stick layout.
 */

#include <fftw3.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/stick_partition.hpp"

namespace q3 {

class FFTGridSticks : public FFTGrid {
  public:
    explicit FFTGridSticks(MPI_Comm communicator) : communicator_(communicator) {
        MPI_Comm_rank(communicator_, &rank_);
        MPI_Comm_size(communicator_, &size_);
    }

    ~FFTGridSticks() override {
        destroy_plans();
    }

    FFTGridSticks(const FFTGridSticks&) = delete;
    FFTGridSticks& operator=(const FFTGridSticks&) = delete;

    /**
     * Size the grid and, until told otherwise, treat every stick as occupied.
     * assign_sticks() narrows that to the sticks inside a cutoff once the
     * basis knows which those are.
     */
    void init(int dimension_1,
              int dimension_2,
              int dimension_3,
              bool use_measured_planning = false) override {
        grid_dimension_1_ = dimension_1;
        grid_dimension_2_ = dimension_2;
        grid_dimension_3_ = dimension_3;
        use_measured_planning_ = use_measured_planning;
        total_grid_points_ = dimension_1 * dimension_2 * dimension_3;

        /**
         * Planes along the third dimension, contiguous ranges, the remainder
         * spread one plane at a time over the lowest ranks.
         */
        int base = dimension_3 / size_;
        int remainder = dimension_3 % size_;
        plane_counts_.assign(static_cast<std::size_t>(size_), base);
        plane_offsets_.assign(static_cast<std::size_t>(size_), 0);
        for (int rank = 0; rank < size_; ++rank) {
            if (rank < remainder) {
                plane_counts_[static_cast<std::size_t>(rank)] += 1;
            }
            plane_offsets_[static_cast<std::size_t>(rank)] =
                rank == 0 ? 0
                          : plane_offsets_[static_cast<std::size_t>(rank - 1)] +
                                plane_counts_[static_cast<std::size_t>(rank - 1)];
        }
        local_planes_ = plane_counts_[static_cast<std::size_t>(rank_)];
        local_plane_offset_ = plane_offsets_[static_cast<std::size_t>(rank_)];
        local_grid_points_ = local_planes_ * dimension_1 * dimension_2;

        std::vector<int> all_occupied(static_cast<std::size_t>(dimension_1) * dimension_2, 1);
        assign_sticks(all_occupied);
    }

    /**
     * Assign sticks to ranks from a weight per stick -- the number of
     * G-vectors on it inside the cutoff, zero for an empty stick -- and rebuild
     * the communication plan and the FFTW plans to match.
     */
    void assign_sticks(const std::vector<int>& stick_weights) override {
        assign_sticks(stick_weights, stick_weights);
    }

    /**
     * Ownership follows `stick_weights`; `wavefunction_stick_weights` says which owned sticks
     * can hold a wavefunction coefficient. Those are listed first on every rank, so the
     * wavefunction transforms work on a prefix of each rank's sticks: one batched 1D plan over
     * the prefix, and an all-to-all with the same displacements and smaller counts.
     */
    void assign_sticks(const std::vector<int>& stick_weights,
                       const std::vector<int>& wavefunction_stick_weights) override {
        partition_.build(grid_dimension_1_, grid_dimension_2_, stick_weights, size_);

        /**
         * Every rank enumerates every rank's sticks in the same order, so the
         * transpose can be unpacked without exchanging index lists: first the
         * sticks a wavefunction can reach, then the rest.
         */
        sticks_by_rank_.assign(static_cast<std::size_t>(size_), {});
        wavefunction_sticks_by_rank_.assign(static_cast<std::size_t>(size_), 0);
        stick_local_index_.assign(static_cast<std::size_t>(grid_dimension_1_) * grid_dimension_2_,
                                  -1);
        for (int pass = 0; pass < 2; ++pass) {
            const bool listing_wavefunction_sticks = (pass == 0);
            for (int first = 0; first < grid_dimension_1_; ++first) {
                for (int second = 0; second < grid_dimension_2_; ++second) {
                    int owner = partition_.owner_of(first, second);
                    if (owner == FFTStickPartition::unassigned) {
                        continue;
                    }
                    std::size_t stick =
                        static_cast<std::size_t>(first) * grid_dimension_2_ + second;
                    /**
                     * A stick carries if it or its mirror does. A gamma-only basis lists half
                     * the sphere, so only one stick of a mirror pair gets a count -- but the
                     * transform writes conj(psi(G)) onto the other, so both must be moved.
                     */
                    std::size_t mirror =
                        static_cast<std::size_t>((grid_dimension_1_ - first) % grid_dimension_1_) *
                            grid_dimension_2_ +
                        (grid_dimension_2_ - second) % grid_dimension_2_;
                    bool carries_wavefunctions = wavefunction_stick_weights[stick] > 0 ||
                                                 wavefunction_stick_weights[mirror] > 0;
                    if (carries_wavefunctions != listing_wavefunction_sticks) {
                        continue;
                    }
                    auto& list = sticks_by_rank_[static_cast<std::size_t>(owner)];
                    if (owner == rank_) {
                        stick_local_index_[stick] = static_cast<int>(list.size());
                    }
                    list.push_back(first * grid_dimension_2_ + second);
                    if (listing_wavefunction_sticks) {
                        wavefunction_sticks_by_rank_[static_cast<std::size_t>(owner)]++;
                    }
                }
            }
        }
        local_sticks_ = static_cast<int>(sticks_by_rank_[static_cast<std::size_t>(rank_)].size());
        local_wavefunction_sticks_ = wavefunction_sticks_by_rank_[static_cast<std::size_t>(rank_)];
        local_g_space_points_ = local_sticks_ * grid_dimension_3_;

        /**
         * All-to-all counts. Rank r sends, to rank q, the plane range of q for
         * each of r's sticks; it receives from q the plane range of r for each
         * of q's sticks.
         */
        send_counts_.assign(static_cast<std::size_t>(size_), 0);
        recv_counts_.assign(static_cast<std::size_t>(size_), 0);
        send_displacements_.assign(static_cast<std::size_t>(size_), 0);
        recv_displacements_.assign(static_cast<std::size_t>(size_), 0);
        for (int other = 0; other < size_; ++other) {
            send_counts_[static_cast<std::size_t>(other)] =
                local_sticks_ * plane_counts_[static_cast<std::size_t>(other)];
            recv_counts_[static_cast<std::size_t>(other)] =
                static_cast<int>(sticks_by_rank_[static_cast<std::size_t>(other)].size()) *
                local_planes_;
        }
        wavefunction_send_counts_.assign(static_cast<std::size_t>(size_), 0);
        wavefunction_recv_counts_.assign(static_cast<std::size_t>(size_), 0);
        for (int other = 0; other < size_; ++other) {
            wavefunction_send_counts_[static_cast<std::size_t>(other)] =
                local_wavefunction_sticks_ * plane_counts_[static_cast<std::size_t>(other)];
            wavefunction_recv_counts_[static_cast<std::size_t>(other)] =
                wavefunction_sticks_by_rank_[static_cast<std::size_t>(other)] * local_planes_;
        }
        for (int other = 1; other < size_; ++other) {
            send_displacements_[static_cast<std::size_t>(other)] =
                send_displacements_[static_cast<std::size_t>(other - 1)] +
                send_counts_[static_cast<std::size_t>(other - 1)];
            recv_displacements_[static_cast<std::size_t>(other)] =
                recv_displacements_[static_cast<std::size_t>(other - 1)] +
                recv_counts_[static_cast<std::size_t>(other - 1)];
        }
        int transpose_size = 0;
        for (int other = 0; other < size_; ++other) {
            transpose_size = std::max(transpose_size,
                                      send_displacements_[static_cast<std::size_t>(other)] +
                                          send_counts_[static_cast<std::size_t>(other)]);
            transpose_size = std::max(transpose_size,
                                      recv_displacements_[static_cast<std::size_t>(other)] +
                                          recv_counts_[static_cast<std::size_t>(other)]);
        }

        std::size_t larger =
            static_cast<std::size_t>(std::max(local_g_space_points_, local_grid_points_));
        stick_buffer_.assign(std::max<std::size_t>(larger, 1), Complex(0.0, 0.0));
        plane_buffer_.assign(std::max<std::size_t>(larger, 1), Complex(0.0, 0.0));
        transpose_buffer_.assign(std::max<std::size_t>(static_cast<std::size_t>(transpose_size), 1),
                                 Complex(0.0, 0.0));

        /**
         * Planned at the first transform, not here: init() lays out every stick and the
         * basis reassigns them straight after, so planning eagerly measured each grid's
         * plans twice -- the first set never used. FFTW_MEASURE times real transforms, on
         * every rank at once, which made planning a second of setup on a full node.
         */
        destroy_plans();
        plans_current_ = false;
    }

    [[nodiscard]] int grid_dimension_1() const override {
        return grid_dimension_1_;
    }
    [[nodiscard]] int grid_dimension_2() const override {
        return grid_dimension_2_;
    }
    [[nodiscard]] int grid_dimension_3() const override {
        return grid_dimension_3_;
    }
    [[nodiscard]] int total_grid_points() const override {
        return total_grid_points_;
    }
    [[nodiscard]] int local_grid_points() const override {
        return local_grid_points_;
    }
    [[nodiscard]] int local_g_space_points() const override {
        return local_g_space_points_;
    }
    /// First plane along the third dimension held here.
    [[nodiscard]] int local_offset() const override {
        return local_plane_offset_;
    }
    /// Local real space is [plane][i1][i2], the plane counted from local_offset().
    [[nodiscard]] int global_real_space_index(int local_index) const override {
        int points_per_plane = grid_dimension_1_ * grid_dimension_2_;
        int index_3 = local_plane_offset_ + local_index / points_per_plane;
        int index_within_plane = local_index % points_per_plane;
        int index_1 = index_within_plane / grid_dimension_2_;
        int index_2 = index_within_plane % grid_dimension_2_;
        return (index_1 * grid_dimension_2_ + index_2) * grid_dimension_3_ + index_3;
    }
    /// No restriction along the first dimension: ownership is by stick.
    [[nodiscard]] int slab_dimension_1() const override {
        return grid_dimension_1_;
    }
    [[nodiscard]] int local_sticks() const {
        return local_sticks_;
    }
    [[nodiscard]] const FFTStickPartition& partition() const {
        return partition_;
    }

    [[nodiscard]] bool co_locates_mirrors() const override {
        return true;
    }

    void gather_local_g_space(const Complex* full_g_space, Complex* local_g_space) const override {
        const auto& mine = sticks_by_rank_[static_cast<std::size_t>(rank_)];
        for (std::size_t stick = 0; stick < mine.size(); ++stick) {
            std::memcpy(local_g_space + stick * static_cast<std::size_t>(grid_dimension_3_),
                        full_g_space + static_cast<std::size_t>(mine[stick]) * grid_dimension_3_,
                        static_cast<std::size_t>(grid_dimension_3_) * sizeof(Complex));
        }
    }

    void scatter_local_g_space(const Complex* local_g_space, Complex* full_g_space) const override {
        const auto& mine = sticks_by_rank_[static_cast<std::size_t>(rank_)];
        for (std::size_t stick = 0; stick < mine.size(); ++stick) {
            std::memcpy(full_g_space + static_cast<std::size_t>(mine[stick]) * grid_dimension_3_,
                        local_g_space + stick * static_cast<std::size_t>(grid_dimension_3_),
                        static_cast<std::size_t>(grid_dimension_3_) * sizeof(Complex));
        }
    }

    [[nodiscard]] bool owns_stick(int miller_1, int miller_2) const override {
        return partition_.owner_of_miller(miller_1, miller_2) == rank_;
    }

    [[nodiscard]] int
    g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const override {
        int wrapped_1 = FFTStickPartition::wrap(miller_1, grid_dimension_1_);
        int wrapped_2 = FFTStickPartition::wrap(miller_2, grid_dimension_2_);
        int wrapped_3 = FFTStickPartition::wrap(miller_3, grid_dimension_3_);
        int local_stick =
            stick_local_index_[static_cast<std::size_t>(wrapped_1) * grid_dimension_2_ + wrapped_2];
        if (local_stick < 0) {
            return -1;
        }
        return local_stick * grid_dimension_3_ + wrapped_3;
    }

    // ─── Zero-copy buffer API ──────────────────────────────────────────

    Complex* input_buffer() const override {
        return stick_buffer_.data();
    }
    Complex* output_buffer() const override {
        return plane_buffer_.data();
    }

    void execute_inverse_raw_on_buffers() const override {
        sticks_to_planes(stick_buffer_.data(), plane_buffer_.data());
    }

    /**
     * The interface promises the result of any transform in output_buffer().
     * The forward transform lands in the stick buffer, so it is copied across
     * -- one memcpy of the G-space count, the same order as the transpose pack
     * -- rather than making every consumer learn which buffer to read.
     */
    void execute_forward_on_buffers() const override {
        planes_to_sticks(plane_buffer_.data(), stick_buffer_.data());
        std::memcpy(plane_buffer_.data(),
                    stick_buffer_.data(),
                    static_cast<std::size_t>(local_g_space_points_) * sizeof(Complex));
    }

    /// The forward transform consumes the plane buffer directly.
    void prepare_forward_from_output() const override {}

    // ─── Wavefunction transforms: the sticks a wavefunction can reach ──

    /// Inverse transform of the stick buffer, moving only the wavefunction sticks.
    void execute_wavefunction_inverse_on_slot(int /*slot*/) const override {
        sticks_to_planes(stick_buffer_.data(), plane_buffer_.data(), true);
    }

    /// Only the wavefunction sticks are computed, so only they are copied to the output.
    void execute_wavefunction_forward_on_slot(int /*slot*/) const override {
        planes_to_sticks(plane_buffer_.data(), stick_buffer_.data(), true);
        std::memcpy(plane_buffer_.data(),
                    stick_buffer_.data(),
                    static_cast<std::size_t>(local_wavefunction_sticks_) * grid_dimension_3_ *
                        sizeof(Complex));
    }

    /// Slot form of wavefunction_inverse_raw; this grid has one buffer set.
    void wavefunction_inverse_raw_on_slot(Complex* data, int /*slot*/) const override {
        wavefunction_inverse_raw(data);
    }

    /// Inverse transform of stick-layout data to local planes, over the wavefunction sticks.
    void wavefunction_inverse_raw(Complex* data) const override {
        std::memcpy(stick_buffer_.data(),
                    data,
                    static_cast<std::size_t>(local_wavefunction_sticks_) * grid_dimension_3_ *
                        sizeof(Complex));
        sticks_to_planes(stick_buffer_.data(), plane_buffer_.data(), true);
        std::memcpy(data, plane_buffer_.data(), local_grid_points_ * sizeof(Complex));
    }

    /// Forward transform of local planes to sticks, filling only the wavefunction sticks.
    void wavefunction_forward(Complex* data) const override {
        std::memcpy(plane_buffer_.data(), data, local_grid_points_ * sizeof(Complex));
        planes_to_sticks(plane_buffer_.data(), stick_buffer_.data(), true);
        std::memcpy(data,
                    stick_buffer_.data(),
                    static_cast<std::size_t>(local_wavefunction_sticks_) * grid_dimension_3_ *
                        sizeof(Complex));
    }

    /// Cap FFTW's plan search for this grid's plans; applied when the plans are next made.
    void limit_planning_time(double seconds_per_plan) override {
        planning_seconds_per_plan_ = seconds_per_plan;
    }

    /// Sticks on this rank that can hold a wavefunction coefficient; they come first.
    [[nodiscard]] int local_wavefunction_sticks() const {
        return local_wavefunction_sticks_;
    }

    // ─── Pointer forms ─────────────────────────────────────────────────

    void forward(Complex* data) const override {
        std::memcpy(plane_buffer_.data(), data, local_grid_points_ * sizeof(Complex));
        planes_to_sticks(plane_buffer_.data(), stick_buffer_.data());
        std::memcpy(data, stick_buffer_.data(), local_g_space_points_ * sizeof(Complex));
    }

    void inverse_raw(Complex* data) const override {
        std::memcpy(stick_buffer_.data(), data, local_g_space_points_ * sizeof(Complex));
        sticks_to_planes(stick_buffer_.data(), plane_buffer_.data());
        std::memcpy(data, plane_buffer_.data(), local_grid_points_ * sizeof(Complex));
    }

    void inverse(Complex* data) const override {
        inverse_raw(data);
        Real scale = 1.0 / static_cast<Real>(total_grid_points_);
        for (int index = 0; index < local_grid_points_; ++index) {
            data[index] *= scale;
        }
    }

    void reduce_g_space(Complex* data, int number_of_elements) const override {
        MPI_Allreduce(
            MPI_IN_PLACE, data, number_of_elements, MPI_C_DOUBLE_COMPLEX, MPI_SUM, communicator_);
    }

  private:
    // ─── The transform ─────────────────────────────────────────────────

    /**
     * Sticks to planes: 1D along the third dimension on each local stick, then
     * the all-to-all, then 2D on each local plane. The all-to-all sends, for
     * each destination, its plane range of every local stick, which is a
     * contiguous run of the stick.
     */
    void sticks_to_planes(Complex* sticks, Complex* planes, bool wavefunction_only = false) const {
        ensure_plans();
        // A wavefunction is zero off its sticks: they are a prefix, transformed and sent alone.
        const int sticks_to_send = wavefunction_only ? local_wavefunction_sticks_ : local_sticks_;
        const std::vector<int>& send_counts =
            wavefunction_only ? wavefunction_send_counts_ : send_counts_;
        const std::vector<int>& recv_counts =
            wavefunction_only ? wavefunction_recv_counts_ : recv_counts_;
        if (sticks_to_send > 0) {
            fftw_execute_dft(wavefunction_only ? plan_wavefunction_sticks_inverse_
                                               : plan_sticks_inverse_,
                             reinterpret_cast<fftw_complex*>(sticks),
                             reinterpret_cast<fftw_complex*>(sticks));
        }

        Complex* send = transpose_buffer_.data();
        for (int other = 0; other < size_; ++other) {
            int plane_start = plane_offsets_[static_cast<std::size_t>(other)];
            int plane_count = plane_counts_[static_cast<std::size_t>(other)];
            Complex* destination = send + send_displacements_[static_cast<std::size_t>(other)];
            for (int stick = 0; stick < sticks_to_send; ++stick) {
                std::memcpy(destination + static_cast<std::size_t>(stick) * plane_count,
                            sticks + static_cast<std::size_t>(stick) * grid_dimension_3_ +
                                plane_start,
                            static_cast<std::size_t>(plane_count) * sizeof(Complex));
            }
        }

        std::vector<Complex>& receive = receive_buffer_;
        receive.resize(std::max<std::size_t>(
            static_cast<std::size_t>(recv_displacements_.back() + recv_counts_.back()), 1));
        MPI_Alltoallv(send,
                      send_counts.data(),
                      send_displacements_.data(),
                      MPI_C_DOUBLE_COMPLEX,
                      receive.data(),
                      recv_counts.data(),
                      recv_displacements_.data(),
                      MPI_C_DOUBLE_COMPLEX,
                      communicator_);

        /**
         * Unpack into dense planes. Sticks nobody owns carry nothing and stay
         * zero, which is the cutoff truncation.
         */
        std::fill(planes, planes + local_grid_points_, Complex(0.0, 0.0));
        std::size_t plane_stride = static_cast<std::size_t>(grid_dimension_1_) * grid_dimension_2_;
        for (int other = 0; other < size_; ++other) {
            const auto& their_sticks = sticks_by_rank_[static_cast<std::size_t>(other)];
            const Complex* source =
                receive.data() + recv_displacements_[static_cast<std::size_t>(other)];
            const std::size_t their_count =
                wavefunction_only
                    ? static_cast<std::size_t>(
                          wavefunction_sticks_by_rank_[static_cast<std::size_t>(other)])
                    : their_sticks.size();
            for (std::size_t stick = 0; stick < their_count; ++stick) {
                int position = their_sticks[stick]; // first * dimension_2 + second
                for (int plane = 0; plane < local_planes_; ++plane) {
                    planes[static_cast<std::size_t>(plane) * plane_stride + position] =
                        source[stick * static_cast<std::size_t>(local_planes_) + plane];
                }
            }
        }

        if (local_planes_ > 0) {
            fftw_execute_dft(plan_planes_inverse_,
                             reinterpret_cast<fftw_complex*>(planes),
                             reinterpret_cast<fftw_complex*>(planes));
        }
    }

    /// Planes to sticks: the same three steps, reversed.
    void planes_to_sticks(Complex* planes, Complex* sticks, bool wavefunction_only = false) const {
        ensure_plans();
        // Only the wavefunction sticks are wanted back: they are packed, sent and transformed.
        const int sticks_to_receive =
            wavefunction_only ? local_wavefunction_sticks_ : local_sticks_;
        const std::vector<int>& send_counts =
            wavefunction_only ? wavefunction_send_counts_ : send_counts_;
        const std::vector<int>& recv_counts =
            wavefunction_only ? wavefunction_recv_counts_ : recv_counts_;
        if (local_planes_ > 0) {
            fftw_execute_dft(plan_planes_forward_,
                             reinterpret_cast<fftw_complex*>(planes),
                             reinterpret_cast<fftw_complex*>(planes));
        }

        /**
         * Pack occupied sticks out of the dense planes, in the layout the
         * receiving side unpacks from above: for each destination rank, its
         * sticks in its order, and for each stick this rank's planes.
         */
        std::vector<Complex>& send = receive_buffer_;
        send.resize(std::max<std::size_t>(
            static_cast<std::size_t>(recv_displacements_.back() + recv_counts_.back()), 1));
        std::size_t plane_stride = static_cast<std::size_t>(grid_dimension_1_) * grid_dimension_2_;
        for (int other = 0; other < size_; ++other) {
            const auto& their_sticks = sticks_by_rank_[static_cast<std::size_t>(other)];
            Complex* destination =
                send.data() + recv_displacements_[static_cast<std::size_t>(other)];
            const std::size_t their_count =
                wavefunction_only
                    ? static_cast<std::size_t>(
                          wavefunction_sticks_by_rank_[static_cast<std::size_t>(other)])
                    : their_sticks.size();
            for (std::size_t stick = 0; stick < their_count; ++stick) {
                int position = their_sticks[stick];
                for (int plane = 0; plane < local_planes_; ++plane) {
                    destination[stick * static_cast<std::size_t>(local_planes_) + plane] =
                        planes[static_cast<std::size_t>(plane) * plane_stride + position];
                }
            }
        }

        Complex* receive = transpose_buffer_.data();
        MPI_Alltoallv(send.data(),
                      recv_counts.data(),
                      recv_displacements_.data(),
                      MPI_C_DOUBLE_COMPLEX,
                      receive,
                      send_counts.data(),
                      send_displacements_.data(),
                      MPI_C_DOUBLE_COMPLEX,
                      communicator_);

        for (int other = 0; other < size_; ++other) {
            int plane_start = plane_offsets_[static_cast<std::size_t>(other)];
            int plane_count = plane_counts_[static_cast<std::size_t>(other)];
            const Complex* source = receive + send_displacements_[static_cast<std::size_t>(other)];
            for (int stick = 0; stick < sticks_to_receive; ++stick) {
                std::memcpy(sticks + static_cast<std::size_t>(stick) * grid_dimension_3_ +
                                plane_start,
                            source + static_cast<std::size_t>(stick) * plane_count,
                            static_cast<std::size_t>(plane_count) * sizeof(Complex));
            }
        }

        if (sticks_to_receive > 0) {
            fftw_execute_dft(wavefunction_only ? plan_wavefunction_sticks_forward_
                                               : plan_sticks_forward_,
                             reinterpret_cast<fftw_complex*>(sticks),
                             reinterpret_cast<fftw_complex*>(sticks));
        }
    }

    // ─── Plans ─────────────────────────────────────────────────────────

    /**
     * Plan on first use. FFTW_MEASURE overwrites the arrays it plans on, and by now a caller
     * may have written its input into them (input_buffer() is the stick buffer), so their
     * contents are saved and put back. Plans are made on these same arrays because they are
     * the ones every transform executes on.
     */
    void ensure_plans() const {
        if (plans_current_) {
            return;
        }
        const std::vector<Complex> saved_sticks = stick_buffer_;
        const std::vector<Complex> saved_planes = plane_buffer_;
        create_plans();
        stick_buffer_ = saved_sticks;
        plane_buffer_ = saved_planes;
        plans_current_ = true;
    }

    void create_plans() const {
        destroy_plans();
        unsigned flags = use_measured_planning_ ? FFTW_MEASURE : FFTW_ESTIMATE;
        /**
         * Measured planning is capped where it does not pay. The full-stick plans serve
         * density transforms -- a few per iteration even on the wavefunction grid -- and are
         * the costliest to measure, every stick in one batch: capped on every grid. The plane
         * and wavefunction-stick plans carry every band transform, so they keep the exhaustive
         * search unless the grid was told otherwise (limit_planning_time()). Measured on a full
         * 128-rank node, the uncapped full-stick plans were most of a second of setup, and
         * capping the band plans as well cost more in the solver than it saved.
         */
        auto cap_planning = [this](double seconds_per_plan) {
            if (use_measured_planning_) {
                fftw_set_timelimit(seconds_per_plan > 0.0 ? seconds_per_plan : FFTW_NO_TIMELIMIT);
            }
        };
        cap_planning(full_stick_planning_seconds_per_plan);
        if (local_sticks_ > 0) {
            int length = grid_dimension_3_;
            auto* buffer = reinterpret_cast<fftw_complex*>(stick_buffer_.data());
            plan_sticks_forward_ = fftw_plan_many_dft(1,
                                                      &length,
                                                      local_sticks_,
                                                      buffer,
                                                      nullptr,
                                                      1,
                                                      length,
                                                      buffer,
                                                      nullptr,
                                                      1,
                                                      length,
                                                      FFTW_FORWARD,
                                                      flags);
            plan_sticks_inverse_ = fftw_plan_many_dft(1,
                                                      &length,
                                                      local_sticks_,
                                                      buffer,
                                                      nullptr,
                                                      1,
                                                      length,
                                                      buffer,
                                                      nullptr,
                                                      1,
                                                      length,
                                                      FFTW_BACKWARD,
                                                      flags);
        }
        cap_planning(planning_seconds_per_plan_);
        if (local_wavefunction_sticks_ > 0) {
            int length = grid_dimension_3_;
            auto* buffer = reinterpret_cast<fftw_complex*>(stick_buffer_.data());
            for (int direction : {FFTW_FORWARD, FFTW_BACKWARD}) {
                fftw_plan plan = fftw_plan_many_dft(1,
                                                    &length,
                                                    local_wavefunction_sticks_,
                                                    buffer,
                                                    nullptr,
                                                    1,
                                                    length,
                                                    buffer,
                                                    nullptr,
                                                    1,
                                                    length,
                                                    direction,
                                                    flags);
                (direction == FFTW_FORWARD ? plan_wavefunction_sticks_forward_
                                           : plan_wavefunction_sticks_inverse_) = plan;
            }
        }
        if (local_planes_ > 0) {
            int lengths[2] = {grid_dimension_1_, grid_dimension_2_};
            int plane_points = grid_dimension_1_ * grid_dimension_2_;
            auto* buffer = reinterpret_cast<fftw_complex*>(plane_buffer_.data());
            plan_planes_forward_ = fftw_plan_many_dft(2,
                                                      lengths,
                                                      local_planes_,
                                                      buffer,
                                                      nullptr,
                                                      1,
                                                      plane_points,
                                                      buffer,
                                                      nullptr,
                                                      1,
                                                      plane_points,
                                                      FFTW_FORWARD,
                                                      flags);
            plan_planes_inverse_ = fftw_plan_many_dft(2,
                                                      lengths,
                                                      local_planes_,
                                                      buffer,
                                                      nullptr,
                                                      1,
                                                      plane_points,
                                                      buffer,
                                                      nullptr,
                                                      1,
                                                      plane_points,
                                                      FFTW_BACKWARD,
                                                      flags);
        }
        fftw_set_timelimit(FFTW_NO_TIMELIMIT);
    }

    void destroy_plans() const {
        for (fftw_plan* plan : {&plan_sticks_forward_,
                                &plan_sticks_inverse_,
                                &plan_wavefunction_sticks_forward_,
                                &plan_wavefunction_sticks_inverse_,
                                &plan_planes_forward_,
                                &plan_planes_inverse_}) {
            if (*plan != nullptr) {
                fftw_destroy_plan(*plan);
                *plan = nullptr;
            }
        }
    }

    MPI_Comm communicator_;
    int rank_ = 0;
    int size_ = 1;

    int grid_dimension_1_ = 0;
    int grid_dimension_2_ = 0;
    int grid_dimension_3_ = 0;
    int total_grid_points_ = 0;
    bool use_measured_planning_ = false;

    std::vector<int> plane_counts_;
    std::vector<int> plane_offsets_;
    int local_planes_ = 0;
    int local_plane_offset_ = 0;
    int local_grid_points_ = 0;

    FFTStickPartition partition_;
    std::vector<std::vector<int>> sticks_by_rank_;
    /// How many of each rank's sticks, the leading ones, can hold a wavefunction coefficient.
    std::vector<int> wavefunction_sticks_by_rank_;
    std::vector<int> stick_local_index_;
    int local_sticks_ = 0;
    int local_wavefunction_sticks_ = 0;
    int local_g_space_points_ = 0;

    std::vector<int> send_counts_, recv_counts_, send_displacements_, recv_displacements_;
    std::vector<int> wavefunction_send_counts_, wavefunction_recv_counts_;

    mutable std::vector<Complex> stick_buffer_;
    mutable std::vector<Complex> plane_buffer_;
    mutable std::vector<Complex> transpose_buffer_;
    mutable std::vector<Complex> receive_buffer_;

    /// Planner time cap per measured band-transform plan, or 0 for none; see limit_planning_time().
    double planning_seconds_per_plan_ = 0.0;
    /// Planner time cap per full-stick plan; see create_plans().
    static constexpr double full_stick_planning_seconds_per_plan = 0.02;
    // Made lazily by ensure_plans(), hence mutable.
    mutable bool plans_current_ = false;
    mutable fftw_plan plan_sticks_forward_ = nullptr;
    mutable fftw_plan plan_sticks_inverse_ = nullptr;
    mutable fftw_plan plan_wavefunction_sticks_forward_ = nullptr;
    mutable fftw_plan plan_wavefunction_sticks_inverse_ = nullptr;
    mutable fftw_plan plan_planes_forward_ = nullptr;
    mutable fftw_plan plan_planes_inverse_ = nullptr;
};

} // namespace q3
