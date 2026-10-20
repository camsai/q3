#pragma once

#include <cstring>
#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

/**
 * FFTGrid — Abstract interface for 3D FFT grids.
 *
 * Provides a unified contract for real <-> reciprocal space transformations,
 * shielding consumers from the details of memory distribution (serial vs. MPI).
 */
class FFTGrid {
  public:
    virtual ~FFTGrid() = default;

    /**
     * Initialize with grid dimensions (dimension_1 x dimension_2 x dimension_3).
     */
    virtual void
    init(int dimension_1, int dimension_2, int dimension_3, bool use_measured_planning = false) = 0;

    /**
     * Grid size along the first dimension.
     */
    [[nodiscard]] virtual int grid_dimension_1() const = 0;
    /**
     * Grid size along the second dimension.
     */
    [[nodiscard]] virtual int grid_dimension_2() const = 0;
    /**
     * Grid size along the third dimension.
     */
    [[nodiscard]] virtual int grid_dimension_3() const = 0;

    /**
     * Total number of real-space grid points across all processes.
     */
    [[nodiscard]] virtual int total_grid_points() const = 0;

    /**
     * Number of real-space grid points stored on the local rank.
     */
    [[nodiscard]] virtual int local_grid_points() const = 0;

    /**
     * Offset into the global grid for the local segment (e.g., start of slab).
     */
    [[nodiscard]] virtual int local_offset() const = 0;

    /**
     * Where this rank's real-space point `local_index` sits in the whole grid,
     * as the flat index i1·d2·d3 + i2·d3 + i3 that a serial grid stores it at.
     *
     * Code that needs the position of a grid point -- a multipole moment, a
     * kernel evaluated at r -- must ask this rather than add local_offset()
     * planes: that is right for a serial grid and a slab along the first
     * dimension, which is this default, and wrong for the stick layout, which
     * holds planes along the third dimension and overrides it.
     */
    [[nodiscard]] virtual int global_real_space_index(int local_index) const {
        return local_offset() * grid_dimension_2() * grid_dimension_3() + local_index;
    }

    /**
     * Number of first-dimension planes owned by this rank for G-vector partitioning.
     *
     * In serial mode this equals grid_dimension_1(). In MPI mode this
     * returns the slab size (local_n1 from FFTW's decomposition), which
     * may differ from local_grid_points() / (dim2*dim3) when the FFT
     * operates on the full grid (Allreduce mode).
     */
    [[nodiscard]] virtual int slab_dimension_1() const {
        return grid_dimension_1(); // Default: full grid (serial)
    }

    /**
     * Number of reciprocal-space points this rank holds.
     *
     * Equal to local_grid_points() for every layout in which G-space and
     * real space share a buffer. The stick decomposition holds G-space as
     * whole sticks and real space as whole planes, so the two counts differ,
     * and a caller sizing a G-space scatter must ask this rather than assume.
     */
    [[nodiscard]] virtual int local_g_space_points() const {
        return local_grid_points();
    }

    /**
     * Whether this rank holds the G-vectors sharing the first two Miller
     * indices. True everywhere for a layout that does not distribute G-space
     * by stick; the plane-wave selector asks this to decide what to keep.
     */
    [[nodiscard]] virtual bool owns_stick(int /*miller_1*/, int /*miller_2*/) const {
        return true;
    }

    /**
     * Assign reciprocal-space ownership from a weight per stick -- the number
     * of G-vectors on it inside the cutoff. Only a layout that distributes by
     * stick has anything to do with it; everything else ignores it.
     */
    virtual void assign_sticks(const std::vector<int>& /*stick_weights*/) {}

    /**
     * Whether a G-vector and its negative are always held by the same rank.
     * True for a serial grid and for the stick layout; false for slabs, which
     * is what forced the gamma-only path off under MPI.
     */
    [[nodiscard]] virtual bool co_locates_mirrors() const {
        return true;
    }

    /**
     * Copy this rank's reciprocal-space points out of a full, replicated
     * G-space array laid out [i1][i2][i3], into the local G-space layout.
     * The slab layout is a contiguous run at local_offset(); the stick layout
     * is not, so consumers must go through this rather than a memcpy.
     */
    virtual void gather_local_g_space(const Complex* full_g_space, Complex* local_g_space) const {
        std::size_t offset = static_cast<std::size_t>(local_offset()) *
                             static_cast<std::size_t>(grid_dimension_2()) *
                             static_cast<std::size_t>(grid_dimension_3());
        std::memcpy(local_g_space,
                    full_g_space + offset,
                    static_cast<std::size_t>(local_g_space_points()) * sizeof(Complex));
    }

    /// The reverse: place this rank's reciprocal-space points into a full array.
    virtual void scatter_local_g_space(const Complex* local_g_space, Complex* full_g_space) const {
        std::size_t offset = static_cast<std::size_t>(local_offset()) *
                             static_cast<std::size_t>(grid_dimension_2()) *
                             static_cast<std::size_t>(grid_dimension_3());
        std::memcpy(full_g_space + offset,
                    local_g_space,
                    static_cast<std::size_t>(local_g_space_points()) * sizeof(Complex));
    }

    /**
     * Forward FFT: real-space -> reciprocal-space (unnormalized).
     */
    virtual void forward(Complex* data) const = 0;

    /**
     * Inverse FFT: reciprocal-space -> real-space (includes 1/N normalization).
     */
    virtual void inverse(Complex* data) const = 0;

    /**
     * Inverse FFT without 1/N normalization (raw FFTW backward).
     */
    virtual void inverse_raw(Complex* data) const = 0;

    /**
     * Map a G-vector (integer Miller indices) to FFT array index.
     * Handles negative indices via wrapping.
     */
    [[nodiscard]] virtual int
    g_vector_to_fft_grid_index(int miller_1, int miller_2, int miller_3) const = 0;

    /**
     * Reduce G-space data across the pool to make it redundant on all ranks.
     *
     * Serial: no-op.
     * MPI:    MPI_Allreduce(MPI_IN_PLACE, ..., MPI_SUM, pool_comm).
     */
    virtual void reduce_g_space(Complex* data, int number_of_elements) const = 0;

    /**
     * Access the FFT grid's internal input buffer for zero-copy transforms.
     *
     * Callers scatter G-space data directly into this buffer, then call
     * execute_inverse_raw_on_buffers() to transform in-place. This eliminates
     * the memcpy overhead of inverse_raw() / forward().
     *
     * The buffer has at least local_grid_points() elements of usable space.
     */
    virtual Complex* input_buffer() const = 0;

    /**
     * Access the FFT grid's internal output buffer for zero-copy transforms.
     *
     * After execute_inverse_raw_on_buffers(), real-space data is here.
     * After execute_forward_on_buffers(), G-space data is here.
     *
     * For in-place plans (MPI FFT), this returns the same pointer as input_buffer().
     * For out-of-place plans (Serial FFT), this is a separate buffer.
     */
    virtual Complex* output_buffer() const = 0;

    /**
     * Execute a raw inverse FFT on the internal buffers (no copy, no normalization).
     *
     * Caller must have populated input_buffer() before calling.
     * Result is available in output_buffer() after this call.
     */
    virtual void execute_inverse_raw_on_buffers() const = 0;

    /**
     * Execute a forward FFT on the internal buffers (no copy).
     *
     * Caller must have populated input_buffer() before calling.
     * Result is available in output_buffer() after this call.
     */
    virtual void execute_forward_on_buffers() const = 0;

    /**
     * Prepare the input buffer for a forward transform after modifying the output buffer.
     *
     * For out-of-place plans (Serial FFT), this copies output_buffer → input_buffer.
     * For in-place plans (MPI FFT), this is a no-op (same buffer).
     *
     * Call this between the real-space multiply and execute_forward_on_buffers().
     */
    virtual void prepare_forward_from_output() const = 0;

    /**
     * How many transforms this grid can have in flight at once.
     *
     * The zero-copy buffer API above is a single set of scratch arrays, so two
     * threads transforming through it at the same time would overwrite each
     * other. A grid that can hand out independent scratch reports more than
     * one slot here and honours the `*_on_slot` calls below; a grid that
     * cannot leaves this at 1 and callers keep their band loops serial.
     *
     * Slot 0 is always the buffers `input_buffer()` and `output_buffer()`
     * return, so a caller that asks for one slot and a caller that never heard
     * of slots drive exactly the same arrays through exactly the same plan.
     * That is what keeps a one-thread run bit-identical to a build with no
     * threading at all.
     */
    [[nodiscard]] virtual int concurrent_transform_slots() const {
        return 1;
    }

    /**
     * The input buffer of transform slot `slot`, the slot-aware form of
     * `input_buffer()`.
     */
    [[nodiscard]] virtual Complex* slot_input_buffer(int /*slot*/) const {
        return input_buffer();
    }

    /**
     * The output buffer of transform slot `slot`, the slot-aware form of
     * `output_buffer()`.
     */
    [[nodiscard]] virtual Complex* slot_output_buffer(int /*slot*/) const {
        return output_buffer();
    }

    /// Inverse transform on slot `slot`, the slot-aware `execute_inverse_raw_on_buffers()`.
    virtual void execute_inverse_raw_on_slot(int /*slot*/) const {
        execute_inverse_raw_on_buffers();
    }

    /// Forward transform on slot `slot`, the slot-aware `execute_forward_on_buffers()`.
    virtual void execute_forward_on_slot(int /*slot*/) const {
        execute_forward_on_buffers();
    }

    /// Output-to-input copy on slot `slot`, the slot-aware `prepare_forward_from_output()`.
    virtual void prepare_forward_from_output_on_slot(int /*slot*/) const {
        prepare_forward_from_output();
    }

    /**
     * In-place inverse transform of a caller-owned array through slot `slot`,
     * the slot-aware form of `inverse_raw()`.
     *
     * The array belongs to the caller, so two threads passing different arrays
     * collide only over the grid's own scratch -- which is what the slot
     * separates.
     */
    virtual void inverse_raw_on_slot(Complex* data, int /*slot*/) const {
        inverse_raw(data);
    }

    /**
     * Pre-create FFTW plans for batched transforms of a given batch size.
     *
     * Must be called before forward_batched() or inverse_raw_batched() with the
     * corresponding batch_size. Calling with a different batch_size replaces
     * the existing batched plans.
     *
     * The default implementation is a no-op (the fallback batched methods loop
     * over single-transform calls, which need no separate plan).
     */
    virtual void initialize_batched_plans(int /*batch_size*/,
                                          bool /*use_measured_planning*/ = false) const {
        // Default: no-op — batched methods fall back to per-transform loops.
    }

    /**
     * Returns the batch_size for which batched plans are currently initialized,
     * or 0 if no batched plans have been created.
     *
     * Callers can use this to determine whether to zero-pad a smaller batch
     * to the pre-initialized plan size (avoiding a plan recreation that would
     * require a collective MPI call).
     */
    virtual int initialized_batched_plan_size() const {
        return 0; // Default: no batched plan
    }

    /**
     * Batched forward FFT: real-space -> reciprocal-space (unnormalized).
     *
     * Transforms batch_size independent 3D grids packed contiguously:
     *   data[0 .. local_grid_points-1]                     = transform 0
     *   data[local_grid_points .. 2*local_grid_points-1]   = transform 1
     *   ...
     *
     * The default implementation loops over forward() for each transform.
     */
    virtual void forward_batched(Complex* data, int batch_size) const {
        const int stride = local_grid_points();
        for (int transform_index = 0; transform_index < batch_size; ++transform_index) {
            forward(data + transform_index * stride);
        }
    }

    /**
     * Batched inverse FFT without 1/N normalization (raw FFTW backward).
     *
     * Transforms batch_size independent 3D grids packed contiguously
     * (same layout as forward_batched).
     *
     * The default implementation loops over inverse_raw() for each transform.
     */
    virtual void inverse_raw_batched(Complex* data, int batch_size) const {
        const int stride = local_grid_points();
        for (int transform_index = 0; transform_index < batch_size; ++transform_index) {
            inverse_raw(data + transform_index * stride);
        }
    }
};

} // namespace q3
