#pragma once

/**
 * How Q3's Complex buffers cross MPI, named in one place because the wrong
 * choice fails on some MPI builds and not others.
 *
 * Two separate hazards live here, and they have different answers.
 *
 * Moving complex data (Bcast, Allgather, Alltoall) needs a datatype whose
 * handle the library recognizes. Complex is std::complex<double>, and
 * MPI_CXX_DOUBLE_COMPLEX is what the MPI standard defines for it. The
 * similar-looking MPI_DOUBLE_COMPLEX is Fortran's DOUBLE COMPLEX, which a
 * library built without Fortran never initializes: it arrives as
 * MPI_DATATYPE_NULL and the job aborts. Q3 passed that Fortran datatype from
 * C++ everywhere until 2026-09-11 (camsai/q3#11). The rule that fixed it
 * stands: nothing in the C++ tree may name a Fortran datatype.
 *
 * Summing complex data is the harder one, and it is why reductions below do
 * not use a complex datatype at all (camsai/q3#17). Under `podman-hpc --mpi`
 * and Shifter's `--module=mpich` the host's MPICH replaces the container's at
 * run time, and the host library rejected MPI_CXX_DOUBLE_COMPLEX for MPI_SUM:
 *
 *     MPIR_SUM_check_dtype(83): MPI_Op MPI_SUM operation not defined
 *                               for this datatype
 *
 * while the same image running its own MPICH summed it correctly. What is
 * established about that, and what is not:
 *
 *   - MPICH classifies MPI_CXX_DOUBLE_COMPLEX under COMPLEX_EXTRA, commented
 *     in its own source as "extra types that are not required to be supported
 *     by the MPI Standard", and gates it on HAVE_CXX_COMPLEX -- a configure
 *     result, so whether MPI_SUM accepts it is a property of how the library
 *     was built. MPI_DOUBLE is in no such group.
 *   - It is NOT a handle-numbering difference, though that is the intuitive
 *     explanation. The values are fixed per MPICH release rather than derived
 *     from what a build enables: configuring 4.2.3 with and without Fortran
 *     yields byte-identical constants, and 3.4.3 -- the generation the Cray
 *     library derives from -- assigns the same 0x4c001035 and the same C
 *     index. Rebuilding the image's MPICH with Fortran, the fix camsai/q3#17
 *     proposes, would therefore have changed nothing.
 *   - Which build option the host library was configured with is not
 *     determinable from here, and does not need to be.
 *
 * So the reduction avoids the question rather than betting on an answer.
 * MPI_DOUBLE is the one datatype observed to survive the substitution -- it is
 * what the one-rank-per-pool path reduces, which is why that path never
 * aborted -- and complex addition is componentwise, so summing 2N doubles is
 * bit-for-bit the same operation as summing N complex numbers. That holds for
 * any MPI, needs nothing from the library beyond the MPI_DOUBLE/MPI_SUM pair
 * the standard requires of every implementation, and costs nothing: no extra
 * copy, no change in message size.
 */

#include "layer_0_core/types/type_aliases.hpp"

#include <mpi.h>

#include <algorithm>

namespace q3 {
namespace execution {

/**
 * The MPI datatype matching Complex on the wire, for moving data -- NOT for
 * reductions, which must go through sum_complex_in_place(). See the file
 * comment.
 */
inline MPI_Datatype complex_datatype() {
    static_assert(sizeof(Complex) == 2 * sizeof(double),
                  "Complex must be layout-compatible with MPI_CXX_DOUBLE_COMPLEX");
    return MPI_CXX_DOUBLE_COMPLEX;
}

/**
 * MPI_SUM over a complex buffer, reduced as pairs of doubles.
 *
 * The only way complex data may be summed in Q3 -- see the file comment for
 * why a complex datatype is not used. Reinterpreting the buffer is well
 * defined: C++ requires an array of N std::complex<double> to be
 * reinterpretable as an array of 2N doubles, real part first.
 *
 * A null `source` reduces `destination` in place; otherwise the two must not
 * overlap, as for MPI_Allreduce itself.
 *
 * Chunked because the element count doubles: a caller may legitimately pass
 * any int, and 2N would overflow one past 2^30 elements. Each chunk is an
 * independent reduction, so splitting changes no result.
 */
inline int sum_complex(const Complex* source,
                       Complex* destination,
                       int number_of_elements,
                       MPI_Comm communicator) {
    constexpr int max_complex_per_call = 1 << 30;
    const auto* source_doubles = reinterpret_cast<const double*>(source);
    auto* destination_doubles = reinterpret_cast<double*>(destination);
    for (int offset = 0; offset < number_of_elements;) {
        const int chunk = std::min(number_of_elements - offset, max_complex_per_call);
        const void* send_buffer = source == nullptr
                                      ? MPI_IN_PLACE
                                      : static_cast<const void*>(source_doubles + 2 * offset);
        const int status = MPI_Allreduce(send_buffer,
                                         destination_doubles + 2 * offset,
                                         2 * chunk,
                                         MPI_DOUBLE,
                                         MPI_SUM,
                                         communicator);
        if (status != MPI_SUCCESS)
            return status;
        offset += chunk;
    }
    return MPI_SUCCESS;
}

/// In-place complex MPI_SUM -- see sum_complex().
inline int sum_complex_in_place(Complex* data, int number_of_elements, MPI_Comm communicator) {
    return sum_complex(nullptr, data, number_of_elements, communicator);
}

} // namespace execution
} // namespace q3
