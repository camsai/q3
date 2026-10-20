#pragma once

/**
 * Thread-team sizing for the OpenMP regions in Q3's hot loops.
 *
 * Every `#pragma omp parallel for` in this codebase carries
 * `num_threads(...) if(... > 1)` with a count from `thread_count_for()`,
 * rather than letting the runtime pick. Two reasons:
 *
 *  - **Zero overhead at one thread.** When the computed count is 1 the `if`
 *    clause is false, so the runtime runs the region inline in the
 *    encountering thread: no team is forked, no barrier is taken, and the
 *    build behaves as if the pragma were not there. That is what makes a
 *    `OMP_NUM_THREADS=1` run of an OpenMP build match a build without it.
 *  - **A team has to earn its keep.** Forking and joining costs a few
 *    microseconds; a loop over a few hundred grid points does not. Each call
 *    site states the smallest slice of work worth handing to a thread and
 *    `thread_count_for()` refuses to start more threads than that allows.
 *
 * Without OpenMP the whole header collapses to constants, so callers need no
 * `#ifdef` of their own.
 */

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace q3 {
namespace execution {
namespace threading {

/**
 * Whether this translation unit was compiled with OpenMP enabled.
 *
 * Reported in the run header so a log says which binary produced it, and used
 * by tests that assert the single-thread path is taken either way.
 */
[[nodiscard]] inline constexpr bool is_compiled_with_openmp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

/**
 * The largest team a parallel region could be given here, which is what
 * `OMP_NUM_THREADS` sets. Always 1 without OpenMP.
 */
[[nodiscard]] inline int maximum_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/**
 * The calling thread's index inside the enclosing parallel region, or 0 when
 * there is none. Used to pick a per-thread scratch slot, which is also why it
 * must stay a plain index into a dense range rather than an opaque id.
 */
[[nodiscard]] inline int thread_index() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/**
 * The size of the team actually running the enclosing parallel region, or 1
 * outside one.
 *
 * `num_threads(n)` is a request, not a promise -- a runtime with dynamic
 * adjustment on may hand back a smaller team. Code that splits an array into
 * one slice per thread must divide by the team it got, not the team it asked
 * for, or the slices at the end belong to threads that never ran and the work
 * is silently skipped.
 */
[[nodiscard]] inline int team_size() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

/**
 * Whether the caller is already inside a parallel region.
 *
 * Hot loops nest -- the band loop in the Hamiltonian applicator contains the
 * scatter/gather loops -- and an inner region that forked its own team would
 * oversubscribe the machine. Inner call sites gate on this and stay serial
 * when an outer region has already claimed the cores.
 */
[[nodiscard]] inline bool is_inside_parallel_region() {
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

/**
 * How many threads to ask for over `work_items`, given that a thread is not
 * worth starting until it has `minimum_items_per_thread` of work to do.
 *
 * Returns 1 -- meaning "run this region serially, do not fork" -- when OpenMP
 * is off, when `OMP_NUM_THREADS` is 1, when an outer region already holds the
 * cores, or when the loop is too short to pay for a team. Never returns more
 * than `maximum_threads()`.
 */
[[nodiscard]] inline int thread_count_for(long long work_items,
                                          long long minimum_items_per_thread) {
    if (work_items <= 0 || minimum_items_per_thread <= 0) {
        return 1;
    }
    if (is_inside_parallel_region()) {
        return 1;
    }
    long long affordable_threads = work_items / minimum_items_per_thread;
    if (affordable_threads < 2) {
        return 1;
    }
    long long available_threads = static_cast<long long>(maximum_threads());
    return static_cast<int>(std::min(affordable_threads, available_threads));
}

/**
 * The half-open slice of `[0, work_items)` that `slice_index` of
 * `number_of_slices` owns, written into `first` and `last`.
 *
 * Deterministic and independent of how the slices are then scheduled, which is
 * what lets a threaded reduction stay bit-identical: a thread always sums the
 * same contiguous run of terms in the same order, and the partial sums are
 * combined by slice index afterwards. The first `work_items % number_of_slices`
 * slices take one extra item.
 */
inline void slice_bounds(long long work_items,
                         int number_of_slices,
                         int slice_index,
                         long long& first,
                         long long& last) {
    long long slices = static_cast<long long>(number_of_slices);
    long long index = static_cast<long long>(slice_index);
    long long base_size = work_items / slices;
    long long remainder = work_items % slices;
    first = index * base_size + std::min(index, remainder);
    last = first + base_size + (index < remainder ? 1 : 0);
}

} // namespace threading
} // namespace execution
} // namespace q3
