#pragma once

/**
 * Resident memory sampling for Q3.
 *
 * Answers "how much memory does this rank hold right now, and what is the most
 * it has ever held", so that a footprint claim can be measured rather than
 * derived from array sizes. The peak matters more than the current value: the
 * Davidson basis grows across iterations and the collapse workspaces appear
 * only at the first restart, so any single early reading understates the run.
 *
 * Where the platform offers it, the sample also splits resident memory into
 * anonymous and shared pages and reports what the allocator believes it is
 * holding. That split is what distinguishes memory Q3 allocated from memory an
 * MPI transport mapped underneath it.
 */

#include <cstdint>
#include <cstdio>
#include <string>

#if defined(__linux__)
#include <cstring>
#include <sys/resource.h>
#if defined(__GLIBC__)
#include <malloc.h>
#define Q3_HAS_MALLOC_TRIM 1
#if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 33)
#define Q3_HAS_MALLINFO2 1
#endif
#endif
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <sys/resource.h>
#endif

namespace q3 {

/**
 * One reading of this process's resident memory, in bytes.
 *
 * `available` is false when no source could be read; every field is then zero.
 * Sampling never throws and never fails the run — a missing reading is worth
 * less than a crashed calculation.
 */
struct ResidentMemorySample {
    /// Resident set size at the moment of sampling.
    std::uint64_t current_bytes = 0;
    /// High-water mark of the resident set over the life of the process.
    std::uint64_t peak_bytes = 0;
    /// Resident anonymous pages: heap and stack, what this process allocated.
    std::uint64_t anonymous_bytes = 0;
    /// Resident shared pages: mapped files and shared-memory segments.
    std::uint64_t shared_bytes = 0;
    /// Bytes the C allocator reports as handed out, where it can be asked.
    std::uint64_t malloc_in_use_bytes = 0;
    /// Whether any source could be read.
    bool available = false;

    static ResidentMemorySample sample();

    static double to_gibibytes(std::uint64_t bytes) {
        return static_cast<double>(bytes) / 1073741824.0;
    }

    static std::string format_gibibytes(std::uint64_t bytes);
};

#if defined(__linux__)

/**
 * Linux reads /proc/self/status, the only source that separates anonymous from
 * shared pages and the only one that reports a peak without a syscall per
 * field. getrusage supplies the peak if the file cannot be opened.
 */
inline ResidentMemorySample ResidentMemorySample::sample() {
    ResidentMemorySample result;
    std::FILE* status_file = std::fopen("/proc/self/status", "r");
    if (status_file != nullptr) {
        char line[256];
        while (std::fgets(line, sizeof(line), status_file) != nullptr) {
            unsigned long long kilobytes = 0;
            if (std::sscanf(line, "VmRSS: %llu kB", &kilobytes) == 1) {
                result.current_bytes = static_cast<std::uint64_t>(kilobytes) * 1024;
                result.available = true;
            } else if (std::sscanf(line, "VmHWM: %llu kB", &kilobytes) == 1) {
                result.peak_bytes = static_cast<std::uint64_t>(kilobytes) * 1024;
                result.available = true;
            } else if (std::sscanf(line, "RssAnon: %llu kB", &kilobytes) == 1) {
                result.anonymous_bytes = static_cast<std::uint64_t>(kilobytes) * 1024;
            } else if (std::sscanf(line, "RssShmem: %llu kB", &kilobytes) == 1) {
                result.shared_bytes = static_cast<std::uint64_t>(kilobytes) * 1024;
            }
        }
        std::fclose(status_file);
    }
    if (result.peak_bytes == 0) {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            result.peak_bytes = static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
            result.available = true;
        }
    }
#if defined(Q3_HAS_MALLINFO2)
    struct mallinfo2 allocator_info = mallinfo2();
    result.malloc_in_use_bytes = static_cast<std::uint64_t>(allocator_info.uordblks) +
                                 static_cast<std::uint64_t>(allocator_info.hblkhd);
#endif
    return result;
}

#elif defined(__APPLE__)

/**
 * macOS has no /proc, so the current size comes from the Mach task info and the
 * peak from whichever of the task info and getrusage is larger. ru_maxrss is in
 * bytes here, unlike Linux. No anonymous/shared split is reported.
 */
inline ResidentMemorySample ResidentMemorySample::sample() {
    ResidentMemorySample result;
    mach_task_basic_info task_information;
    mach_msg_type_number_t information_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(),
                  MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&task_information),
                  &information_count) == KERN_SUCCESS) {
        result.current_bytes = static_cast<std::uint64_t>(task_information.resident_size);
        result.peak_bytes = static_cast<std::uint64_t>(task_information.resident_size_max);
        result.available = true;
    }
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        std::uint64_t reported_peak = static_cast<std::uint64_t>(usage.ru_maxrss);
        if (reported_peak > result.peak_bytes) {
            result.peak_bytes = reported_peak;
        }
        result.available = true;
    }
    return result;
}

#else

/// Platforms without a known source report nothing rather than guessing.
inline ResidentMemorySample ResidentMemorySample::sample() {
    return ResidentMemorySample{};
}

#endif

inline std::string ResidentMemorySample::format_gibibytes(std::uint64_t bytes) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f GiB", to_gibibytes(bytes));
    return std::string(buffer);
}

/**
 * Hand freed heap back to the operating system.
 *
 * Both allocators keep the pages they were given after every block in them
 * has been freed, and the resident set -- what a scheduler or an OOM killer
 * counts -- keeps them too. At four ranks on the 54-atom silicon cell a rank
 * sat at 0.85 GiB with 0.32 GiB of heap in use: the rest was the setup
 * transients and the solver's per-iteration temporaries, dead but resident.
 * glibc's malloc_trim and macOS's pressure relief return what they can, and
 * this is called at phase boundaries. Measured, it does not lower the peak
 * inside an SCF iteration -- the next iteration's temporaries reuse the
 * pages it would have returned -- but it does lower what a later phase
 * starts from: the one-rank run of that cell, whose stress used to stack on
 * the retained pages, peaks at 4.12 GiB instead of 4.83. Returns the bytes
 * released where the platform reports them, and 0 where it does not.
 */
inline std::uint64_t release_unused_heap() {
#if defined(Q3_HAS_MALLOC_TRIM)
    malloc_trim(0);
    return 0;
#elif defined(__APPLE__)
    return static_cast<std::uint64_t>(malloc_zone_pressure_relief(nullptr, 0));
#else
    return 0;
#endif
}

} // namespace q3
