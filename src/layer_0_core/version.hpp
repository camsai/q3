#pragma once

/**
 * Q3's own build identity, for the provenance block of a results file.
 *
 * The commit comes from q3_git_commit.h, which the build regenerates on every
 * compile (cmake/write_git_commit_header.cmake) rather than capturing once when
 * cmake last ran. A captured value goes stale the moment anything is committed
 * afterwards, and a provenance record that confidently names the wrong source
 * is worse than one that admits it does not know: the first is
 * indistinguishable from a real answer, the second is not. For the same reason
 * the generator reports nothing for a dirty worktree, which is not the commit
 * it sits on, and nothing when there is no repository to ask.
 *
 * The version string is the release identity and tracks the one declared in
 * pyproject.toml; they are checked against each other by a unit test, because
 * two version strings that drift apart are worse than one.
 */

#if __has_include(<q3_git_commit.h>)
#include <q3_git_commit.h>
#endif

namespace q3 {

inline constexpr const char* version = "0.1.0";

#ifdef Q3_GIT_COMMIT
inline constexpr const char* commit = Q3_GIT_COMMIT;
#else
inline constexpr const char* commit = "";
#endif

/// The BLAS/LAPACK libraries this build links, as CMake found them; empty if unknown.
#ifdef Q3_LINEAR_ALGEBRA_LIBRARIES
inline constexpr const char* linear_algebra_libraries = Q3_LINEAR_ALGEBRA_LIBRARIES;
#else
inline constexpr const char* linear_algebra_libraries = "";
#endif

} // namespace q3
