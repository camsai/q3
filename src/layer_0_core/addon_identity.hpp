#pragma once

/**
 * Which build-time add-ons this build compiled in, for `--version` and the
 * provenance block of a results file.
 *
 * The list comes from q3_addon_identity.hpp, which cmake/Q3Addons.cmake writes
 * at configure time from the add-ons present in the tree. A result computed
 * with an add-on can depend on it, so the record names it; a core-only build
 * reports an empty list rather than omitting the field, so the absence of
 * add-ons is stated rather than inferred. Without the generated header (a tool
 * compiling a header on its own, outside the build) the list is empty, the
 * same honest default version.hpp uses for the commit.
 */

#include <vector>

#if __has_include(<q3_addon_identity.hpp>)
#include <q3_addon_identity.hpp>
#endif

namespace q3 {

/// One add-on compiled into this build.
struct AddonIdentity {
    const char* name;
    const char* version;
};

/// The add-ons compiled into this build, in the order the build discovered them.
inline const std::vector<AddonIdentity>& active_addons() {
    static const std::vector<AddonIdentity> addons{
#ifdef Q3_ADDON_IDENTITY_ENTRIES
        Q3_ADDON_IDENTITY_ENTRIES
#endif
    };
    return addons;
}

} // namespace q3
