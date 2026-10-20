#pragma once

/**
 * Registry — an ordered set of interchangeable implementations, found by name.
 *
 * Each plug-in point a build-time add-on can extend keeps one: input formats,
 * pseudopotential formats, PAW one-centre conventions, volumetric formats.
 * Core seeds its own entries the first time a registry is used, and an add-on
 * appends its entries from the registration function every entry point calls
 * before it reads any input (io/addon_registration.hpp). Nothing registers
 * itself during static initialisation: the order in which translation units
 * initialise is unspecified, and a registration that depends on it can
 * silently drop out of one binary while working in another.
 *
 * An entry answers to the names its `names()` returns. The first is canonical,
 * which is the one usage text and error messages print. A name belongs to one
 * entry only, so registering a second entry under a taken name throws instead
 * of shadowing the first.
 *
 * Entries keep their registration order — core's first, then each add-on's in
 * the order the build discovered them — so iteration is deterministic.
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q3 {

/// An ordered set of interchangeable implementations of `Entry`, found by name.
template <typename Entry> class Registry {
  public:
    /// Add `entry`; throws std::logic_error when one of its names is already registered.
    void add(std::unique_ptr<Entry> entry) {
        for (const std::string& name : entry->names()) {
            if (find(name) != nullptr)
                throw std::logic_error("The name '" + name + "' is already registered");
        }
        entries_.push_back(std::move(entry));
    }

    /// The entry answering to `name`, or nullptr when none does.
    [[nodiscard]] const Entry* find(const std::string& name) const {
        for (const auto& entry : entries_) {
            for (const std::string& entry_name : entry->names()) {
                if (entry_name == name)
                    return entry.get();
            }
        }
        return nullptr;
    }

    /// Every entry, in registration order.
    [[nodiscard]] const std::vector<std::unique_ptr<Entry>>& entries() const {
        return entries_;
    }

    /**
     * Every entry's canonical name in registration order, joined by
     * `separator` — the list a usage line or an "unknown name" error prints.
     */
    [[nodiscard]] std::string canonical_names(const std::string& separator) const {
        std::string joined;
        for (const auto& entry : entries_) {
            if (!joined.empty())
                joined += separator;
            joined += entry->names().front();
        }
        return joined;
    }

  private:
    std::vector<std::unique_ptr<Entry>> entries_;
};

} // namespace q3
