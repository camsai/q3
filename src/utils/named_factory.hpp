#pragma once

/**
 * NamedFactory — a registry entry that only builds one implementation.
 *
 * Some plug-in points need nothing from an implementation beyond a name and a
 * way to construct it: a volumetric file format, a PAW one-centre convention.
 * This entry carries exactly that, so each implementation does not need an
 * entry class of its own. See utils/registry.hpp for how entries are found.
 */

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace q3 {

/// A registry entry that builds one implementation of `Product`.
template <typename Product> class NamedFactory {
  public:
    using Builder = std::function<std::unique_ptr<Product>()>;

    /// An entry answering to `names` (the first is canonical) that builds with `builder`.
    NamedFactory(std::vector<std::string> names, Builder builder)
        : names_(std::move(names)), builder_(std::move(builder)) {}

    /// The names this entry answers to; the first is canonical.
    [[nodiscard]] const std::vector<std::string>& names() const {
        return names_;
    }

    /// A newly built implementation.
    [[nodiscard]] std::unique_ptr<Product> create() const {
        return builder_();
    }

  private:
    std::vector<std::string> names_;
    Builder builder_;
};

} // namespace q3
