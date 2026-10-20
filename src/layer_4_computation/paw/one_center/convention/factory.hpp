#pragma once

/**
 * Pick the one-center convention a dataset's numbers were generated for.
 *
 * Looks up the dataset's own declared `one_center_convention` in the
 * convention registry, not the file format and not some correlated
 * capability. The thing that makes a reference subtraction correct is what
 * the numbers mean, not where they were read from, and a capability test
 * would silently do the wrong thing the first time some other reader produced
 * either kind.
 *
 * A dataset declaring a convention this build does not provide is refused
 * rather than computed in the default one, because the default would be wrong
 * by a per-element constant while every reference-independent observable
 * still agreed.
 */

#include <memory>
#include <stdexcept>

#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_4_computation/paw/one_center/convention/init.hpp"
#include "layer_4_computation/paw/one_center/convention/registry.hpp"

namespace q3 {
namespace paw_one_center {

inline std::unique_ptr<PawOneCenterConvention>
create_one_center_convention(const PawPseudopotential& pseudopotential) {
    const auto& registry = one_center_convention_registry();
    if (const auto* factory = registry.find(pseudopotential.one_center_convention))
        return factory->create();
    throw std::runtime_error("The PAW dataset for " + pseudopotential.element +
                             " is expressed in the one-center convention '" +
                             pseudopotential.one_center_convention +
                             "', which this build does not provide; the conventions it provides "
                             "are: " +
                             registry.canonical_names(", "));
}

} // namespace paw_one_center
} // namespace q3
