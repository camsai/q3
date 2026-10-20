#pragma once

/**
 * PseudopotentialSet — polymorphic container for pseudopotential_set.
 *
 * Stores unique_ptrs internally but exposes const Pseudopotential& via
 * operator[], so all existing `.field` access patterns work unchanged.
 * This avoids changing 20+ function signatures from .field to ->field.
 */

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_3_representation/pseudopotential/init.hpp"
#include "layer_3_representation/pseudopotential/paw/init.hpp"
#include "layer_3_representation/pseudopotential/ultrasoft/init.hpp"

namespace q3 {

/**
 * Polymorphic container for pseudopotentials.
 *
 * Stores unique_ptrs internally but exposes const Pseudopotential& via
 * operator[], so all existing `.field` access patterns work unchanged.
 */
class PseudopotentialSet {
  public:
    PseudopotentialSet(const PseudopotentialSet&) = delete;
    PseudopotentialSet& operator=(const PseudopotentialSet&) = delete;
    PseudopotentialSet(PseudopotentialSet&&) = default;
    PseudopotentialSet& operator=(PseudopotentialSet&&) = default;

    PseudopotentialSet() = default;

    /// Move a unique_ptr into the set.
    void push_back(std::unique_ptr<Pseudopotential> pseudopotential) {
        pseudopotentials_.push_back(std::move(pseudopotential));
    }

    /// Access by index (const reference).
    [[nodiscard]] const Pseudopotential& operator[](int index) const {
        assert(index >= 0 && index < static_cast<int>(pseudopotentials_.size()));
        return *pseudopotentials_[index];
    }

    /// Access by index (mutable reference).
    Pseudopotential& operator[](int index) {
        assert(index >= 0 && index < static_cast<int>(pseudopotentials_.size()));
        return *pseudopotentials_[index];
    }

    /**
     * Access as ultrasoft pseudopotential.
     *
     * A PAW dataset also satisfies this cast — it extends the ultrasoft class
     * because it carries the same augmentation apparatus — so the check below
     * tests the storage layout, not the pseudopotential kind. Use
     * pseudopotential_type_enum() to distinguish the two.
     */
    [[nodiscard]] const UltrasoftPseudopotential& as_ultrasoft(int index) const {
        assert(index >= 0 && index < static_cast<int>(pseudopotentials_.size()));
        const auto* result =
            dynamic_cast<const UltrasoftPseudopotential*>(pseudopotentials_[index].get());
        if (result == nullptr) {
            throw std::logic_error(
                "as_ultrasoft(" + std::to_string(index) + "): species '" +
                pseudopotentials_[index]->element +
                "' carries no augmentation, so there is no ultrasoft form to return. A cell "
                "mixing norm-conserving species with ultrasoft or PAW ones reaches this when a "
                "caller asks for the augmented form per species without gating on the species. "
                "This was an assert, which a release build compiles out -- so the null it "
                "guarded was dereferenced instead of reported.");
        }
        return *result;
    }

    /// Access as PAW pseudopotential.
    [[nodiscard]] const PawPseudopotential& as_paw(int index) const {
        assert(index >= 0 && index < static_cast<int>(pseudopotentials_.size()));
        const auto* result =
            dynamic_cast<const PawPseudopotential*>(pseudopotentials_[index].get());
        if (result == nullptr) {
            throw std::logic_error(
                "as_paw(" + std::to_string(index) + "): species '" +
                pseudopotentials_[index]->element +
                "' is not a PAW dataset, so there is no one-center form to return. A cell mixing "
                "PAW species with norm-conserving or ultrasoft ones reaches this when a caller "
                "asks for the one-center form per species without gating on the species. This "
                "was an assert, which a release build compiles out -- so the null it guarded was "
                "dereferenced instead of reported.");
        }
        return *result;
    }

    [[nodiscard]] int size() const {
        return static_cast<int>(pseudopotentials_.size());
    }

    [[nodiscard]] bool empty() const {
        return pseudopotentials_.empty();
    }

    /**
     * Check whether any pseudopotential in this set is of the given type.
     */
    [[nodiscard]] bool has_pseudopotential_type(PseudopotentialType target_type) const {
        for (int index = 0; index < size(); ++index) {
            if ((*this)[index].pseudopotential_type_enum() == target_type) {
                return true;
            }
        }
        return false;
    }

    /**
     * Whether any species needs the augmentation machinery.
     *
     * The question the plane-wave side actually asks — it wants to know
     * whether to build Q(G) tables, an S-operator and a per-step D matrix, and
     * both ultrasoft and PAW datasets answer yes. Asking it by capability
     * rather than by `== ultrasoft` is what lets a PAW dataset reach that
     * machinery at all: PAW is a layered extension of ultrasoft here, so a
     * type equality test silently routes it down the norm-conserving path
     * where its augmentation charges are simply dropped.
     */
    [[nodiscard]] bool has_augmentation_charges() const {
        for (int index = 0; index < size(); ++index) {
            if ((*this)[index].has_augmentation_charges()) {
                return true;
            }
        }
        return false;
    }

    /// Deep-clone the entire set (preserves derived types).
    [[nodiscard]] PseudopotentialSet clone() const {
        PseudopotentialSet result;
        for (const auto& pseudopotential : pseudopotentials_) {
            result.push_back(pseudopotential->clone());
        }
        return result;
    }

  private:
    std::vector<std::unique_ptr<Pseudopotential>> pseudopotentials_;
};

} // namespace q3
