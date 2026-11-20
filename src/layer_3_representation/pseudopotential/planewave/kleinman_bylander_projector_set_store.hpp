#pragma once

/**
 * KleinmanBylanderProjectorSetStore — one projector set per k-point, each
 * built when it is first asked for.
 *
 * What the SCF iterator owns in place of a matrix per k-point, and what it
 * hands to everything that needs projectors: the eigensolver, the starting
 * rotation and the projector occupations directly, and forces, stress and
 * exports through the SCF state. at() returns the k-point's set, building it
 * through the store's one ProjectorSetBuilder if it is not built or was built
 * from a geometry that has since changed.
 *
 * Memory. A store keeps every set it builds, which is what the SCF needs: every
 * solve reads every k-point's set again. After keep_latest_set_only() it holds
 * one set at a time, freeing the others whenever a set has to be built: what
 * follows the SCF walks the k-points and needs one set at a time, and holding
 * them all would put the whole cache under the run's peak again. A reference
 * from at() is then good only until at() is called for another k-point.
 * keep_all_sets() returns to keeping everything, for the SCF and for any
 * reader that uses two k-points' sets together.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/pseudopotential/planewave/projector_set_builder.hpp"

namespace q3 {

/// One Kleinman-Bylander projector set per k-point, built on demand; see the file comment.
class KleinmanBylanderProjectorSetStore {
  public:
    /**
     * A store for the k-points of `kpoint_grid`, each over the plane waves listed for it in
     * `g_vector_indices_by_kpoint` (indices into the basis's G-vectors; for a run, the basis's
     * own table). The crystal, the grid and the lists must outlive the store; the builder is
     * the store's own.
     */
    KleinmanBylanderProjectorSetStore(
        const Crystal& crystal_structure,
        const KPointsGrid& kpoint_grid,
        const std::vector<std::vector<int>>& g_vector_indices_by_kpoint,
        std::unique_ptr<ProjectorSetBuilder> builder)
        : crystal_structure_(crystal_structure)
        , kpoint_grid_(kpoint_grid)
        , g_vector_indices_by_kpoint_(g_vector_indices_by_kpoint)
        , builder_(std::move(builder))
        , projector_sets_(static_cast<std::size_t>(kpoint_grid.number_of_kpoints()))
        , last_use_(static_cast<std::size_t>(kpoint_grid.number_of_kpoints()), 0) {
        if (!builder_) {
            throw std::invalid_argument("KleinmanBylanderProjectorSetStore: builder is null");
        }
    }

    /// Number of k-points, and so of projector sets.
    [[nodiscard]] int number_of_kpoints() const {
        return static_cast<int>(projector_sets_.size());
    }

    /**
     * The projector set of one k-point, built now if it is not built or no longer describes
     * the crystal. The reference stays valid for the store's lifetime; what it refers to is
     * rebuilt in place when the geometry changes, and freed by release() or, while the store
     * keeps the latest set only, by asking for another k-point.
     */
    [[nodiscard]] const KleinmanBylanderProjectorSet& at(int kpoint_index) {
        KleinmanBylanderProjectorSet& projector_set =
            projector_sets_[static_cast<std::size_t>(kpoint_index)];
        const Vector3D& kpoint_crystal_coordinates = kpoint_grid_[kpoint_index].coordinates;
        const std::vector<int>& g_vector_indices =
            g_vector_indices_by_kpoint_[static_cast<std::size_t>(kpoint_index)];
        const int number_of_plane_waves = static_cast<int>(g_vector_indices.size());
        last_use_[static_cast<std::size_t>(kpoint_index)] = ++use_counter_;
        if (projector_set.is_built_from(
                kpoint_crystal_coordinates, crystal_structure_, number_of_plane_waves)) {
            make_room_on_the_device(kpoint_index, set_bytes(projector_set));
            return projector_set;
        }
        if (!keeps_all_sets_) {
            for (KleinmanBylanderProjectorSet& other_set : projector_sets_) {
                other_set.release();
            }
        }
        // Before the build, for a builder that fills the set on the device as it builds.
        make_room_on_the_device(kpoint_index, expected_set_bytes(number_of_plane_waves));
        builder_->build(
            kpoint_crystal_coordinates, g_vector_indices, number_of_plane_waves, projector_set);
        make_room_on_the_device(kpoint_index, set_bytes(projector_set));
        return projector_set;
    }

    /**
     * Limit the device memory the sets hold together, in bytes; zero, the default, is no limit.
     *
     * The device copies are a cache: any of them can be given up and made again. When handing
     * out a set would take the device copies over the limit, the sets used longest ago give
     * theirs up first (see KleinmanBylanderProjectorSet::evict_from_device()), never the one
     * being handed out. A run with more k-points than fit therefore uploads or rebuilds a set
     * each time it comes back to a k-point, which is slower and still correct, rather than
     * running out of device memory. A single set larger than the limit is still handed out.
     */
    void set_device_memory_budget(std::size_t budget_bytes) {
        device_memory_budget_bytes_ = budget_bytes;
    }

    /// The limit set by set_device_memory_budget(); zero when there is none.
    [[nodiscard]] std::size_t device_memory_budget() const {
        return device_memory_budget_bytes_;
    }

    /// Bytes one k-point's set holds on a device at the moment, without handing it out.
    [[nodiscard]] std::size_t device_bytes(int kpoint_index) const {
        return projector_sets_[static_cast<std::size_t>(kpoint_index)].device_bytes();
    }

    /// Bytes the sets hold on a device at the moment, all of them together.
    [[nodiscard]] std::size_t device_bytes() const {
        std::size_t total_bytes = 0;
        for (const KleinmanBylanderProjectorSet& projector_set : projector_sets_) {
            total_bytes += projector_set.device_bytes();
        }
        return total_bytes;
    }

    /// Whether a k-point's set is built at the moment, without building it.
    [[nodiscard]] bool is_built(int kpoint_index) const {
        return projector_sets_[static_cast<std::size_t>(kpoint_index)].is_built();
    }

    /// Free every set, on the host and on the device. Each is built again when next asked for.
    void release() {
        for (KleinmanBylanderProjectorSet& projector_set : projector_sets_) {
            projector_set.release();
        }
    }

    /// Keep every set that gets built, as a new store does.
    void keep_all_sets() {
        keeps_all_sets_ = true;
    }

    /// From now on, free the other sets whenever one has to be built.
    void keep_latest_set_only() {
        keeps_all_sets_ = false;
    }

  private:
    [[nodiscard]] static std::size_t set_bytes(const KleinmanBylanderProjectorSet& projector_set) {
        return static_cast<std::size_t>(projector_set.number_of_projectors()) *
               static_cast<std::size_t>(projector_set.number_of_plane_waves()) * sizeof(Complex);
    }

    /**
     * What a set over this many plane waves will take once built: every k-point has the same
     * projectors, so the count is read from any set that is built. Zero when none is, and then
     * the limit is applied after the build instead.
     */
    [[nodiscard]] std::size_t expected_set_bytes(int number_of_plane_waves) const {
        for (const KleinmanBylanderProjectorSet& projector_set : projector_sets_) {
            if (projector_set.is_built()) {
                return static_cast<std::size_t>(projector_set.number_of_projectors()) *
                       static_cast<std::size_t>(number_of_plane_waves) * sizeof(Complex);
            }
        }
        return 0;
    }

    /**
     * Have the least recently used other sets give up their device copies until a device copy
     * of `needed_bytes` for this k-point fits under the limit, or no other set holds one.
     */
    void make_room_on_the_device(int kpoint_index, std::size_t needed_bytes) {
        if (device_memory_budget_bytes_ == 0) {
            return;
        }
        const std::size_t kept_index = static_cast<std::size_t>(kpoint_index);
        while (true) {
            std::size_t other_bytes = 0;
            std::size_t least_recently_used = projector_sets_.size();
            for (std::size_t index = 0; index < projector_sets_.size(); ++index) {
                if (index == kept_index || projector_sets_[index].device_bytes() == 0) {
                    continue;
                }
                other_bytes += projector_sets_[index].device_bytes();
                if (least_recently_used == projector_sets_.size() ||
                    last_use_[index] < last_use_[least_recently_used]) {
                    least_recently_used = index;
                }
            }
            if (least_recently_used == projector_sets_.size() ||
                other_bytes + needed_bytes <= device_memory_budget_bytes_) {
                return;
            }
            projector_sets_[least_recently_used].evict_from_device();
        }
    }

    const Crystal& crystal_structure_;
    const KPointsGrid& kpoint_grid_;
    const std::vector<std::vector<int>>& g_vector_indices_by_kpoint_;
    std::unique_ptr<ProjectorSetBuilder> builder_;
    std::vector<KleinmanBylanderProjectorSet> projector_sets_;
    bool keeps_all_sets_ = true;
    /// Zero is no limit; see set_device_memory_budget().
    std::size_t device_memory_budget_bytes_ = 0;
    /// When each k-point's set was last handed out, as a count of at() calls.
    std::vector<std::uint64_t> last_use_;
    std::uint64_t use_counter_ = 0;
};

} // namespace q3
