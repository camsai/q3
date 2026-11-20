#pragma once

/**
 * KleinmanBylanderProjectorSet — the projectors <k+G|β> of one k-point, and
 * what they were built from.
 *
 * The one owner of a k-point's projector matrix. Everything that needs the
 * projectors — the Hamiltonian, the overlap, the starting rotation, the
 * projector occupations, forces, stress, exports — reads this set instead of
 * building a matrix of its own, on the host through host() or on a device
 * through device(backend). The matrix is a DeviceMirroredArray, so the set knows
 * whether its device copy is current: rebuilding the set on the host makes the
 * device copy stale, and the next device() uploads it again. No consumer keeps
 * an upload of its own that could outlive the geometry it was made for.
 *
 * The set also records the k-point, the cell and the atomic positions it was
 * built from, so its owner can ask whether it still describes the present
 * geometry rather than having to remember to invalidate it.
 *
 * Layout: number_of_projectors × number_of_plane_waves, one projector after
 * another, so projector p starts at element p × number_of_plane_waves. The
 * projectors are ordered atom by atom, each atom's beta functions in order,
 * each with its 2l+1 magnetic components.
 */

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "layer_0_core/execution/gpu/device/device_mirrored_array.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"

namespace q3 {

/// The Kleinman-Bylander projectors of one k-point, on host and device; see the file comment.
class KleinmanBylanderProjectorSet {
  public:
    KleinmanBylanderProjectorSet() = default;

    // ─── What it holds ──────────────────────────────────────────────────

    /// Whether the set has been built and not released since.
    [[nodiscard]] bool is_built() const {
        return is_built_;
    }

    /// Number of projectors: every beta function of every atom, with its 2l+1 components.
    [[nodiscard]] int number_of_projectors() const {
        return number_of_projectors_;
    }

    /// Number of plane waves each projector is expanded over.
    [[nodiscard]] int number_of_plane_waves() const {
        return number_of_plane_waves_;
    }

    /// True when there is no element to read: not built, no projectors, or no plane waves.
    [[nodiscard]] bool empty() const {
        return projectors_.size() == 0;
    }

    /// The whole matrix on the host, to read; see the file comment for the layout.
    [[nodiscard]] const Complex* host() const {
        return projectors_.host();
    }

    /// One projector's coefficients over the plane waves, on the host, to read.
    [[nodiscard]] const Complex* projector(int projector_index) const {
        return projectors_.host() + static_cast<std::size_t>(projector_index) *
                                        static_cast<std::size_t>(number_of_plane_waves_);
    }

    /**
     * The whole matrix on a device, to read: uploaded on the first call and again only after
     * the set has been rebuilt.
     */
    [[nodiscard]] const Complex*
    device(const std::shared_ptr<execution::gpu::device::GPUDeviceBackend>& device_backend) const {
        return projectors_.device(device_backend);
    }

    /**
     * The matrix as the array that owns its host and device copies, for a consumer that reads
     * it wherever it is current, such as a BandProjectionCalculator.
     */
    [[nodiscard]] const execution::gpu::device::DeviceMirroredArray<Complex>& matrix() const {
        return projectors_;
    }

    /// Whether a device copy exists and holds the present projectors.
    [[nodiscard]] bool device_is_current() const {
        return projectors_.device_is_current();
    }

    // ─── What it was built from ─────────────────────────────────────────

    /**
     * Whether the set was built for this k-point, this cell and these atomic positions, over
     * this many plane waves. False for a set that is not built.
     */
    [[nodiscard]] bool is_built_from(const Vector3D& kpoint_crystal_coordinates,
                                     const Crystal& crystal_structure,
                                     int number_of_plane_waves) const {
        if (!is_built_ || number_of_plane_waves != number_of_plane_waves_ ||
            !same_vector(kpoint_crystal_coordinates, kpoint_crystal_coordinates_) ||
            crystal_structure.two_pi_over_lattice_parameter() != two_pi_over_lattice_parameter_ ||
            crystal_structure.number_of_atoms() != static_cast<int>(atomic_positions_.size())) {
            return false;
        }
        for (int direction = 0; direction < 3; ++direction) {
            if (!same_vector(crystal_structure.lattice_vectors()[direction],
                             lattice_vectors_[direction])) {
                return false;
            }
        }
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); ++atom_index) {
            const std::size_t atom = static_cast<std::size_t>(atom_index);
            if (crystal_structure.species_type_index(atom_index) != species_type_indices_[atom] ||
                !same_vector(crystal_structure.fractional_position(atom_index),
                             atomic_positions_[atom])) {
                return false;
            }
        }
        return true;
    }

    // ─── Building, for ProjectorSetBuilder implementations ──────────────

    /**
     * Start a build: size the matrix, zero it, and record what it is being built from. Returns
     * the host storage to fill, in the layout of the file comment. The device copy, if there is
     * one, is stale from this call on.
     */
    [[nodiscard]] Complex* begin_build(const Vector3D& kpoint_crystal_coordinates,
                                       const Crystal& crystal_structure,
                                       int number_of_projectors,
                                       int number_of_plane_waves) {
        const std::size_t number_of_elements = static_cast<std::size_t>(number_of_projectors) *
                                               static_cast<std::size_t>(number_of_plane_waves);
        if (projectors_.size() != number_of_elements) {
            projectors_.resize(number_of_elements);
        }
        Complex* storage = projectors_.mutable_host();
        std::fill_n(storage, number_of_elements, Complex(0.0, 0.0));
        record_what_it_is_built_from(kpoint_crystal_coordinates,
                                     crystal_structure,
                                     number_of_projectors,
                                     number_of_plane_waves);
        return storage;
    }

    /**
     * Start a build on a device: size the matrix, record what it is being built from, and
     * return the device storage to fill, every element of it, in the layout of the file
     * comment. Nothing is uploaded and no host copy is made: the host gets the projectors only
     * if something later asks for them there.
     */
    [[nodiscard]] Complex*
    begin_device_build(const Vector3D& kpoint_crystal_coordinates,
                       const Crystal& crystal_structure,
                       int number_of_projectors,
                       int number_of_plane_waves,
                       const std::shared_ptr<execution::gpu::device::GPUDeviceBackend>& backend) {
        const std::size_t number_of_elements = static_cast<std::size_t>(number_of_projectors) *
                                               static_cast<std::size_t>(number_of_plane_waves);
        if (projectors_.size() != number_of_elements) {
            projectors_.reset(number_of_elements);
        }
        Complex* storage = projectors_.overwrite_device(backend);
        record_what_it_is_built_from(kpoint_crystal_coordinates,
                                     crystal_structure,
                                     number_of_projectors,
                                     number_of_plane_waves);
        return storage;
    }

    /// Bytes the set holds on a device, current or not; zero when it has no device copy.
    [[nodiscard]] std::size_t device_bytes() const {
        return projectors_.device_is_allocated() ? projectors_.size() * sizeof(Complex) : 0;
    }

    /// Whether the host holds a copy of the matrix. False for a set built on a device.
    [[nodiscard]] bool host_is_allocated() const {
        return projectors_.host_is_allocated();
    }

    /**
     * Give up the device copy. A set the host also holds keeps its projectors and uploads them
     * again when next asked for on a device. A set that lives on the device alone is released
     * altogether, since bringing it down would cost more than building it again: it is not
     * built afterwards.
     */
    void evict_from_device() {
        if (projectors_.host_is_current()) {
            projectors_.release_device();
        } else {
            release();
        }
    }

    /// Free the matrix, on the host and on the device. The set is not built afterwards.
    void release() {
        projectors_.reset(0);
        number_of_projectors_ = 0;
        number_of_plane_waves_ = 0;
        atomic_positions_.clear();
        species_type_indices_.clear();
        is_built_ = false;
    }

  private:
    /// What is_built_from() compares with: the k-point, the cell and the atoms of this build.
    void record_what_it_is_built_from(const Vector3D& kpoint_crystal_coordinates,
                                      const Crystal& crystal_structure,
                                      int number_of_projectors,
                                      int number_of_plane_waves) {
        number_of_projectors_ = number_of_projectors;
        number_of_plane_waves_ = number_of_plane_waves;
        kpoint_crystal_coordinates_ = kpoint_crystal_coordinates;
        two_pi_over_lattice_parameter_ = crystal_structure.two_pi_over_lattice_parameter();
        lattice_vectors_ = crystal_structure.lattice_vectors();
        const std::size_t number_of_atoms =
            static_cast<std::size_t>(crystal_structure.number_of_atoms());
        atomic_positions_.resize(number_of_atoms);
        species_type_indices_.resize(number_of_atoms);
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); ++atom_index) {
            const std::size_t atom = static_cast<std::size_t>(atom_index);
            atomic_positions_[atom] = crystal_structure.fractional_position(atom_index);
            species_type_indices_[atom] = crystal_structure.species_type_index(atom_index);
        }
        is_built_ = true;
    }

    [[nodiscard]] static bool same_vector(const Vector3D& first, const Vector3D& second) {
        return first.x == second.x && first.y == second.y && first.z == second.z;
    }

    execution::gpu::device::DeviceMirroredArray<Complex> projectors_;
    int number_of_projectors_ = 0;
    int number_of_plane_waves_ = 0;
    bool is_built_ = false;

    Vector3D kpoint_crystal_coordinates_;
    Real two_pi_over_lattice_parameter_ = 0.0;
    Matrix3x3 lattice_vectors_;
    /// Fractional positions and species of the atoms, in the crystal's order.
    std::vector<Vector3D> atomic_positions_;
    std::vector<int> species_type_indices_;
};

} // namespace q3
