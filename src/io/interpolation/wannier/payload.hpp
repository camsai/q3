#pragma once

/**
 * The Wannier method's payload inside an interpolation record: WannierInputs
 * to bytes and back.
 *
 * Split from the envelope in `io/interpolation/checkpoint.hpp` because it is
 * the one part of that file that knows a method. The envelope holds what every
 * record has -- magic, version, method name, lattice, Fermi level -- and
 * dispatches to a payload; a second method adds a file beside this one and a
 * case there, and touches nothing else.
 *
 * `wannier/` is the METHOD, not the file format: `wannier90/` beside it holds
 * the interchange format that another code reads. This payload is Q3's own
 * `.q3ip` layout and is not interchangeable with anything.
 *
 * Layout, in order: the grid, band, trial-orbital and spin counts, the Fermi
 * level, the k-points, the neighbour directions, the overlaps, the
 * projections, the eigenvalues, the projectabilities, the trial-orbital
 * centres. Anything added here is a
 * format change and needs the envelope's version bumped: a field added without
 * one decodes as garbage in a consumer far from the mistake.
 */

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/interpolation/internal/binary_stream.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"

namespace q3::internal::wannier_payload {

using namespace q3::internal::interpolation_stream;

inline void write(std::ostream& stream, const WannierInputs& record) {
    for (int axis = 0; axis < 3; axis++) {
        write_scalar<int32_t>(stream, record.kpoint_grid_dimensions[axis]);
    }
    write_scalar<int32_t>(stream, record.number_of_bands);
    write_scalar<int32_t>(stream, record.number_of_trial_orbitals);
    write_scalar<int32_t>(stream, record.number_of_spin_components);
    write_scalar<double>(stream, record.fermi_energy_rydberg);

    write_scalar<int64_t>(stream, static_cast<int64_t>(record.kpoint_coordinates.size()));
    for (const Vector3D& kpoint : record.kpoint_coordinates) {
        write_vector3d(stream, kpoint);
    }

    write_scalar<int64_t>(stream, static_cast<int64_t>(record.neighbor_directions.size()));
    for (const KPointNeighborDirection& direction : record.neighbor_directions) {
        write_scalar<int32_t>(stream, direction.grid_step.x);
        write_scalar<int32_t>(stream, direction.grid_step.y);
        write_scalar<int32_t>(stream, direction.grid_step.z);
        write_vector3d(stream, direction.reciprocal_lattice_translation);
        write_scalar<double>(stream, direction.weight);
    }

    write_scalar<int64_t>(stream, static_cast<int64_t>(record.overlap_matrices.size()));
    for (const auto& per_direction : record.overlap_matrices) {
        write_nested_arrays(stream, per_direction);
    }
    write_nested_arrays(stream, record.projection_matrices);
    write_nested_arrays(stream, record.eigenvalues_rydberg);
    write_nested_arrays(stream, record.projectabilities);
    /**
     * Format version 2: the trial-orbital centres. Zero of them is a legitimate
     * count -- a record imported from projections and eigenvalues has none.
     */
    write_scalar<int64_t>(stream, static_cast<int64_t>(record.trial_orbital_centers_bohr.size()));
    for (const Vector3D& center : record.trial_orbital_centers_bohr) {
        write_vector3d(stream, center);
    }
}

inline WannierInputs read(std::istream& stream) {
    WannierInputs record;
    for (int axis = 0; axis < 3; axis++) {
        record.kpoint_grid_dimensions[axis] = read_scalar<int32_t>(stream);
    }
    record.number_of_bands = read_scalar<int32_t>(stream);
    record.number_of_trial_orbitals = read_scalar<int32_t>(stream);
    record.number_of_spin_components = read_scalar<int32_t>(stream);
    record.fermi_energy_rydberg = read_scalar<double>(stream);
    for (int axis = 0; axis < 3; axis++) {
        if (record.kpoint_grid_dimensions[axis] <= 0) {
            throw std::runtime_error("interpolation record: non-positive k-point grid dimension; "
                                     "the file is corrupt.");
        }
    }
    if (record.number_of_bands <= 0 || record.number_of_trial_orbitals <= 0) {
        throw std::runtime_error("interpolation record: non-positive band or trial-orbital "
                                 "count; the file is corrupt.");
    }

    const int64_t number_of_kpoints = read_scalar<int64_t>(stream);
    if (number_of_kpoints < 0) {
        throw std::runtime_error("interpolation record: negative k-point count; the file is "
                                 "corrupt.");
    }
    if (static_cast<int64_t>(record.kpoint_grid_dimensions[0]) * record.kpoint_grid_dimensions[1] *
            record.kpoint_grid_dimensions[2] !=
        number_of_kpoints) {
        throw std::runtime_error("interpolation record: the k-point count does not match the "
                                 "grid dimensions; the file is corrupt.");
    }
    require_bytes_available(stream,
                            static_cast<std::uint64_t>(number_of_kpoints) * 3 * sizeof(double));
    record.kpoint_coordinates.reserve(static_cast<std::size_t>(number_of_kpoints));
    for (int64_t index = 0; index < number_of_kpoints; index++) {
        record.kpoint_coordinates.push_back(read_vector3d(stream));
    }

    const int64_t number_of_directions = read_scalar<int64_t>(stream);
    if (number_of_directions < 0) {
        throw std::runtime_error("interpolation record: negative neighbour count; the file is "
                                 "corrupt.");
    }
    require_bytes_available(stream,
                            static_cast<std::uint64_t>(number_of_directions) *
                                (3 * sizeof(int32_t) + 3 * sizeof(double) + sizeof(double)));
    for (int64_t index = 0; index < number_of_directions; index++) {
        KPointNeighborDirection direction;
        direction.grid_step.x = read_scalar<int32_t>(stream);
        direction.grid_step.y = read_scalar<int32_t>(stream);
        direction.grid_step.z = read_scalar<int32_t>(stream);
        direction.reciprocal_lattice_translation = read_vector3d(stream);
        direction.weight = read_scalar<double>(stream);
        record.neighbor_directions.push_back(direction);
    }

    const int64_t overlap_kpoints = read_scalar<int64_t>(stream);
    if (overlap_kpoints < 0) {
        throw std::runtime_error("interpolation record: negative overlap count; the file is "
                                 "corrupt.");
    }
    require_bytes_available(stream, static_cast<std::uint64_t>(overlap_kpoints) * sizeof(int64_t));
    record.overlap_matrices.reserve(static_cast<std::size_t>(overlap_kpoints));
    for (int64_t index = 0; index < overlap_kpoints; index++) {
        record.overlap_matrices.push_back(read_nested_arrays<Complex>(stream));
    }
    record.projection_matrices = read_nested_arrays<Complex>(stream);
    record.eigenvalues_rydberg = read_nested_arrays<Real>(stream);
    record.projectabilities = read_nested_arrays<Real>(stream);
    const int64_t number_of_centers = read_scalar<int64_t>(stream);
    if (number_of_centers != 0 && number_of_centers != record.number_of_trial_orbitals) {
        throw std::runtime_error("interpolation record: " + std::to_string(number_of_centers) +
                                 " trial-orbital centres for " +
                                 std::to_string(record.number_of_trial_orbitals) +
                                 " trial orbitals; the file is corrupt.");
    }
    record.trial_orbital_centers_bohr.clear();
    for (int64_t index = 0; index < number_of_centers; index++) {
        record.trial_orbital_centers_bohr.push_back(read_vector3d(stream));
    }

    /**
     * Every per-k array either covers the k-point list or is absent -- an
     * imported record carries no overlaps -- and nothing in between; a partial
     * one would misassign k-points silently downstream.
     */
    auto check_coverage = [&](std::size_t count, const char* name) {
        if (count != 0 && count != record.kpoint_coordinates.size()) {
            throw std::runtime_error(std::string("interpolation record: ") + name + " cover " +
                                     std::to_string(count) + " k-points, the record has " +
                                     std::to_string(record.kpoint_coordinates.size()) +
                                     "; the file is corrupt.");
        }
    };
    check_coverage(record.overlap_matrices.size(), "the overlaps");
    check_coverage(record.projection_matrices.size(), "the projections");
    check_coverage(record.eigenvalues_rydberg.size(), "the eigenvalues");
    check_coverage(record.projectabilities.size(), "the projectabilities");

    /**
     * Shapes inside each k-point, too. A file whose layout drifted from this
     * reader's (a field added without a version bump, a truncated write)
     * decodes into plausible-looking counts with impossible blocks; refusing
     * here names the file, where a consumer's later out-of-range access
     * would name nothing.
     */
    if (record.number_of_bands <= 0 || record.number_of_trial_orbitals < 0) {
        throw std::runtime_error("interpolation record: impossible band or orbital count; the "
                                 "file is corrupt or predates this reader's format.");
    }
    const std::size_t projection_block =
        static_cast<std::size_t>(record.number_of_bands) * record.number_of_trial_orbitals;
    for (const std::vector<Complex>& block : record.projection_matrices) {
        if (block.size() != projection_block) {
            throw std::runtime_error("interpolation record: a projection block has " +
                                     std::to_string(block.size()) + " entries, expected " +
                                     std::to_string(projection_block) +
                                     "; the file is corrupt or predates this reader's format.");
        }
    }
    for (const std::vector<Real>& row : record.eigenvalues_rydberg) {
        if (row.size() != static_cast<std::size_t>(record.number_of_bands)) {
            throw std::runtime_error("interpolation record: an eigenvalue row has " +
                                     std::to_string(row.size()) + " entries, expected " +
                                     std::to_string(record.number_of_bands) +
                                     "; the file is corrupt or predates this reader's format.");
        }
    }
    return record;
}

} // namespace q3::internal::wannier_payload
