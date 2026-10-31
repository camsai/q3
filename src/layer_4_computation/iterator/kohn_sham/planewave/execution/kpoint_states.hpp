#pragma once

/**
 * A planewave run's per-k-point results in global order.
 *
 * Under k-point pools each pool's iterator holds only its own k-point states,
 * so eigenvalues, occupations, weights and coordinates read straight off it
 * describe a fraction of the grid, in pool-local order. Everything that reports
 * them goes through here, which gathers one contribution per pool through the
 * execution policy; a serial policy hands the local data back unchanged.
 * Shared rather than repeated per flavor: the non-self-consistent flavor read
 * its eigenvalues straight off the iterator while it ran only serially, and
 * that copy would have reported one pool's k-points under MPI.
 */

#include <cstddef>
#include <vector>

#include "generated/esse/properties/non_scalar/eigenvalues_at_kpoints.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"

namespace q3 {

/**
 * ESSE's eigenvalues_at_kpoints record for the whole run: every k-point state,
 * with its coordinates (crystal), weight, eigenvalues and occupations. Q3
 * stores one spin channel per state (a spin-polarized run's states are its
 * k-points twice over, spin up first), so each entry holds one unlabelled
 * channel.
 *
 * Collective under MPI: every rank must call it.
 */
[[nodiscard]] inline generated::EigenvaluesAtKpoints assemble_global_eigenvalues_at_kpoints(
    const PlanewavePseudopotentialKohnShamSCFIterator& iterator) {
    const auto& eigenvalue_matrix = iterator.eigenvalues();
    int local_number_of_kpoint_states = eigenvalue_matrix.rows();
    int number_of_bands = eigenvalue_matrix.cols();
    /**
     * kpoint_grid() holds one entry per PHYSICAL k-point; eigenvalue_matrix /
     * occupation() are indexed by k-point STATE (physical kpoint * spin, see
     * build_global_kpoint_grid_from_input / accumulator.hpp for the same
     * convention) -- for a spin-polarized run number_of_kpoints above is the
     * state count, so kpoint_grid() lookups need the physical index instead.
     */
    int local_number_of_physical_kpoints = iterator.kpoint_grid().number_of_kpoints();
    int number_of_spin_components = iterator.quantum_system().number_of_spin_components;
    int number_of_values_per_state = 2 * number_of_bands + 4;
    std::vector<Real> local_state_data(static_cast<std::size_t>(local_number_of_kpoint_states) *
                                       number_of_values_per_state);
    for (int kpoint_index_state = 0; kpoint_index_state < local_number_of_kpoint_states;
         ++kpoint_index_state) {
        int local_physical_kpoint_index = kpoint_index_state % local_number_of_physical_kpoints;
        std::size_t state_offset =
            static_cast<std::size_t>(kpoint_index_state) * number_of_values_per_state;
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            local_state_data[state_offset + band_index] =
                eigenvalue_matrix(kpoint_index_state, band_index);
            local_state_data[state_offset + number_of_bands + band_index] =
                iterator.occupation(kpoint_index_state, band_index);
        }
        const auto& kpoint = iterator.kpoint_grid()[local_physical_kpoint_index];
        local_state_data[state_offset + 2 * number_of_bands] = kpoint.weight;
        local_state_data[state_offset + 2 * number_of_bands + 1] = kpoint.coordinates.x;
        local_state_data[state_offset + 2 * number_of_bands + 2] = kpoint.coordinates.y;
        local_state_data[state_offset + 2 * number_of_bands + 3] = kpoint.coordinates.z;
    }

    std::vector<Real> global_state_data =
        iterator.execution_policy().assemble_global_kpoint_state_data(
            local_state_data,
            local_number_of_physical_kpoints,
            number_of_spin_components,
            number_of_values_per_state);
    int global_number_of_kpoint_states =
        static_cast<int>(global_state_data.size()) / number_of_values_per_state;
    generated::EigenvaluesAtKpoints eigenvalues;
    eigenvalues.name = generated::EigenvaluesAtKpointsNameEnum::EIGENVALUES_AT_KPOINTS;
    eigenvalues.units = generated::EigenvaluesAtKpointsUnitsEnum::RY;
    for (int kpoint_index_state = 0; kpoint_index_state < global_number_of_kpoint_states;
         ++kpoint_index_state) {
        const auto state =
            global_state_data.begin() +
            static_cast<std::ptrdiff_t>(kpoint_index_state) * number_of_values_per_state;
        generated::EigenvaluesAtKpointsBandEnergies channel;
        channel.energies.assign(state, state + number_of_bands);
        channel.occupations =
            std::vector<double>(state + number_of_bands, state + 2 * number_of_bands);
        generated::EigenvaluesAtKpointsEigenvaluesAtKPoint kpoint;
        kpoint.weight = state[2 * number_of_bands];
        kpoint.kpoint =
            std::vector<double>(state + 2 * number_of_bands + 1, state + 2 * number_of_bands + 4);
        kpoint.eigenvalues = {channel};
        eigenvalues.values.push_back(kpoint);
    }
    return eigenvalues;
}

/**
 * Gather per-k-point-state blocks of `values_per_state` reals into global
 * order, as assemble_global_eigenvalues_at_kpoints() does for the eigenvalues.
 *
 * `local_blocks` holds one block per local k-point state, spin-major over the
 * pool's physical k-points. Collective under MPI.
 */
inline std::vector<std::vector<Real>>
assemble_global_kpoint_state_blocks(const PlanewavePseudopotentialKohnShamSCFIterator& iterator,
                                    const std::vector<std::vector<Real>>& local_blocks,
                                    int values_per_state) {
    const int local_number_of_physical_kpoints = iterator.kpoint_grid().number_of_kpoints();
    const int number_of_spin_components = iterator.quantum_system().number_of_spin_components;
    std::vector<Real> local_data;
    local_data.reserve(local_blocks.size() * static_cast<std::size_t>(values_per_state));
    for (const auto& block : local_blocks) {
        local_data.insert(local_data.end(), block.begin(), block.end());
    }
    const std::vector<Real> global_data =
        iterator.execution_policy().assemble_global_kpoint_state_data(
            local_data,
            local_number_of_physical_kpoints,
            number_of_spin_components,
            values_per_state);
    std::vector<std::vector<Real>> global_blocks(global_data.size() / values_per_state);
    for (std::size_t state_index = 0; state_index < global_blocks.size(); ++state_index) {
        global_blocks[state_index].assign(
            global_data.begin() + static_cast<std::ptrdiff_t>(state_index * values_per_state),
            global_data.begin() +
                static_cast<std::ptrdiff_t>((state_index + 1) * values_per_state));
    }
    return global_blocks;
}

} // namespace q3
