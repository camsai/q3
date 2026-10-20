#pragma once

/**
 * An orbital set measured in the S metric: S applied to each orbital, and the
 * Gram matrix O_ij = ⟨φ_i|S|φ_j⟩ that Löwdin orthogonalization needs when the
 * basis is not orthonormal.
 *
 * Every consumer of a non-orthonormal orbital set -- Wannier trial orbitals,
 * the PDOS / Löwdin population projectors -- needs the same two operations, so
 * they live beside the operator they apply rather than inside whichever
 * consumer happened to need them first. `overlap_operator` may be null, which
 * is the norm-conserving S = I case: the orbitals come back unchanged and the
 * Gram matrix is the plain ⟨φ_i|φ_j⟩ of
 * compute_orbital_overlap_matrix(), element for element.
 *
 * Ultrasoft and PAW pseudopotentials are what make the metric non-trivial:
 * both relax norm conservation, so the pseudo-orbitals are orthonormal with
 * respect to S = 1 + Σ_ij q_ij |β_i⟩⟨β_j| rather than to the identity
 * (Vanderbilt, PRB 41, 7892 (1990); Blöchl, PRB 50, 17953 (1994); Kresse &
 * Joubert, PRB 59, 1758 (1999)).
 */

#include <complex>
#include <cstring>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"

namespace q3 {

/**
 * Apply `overlap_operator` to every orbital in `orbitals`, batched through
 * one apply_batch() call. Returns `orbitals` unchanged (a copy) when
 * `overlap_operator` is null -- the norm-conserving S = I case, kept as an
 * explicit branch here rather than pushed onto every caller.
 */
[[nodiscard]] inline std::vector<std::vector<Complex>>
apply_overlap_operator_to_orbitals(const std::vector<std::vector<Complex>>& orbitals,
                                   const OverlapOperator* overlap_operator) {
    if (overlap_operator == nullptr) {
        return orbitals;
    }

    int number_of_orbitals = static_cast<int>(orbitals.size());
    if (number_of_orbitals == 0) {
        return {};
    }
    int number_of_plane_waves = static_cast<int>(orbitals[0].size());

    std::vector<Complex> flattened_orbitals(static_cast<std::size_t>(number_of_orbitals) *
                                            number_of_plane_waves);
    for (int orbital_index = 0; orbital_index < number_of_orbitals; orbital_index++) {
        std::memcpy(flattened_orbitals.data() +
                        static_cast<std::size_t>(orbital_index) * number_of_plane_waves,
                    orbitals[orbital_index].data(),
                    static_cast<std::size_t>(number_of_plane_waves) * sizeof(Complex));
    }

    std::vector<Complex> flattened_s_applied(flattened_orbitals.size());
    overlap_operator->apply_batch(number_of_plane_waves,
                                  number_of_orbitals,
                                  flattened_orbitals.data(),
                                  flattened_s_applied.data());

    std::vector<std::vector<Complex>> s_applied_orbitals(
        number_of_orbitals, std::vector<Complex>(number_of_plane_waves));
    for (int orbital_index = 0; orbital_index < number_of_orbitals; orbital_index++) {
        std::memcpy(s_applied_orbitals[orbital_index].data(),
                    flattened_s_applied.data() +
                        static_cast<std::size_t>(orbital_index) * number_of_plane_waves,
                    static_cast<std::size_t>(number_of_plane_waves) * sizeof(Complex));
    }
    return s_applied_orbitals;
}

/**
 * The orbital overlap matrix O_ij = ⟨φ_i|S|φ_j⟩, column-major, in the same
 * layout compute_orbital_overlap_matrix() (the S = I case) uses -- so both
 * feed the same lowdin_orthogonalize_orbitals(orbitals, overlap) entry point
 * unchanged. `overlap_operator` may be null (S = I).
 */
[[nodiscard]] inline std::vector<Complex>
compute_orbital_overlap_matrix_in_s_metric(const std::vector<std::vector<Complex>>& orbitals,
                                           const OverlapOperator* overlap_operator) {
    int number_of_orbitals = static_cast<int>(orbitals.size());
    if (number_of_orbitals == 0) {
        return {};
    }
    int number_of_plane_waves = static_cast<int>(orbitals[0].size());
    std::vector<std::vector<Complex>> s_applied_orbitals =
        apply_overlap_operator_to_orbitals(orbitals, overlap_operator);

    std::vector<Complex> overlap_column_major(static_cast<std::size_t>(number_of_orbitals) *
                                              number_of_orbitals);
    for (int column = 0; column < number_of_orbitals; column++) {
        for (int row = 0; row < number_of_orbitals; row++) {
            Complex value(0.0, 0.0);
            for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
                value += std::conj(orbitals[row][g_index]) * s_applied_orbitals[column][g_index];
            }
            overlap_column_major[static_cast<std::size_t>(row) + column * number_of_orbitals] =
                value;
        }
    }
    return overlap_column_major;
}

} // namespace q3
