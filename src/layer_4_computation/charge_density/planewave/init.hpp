#pragma once

/**
 * PlanewaveChargeDensitySymmetrizer — averages the reciprocal-space charge density
 * over the crystal's space group.
 *
 * It groups G-vectors into shells related by the symmetry operations, then for each
 * shell replaces every member's density component by the (phase-corrected) average
 * over the orbit. Non-symmorphic operations contribute the fractional-translation
 * phase. The symmetrizer owns only the precomputed symmetry operations and shells;
 * the plane-wave basis and the density are passed in per call.
 *
 * The operations are every one spglib reports. For a supercell that includes the
 * pure translations of the smaller cell it repeats: a 2x2x2 diamond cell has 384
 * operations, 48 rotations times 8 translations. spglib's list is closed under
 * composition, so the average is a projector onto densities invariant under the
 * whole group -- the translations remove every Fourier component the repeated
 * cell cannot carry, and the rotations act exactly as they do in the primitive
 * cell. What matters is that this is the same group the k-point mesh was reduced
 * with (SymmetryAnalyzer::reduce_kpoints_internal reads the same list): a reduced
 * wedge only stands in for the full mesh once the density is averaged over the
 * operations that produced it. Quantum ESPRESSO drops every translation-carrying
 * operation of a supercell because its own operation search is not guaranteed to
 * produce a group; carried over here, that left a 2x2x2 silicon cell averaged
 * over the 12-operation subgroup fixing the origin, which happens to leave a
 * shifted mesh invariant, so the supercell converged to the unsymmetrized-mesh
 * density and sat 1.4e-4 Ry above eight primitive cells (camsai/q3#6).
 *
 * A G-vector that every rotation leaves in place is a shell of its own, and
 * for it the average is a multiplication: by the mean of the operations'
 * fractional-translation phases, exp(2 pi i G.t). Those phases form a
 * character of the group, so the mean is exactly 1 or exactly 0. Where it is
 * 0 a screw axis or a glide forbids the component outright -- the systematic
 * absences of crystallography, (0, 0, l) with l odd under the 6_3 screw and
 * c-glides of wurtzite's P6_3mc -- and those components are set to zero rather
 * than averaged. Only a polar point group leaves a G-vector other than 0 in
 * place (inversion, or any rotation turning the axis over, pairs G with -G),
 * and only a translation along that axis makes the phases cancel. A wedge
 * k-point leaves such components nonzero whenever its time-reversal partner
 * lies outside its own star, as it does off wurtzite's k_z = 0 and 1/2 planes,
 * and skipping one-member shells -- as this class did until 2026-10-09 -- let
 * them through: the density lost the screw axis, and the levels it makes
 * degenerate at K, A, L and H came out split by meV (camsai/q3#24).
 */

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class PlanewaveChargeDensitySymmetrizer {
  public:
    /// Precompute the symmetry operations and G-vector shells for the given cell/basis.
    void build_shells(const Crystal& crystal_structure, const PlanewaveBasis& planewave_basis) {
        if (planewave_basis.is_gamma_only()) {
            symmetry_initialized_ = false;
            return;
        }

        symmetry_operations_ = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);
        log::info("     Density symmetrization: {} symmetry operations",
                  symmetry_operations_.size());

        int number_of_symmetry_operations = static_cast<int>(symmetry_operations_.size());

        if (number_of_symmetry_operations <= 1) {
            symmetry_initialized_ = false;
            return;
        }

        int number_of_g_vectors = planewave_basis.number_of_g_vectors();
        const auto& g_vectors = planewave_basis.g_vectors();

        // Build a hash map from Miller indices to G-vector index for fast lookup.

        // Key = (miller_index_1, miller_index_2, miller_index_3) packed into a single int64.
        auto miller_to_key =
            [](int miller_index_1, int miller_index_2, int miller_index_3) -> int64_t {
            constexpr int64_t offset = 1000;
            return (int64_t)(miller_index_1 + offset) * (2 * offset + 1) * (2 * offset + 1) +
                   (int64_t)(miller_index_2 + offset) * (2 * offset + 1) +
                   (int64_t)(miller_index_3 + offset);
        };

        std::unordered_map<int64_t, int> miller_index_map;
        miller_index_map.reserve(number_of_g_vectors);
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
            const auto& miller = g_vectors[g_vector_index].miller;
            miller_index_map[miller_to_key(miller.x, miller.y, miller.z)] = g_vector_index;
        }

        std::vector<bool> assigned(number_of_g_vectors, false);
        symmetry_shells_.clear();
        symmetry_shells_.reserve(number_of_g_vectors);

        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
            if (assigned[g_vector_index]) {
                continue;
            }

            SymmetryShell shell;
            const auto& miller = g_vectors[g_vector_index].miller;
            std::array<int, 3> miller_indices = {miller.x, miller.y, miller.z};

            for (int operation_index = 0; operation_index < number_of_symmetry_operations;
                 operation_index++) {
                std::array<int, 3> rotated = rotate_miller_indices(operation_index, miller_indices);
                int64_t key = miller_to_key(rotated[0], rotated[1], rotated[2]);
                auto it = miller_index_map.find(key);
                if (it == miller_index_map.end()) {
                    continue;
                }
                int rotated_index = it->second;
                if (assigned[rotated_index]) {
                    continue;
                }
                bool already_in_shell = false;
                for (int existing_index : shell.g_vector_indices) {
                    if (existing_index == rotated_index) {
                        already_in_shell = true;
                        break;
                    }
                }
                if (!already_in_shell) {
                    shell.g_vector_indices.push_back(rotated_index);
                    assigned[rotated_index] = true;
                }
            }

            if (!shell.g_vector_indices.empty()) {
                build_rotated_member_indices(g_vectors, shell);
                symmetry_shells_.push_back(std::move(shell));
            }
        }

        systematically_absent_g_vector_indices_.clear();
        for (const auto& shell : symmetry_shells_) {
            if (shell.g_vector_indices.size() != 1) {
                continue;
            }
            const auto& miller = g_vectors[shell.g_vector_indices[0]].miller;
            if (is_forbidden_by_a_translation({miller.x, miller.y, miller.z})) {
                systematically_absent_g_vector_indices_.push_back(shell.g_vector_indices[0]);
            }
        }

        symmetry_initialized_ = true;
    }

    /// Symmetrize the spin-resolved real-space charge density in place.
    /**
     * Symmetrize charge density given in G-space and produce both symmetrized
     * G-space and R-space outputs with a single G→R FFT.
     *
     * This avoids the R→G FFT that the real-space interface would require,
     * saving ~100ms per call for large grids. The caller provides the G-space
     * density (typically available from construct_density_from_states or
     * add_augmentation), and receives the symmetrized G-space density in-place
     * plus the symmetrized R-space density in the output parameter.
     */
    void symmetrize_from_g_space(const PlanewaveBasis& planewave_basis,
                                 std::vector<std::vector<Complex>>& charge_density_g_components,
                                 std::vector<std::vector<Real>>& charge_density_real,
                                 int number_of_spin_components) const {
        if (!symmetry_initialized_) {
            // No symmetry to apply — just do G→R transform
            for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
                planewave_basis.rho_g_to_r(charge_density_g_components[spin_index],
                                           charge_density_real[spin_index]);
            }
            return;
        }

        int number_of_symmetry_operations = static_cast<int>(symmetry_operations_.size());

        std::vector<bool> is_non_symmorphic(number_of_symmetry_operations, false);
        for (int operation_index = 0; operation_index < number_of_symmetry_operations;
             operation_index++) {
            const auto& fractional_translation =
                symmetry_operations_[operation_index].fractional_translation;
            is_non_symmorphic[operation_index] =
                (std::abs(fractional_translation.x) >
                     math::tolerances::zero_vector_magnitude_threshold ||
                 std::abs(fractional_translation.y) >
                     math::tolerances::zero_vector_magnitude_threshold ||
                 std::abs(fractional_translation.z) >
                     math::tolerances::zero_vector_magnitude_threshold);
        }

        for (const auto& shell : symmetry_shells_) {
            if (static_cast<int>(shell.g_vector_indices.size()) <= 1) {
                continue;
            }
            symmetrize_shell(planewave_basis,
                             number_of_spin_components,
                             shell,
                             is_non_symmorphic,
                             charge_density_g_components);
        }
        zero_systematically_absent_components(number_of_spin_components,
                                              charge_density_g_components);

        // Single G→R transform for each spin component
        for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
            planewave_basis.rho_g_to_r(charge_density_g_components[spin_index],
                                       charge_density_real[spin_index]);
        }
    }

    void symmetrize(const PlanewaveBasis& planewave_basis,
                    std::vector<std::vector<Real>>& charge_density_real,
                    int number_of_spin_components) const {
        if (!symmetry_initialized_) {
            return;
        }

        int number_of_symmetry_operations = static_cast<int>(symmetry_operations_.size());

        std::vector<std::vector<Complex>> charge_density_g_components(number_of_spin_components);
        for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
            planewave_basis.rho_r_to_g(charge_density_real[spin_index],
                                       charge_density_g_components[spin_index]);
        }

        std::vector<bool> is_non_symmorphic(number_of_symmetry_operations, false);
        for (int operation_index = 0; operation_index < number_of_symmetry_operations;
             operation_index++) {
            const auto& fractional_translation =
                symmetry_operations_[operation_index].fractional_translation;
            is_non_symmorphic[operation_index] =
                (std::abs(fractional_translation.x) >
                     math::tolerances::zero_vector_magnitude_threshold ||
                 std::abs(fractional_translation.y) >
                     math::tolerances::zero_vector_magnitude_threshold ||
                 std::abs(fractional_translation.z) >
                     math::tolerances::zero_vector_magnitude_threshold);
        }

        for (const auto& shell : symmetry_shells_) {
            if (static_cast<int>(shell.g_vector_indices.size()) <= 1) {
                continue;
            }
            symmetrize_shell(planewave_basis,
                             number_of_spin_components,
                             shell,
                             is_non_symmorphic,
                             charge_density_g_components);
        }
        zero_systematically_absent_components(number_of_spin_components,
                                              charge_density_g_components);

        for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
            planewave_basis.rho_g_to_r(charge_density_g_components[spin_index],
                                       charge_density_real[spin_index]);
        }
    }

    [[nodiscard]] bool is_initialized() const {
        return symmetry_initialized_;
    }
    [[nodiscard]] int number_of_shells() const {
        return static_cast<int>(symmetry_shells_.size());
    }
    [[nodiscard]] int number_of_symmetry_operations() const {
        return static_cast<int>(symmetry_operations_.size());
    }
    /// G-vectors whose component a screw axis or glide forbids; zero unless the group is polar.
    [[nodiscard]] int number_of_systematically_absent_g_vectors() const {
        return static_cast<int>(systematically_absent_g_vector_indices_.size());
    }

  private:
    struct SymmetryShell {
        std::vector<int> g_vector_indices;

        /**
         * Which member of this shell each symmetry operation carries each
         * member to: `rotated_member_indices[member][operation]`, or -1 when
         * the rotated G-vector is not in the basis.
         *
         * Built once with the shell, because it is a function of the Miller
         * indices and the point group and neither moves during an SCF. It
         * used to be rediscovered on every iteration by a linear search over
         * the shell for every (member, operation) pair -- O(N_ops * size^2)
         * per shell, per iteration, to recompute an answer that could not
         * have changed.
         */
        std::vector<std::vector<int>> rotated_member_indices;
    };
    std::vector<SymmetryOperation> symmetry_operations_;
    std::vector<SymmetryShell> symmetry_shells_;
    /// One-member shells whose translation phases cancel; zeroed after every average.
    std::vector<int> systematically_absent_g_vector_indices_;
    bool symmetry_initialized_{false};

    /**
     * Whether the group average of the component at `miller` is zero.
     *
     * The operations that leave G in place multiply its component by
     * exp(2 pi i G.t), and those phases are a character of that subgroup:
     * composing {R|t} with {R'|t'} adds R t' to t, and R^T G = G turns the
     * extra phase into exp(2 pi i G.t'). The mean of a character is 1 when it
     * is trivial and 0 otherwise, so the mean is compared against the midpoint
     * rather than any phase against a tolerance -- spglib reports translations
     * unrounded, and a long supercell's pure translations make phases as small
     * as 1/N of a turn.
     */
    [[nodiscard]] bool is_forbidden_by_a_translation(const std::array<int, 3>& miller) const {
        Complex phase_sum(0.0, 0.0);
        int fixing_operations = 0;
        for (int operation_index = 0;
             operation_index < static_cast<int>(symmetry_operations_.size());
             operation_index++) {
            if (rotate_miller_indices(operation_index, miller) != miller) {
                continue;
            }
            const auto& translation = symmetry_operations_[operation_index].fractional_translation;
            Real argument =
                constants::two_pi *
                (miller[0] * translation.x + miller[1] * translation.y + miller[2] * translation.z);
            phase_sum += Complex(std::cos(argument), std::sin(argument));
            fixing_operations++;
        }
        return fixing_operations > 0 && std::abs(phase_sum) < 0.5 * fixing_operations;
    }

    void zero_systematically_absent_components(
        int number_of_spin_components,
        std::vector<std::vector<Complex>>& charge_density_g_components) const {
        for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
            for (int g_vector_index : systematically_absent_g_vector_indices_) {
                charge_density_g_components[spin_index][g_vector_index] = Complex(0.0, 0.0);
            }
        }
    }

    std::array<int, 3> rotate_miller_indices(int operation_index,
                                             const std::array<int, 3>& miller) const {
        const auto& reciprocal_rotation = symmetry_operations_[operation_index].reciprocal_rotation;
        return {{reciprocal_rotation[0][0] * miller[0] + reciprocal_rotation[0][1] * miller[1] +
                     reciprocal_rotation[0][2] * miller[2],
                 reciprocal_rotation[1][0] * miller[0] + reciprocal_rotation[1][1] * miller[1] +
                     reciprocal_rotation[1][2] * miller[2],
                 reciprocal_rotation[2][0] * miller[0] + reciprocal_rotation[2][1] * miller[1] +
                     reciprocal_rotation[2][2] * miller[2]}};
    }

    /**
     * Fill one shell's rotation map: for each member and each operation, the
     * index *within this shell* of the member the operation rotates it to.
     *
     * A shell is closed under the group by construction, so a rotated
     * G-vector that is in the basis at all is in this shell -- which is what
     * lets the map be an index into the shell rather than into the basis, and
     * why the search it replaces was over the shell.
     *
     * -1 marks an operation whose rotated vector is not in the basis, the
     * same value the search produced when it found nothing, so the
     * symmetrization reads it unchanged.
     */
    void build_rotated_member_indices(const std::vector<GVector>& g_vectors,
                                      SymmetryShell& shell) const {
        const int shell_size = static_cast<int>(shell.g_vector_indices.size());
        const int number_of_symmetry_operations = static_cast<int>(symmetry_operations_.size());

        std::vector<std::array<int, 3>> shell_miller_indices(shell_size);
        for (int member_index = 0; member_index < shell_size; member_index++) {
            const auto& miller = g_vectors[shell.g_vector_indices[member_index]].miller;
            shell_miller_indices[member_index] = {miller.x, miller.y, miller.z};
        }

        shell.rotated_member_indices.assign(shell_size,
                                            std::vector<int>(number_of_symmetry_operations, -1));
        for (int member_index = 0; member_index < shell_size; member_index++) {
            for (int operation_index = 0; operation_index < number_of_symmetry_operations;
                 operation_index++) {
                const std::array<int, 3> rotated =
                    rotate_miller_indices(operation_index, shell_miller_indices[member_index]);
                for (int candidate_index = 0; candidate_index < shell_size; candidate_index++) {
                    if (rotated == shell_miller_indices[candidate_index]) {
                        shell.rotated_member_indices[member_index][operation_index] =
                            candidate_index;
                        break;
                    }
                }
            }
        }
    }

    void symmetrize_shell(const PlanewaveBasis& planewave_basis,
                          int number_of_spin_components,
                          const SymmetryShell& shell,
                          const std::vector<bool>& is_non_symmorphic,
                          std::vector<std::vector<Complex>>& charge_density_g_components) const {
        const auto& g_vectors = planewave_basis.g_vectors();
        int number_of_symmetry_operations = static_cast<int>(symmetry_operations_.size());
        int number_of_components = number_of_spin_components;
        int shell_size = static_cast<int>(shell.g_vector_indices.size());

        std::vector<bool> done(shell_size, false);

        for (int shell_member_index = 0; shell_member_index < shell_size; shell_member_index++) {
            if (done[shell_member_index]) {
                continue;
            }

            std::vector<Complex> averaged_density(number_of_components, Complex(0.0, 0.0));
            const std::vector<int>& rotated_shell_indices =
                shell.rotated_member_indices[shell_member_index];

            for (int operation_index = 0; operation_index < number_of_symmetry_operations;
                 operation_index++) {
                if (rotated_shell_indices[operation_index] < 0) {
                    continue;
                }

                int source_global_index =
                    shell.g_vector_indices[rotated_shell_indices[operation_index]];

                if (is_non_symmorphic[operation_index]) {
                    /**
                     * Correct formula: rho_sym(G) = (1/N) sum_op e^{+2pi i G_source.t}
                     * rho(G_source) where G_source = R_op^{-T} G (the reciprocal-space rotation of
                     * G).
                     *
                     * Derivation: the symmetry constraint rho(G) = e^{+2pi i G_source.t}
                     * rho(G_source) gives contribution rho(G_source) * e^{+2pi i G_source.t} =
                     * rho(G) for each op when the density is self-consistent. Phase uses G_source
                     * (not G_target).
                     */
                    const auto& source_miller = g_vectors[source_global_index].miller;
                    const auto& fractional_translation =
                        symmetry_operations_[operation_index].fractional_translation;
                    Real argument =
                        constants::two_pi * (source_miller.x * fractional_translation.x +
                                             source_miller.y * fractional_translation.y +
                                             source_miller.z * fractional_translation.z);
                    Complex phase_factor(std::cos(argument), std::sin(argument));
                    for (int spin_index = 0; spin_index < number_of_components; spin_index++) {
                        averaged_density[spin_index] +=
                            charge_density_g_components[spin_index][source_global_index] *
                            phase_factor;
                    }
                } else {
                    for (int spin_index = 0; spin_index < number_of_components; spin_index++) {
                        averaged_density[spin_index] +=
                            charge_density_g_components[spin_index][source_global_index];
                    }
                }
            }

            for (int spin_index = 0; spin_index < number_of_components; spin_index++) {
                averaged_density[spin_index] /= number_of_symmetry_operations;
            }

            for (int operation_index = 0; operation_index < number_of_symmetry_operations;
                 operation_index++) {
                if (rotated_shell_indices[operation_index] < 0) {
                    continue;
                }

                int target_global_index =
                    shell.g_vector_indices[rotated_shell_indices[operation_index]];

                if (is_non_symmorphic[operation_index]) {
                    /**
                     * Write-back: rho_sym(G_target) = rho_sym(G_ref) * e^{-2pi i G_target.t_map}
                     * where op_map takes G_ref -> G_target = R_map^{-T} G_ref.
                     */
                    const auto& target_miller = g_vectors[target_global_index].miller;
                    const auto& fractional_translation =
                        symmetry_operations_[operation_index].fractional_translation;
                    Real argument =
                        constants::two_pi * (target_miller.x * fractional_translation.x +
                                             target_miller.y * fractional_translation.y +
                                             target_miller.z * fractional_translation.z);
                    Complex phase_factor(std::cos(argument), -std::sin(argument));
                    for (int spin_index = 0; spin_index < number_of_components; spin_index++) {
                        charge_density_g_components[spin_index][target_global_index] =
                            averaged_density[spin_index] * phase_factor;
                    }
                } else {
                    for (int spin_index = 0; spin_index < number_of_components; spin_index++) {
                        charge_density_g_components[spin_index][target_global_index] =
                            averaged_density[spin_index];
                    }
                }
                done[rotated_shell_indices[operation_index]] = true;
            }
        }
    }
};

} // namespace q3
