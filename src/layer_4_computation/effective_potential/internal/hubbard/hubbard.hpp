#pragma once

/**
 * DFT+U (Dudarev rotationally-invariant form), energy and potential.
 *
 * E_U = sum_{I,sigma} (U_I/2) Tr[n^{Isigma}(1 - n^{Isigma})]
 * V_U|psi> = sum_{I,m,m'} (U_I/2)(delta_mm' - 2 n^{Isigma}_{m'm}) |phi^I_m><phi^I_m'|psi>
 *
 * where n^{Isigma}_{mm'} = sum_{k,v} w_k f_{kv} <psi_{kv}|phi^I_m><phi^I_m'|psi_{kv}> is the
 * occupation matrix of the Hubbard-active atomic orbitals (indices m, m' run over the 2l+1
 * states of one atom's chosen shell). "Atomic" projector convention (raw atomic pseudo-orbitals,
 * no Lowdin orthogonalization) -- matches QE's HUBBARD (atomic) card.
 *
 * S.L. Dudarev et al., Phys. Rev. B 57, 1505 (1998).
 */

#include <cctype>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_0_core/math/linear_algebra/blas.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp"

namespace q3 {

/// One user-specified Hubbard-active shell (input.hubbard[i]).
struct HubbardSpecification {
    std::string species_symbol;
    std::string orbital_label; // matches the UPF PP_CHI label verbatim, e.g. "3D"
    Real u_rydberg;
};

/**
 * One seeding directive. Species is matched by ATOMIC_SPECIES symbol (the
 * same per-sublattice granularity the Hubbard specification itself uses, so
 * an antiferromagnet's Fe1/Fe2 can be seeded independently). Indices are
 * 0-based here; the input parsers convert from the 1-based drop-in and YAML
 * forms and bounds-check what they can before the manifold sizes are known.
 */
struct HubbardStartingEigenvalue {
    std::string species_symbol;
    int eigenvalue_index;   ///< 0-based position in the ascending eigenvalue list.
    int spin_index;         ///< 0-based spin channel.
    Real target_occupation; ///< [0, 1] in the per-spin convention.
};

/**
 * Build a seeding directive from the 1-based indices both input formats use,
 * validating once.
 *
 * Both parsers accept the same directive with the same numbering, so the
 * range rules and the 1-based to 0-based conversion live here rather than
 * being written out twice and drifting apart. Named arguments in, so a
 * reordering of the struct cannot silently mis-assign at a call site.
 */
[[nodiscard]] inline HubbardStartingEigenvalue
make_hubbard_starting_eigenvalue(const std::string& species_symbol,
                                 int one_based_eigenvalue_index,
                                 int one_based_spin_index,
                                 Real target_occupation,
                                 const std::string& context) {
    if (one_based_eigenvalue_index < 1)
        throw std::runtime_error(context +
                                 " eigenvalue index must be >= 1 (1-based, ascending), got " +
                                 std::to_string(one_based_eigenvalue_index));
    if (one_based_spin_index != 1 && one_based_spin_index != 2)
        throw std::runtime_error(context + " spin must be 1 or 2, got " +
                                 std::to_string(one_based_spin_index));
    if (target_occupation < 0.0 || target_occupation > 1.0)
        throw std::runtime_error(context + " value must be in [0, 1], got " +
                                 std::to_string(target_occupation));
    return {species_symbol,
            one_based_eigenvalue_index - 1,
            one_based_spin_index - 1,
            target_occupation};
}

/**
 * One resolved Hubbard-active manifold: a specific atom's specific shell, with the
 * indices of its 2l+1 orbitals into the full enumerate_atomic_orbitals() list.
 */
struct HubbardManifold {
    int atom_index;
    Real u_rydberg;
    int angular_momentum;
    std::vector<int> orbital_indices; // size 2*angular_momentum + 1
};

/**
 * Whether a user's shell specification names this orbital.
 *
 * The primary form is the dataset's own PP_CHI label, e.g. "3D". But labels are
 * optional in the UPF format and PseudoDojo's PAW files omit them entirely —
 * they give `l` and `occupation` and nothing else — which put every PAW dataset
 * out of reach of a label-only match. So a bare orbital letter ("d", "D", "f")
 * is also accepted and matches on angular momentum.
 *
 * That is unambiguous for what DFT+U is actually used for: a Hubbard U is
 * applied to the one partially-filled d or f shell, and a dataset does not ship
 * two of those. When the dataset does label its shells, the explicit label
 * still selects among them exactly as before.
 */
[[nodiscard]] inline bool shell_specification_matches(const std::string& specification,
                                                      const AtomicOrbitalLabel& label) {
    auto upper = [](std::string text) {
        for (char& character : text)
            character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
        return text;
    };
    std::string wanted = upper(specification);
    if (!label.shell_label.empty() && upper(label.shell_label) == wanted)
        return true;
    if (wanted.size() != 1)
        return false;
    static const std::string orbital_letters = "SPDFGH";
    size_t position = orbital_letters.find(wanted[0]);
    return position != std::string::npos && static_cast<int>(position) == label.angular_momentum;
}

/**
 * Resolves user Hubbard specifications against the crystal and the full atomic-orbital
 * label list, failing loudly (not silently) if a species or shell is not found.
 */
[[nodiscard]] inline std::vector<HubbardManifold>
resolve_hubbard_manifolds(const Crystal& crystal_structure,
                          const std::vector<AtomicOrbitalLabel>& orbital_labels,
                          const std::vector<HubbardSpecification>& specifications) {
    std::vector<HubbardManifold> manifolds;
    for (const auto& specification : specifications) {
        bool species_found = false;
        for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms(); atom_index++) {
            if (crystal_structure.element(atom_index) != specification.species_symbol) {
                continue;
            }
            species_found = true;

            /**
             * Group the matches by shell, then keep the most occupied one.
             *
             * A dataset may carry several channels of the same l — the Ni PAW
             * file has two d channels, occupancies 8 and 0 — so a bare orbital
             * letter matches more than one. The Hubbard U belongs to the
             * partially-filled valence shell, so the most occupied match is the
             * intended one; an explicit label still picks a specific shell, and
             * a genuine tie is refused below rather than resolved by file order.
             */
            std::vector<std::vector<int>> matched_shells;
            std::vector<Real> matched_occupations;
            int angular_momentum = -1;
            for (int label_index = 0; label_index < static_cast<int>(orbital_labels.size());
                 label_index++) {
                const auto& label = orbital_labels[label_index];
                if (label.atom_index != atom_index ||
                    !shell_specification_matches(specification.orbital_label, label))
                    continue;
                if (label.magnetic_quantum_number == 0) {
                    matched_shells.emplace_back();
                    matched_occupations.push_back(label.shell_occupation);
                }
                if (!matched_shells.empty())
                    matched_shells.back().push_back(label_index);
                angular_momentum = label.angular_momentum;
            }

            std::vector<int> orbital_indices;
            if (!matched_shells.empty()) {
                size_t best = 0;
                for (size_t candidate = 1; candidate < matched_shells.size(); candidate++)
                    if (matched_occupations[candidate] > matched_occupations[best])
                        best = candidate;
                int ties = 0;
                for (Real occupation : matched_occupations)
                    if (occupation == matched_occupations[best])
                        ties++;
                if (matched_shells.size() > 1 && ties > 1)
                    throw std::runtime_error(
                        "Hubbard: '" + specification.orbital_label + "' matches " +
                        std::to_string(matched_shells.size()) + " shells on atom " +
                        std::to_string(atom_index) +
                        " with equal occupancy, so the intended one cannot be inferred. "
                        "Name the shell explicitly.");
                orbital_indices = matched_shells[best];
            }
            if (orbital_indices.empty()) {
                std::string available;
                for (const auto& label : orbital_labels) {
                    if (label.atom_index != atom_index || label.magnetic_quantum_number != 0)
                        continue;
                    available +=
                        (available.empty() ? "" : ", ") +
                        (label.shell_label.empty()
                             ? "(unlabelled, l=" + std::to_string(label.angular_momentum) + ")"
                             : label.shell_label);
                }
                throw std::runtime_error(
                    "Hubbard: orbital shell '" + specification.orbital_label +
                    "' not found in the pseudopotential for species '" +
                    specification.species_symbol + "' (atom " + std::to_string(atom_index) +
                    "). Available: " + (available.empty() ? "none" : available) +
                    ". A bare orbital letter such as 'd' selects by angular momentum, which "
                    "is what unlabelled datasets need.");
            }
            /**
             * A single specification must not sweep up two shells of the same
             * l — an unlabelled dataset with two d channels would otherwise
             * silently build a 10-orbital "manifold" where 5 are meant.
             */
            if (static_cast<int>(orbital_indices.size()) != 2 * angular_momentum + 1) {
                throw std::runtime_error(
                    "Hubbard: '" + specification.orbital_label + "' matched " +
                    std::to_string(orbital_indices.size()) + " orbitals on atom " +
                    std::to_string(atom_index) +
                    ", but a shell with l=" + std::to_string(angular_momentum) + " has " +
                    std::to_string(2 * angular_momentum + 1) +
                    ". Name the shell explicitly to disambiguate.");
            }
            manifolds.push_back(
                {atom_index, specification.u_rydberg, angular_momentum, orbital_indices});
        }
        if (!species_found) {
            throw std::runtime_error("Hubbard: species '" + specification.species_symbol +
                                     "' specified in the hubbard input list is not present in "
                                     "the crystal");
        }
    }
    return manifolds;
}

/**
 * q_ij for one atom's projector channel pair, or zero when the species has none.
 *
 * The augmentation overlap is diagonal in the combined lm index -- the same rule
 * `augment_hubbard_projectors_with_overlap` applies when the SCF builds S|phi>. The
 * derivative kernels must agree with it or the force and stress differentiate a
 * different S than the energy used.
 *
 * The species gate is the part that must not be dropped.
 * `UltrasoftPlanewaveBasis::augmentation_overlap_integral` reaches the dataset through
 * `PseudopotentialSet::as_ultrasoft`, whose null check is an `assert` and therefore
 * compiled out in Release. A mixed cell -- one ultrasoft or PAW species alongside one
 * norm-conserving species -- with a Hubbard manifold walks the norm-conserving atoms
 * into that cast and dereferences null in exactly the builds users run. Debug builds
 * and the single-species test cells both hide it.
 *
 * Shared rather than repeated because it already went wrong: five hand-rolled copies of
 * this assembly existed across the force and stress kernels and only the one in
 * `forces/.../nonlocal.hpp` carried the gate.
 */
[[nodiscard]] inline Real
hubbard_augmentation_overlap(const UltrasoftPlanewaveBasis& ultrasoft_basis,
                             const NormConservingPlanewaveBasis& pseudopotential_basis,
                             int species_index,
                             int channel_row,
                             int channel_column) {
    if (!pseudopotential_basis.pseudopotential_set()[species_index].has_augmentation_charges()) {
        return 0.0;
    }
    if (pseudopotential_basis.projector_combined_lm_index(species_index, channel_row) !=
        pseudopotential_basis.projector_combined_lm_index(species_index, channel_column)) {
        return 0.0;
    }
    return ultrasoft_basis.augmentation_overlap_integral(
        species_index,
        pseudopotential_basis.projector_to_beta_index(species_index, channel_row),
        pseudopotential_basis.projector_to_beta_index(species_index, channel_column));
}

/**
 * All Hubbard projector overlaps <phi_j|chi_b> at one k-point, in a single ZGEMM.
 *
 * Both DFT+U consumers need the same contraction over plane waves: the SCF iterator needs
 * <phi_j|psi_b> to build the occupation matrix each iteration, and HubbardForceCalculator
 * needs that plus <phi_j|-i(k+G)_d|psi_b> for the three Cartesian gradients. Written as
 * scalar loops it is an O(N_pw * N_orb * N_band) nest running at dot-product speed with no
 * reuse of either operand; it is a matrix-matrix product, so it is one ZGEMM here. Both
 * call sites batch over ALL Hubbard orbitals at once (not per manifold), so the cost does
 * not grow with the number of Hubbard atoms beyond the GEMM's own N_orb dimension.
 *
 * This mirrors the Kleinman-Bylander projector overlaps in
 * layer_5_properties/forces/planewave/pseudopotential/internal/nonlocal.hpp, which has
 * always batched its own structurally identical "4 overlaps per band" the same way.
 *
 * `packed_orbitals` is column-major (number_of_plane_waves x total_hubbard_orbitals): the
 * Hubbard orbitals laid out contiguously by GLOBAL orbital index, i.e. manifold i's orbital
 * m occupies column `manifold_row_offsets[i] + m`.
 * `band_functions` is column-major (number_of_plane_waves x number_of_bands): the bands to
 * contract against -- psi itself, or psi pre-multiplied by -i(k+G)_direction.
 * `overlaps` is resized to and written as column-major
 * (total_hubbard_orbitals x number_of_bands), so element (j, b) is at
 * `j + b * total_hubbard_orbitals`.
 *
 * Under MPI band (G-vector) parallelism each rank passes only its local plane waves, making
 * the result a partial sum -- callers must reduce across the band-parallel group, exactly as
 * they did with the scalar loops.
 */
inline void compute_hubbard_projector_overlaps(int number_of_plane_waves,
                                               int total_hubbard_orbitals,
                                               int number_of_bands,
                                               const Complex* packed_orbitals,
                                               const Complex* band_functions,
                                               std::vector<Complex>& overlaps) {
    overlaps.assign(static_cast<size_t>(total_hubbard_orbitals) * number_of_bands,
                    Complex(0.0, 0.0));
    if (number_of_bands == 0 || total_hubbard_orbitals == 0 || number_of_plane_waves == 0) {
        return;
    }

    Complex alpha_blas(1.0, 0.0);
    Complex beta_blas(0.0, 0.0);
    blas::zgemm_("C",
                 "N",
                 &total_hubbard_orbitals,
                 &number_of_bands,
                 &number_of_plane_waves,
                 reinterpret_cast<const std::complex<double>*>(&alpha_blas),
                 reinterpret_cast<const std::complex<double>*>(packed_orbitals),
                 &number_of_plane_waves,
                 reinterpret_cast<const std::complex<double>*>(band_functions),
                 &number_of_plane_waves,
                 reinterpret_cast<const std::complex<double>*>(&beta_blas),
                 reinterpret_cast<std::complex<double>*>(overlaps.data()),
                 &total_hubbard_orbitals);
}

/**
 * The full-sphere overlap that a Gamma-only half-sphere sum stands for.
 *
 * At a single unshifted Gamma point Q3 keeps one G of each +-G pair, and its
 * half sphere *includes* G = 0 (the selection rule is `index_1 >= 0` when the
 * other two vanish). Every function this file contracts -- the Hubbard atomic
 * orbitals, the beta projectors inside S, and the wavefunctions -- is real in
 * real space, so f(-G) = conj(f(G)), and the -G half of sum_G conj(f) g is the
 * complex conjugate of the +G half. The full-sphere sum is therefore real and
 * equal to 2*Re[half], with the G = 0 term -- present once in the half sphere
 * and counted twice by the doubling -- removed again.
 *
 * A summand carrying a factor of G_alpha, which is every position-derivative
 * overlap here, vanishes at G = 0 and passes `g_zero_product = 0`. Getting that
 * distinction wrong is exactly what made four force terms wrong at once on
 * 2026-08-08 (KNOWN-ISSUES.md, "Gamma-only forces were wrong in four terms"),
 * and the KB nonlocal force and Hamiltonian apply both carry the same pair of
 * rules inline.
 */
[[nodiscard]] inline Complex gamma_only_full_sphere_overlap(Complex half_sphere_sum,
                                                            Real g_zero_product) {
    return {2.0 * half_sphere_sum.real() - g_zero_product, 0.0};
}

/**
 * Position of G = 0 in one k-point's plane-wave list, or -1 if it is absent.
 *
 * It is the first entry in practice (the list is sorted by |k+G|^2), but the
 * index is searched rather than assumed so that the Gamma correction is simply
 * skipped, not misapplied, anywhere the entry is missing.
 */
[[nodiscard]] inline int find_g_zero_plane_wave_index(const PlanewaveBasis& planewave_basis,
                                                      const std::vector<int>& plane_wave_indices,
                                                      int number_of_plane_waves) {
    for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
        if (planewave_basis.g_vectors()[plane_wave_indices[static_cast<size_t>(g_index)]]
                .magnitude_squared < math::tolerances::numerical_zero_threshold) {
            return g_index;
        }
    }
    return -1;
}

/**
 * Rewrite `compute_hubbard_projector_overlaps`' result in place as full-sphere values.
 *
 * `overlaps` is the column-major (total_hubbard_orbitals x number_of_bands)
 * block that function produces, and `packed_orbitals` / `band_functions` are
 * the operands it was given, unchanged. `band_functions_carry_g_component`
 * says whether the band operand was pre-multiplied by -i(k+G)_direction, in
 * which case the summand vanishes at G = 0 and no correction is subtracted.
 *
 * `g_zero_index` is `find_g_zero_plane_wave_index`'s answer; -1 skips the
 * subtraction. Both operations are linear in the G-sum, so applying them to a
 * rank's partial sum and reducing afterwards gives the same result as the
 * other order -- which is what lets a rank that does not hold G = 0 pass -1 and
 * contribute its half of the sum untouched.
 */
inline void
convert_gamma_only_projector_overlaps_to_full_sphere(int number_of_plane_waves,
                                                     int total_hubbard_orbitals,
                                                     int number_of_bands,
                                                     const Complex* packed_orbitals,
                                                     const Complex* band_functions,
                                                     int g_zero_index,
                                                     bool band_functions_carry_g_component,
                                                     std::vector<Complex>& overlaps) {
    for (int orbital_index = 0; orbital_index < total_hubbard_orbitals; orbital_index++) {
        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            Real g_zero_product = 0.0;
            if (!band_functions_carry_g_component && g_zero_index >= 0) {
                g_zero_product =
                    (std::conj(packed_orbitals[static_cast<size_t>(orbital_index) *
                                                   number_of_plane_waves +
                                               g_zero_index]) *
                     band_functions[static_cast<size_t>(band_index) * number_of_plane_waves +
                                    g_zero_index])
                        .real();
            }
            size_t flat_index =
                static_cast<size_t>(band_index) * total_hubbard_orbitals + orbital_index;
            overlaps[flat_index] =
                gamma_only_full_sphere_overlap(overlaps[flat_index], g_zero_product);
        }
    }
}

/**
 * Accumulates one (k-point, spin, band-set)'s contribution into a manifold's occupation
 * matrix. `overlaps(band_index, m)` = <phi_m|psi_{kv}> for this manifold's orbitals m,
 * this k-point, this spin. `occupation_matrix` is row-major manifold_size x manifold_size,
 * accumulated INTO (call once per k-point per spin; caller sums over all k-points).
 *
 * Kept as a scalar (band x m x m') nest deliberately: manifold_size is 2l+1, so it is 5 for
 * a d shell and 7 for an f shell and cannot grow. Measured against the equivalent ZGEMM
 * (n = O~^H O~ with O~ = sqrt(w) O) on OpenBLAS 0.3.26, this kernel costs 0.35 us at the
 * d-shell/10-band size and 20 us at f-shell/300-band, and the ZGEMM form is 1.5-2.9x faster
 * -- i.e. it would save single-digit microseconds per k-point/spin/manifold against an
 * overlap build (above) that costs milliseconds to seconds, while adding a weight-scaling
 * pass, a temporary, and the loss of this function's accumulate-in-place (beta=1) contract.
 */
inline void accumulate_hubbard_occupation_matrix(int number_of_bands,
                                                 int manifold_size,
                                                 const Matrix<Complex>& overlaps,
                                                 const std::vector<Real>& band_occupations,
                                                 Real kpoint_weight,
                                                 std::vector<Complex>& occupation_matrix) {
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        Real weight = kpoint_weight * band_occupations[band_index];
        if (weight <= 0.0) {
            continue;
        }
        for (int m = 0; m < manifold_size; m++) {
            Complex overlap_m = overlaps(static_cast<size_t>(band_index), static_cast<size_t>(m));
            for (int m_prime = 0; m_prime < manifold_size; m_prime++) {
                Complex overlap_m_prime =
                    overlaps(static_cast<size_t>(band_index), static_cast<size_t>(m_prime));
                occupation_matrix[static_cast<size_t>(m * manifold_size + m_prime)] +=
                    weight * std::conj(overlap_m) * overlap_m_prime;
            }
        }
    }
}

/// Result of the Dudarev energy/potential formula for one manifold, one spin channel.
struct HubbardEnergyAndPotential {
    Real energy = 0.0;
    std::vector<Complex> d_matrix; // row-major manifold_size x manifold_size
};

/**
 * Hellmann-Feynman gradient of E_U = (U/2)Tr[n(1-n)] w.r.t. one Cartesian component of one
 * manifold's own atom position, for one k-point/spin's contribution (caller sums over
 * k-points and spins). Derivation: dE_U/dtau = sum_mm' D_mm' dn_mm'/dtau (chain rule through
 * d/dn Tr[n(1-n)] = 1-2n, which is exactly D/U up to the 1/2 already in D's definition), and
 * dn_mm'/dtau expands via the product rule on n_mm' = <psi|phi_m><phi_m'|psi> into two terms
 * that collapse to 2*Re[...] using D's Hermiticity (D_mm' = conj(D_m'm), mirroring how the
 * energy/potential formula above is itself built from Hermitian n) -- exactly the same
 * collapse the Kleinman-Bylander nonlocal force
 * (forces/planewave/pseudopotential/internal/nonlocal.hpp) already relies on for its own,
 * structurally identical D_ab-weighted sum.
 *
 * `overlaps(band,m)` = <phi_m|psi_v> (as accumulate_hubbard_occupation_matrix's own
 * docstring); `direction_derivative_overlaps(band,m)` = <phi_m|-i(k+G)_direction|psi_v>,
 * built by pre-multiplying psi_v(G) by -i(k+G)_direction before the SAME overlap sum --
 * exactly nonlocal.hpp's own "component 1,2,3" construction for KB projectors, reused here
 * verbatim because Q3's atomic orbitals and KB projectors share the identical G-space
 * structure-factor convention exp(-i(k+G)*tau) (both built from
 * atomic_orbital_builder.hpp / projector.hpp's shared phase_factor.hpp), so the position
 * derivative of either has the identical -i(k+G) form.
 *
 * `d_matrix` is this manifold/spin's Dudarev D-matrix from compute_hubbard_energy_and_potential
 * (row-major manifold_size x manifold_size). Returns the scalar force contribution (Ry/bohr)
 * to add to this manifold's own atom, this Cartesian direction.
 */
[[nodiscard]] inline Real
accumulate_hubbard_force_gradient(int number_of_bands,
                                  int manifold_size,
                                  const Matrix<Complex>& overlaps,
                                  const Matrix<Complex>& direction_derivative_overlaps,
                                  const std::vector<Real>& band_occupations,
                                  Real kpoint_weight,
                                  const std::vector<Complex>& d_matrix) {
    Real force_contribution = 0.0;
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        Real weight = kpoint_weight * band_occupations[band_index];
        if (weight <= 0.0) {
            continue;
        }
        Complex band_sum(0.0, 0.0);
        for (int m = 0; m < manifold_size; m++) {
            Complex derivative_overlap_m = direction_derivative_overlaps(
                static_cast<size_t>(band_index), static_cast<size_t>(m));
            Complex derivative_overlap_m_conj = std::conj(derivative_overlap_m);
            for (int m_prime = 0; m_prime < manifold_size; m_prime++) {
                Complex overlap_m_prime =
                    overlaps(static_cast<size_t>(band_index), static_cast<size_t>(m_prime));
                band_sum += d_matrix[static_cast<size_t>(m * manifold_size + m_prime)] *
                            derivative_overlap_m_conj * overlap_m_prime;
            }
        }
        force_contribution += weight * 2.0 * band_sum.real();
    }
    return force_contribution;
}

/**
 * E_U = (U/2) Tr[n(1-n)]; V_mm' = U(delta_mm'/2 - n_m'm). `occupation_matrix` is row-major
 * manifold_size x manifold_size (n_mm' at index m*manifold_size + m').
 */
[[nodiscard]] inline HubbardEnergyAndPotential compute_hubbard_energy_and_potential(
    int manifold_size, Real u_rydberg, const std::vector<Complex>& occupation_matrix) {
    HubbardEnergyAndPotential result;
    result.d_matrix.assign(static_cast<size_t>(manifold_size) * manifold_size, Complex(0.0, 0.0));

    Real trace_n = 0.0;
    Real trace_n_squared = 0.0;
    for (int m = 0; m < manifold_size; m++) {
        trace_n += occupation_matrix[static_cast<size_t>(m * manifold_size + m)].real();
        for (int m_prime = 0; m_prime < manifold_size; m_prime++) {
            Complex element = occupation_matrix[static_cast<size_t>(m * manifold_size + m_prime)];
            trace_n_squared += std::norm(element); // |n_mm'|^2
        }
    }
    result.energy = 0.5 * u_rydberg * (trace_n - trace_n_squared);

    for (int m = 0; m < manifold_size; m++) {
        for (int m_prime = 0; m_prime < manifold_size; m_prime++) {
            Real kronecker_delta = (m == m_prime) ? 1.0 : 0.0;
            /**
             * Note the index order: n_m'm, not n_mm' (Dudarev's formula couples V_mm' to
             * the TRANSPOSE-indexed occupation-matrix element).
             */
            Complex occupation_m_prime_m =
                occupation_matrix[static_cast<size_t>(m_prime * manifold_size + m)];
            result.d_matrix[static_cast<size_t>(m * manifold_size + m_prime)] =
                u_rydberg * (0.5 * kronecker_delta - occupation_m_prime_m);
        }
    }
    return result;
}

} // namespace q3
