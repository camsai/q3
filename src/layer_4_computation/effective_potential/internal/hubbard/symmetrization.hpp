#pragma once

/**
 * Symmetrizing the DFT+U on-site occupation matrix.
 *
 * The occupation matrix is accumulated as a sum over the *irreducible* wedge:
 *
 *     n_mm' = Σ_{k∈IBZ} w_k Σ_v f_kv ⟨φ_m|ψ_kv⟩⟨ψ_kv|φ_m'⟩
 *
 * Weighted IBZ sums reproduce full-zone sums only for quantities that are
 * invariant under the point group. `n` is not: it carries the manifold's
 * angular indices, and each operation rotates them. Summing the wedge and
 * stopping leaves a matrix that is *not* symmetry-adapted, which is wrong even
 * though its trace is right — the trace is the one invariant the omission
 * preserves.
 *
 * The restoration is the group average
 *
 *     n_a  ←  (1/N_R) Σ_R  D(R)ᵀ · n_{R(a)} · D(R)
 *
 * with D(R) the rotation matrix of the real spherical harmonics of the
 * manifold's l, and R(a) = `atom_map[a]` the atom that a maps *to*. Read the
 * other way round, it is the statement the crystal makes: n_{R(a)} = D(R) n_a
 * D(R)ᵀ, so an already-symmetric n is a fixed point of this and of nothing
 * else — the same pairing, and the same argument for it, that
 * `SymmetryAnalyzer::symmetrize_vector` sets out for forces. Averaging the
 * on-site occupation matrix over the point group is the standard restoration in
 * DFT+U implementations, Quantum ESPRESSO's among them. The loop nesting and
 * the placement of the 1/N remain free choices the formula does not fix, and no
 * claim is made here about how any other implementation resolves them.
 *
 * **How this showed up.** Silicon's 3p manifold sits on a tetrahedral site, so
 * symmetry requires n = x·1 and therefore Tr[n²] = (Tr n)²/3. QE's numbers
 * satisfy that identity exactly; Q3's exceeded it by 0.154, and since
 * E_U = (U/2)(Tr n − Tr n²) that surplus fed straight into the energy — Q3 gave
 * E_U = 0.1204 Ry against QE's 0.1918 while Tr[n] agreed to 0.5%. Equal traces
 * and unequal energies is the signature of exactly this: the *structure* of the
 * matrix is wrong while its filling is right.
 *
 * Running the same case over the full zone (`disable_symmetry: true`), where no
 * symmetrization is needed because the k-sum performs it, gives 0.191758062
 * against QE's 0.19175802 — eight significant figures, and the confirmation
 * that nothing else was involved.
 *
 * D(R) is built numerically rather than from closed-form Wigner matrices,
 * using the defining relation and an angular quadrature that is exact for the
 * degree involved. That keeps this correct for any l the manifold can carry
 * without a table of hard-coded rotation formulae, and it reuses the same
 * grid and harmonics the PAW one-center engine already validates.
 * `build_real_harmonic_rotation_matrix` and `cartesian_rotation_of` live in
 * `layer_4_computation/paw/one_center/harmonic_rotation.hpp`, shared with the
 * PAW on-site occupation symmetrizer, which averages a different array over
 * the same group.
 */

#include <array>
#include <cmath>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/symmetry/operation.hpp"
#include "layer_4_computation/paw/one_center/harmonic_rotation.hpp"

namespace q3 {

/**
 * The other half of the group the wedge was reduced with: time reversal.
 *
 * `reduce_kpoints_internal` folds −Rk into Rk as well as Rk into k, so the
 * irreducible wedge is the quotient by the *magnetic* group {R} × {1, Θ}, not
 * by {R} alone. Averaging over the spatial operations therefore restores only
 * part of what the reduction removed.
 *
 * For a collinear Hamiltonian (real potential, real symmetric D matrices — Q3
 * has no spin-orbit or noncollinear path) Θ acts by complex conjugation:
 * ψ_{−k} = ψ_k*, so with real atomic orbitals ⟨φ_m|ψ_{−k}⟩ = ⟨φ_m|ψ_k⟩* and
 *
 *     n^{(−k)} = (n^{(k)})* = (n^{(k)})ᵀ    (n is Hermitian)
 *
 * A full-zone sum contains both members of every ±k pair and its imaginary
 * part cancels exactly; a wedge sum keeps only one and leaves the whole
 * antisymmetric imaginary part behind. The spatial average cannot remove it —
 * D(R) is real, so D·(iB)·Dᵀ stays antisymmetric-imaginary — which is why this
 * has to be a separate step rather than another operation in the loop above.
 *
 * The average over {1, Θ} is (n + nᵀ)/2 = Re n. Quantum ESPRESSO reaches the
 * same place from the other direction, accumulating only the real part of the
 * projector outer product into a real occupation array for the collinear case
 * (its complex branch is the noncollinear one, where Θ is not a symmetry of a
 * single spin channel).
 *
 * Unconditional, and safe to be: a full-zone n is already real to 1e-11, and
 * every Monkhorst-Pack mesh is closed under k → −k, so the projection is a
 * no-op exactly when it is not needed.
 *
 * **What it was worth.** Zincblende SiC, PAW, U = 3 eV on silicon with carbon
 * displaced off its site: the wedge left max|Im n| = 6.8e-3 against an
 * off-diagonal real part of 1.06e-2, moving Tr[n²] by 1.5e-4 and the total
 * energy by 7.7e-6 Ry away from the full-zone answer. Because the Hubbard
 * force contracts a *real* part (`accumulate_hubbard_force_gradient`'s
 * `2·Re[...]`, which is itself the time-reversal average) against a D matrix
 * built from a complex n, the two sides disagreed and the analytic force sat
 * 0.3% off Q3's own energy gradient.
 */
inline void symmetrize_hubbard_occupation_matrices_under_time_reversal(
    std::vector<std::vector<std::vector<Complex>>>& occupation_matrices) {
    for (auto& per_spin_matrices : occupation_matrices)
        for (auto& matrix : per_spin_matrices)
            for (auto& element : matrix)
                element = Complex(element.real(), 0.0);
}

/// One manifold's identity: which atom, which l, and where its rows live.
struct HubbardManifoldLayout {
    int atom_index{};
    int angular_momentum{};
    int manifold_size{};
};

/**
 * Replace each manifold's occupation matrix by the point-group average.
 *
 * `occupation_matrices[manifold][spin]` is row-major (manifold_size ×
 * manifold_size) and is modified in place. Manifolds are matched to their
 * symmetry images by (atom, l): an operation maps atom `a` to
 * `atom_map[a]`, and the manifold on that atom with the same l supplies the
 * matrix to rotate. A manifold whose image is not itself a Hubbard manifold —
 * possible only if the user applied U to some but not all symmetry-equivalent
 * atoms — is left alone, because averaging against a matrix that was never
 * accumulated would be worse than not averaging.
 *
 * **The pairing is the one thing here that a test has to pin.** Reindexing the
 * group sum by R → R⁻¹ turns this average into the one that pairs `atom_map`
 * with D(R) instead, with `atom_map` replaced by its inverse — so the two agree
 * *exactly* whenever every Hubbard atom's permutation is its own inverse, which
 * is always true of one or two Hubbard atoms and of one per species. Q3 shipped
 * the other pairing until 2026-09-16 and nothing caught it, because every DFT+U
 * case it is gated on is in one of those classes. Seeing the difference takes an
 * orbit of three or more equivalent atoms carried round a cycle, on sites whose
 * own symmetry still leaves the manifold free; that is what the cubic-perovskite
 * cases in `tests/unit/.../hubbard/test_hubbard.cpp` are for.
 */
inline void symmetrize_hubbard_occupation_matrices(
    const Crystal& crystal_structure,
    const std::vector<SymmetryOperation>& symmetry_operations,
    const std::vector<HubbardManifoldLayout>& layouts,
    std::vector<std::vector<std::vector<Complex>>>& occupation_matrices) {
    int number_of_operations = static_cast<int>(symmetry_operations.size());
    if (number_of_operations <= 1 || layouts.empty())
        return;

    /**
     * Rotation matrices are per (operation, l), not per manifold: two
     * manifolds sharing an l share the matrix, and building them is the
     * expensive part.
     */
    std::vector<int> distinct_angular_momenta;
    for (const auto& layout : layouts) {
        if (std::find(distinct_angular_momenta.begin(),
                      distinct_angular_momenta.end(),
                      layout.angular_momentum) == distinct_angular_momenta.end())
            distinct_angular_momenta.push_back(layout.angular_momentum);
    }
    std::vector<std::vector<std::vector<Real>>> rotation_matrices(number_of_operations);
    for (int operation_index = 0; operation_index < number_of_operations; operation_index++) {
        Matrix3x3 cartesian =
            cartesian_rotation_of(crystal_structure, symmetry_operations[operation_index]);
        rotation_matrices[operation_index].resize(distinct_angular_momenta.size());
        for (size_t slot = 0; slot < distinct_angular_momenta.size(); slot++)
            rotation_matrices[operation_index][slot] =
                build_real_harmonic_rotation_matrix(distinct_angular_momenta[slot], cartesian);
    }

    auto find_manifold = [&](int atom_index, int angular_momentum) {
        for (size_t index = 0; index < layouts.size(); index++)
            if (layouts[index].atom_index == atom_index &&
                layouts[index].angular_momentum == angular_momentum)
                return static_cast<int>(index);
        return -1;
    };

    auto original = occupation_matrices;

    for (size_t manifold_index = 0; manifold_index < layouts.size(); manifold_index++) {
        const auto& layout = layouts[manifold_index];
        int manifold_size = layout.manifold_size;
        size_t rotation_slot = static_cast<size_t>(std::find(distinct_angular_momenta.begin(),
                                                             distinct_angular_momenta.end(),
                                                             layout.angular_momentum) -
                                                   distinct_angular_momenta.begin());

        for (size_t spin_index = 0; spin_index < occupation_matrices[manifold_index].size();
             spin_index++) {
            std::vector<Complex> averaged(static_cast<size_t>(manifold_size) * manifold_size,
                                          Complex(0.0, 0.0));
            int contributing_operations = 0;

            for (int operation_index = 0; operation_index < number_of_operations;
                 operation_index++) {
                const auto& operation = symmetry_operations[operation_index];
                if (layout.atom_index >= static_cast<int>(operation.atom_map.size()))
                    continue;
                int image_atom = operation.atom_map[layout.atom_index];
                int image_manifold = find_manifold(image_atom, layout.angular_momentum);
                if (image_manifold < 0)
                    continue;

                const auto& rotation = rotation_matrices[operation_index][rotation_slot];
                const auto& source = original[image_manifold][spin_index];

                // (Dᵀ · n · D)_{m m'} = Σ_pq D_{p m} n_{pq} D_{q m'}
                for (int m = 0; m < manifold_size; m++) {
                    for (int m_prime = 0; m_prime < manifold_size; m_prime++) {
                        Complex accumulated(0.0, 0.0);
                        for (int p = 0; p < manifold_size; p++) {
                            Real left = rotation[static_cast<size_t>(p) * manifold_size + m];
                            if (left == 0.0)
                                continue;
                            for (int q = 0; q < manifold_size; q++) {
                                Real right =
                                    rotation[static_cast<size_t>(q) * manifold_size + m_prime];
                                if (right == 0.0)
                                    continue;
                                accumulated += left * right *
                                               source[static_cast<size_t>(p) * manifold_size + q];
                            }
                        }
                        averaged[static_cast<size_t>(m) * manifold_size + m_prime] += accumulated;
                    }
                }
                contributing_operations++;
            }

            if (contributing_operations == 0)
                continue;
            for (auto& element : averaged)
                element /= static_cast<Real>(contributing_operations);
            occupation_matrices[manifold_index][spin_index] = averaged;
        }
    }
}

} // namespace q3
