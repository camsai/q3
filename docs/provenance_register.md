# Provenance register

This register makes provenance review measurable. Each row is a module that
another computational materials science code also implements, and records
three things about it:

1. **Published sources** — the papers (and equations, where the code names
   them) the implementation is written from, as cited in Q3's own source today.
2. **Validation** — the gates and tests that check its numbers.
3. **Side-by-side review** — who read the implementation next to the code it
   might derive from, when, against what, and with what result.

The third column is the one no automated check can fill. A wording search finds
red-flag phrases; it does not find carried-over structure
([PROVENANCE.md § 4.4](../PROVENANCE.md#44-what-a-search-cannot-catch)). The
review procedure, and the rule that a review is a research session done by
someone who did not write the code, are in
[PROVENANCE.md § 3](../PROVENANCE.md#3-the-recipe).

## Register

Paths are relative to the repository root. "Not cited in the source" means the
module names its method but cites no paper for it; those gaps are listed under
[Citation gaps](#citation-gaps).

| Module                              | Code                                                                                                                                                                                                                         | Published sources                                                                                                                                                                                                                                                                                                        | Validation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Reviewer | Date | Compared against | Result           |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---- | ---------------- | ---------------- |
| PAW/USPP augmentation and D-matrix  | `src/layer_3_representation/pseudopotential/ultrasoft/`, `src/layer_3_representation/pseudopotential/paw/compensation_charge.hpp`, `src/layer_4_computation/paw/one_center/` (`d_matrix.hpp`, `energy.hpp`, `densities.hpp`) | Vanderbilt, PRB 41, 7892 (1990); Blöchl, PRB 50, 17953 (1994) (one-center split, multipole moments of the compensation charge); Kresse & Joubert, PRB 59, 1758 (1999) (compensation-charge shape, linearized frozen core). No equation numbers cited.                                                                    | `tests/unit/layer_4_computation/paw/one_center/test_energy_and_d_matrix.cpp` (`OneCenterDMatrixTest.IsTheDerivativeOfTheOneCenterEnergy`, a finite-difference gate; `OneCenterDMatrixTest.GgaDerivativeIdentityHolds`); `tests/unit/layer_3_representation/pseudopotential/planewave/ultrasoft/internal/test_charge_density_augmentation.cpp`; `tests/unit/layer_3_representation/pseudopotential/paw/test_compensation_charge.cpp`; `tests/validation/test_qe_energy_comparison.py` (`test_qe_energy_comparison`, `test_qe_paw_energy_comparison`, against Quantum ESPRESSO 7.5 + LibXC 7.0.0); `tests/validation/test_paw_irreducible_wedge_consistency.py` | —        | —    | —                | not yet reviewed |
| Force and stress augmentation terms | `src/layer_5_properties/forces/planewave/pseudopotential/ultrasoft/internal/augmentation.hpp`, `src/layer_5_properties/stress_tensor/planewave/pseudopotential/ultrasoft/internal/augmentation.hpp`                          | Vanderbilt, PRB 41, 7892 (1990) (the Q_ij(G) expansion the stress term differentiates). The force term is derived in its own header from the structure-factor derivative; no paper cited. The stress module names the Nielsen–Martin formalism but does not cite it.                                                     | `tests/validation/test_paw_pressure_and_gamma_forces.py` (pressure is −dE/dV of Q3's own energy, origin invariance of the stress, Γ-only forces are −dE/dx); `tests/validation/test_paw_forces_and_stress.py` (analytic forces against central finite differences on ultrasoft and PAW datasets, full zone and irreducible wedge)                                                                                                                                                                                                                                                                                                                             | —        | —    | —                | not yet reviewed |
| Symmetry                            | `src/layer_1_structure/symmetry/symmetry.hpp`, `src/layer_1_structure/symmetry/operation.hpp`, `src/layer_4_computation/paw/one_center/occupation_symmetrization.hpp`                                                        | Space-group detection is delegated to the spglib library; no paper cited for spglib or for Q3's own irreducible-wedge reduction and symmetrization.                                                                                                                                                                      | `tests/unit/layer_1_structure/symmetry/test_symmetry.cpp` (`ReduceKpointsInternalUsesActualCrystalSymmetryNotBravaisLattice` checks Q3's reduction against spglib's; vector symmetrization under reduced symmetry); `tests/unit/layer_1_structure/symmetry/test_operation.cpp`; `tests/validation/test_screw_axis_wedge_consistency.py`; `tests/validation/test_paw_irreducible_wedge_consistency.py`; `tests/validation/test_stress_baseline.py` (`test_low_symmetry_reduction_matches_the_explicit_full_zone`)                                                                                                                                              | —        | —    | —                | not yet reviewed |
| k-point grids                       | `src/layer_1_structure/reciprocal_space/kpoints_grid.hpp`, `src/layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp`                                                                                               | Monkhorst–Pack meshes, named but not cited. Neighbor shells: Marzari & Vanderbilt, PRB 56, 12847 (1997), Eq. (B1) (completeness condition).                                                                                                                                                                              | `tests/unit/layer_1_structure/reciprocal_space/test_kpoints_grid.cpp` (counts, weights summing to 2, points in the zone); `tests/unit/layer_1_structure/reciprocal_space/test_kpoint_neighbor_shells.cpp`; `tests/unit/layer_1_structure/symmetry/test_symmetry.cpp` (`ReducedGeneratedGridsPreserveDimensionsShiftsWeightsAndReduction`); `tests/validation/test_supercell_folding_consistency.py`                                                                                                                                                                                                                                                           | —        | —    | —                | not yet reviewed |
| Ewald                               | `src/layer_4_computation/effective_potential/internal/electrostatic/ewald_summation.hpp`, `src/layer_5_properties/forces/planewave/internal/ewald.hpp`, `src/layer_5_properties/stress_tensor/planewave/internal/ewald.hpp`  | Not cited in the source.                                                                                                                                                                                                                                                                                                 | `tests/unit/layer_4_computation/effective_potential/internal/electrostatic/test_ewald_summation.cpp` (`EwaldTest.SiliconEwaldEnergy` against a reference value; lattice-image invariance); `tests/unit/layer_5_properties/forces/planewave/internal/test_ewald.cpp`; `tests/unit/layer_5_properties/stress_tensor/planewave/internal/test_ewald.cpp`; `tests/validation/test_stress_baseline.py` (`test_ewald_tensor_is_the_gradient_of_the_ewald_energy`, `test_gamma_ewald_energy_and_stress_are_storage_invariant`, `test_low_symmetry_stress_and_every_component_match_quantum_espresso`)                                                                 | —        | —    | —                | not yet reviewed |
| Broyden mixing                      | `src/layer_4_computation/iterator/internal/mixing/reciprocal_space/broyden.hpp`, `src/layer_4_computation/iterator/internal/mixing/preconditioner/`                                                                          | D. D. Johnson, PRB 38, 12807 (1988) (modified Broyden); Kerker, PRB 23, 3082 (1981) (preconditioner); Raczkowski, Canning & Wang, PRB 64, 121101(R) (2001) (local Thomas–Fermi preconditioner). No equation numbers cited. Placing the preconditioner before the history is documented in the header as Q3's own choice. | `tests/unit/layer_4_computation/iterator/internal/mixing/reciprocal_space/test_broyden.cpp` (`ConvergesTowardFixedPoint`, `NonUniformMetricWeightsProduceHandVerifiedResult`, `TwoSpinChannelsConvergeWithMagnetizationMetric`); `tests/unit/layer_4_computation/iterator/internal/mixing/preconditioner/test_kerker.cpp`, `test_local_thomas_fermi.cpp`; end to end through every SCF case in `tests/regression/parity_matrix.yml` and `tests/validation/test_qe_energy_comparison.py`                                                                                                                                                                       | —        | —    | —                | not yet reviewed |
| Davidson                            | `src/layer_4_computation/diagonalizer/planewave/davidson.hpp`, `src/layer_4_computation/diagonalizer/planewave/internal/subspace_manager/`                                                                                   | Not cited in the source.                                                                                                                                                                                                                                                                                                 | No dedicated unit test of the iteration (`tests/unit/layer_4_computation/diagonalizer/planewave/test_factory.cpp` covers construction only). Checked end to end by `tests/validation/test_qe_energy_comparison.py`, `tests/validation/test_band_structure.py`, `tests/validation/test_gpu_energy_parity.py` and `tests/validation/test_mpi_band_parallelism.py`                                                                                                                                                                                                                                                                                               | —        | —    | —                | not yet reviewed |

## Citation gaps

Found while seeding the register. Each is a module that implements a published
method without citing it; the fix is a citation in the module's header, written
in the approved form of
[PROVENANCE.md § 4.1](../PROVENANCE.md#41-approved-forms), after checking the
reference against the paper itself.

| Module                         | Missing citation                    | Standard reference to check                                     |
| ------------------------------ | ----------------------------------- | --------------------------------------------------------------- |
| Ewald (energy, force, stress)  | The Ewald summation                 | P. P. Ewald, Ann. Phys. 369, 253 (1921)                         |
| k-point grids                  | Monkhorst–Pack meshes               | H. J. Monkhorst & J. D. Pack, PRB 13, 5188 (1976)               |
| Davidson                       | The block Davidson method           | E. R. Davidson, J. Comput. Phys. 17, 87 (1975)                  |
| Stress (augmentation and rest) | The Nielsen–Martin stress formalism | O. H. Nielsen & R. M. Martin, PRB 32, 3780 (1985)               |
| Symmetry                       | spglib                              | The spglib paper the library's documentation asks users to cite |
| All PAW/USPP rows              | Equation numbers for each term      | The papers already cited in the row                             |

The validation column has one gap of the same kind: the Davidson iteration has
no unit test of its own, so a defect in it is found only through end-to-end
energies.

## Recording a side-by-side review

A review is a research session ([PROVENANCE.md § 3](../PROVENANCE.md#3-the-recipe)),
done by someone who did not write the module. It may read GPL source; it never
opens proprietary source. It writes no code: a defect it finds is fixed later,
in a separate implementation session.

When the review is done, fill in the row's last four columns in a pull request
of its own:

-   **Reviewer** — the person, or for an AI session the person who ran it and
    the tool, so that "did not write the code" can be checked against the
    module's history.
-   **Date** — the date the review finished, as `YYYY-MM-DD`.
-   **Compared against** — the code and version read (for example "Quantum
    ESPRESSO 7.5"), or "published sources only" when no implementation was
    read. Name no proprietary engine here; a proprietary engine's source is
    never read, so it is never what a review compares against.
-   **Result** — `no structural correspondence found`, or `findings` with a
    link to the issue that tracks them. A finding is fixed by rewriting the
    code, never by deleting its provenance comment
    ([PROVENANCE.md § 4.3](../PROVENANCE.md#43-do-not-fix-this-by-deleting-citations)).

A row whose module is later rewritten goes back to `not yet reviewed`: the
review covered the code as it was, not the code as it is.

Adding a module is the same pull request that adds the module, with its review
columns at `not yet reviewed`.
