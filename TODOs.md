# TODOs

Planned and in-progress work for Q3, grouped into the same three areas as
[FUNCTIONALITY.md](FUNCTIONALITY.md): **Science**, **Performance**, and
**Engineering**. When an item ships, it moves from here into the matching area of
[FUNCTIONALITY.md](FUNCTIONALITY.md).

**Queue convention:** unchecked boxes are the live work queue. Checked boxes
still present in this file are legacy implementation records retained for
their root-cause detail; they are not active plan items. New work continues to
follow the handoff-on-ship rule above rather than adding to that legacy set.

Each item carries a horizon tag. Because Q3 is developed at AI-agent velocity,
these are calibrated to that pace and gated primarily by **validation and review
bandwidth**, not implementation time: **[S]** short-term (days), **[M]** mid-term
(weeks), **[L]** long-term (weeks–months, usually because reference data or
hardware access is the blocker).

**Current scope:** NC and US pseudopotentials are validated on serial, MPI
(k-point pool + band), and GPU (CUDA, Metal) execution paths. PAW is validated
on serial for UPF v2 against QE, and
its UPF energy, forces, and stress are serial/MPI k-point-pool invariant. PAW
energy also has CPU/GPU parity coverage.
DFT+U atomic-projector and ortho-atomic energy, forces, and stress are implemented
for every dataset family. The remaining reference-evidence and backend limits are
listed in §1.3.

---

**Sequencing:** the authoritative feature sequencing (PAW → DFPT → hybrids; properties
native-first with living-ecosystem interop — ASE/Phonopy/phono3py/i-PI/PLUMED/icet;
native GW ladder with BerkeleyGW/Yambo exporters kept as validation instruments;
COMMIT/GATED/PARKED tiers with a review-unit budget) with per-item pinned validation
criteria, and the demand gates that schedule them, live in the maintainers'
planning tree.

## By horizon (index)

-   **[M] Next:** calculator decomposition (§1), the Wannier campaign's threshold re-examination (§1.3), library API & layer cleanup (§3), ESSE schema coverage (§3), CPU batched FFTs (§2), GPU-resident pipeline + CUDA kernels (§2), move hot kernels to .cpp (§3.3), JAX primitives (§4)
-   **[L] Later:** PAW parallel-backend validation (§1), MPI+GPU / NCCL (§2), HIP/SYCL hardware validation (§2), benchmarks (§2), advanced functionals (§1)

**Recently shipped 2026-08-08**: spin-polarized ultrasoft and PAW (the moment
had been collapsing silently — Fe
ultrasoft 0.0144 → 2.2482 μB vs QE); PAW forces and stress; Gamma-only forces, which were wrong in four terms for
_every_ pseudopotential family; DFT+U for ultrasoft and PAW.

**Recently shipped** (now in [FUNCTIONALITY.md](FUNCTIONALITY.md) §2): MPI
k-point pool + band parallelism; CUDA / Metal / HIP / SYCL GPU backends with
mixed precision; wavefunction-specific FFT grid; ZGEMM-batched KB projectors;
Kerker residual preconditioning; OpenMP threading of the hot loops; kbar constant
(`RYDBERG_TO_KILOBAR`); SCF iterator factory rename; DFT+U ortho-atomic
projectors; JSON output units &
validation; Broyden G-space BLAS acceleration; namelist vs YAML parser parity
for spin/smearing; deterministic FFT planning knob (+ tightened Fermi-Dirac
MPI tolerance); input/output reference doc; Makov-Payne isolated-system
correction (both orders); Martyna-Tuckerman isolated-system correction
(post-SCF evaluation); PAW serial SCF for UPF v2 datasets.

---

## 1. Science (Materials Science)

### 1.1 PAW (end-to-end)

PAW dispatch is explicit and never falls through to NC/US behavior.

-   [x] **[L]** **PAW data model** — `PawPseudopotential` extends `UltrasoftPseudopotential` with partial waves, all-electron/pseudo core data, occupations, one-center potentials, and compensation multipoles. `PawUPFv2Reader` consumes UPF v2 PAW sections. UPF v1 PAW remains rejected
-   [x] **[L]** **PAW representation & SCF** — plane-wave overlap/augmentation, one-center Hartree/XC and D-matrix terms, compensation-charge synthesis, and dataset-specific one-center conventions (absolute for QE/UPF; others registered by build-time add-ons) are active in the serial Kohn-Sham iterator
-   [x] **[L]** **PAW properties** — forces and stress complete 2026-08-08. Two force terms had been missing (the augmentation dQᵢⱼ/dR contribution was assembled in the wrong index space, and the generalized-eigenvalue `−ε_nk q_ij` term was absent from the force while present in the stress); the core-correction stress separately had a translation-invariance bug that made the pressure depend on the basis origin. Every term was gated against the numerical derivative of Q3's own energy
-   [x] **[L]** **PAW validation** — Si, Cu, and diamond UPF cases pass the LibXC-enabled QE gate at 1 mRy/atom
-   [x] **[L]** **PAW spin** — fixed 2026-08-08. Four defects, of which the dominant was that the compensation charge was split between spin channels by the ratio of the _smooth_ densities, forcing the on-site moment to match the interstitial one
-   [x] **[M]** **Symmetrize the PAW on-site occupations** — done 2026-09-16. ρ_ij was
        accumulated over the irreducible k-points and handed to the one-centre term
        unsymmetrized, so a symmetry-reduced mesh split degenerate levels (diamond's Γ
        valence maximum by 19 meV, copper's 3p by 0.277 eV) and moved the total energy
        against the full mesh (1.9–2.5e-4 Ry). `OnSiteOccupationSymmetrizer` now averages
        each atom's ρ_ij over the point group with its channels' real-harmonic rotations
        and the permutation of equivalent atoms, standing down wherever the density
        symmetrizer does. Diamond at its own 60/480 Ry agrees with the full mesh to
        **2.6e-10 Ry**, silicon PAW to 2e-11 Ry, copper's 3p triplet is degenerate, and the
        ultrasoft and norm-conserving controls are unchanged. Gated by
        `tests/validation/test_paw_irreducible_wedge_consistency.py` and
        `tests/unit/layer_4_computation/paw/one_center/test_occupation_symmetrization.cpp`
-   [x] **[S]** **Rotate the DFT+U occupation matrix with the other end of the atom map** —
        done 2026-09-16. `symmetrize_hubbard_occupation_matrices` averaged
        `D(R) · n_{atom_map[a]} · D(R)ᵀ` where the crystal's own covariance calls for
        `D(R)ᵀ · n_{atom_map[a]} · D(R)`, the pairing `SymmetryAnalyzer::symmetrize_vector`
        uses for forces and the PAW on-site occupations use. The two agree exactly whenever
        every Hubbard atom's permutation is its own inverse, which covers every DFT+U case
        Q3 is gated on — so nothing shipped moved, and the whole DFT+U validation set
        confirms it. On a cubic perovskite with U on the three cycled oxygens the old
        average left a covariance violation of 0.272 and moved 0.114 on a second pass;
        both are now at 1e-16, gated by four cases in
        `tests/unit/layer_4_computation/effective_potential/internal/hubbard/test_hubbard.cpp`
-   [ ] **[L]** **Advanced functionals for PAW** — the missing meta-GGA kinetic-energy-density one-center terms

### 1.2 Electronic structure

-   [ ] **[M]** **Spherical harmonics at degrees 5 and 6, for f-channel datasets** —
        `compute_real_spherical_harmonic`
        ([harmonics.hpp](src/layer_0_core/math/spherical/harmonics.hpp)) has closed forms to
        degree 4 and now throws above it, so a dataset with f projectors is refused at
        `generate_gaunt_coefficients` rather than run on a table whose collocation matrix is
        singular in the rows the missing harmonics should fill. Lifting the limit means
        harmonics to degree 6, either as two more closed-form blocks or by routing the scalar
        function through the same normalized-Legendre recursion
        `compute_real_spherical_harmonics_batch` already uses for arbitrary degree — the
        recursion is the smaller change, but it would move every l ≤ 4 value in its last bits,
        so it is a `numerics-change` on every ultrasoft and PAW result and wants the parity
        matrix. Until then Q3 supports projector channels up to d.

-   [x] **[M]** **Validate ultrasoft LSDA vs QE** — done 2026-08-08 with GBRV `fe_pbe_v1.5`, the only magnetic ultrasoft dataset available. It found the path was _broken_, not merely unvalidated: 0.0144 μB and 473 meV/atom off, now **2.2482 μB** and **0.06 meV/atom** against QE
-   [x] **[M]** **Split GGA vs LDA** effective-potential paths into separate calculator classes + factory — done: `requires_density_gradient()` is now consulted only by two factories, `create_exchange_correlation_potential_calculator()` (effective potential; local-density vs gradient-corrected V_xc strategies under `effective_potential/planewave/internal/exchange_correlation_potential/`) and `create_exchange_correlation_gradient_stress_calculator()` (stress; Null object for local functionals under `stress_tensor/planewave/internal/exchange_correlation_gradient/`). NLCC subclassing kept on its own axis via `ExchangeCorrelationStressDensitySource`. Gate met: PZ, PBE and Ni-LSDA-PBE runs bit-identical to the pre-decomposition build (every results-JSON numeric incl. stress, plus `charge-density.dat` and `effective-potential.q3cp` bytes, under deterministic FFT planning)
-   [ ] **[L]** **Advanced functionals & smearing** — meta-GGA, non-collinear
-   [ ] **[S]** **`occupations = 'from_input'`** — fixed per-band (and per-spin) occupations
        read from QE's `OCCUPATIONS` card; the card is refused at parse
        time today. Needs an `InputData` occupation table, a fixed-occupation scheme that
        bypasses the Fermi-level search, and a QE comparison (`requires_qe`) on an atom or
        molecule in a box. (Methfessel-Paxton order > 1, the other half of this item, shipped —
        see [FUNCTIONALITY.md](FUNCTIONALITY.md) §1.)
-   [ ] **[M]** **Per-atom `starting_magnetization`** — the initial moment is per species
        (`InputData::starting_magnetization`, parallel to `species`), so QE and YAML users
        must split the species of a same-species antiferromagnet (`Cr1`/`Cr2`). Needs a
        per-atom initial-moment vector carried into the initial spin density.

### 1.3 Properties

-   [x] **[S]** **Validate the localizer against Wannier90's converged spreads** — done,
        `tests/validation/test_wannier90_parity.py` (tier 3, monthly; skips without the 4.0.1 oracle):
        on Q3's own exported matrices, Wannier90 reaches the same Ω_I to 1e-6 A² or better on
        silicon, GaAs, copper and Si₃C, the same Ω to 1e-6 (isolated) / 8e-4 A² (the
        disentangled alloy), and the same centres. Two things it took to get there are recorded
        in that file's docstring: Wannier90 4.0's `dis_proj_min/max` do not freeze without
        `dis_froz_proj`, and Q3's guiding centres had to become the trial-orbital positions.
-   [x] **[S]** **Why the alloy localizes to Ω_D = 533 bohr²** — it did not have to. The
        guiding centres were the starting gauge's own Berry-phase centres, which on this
        structure sit on inconsistent branches across k and land 5 Å from any atom; guiding
        to them kept them. Guiding to the trial-orbital positions instead (record format 2,
        `WannierInputs::trial_orbital_centers_bohr`) takes the alloy from Ω = 170.6 to 26.0 A²,
        against Wannier90's 26.0 on the same matrices, with Ω_D ≈ 0.08 A²
-   [ ] **[S]** **Tighten the loose off-mesh thresholds** — with the localizer validated, the
        150 meV guard in
        `tests/validation/test_wannier_interpolation.py::test_interpolated_bands_track_the_direct_calculation_along_a_path`
        and the loose bound in `tests/validation/test_automatic_wannier.py` are what remains of
        the localization item above; both were development-configuration guards, not accuracy
        gates, and should now be set from measured numbers
-   [ ] **[M]** **Route `yaml.hpp`'s `parse_interpolation` through the generated
        `InputControlInterpolation`** — the manifest's `control.interpolation` defaults live
        in `schemas/q3/input.json` and the engine's in `specification.hpp`;
        `tests/python/test_interpolation_settings.py` holds the two equal, which closes the
        drift, but the parser still copies the keys by hand. The generated struct types
        `frozen_window_above_fermi_ev` as `nlohmann::json` and its vectors as `int64_t`, so
        routing the parser through it today adds a second conversion layer (~250 lines)
        rather than deleting one; worth doing once the generator learns a `oneOf` of
        number-or-word and integer widths.
-   [ ] **[M]** **Re-examine the Wannier accept thresholds against the campaign table** — the
        protocol around the automatic path is built (`mesh_for_spacing`, the `quality` block,
        `frozen_window_above_fermi_ev: auto` with the singular-gauge exclusion,
        `q3.workflows.wannier.run_automatic_wannier_workflow` with the band distance η and the
        retry ladder, the campaign `scripts/validation/run_wannier_campaign.py`), but its 10 / 100 meV gates were set
        before any automatic model existed. The campaign
        (`python scripts/validation/run_wannier_campaign.py`, or
        `pytest tests/validation/test_wannier_campaign.py --tier3`, which the scheduled
        workflow runs monthly and whose table it keeps)
        has run its 40-row roster, the augmented rows rerun on the trial-shell rule and the
        PAW NSCF fix (2026-09-14): 34 accepted at η 0.69–9.70 meV, 6 rejected — every
        norm-conserving copper geometry at 87.6–254 meV on its 4p-less dataset, and copper
        PAW, whose localisation stalls on every attempt (218–302 meV). Nothing lands between
        9.7 and 87.6 meV, so the 10 meV gate is not exercised by that roster, while the 100 meV
        η_max gate is (seven accepted rows at 63–82 meV). Re-set both from the table — leaving
        copper ultrasoft's accept out until its localisation reaches Wannier90's minimum — and
        move the protocol plan to `implemented/` on it, not before
-   [ ] **[S]** **Semicore shells out of the trial set, and an SCDM rung** — the trial-shell
        rule (`q3.workflows.wannier.trial_shells`) leaves out an augmented dataset's repeated
        empty channels but keeps every occupied shell, semicore included, so gallium arsenide
        keeps `bands: [19, 22]` as an override and copper's norm-conserving file carries 3s and
        3p into the model; the rule already identifies those shells, and leaving them out also
        needs the band window to start above them. No shipped dataset lacks atomic
        wavefunctions, so the SCDM fallback of the parent plan stays unbuilt until one does
-   [ ] **[S]** **Gate the 8×8×8 Wannier result** — the 10 meV headline was measured by hand
        (`run_interpolation_workflow(..., grid=(8, 8, 8))` against QE's direct bands); the committed fixture and the validation
        suite use the 4×4×4 development grid, so nothing notices if it regresses. Add the 8×8×8
        case as a tier-2 test with its own reference
-   [ ] **[S]** **Cross-validate the ultrasoft/PAW Wannier kernels** — the NSCF hook dispatches
        augmented datasets to the S-metric projections and the augmentation term in M_mn(k,b),
        which pass their internal identities (b→0 normalization, singular values bounded by 1
        up to the radial-table interpolation slack); only the norm-conserving path is compared
        against `pw2wannier90.x`. **The capture now reaches `build_ultrasoft`** — it had
        downcast the iterator's by-value base quantum system rather than the polymorphic
        snapshot the factory built, so the cast was null on every run and every augmented
        dataset was refused before the builder was entered; the snapshot also needed the
        electronic state copied into it, since it carries the pseudopotential data and the
        by-value member carries the wavefunctions. On a capture whose PAW NSCF carries D's
        one-centre term (2026-09-14), silicon agrees with Wannier90 on both augmented
        families -- Ω_I, Ω and the centres -- and copper does not: on ultrasoft Q3's
        localisation stalls at Ω 9.74 Å² where Wannier90 reaches 7.91, and copper PAW's
        eighteen functions stall both localisers and put Ω_I 7.7e-4 Å² apart. Both copper
        cases are xfail with the numbers recorded. Close it by making Q3's localiser reach
        Wannier90's minimum on copper ultrasoft, and by posing copper PAW on the trial
        shells the automatic protocol uses before comparing it again — see KNOWN-ISSUES.md
        § Wannier interpolation
-   [ ] **[S]** **Cross-validate the ultrasoft/PAW PDOS projections** — the NSCF
        projection hook now takes `|<phi'_i|S|psi_nk>|^2` in the S metric for datasets
        that carry augmentation charges, and Löwdin-orthogonalizes the atomic orbitals
        in the same metric. Its own sum rule holds — the Löwdin charges account for the
        valence electron count up to the basis-set spilling, checked on a silicon
        ultrasoft and a silicon PAW case in `tests/validation/test_pdos_s_metric.py` —
        but only the norm-conserving path is compared against `projwfc.x`. The ultrasoft
        comparison is written
        (`test_qe_projwfc_comparison.py::test_si_ultrasoft_pdos_matches_qe_projwfc`) and
        self-skips without the pinned QE LibXC build; run it on a machine that has one,
        and add a PAW case once a dataset QE accepts is committed. The parity matrix's
        `silicon-projections-ultrasoft` row now holds these weights fixed, which says
        they do not move, not that they agree with `projwfc.x`
-   [ ] **[S]** **Wannier interface completeness** — carry `missing_g_vector_count` (computed
        per (k,b) by the overlap kernel, dropped by the builder) into the record as the
        cutoff-sphere-edge diagnostic; trial orbitals with l > 0 (the hydrogenic set is s-only;
        the real spherical harmonics and the i^l convention already exist in the atomic-orbital
        builder); spin-polarized and Γ-only interface records (both refused today — the first needs a
        per-spin record layout, not just plumbing)
-   [ ] **[M]** **A second interpolation type: Fourier star-function fitting** — the umbrella
        (`control.interpolation`, `BandInterpolator`, the interpolation record, `q3_utility interpolate`)
        has one method today, and is only honest once a method of a different KIND exists. Next:
        fit each band's eigenvalues with symmetry-adapted star functions from the irreducible-wedge
        eigenvalues (Shankland 1971; Koelling & Wood 1986; Pickett, Krakauer & Allen, PRB 38, 2721
        (1988); the BoltzTraP method). Gauge-free, metals are fine, and every existing SCF already
        produces its input, so no interface pass is needed; bands are interpolated by index, so
        crossings are smoothed rather than resolved — right for DOS and transport, wrong for band
        topology. Small (a few hundred lines; spglib already provides the operations for the
        stars). After it, in this order: projected-atomic-orbital tight binding (Agapito, Curtarolo
        & Buongiorno Nardelli, PRB 88, 165127 (2013), reusing the S-metric projections and the
        pseudo-atomic orbital set) and Shirley's optimal basis (PRB 54, 16464 (1996); Prendergast &
        Louie, PRB 80, 235126 (2009), gauge-free and entangled manifolds without disentanglement).
        Deliberately NOT part of PR #144; sequenced as the second method in the
        interpolation-umbrella plan of 2026-09-05, §5.
-   [ ] **[M]** **Tetrahedron DOS integration** — Q3 has no tetrahedron integrator, so the
        Wannier program's stronger DOS gate (interpolated DOS(E_F) against tetrahedron within
        2 %) cannot run; today's gate is Gaussian against Gaussian at equal broadening. Belongs
        beside `LocalOrbitalDensityOfStatesCalculator` as an alternative strategy, and serves
        the direct DOS workflow equally
-   [ ] **[M]** **Independent PAW Hubbard-stress evidence** — the PAW path is exercised
        and shares the validated ultrasoft branch, but the committed modern PAW datasets
        omit labelled atomic wavefunctions and QE refuses their Hubbard cards. QE's local
        source inventory contains a complete labelled PSLibrary copper PAW dataset and
        reports a `-3711.29 kbar` Hubbard term on the matched deck. The older `l_max_aug`
        spelling is now parsed, but Q3 still diverges on this representation and does not
        apply its `core_energy` reference constant. That unsupported subset is now refused
        before SCF. Root-cause and validate the complete convention before promoting the
        case to an independent parity gate
-   [ ] **[M]** **DFT+U backend coverage** — the GPU Hamiltonian applicator and direct
        diagonalizer still refuse DFT+U because neither reaches the Hubbard term. Keep the
        now-tested setup-time refusals until CPU/GPU and Davidson/direct energy plus
        occupation-matrix parity pass
-   [ ] **[M]** **Split NC vs US nonlocal stress** — `nonlocal.hpp` (~800 lines) into separate calculators
-   [ ] **[M]** **Compose NLCC/augmentation** in force/stress orchestrators instead of `if (has_nlcc)` / `if (has_ultrasoft())` guards inside calculators
-   [x] **[M]** **DFT-D3(BJ) forces & stress** — analytical gradients of the dispersion energy w.r.t. atomic positions (direct pairwise term + coordination-number chain rule) and cell vectors (Nielsen-Martin convention); `D3BJDispersionCalculator::compute_gradients()`. Validated by finite-difference unit tests (`test_d3bj.cpp`) and, for forces, an independent QE cross-check via the two-run differencing trick (`test_dispersion_forces_match_qe_via_differencing`, agreement to 1e-8 Ry/bohr)
-   [ ] **[L]** **Semimetal Fermi-level agreement at extreme cell separation** — found while validating graphite D3: Q3 and QE total energies (non-dispersion part) diverge by ~9.5 meV/atom for two fully-decoupled graphene layers at 15 Å separation, though they agree to ~0.5 meV/atom at 8 Å. Ewald/Hartree/XC/smearing-entropy all match QE component-by-component at both separations, so this isn't a bulk SCF bug — it isolates to eigenvalue/occupation resolution for a semimetal with an exactly-degenerate Dirac point at E_F once interlayer splitting no longer lifts the degeneracy, likely a Fermi-bisection or k-point-sampling sensitivity specific to this regime. Not blocking (worked around by using 8 Å as the graphite c-scan's asymptotic reference instead of 15 Å — see `tests/validation/test_qe_graphite_d3_dispersion.py`), but worth root-causing before any future work targets other semimetal/degenerate-Fermi-surface systems (e.g. other layered 2D materials)
-   [x] **[M]** **Makov-Payne isolated-system correction** (both orders) — added
        `control.isolated_system_correction: "none"|"makov-payne"` (default `"none"`,
        opt-in) and `MakovPayneCorrectionCalculator`
        ([layer_5_properties/isolated_system_correction/planewave/makov_payne.hpp](src/layer_5_properties/isolated_system_correction/planewave/makov_payne.hpp);
        selected via the `IsolatedSystemCorrectionCalculator` factory + Null-object
        pattern in the same directory, mirroring `create_dispersion_calculator()`),
        invoked from `SelfConsistentFieldFlavor` alongside forces/stress. Cubic lattices
        only (`bravais_lattice_index` 1/2/3 = SC/FCC/BCC), gated with an explicit throw
        otherwise. - **Formula source correction**: this item originally cited a local
        `reference/` path for the formula, which does not exist in this checkout (no
        `reference/` directory at all — confirmed by direct search). Rather
        than reconstruct the formula from memory/the 1995 paper alone (Makov & Payne,
        PRB 51, 4014 — whose Eq. 15 has the wrong sign for the quadrupole term, as
        Quantum ESPRESSO itself documents), the correction and the dipole/quadrupole
        moments it needs were matched term-by-term to what current Quantum ESPRESSO
        actually computes. Confirmed the Madelung constants (SC=2.8373,
        FCC=2.8883, BCC=2.8885) match this item's original note. **Correction to the
        original plan**: it assumed `ibrav=-3` (BCC's alternate-axes convention) shares
        BCC's Madelung constant; QE's actual bound check is on the signed index (1..3,
        with no absolute value taken) and rejects `-3` outright, same as any non-cubic lattice —
        `MakovPayneCorrectionCalculator` matches that exactly (regression test:
        `RejectsBccAlternateAxesConvention`). - **Exact algorithm** (both terms already sign-flipped to QE's printed convention,
        so `corrected_total = total_energy + E_1st + E_2nd`):
        `E_1st = alpha_M(ibrav) * qq^2 / alat`;
        `E_2nd = -(2*pi/3) * (qq*aa - bb) / alat^3 * e2` (e2=2.0 Ry), where
        `qq = zvtot - N_electrons` (net charge, +1 for one electron removed),
        `aa = Tr[Q_ion - Q_el]` (trace of the ionic-minus-electronic quadrupole tensor
        about `x0`), `bb = |dipole_ion - dipole_el|^2`,
        `x0 = sum(Z_i*tau_i)/sum(Z_i)` (valence-charge-weighted ionic center). Electronic
        dipole/quadrupole moments are a real-space sum over the dense (charge-density)
        grid with minimum-image wrapping of `(r - x0)`; ionic moments are **not**
        wrapped, matching QE exactly (reproducing QE's number requires reproducing
        whatever atomic-position gauge the reference calculation used — this is the
        likely source of the small residual noted below, not a formula bug). NLCC core
        charge is excluded (`charge_density_real` is the pure valence density in this
        codebase already — core charge is a separate array added only where XC needs
        it, matching how QE itself keeps the two apart). - **Validation against the QE reference** (FCC Si, `tot_charge=1.0`,
        `celldm(1)=10.3344`, via `examples/total_energy--semiconductor--silicon/nc_charged.yml` - `control.isolated_system_correction: makov-payne`): 1st order =
        **0.27948401 Ry**, matching QE's own reference value to all 8 significant
        figures printed (this term depends only on the net charge and lattice geometry,
        with no cross-code SCF-density or ionic-gauge ambiguity, so an exact match is
        the expected outcome of a correct implementation). 2nd order = **-0.91521864
        Ry** vs. QE's reference of -0.91595649 Ry — agreement to ~0.08% (0.0007 Ry),
        attributed to the un-wrapped-ionic-moment gauge-dependence noted above combined
        with ordinary cross-code differences in the converged self-consistent density
        (not a formula bug, given the exact 1st-order match). `tests/validation/test_makov_payne.py`
        asserts both to tolerances consistent with what's actually achievable
        (1e-5 Ry for the 1st order, 1e-3 Ry for the 2nd). - **Testing**: `tests/unit/layer_5_properties/isolated_system_correction/planewave/test_init.cpp`
        (3 tests) covers the cubic-lattice gate (free lattice vectors, `ibrav=-3`,
        hexagonal, all rejected); `test_yaml.cpp` covers the new key (default, valid
        value, rejected invalid value); `tests/validation/test_makov_payne.py` covers
        the end-to-end QE-reference comparison above and confirms the JSON keys are
        absent when the correction isn't opted into. Full CTest 775/775;
        `tests/python/` + `tests/regression/` 79/79 (Crystal's new
        `bravais_lattice_index` field touches a widely-used class); broader
        `tests/validation/` sweep unaffected. - **Architecture note**: `Crystal` did not retain `bravais_lattice_index` after
        construction (a transient constructor parameter, discarded after
        `set_from_lattice_type`) — added a member + accessor, set in
        `Crystal::from_input_data()` (the "free"/explicit-lattice-vectors path,
        `Crystal::from_lattice_vectors()`, leaves it at its default `0`, matching QE's
        own ibrav=0 convention for that case). - **qq** (net charge) can be taken directly from `config().total_charge`, or — matching QE's own more robust approach — from the actually-integrated electronic charge (`dipole(0)` below) minus the ionic charge; the latter is a free byproduct of the same density-integration pass and self-consistently absorbs any residual SCF density error. - **The hard part — electronic dipole/quadrupole moments**: real-space integration of the converged density over the FFT grid, using **minimum-image wrapping** relative to the ion-charge-weighted center `x0 = Σ(Z_i·τ_i)/Σ(Z_i)`: for each grid point, convert its Cartesian position minus `x0` to fractional/crystal coordinates via `(lattice_vectors_bohr.transpose().inverse())`, wrap each component with `frac -= round(frac)`, convert back to Cartesian, accumulate `Σρ·r`, `Σρ·r²` per component, then multiply by the grid-point volume (`cell_volume / total_grid_points`). This is the ONLY genuinely new numerical subsystem needed; ionic dipole/quadrupole (simple sum over atoms, no grid) is comparatively trivial. - **Before implementing**: verify the exact (i,j,k)→flat-index convention of `charge_density_real_` / `FFTGrid` against an existing real-space consumer (e.g. the XC or local-potential calculator) — this was not yet confirmed and is the most likely source of an indexing bug. - **Hook point**: `report_convergence_details()` in `layer_4_computation/iterator/kohn_sham/planewave/init.hpp` (called once at convergence; already has `charge_density_real_`, `crystal_structure()`, and `config()` in scope — no need to expose the density outside the iterator). Do NOT fold the correction into `energies_.total()` — like QE, report it as a separate, clearly-labeled "corrected" energy alongside the uncorrected total; it has no well-defined per-atom forces/stress and must not silently affect relaxation or convergence. - Validate against QE's own separately-printed 1st-order, 2nd-order, and total lines (`assume_isolated = 'makov-payne'`) on the same charged Si cell already validated here.
-   [x] **[M]** **Martyna-Tuckerman isolated-system correction** — the second scheme through
        the `IsolatedSystemCorrectionCalculator` factory (`create_isolated_system_correction_calculator`,
        `control.isolated_system_correction: "martyna-tuckerman"`), added in
        `MartynaTuckermanCorrectionCalculator`
        ([layer_5_properties/isolated_system_correction/planewave/martyna_tuckerman.hpp](src/layer_5_properties/isolated_system_correction/planewave/martyna_tuckerman.hpp)).
        Unlike Makov-Payne, it has **no lattice-type gate** — valid for any Bravais lattice,
        with a cheap, warning-only (not throwing) locality heuristic instead (see below). - **Scope (read this before comparing to QE): post-SCF only, NOT QE's self-consistent
        `assume_isolated='m-t'`.** QE's own Martyna-Tuckerman (G. J. Martyna, M. E. Tuckerman,
        J. Chem. Phys. 110, 2810 (1999)) folds a corrected local potential
        into the SCF Hamiltonian every iteration, so its converged density differs from the
        ordinary-periodic-kernel density. Q3's `IsolatedSystemCorrectionCalculator` interface
        is architecturally post-SCF (every scheme, Makov-Payne included, is evaluated once
        after convergence) — `MartynaTuckermanCorrectionCalculator` reports what the MT
        kernel difference contributes to the ALREADY-CONVERGED (periodic-kernel) density, not
        what QE would converge to under its own corrected potential. The self-consistent
        version (feeding the correction back into `V_eff` every SCF iteration, a materially
        larger piece of work touching the effective-potential calculator and SCF iterator, not
        just a post-convergence report) is explicitly **not implemented** and is the natural
        next step — see "Future work" below. - **Formula source**: matched to what current Quantum ESPRESSO
        actually computes for its Martyna-Tuckerman correction, rather than
        reconstructed from memory or the paper alone (the same care already
        recorded for Makov-Payne above). - **Exact algorithm**: `E_MT = (e^2/2) * Omega * sum_G |rho_ion(G) - rho_el(G)|^2 * W(G)`,
        summed over ALL G (including G=0, unlike the ordinary Hartree/Ewald sums) on the
        charge-density G-vector grid. `rho_ion(G)` is the usual ionic structure factor over
        cell volume (same convention as `compute_ewald()`); `rho_el(G)` is the forward FFT of
        the converged, spin-summed electron density (`PlanewaveBasis::rho_r_to_g()`). `W(G)`
        (QE's Martyna-Tuckerman kernel) is built by (1) picking a convergence parameter
        `alpha` (bohr^-2) via the same kind of erfc-tail-bound search as Ewald, but
        numerically matching QE's own alpha-selection formula exactly (`sqrt(2*alpha/tpi)`,
        not Q3's own differently-scaled `compute_ewald()` search); (2) evaluating
        `erf(sqrt(alpha)*r)/r` on the dense
        real-space grid at each point's **minimum-image distance from the origin**, forward-FFT
        to G-space; (3) subtracting the free-space analytic FT of that same function
        (the smooth-Coulomb transform, including QE's specially-derived finite G→0 limit,
        reproduced exactly) and applying a residual Gaussian filter. See the class docstring
        for the full derivation and QE cross-reference. - **Minimum-image distance is a parallelepiped proxy, not the exact Wigner-Seitz
        polyhedron**: QE constructs the true
        Wigner-Seitz cell for a general (possibly oblique) lattice. Reproducing that exactly
        was judged out of scope for this session (a nontrivial geometric algorithm in its own
        right); `MartynaTuckermanCorrectionCalculator` instead reuses
        `MakovPayneCorrectionCalculator`'s own already-established fractional-coordinate
        minimum-image wrap (`frac -= round(frac)`) as a documented approximation — exact for
        orthogonal lattices (simple cubic), and exact near the cell center for any lattice
        (both constructions share the same inscribed sphere), but only approximate near the
        cell boundary of an oblique primitive cell (FCC/BCC). Tracked as a documented
        limitation, not silently assumed away. - **Validity check (cheap, warning-only, not a gate)**: because the class has no
        lattice-type restriction, it instead checks — folded into the same real-space grid
        pass that builds `W(G)`, at no extra asymptotic cost — what fraction of `|charge
density|` lies within a sphere of radius = half the shortest lattice vector, centered
        on the valence-charge-weighted ionic center (the region where the parallelepiped proxy
        above is provably exact). Below 99%, it logs a warning (not a throw) rather than
        blocking the run, since the check is itself a heuristic, not an exact geometric
        membership test. - **Physical completeness vs. QE's own public API**: QE exposes the MT
        correction as two SEPARATE numbers (electron-electron and ion-ion); the
        electron-ion cross term is captured only through the corrected local potential's
        feedback into the self-consistent potential — machinery Q3's post-SCF scheme doesn't
        have. To still capture that cross term (needed for the monopole-limit validation
        below to actually work), `MartynaTuckermanCorrectionCalculator` sums
        `|rho_ion(G) - rho_el(G)|^2 * W(G)` directly over the TOTAL charge, the mathematically
        complete generalization of "MT kernel minus periodic kernel, applied to the whole
        system's electrostatic energy" — not merely the sum of QE's own two published
        numbers. See the class docstring's section on how this relates to that two-number split. - **Validation — two first-principles limits, verified independently in a throwaway
        numpy prototype (not checked in) before being wired into the C++ unit tests**, on a
        synthetic single-atom simple-cubic system with a hand-built isotropic Gaussian
        "electron density": 1. **Neutral, concentrically-symmetric limit**: a point ion and an isotropic Gaussian
        electron cloud of the same integrated charge, centered at the same point, give a
        correction consistent with zero (~1e-9 Ry in the numpy prototype at properly
        resolved grids; the C++ unit test asserts < 1e-5 Ry) — matching the classical
        concentric-spherical-shell (Gauss's law) expectation that such a configuration
        produces (almost) no field outside its own compact support, hence (almost) no
        periodic-image interaction of any multipole order. 2. **Monopole (large-cell) limit vs. Makov-Payne**: for a net-charged compact
        (isotropic Gaussian) density, `MartynaTuckermanCorrectionCalculator`'s total energy
        and `MakovPayneCorrectionCalculator`'s first-order (Madelung monopole) term,
        evaluated on the exact same synthetic SCF state, both describe the same leading-order
        physics (a point-like net charge's periodic self-interaction) and their ratio
        trends monotonically toward 1 as the cell grows: numpy prototype ratios 1.052 →
        1.023 → 1.010 → 1.006 across four increasing cell sizes (fixed Gaussian width,
        fixed plane-wave cutoff); the C++ unit test reproduces the same trend at three
        sizes (`ApproachesMakovPayneFirstOrderTermAsCellGrows`) and passed on the first run
        with no tolerance tuning needed, corroborating both the prototype and the ported
        implementation independently. 3. **End-to-end regression anchor** (same charged FCC Si cell as Makov-Payne's own
        validation, `examples/total_energy--semiconductor--silicon/nc_charged.yml` +
        `control.isolated_system_correction: martyna-tuckerman`): energy correction =
        **0.02299160 Ry** (uncorrected total energy −17.29961509 Ry). This is a
        **self-measured anchor, not a QE cross-check** — QE has no directly comparable
        number to check against, since its own `assume_isolated='m-t'` is self-consistent
        (see the scope note above); comparing Q3's post-SCF number to QE's self-consistent
        one would conflate the architectural scope difference with any actual formula
        error. Mirrors the precedent set by the DFT+U ortho-atomic regression anchors
        (`NiFCC_HubbardOrthoAtomicConvergesToDifferentEnergyThanAtomic`,
        `tests/integration/test_scf_integration.cpp`) for a case this project could not
        cross-check against an established code either. **Also note**: FCC Si's valence
        density is delocalized (covalent bonding), not compact — exactly the kind of
        system the class's own locality check is designed to flag, and it does (only
        ~54% of `|charge density|` within the locality radius for this system, logged as
        a warning during this validation run). This end-to-end test therefore validates
        the WIRING (factory, YAML parsing, results.json output), not the physical
        correctness of this specific number for isolated Si⁺ — physical correctness is
        established separately, by the two synthetic limits above. - **Testing**: `tests/unit/layer_5_properties/isolated_system_correction/planewave/test_martyna_tuckerman.cpp`
        (3 tests: the two validation limits above, plus a factory-name check); `test_factory.cpp`
        gained `MartynaTuckermanResolvesToMartynaTuckermanCalculator` (its pre-existing
        `UnknownSchemeThrows` test used `"martyna-tuckerman"` as an example unrecognized string —
        now switched to `"point-countercharge"`, a real not-yet-implemented QE-family scheme
        name, so the test can't silently start passing for the wrong reason once THAT scheme is
        added too); `test_yaml.cpp` gained `IsolatedSystemCorrectionParsesMartynaTuckerman` (same
        `"point-countercharge"` swap in its own rejects-invalid-value test, for the same reason);
        `tests/validation/test_martyna_tuckerman.py` covers the end-to-end regression anchor
        above and confirms the JSON keys are absent when the correction isn't opted into. Full
        CTest and `tests/python`/`tests/regression`/`tests/validation` green (see this PR's own
        gate for exact counts). - **Schema check**: `schemas/q3/input.json` does **not** enumerate
        `isolated_system_correction` at all (confirmed by direct search) — it wasn't added when
        Makov-Payne shipped either, so there is no existing enum to extend; left untouched
        rather than introducing a field this schema doesn't otherwise track. `docs/io_reference.md`
        and `FUNCTIONALITY.md` updated instead (this project's actual source of truth for the
        YAML `control.*` allow-list, per `yaml.hpp`'s own `reject_unknown_keys()` — the schema
        under `schemas/q3/` is a separate, currently-partial mirror, primarily feeding
        `src/generated/q3/input.hpp`, which has no consumer yet either).
        **Superseded:** `schemas/q3/input.json` now mirrors the YAML manifest itself rather
        than the flattened `InputData`, and its per-level key sets are generated into
        `src/generated/q3/input.hpp` and consumed by `reject_unknown_keys()` — so the schema
        _is_ the allow-list, `isolated_system_correction` included, and a new `control.*` key
        goes there first. See `schemas/README.md` § Key registries. - **Future work — self-consistent Martyna-Tuckerman**: feed the MT-corrected local
        potential back into the effective-potential calculator every SCF
        iteration (a new term alongside Hartree/XC/local-pseudopotential, analogous in shape to
        the still-open slab-dipole item below, not a post-convergence report), so the converged
        density itself relaxes under the corrected potential exactly like QE's
        `assume_isolated='m-t'`. Only then would a direct Q3-vs-QE total-energy comparison on
        `assume_isolated='m-t'` be a like-for-like check rather than conflating architecture with
        formula. Needs: (1) the local-potential correction term added to
        `PlanewaveEffectivePotentialCalculator`; (2) re-deriving the corresponding force term if
        forces/relaxation should also see the correction; (3) a genuine QE
        `assume_isolated='m-t'` cross-check once (1) lands, replacing this item's self-measured
        anchor above.
-   [ ] **[M]** **Slab dipole (sawtooth) correction** — QE `tefield=.true., dipfield=.true.` analog for slab total-energy/force correction; deliberately deferred alongside Makov-Payne (Window 2 roadmap item). Needs a per-iteration sawtooth counter-potential added to the local potential (not a post-SCF correction like Makov-Payne), so it is architecturally a different, larger piece of work — a new term in the effective-potential calculator, not a post-convergence report.
-   [x] **[L]** **Forces had spurious components on any structure with reduced (non-maximal) point-group symmetry** — found 2026-07-25 while building the Γ-point phonon workflow (`feature/2026-07-25-4-gamma-phonons`): finite-difference force constants on Si (2-atom FCC/diamond) came out with the Γ-optical mode split into three DIFFERENT frequencies (350.5, 680.1, 741.8 cm⁻¹) instead of the physically-required exact triple degeneracy (diamond structure has Oh symmetry at Γ; this is why bulk Si shows one sharp Raman line, not three).
    -   **Root cause, fixed 2026-07-26 in `feature/2026-07-26-0-fix-reduced-symmetry-forces`**: `SymmetryAnalyzer::symmetrize_vector()` (`layer_1_structure/symmetry/symmetry.hpp`) applied the crystal's real-space rotation matrix (`SymmetryOperation::rotation`, documented as `x' = rotation * x` for POSITIONS) directly to forces. Forces are a covariant (gradient-like) quantity, not a position; QE's own convention (its vector symmetrization computes `vect_i = Σ_j s(i,j,isym)*work_j`, i.e. applies `s` directly — and `s = rotation^T` since QE's rotated ATOM POSITIONS are `rau = s^T·xau`, i.e. `rotation = s^T`) shows forces need `rotation^T`, not `rotation`. A previous version of the code's own comment had swapped `s(i,j)`/`s(j,i)` while re-deriving this from the array-index notation, landing on the wrong (untransposed) convention. Invisible for the full 48-operation cubic point group (every previously-validated system in this codebase), because `symmetrize_matrix()` (used for stress) already had the correct `R^T` convention independently, and the one existing `symmetrize_vector` unit test used an isotropic `(1,1,1)` input under the full point group, which averages to zero regardless of transpose direction — not for the 8-operation subgroup ("Imma", #74) a single-atom Cartesian-axis displacement reduces to. Fix is a single transposed-index change; see `tests/unit/layer_1_structure/symmetry/test_symmetry.cpp`'s `VectorSymmetrizationUnderReducedSymmetry*` tests for a minimal, hand-verified (against an independent QE comparison) regression test that fails without the fix and passes with it.
    -   **Exact repro** (kept for reference / re-deriving the follow-up below): Si FCC (`examples/data/materials/silicon.json`, NC `Si.upf`, PBE, ecutwfc=48/ecutrho=192 Ry), atom 0 (equilibrium fractional `[0.875,0.875,0.875]`) displaced by Cartesian Δx = -0.01 Å only (reduces `Fd-3m #227` to `Imma #74`, 8 ops). QE's force for this geometry is purely along x and essentially k-grid-independent (`[0.0108,0,0]` Ry/bohr Γ-only, `[0.0051,0,0]` at 6×6×6-shifted). Post-fix, Q3 at 6×6×6-shifted gives `[0.00526,0,0]` Ry/bohr — matches QE to ~3%, exactly zero y/z.
-   [x] **[L]** **Q3's IBZ k-point reduction did not correctly reconstruct vector quantities (forces, stress) from the irreducible-BZ representative** — found 2026-07-26 immediately after fixing the `symmetrize_vector` transpose bug above, while validating that fix's effect on the full Γ-phonon calculation: the symmetry-forbidden force components were correctly zero (the fix above works), but the _surviving_ (symmetry-allowed) force component could still be badly wrong -- including sign-flipped -- when Q3's automatic Monkhorst-Pack + IBZ reduction was used, on the same class of reduced-symmetry displaced structures.
    -   **Exact repro**: same Si FCC setup as above, atom 0 displaced by Cartesian Δz = -0.01 Å instead of Δx (reduces to a differently-oriented `Imma`-type 8-op subgroup). At 6×6×6-shifted `[1,1,1]`, QE gives `+0.00509` Ry/bohr (z-component, matching the same-order-of-magnitude, same-sign pattern as the Δx case). Q3's automatic IBZ-reduced calculation gave **`-0.00638`** — sign-flipped, ~25% larger in magnitude. Feeding Q3 the identical structure through an **explicit, full, unreduced 216-point k-point list** (`control.kpoints_list`, bypassing automatic MP-grid + IBZ reduction entirely) gave **`+0.00481`** — correct sign, ~5.5% off QE (an unrelated, expected discretization-level gap, not this bug). This isolated the bug to the automatic IBZ-reduction path, not the underlying nonlocal force formula.
    -   **Root cause, fixed 2026-07-27 in `feature/2026-07-27-0-ibz-vector-reconstruction`**: the leading hypothesis above (that the k-point-summed force/stress calculators needed to explicitly rotate each star member's contribution, i.e. `Σ_{k in star} f(k) = Σ_isym R_isym · f(k_IBZ)` rather than a plain `weight * f(k_IBZ)` scaling) turned out to be a plausible-sounding but WRONG diagnosis — empirically ruled out below. The actual bug was one level upstream, in k-point **equivalence-class construction** itself: `SymmetryAnalyzer::reduce_kpoints_internal()` (the default `ibz_reduction_method`) built its k-point equivalence classes from `compute_bravais_lattice_rotations()` — the Bravais **lattice's** holohedral point group (lattice vectors only, e.g. all 48 operations of cubic FCC) — instead of the crystal's actual, atomic-basis-aware point group (8 operations for this displaced structure). Confirmed by direct code reading: `compute_bravais_lattice_rotations()` takes only a `Crystal` and reads `crystal.lattice_vectors()`, never touching atomic positions/types, unlike `get_symmetry_operations()` (used correctly elsewhere, e.g. density symmetrization), which calls spglib's `spg_get_dataset()` with positions and types and so correctly detects the smaller, basis-respecting group. `reduce_kpoints_internal()`'s own docstring even asserted it "uses Q3's own symmetry operations (from spglib's `spg_get_dataset`)" — describing the intended, correct behavior its actual implementation didn't deliver, strong evidence this was an unintentional divergence, not a deliberate design choice. Treating k-points as equivalent under rotations that are not real symmetries of the decorated crystal corrupts the k-point sum for **any** k-resolved quantity gathered per-representative-and-scaled, not only vectors — it just happened to be invisible for every previously-validated system (all at full, lattice-matching symmetry, per the `symmetrize_vector` bug's own writeup above) and for scalar totals close enough in practice not to have been separately flagged. A second, compounding issue in the same function: even given the right group, the k'=R·k reconstruction step applied `.rotation` (the real-space/position convention) to a **reciprocal-space** k-vector; the correct convention is `.reciprocal_rotation` = `(R^{-1})^T` (already used for exactly this purpose by the density symmetrizer's G-vector shells) — a crystallographic rotation matrix in a generally oblique fractional basis (e.g. FCC's rhombohedral primitive cell) is not its own inverse-transpose.
    -   **Fix**: `reduce_kpoints_internal()` now calls `get_symmetry_operations(crystal, symmetry_precision)` (deleting the now-redundant, ~160-line `compute_bravais_lattice_rotations()` entirely) and uses each operation's `reciprocal_rotation` for the k-point mapping step. No change was needed in any force/stress calculator, or in the naive `weight * f(k_IBZ)` reconstruction pattern itself — once fed the correct equivalence classes, that pattern is already correct (see verification below), which is why the original "vector reconstruction formula" hypothesis was a red herring: the star was actually always correctly reconstructible from a single representative for this problem, provided that representative's own residual little-group-of-the-perturbation happens to leave the force vector itself invariant component-by-component -- a general audit of every k-point-summed calculator against a genuinely un-recoverable-from-one-representative case (e.g. a low-symmetry crystal with multiple atoms permuted, not fixed, by the star's generating rotations) was not attempted and may still be worth a future, separate look if a case surfaces where this reasoning doesn't apply.
    -   **Verification**: on the exact Δz repro above, the fixed default path now gives `+0.00480456` Ry/bohr — matching, to displayed precision, both an independent spglib-based reduction (`ibz_reduction: spglib`, `SymmetryAnalyzer::reduce_kpoints()`, which already correctly used positions/types via `spg_get_ir_reciprocal_mesh`) and the fully unreduced explicit k-point list (`+0.00481` above), and reproducing the same order of the QE-cross-checked value. **Stress** (previously only spot-checked, not confirmed) showed the identical pattern: pre-fix `[[-26.03,-18.59,0],[-18.59,-26.03,0],[0,0,52.27]]` kbar (qualitatively wrong) vs. post-fix `[[0.14,-25.21,0],[-25.21,0.14,0],[0,0,0.08]]` kbar, again matching the spglib and unreduced-list references exactly — so stress needed no separate audit; it shares the same upstream k-point grid. New regression coverage:
        -   `tests/unit/layer_1_structure/symmetry/test_symmetry.cpp`'s `ReduceKpointsInternalUsesActualCrystalSymmetryNotBravaisLattice` (fast, no SCF: cross-checks `reduce_kpoints_internal` against `reduce_kpoints` on the existing Δx-displaced-Si helper, asserting equal irreducible-k-point counts and conserved total weight).
        -   `tests/validation/test_gamma_phonons.py`'s `test_silicon_optical_modes_triply_degenerate` now passes outright (`xfail` marker removed, was previously an unexpected-pass/`XPASS` signal once this landed).
        -   `test_silicon_gamma_phonons_match_qe` (the direct, wired-in QE comparison; `xfail` also removed) still needs a run with an established reference DFT code available to confirm formally, matching this repo's usual QE-gated-test caveat — not run as part of this fix.
    -   Diagnostic scripts/inputs used to isolate this are NOT checked in (per `agents/workdir/tmp/`, gitignored); reproduce from the exact repro description above rather than searching for them.
-   [ ] **[M]** **Shifted Monkhorst-Pack meshes on reduced-symmetry cells are sampled at the bare mesh points, where QE samples their star** — found 2026-10-08 running `tests/validation/test_phonopy_phonons.py`'s QE gate for the first time. On silicon's 2x2x2 supercell with one Phonopy-displaced atom (space group Cm, 2 operations), the example's 1x1x1 shifted mesh is **1** k-point in Q3 and **3** in QE (weights 0.5, 0.5, 1): QE's `irreducible_BZ` expands each mesh point to its star under the lattice's 48 rotations, then reduces by the crystal's 2. Q3's forces carry a residual of ~0.074 eV/Å on every atom that QE's do not, and the response itself comes out low: Γ-optical 14.38 THz against QE's 15.22 through the same Phonopy, a gap the workflow's docs had put down to k-point convergence. Given QE's 3 k-points as an explicit `control.kpoints_list`, Q3's forces match QE's to 6e-4 eV/Å, so the forces are right and the mesh generation is the difference. The fix is QE's: generate the mesh, then complete each point's star under the Bravais lattice's rotations before reducing by the crystal's own group (the 2026-07-27 fix above rightly stopped _reducing_ by the lattice group; this is _expanding_ by it). Likely also behind the 6x6x6 ~5.5% gap noted above, and worth re-measuring on the elastic workflow's strained cells. The gate is `xfail(strict=True)`, so it fails loudly once this is fixed

---

## 2. Performance (HPC)

### 2.1 CPU

-   [ ] **[S]** **Converge a non-self-consistent pass's empty bands to its tolerance** — a
        single pass lets an empty band stop once its eigenvalue moves by less than 1e-5 Ry
        an iteration, the self-consistent rule, though no iteration follows it. On copper's
        4×4×4 Wannier mesh with sixteen bands and the NSCF's basis of four times the band
        count, that leaves band 16 1.2 meV from direct diagonalization at one k-point,
        against at most 0.056 meV for bands 1–15. Holding every band to the pass's tolerance
        brings all sixteen within 0.005 meV for 21% more Hamiltonian applications; gate on
        the same comparison, which retires the KNOWN-ISSUES.md workaround of requesting
        extra bands. The 68.8 meV this item used to quote, and the conclusion that a
        tighter test was not the fix, were measured under the looser re-diagonalization
        resolved on 2026-10-08

-   [x] **[S]** **Accelerate Broyden G-space operations** — the smooth-only G-space
        vector math in [broyden.hpp](src/layer_4_computation/iterator/internal/mixing/reciprocal_space/broyden.hpp)
        (residual, delta vectors, gamma corrections, metric dot products) was hand-rolled
        scalar loops with no BLAS or SIMD. Targeted the two hot spots that actually scale
        with `history_depth` (default 8): - **Metric dot products** (Gram matrix + projection): `compute_metric_dot_product()`
        re-applied the per-G Hartree/flat metric weight on _every_ pairwise access, i.e.
        O(history*depth²) weight applications for the Gram matrix alone. Added
        `metric_scale_factor()`/`build_metric_scaled_copy()`, which bake the (fixed, only
        metric-dependent) weight into a metric-scaled twin of each history vector **once**,
        when it enters history (O(history_depth) total). The Gram matrix and projection
        loops then call `dot_product_of_scaled_and_raw()`, which reduces to a plain,
        branch-free complex conjugate dot product per spin channel via the existing
        `q3_complex_conjugate_dot_product` helper ([blas.hpp](src/layer_0_core/math/linear_algebra/blas.hpp),
        the same BLAS-Level-1-style primitive `layer_0_core/execution/serial.hpp` already
        uses) — no per-element weight lookup or `g_index < first_nonzero_g_index*`branch
        inside the O(history_depth²) loop.
    -   **Gamma corrections + final linear mixing**:`rho*in -= Σγᵢ·Δρᵢ`,
        `F_prec -= Σγᵢ·ΔFᵢ`, and the final `ρ_new = ρ_in + α·F_prec` step are textbook
        AXPY (`y += alpha\*x`) patterns; replaced the hand-rolled per-element loops with
        direct calls to BLAS `zaxpy*`(already declared in blas.hpp, previously unused
        outside`layer*0_core/execution/serial.hpp`).
    -   **Deliberately left as plain fused loops**: raw residual (`F = ρ_out - ρ_in`) and
        delta-vector computation (`ΔF`, `Δρ`) are elementwise _assignments_, not
        accumulations — a single-pass fused subtract-and-store loop already reads each
        input array once and writes the output once. There is no BLAS "vector subtract"
        primitive; routing this through `zaxpy*` would require an extra copy first,
        _increasing_ memory traffic rather than reducing it. Left untouched, with a
        reasoned no-op documented rather than silently skipped.
    -   **Bit-reproducibility caveat**: the refactor is mathematically equivalent to the
        original per-pair weighted formula but is **not** guaranteed bit-identical to it
        (folding a per-G weight into one dot-product operand before multiplying, versus
        multiplying the unweighted product by the same weight afterward, does not
        distribute exactly in IEEE754 for arbitrary — i.e. non-power-of-2 — weights).
        Verified this doesn't matter in practice: a hand-derived 2-iteration, non-uniform-metric
        regression test (`BroydenMixer.NonUniformMetricWeightsProduceHandVerifiedResult`,
        weights 2.0/5.0, exercising the full Gram-matrix/DSYTRF/DSYTRI/projection/correction
        path with `actual_history == 1`) matches the hand-computed expected density to
        1e-9; a real end-to-end SCF run (`examples/total_energy--semiconductor--silicon/nc.yml`,
        Broyden is the default mixer — `data.hpp`'s `mixing_mode` default) gave
        **identical SCF iteration count (7) and Davidson iteration count (45)** before/after,
        with the converged total energy differing only at the last printed digit
        (`-16.91341793199311`Ry vs`-16.913417931993106`Ry, a ~4e-15 absolute
        difference — machine-epsilon-level noise, not a physical discrepancy).
    -   Added`BroydenMixer.TwoSpinChannelsConvergeWithMagnetizationMetric`for coverage:
        every pre-existing`test_broyden.cpp` test was single-spin, so the
        magnetization-channel (`spin_index >= 1`) branch of `metric_scale_factor()`had
        never been exercised by any test before this branch.
    -   Regression tests: 2 new tests in`test_broyden.cpp`(11/11 passing); full CTest
        760/760;`tests/validation/` (excluding MPI/GPU-marked tests) unchanged pass/skip
        counts.
-   [ ] **[M]** **Wire up CPU batched FFTs across bands** — the batched API exists
        (`FFTGrid::forward_batched` / `inverse_raw_batched`, and `fftw_mpi_plan_many_dft`
        in the MPI grid) but is currently **dead code**: no compute path calls it and
        `SerialFFTGrid` has no `fftw_plan_dft_many` override. The serial/CPU
        `apply_hamiltonian` still transforms one band (or one Γ pair) at a time.
        Route the Davidson batch path through the batched API (the GPU path already
        batches via cuFFT `plan_many`).

### 2.2 GPU

-   [ ] **[S]** **Compile the GPU targets in CI** — no CI job builds `q3_gpu` or
        `q3_mpi_gpu_cuda`, so any change to the GPU sources breaks the build silently and
        is only found by whoever next builds with `-DQ3_WITH_GPU=ON`. A compile-only job
        would catch it; it needs a macOS runner for the Metal backend (and CUDA toolkit
        for the other), which is a runner-cost decision rather than a technical one. Until
        then, changes touching `src/**/gpu*` should be built and run locally with
        `-DQ3_WITH_GPU=ON` and `tests/validation/test_gpu_energy_parity.py`, and say so in
        the pull request. Compiling alone would not have been enough on 2026-09-14: a
        wrong GPU energy reached main and was found only by running that suite by hand
        (KNOWN-ISSUES.md, Resolved).

GPU acceleration (CUDA, Metal, HIP, SYCL) has shipped for the Davidson `H|ψ⟩` and
subspace path. Remaining work:

-   [ ] **[L]** **Multi-GPU beyond k-point pools** _(Phase 5e)_ — k-point-parallel multi-GPU
        shipped as `q3_mpi_gpu_cuda` (one rank, one device, one pool). What is left is the
        band axis: `GPUPlanewaveHamiltonianApplicator` hard-fails under band-distributed MPI,
        and spreading one k-point across devices needs NCCL collectives. Device assignment is
        still the launcher's job (`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`), since `cuda.cpp`
        calls `cudaSetDevice(0)`. The single-process GPU binaries refuse a multi-task launch
        rather than duplicating the calculation (`src/io/launcher.hpp`).
-   [ ] **[L]** **Native GPU FFT for SYCL and Metal** — `SYCLGPUFFTGrid` delegates to
        CPU FFTW (add oneMKL DFT descriptors); Metal FP64 FFT / eigensolve also fall back
        to CPU (Apple GPUs lack FP64).
-   [ ] **[L]** **HIP/SYCL hardware validation** — HIP is compile-only in CI (no AMD
        hardware); SYCL runs only on the CPU-fallback device. Neither has a validated
        energy or benchmark on real accelerators.

### 2.3 MPI

-   [ ] **[S]** **Stage the subspace collapse through the wavefunction storage, for exact
        parity** — the solver holds one band-sized rotation scratch that Quantum ESPRESSO
        does not, because that code stages its collapse through its wavefunction array,
        which is contiguous per k-point. Q3's is a vector per band, so a single GEMM cannot
        target it. It does not have to: rotate in blocks of a few dozen columns into a small
        buffer and copy each block out band by band into
        `quantum_system.wavefunction_band_data(...)`, which is free scratch between the seed
        and the extraction — nothing reads it in between. That removes the last
        `nbnd x npw` array, taking the 432-atom deck from 40.6 GB to 37.8 GB per rank
        against Quantum ESPRESSO's 36.4, within 4%. The cost is two extra copies of `nbnd x npw` per
        collapse and a blocked rather than single GEMM.
-   [ ] **[M]** **Cache nonlocal projectors for the k-point being solved, not for all of
        them** — `cached_kleinman_bylander_projectors_` holds one `nkb x npw` matrix per
        k-point for the whole run, so a multi-k-point calculation carries `nkpt` copies
        where Quantum ESPRESSO carries one and rebuilds it per k-point in the loop. It costs
        nothing on a gamma-only supercell, which has a single k-point, and it is the second
        largest term in the footprint there regardless (20.8 GB on the 432-atom deck). On a
        100-atom cell with ten k-points it would be the difference between 1.2 GB and
        12 GB. The trade is recomputation time against memory, so it wants a measurement
        and probably a knob rather than an unconditional change.
-   [x] **[M]** **Gamma-only under a shared pool** — done 2026-09-12. The stick FFT
        co-locates every G-vector with its negative, so the half sphere is rebuilt locally
        again, and the `G = 0` corrections -- the `2 Re(...) - G0` terms in the subspace
        manager, the applicator's DGEMM correction, the `psi[0]` packing in the initializer
        and accumulator, the ultrasoft overlap operator -- now ask
        `PlanewaveBasis::gamma_origin_is_local()`, true on the one rank whose plane-wave
        list begins with the origin. The refusal in
        `planewave/init.hpp` is gone; the iterator asks the grid whether it co-locates
        mirrors after building it and falls back to the full sphere only if it does not.
        Gated by `test_gamma_only_survives_a_shared_pool` and two MPI unit tests on the
        subspace manager.
-   [x] **[M]** **Stress builds every derivative projector at once** — done 2026-09-14. The
        nonlocal stress works through the cell in blocks of consecutive atoms, which the D
        matrix allows because it couples an atom's own projectors only. A block holds its
        projectors and, per stress component, one derivative with the radial and angular
        parts already summed (they enter each component with the same weights), side by side
        in one array contracted with the bands in a single product. That last part matters:
        per-atom calls with separate kinds copied the wavefunctions into the BLAS buffers on
        every call and ran 47% slower than the whole cell. On the 54-atom silicon cell at one
        rank the run's sampled peak fell from 4.12 to 1.84 GiB, below the SCF's own, and the
        nonlocal stress from 3.0 to 1.4 s; the stress tensor agrees with the whole-cell
        version to 5e-11 kbar on displaced norm-conserving, ultrasoft and PAW silicon and on
        spin-polarized iron. Shared-pool runs still skip the stress.
-   [ ] **[S]** **Stop transporting empty sticks** — every stick is owned today because two
        consumers read the whole grid box rather than the cutoff sphere: the dense-to-smooth
        potential copy in `planewave_potential_utils.hpp` and the smooth-to-dense density
        transfer in the accumulator. Make those two loops sphere-only, then let
        `stick_weights_for_grid` leave out-of-sphere sticks at weight zero. Roughly halves
        the all-to-all volume.
-   [ ] **[M]** **Distribute the density G-vector list** — `PlanewaveBasis::g_vectors_` is
        the full density sphere on every rank whatever the grid does, and Ewald, the local
        potential, the mixer history and the G-vector table loop over it. About twenty
        consumer sites, each needing its own reduction. Small in bytes next to the solver,
        but it is the last replicated term that scales with the cell.
-   [ ] **[S]** **Shrink what is left of the wavefunction-initialization spike** — the
        atomic-orbital start still holds four arrays of `max(natwfc, nbnd) x npw` at once
        (the identity-overlap copy is gone), and the orbitals are built into a vector of
        vectors and then packed rather than written into the contiguous buffer directly.
        Measured on the 54-atom silicon cell, that step adds 0.68 GiB of resident memory
        that is never returned to the system, so the cost outlives the arrays.

-   [x] **[S]** **MPI in CI** — this item was stale as of 2026-07-27: a `mpi-validation` job
        (`.github/workflows/ci.yml`) already builds `-DQ3_WITH_MPI=ON` and runs both
        `test_mpi_topology_invariance.py` and `test_mpi_band_parallelism.py` (1/2/4 ranks); the
        `q3_tests_mpi` GoogleTest target already has 10 real unit tests
        (`tests/unit/layer_0_core/execution/test_mpi_execution_policy.cpp`), not an empty
        placeholder. Unclear exactly when this landed since the item was never flipped when it
        did — a reminder to flip TODOs items in the same PR that closes them, per this file's own
        "handoff on ship" convention.
-   [x] **[S]** **Extend MPI validation tests with LSDA + smearing cases** — added to both files
        using the existing Ni FCC LSDA example (`examples/total_energy--metal--magnetic--nickel/nc.yml`),
        k-grid overlaid down to 4x4x4+shift (10 irreducible k-points) purely for MPI-test speed
        (NOT 2x2x2+shift -- only 2 irreducible k-points there, fewer than the 4 pools these
        tests request, which Q3 silently clamps down to match, corrupting the intended
        topology; see the k-point-pool-clamping note this same investigation added to both
        test files' docstrings). `test_mpi_topology_invariance.py` gained
        `test_ni_lsda_scf_runs_on_n_ranks` (1/2/4 ranks, real/passing) and
        `test_ni_lsda_smearing_topology_invariance_1_vs_4_ranks`, parametrized over all four
        supported smearing schemes -- **currently marked `xfail` (non-strict)**, see the new
        item immediately below. `test_mpi_band_parallelism.py` gained
        `test_hubbard_lsda_band_parallelism_topology_invariance` (LSDA + DFT+U/Hubbard, 4-pool
        vs 2x2-pool vs 1-pool-of-4 topologies) -- **real, reliably passing** (4/4 repeated runs
        on real `mpirun`, 1/2/4 ranks): this is also the first real-MPI-hardware confirmation
        of the `reduce_within_group` fix for Hubbard's occupation-matrix accumulation
        (feature/2026-07-27-1-hubbard-forces-stress, §1.3 above), which had previously only
        been validated by code-reading symmetry since no MPI build was available in that
        environment. No CI workflow change needed: `.github/workflows/ci.yml`'s
        `mpi-validation` job already runs both files wholesale, so the new cases (and the
        xfail below) are covered automatically.
-   [x] **[S]** **SCF convergence branch desyncs MPI collective call counts under k-point-pool
        parallelism + smearing** — found 2026-07-27 while writing the LSDA+smearing MPI test
        extension above: `test_ni_lsda_smearing_topology_invariance_1_vs_4_ranks` was
        **nondeterministic** across repeated runs of the identical input/topology -- sometimes
        passing, sometimes exceeding the 1e-8 Ry tolerance by up to ~2.4e-7 Ry, sometimes
        crashing outright with `MPI_ERR_TRUNCATE: message truncated` on a pool-scoped ("SPLIT
        FROM 0") communicator. Reproduced on a plain LSDA+smearing system with no Hubbard/DFT+U
        involved (so distinct from, and unfixed by, the `reduce_within_group` Hubbard fix).
    -   **Root cause, confirmed 2026-07-27 by instrumented per-rank hexfloat tracing of `dr2`
        and fixed in `feature/2026-07-27-5-fix-mpi-scf-convergence-desync`**: the density
        residual (`compute_residual_between`, `iterator/kohn_sham/planewave/init.hpp`) is
        computed independently on every rank from that rank's own `rho_r_to_g` FFTs, and the
        rho FFT grid uses `FFTW_MEASURE` -- each rank timing-selects its own FFT algorithm at
        startup, so bit-identical (Allreduce-replicated) input densities yield bit-DIFFERENT
        residuals on different pool ranks, re-rolled every run (measured on the fermi-dirac
        Ni repro, 4 ranks / 4 pools: 4 distinct bit patterns from **iteration 1** onward,
        starting at last-bit level ~1e-15 relative and amplified through the nonlinear
        mixing-history feedback to factors of 2-30x near convergence -- dr2 tends to zero
        while the per-rank noise it is measured against does not). That per-rank value gates
        three branches that alter the inter-pool collective
        sequence: the SCF loop exit in `iterate()`, the skip-mixing early return in
        `perform_iteration()`, and the first-iteration re-diagonalization. In 6/6 instrumented
        crash runs, the final iteration showed a split decision (e.g. rank 0 at 8.06e-09 →
        stop, ranks 1-3 at 1.1-2.7e-08 → continue): the stopping rank proceeds to
        finalize/forces/stress collectives while the others enter the next iteration's
        Fermi-bisection and charge-density Allreduces -- `MPI_ERR_TRUNCATE` when the
        mismatched calls differ in message size, silently corrupted totals when they happen
        to match. The prior "suspected mechanism" writeup here blamed the skipped
        `mix_density()` FFTW-MPI collectives -- that part was WRONG for the observed 4x1
        topology (1 rank/pool → mixing is rank-local, no collectives); the desynced
        collectives are Fermi/density/forces, and the divergence source is per-rank
        `FFTW_MEASURE` planning, not a missing reduction.
    -   **Fix**: `compute_residual_between` now reduces the residual across pools with
        `reduce_scalar_max` (inter-pool `MPI_MAX`) before anything branches on it -- every
        rank sees the same value, and MAX is conservative (the SCF stops only when the
        slowest pool's estimate has converged). Within a pool no synchronization is needed:
        band ranks compute the residual from pool-collectively-transformed, replicated
        G-space data, so they already agree bitwise. Serial path unchanged (policy no-op).
        Verification: the previously 6/6-crashing fermi-dirac repro passes 10/10 with the
        fix (30/30 clean MPI runs total across the verification batches), and same-topology
        repeated runs of the stiff smearing schemes became **bit-deterministic** (gaussian /
        methfessel-paxton / marzari-vanderbilt: zero energy spread across 3x np=4 and
        2x np=1 repeats each). The `xfail` marker on
        `test_ni_lsda_smearing_topology_invariance_1_vs_4_ranks` is REMOVED (per this file's
        own `test_gamma_phonons.py` precedent); serial C++ suite unchanged.
    -   **Post-fix finding that recalibrated the test tolerances** (measured, 2026-07-27):
        with desync eliminated, the GENUINE 1-vs-4-rank converged-energy differences are
        gaussian 4.3e-08, methfessel-paxton 2.0e-09, marzari-vanderbilt 1.6e-08 Ry --
        stopping-iteration and Allreduce-vs-serial rounding differences on a metallic
        soft-moment system, NOT corruption. This means the pre-fix sub-1e-7 "tolerance
        breaches" (gaussian 4.36e-08, marzari-vanderbilt 1.53e-08) matched the healthy
        deltas almost exactly and were largely or wholly legitimate physics failing an
        unrealistically tight 1e-8 tolerance borrowed from the insulating Si case; the
        bug's unambiguous symptoms were the crashes and the run-to-run nondeterminism.
        Fermi-Dirac at degauss 0.02 Ry is qualitatively different: its moment restoring
        force is so soft that dr2 < 1e-8 is satisfied anywhere along a nearly-degenerate
        moment valley (converged moments 0.304-0.306 across identical repeated np=4 runs,
        energies spreading ~1e-4 via FFTW_MEASURE run-to-run plan differences; np=1 runs
        stayed deterministic in a 2-run sample), so only a valley-scale comparison is
        physically meaningful for it. The test now uses per-scheme calibrated tolerances
        (`SMEARING_TOPOLOGY_TOLERANCE_RYDBERG`: 2e-7 stiff schemes / 5e-3 fermi-dirac) with
        the measured table in its docstring; the crash mode stays guarded by all four
        schemes' exit codes.
-   [x] **[S]** **Deterministic FFT planning knob** — implements the follow-up flagged
        above: `control.use_deterministic_fft_planning: true` (`InputData::use_deterministic_fft_planning`,
        default `false`) threads a `use_measured_fft_planning` bool through
        `PlanewaveBasis::generate()` -> `PlanewaveFFTPair::initialize()` -> each grid's
        `FFTGrid::init()`, replacing the hardcoded `true` (FFTW*MEASURE, per-process-timed) at
        all three of `PlanewaveFFTPair`'s call sites with the configurable value; `false`
        (the new default's \_complement*, i.e. requesting determinism) selects FFTW*ESTIMATE
        (fixed heuristic, no runtime timing). Verified: the previously-nondeterministic
        Fermi-Dirac Ni repro (4 ranks / 4 pools, the exact case above) gives **bit-identical**
        total energy, iteration count, and converged magnetization across 5 repeated runs with
        the flag on (regression test:
        `test_deterministic_fft_planning_gives_bit_identical_repeated_runs`,
        `tests/validation/test_mpi_topology_invariance.py`, asserts spread < 1e-9 Ry across 3
        runs). Serial and namelist-parser behavior unchanged (default `false` preserves
        current FFTW_MEASURE performance); the namelist parser does not yet expose this key
        (out of scope here — namelist/YAML control-key parity generally is a separate, larger
        gap than just this one flag, see the "Namelist parser vs YAML parser parity" item
        below, which is scoped to spin/smearing only and does not cover this).
        **Follow-up completed (2026-07-28)**: `run_ni_lsda_scf_on_n_ranks()` (shared by
        `test_ni_lsda_smearing_topology_invariance_1_vs_4_ranks` and the plain
        `test_ni_lsda_scf_runs_on_n_ranks` smoke test) now sets
        `control.use_deterministic_fft_planning: true` for all four smearing schemes. Re-measured
        the Fermi-Dirac case (the one whose 5e-3 tolerance was mostly covering FFTW_MEASURE
        run-to-run noise, not genuine cross-topology physics): 3x np=1 + 3x np=4 repeated runs
        gave energies bit-identical to 12 printed decimal digits across all 6 runs (same-topology
        \_and* cross-topology spread both exactly 0.0). Tightened
        `SMEARING_TOPOLOGY_TOLERANCE_RYDBERG["fermi-dirac"]` from `5.0e-3` to `1.0e-4` — a 50x
        tightening, matching the magnitude suggested when this flag was first scoped, while
        keeping a >1000x margin over the single-sample zero measurement rather than asserting
        that literal value. The three already-tight schemes
        (gaussian/methfessel-paxton/marzari-vanderbilt, 2e-7) are unaffected by this since their
        tolerance floor was genuine cross-topology convergence-path divergence on a soft-moment
        system, not FFT-plan nondeterminism — re-verified all four schemes still pass
        (`tests/validation/test_mpi_topology_invariance.py`, 13/13, including 3 independent
        repeats of the fermi-dirac parametrization at the new tolerance).
-   [ ] **[M]** **Distribute charge-density G-vectors** — band parallelism distributes
        wavefunction plane-wave coefficients by FFTW slab, but the charge-density G-vector
        set stays replicated on every band rank, and a zero-plane-wave slab silently falls
        back to fully replicated G-vectors. Distribute the ρ-grid G-vectors and balance
        slab plane-wave counts. Also: stress computation is skipped when `band_size > 1`.

### 2.4 Benchmarks & docs

-   [ ] **[M]** **Scheduled wall-clock comparison against Quantum ESPRESSO** — Q3's own
        timings are now recorded and gated against a committed baseline (the harness,
        the comparison and the generated PERFORMANCE.md section are in
        [FUNCTIONALITY.md](FUNCTIONALITY.md) §2), but nothing runs the same decks through
        QE on a schedule and tracks the ratio, so the head-to-head numbers in
        [PERFORMANCE.md](PERFORMANCE.md) are still measured by hand and age silently.
        Needs the pinned QE build (7.5 + LibXC 7.0.0), which is on no CI runner today.
        The CUDA workflow's benchmark steps also still only echo their timings to the
        log; they should emit the same JSON records as the CPU harness, which would give
        a GPU runner class a baseline of its own. That leg belongs in a workflow that
        runs only when dispatched, as `cuda.yml` does: the one GPU runner is
        self-hosted, and pull-request code does not run on it.
-   [ ] **[S]** **Band-parallel MPI vs FFTW-threading benchmarks** — run multi-run averaged
        benchmarks for np=1/OMP=2 vs np=2/OMP=1 on the Si NC 3×3×3 supercell to produce
        official numbers for [PERFORMANCE.md](PERFORMANCE.md) (moved here from that file's
        TODO list per the single-source-of-truth rule).
-   [ ] **[L]** **PAW performance tuning** — the matched copper profile measures smooth
        augmentation at about 1 ms per SCF iteration and post-SCF PAW stress augmentation
        at about 108 ms. The PAW one-center refresh has a 62.3 ms median and accounts for
        nearly all of density construction. Split its angular-quadrature, radial-integration,
        and D-matrix phases, then cache proven invariant work before considering a lower
        correction frequency.
-   [ ] **[L]** **Parallelization documentation** — document the strategy alongside the
        layered architecture (k-point MPI, band parallelism, G-vector parallelism, GPU
        offloading); much of this now exists in code but not in the architecture docs.

---

## 3. Engineering (Developer Experience)

---

## 4. Python Layer & Workflows

-   [ ] **[M]** **Variable-cell relaxation in bindings mode** — `q3_core.SCFSolver.solve()`
        (src/bindings/q3_core_api.hpp) takes only fractional coordinates; there is no API to
        update the cell it was constructed with. `Q3ASECalculator` now raises rather than
        silently ignoring a cell mismatch in bindings mode (see FUNCTIONALITY.md's
        Variable-Cell Relaxation note for the bug history this guards against), and
        `run_vc_relax_workflow` forces `mode="subprocess"` accordingly — correct, but slower
        (one process spawn + full YAML/PP-path resolution per SCF step) than bindings mode's
        warm-started, in-process `SCFSolver`. Extending the nanobind API with a
        `set_cell(vectors)` method (mirroring the C++-side `set_free_lattice_vectors` used by
        the YAML "vectors" input path) and re-running `Basis`/`Pseudopotential` setup on cell
        change would let variable-cell relaxation use bindings mode too.
-   [x] **[S]** **`FixSymmetry` for variable-cell relaxation** — `run_relax_workflow` supported
        `use_symmetry` (ASE's `FixSymmetry` constraint); `run_vc_relax_workflow` did not. Added
        in `feature/2026-07-27-2b-vc-relax-fix-symmetry`, mirroring the ionic-only workflow's
        identical parameter and default (`True`); applied to `atoms` before wrapping in the cell
        filter, so both ionic displacements and cell changes are constrained. Verified the new
        default doesn't change `test_vc_relaxation.py`'s existing (isotropic Si) results.
-   [ ] **[S]** **Anisotropic variable-cell relaxation validation** — `test_vc_relaxation.py`
        validates recovery from an ISOTROPIC starting strain only (matching the original
        five-feature-chain plan's scope). A genuinely anisotropic or sheared starting cell
        (e.g. tetragonal distortion of a cubic crystal) would exercise the full 3x3 cell-filter
        machinery more thoroughly than a purely hydrostatic scan does.
        **Concrete blocker found 2026-07-27**: `_material_json_to_ase_atoms`'s (`relax.py`)
        `"FCC"` branch builds the cell from `lattice.a` alone (`ibrav=2` formula) and silently
        ignores `lattice.b`/`lattice.c` entirely — setting them to different values in the
        material JSON does NOT produce an anisotropic cell; it is still exactly cubic. The only
        path that honors independent `a`/`b`/`c`/angles is the generic/triclinic fallback
        branch (anything outside `{FCC, BCC, SC, CUB, HEX}`), which carries its own explicit,
        unverified risk (see that branch's own `WARNING` comment): Q3's C++ parser may build the
        Cartesian frame for a generic triclinic lattice differently than this Python fallback
        does, which would silently feed BFGS forces in the wrong frame. Before attempting this
        item, either (a) verify the two frame conventions agree for a concrete generic-lattice
        case (compare a Q3-computed force against this Python conversion's frame, analogous to
        how the FCC/BCC branches were originally verified), or (b) add a properly-verified
        tetragonal-specific branch (conventional 4-atom cubic cell, one axis-differentiated ibrav
        formula) rather than relying on the generic fallback.
-   [ ] **[M]** **Elastic tensor beyond cubic symmetry** — `q3.workflows.elastic` fits the
        three independent constants of a cubic crystal (C11, C12, C44) from two strain
        patterns, and says so in its module docstring and in `FUNCTIONALITY.md`. A hexagonal
        crystal has six independent constants and an orthorhombic one nine; applying the cubic
        strain set to either returns three numbers that are weighted averages of several real
        constants rather than any of them. What is needed is a strain set per Laue class, or
        all six Voigt patterns and a least-squares fit of the full 6×6 tensor (any symmetry, at
        three times the SCF runs) — a table of strain patterns plus a symmetry lookup, not new
        physics. Guard the cubic path with an explicit refusal once a symmetry is detected that
        it cannot speak for, rather than letting it answer anyway
-   [ ] **[M]** **Ion-relaxed (internal-strain) C44** — the elastic workflow holds fractional
        coordinates fixed at every strain, so its C44 is the **clamped-ion** value. For silicon
        that is the whole difference between the measured 99.8 GPa and the ~75 GPa that
        compares with the experimental 79.6 GPa: under a C44 shear the diamond structure's two
        sublattices slide relative to each other by the Kleinman internal-strain parameter, and
        that relaxation softens the response. C11 and C12 are unaffected — the uniaxial strains
        that determine them preserve each site's symmetry, so no internal displacement is
        induced at all — and are already fully relaxed as reported. The fix is to relax ions at
        each strained cell before reading its stress: an ASE optimizer on the same `Atoms`,
        with the cell held fixed, between `set_cell` and `get_stress`. The same argument
        applies to an equation-of-state scan of a structure with a free internal coordinate
        (silicon has none, so its scan is exact)
-   [ ] **[S]** **`cell_dofree`: constrained and partial cell relaxation** — the `&CELL`
        namelist's `cell_dofree` (relax the shape at fixed volume, one axis only, 2D only, …)
        is still refused by `src/io/input/parser/qe.hpp`, while its neighbour `press` now names
        `run_vc_relax_workflow(target_pressure_gpa=...)`. ASE's cell filters already take
        `mask`, `hydrostatic_strain` and `constant_volume`, which cover most of what
        `cell_dofree` spells, so this is a parameter and a mapping table rather than new
        machinery — but note ASE's own warning that `constant_volume` only approximately
        conserves the volume and breaks energy/force consistency, so a fixed-volume shape
        relaxation needs FIRE rather than BFGS and should say so
-   [ ] **[M]** **JAX primitives** — make `E(R)` differentiable end-to-end via `q3.jax.primitives`;
        stub already exists at `src/py/q3/jax/primitives.py`.

    **Why it's useful.** Once the total energy is a differentiable JAX function of atomic
    positions and cell parameters, you unlock:

    -   **Force constants / dynamical matrix** — NOTE (2026-07-09): the Hellmann-Feynman
        `custom_jvp` supplies FIRST derivatives only; `jax.jacobian(jax.grad(energy))` through
        the SCF fixed point requires the implicit-function-theorem machinery that is
        mathematically DFPT (see the roadmap: A3 + the post-A3 SCF fixed-point adjoint).
        Second derivatives do not come "for free" from the HF rule.
    -   **Elastic tensor** — same caveat: `jax.hessian` w.r.t. cell parameters needs the
        SCF adjoint, not just the HF JVP. The finite-difference route exists today in
        `q3.workflows.elastic` (cubic only); this item is about getting the same tensor
        analytically, without one SCF per strain.
    -   **Geometry optimization** — standard JAX/optax gradient-based optimizers consume
        `value_and_grad`; enables a JIT-compiled relax loop with no subprocess overhead.
    -   **ML potential training** — use Q3 as the reference code inside a JAX neural network
        potential training loop (NequIP, MACE); backprop through Q3 for energy/force matching.
    -   **Sensitivity analysis** — propagate uncertainty in pseudopotential or XC parameters
        through to observables via `jax.vjp`.

    **Why it's non-trivial.** The SCF loop is an implicit function — it finds the density ρ
    satisfying F(ρ) = ρ (a fixed point). You cannot unroll and autodiff through it naively.

    **Implementation strategy (Hellmann-Feynman `custom_jvp`).**
    The Hellmann-Feynman theorem states that at self-consistency the force is:

        F_I = -dE/dR_I = -⟨Ψ| ∂H/∂R_I |Ψ⟩

    The wavefunction terms vanish because ∂E/∂Ψ = 0 at the SCF fixed point — so you only
    need to differentiate the Hamiltonian, not the wavefunction. Q3 already computes
    Hellmann-Feynman forces analytically in `forces.hpp` (ion-ion gradient + local PP
    gradient + nonlocal KB projector Pulay correction). The `custom_jvp` rule for
    `scf_energy(positions)` is therefore: _call `q3_core.SCFSolver.solve()`, return the
    already-computed HF forces as the JVP tangent_. No extra SCF calls needed; the
    derivative comes for free from the converged calculation.

    Requires `q3_core` (Mode B nanobind bindings) to be built.

-   [x] **[M]** **`mat3ra-made` k-path** — done. `mat3ra-made` is a declared dependency;
        `q3.workflows.kpath` takes the AFLOW high-symmetry points from
        `ReciprocalLattice.symmetry_points` and the recommended paths from
        `default_kpoint_path`, and `q3.workflows.interpolation`'s k-mesh rule is
        `get_dimensions_from_spacing`. What Q3 still owns is the one thing made does not
        model: **which primitive vectors the engine builds.** made keeps only a, b, c and
        the angles and rebuilds vectors in its own frame, while `lattice.hpp` fixes QE's
        ibrav 2/3/4 orientations — so a tabulated fractional triple is re-expressed on the
        way in (identity for CUB/TET/ORC, an integer determinant-one matrix for FCC, BCC
        and HEX). Reading AFLOW's numbers as if that matrix were the identity is what put
        four points outside or inside the Brillouin zone before this landed. seekpath was
        not needed.

        Still refused, and made cannot fix it: the eight centred and low-symmetry types.
        made has their points, expressed in the primitive cell of that centring, while
        Q3's parser builds the **conventional** cell from a/b/c/angles for every non-cubic
        type — so those coordinates would be read against a different lattice. Teaching
        `q3.structure.lattice` to build centred primitive cells is the prerequisite, and
        `conventional_cell_parameters` is where that conversion already lives.

-   [ ] **[M]** **`q3.io.input_writer`** — convert a `mat3ra.made` Material object → Q3 YAML (counterpart to existing `output_writer.py`)

-   [ ] **[M]** **The rest of the Mat3ra Python stack** — what is left after the k-path,
        measured 2026-09-12 rather than guessed. The implemented Python-layer plan
        designed this layer around made, `mat3ra-code` and `ase`; three of the four
        deviations the 2026-06-30 audit found are now closed (the k-path above, and the
        three copies of the cell construction that are now one). The rest:

    1.  **Units: four modules and `properties/` still read the older table.**
        `q3.constants` takes its factors from `mat3ra.utils.constants`, the table the stack
        generates, which agrees with the engine's CODATA constants to 7e-10
        (`tests/python/test_constants.py`), and everything under `workflows/wannier/`,
        `workflows/interpolation.py`, `workflows/band_distance.py` and
        `structure/lattice.py` reads it. `io/output_writer.py`, `workflows/relax.py`,
        `workflows/vc_relax.py`, `calculators/ase_calculator.py` and `properties/` (the
        eV/Å³-to-GPa factor) still read `mat3ra.code.constants.Coefficients`, a
        hand-copied table whose `EV_TO_RY` is 3.6e-7 from the engine's. Moving them is a
        one-line import each; what makes it its own change is that it moves every energy
        those modules convert in the seventh digit, so the tests that pin converted values
        have to be re-read first.
    2.  **`structure/ase_atoms.py` (188 lines) — blocked, not dropped.** It duplicates
        made's `to_ase` / `from_ase`, but made's converter imports pymatgen at load time
        and pymatgen is only in made's `tools` extra; and 45 of those lines are a
        guarantee made's round trip does not make — species labels such as an
        antiferromagnet's `Cr1` / `Cr2`, which carry their own pseudopotentials and
        moments and collapse to `Cr` through chemical symbols. File the load-time import
        against made; keep the label logic whatever happens.
    3.  **`io/output_writer.py` (554 lines)** is the job `mat3ra-express` exists for, but
        express emits ESSE properties and Q3's result schemas carry no ESSE `$ref` at all,
        despite the HARD RULE in AGENTS.md § JSON Formatting. Making the results
        ESSE-shaped is the prerequisite; after it, an express `apps/q3` parser is a thin
        JSON reader rather than a private extractor.
    4.  **`workflows/wannier/` — split, one class per module (2026-09-20).** The package is
        a facade over `driver` (the entry point), `protocol` (`WannierProtocol`, the mesh
        loop, under a caller's `protocol_options`, returning a `workflow_result`),
        `ladder_state`, `accept_gate`, `verdict` and `evaluation` (what the rungs share),
        `rungs/` (one class per rung, `LADDER` in order), `diagnostic`, `route`,
        `attempt_status` and `rung_name` (the engine's codes and the words of the history:
        the code alone routes, there is no prose fallback), `trial_shells` (the rule) and
        `trial_orbitals` (the rule over a material); `settings` holds every fixed choice and
        `utils` the helpers with no state. What is file I/O sits a tier down: the UPF shell
        reader is `q3.io.pseudo_atomic_shells`, the manifest's file references are
        `q3.io.input_reader`, and the two blocks are `q3.io.interpolation_settings` and
        `q3.io.interpolation.results`. The engine chain stayed in
        `workflows/interpolation.py` with the mesh and k-path derivations both workflows
        share.

        What the campaign reuses: `ase.build.bulk`, `ase.spacegroup.crystal`,
        `set_cell(scale_atoms=True)`, `ase.neighborlist` for the shortest bond (which covers
        a one-atom cell over its periodic images, as `get_all_distances(mic=True)` does not),
        and q3's own `ase_atoms_to_material_json`, `material_json_to_ase_atoms`,
        `write_material_json` and `write_scf_input`. One candidate checked and **rejected**:
        its `rattled` displaces every atom by a bounded uniform fraction of the shortest
        bond where ASE's `rattle` is an unbounded Gaussian — swapping it would change the
        roster's physics and invalidate the seeds the campaign table was measured with.

        A note on the dependency, since it is the reason this was put off: made used to pin
        `numpy<=1.26.4` in its **core** metadata on pymatgen's behalf, though only its
        `tools` extra imports pymatgen. That did not cap Q3 cleanly — it made pip, given
        the numpy 2 that `ase`'s newest scipy wants, backtrack made to a 2024 release
        without `mat3ra.made.reciprocal` and fail at import. made has since moved the pin
        into `[tools]` and declared the scipy its core already imported; from
        mat3ra-made 2026.9.12.post0, `.[test,relax,build]` resolves to numpy 2.5 with no
        backtracking. Keep the `mat3ra-made>=` floor: below it the old behaviour returns.

    5.  **Strain to mat3ra-made; equation of state and elastic constants to
        mat3ra-prode (2026-10-09).** `q3.workflows.equation_of_state` and `q3.workflows.elastic` (and
        the target-pressure path of `vc_relax`) are `run_*` orchestration only. What they
        call is in two places marked as the pieces to upstream, `q3/structure/strain.py`
        and `q3/properties/`; once made has them, both become imports and the workflow
        bodies do not change. made already has the strain operation,
        `create_strain(material, strain_matrix=..., scale_factor=...)` and
        `get_isotropic_strain_matrix` in
        `mat3ra.made.tools.build_components.operations.core.modifications.strain`, but Q3
        cannot call it, for two reasons that are made's to fix:

        -   It is under `mat3ra.made.tools`, which imports pymatgen at load time, so it
            needs made's `tools` extra (`pymatgen==2024.4.13`, `numpy<=1.26.4`) — the
            numpy-1 pin Q3's `mat3ra-made>=` floor exists to stay clear of. The same holds
            for `to_ase` / `from_ase` (item 2).
        -   It strains made's `Lattice`, which keeps a, b, c and the angles and rebuilds
            the vectors in a frame of its own. Measured on the silicon example: a Voigt yz
            shear of 0.01 comes back with angles (59.595°, 59.9997°, 59.865°) — a mixed
            deformation, because made's FCC frame does not have the cubic axes along x, y
            and z — so the cubic C44 would be contaminated by C11 − C12. A volume scan is
            unaffected; an elastic constant is not.

        What to move upstream (signatures as Q3 has them today; arrays are numpy,
        stresses are ASE's tensile-positive Voigt order). Strain goes to mat3ra-made;
        everything under `q3/properties/` — stress units, the equation of state and the
        elastic constants — is a derived property and goes to mat3ra-prode, the stack's
        property package:

        -   strain — beside `create_strain`, importable without the `tools` extra, and
            keeping the vectors' orientation (in made's own terms, a frame-preserving
            `strain(material, strain_matrix)`):
            -   `isotropic_strain_matrix(linear_scale_factor: float) -> np.ndarray`
            -   `strain_matrix_from_voigt(voigt_strain) -> np.ndarray` — I + ε,
                engineering shear
            -   `strain_atoms(atoms, reference_cell, strain_matrix) -> None`
        -   stress units:
            -   `voigt_stress_gpa(voigt_stress_ev_per_angstrom3) -> np.ndarray`
            -   `hydrostatic_pressure_gpa(voigt_stress_ev_per_angstrom3) -> float` —
                compression positive
            -   `pressure_gpa_to_ev_per_angstrom3(pressure_gpa: float) -> float`
        -   equation of state:
            -   `DEFAULT_LINEAR_SCALE_FACTORS` — 0.96 … 1.04
            -   `fit_birch_murnaghan(volumes_angstrom3, energies_ev) -> dict` — ESSE
                `volume` and `total_energy`, plus `bulk_modulus_gpa` and
                `bulk_modulus_pressure_derivative`, ESSE having no schema for either
        -   cubic elastic constants — the sign-symmetric set and the fit move together,
            since the zero-intercept slopes are what cancel the k-point-set artefact (see
            the module docstring):
            -   `DEFAULT_STRAIN_MAGNITUDES` — 0.005, 0.01; `CUBIC_STRAIN_PATTERNS`
            -   `cubic_voigt_strain_set(strain_magnitudes) -> list[tuple[str, np.ndarray]]`
                — reference first, every magnitude in both signs
            -   `fit_cubic_elastic_constants(points) -> dict[str, float]`

    6.  **Phonon properties to mat3ra-prode; two ESSE schemas missing (2026-10-09).**
        `q3.workflows.phonons` is `run_*` orchestration only: Phonopy and ASE compute,
        and `q3/properties/phonons.py` gives their arrays ESSE shapes. It imports neither
        library, so it moves to mat3ra-prode as it is:

        -   `phonon_frequencies_ev_to_thz_and_cm1(frequencies_ev) -> (thz, cm1)` — needs
            ASE's units
        -   `phonon_dispersions(qpoints, frequencies, units="THz") -> dict` — ESSE
            `phonon_dispersions`, frequencies given `[q-point][branch]`
        -   `phonon_density_of_states(frequency_points_thz, states_per_thz) -> dict` —
            ESSE `phonon_dos`
        -   `zero_point_energy(value, units="kJ/mol") -> dict` — ESSE `zero_point_energy`
        -   `harmonic_thermal_properties(temperatures, free_energy, entropy, heat_capacity)`
            — see below
        -   `force_constants_file(path) -> dict` — ESSE `file_content`

        Two results have no ESSE schema, checked against mat3ra-esse 2026.10.8, and are
        mat3ra-esse items:

        -   **Temperature-dependent thermal properties.** `heat_capacity_cv` and
            `total_entropy` are single scalars in cal/mol-K that do not record the
            temperature they hold at (molecular thermochemistry), and there is no
            free-energy property. Q3 returns free energy, entropy and heat capacity each as
            ESSE's abstract `core/abstract/2d_plot` (temperature in K on x, Phonopy's
            kJ/mol and J/K/mol on y): the base `phonon_dos` already extends, so a property
            schema added later is a `name` and a units enum on the same data.
        -   **Force constants.** They stay in Phonopy's `phonopy_params.yaml`, reported as
            the ESSE `file_content` property; a schema would need the supercell matrix
            and an (N, N, 3, 3) array in eV/Å².

        Two pieces of workflow plumbing are marked for mat3ra-wode:
        `q3.workflows.kpath.band_path_from_input` (the band path from the input file's
        defaults) and the k-grid rescaling in `q3.io.write_supercell_scf_input`. wode's
        `PointsGridDataProvider` is the right home for the second, keeping a grid by
        KPPRA across cells, but its `calculate_dimensions` is not implemented in Python
        yet.

---

### 3.1 I/O & Interoperability

-   [x] **[S]** **Unify JSON output units** — audited: this item's premise (a separately-written
        `_total_energy.json` in eV) does **not exist anywhere in the current tree** (confirmed
        by repo-wide search for both the filename pattern and the literal string `"eV"` in
        `src/`). Every entry point (`main.cpp`, `main_mpi.cpp`) writes exactly one file,
        `<prefix>_results.json`, via `FlavorResult::write_results_json()`
        (`flavors/result.hpp`, since removed) -- already
        consistently Ry (energies), Ry/bohr (forces), kbar (stress), matching every live Python
        consumer (`ase_calculator.py`, `output_writer.py`, `jax/primitives.py`, the workflow
        modules) and every committed QE-comparison reference fixture
        (`examples/*/reference/results.json`). Git history (`dde66b7`, "refactor: flavors
        architecture… remove result wrapper types") explains the stale premise: the _old_,
        now-deleted per-property writer hierarchy already nested ESSE-shaped
        `{"name":"total_energy","value":…,"units":"Ry"}` inside `_results.json`; the newer flat
        writer regressed by dropping the `units` field, not by changing to a different unit.
        ESSE itself defines no single canonical unit per property (an _enum_ of legal units,
        e.g. energy: `{Ry, eV, …}`) -- Ry/Ry-bohr/kbar are all ESSE-legal, so "unifying" here
        meant confirming there was nothing left to unify (already one file, one unit set) and
        making that fact self-describing — see the item below.
-   [x] **[S]** **Add `units` fields** to all ESSE-shaped result JSON (forces, stress, contributions) —
        added a `units_json()` helper on `FlavorResult` (`flavors/result.hpp`) returning
        `{"energy": "Ry", "forces": "Ry/bohr", "stress": "kbar", "eigenvalues": "Ry"}`, emitted
        as a new top-level `"units"` key from both `write_results_json()` and `to_json_string()`
        (the nanobind Python binding's direct-return path) -- purely additive, since no
        consumer needs a unit different from what's already written; confirmed by auditing
        every JSON consumer (Python calculators/workflows, `tests/regression/`,
        `tests/integration/test_cli.cpp`, every `tests/validation/test_*.py` that reads
        `total_energy_ry`/`forces_ry_per_bohr`/`stress_kbar`) -- all key-access by name, none do
        a strict whole-object equality check, so the new key breaks nothing. Regression tests:
        `tests/unit/layer_4_computation/flavors/test_result.cpp` (new file), asserting the
        `units` object's contents and that `write_results_json`/`to_json_string` agree on it.
        The unused, never-wired-up ESSE per-property scaffolding
        (`src/generated/esse/properties/scalar/total_energy.hpp`) already has a proper
        `RY` enum value in its `TotalEnergyUnitsEnum` (not hardcoded to `eV` as its one example
        fixture might suggest) -- no fix needed there; whenever it's wired up (see below), it
        should simply emit `RY` to match this file.
-   [x] **[S]** **Consolidate duplicate energy serialization** — done: flavors now return
        `generated::Results` itself, and `write_results_json()` and `results_json_string()` both
        serialize that one struct. Original item: `FlavorResult::write_results_json()`
        (`flavors/result.hpp`, since removed) and `to_json_string()`
        are two hand-synced copies of the identical schema in the same file (confirmed
        2026-07-28 while adding the `units` field above -- both needed updating together, by
        hand); one code path for both would remove that duplication. **Correction to this
        item's own previous description**: it claimed the per-property ESSE writers
        (`<prefix>_total_energy.json` etc.) are "emitted from `self_consistent_field.hpp` via
        `src/generated/esse/`" -- confirmed false by direct grep: `self_consistent_field.hpp`
        never references `src/generated/esse/` at all, and no per-property JSON file is ever
        written by any live code path. That scaffolding is unused, dead code, not a second
        live output path to consolidate against; the actual duplication needing consolidation
        is the two `FlavorResult` methods themselves.
-   [x] **[S]** **Parse-time validation: reject unknown YAML manifest keys** — `YAMLInputParser` previously read each known key with a plain `if (node["key"])` check and silently ignored anything else, so a misspelled or unsupported field (e.g. `convergance_threshold`, `smearing.scheme` instead of `smearing.type`) fell back to its default with no error at all -- the setting the user thought they'd changed simply never applied. Fixed by adding `reject_unknown_keys()` (`yaml.hpp`) and calling it at every manifest level Q3 fully owns the key set for: root, `model`, `method` (+ `cutoffs`, `self_consistent_field`/`scf` and its `mixing`/`smearing`/`kgrid` sub-objects and each `hubbard` list entry, `band_structure`), and `control` (+ each `band_path` list entry). Cross-checked the allow-lists against every real key-path actually used across `examples/**/*.yml` and `tests/**/*.yml` (via a one-off script, not committed) before writing the fixed lists -- this caught `model.type`/`model.subtype`/`method.type` as accepted-but-currently-unused keys present in every single example (self-documenting the calculation even though DFT/Kohn-Sham + planewave-pseudopotential are Q3's only supported model/method today); missing these from the allow-list would have made every existing example throw on its very first field.
    -   **Real bug found and fixed by this same audit**: `q3.io.input_writer.write_scf_input()` wrote `method.scf.smearing.{scheme,width}`, but the parser only ever read `.{type,degauss}` -- so any caller passing `smearing=`/`smearing_width=` got an SCF run with NO smearing applied at all, silently. Not caught earlier because no validation/integration test exercised `write_scf_input`'s smearing parameters end-to-end (only `tests/python/test_input_writer.py`'s structural assertions, which didn't check smearing). Fixed the writer to emit the correct keys; added `test_write_scf_input_smearing_uses_parser_keys` as a regression guard.
    -   **Deliberately not covered**: inline YAML-embedded `material`/`lattice`/`basis` (the `material: {lattice: ..., basis: ...}` form, as opposed to `material: <path>.json`) is NOT validated -- confirmed via the same cross-check that every committed example references an external material JSON file instead, so this path carries far less real-world typo risk; revisit if inline material embedding sees real usage.
    -   **Still open**: aligning the pw.x namelist parser against the YAML parser for spin/smearing key parity was not attempted here (a separate, independent piece of work) -- see the item below.
-   [x] **[M]** **Drop-in case set: the reference codes' own input files** (2026-08-08) —
        `tests/validation/test_reference_input_drop_in.py`, 12 cases. Every other comparison test
        generates the reference code's input _from_ Q3's YAML, which checks the physics but leaves
        a loophole: a wrong translation in the generator makes both sides agree on the wrong
        calculation. Here the reference file is primary and Q3 consumes it byte for byte.
    -   **10 QE cases** are the committed `scf_qe.in`/`scf_qe.out` pairs already in `examples/` —
        QE's own input and the output QE produced from it with the LibXC build AGENTS.md requires.
        Coverage: NC / ultrasoft / PAW-from-UPF; semiconductor, insulator, metal-with-smearing;
        ferromagnetic Fe and Ni and antiferromagnetic Cr (moment checked as well as energy);
        Gamma-only, where Q3 stores half the G sphere and QE stores all of it; and DFT+U through
        QE's HUBBARD card, where E_U is checked separately because a wrong E_U can hide inside a
        total with other terms to trade against.
    -   **Two new fixtures** generated here: `reference/us_hubbard/` (silicon + U = 3 eV on 3p) and
        `reference/nc/scf_qe_gamma.in`.
    -   **The reference numbers are committed as `reference.json`, not read from the `.out` files.**
        `.gitignore` carries `*.out`, so QE's outputs are untracked — a test reading them directly
        passes on the machine that ran QE and dies on a fresh clone with a missing file. The JSON
        beside each input is the committed record; where a `.out` _is_ present the test re-reads it
        and asserts the JSON still matches, so a stale record cannot survive on any machine able to
        notice. `tests/validation/refresh_qe_references.py` regenerates them. Verified by moving all
        29 `.out` files aside and re-running: 14 passed.
    -   Inputs are staged into a temporary tree that reproduces their _relative_ layout rather than
        being rewritten; rewriting `pseudo_dir` to an absolute path would quietly weaken the claim
        being tested, and running in place would write into the committed example tree.
    -   Added `--check` (`q3 <input> --check`) to support the corpus test: parse, report what the
        input resolved to, exit. Also the cheap answer to "will Q3 read this deck, and as what?".
    -   **Real defects on its first run**, in files that had been sitting in
        `examples/`: (1) `mixing_mode = 'plain'` — QE's _default_ — was stored verbatim and killed
        Q3 in the mixer factory; (2) Q3 reported no _absolute_ magnetization,
        which is the only number that distinguishes a converged antiferromagnet from one whose
        moments collapsed — both give zero total. Both fixed.
    -   **Measured agreement**: NC 2e-6 Ry, Gamma-only 4e-6, ultrasoft 2e-5, DFT+U total and E_U
        both within 2e-4, PAW 2.9e-4 to 6.9e-4 (the known one-center residual), Cr AFM 2.5e-4
        (1.7 meV/atom, with |m| 2.5929 against QE's 2.60). PAW and chromium carry their own
        tolerances set from these measurements, with the reason recorded beside each.
-   [x] **[M]** **QE namelist parser strictness** (2026-08-08) — `QEInputParser` silently ignored
        everything it did not recognize. That is how a `HUBBARD` card came to be dropped without
        a word: a PAW+U input ran to convergence and reported the **no-U** energy. Cards were
        fixed first; this closes the namelist-key half. - **Policy** (not YAML's — a real QE input carries dozens of inert I/O
        keys, and refusing all of them would make real inputs unreadable): honoured /
        physics-neutral / inert-at-its-QE-default / recognized-but-unimplemented-with-a-reason /
        unknown. Four tables in `qe_input::`. - **The default rule matters.** Inputs state defaults constantly; `noinv = .false.` and
        `tot_magnetization = -10000` (QE's "unspecified" sentinel) ask for exactly what Q3 does,
        so refusing them would be a false refusal. Adding it took QE's own regression corpus from
        93 to 96 of 150 `scf` inputs; honouring `restart_mode='from_scratch'` — QE's default,
        stated in most real inputs — took it from 52 to 93. - **Found by turning the policy on**: Q3 was _assuming_ `dftd3_version=4,
dftd3_threebody=.false.` and ignoring what the file said, so `vdw_corr='dft-d3'` meant
        something different in Q3 than in the input without saying so. Both are now checked. - **`celldm(2..6)` were dropped**, so every Bravais index needing an axial ratio or an
        angle — hexagonal, tetragonal, orthorhombic, monoclinic, triclinic — was built with
        c/a = 0, a collapsed cell. Plumbed through `InputData` → `build_crystal_from_input` →
        `Crystal::from_input_data`; the lattice code already supported all of them. QE's
        crystallographic `A`/`B`/`C`/`cosAB`/`cosAC`/`cosBC` form is converted exactly as
        Quantum ESPRESSO itself does, cosine-to-celldm mapping included. - **`&IONS`/`&CELL` were not recognized as namelists at all**, so an unknown header left
        the parser scanning the _previous_ namelist's keys and everything inside was dropped.
        `ion_dynamics`/`cell_dynamics` are now read and restricted to `bfgs`, which is what Q3
        relaxes with. Also newly honoured: `nosym`, `assume_isolated`, `nstep`, `forc_conv_thr`,
        `etot_conv_thr`, `restart_mode`, `scf_must_converge`. - **Two parsing bugs fixed underneath**: key matching was a bare substring search, so
        `conv_thr` matched inside `etot_conv_thr` (harmless only because they live in different
        namelists) and the single-letter key `A` would have matched almost anything; matching is
        now anchored to whole keys. Multi-assignment lines (`ibrav= 2, celldm(1)=10.2, nat= 2`,
        the style QE's own examples use) are split, so strictness sees every key rather than the
        first. Fortran's array-continuation form (`starting_magnetization = 0.5, 0.3`) is refused
        rather than silently truncated to its first element. - 20 tests in `tests/unit/io/input/parser/test_qe.cpp`. Verified behaviour-preserving on
        the real case: Q3 reading QE's own DFT+U input still gives −15.517022664 Ry, bit-identical
        to before the refactor.
-   [x] **[S]** **Namelist parser vs YAML parser parity for spin/smearing** — audited `qe.hpp`
        (`QEInputParser`) against `yaml.hpp` (`YAMLInputParser`) field-by-field for spin and
        smearing configuration.
    -   **Spin polarization: parity confirmed, no gap in the physics.** QE's `nspin` +
        indexed `starting_magnetization(i)` (Fortran-index convention, unindexed = species 1)
        and YAML's `number_of_spin_components` + species-symbol-keyed `starting_magnetization`
        map both resolve to the identical `InputData` fields (`number_of_spin_components`,
        `starting_magnetization` — a flat `vector<Real>` parallel to `species`). AFM
        sublattices already work identically in both (QE via `ATOMIC_SPECIES` card order +
        index, YAML via `Cr1`/`Cr2`-style symbols) -- proven by the parallel
        `chromium_afm_input` (`test_qe.cpp`) / `chromium_afm_yaml` (`test_yaml.cpp`) fixtures,
        which already tested the identical physical case on both parsers. This was an
        idiomatic difference (index vs. symbol), not a gap.
    -   **Smearing: parity confirmed at the value level, no gap in the physics.** QE's
        `degauss`/`smearing` (+ the `occupations='smearing'` compatibility default to
        Gaussian) and YAML's `smearing.type`/`smearing.degauss` both feed the same
        `InputData::smearing`/`smearing_width` fields, and both resolve smearing-scheme
        aliases (`gauss`, `m-p`/`mp`, `cold`/`m-v`/`mv`, `f-d`/`fd`) via the single shared
        `create_smearing` factory (`layer_4_computation/iterator/internal/smearing/factory.hpp`)
        -- there is no parser-level place alias support could diverge, by construction.
    -   **Real gap found and fixed**: YAML explicitly guards against its own most likely
        typo -- writing the QE abbreviation `nspin` into a YAML manifest throws with guidance
        pointing to the correct key. The namelist parser had no mirror-image guard: writing
        the YAML manifest key `number_of_spin_components` into a `&system` namelist block was
        silently ignored (not a substring of any existing key check, so it fell through with
        no error and no warning, leaving the run unpolarized). Fixed by adding the
        equivalent throw to `qe.hpp`, mirroring the exact idiom of the existing `nspin` guard.
        Added `QEParserSpinTest.ManifestSpinKeyRejectedWithGuidance` as a regression test
        (mirrors `YAMLParserSpinTest.NspinKeyRejectedWithGuidance`).
    -   **Real gap found and fixed**: the namelist parser's smearing/`degauss`/
        `occupations='smearing'` code path had **zero test coverage** (confirmed via grep --
        no `smear`/`degauss` matches anywhere in `test_qe.cpp`) despite being fully
        implemented and, per the audit above, functionally correct. Added a
        `QEParserSmearingTest` group (mirroring the existing `QEParserSpinTest` naming/style):
        parses smearing type + `degauss` correctly; `occupations='smearing'` with no explicit
        type defaults to Gaussian; default (no `occupations`/`smearing` keys at all) is
        `"none"`; and a regression guard confirming an explicit smearing type set _before_ a
        later `occupations='smearing'` line is not overwritten by the Gaussian-default logic
        (order-independence of that interaction, which the existing code already got right
        but had never been asserted).
    -   **Followed up 2026-08-08** (was: "not attempted, systemic"): `QEInputParser` now
        classifies every namelist key. It had no unknown-key mechanism at all, unlike YAML's
        pervasive `reject_unknown_keys`, so a typo anywhere was silently dropped. The shape
        is not YAML's, because a QE input legitimately
        carries dozens of inert I/O keys and refusing all of them would make real inputs
        unreadable: honoured / physics-neutral / inert-at-its-default / unimplemented-with-a-
        reason / unknown. See the strictness item below.
-   [ ] **[S]** **`units.forces` spells its unit differently from ESSE** — `results.json`
        declares `"forces": "Ry/bohr"`, and ESSE's `definitions/units.json#/force` enum spells
        the same unit `"Ry/a.u."` (bohr _is_ the atomic unit of length, so the values are
        identical and only the string differs). Every other unit the file declares is `$ref`'d
        straight to its ESSE enum in [schemas/q3/results.json](schemas/q3/results.json); this
        one carries a one-value local enum and a note saying why. Changing the string would
        move a value in a file every existing consumer already reads, so it needs a deliberate
        migration -- emit both for a release, or have ESSE accept the spelling -- rather than
        a quiet edit. Until then the schema documents the divergence instead of hiding it.
-   [ ] **[M]** **Checkpoint when ranks share a k-point pool** — with one rank per pool
        `q3_mpi` writes the full `<prefix>.save/` (each pool its own `wfc<N>.dat`, the root the
        pool-reduced density and potential; pinned by `tests/validation/test_mpi_checkpoint.py`).
        Ranks that share a pool divide each k-point's coefficients (or, with
        `--distribute-bands`, its bands) and the FFT grids between them, so `write_checkpoints()`
        still refuses there. Lifting it needs the coefficients gathered onto each pool's root in
        global plane-wave order, and the real-space grids onto the world root; the same gather
        is what a multi-rank BerkeleyGW export needs (see the export item below). The
        automatic topology never chooses this layout, so it matters only for a gamma-only
        supercell, whose single k-point leaves sharing a pool as its only multi-rank layout.
-   [ ] **[M]** **LSDA `.save` weight convention** — k-point weights are halved per spin channel for nspin=2 internally, but QE keeps `wk` un-halved in its data files; decide whether the QE-format checkpoint writers should emit un-halved weights so QE post-processors can read Q3 LSDA `.save` directories (LSDA checkpoint interop is currently unvalidated; the `wfcN.dat` writer also emits one file per k-point with `ispin = 1` and the charge-density writer a single spin record, where QE writes one file per (k, spin) and `nspin` density records)
-   [ ] **[M]** **ESSE schema codegen** — expand `src/generated/` coverage; optional schema validation at build time (`--check`)
-   [ ] **[M]** **BerkeleyGW mean-field export: HDF5 and untested coverage** —
        `mean_field_export: berkeleygw` (`io/mean_field_export/berkeleygw/`) writes WFN, RHO, and
        (for the spin-unpolarized, non-gamma-only case) vxc.dat (validated: byte-exact
        header/body layout, k-point-weight and coefficient-norm invariants, and an end-to-end
        export from a real converged silicon SCF state — see [FUNCTIONALITY.md](FUNCTIONALITY.md)
        §3.1). vxc.dat computes `<psi_nk|Vxc|psi_nk>` as a G-space convolution over psi_nk's own
        plane-wave coefficients and the already-available G-space Vxc
        (`exchange_correlation_potential_g`), sidestepping the dense-grid wavefunction transform
        pw2bgw itself uses (see `exchange_correlation.hpp`'s class comment for the full
        derivation) — measured on a real silicon SCF (2x2x2 k-mesh, dual=4, 1546/1557 plane
        waves at the 2 irreducible k-points, 8 bands): 0 of the 38,514,920 (G,G') pairs examined
        fell outside the stored charge-density G-set, and the largest computed imaginary part
        was ~7e-16 eV, both consistent with the derivation being exact for the standard dual=4
        cutoff ratio. Left open:
    -   **HDF5 `WFN.h5`/`RHO.h5`/`vxc.h5` not implemented** — only the original Fortran-binary/
        ASCII formats are supported; modern BerkeleyGW increasingly expects the HDF5 variant.
    -   **Spin-polarized (LSDA) and gamma-only coverage unverified end-to-end** — the LSDA body
        ordering (spin varies fastest within a band, per k-point) and the kw/ifmin/ifmax
        spin-degeneracy-factor derivation are implemented and unit-tested against a synthetic
        two-spin state, but not validated against a real converged LSDA SCF run or an actual
        BerkeleyGW/QE reference (none available in this environment). Gamma-only always exports
        WFN/RHO as WFN-Complex (flavor 2); pw2bgw's real-storage flavor 1 (`real_wfng()`'s
        time-reversal-pair reordering) is not implemented. vxc.dat additionally REJECTS both
        cases outright (nspin>1: only a spin-averaged Vxc is exposed; gamma-only: the direct
        G-space convolution would need the same `c(-G) = conj(c(G))` half-sphere reconstruction
        the real Hamiltonian applicator does, which is not implemented here) rather than writing
        a silently-wrong file — see `exchange_correlation.hpp`'s class comment.
    -   **No true MPI export — multi-rank runs refuse it (verified, guarded)** — measured on a
        2-rank silicon run before the guard existed: with 2 k-point pools the export wrote a
        WELL-FORMED WFN containing 1 of 2 k-points with its weight renormalized to 1.0
        (undetectable downstream); with 1 pool (band groups) each rank holds only a G-slab of
        every band (809 of 1546 plane waves at k-point 1) and the two ranks interleaved
        slab-partial records onto the same file — corrupt Fortran framing, exit code 0. Since no
        rank holds the complete mean field under any multi-rank topology,
        `export_and_record_mean_field_under_topology()` now refuses loudly for
        `number_of_pools() > 1 || group_size() > 1` and points at the working route (serial NSCF
        restoring the checkpoint), which can start from a pooled `q3_mpi` SCF now that one rank
        per pool writes the checkpoint; `tests/validation/test_mpi_mean_field_export.py` pins both the
        refusal and the single-rank `q3_mpi` positive case. A real multi-rank export needs a
        gather step (k-points across pools, G-slabs within a group — `allgather_within_group()`
        exists for the latter) feeding the record on the root rank.
    -   **`VSC` and `VKB` are not in the mean-field record** — `q3_utility convert` cannot
        produce them, because `mean-field.q3mf` stores neither the self-consistent potential
        nor the projectors; they are written only by the run itself. Storing V_sc(G) is
        cheap; the projectors would better be rebuilt from the datasets at conversion than
        stored, which needs the pseudopotentials and the crystal in the converter.
    -   **`VKB` is norm-conserving only** — ultrasoft and PAW make the problem generalized
        (H ψ = E S ψ), and DFT+U adds a projector term of its own; neither fits the file, so
        both are refused. `VXC` is spin-unpolarized only: the record carries the
        spin-averaged V_xc(G).
    -   **The mean-field record duplicates `wfc<N>.dat`** — `mean-field.q3mf`
        (`io/mean_field_record/`) stores its own copy of the wavefunction coefficients, so a run
        that requests an export writes them twice. It is written only when
        `mean_field_export` is set, precisely to keep that cost off every other run. Teaching
        `q3_utility convert` to read `wfc<N>.dat` and `charge-density.dat` directly would retire the
        duplication. The reader gap that blocked it is closed —
        `io/charge_density/.../checkpoint/qe_reader.hpp` and
        `io/wavefunction/.../checkpoint/qe_reader.hpp` exist and `q3_utility volumetric` is built on
        them — so what is left is sourcing a `MeanFieldRecord` from those files instead of from
        the duplicated copy, which also needs the symmetry, occupation and k-point metadata the
        record carries and those two files do not.
    -   **[S] No `kgrid` for the wedge of a grid the rotations do not preserve** — the header
        recovers `kgrid`/`kshift` from an explicit `kpoints_list` when the list, or its star under
        the crystal's rotations, fills a grid exactly; BerkeleyGW's `epsilon` refuses a zero
        `kgrid` ("kgrid out of bounds in wfn"). The symmetry-reduced wedge of a shifted FCC grid is not
        closed under the point group, so it still gets 0 0 0 and a warning. A `kgrid`/`kshift`
        override for the export, as `pw2bgw.x` has (`wfng_nk1..3`, `wfng_dk1..3`), would cover
        it (camsai/q3#21).
    -   **[S] `vxc.dat` covers every band** — an NSCF export computes `<psi_nk|Vxc|psi_nk>` for
        all bands in WFN, which for ZnO (200 bands × 32 k-points, 80 Ry) took ~11 min of a
        13.5-min NSCF step. `pw2bgw.x` computes only the `vxc_diag_nmin`–`vxc_diag_nmax` window
        sigma asks for; a band window for the export would do the same (camsai/q3#21).

### 3.2 Architecture & API

-   [ ] **[M]** **Extract `CalculationRunner` from `main.cpp`** — setup + SCF workflow into a library entry point (`q3::run(const InputData&)`). Now higher-value: there are **six** near-duplicate entry points (`main`, `main_mpi`, `main_gpu`, `main_gpu_cuda`, `main_gpu_hip`, `main_gpu_sycl`), each re-implementing the setup+SCF flow — a shared runner collapses that surface and stops the six from drifting. (`main_utility.cpp` is a seventh entry point but not part of this cluster: its subcommands run no SCF and share none of the setup flow.)
-   [ ] **[M]** **Define public headers** under `include/q3/` and install `q3_core` for Mat3ra embedding
-   [ ] **[M]** **Replace post-construction iterator wiring** — raw pointers and public mutable fields with constructor injection or a config struct
-   [ ] **[M]** **The Wannier capture kernels still take their pseudopotential type as a null
        pointer** — `PlanewaveWannierCapture` (`interpolator/method/wannier/planewave/init.hpp`)
        has `build()` and `build_ultrasoft()` share one body that receives the augmentation
        tables as a pointer, `nullptr` meaning norm-conserving, and branches on it twice;
        the entry points take nine and ten positional arguments. It is the shape AGENTS.md rules
        out (no null as a mode, an interface plus a factory per physical context) and it
        predates the restructuring of everything downstream of the record, which left it
        alone on purpose: the capture is the one part that touches wavefunctions, its
        numbers are pinned against `pw2wannier90.x`, and the fix is the `OverlapOperator`
        shape — a norm-conserving and an augmented metric behind one interface, with the
        grid, bands and execution policy held by the capture object — which wants its own
        change and its own parity run.
-   [ ] **[M]** **Layer boundary cleanup** — move the third-party ibrav helper out of `io/`; decouple layer 5 result types from `io/mixins/` via output adapters; decide where `layer_4_computation/flavors/` belongs — flavors orchestrate post-SCF properties and therefore `#include` layer_5 headers (forces, stress, isolated-system correction) from a layer_4 path, an upward include the C++ layering rule otherwise forbids (they are de-facto a layer above 5; either move the directory or bless flavors as a vertical slice like `io/`)

-   [ ] **[S]** **PocketFFT everywhere, then as the default** — `Q3_FFT_BACKEND=pocketfft`
        covers the serial FFT grid only. The MPI stick-distributed grid and the GPU builds'
        CPU fallbacks still plan with FFTW, so CMake refuses those combinations; give them
        PocketFFT paths, then decide whether to make `pocketfft` the default so released
        binaries and containers link no GPL library.
-   [ ] **[S]** **CUDA image: GPL FFTW + proprietary CUDA runtime under counsel review;
        fallback is a PocketFFT CPU path for GPU builds (§3.2)** — the `-cuda` image ships
        FFTW (GPL-2.0-or-later) beside NVIDIA's CUDA runtime libraries under NVIDIA's EULA.
        It is published with the same GPL compliance as the other images
        (`THIRD_PARTY_NOTICES.md`, the release's source bundle); if counsel rules the
        combination out, the item above is the way to drop FFTW from GPU builds.

### 3.3 Conventions & cleanup

-   [ ] **[S]** **`LocalOrbitalHamiltonian::maximum_hamiltonian_magnitude_beyond` has no caller**
        (`src/layer_3_representation/local_orbital/init.hpp`). The decay of H(R) is measured
        by `HamiltonianDecay::measured_on`, over shells rather than beyond a radius; delete
        the method, or give it the use it was written for.
-   [ ] **[S]** **Layer-3 refusals that name a manifest key** — `PseudoAtomicOrbitalSet`
        words its errors with `control.interpolation.trial_shells`, which is the vocabulary
        of the layer that parsed the manifest, not of the layer refusing. The key is what
        lets a user act on the message, so the fix is a typed error the input layer words,
        not a vaguer sentence.
-   [ ] **[S]** **Clear the pre-commit backlog and drop the CI `SKIP` list** — the new
        `lint` CI job (`.github/workflows/ci.yml`) runs `pre-commit run --all-files`, but
        skips the hooks that do not pass on the tree today. Measured 2026-07-29; each
        needs its own cleanup commit, kept separate because the diffs are large,
        mechanical, and touch files unrelated to whatever change ships them:
    -   `prettier` — **9 markdown files** (`FUNCTIONALITY.md`, `PROMPTS.md`, `TODOs.md`,
        `LICENSE.md`, `TROUBLESHOOTING.md`, `AGENTS-generic.md`, `schemas/README.md`,
        `docs/differentiability.md`, `docs/io_reference.md`). Pure reflow; the risk is
        that it rewrites large tracked documents in a way that buries real edits in
        review, so it wants to be its own PR.
    -   ~~`ruff-format` — 3 Python files.~~ **Cleared 2026-07-29** in the Python
        import-layering change; the last 2 files were formatted and `ruff-format`
        dropped from the CI `SKIP` list. The pinned hook version (ruff 0.11.13) and
        a current local ruff (0.15.8) agreed on the output, so unlike `clang-format`
        this hook has no version-skew problem to settle first.
    -   ~~`clang-format` — **4 C++ files** (`src/io/input/parser/yaml.hpp`,
        `.../fft/gpu/hip.cpp`, `tests/unit/io/input/parser/test_yaml.cpp`,
        `tests/unit/.../test_hubbard_ortho_atomic.cpp`).~~ **Cleared 2026-07-29**:
        decided the version question in favor of the pinned pre-commit hook
        (clang-format 19.1.7) as the single source of truth; system clang-format
        is advisory only (documented in AGENTS-generic.md § Linter). Running the
        pinned hook across the whole tree made **zero** changes — the 4 files
        above were already reformatted as a side effect of the same-day
        multiline-comment-linter-rule change (commit `f81283a`), which also
        resolved `yaml.hpp`'s version-skew drift: both clang-format 19.1.7 and
        the system's 18.1.3 now agree on all four files. `clang-format` dropped
        from the CI `SKIP` list.
    -   ~~`clang-tidy` — needs a compile database (`-p build`)...~~ **Cleared
        2026-07-29**: wired into `build-and-test` right after the Build step
        (`.github/workflows/ci.yml`), gating hard — **zero findings** across all 104
        files this configuration compiles (clang-tidy 18.1.3). A trustworthy
        measurement needed two real bugs fixed first, not just flipping the switch:
        (1) `.clang-tidy`'s `HeaderFilterRegex` read `.*/q3-minimal/src/.*` — a stale
        pre-rename checkout-directory name that never matched any real path, so
        every project-header diagnostic (as opposed to the `.cpp` file passed on the
        command line) was silently suppressed alongside the third-party ones it was
        meant to filter; fixed to `.*/q3/src/.*` and verified by planting a
        violation in a header and confirming it was then caught. (2) the hook's file
        list (every non-`src/bindings/` `.cpp` repo-wide) included files this
        configuration never compiles — GPU vendor backends and
        `main_gpu*.cpp`/`main_mpi.cpp`, gated behind `Q3_WITH_CUDA`/`_HIP`/`_SYCL`/
        `_MPI`, all OFF by default — which have no compile-database entry, so
        clang-tidy's fallback guess at compiler flags reported unrelated "compiler
        error" noise (e.g. `'mpi.h' file not found`) instead of a real finding;
        `scripts/repo/run_clang_tidy.py` (the hook's new entry point) now drives the file
        list from `build/compile_commands.json` itself, so coverage tracks whatever
        this configuration actually built rather than a static, driftable list.
-   [ ] **[?]** **Decide the `init.hpp` convention** — this item called for migrating
        `init.hpp` → `snake_case(class_name).hpp`, but the code has since moved the
        _other_ way (a commit renamed `core.hpp` → `init.hpp` as a "Pythonic init
        convention", and there are now ~79 `init.hpp` files). Either update AGENTS.md to
        bless the `init.hpp`-as-package-entry convention, or commit to the rename — but
        stop drifting between the two.
-   [ ] **[L]** **Retire dynamic type queries** — replace `dynamic_cast`, ultrasoft/PAW
        downcasts, `as_ultrasoft()`/`as_paw()` and `has_*()` flag branching with
        per-physics classes (AGENTS.md § Calculator Design). The
        `cpp-type-queries` ratchet holds the count at its baseline; each refactor
        lowers it with `scripts/repo/check_cpp_type_queries.py --write-baseline`.
-   [ ] **[M]** **Document the undocumented public API** — 1022 public header
        declarations and 13 `q3` Python definitions lack a doc comment; the
        `docstring-coverage` ratchet stops new ones and is tightened as they are
        documented.
-   [ ] **[M]** **Split multi-type headers** — config struct + class in the same file violates one-class-per-file rule
-   [ ] **[M]** **Extract shared kernels** — radial Bessel transform (repeated across projector/charge_core/potential_local) and Ewald G-space utilities shared by force/stress
-   [ ] **[M]** **Move hot compute kernels from headers to `.cpp` files** — Currently Q3 is
        almost entirely header-only. Moving performance-critical functions (Davidson
        inner loops, XC kernel, FFT scatter/gather) to `.cpp` translation units
        would give the compiler a single definition to auto-vectorize and enables
        LTO to inline only where beneficial. QE's Fortran loops auto-vectorize
        more aggressively than C++ template code in headers.
-   [ ] **[L]** **LAPACK/LibXC facades** — consolidate scattered `extern "C"` declarations and stop leaking LibXC headers into output writers

### 3.4 Error handling & logging

-   [ ] **[M]** **Standardize throw vs assert** — unimplemented XC virtuals use `assert` (silent in `-DNDEBUG`); likewise `PseudopotentialSet::as_ultrasoft()`/`as_paw()` (`dynamic_cast` + `assert`, a null dereference in Release on a mixed NC/US cell), the GPU MPI guard and the Fermi-level null check
-   [ ] **[M]** **Align logger defaults** with AGENTS-generic; add `--log-file` CLI flag

### 3.5 Documentation & tooling

-   [x] **[S]** **Input/output reference** — added [docs/io_reference.md](docs/io_reference.md):
        every YAML manifest field (root/model/method/scf/mixing/smearing/kgrid/hubbard/
        band_structure/control/band_path), the QE-namelist format with a full per-key
        YAML-equivalence table (including the vocabulary gaps already documented in the
        namelist-parity item above), every CLI flag per binary (`q3`/`q3_mpi`/GPU) including
        `--pools`/`--precision`/`--log-level` and which binaries actually act on which flags,
        and every output artifact (`<prefix>_results.json` schema incl. the new `units` key,
        the `.save/` binary checkpoint set with exact byte layouts, the mid-SCF restart
        checkpoint, and the Python-workflow-level outputs for band structure/DOS/PDOS/
        phonons/relaxation). Researched via three parallel deep-read agents (YAML schema,
        CLI+namelist, output formats) cross-checked against `examples/**/*.yml` and real
        `.save`/results.json files, not just the parser source. Surfaced four previously
        undocumented real issues in the process, each filed as its own TODO item above/below
        rather than silently fixed as a documentation side-effect: the `--pools`
        default-topology doc/behavior mismatch, a byte-offset bug in the wavefunction-checkpoint
        test reader, the orphaned (write-only) mid-SCF checkpoint, and the CLI-vs-bindings
        `results.json` naming mismatch.
-   [ ] **[M]** **Reconcile docs/ontology.md with the implementation** — the DataHolder / DataCalculator / Result pattern it describes (`ForcesDataHolder`, `AtomicForcesResult`, `LoggableMixin`, …) diverged from the code (results flow through `generated::Results` + `contributions.hpp` structs + generated ESSE types). Either update the ontology (and [ontology.drawio](docs/ontology.drawio)) to the implemented design, or decide the pattern is still the target and file the refactor. Concrete class trees (XC, diagonalizers, mixers, smearing) were re-synced 2026-07-08.
