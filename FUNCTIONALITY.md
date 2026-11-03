# Functionality

Present-state capability inventory for Q3, grouped into three user-facing areas:
**Science** (what you can model and compute), **Performance** (how it runs), and
**Engineering** (how you work with it). This file describes **what exists now**;
planned work lives in [TODOs.md](TODOs.md) and codebase metrics in
[STATISTICS.md](STATISTICS.md).

**Status legend**

| Symbol | Meaning                                                                                                     |
| :----: | ----------------------------------------------------------------------------------------------------------- |
|   ✅   | Validated — cross-checked against an established plane-wave DFT code (LibXC-enabled) and confirmed to agree |
|   🟡   | Implemented — present in code, not yet validated against a reference                                        |
|   —    | Not applicable for this pseudopotential type                                                                |

A feature graduates from [TODOs.md](TODOs.md) into the matching area here once it ships.

---

## 1. Science (Materials Science)

What Q3 can model and the physical quantities it produces. NC, US and PAW
ground-state SCF are implemented and validated, including spin polarization,
forces and stress. PAW accepts UPF v2 datasets; further dataset and input formats
plug in as build-time add-ons (§3.1). Parallel
PAW backend validation remains planned (see [TODOs.md](TODOs.md) §1.1).

### 1.1 Crystal Structures & Symmetry

| Feature                                                                                     | Status |
| ------------------------------------------------------------------------------------------- | :----: |
| All 14 Bravais lattice types (ibrav)                                                        |   ✅   |
| Free / explicit lattice vectors (ibrav = 0)                                                 |   ✅   |
| General-lattice input: YAML `type` + a/b/c/α/β/γ or `lattice.vectors`; QE `CELL_PARAMETERS` |   ✅   |
| Symmetry detection (spglib)                                                                 |   ✅   |
| Monkhorst-Pack k-grid + offsets                                                             |   ✅   |
| IBZ reduction (spglib)                                                                      |   ✅   |
| Density symmetrization over the full space group (supercell translations included)          |   ✅   |
| Gamma-only                                                                                  |   ✅   |

Generated Monkhorst-Pack dimensions and 0/1 shifts survive both symmetry
reduction implementations and k-point-pool distribution. Explicit k-point
lists retain zero grid metadata, so downstream consumers can distinguish the
two forms. A real shifted `4x4x4 -> 10` silicon reduction gates the BerkeleyGW
WFN header's `kgrid` and `kshift` fields. `control.disable_symmetry` switches
all of it off — no mesh reduction and no symmetrization of the density, forces
or stress, QE's `nosym` + `noinv` — so a shifted mesh then gives the raw-mesh
energy rather than the symmetrized one.

### 1.2 Pseudopotentials

| Feature                         | NC  | US  | PAW |  Status   |
| ------------------------------- | :-: | :-: | :-: | :-------: |
| Local potential V_loc(G)        | ✅  | ✅  | ✅  | Validated |
| KB projectors β(G)              | ✅  | ✅  | ✅  | Validated |
| (-i)^l phase convention         | ✅  | ✅  | ✅  | Validated |
| D^ion matrix                    | ✅  | ✅  | ✅  | Validated |
| S-matrix (overlap)              |  —  | ✅  | ✅  | Validated |
| Q-function augmentation         |  —  | ✅  | ✅  | Validated |
| NLCC core charge                | ✅  | ✅  | ✅  | Validated |
| Atomic charge ρ_at(G)           | ✅  | ✅  | ✅  | Validated |
| All-electron one-center package |  —  |  —  | ✅  | Validated |

**The columns above compose within one cell.** Species carrying different
pseudopotential families can be combined in a single calculation — an
ultrasoft nickel alongside a norm-conserving carbon, say — which matters
whenever one element needs a particular dataset and the others do not. The
augmentation machinery is built whenever _any_ species asks for it, and each
species is then asked individually: one without augmentation charges
contributes q_ij = 0 to the overlap operator and no augmentation charge to the
density, while its projectors still enter the nonlocal potential normally.

`PawPseudopotential` holds the all-electron and pseudo partial waves, core
densities, one-center potentials, occupations, and compensation multipoles
alongside the ultrasoft augmentation apparatus it extends. UPF v2 supplies
radial augmentation functions directly. A dataset format that supplies other
data plugs in as a build-time add-on, together with the one-center convention
its datasets use (§3.1).

Serial PBE validation: Q3+UPF agrees with the
LibXC-enabled QE 7.5 build within 1 mRy/atom for Si, Cu, and diamond. UPF v1
PAW remains rejected at read time.

### 1.3 Exchange–Correlation & Spin

All functionals use LibXC. Named functionals (`PZ`, `LDA`, `PBE`, `PBEsol`,
`BLYP`, `revPBE`, `RPBE`) map to LibXC functional IDs in the factory; arbitrary
LDA/GGA combinations may be selected via `libxc:X:C` or QE-style `XC-...`
identifiers. Names are matched case-insensitively, and each half of a `libxc:`
pair may be given as a LibXC id or as the name LibXC knows it by, so
`libxc:117:130` and `libxc:gga_x_rpbe:gga_c_pbe` select the same functional.
Every exchange and correlation component is classified during construction.
Meta-GGA and hybrid families are refused before SCF with the identifier, detected
family, and missing Q3 capability in the error; this also prevents BerkeleyGW
`vxc.dat` from representing a semilocal-only fragment under a nonlocal functional name.

An input that names no functional runs the one its pseudopotentials were
generated with, as pw.x does without `input_dft`, whether the deck is Q3 YAML or
QE-format. Datasets that disagree, or that state a functional Q3 does not map,
are refused by name rather than approximated — see
[io_reference.md §1.2](docs/io_reference.md#12-model).

| Functional          | Type    | Unpolarized | LSDA | Status |
| ------------------- | ------- | :---------: | :--: | :----: |
| PZ                  | LDA     |     ✅      |  🟡  |   ✅   |
| PBE                 | GGA     |     ✅      |  ✅  |   ✅   |
| Any via `libxc:X:C` | LDA/GGA |     ✅      |  🟡  |   🟡   |

| Spin mode   | nspin | NC  | US  | PAW | Status |
| ----------- | :---: | :-: | :-: | :-: | :----: |
| Unpolarized |   1   | ✅  | ✅  | ✅  |   ✅   |
| LSDA        |   2   | ✅  | ✅  | ✅  |   ✅   |

LSDA supports per-species `starting_magnetization` through both parsers
(AFM-capable, e.g. `Cr1`/`Cr2` sublattice labels) and per-atom starting
moments — YAML `method.scf.atomic_starting_magnetization` (one value per
atom) — so a same-species antiferromagnet needs no relabeling; such sites are also
kept symmetry-inequivalent, and the per-atom AFM Cr cell matches the
two-species formulation (`tests/validation/test_per_atom_starting_magnetization.py`).
QE input stays per species, as in QE itself. LSDA reproduces the
unpolarized energy at zero moment to ~1e-9 Ry
(`tests/validation/test_lsda_consistency.py`). Cross-code validated (NC,
PBE) against QE on three Window 1 roadmap systems — Fe BCC-FM
(ΔE = 0.11 mRy, Δmoment = 0.005 μB), Ni FCC-FM (ΔE = 0.013 mRy,
Δmoment = 0.005 μB), Cr BCC-AFM (ΔE = 0.25 mRy, genuine antiferromagnetic
order confirmed via both codes' absolute-vs-total magnetization) — all well
inside the roadmap's ΔE < 1 mRy / Δmoment < 0.02 μB gates
(`tests/validation/test_qe_energy_comparison.py::test_qe_lsda_energy_and_moment_comparison`,
reference inputs/outputs in `examples/total_energy--metal--magnetic--{iron,nickel,chromium}/reference/`).

**Ultrasoft and PAW spin polarization** was fixed and validated 2026-08-08.
The nonlocal D matrix is now built per spin channel and the compensation
charge per channel from that channel's own occupancies; previously both were
spin-averaged, which is exact for a non-magnetic cell and collapses the moment
in a magnetic one. Ferromagnetic iron with a GBRV ultrasoft
dataset — a path that had never been measured — went from 0.0144 to **2.2482 μB**
and from 473 to **0.06 meV/atom** against QE. Non-magnetic cells at
`nspin = 2` reproduce their `nspin = 1` energies bit-for-bit
(`tests/validation/test_spin_polarized_nonlocal.py`).

The **augmentation force and stress became spin-resolved 2026-08-09**,
completing that work. They had kept integrating the spin-_averaged_ potential
against the spin-_summed_ occupancies while their nonlocal partners carried a
spin index, dropping ∫(V↑−V↓)(ρ↑−ρ↓) dQ/dR / 2 — zero for a non-magnetic cell,
which is why unpolarized tests never saw it. The per-spin quantities already
existed in the iterator; only the SCF state handed to layer 5 lacked them.
Gated by a finite-difference check against Q3's own energy on ferromagnetic
bcc iron (ultrasoft, conventional two-atom cell, one atom displaced), with a
companion assertion that the cell carries a moment so the gate cannot pass
vacuously.

### 1.4 Ground-State SCF

| Feature                      | NC  | US  | PAW |   Status    | Notes                                                                                                |
| ---------------------------- | :-: | :-: | :-: | :---------: | ---------------------------------------------------------------------------------------------------- |
| Total energy                 | ✅  | ✅  | ✅  |  Validated  | PAW validated against QE                                                                             |
| Atomic ρ superposition       | ✅  | ✅  | ✅  |  Validated  | Initial density                                                                                      |
| Hartree potential            | ✅  | ✅  | ✅  |  Validated  |                                                                                                      |
| Ewald energy (ion-ion)       | ✅  | ✅  | ✅  |  Validated  |                                                                                                      |
| Dispersion (DFT-D3 BJ)       | ✅  | 🟡  | 🟡  |  Validated  | Energy + forces + stress (NC validated vs QE)                                                        |
| Augmentation charge          |  —  | ✅  | ✅  |  Validated  | `add_augmentation_charges`                                                                           |
| Effective D-matrix           |  —  | ✅  | ✅  |  Validated  | D_ij = D^ion + ∫ V_eff Q_ij dG; PAW includes one-center correction                                   |
| Charged cells (net Q)        | ✅  | 🟡  | 🟡  |  Validated  | `total_charge` input, QE `tot_charge` convention                                                     |
| Makov-Payne correction       | ✅  |  —  |  —  |  Validated  | Cubic lattices only; `isolated_system_correction: makov-payne`                                       |
| Martyna-Tuckerman correction | ✅  |  —  |  —  |  Validated  | Any Bravais lattice; post-SCF only (see note below); `isolated_system_correction: martyna-tuckerman` |
| DFT+U (Dudarev)              | ✅  | ✅  | 🟡  | Implemented | Atomic and ortho-atomic energy/force/stress for all three dataset families (see note)                |

DFT-D3(BJ) additionally validated on AB (Bernal) graphite (roadmap Window 2
pinned system): the isolated dispersion term matches QE to <2e-9 Ry at every
scanned interlayer spacing (`D3BJDispersionCalculator.Graphite*` in
`test_d3bj.cpp`, sub-millisecond, runs on every PR). The equilibrium
interlayer spacing and interlayer binding energy were extracted from a
c-scan (fixed a=2.464 Å, quadratic fit near the discrete minimum) in both
Q3 and QE: equilibrium spacing Δ = 0.0001 Å (roadmap gate: ≤ 0.01 Å) and
binding energy Δ = 0.54 meV/atom (roadmap gate: ≤ 1 meV/atom)
(`tests/validation/test_qe_graphite_d3_dispersion.py`, tier-2/slow, ~2 min).

Forces and stress (`D3BJDispersionCalculator::compute_gradients()`) cover
both the direct pairwise term and the coordination-number chain rule
(C6 depends on each atom's full coordination environment, not just the
pair in question). Validated by finite-difference unit tests against the
already-validated energy (`test_d3bj.cpp`, forces to 1e-7 Ry/bohr, stress
to 1e-8 Ry/bohr³) and, independently, against QE forces on a distorted
graphite geometry via a two-run differencing trick (QE's `dft-d3` module
requires plain `input_dft='pbe'`, so the two runs — with and without
`vdw_corr` — are differenced to cancel every non-dispersion force term):
agreement to 1e-8 Ry/bohr
(`test_dispersion_forces_match_qe_via_differencing`, tier-2/slow).

`total_charge` (`InputData::total_charge`, YAML `method.scf.total_charge`,
QE `tot_charge`): positive = electrons removed (cationic), negative =
electrons added (anionic), matching QE's sign convention exactly. Needs no
compensating-background term — the Ewald ion-ion sum and the Hartree G=0
skip are both keyed only to the fixed ionic charge (verified against QE,
whose Ewald energy has no `tot_charge` dependence at all), so only the
electron count changes. Validated on a charged FCC Si cell (`tot_charge =
1.0`, Gaussian smearing — required, since an odd electron count has no
fixed-occupation representation): Q3 vs QE agreement 0.0002 mRy, deep
inside the 0.05 mRy tolerance
(`tests/validation/test_qe_energy_comparison.py::test_qe_charged_cell_energy_comparison`).
The isolated-system correction for comparing charged-cell energies across
cell sizes (`control.isolated_system_correction: makov-payne`,
`MakovPayneCorrectionCalculator`) reproduces QE's exact convention (both
the 1st-order Madelung/monopole and 2nd-order dipole+quadrupole terms —
the 2nd order is 3.3x _larger_ than the 1st for this system, not a small
refinement, so an implementation without it would be badly misleading).
Validated on the same charged FCC Si cell above: 1st order matches QE to
all 8 printed significant figures (0.27948401 Ry); 2nd order agrees to
~0.08% (-0.91521864 Ry vs. QE's -0.91595649 Ry), consistent with the
expected gauge-dependence of Makov-Payne's un-wrapped ionic moments
(Q3 and the QE reference run need not share the exact same atomic-position
gauge) rather than a formula error. Cubic lattices only (SC/FCC/BCC); see
TODOs.md §1.3 for the exact algorithm and full derivation notes.

The Martyna-Tuckerman isolated-system correction
(`control.isolated_system_correction: martyna-tuckerman`,
`MartynaTuckermanCorrectionCalculator`) is the second scheme through the
same factory: an exact reciprocal-space kernel-difference sum (every
multipole order at once, not a truncated series), valid for any Bravais
lattice (no cubic-only gate, unlike Makov-Payne). **This is the post-SCF
(non-self-consistent) evaluation only** — evaluated once on the density
already converged under the ordinary periodic Coulomb kernel, which
differs from QE's self-consistent `assume_isolated='m-t'` (density not
relaxed under the MT-corrected potential); see the class docstring for the
full scope statement. Validated by two unit-test limits derived from first
principles and confirmed both analytically and in an independent numpy
prototype before being wired into C++: (1) a neutral, concentrically
symmetric synthetic charge distribution (point ion + isotropic Gaussian
electron cloud at the same center) gives a correction consistent with zero
to ~1e-9 Ry, matching the classical shell-theorem expectation; (2) for a
net-charged compact synthetic density, the correction's ratio to
Makov-Payne's first-order (Madelung monopole) term on the same state
trends monotonically toward 1 as the cell grows (1.07 → 1.03 → 1.01 across
three increasing cell sizes), confirming both the sign and the expected
large-cell limit against an independently-validated scheme. End-to-end
regression anchors on the same charged FCC Si cell used above are
self-measured (no QE `assume_isolated='m-t'` cross-check yet — QE's own
version is self-consistent, so a literal Q3-vs-QE total-energy comparison
would conflate the post-SCF-vs-self-consistent scope difference with any
formula difference); see TODOs.md §1.3 for the exact algorithm, its
literature grounding, and what a future self-consistent version would need.

DFT+U (`method.scf.hubbard`) is Dudarev's rotationally invariant formulation,
applied self-consistently as a second nonlocal-projector term. A shell may be
named by its dataset label (`3D`) or by a bare orbital letter (`d`), which
selects by angular momentum and occupancy. Q3 logs `Tr[n]` per manifold and
spin so a missing or misidentified shell is visible.

**Energy and potential.** Atomic projectors work for norm-conserving,
ultrasoft, and PAW datasets; the latter two use the all-electron inner product
`⟨φ|S|ψ⟩`. Reading QE's own committed ultrasoft Hubbard input, Q3 matches the
total energy to **4.5e-7 Ry** and `E_U` to **3e-8 Ry**. The gate covers the
`Tr[V_U n]` double-counting correction, spatial and time-reversal averaging of
the occupation matrix, and irreducible-wedge versus full-zone consistency.

**Forces.** Atomic-projector forces cover all three dataset families. Their
ultrasoft/PAW path includes both `dφ/dR` and `dS/dR`, so a Hubbard force can
appear on an atom with no Hubbard manifold. Silicon ultrasoft agrees with the
derivative of Q3's own energy to 0.044%; zincblende SiC PAW agrees to 0.013%,
and removing the carbon-side `dS/dR` term makes that discriminating
contribution exactly zero. Gamma-only energy agrees with Q3's explicit
full-sphere result to 8e-9 Ry and its force agrees with a central difference
to about 3e-5 relative.

`method.scf.hubbard_projectors: "atomic"|"ortho-atomic"` selects QE's two
projector conventions. Ortho-atomic projectors use one joint Löwdin transform
over every orbital in the cell. Norm-conserving datasets use `⟨φ|φ⟩`; ultrasoft
and PAW use the generalized `⟨φ|S|φ⟩` metric from the same production overlap
operator as Davidson. Their derivatives include `d(O^-1/2)/dR`,
`d(O^-1/2)/dε`, and, where present, `dS/dR` and `dS/dε`.

The ultrasoft silicon ortho-atomic total and Hubbard energies match LibXC QE to
**5.8e-7 Ry** and **1.1e-8 Ry**, with occupation traces within **5e-5**. On PAW
SiC, the complete force agrees with Q3's own energy derivative within
**2.4e-6 Ry/bohr**, and all six stress components agree within **0.05 kbar**.
The corresponding norm-conserving gates agree to **4.9e-5** relative for force
and **0.05 kbar** for stress; removing only the transform derivative misses by
**36.3 kbar**. Gamma half/full parity is gated for energy, force, total stress,
augmentation stress, and Hubbard stress.

**Stress.** Atomic-projector Hubbard stress is implemented for every dataset
family. Its isolated term matches QE's printed decomposition on
norm-conserving and ultrasoft silicon and ultrasoft copper, including
Gamma-only and both p and d manifolds. PAW exercises the same ultrasoft branch
and produces a non-zero term, but lacks an independent QE gate because the
committed, currently supported PAW datasets omit the orbital labels QE requires. Atomic and
ortho-atomic PAW stress are therefore gated against the full tensor of Q3's own
energy derivatives; the ortho-atomic test covers two strain steps before
committing its tolerance.

The remaining limits fail visibly. The GPU applicator and direct diagonalizer
refuse DFT+U until they have parity coverage. Ultrasoft ortho-atomic energy and
the insulating norm-conserving topology gates retain the repository's
`1e-8 Ry` standard. The metallic nickel norm-conserving ortho-atomic gate uses
a `5e-8 Ry` allowance for MPI reduction-order noise at a `1e-16` density
threshold and enables deterministic FFT planning so `FFTW_MEASURE` plan
selection is not folded into the topology comparison.

#### DFT+U support matrix

| Execution path               | Atomic projector | Ortho-atomic projector | Contract                                                                          |
| ---------------------------- | :--------------: | :--------------------: | --------------------------------------------------------------------------------- |
| Serial CPU + Davidson        |   NC, US, PAW    |      NC, US, PAW       | Energy, forces, and stress enabled                                                |
| MPI k-point pools + Davidson |   NC, US, PAW    |      NC, US, PAW       | Energy, forces, and stress enabled                                                |
| MPI band groups + Davidson   |   NC, US, PAW    |      NC, US, PAW       | Energy and forces enabled; stress refused before property evaluation              |
| GPU applicator               |     Refused      |        Refused         | Rejected before GPU backend initialization                                        |
| Direct diagonalizer          |     Refused      |        Refused         | Rejected before SCF because this path bypasses the Hubbard Hamiltonian applicator |

The mandatory PAW gates use redistributable committed UPF data, and no committed
input embeds a host-specific path.
Older PAW UPFs that use `l_max_aug` are parsed, but a PAW UPF declaring a
`core_energy` reference constant is refused during loading until Q3 applies and
validates that absolute-energy convention. This prevents the known PSLibrary
subset from entering SCF with a silently wrong one-center energy. A dataset with f
projector channels is refused for a related reason: the augmentation expansion
would need spherical harmonics to degree 6 and Q3 has closed forms to 4, so the
Gaunt table it would be built from cannot be formed exactly — see
[TODOs.md](TODOs.md) §1.2.

| SCF machinery   | Options                                                                                    | Status |
| --------------- | ------------------------------------------------------------------------------------------ | :----: |
| Diagonalization | Full LAPACK (`direct`, default), Davidson (`davidson`)                                     |   ✅   |
| Density mixing  | Broyden (reciprocal, default), Anderson (real-space)                                       |   ✅   |
| Occupations     | Fixed, Gaussian, Methfessel-Paxton (order N ≥ 1), Marzari-Vanderbilt, Fermi-Dirac smearing |   ✅   |

Both diagonalizers handle the generalized eigenvalue problem for US (S-matrix via
`OverlapOperator`). Mixing parameters: `mixing_beta` (default 0.7), `mixing_ndim`
(default 8); for US, mixing acts on total ρ (smooth + augmentation). Broyden's G-space
metric dot products and gamma-correction updates are BLAS-accelerated (metric weights
precomputed once per history entry rather than per pairwise access; ZAXPY for the
correction/mixing steps).

Methfessel-Paxton (order 1), Marzari-Vanderbilt ("cold smearing"), and
Fermi-Dirac smearing cross-validated (NC, PBE, Cu FCC, matched `degauss`)
against QE: ΔE = 0.00018, 0.00018, and 0.00019 mRy respectively, all inside
the roadmap's ΔE < 1e-6 Ry gate
(`tests/validation/test_qe_energy_comparison.py::test_qe_energy_comparison`,
`Cu-NC-MethfesselPaxton`/`Cu-NC-MarzariVanderbilt`/`Cu-NC-FermiDirac`).

Methfessel-Paxton of any order N ≥ 1 (Phys. Rev. B 40, 3616) is selected by
`method.scf.smearing.order` in YAML; order 1 is the default and is
bit-identical to the original closed form.
Quantum ESPRESSO's pw.x exposes order 1 only, so higher orders are validated
against the expansion's defining identities (vanishing moments of δ_N up to
x^{2N}, dS_N/dx = −x δ_N) in
`tests/unit/layer_4_computation/iterator/internal/smearing/test_methfessel_paxton.cpp`
and end-to-end on Cu in `tests/validation/test_methfessel_paxton_order.py`.

### 1.5 Properties

**Forces (Hellmann-Feynman)**

| Component               | NC  | US  | PAW |  Status   |
| ----------------------- | :-: | :-: | :-: | :-------: |
| Local potential         | ✅  | ✅  | ✅  | Validated |
| Ewald (ion-ion)         | ✅  | ✅  | ✅  | Validated |
| Nonlocal (KB)           | ✅  | ✅  | ✅  | Validated |
| NLCC (core charge)      | ✅  | ✅  | ✅  | Validated |
| Augmentation (dQ_ij/dτ) |  —  | ✅  | ✅  | Validated |

**Stress tensor (Nielsen-Martin)**

| Component             | NC  | US  | PAW |  Status   | Notes                                |
| --------------------- | :-: | :-: | :-: | :-------: | ------------------------------------ |
| Kinetic               | ✅  | ✅  | ✅  | Validated |                                      |
| Hartree               | ✅  | ✅  | ✅  | Validated |                                      |
| Exchange-correlation  | ✅  | ✅  | ✅  | Validated | Includes NLCC core correction        |
| Ewald                 | ✅  | ✅  | ✅  | Validated |                                      |
| Local pseudopotential | ✅  | ✅  | ✅  | Validated | Finite-difference dV_loc/dG          |
| Nonlocal (KB)         | ✅  | ✅  | ✅  | Validated | Analytical + finite-difference dβ/dq |
| Core correction       | ✅  | ✅  | ✅  | Validated |                                      |
| Augmentation          |  —  | ✅  | ✅  | Validated |                                      |

**Low-symmetry and Gamma stress** gained a dedicated no-U baseline on
2026-08-11. On an inversion-only triclinic cell, the symmetry-reduced and
explicit full-zone tensors now agree to `1e-7 kbar`; every kinetic, Hartree,
XC, Ewald, local, nonlocal, and core-correction component agrees with the
LibXC-enabled QE reference to `0.03 kbar`. All six tensor components agree
with central derivatives of Q3's own total energy to `0.006 kbar`, and the
Ewald tensor agrees with the Ewald-energy derivative to `0.001 kbar`.

The audit fixed two independent defects. Ewald reciprocal-space shear was
written into the lower triangle while its real-space part used the upper
triangle, so final symmetrization discarded the reciprocal shear. At a
single Gamma point, the stress calculators did not consistently reconstruct
the omitted `-G` half sphere while keeping `G=0` single-counted; the nonlocal
term additionally needed full-sphere projector overlaps before contraction.
Gamma half/full-sphere energy, total stress, and every component now agree,
with Ewald energy and stress held to numerical roundoff by
`tests/validation/test_stress_baseline.py`. Full contribution tensors are
serialized under `stress_contributions` in `results.json`.

**PAW forces and stress** were completed 2026-08-08. Two force terms had been missing: the
augmentation dQ_ij/dR contribution was assembled in the wrong index space
(radial beta pairs against lm-expanded channels), and the generalized-eigenvalue
term — ultrasoft and PAW solve H|ψ⟩ = εS|ψ⟩, so the force contracts
(D_ij − ε_nk q_ij) — was absent from the force while present in the stress.
The stress separately had a translation-invariance bug in its core-correction
term that made the pressure depend on the choice of basis origin.

Force and stress terms are gated against the numerical derivative of Q3's
_own_ total energy, which is what localizes a defect — a force disagreeing
with an external code could be either the force or the energy, while one
disagreeing with its own energy can only be the force. On the committed
silicon PAW dataset, the pressure is gated against its volume derivative on a
fixed basis, the stress against a shift of the origin, and the force at a
single Γ point (`tests/validation/test_paw_pressure_and_gamma_forces.py`); the
PAW force away from Γ is gated by the SiC DFT+U case
(`tests/validation/test_paw_forces_and_stress.py`).

**Gamma-only forces** were fixed in the same pass and are covered by their own
gate, which reads 0.99991 of Q3's own dE/dR on the committed silicon PAW
dataset (`test_gamma_only_forces_are_the_gradient_of_q3s_own_energy` in
`tests/validation/test_paw_pressure_and_gamma_forces.py`). At a single
unshifted Γ point Q3 keeps only the positive half of G-space, and the local,
NLCC, Ewald and augmentation force sums each lacked the resulting factor of
two, while the nonlocal term double counted G = 0.
Silicon's Γ force was 0.637 of Q3's own dE/dR with a norm-conserving dataset
and is now 1.00004; energies were correct throughout.

**Structural Relaxation & Band Structure**

| Feature                               | Status | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| :------------------------------------ | :----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Geometry Relaxation                   |   ✅   | High-level workflow in `q3.workflows.relax` (uses ASE optimizers); runs a Q3 YAML input or a pw.x `relax` deck as is (any `ibrav`, `if_pos` honoured) on `q3`, `q3_mpi` or a GPU binary, with an optional `execution_policy`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Variable-Cell Relaxation              |   ✅   | `q3.workflows.vc_relax`; ASE `FrechetCellFilter` + BFGS, QE-validated; a pw.x `vc-relax` deck runs as is, on any binary; `use_symmetry` (`FixSymmetry`) supported, matching ionic-only relaxation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Band Structure                        |   ✅   | High-level workflow in `q3.workflows.band_structure` (crystal path)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Density of States                     |   ✅   | High-level workflow in `q3.workflows.dos`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Projected DOS (PDOS) + Löwdin charges |   ✅   | `q3.workflows.pdos`; ortho-atomic (Löwdin) `\|⟨φ'_i\|S\|ψ_nk⟩\|²` projections, taken in the S metric for ultrasoft/PAW                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Γ-Point Phonons                       |   🟡   | `q3.workflows.phonons`; finite-difference, both blocking Q3 bugs fixed (below)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Phonon dispersions + thermodynamics   |   🟡   | `q3.workflows.phonons`'s Phonopy path; displaced supercells → band paths, free energy / entropy / C_v (below)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Wannier interpolation                 |   🟡   | `q3.workflows.interpolation`, `q3.workflows.wannier` + `q3_utility interpolate`; isolated manifolds, band selection, projectability disentanglement for entangled windows, maximally localized Wannier functions (Marzari-Vanderbilt spread minimisation with guiding centres), and the protocol around them: the mesh from a 0.2 Å⁻¹ spacing, the trial orbitals from the datasets' own shells less an augmented dataset's repeated empty channels, a quality report (per-function spreads and their variance, centre–atom and centre–bond distances, H(R) decay), the frozen window resolved from the bands with the singular-gauge exclusion reported, the band distance η against a reference calculation, and a retry ladder that accepts, retries or rejects unattended; a campaign driver runs a fixed roster and reports one table. Still hand-set: leaving semicore shells out, and SCDM ([TODOs.md](TODOs.md) §1.3) |
| Equation of state (Birch–Murnaghan)   |   ✅   | `q3.workflows.equation_of_state`; volume scan through `Q3ASECalculator` → ASE's third-order fit (`q3.properties.equation_of_state`) → V0, B0, B0'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Elastic constants                     |   🟡   | `q3.workflows.elastic`; cubic C11/C12/C44 as stress–strain slopes over ± strains (`q3.properties.elastic_constants`), clamped-ion C44 ([TODOs.md](TODOs.md) §4)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Target-pressure relaxation            |   ✅   | `run_vc_relax_workflow(target_pressure_gpa=…)`; minimises E + PV, the QE `&CELL press` analog                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

Relaxation, band structure, and density of states are Python-orchestrated
multi-step workflows, not single engine flavors: each chains
`self_consistent_field`/`non_self_consistent_field` runs (`main.cpp`'s
dispatch comment states this directly). The YAML `calculation:
relaxation`/`band_structure` ("bands")/`density_of_states` ("dos")
convention `examples/{relaxation,band_structure,density_of_states}--*/`
uses is consumed by the matching Python workflow, which reads the same
file and the workflow-specific `control` keys it sets (`band_path` and
`band_path_points_per_segment` for band structure); it is **not** a
calculation type the `q3` binary runs on its own -- `create_flavor()`
(`src/layer_4_computation/flavors/factory.hpp`) raises a specific error
naming the right `q3.workflows` function if one of these three strings
reaches it directly, e.g. from running an example file with
`./build/q3 <file>.yml` instead of through the workflow.

`run_relax_workflow()` and `run_dos_workflow()` read the matching
`control.relax_max_steps`/`relax_force_threshold` and
`control.dos_broadening`/`dos_energy_min`/`dos_energy_max`/`dos_energy_steps`
keys from their own input file as defaults when the caller doesn't pass
them explicitly (`relax_force_threshold` converts Ry/bohr to eV/Å for
ASE's `fmax`) — the same treatment `run_band_structure_workflow` applies
to `band_path`/`band_path_points_per_segment`. `relax_energy_threshold`
is schema-accepted and set by every `examples/relaxation--*/` file but
has no effect on this ASE-BFGS-based workflow, which only has a
force-based convergence criterion; the workflow now warns instead of
silently ignoring it when a file sets it.

**Equation of state, elastic constants and target pressure.** All three are
Python workflows over repeated SCF runs on deformed cells, driven through ASE
and `Q3ASECalculator` (which writes each point's own `material.json` +
`input.yml`, so any point can be re-run by hand). The workflows hold only the
`run_*` orchestration: the strain is `q3.structure.strain`, the fits and the
stress conversions are `q3.properties`, marked to move to mat3ra-made and
mat3ra-prode respectively ([TODOs.md](TODOs.md) §4). The Birch–Murnaghan fit is
ASE's `EquationOfState`; V0 and E0 come back as the ESSE `volume` and
`total_energy` properties, while B0, B0' and the C_ij have no ESSE schema.

Measured on the silicon norm-conserving PBE example (48 Ry, 4×4×4 k-grid,
`tests/validation/test_equation_of_state.py` and
`tests/validation/test_elastic_constants.py`):

| quantity | Q3                     | PBE literature | experiment |
| :------- | :--------------------- | :------------- | :--------- |
| V0       | 40.891 Å³              | —              | —          |
| B0       | 88.1 GPa               | 88–89 GPa      | 97.9 GPa   |
| B0'      | 4.29                   | ~4.2           | ~4.2       |
| C11      | 152.6 GPa              | ~153 GPa       | 165.7 GPa  |
| C12      | 56.2 GPa               | ~57 GPa        | 63.9 GPa   |
| C44      | 99.8 GPa (clamped ion) | ~100 GPa       | 79.6 GPa   |

V0 returns to the input cell — itself a PBE-relaxed MaterialsProject
structure — to 0.007 %. Two things cross-check without any external
reference. The bulk modulus from the elastic constants, (C11 + 2C12)/3 =
88.3 GPa, agrees with the Birch–Murnaghan B0 to 0.2 %, though one fits
energies across volume and the other differentiates the stress with respect
to strain. And the fitted P(V) reproduces Q3's own computed stress at every
scanned point to better than 0.05 GPa over ±12 % in volume, although the fit
never sees a stress.

C44 is the **clamped-ion** value: fractional coordinates are held fixed, so
the sublattice sliding a shear induces in the diamond structure is not
included. That relaxation is the whole of the difference between ~100 GPa and
the ~75 GPa that compares with experiment — see [TODOs.md](TODOs.md) §4.

The strain set is applied in **both signs** of every magnitude, which is the
method rather than a refinement of it. Deforming the cell changes the
symmetry-reduced k-point set, adding a stress term that is even in the strain:
measured at 0.985 GPa in silicon's shear component at 4×4×4 — twice the
physical response — and constant in the magnitude to four digits, which is how
it is told apart from third-order elasticity (which scales as the square, as
the uniaxial pattern's own even part does). A zero-intercept fit over a
sign-symmetric set removes every even term from the estimate exactly (the
reference cell's own stress included), so C44 is k-converged at 4×4×4
(99.82 GPa against 99.78 at 6×6×6) while a one-sided difference at 2×2×2 would
have overestimated it sevenfold.

Target-pressure relaxation is Q3's answer to QE's `&CELL press`, which the QE
parser refuses while naming this parameter. `run_vc_relax_workflow` hands
ASE's cell filter the enthalpy E + PV, so the converged cell is the one whose
own stress balances the target rather than the one whose stress vanishes.
Validated against the equation of state, which reaches the same volume
through the fitted P(V) and a different quantity: on silicon at 10 GPa the two
agree to 0.03 % in volume (`tests/validation/test_vc_relaxation.py`).
`cell_dofree` (constrained/partial cell relaxation) stays refused.

Γ-point phonons (`run_gamma_phonons_workflow`, central-difference forces
via `Q3ASECalculator` -> mass-weighted dynamical matrix -> ASR ->
diagonalize at q=0, using ASE's `Phonons` class) are implemented and
unit-tested on their own math (`tests/python/test_properties_phonons.py`), and the
acoustic sum rule correctly zeroes the three acoustic modes on real Q3
forces (`tests/validation/test_gamma_phonons.py::test_silicon_acoustic_modes_vanish_after_asr`).
Validating this workflow found and fixed a real Q3 bug: forces on any
structure with reduced (non-maximal) point-group symmetry had spurious
components that must be exactly zero by symmetry, traced to
`SymmetryAnalyzer::symmetrize_vector` applying the crystal rotation matrix
un-transposed to a covariant quantity (forces) — invisible for the full
48-operation cubic point group every other validated system in this
codebase uses, not for the reduced subgroups a displaced structure has.
Fixed in `feature/2026-07-26-0-fix-reduced-symmetry-forces`; see
`tests/unit/layer_1_structure/symmetry/test_symmetry.cpp` for a direct
regression test.

A second, deeper issue was found immediately after (validating the first
fix against the full Γ-phonon calculation, not just a single displacement)
and is now also fixed: `SymmetryAnalyzer::reduce_kpoints_internal` — the
default automatic Monkhorst-Pack + IBZ reduction path — built k-point
equivalence classes from the Bravais LATTICE's holohedral point group
(lattice vectors only), not the crystal's actual, atomic-basis-aware point
group. For any structure whose basis reduces the true symmetry below the
lattice's (any finite-difference-displaced structure), that group is too
large, silently treating k-points as equivalent under rotations that are
not real symmetries of the decorated crystal and corrupting the
k-point-summed forces and stress. Fixed in
`feature/2026-07-27-0-ibz-vector-reconstruction` by reusing
`get_symmetry_operations()` (the same spglib-backed, basis-aware detection
already used for density symmetrization) in place of the hand-rolled
Bravais-only group, and its `reciprocal_rotation` field (k-vectors are
reciprocal-space quantities; a crystallographic rotation matrix in a
generally oblique fractional basis is not its own inverse-transpose).
Verified two ways: (a) on the exact displaced-Si case TODOs.md's prior
investigation had already checked against an established plane-wave DFT
code, the fixed default path now reproduces both the QE-cross-checked
force (`+0.0048` Ry/bohr) and stress exactly, matching an independent
spglib-based reduction and a fully unreduced explicit k-point list to
displayed precision — where it previously gave a sign-flipped force and a
qualitatively wrong stress tensor; (b) `test_silicon_optical_modes_triply_degenerate`
now passes outright (was `xfail`) — see
`tests/unit/layer_1_structure/symmetry/test_symmetry.cpp`'s
`ReduceKpointsInternalUsesActualCrystalSymmetryNotBravaisLattice` for the
regression guard, and TODOs.md §1.3 for the full writeup.
`test_silicon_gamma_phonons_match_qe` (the direct, wired-in QE comparison)
still needs a run with an established reference DFT code available to
confirm formally — not possible in every environment — so this row stays
🟡 pending that run, rather than the DFT+U-style "partially validated"
language this legend deliberately has no tier for.

**Phonon dispersions and thermodynamics** (`run_phonopy_workflow`, same
module) drive Phonopy as a library: it generates the symmetry-reduced
displaced supercells, `Q3ASECalculator` evaluates the forces on each,
the force sets go back as force constants (acoustic sum rule through
Phonopy's own `symmetrize_force_constants`), and Phonopy answers at any
q — a dispersion along a `q3.workflows.kpath` band path, the density of
states, and harmonic free energy, entropy and heat capacity from a q-mesh.
The dispersion, the density of states and the zero-point energy come back
as ESSE properties (`phonon_dispersions` in THz, one series per branch);
ESSE has no temperature-dependent schema for the thermal properties, which
come back as its abstract two-dimensional plot against temperature, and none
for force constants, which stay in `phonopy_params.yaml` named by an ESSE
`file_content` property ([TODOs.md](TODOs.md) §4). Silicon in a
2x2x2 supercell costs two 16-atom SCF runs: Phonopy reduces the
displacement set to one direction, doubled to ±u. On the `nc.yml`
example's own settings it gives a Γ-optical mode of 14.38 THz (against
15.53 THz measured), TA(X) = 4.15 THz and TA(L) = 2.94 THz (4.49 and 3.45
measured), C_v(1000 K) = 48.9 J/K/mol of unit cells against the
Dulong–Petit 49.9, and a zero-point energy of 0.058 eV/atom.

The ~7% shortfall is not k-point convergence, as first thought: QE through
the same Phonopy, on the same nominal mesh, gives 15.22 THz. On the
displaced supercell (space group Cm) the 1x1x1 shifted mesh is one k-point
in Q3 and three in QE, which completes each mesh point's star under the
lattice's rotations; given QE's three points explicitly, Q3's forces match
QE's to 6e-4 eV/Å. The difference is the engine's mesh generation, tracked
in [TODOs.md](TODOs.md) §1.3.

The central difference is not optional here, and finding that out is
what the validation bought. Phonopy's default displacement set uses one
sign wherever site symmetry makes ±u equivalent, which holds for the
exact energy surface but not for a k-sampled one: on that same 2x2x2
shifted mesh, silicon's ±0.01 Å forces are −0.00887 and +0.00150 Ry/bohr,
an antisymmetric half that is the physics and a symmetric half of
−0.00368 Ry/bohr that is sampling error — one that falls to +0.00028 by
6x6x6 and vanishes identically on a Γ-centred mesh. A forward difference
keeps all of it and reports a Γ-optical mode of 4.9 THz instead of 14.4,
with no diagnostic anywhere; the central difference cancels it by
construction, which is why the Γ-only path above never saw it. It is the
same sampling effect as above: the residual vanishes with QE's k-points,
and what the central difference cannot cancel is the response itself
coming out low.

`tests/validation/test_phonopy_phonons.py` covers the workflow against
the real binary, and carries the Si-vs-QE+Phonopy comparison the
dispersions need to graduate to ✅: the same Phonopy displaced supercells
pushed through `pw.x`, so a disagreement is a disagreement about forces
and nothing else. It needs the pinned reference build and self-skips
without it. Run, it fails by 0.84 THz at Γ for the mesh reason above, and
is `xfail(strict=True)` until the engine samples the star, so this row
stays 🟡.

Variable-cell relaxation (Si, +2% isotropic strain) recovers the
unstrained equilibrium lattice constant to 0.14% and agrees with an
independent QE `vc-relax` run on the identical starting geometry to
0.004% (`tests/validation/test_vc_relaxation.py`). This workflow had
three compounding bugs before this validation existed and none of them
were caught by any prior test: (1) `Q3ASECalculator` never wrote the
optimizer's updated cell back into the per-step material JSON at all, so
Q3 silently re-solved the ORIGINAL cell on every step regardless of what
the ASE cell filter did; (2) the Q3-to-ASE stress conversion was missing
a sign flip between Q3's Nielsen-Martin convention and ASE's
tensile-positive convention, which would have driven the cell filter the
wrong way once (1) was fixed; (3) `_material_json_to_ase_atoms`'s FCC/BCC
cell construction used the ESSE JSON's primitive lattice length directly
instead of converting it to the conventional celldm(1) Q3 itself uses,
making the cell fed back to Q3 too small by `1/√2` (FCC) or `√3/2` (BCC).
All three were dormant and invisible for ionic-only relaxation (Q3 never
saw the ASE cell at all there) and only became consequential once (1) was
fixed. `Q3ASECalculator`'s bindings mode (`q3_core` nanobind path) has no
API to update the solver's cell at all, so it now raises rather than
silently ignoring a mismatched or changing cell; `run_vc_relax_workflow`
forces subprocess mode accordingly.

Wannier interpolation is native end to end: an opt-in NSCF pass
(`control.interpolation`) writes the overlap, projection and
eigenvalue matrices as an interpolation record, and `q3_utility interpolate` builds
the projection gauge U = A(A†A)^(−1/2), transforms to a real-space H(R) on
the Wigner-Seitz supercell, and interpolates bands and DOS at arbitrary
k-points. Verified at three levels: the k → R → k round trip reproduces the
input eigenvalues to 1e-14 eV at the coarse mesh points (an exact identity
there); an analytic cubic band is reproduced off-mesh to 1e-10; and against
Q3's own direct calculation on a Γ–X–L path the interpolation error falls
from 67.8 meV rms on a 4×4×4 grid to 10.0 meV on 8×8×8 — the convergence
being what distinguishes interpolation error from a bookkeeping bug. Against
Quantum ESPRESSO on the same cell and grid, Q3's NSCF eigenvalues agree to
0.006 meV, and the interpolation from Q3's own interface matrices agrees
with the one from `pw2wannier90.x`'s to 0.6 meV rms along Γ–X–L–Γ — the
residual being the two codes' projection-gauge conventions (Q3
orthonormalizes the trial orbitals before projecting, pw2wannier90 does
not), a hundredth of the interpolation error they share
(`tests/validation/test_wannier_interpolation.py`,
`examples/band_structure--semiconductor--silicon--wannier/`, tutorial 08).

The construction is layered as a basis-agnostic local-orbital
representation (`LocalOrbitalHamiltonian`, layer 3: H(R) blocks, optional
S(R), weighted Bloch sums) consumed by generic band-structure and DOS
calculators, so an atomic-orbital or Slater-Koster Hamiltonian would run
through the same code. Trial orbitals are a strategy behind one interface
(the pseudopotentials' atomic wavefunctions, or hydrogenic s orbitals at
`control.interpolation.projection_centers`). Ultrasoft and PAW datasets are
dispatched to the augmented interface kernels — the S metric in A_mn(k) and
the augmentation term in M_mn(k,b) — which pass their internal identities;
only the norm-conserving path is cross-validated against Quantum ESPRESSO.

Entangled windows are handled by projectability disentanglement
(`control.interpolation.disentanglement`), and the gauge inside the chosen
subspace is minimised toward maximum localization by default
(`control.interpolation.localization`), with guiding centres fixing the branch
of Im ln so the minimisation does not stall where a diagonal overlap crosses
the negative real axis. Six examples cover isolated, offset, entangled,
symmetry-free and mixed-chemistry cases, gated by a `wannier-tier2` CI job.
The **norm-conserving** construction is validated against Wannier90 4.0.1 run
as a black box on Q3's own exported matrices
(`tests/validation/test_wannier90_parity.py`): Ω_I agrees to 10⁻⁶ Å² or
better on all four norm-conserving cases, and the minimised Ω to 10⁻⁶ Å² on
the isolated manifolds and 8×10⁻⁴ Å² on the disentangled alloy.

The **augmented** construction is validated on silicon, not on copper. The
same harness runs four PAW and ultrasoft cases: silicon agrees with Wannier90
on both families, Ω_I, Ω and the centres, while on copper the localisers do
not reach the same minimum — Q3's stalls short of Wannier90's on ultrasoft,
and both stall on PAW's eighteen functions. See
[KNOWN-ISSUES.md](KNOWN-ISSUES.md), and treat a copper PAW or ultrasoft
Wannier model as unvalidated.

The protocol around those mechanics is in `q3.workflows.wannier`: the mesh
from a 0.2 Å⁻¹ spacing (the lever that matters — the same automatic recipe on
silicon gives 826 / 110 / 12.6 meV at 4×4×4 / 6×6×6 / 8×8×8 against at most
1.4× from the gauge), the published E_F + 2 eV frozen window with the
singular-gauge exclusion reported, a quality report the run writes itself,
the band distance η against a reference calculation, and a retry ladder that
accepts, retries or rejects unattended. 🟡 because its 10 / 100 meV gates
have not yet been re-set from a population — the campaign
(`scripts/validation/run_wannier_campaign.py`) has run its 40-row roster, and
nothing on it lands between 9.7 and 87.6 meV, so the 10 meV gate is untested
there — and because leaving semicore shells out of the trial set and SCDM
are not built. Tracked in
[TODOs.md](TODOs.md) §1.3.

The construction is composed, not branched: `WannierHamiltonianBuilder`
checks a record's inputs once, asks a `SubspaceSelection` (the window as it
stands, or projectability disentanglement) and a `GaugeChoice` (the
projection gauge, or the Marzari-Vanderbilt minimum), each chosen by a
factory from one `WannierConstructionSettings`, and Fourier-transforms; the
kernels underneath work on a layer-0 column-major `DenseMatrix` with one
`zgemm`, one Hermitian eigensolver and one SVD behind them. Every failure a
driver can act on carries a `WannierDiagnostic` code that `q3_utility` prints
on the line after the prose, and the results file has one schema
(`schemas/q3/interpolation/results.json`) that generates the writer and is
mirrored by the Python reader under a parity test. The protocol's retry
ladder is one class per rung over a frozen state, routed on the code with
the prose as the fallback for an older binary; its campaign-scale tests are
tier 3, run monthly by the scheduled workflow, which keeps the table.

PDOS/Löwdin charges cross-validated against QE's `projwfc.x` on Si
(insulator, non-magnetic) and Ni (FCC, LSDA-FM): max Löwdin charge
Δ = 0.00024 e (Si) / 0.00097 e (Ni), both well inside the roadmap's
Δ < 0.01 e gate; spilling parameter agrees to Δ ≤ 0.0001; per-shell PDOS
integrals agree to <1% with correlation 0.97–0.999 across every shell,
including Ni's two same-_l_ shells (3S, 4S)
(`tests/validation/test_qe_projwfc_comparison.py`). Both codes
independently converge SCF, then run a matched-resolution NSCF (explicit
full Monkhorst-Pack grid) before projecting — an end-to-end agreement
check, not a comparison against a frozen reference.

Ultrasoft and PAW datasets are projected in the S metric: those schemes give
up norm conservation, so the states solve `H|ψ⟩ = ε S|ψ⟩` with
`S = 1 + Σ q_ij |β_i⟩⟨β_j|` and the projection that conserves charge is
`⟨φ'_i|S|ψ_n⟩`, with the atomic orbitals Löwdin-orthogonalized in the same
metric (Vanderbilt, PRB 41, 7892 (1990); Blöchl, PRB 50, 17953 (1994);
Kresse & Joubert, PRB 59, 1758 (1999)). The sum rule that follows — the
Löwdin charges accounting for the valence electron count, up to the
basis-set spilling — is checked for a silicon ultrasoft and a silicon PAW
case in `tests/validation/test_pdos_s_metric.py`: on a 4×4×4 grid and
against 8 valence electrons the charges sum to 7.947881 e (spilling 0.0065)
and 7.935842 e (spilling 0.0080), beside norm-conserving silicon's
7.930530 e (0.0087). The cross-code comparison against `projwfc.x` runs on
the norm-conserving cases above ([TODOs.md](TODOs.md) §1.3).

### 1.6 Validated Systems

| System            | PP Type | ecutwfc (Ry) | k-grid             | Functional | Δ Total Energy vs reference  |
| ----------------- | ------- | -----------: | ------------------ | ---------- | ---------------------------- |
| Si FCC            | NC      |           48 | 2×2×2 (shifted)    | PBE        | < 0.001 Ry                   |
| Si FCC            | US      |           40 | 2×2×2 (shifted)    | PBE        | < 0.001 Ry                   |
| Cu FCC            | NC      |          120 | 4×4×4 (shifted)    | PBE        | < 0.001 Ry                   |
| Cu FCC            | US      |           40 | 4×4×4 (shifted)    | PBE        | < 0.001 Ry                   |
| Fe BCC (LSDA-FM)  | NC      |           90 | 12×12×12 (shifted) | PBE        | 0.11 mRy (Δmoment 0.005 μB)  |
| Ni FCC (LSDA-FM)  | NC      |           60 | 12×12×12 (shifted) | PBE        | 0.013 mRy (Δmoment 0.005 μB) |
| Cr BCC (LSDA-AFM) | NC      |           90 | 12×12×12 (shifted) | PBE        | 0.25 mRy                     |

---

## 2. Performance (HPC)

**Shared memory**

| Feature                            | Status | Notes                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------- | :----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenMP threading                   |   ✅   | Threaded across bands in the Davidson H\|ψ⟩ loop and in density accumulation, and across grid slices in the XC functional; the sphere/grid scatter and gather run inside the band-parallel region. Team sizes come from `layer_0_core/execution/threading.hpp`, which returns one thread for a loop too short to pay for a team, so `OMP_NUM_THREADS=1` forks nothing and matches a build without OpenMP bit for bit |
| FFTW3 with OpenMP threads          |   🟡   | `fftw_plan_with_nthreads`, now used only where band-level threading is not available. With more than one thread the serial grid plans single-threaded transforms and gives each thread its own scratch, so threads go to whole transforms rather than inside one                                                                                                                                                     |
| Wavefunction-specific FFT grid     |   ✅   | Smooth grid (2×ecutwfc) vs dense density grid; QE-format input, and YAML with `control.use_exact_wavefunction_grid`, get the exact 4×ecutwfc grid pw.x builds                                                                                                                                                                                                                                                        |
| ZGEMM-batched KB projectors        |   ✅   | Level-3 GEMM across all bands (serial + GPU)                                                                                                                                                                                                                                                                                                                                                                         |
| Kerker residual preconditioning    |   ✅   | Auto-enabled for smeared/metallic systems                                                                                                                                                                                                                                                                                                                                                                            |
| Local Thomas-Fermi preconditioning |   🟡   | `mixing.preconditioning: local-tf`; screening built from the local density, solved by conjugate gradients. Unit-tested (reduces to Kerker for a uniform density); the convergence gain on inhomogeneous cells is not yet measured                                                                                                                                                                                    |

**MPI (distributed memory)**

| Feature                                      | Status | Notes                                                                                                                                                          |
| -------------------------------------------- | :----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| K-point pool parallelism                     |   ✅   | `--pools N` or manifest setting; omitted/zero means one pool                                                                                                   |
| Band-group parallelism                       |   ✅   | Wavefunction G-vectors distributed by FFT stick within a pool, gamma-only kept; opt-in via `--pools`, no measured speedup                                      |
| Band distribution                            |   ✅   | `--distribute-bands` splits states across a pool at the full G-sphere; divides work, not storage                                                               |
| Non-self-consistent runs under MPI           |   ✅   | `q3_mpi` and `q3_mpi_gpu_cuda` run NSCF as well as SCF, on one rank per k-point pool; eigenvalues and projections gathered in list order                       |
| One execution policy for a whole workflow    |   ✅   | `q3.binary.ExecutionPolicy` (ranks, pools, launcher) passed to any `q3.workflows` function reaches every engine run it makes: SCF, NSCF, each relaxation step  |
| Per-rank memory report                       |   ✅   | Estimate of the named arrays before the SCF loop, measured peak resident set at the end                                                                        |
| Davidson basis size                          |   ✅   | `method.scf.davidson.basis_expansion_factor` (default 2, minimum 2; 4 for a non-self-consistent run unless set) sizes the working basis                        |
| Fermi bisection without per-step collectives |   ✅   | Eigenvalue table gathered once per call and bisected locally in global k-point order, so E_F is bit-identical across pool topologies for identical eigenvalues |
| ScaLAPACK subspace diagonalizer              |   🟡   | Used when `band_size > 1`; redundant-serial fallback otherwise                                                                                                 |

The 10⁻⁸ Ry topology invariance gate is confirmed for norm-conserving silicon,
localized Makov-Payne and Martyna-Tuckerman corrections, LSDA nickel, atomic and
ortho-atomic DFT+U, ultrasoft silicon, and PAW silicon across serial and one/two/four-rank
runs. The matrix records actual pool/band topology, binary hashes, all reported energy
components (including Hubbard and PAW one-center terms), stable result metadata, and
repeated-run spread. LSDA + smearing under
k-point-pool parallelism is also validated (all four smearing schemes, 1/2/4 ranks) after
fixing a nondeterministic MPI collective-desync bug: the SCF convergence decision was made
per-rank from a residual whose per-rank `FFTW_MEASURE` FFTs made it bit-divergent across
pools, desyncing the pools' collective call sequences (crashes or corrupted totals); the
residual is now `MPI_MAX`-synchronized across pools before any branch consumes it — see
[TODOs.md](TODOs.md) §2.3 for the full root-cause writeup. For metallic soft-moment systems
the _genuine_ cross-topology converged-energy difference is ~10⁻⁸–10⁻⁷ Ry (stopping-point
and summation-order differences), and hot Fermi-Dirac smearing only pins the energy to its
flat moment valley (~10⁻⁴ Ry) — the LSDA smearing tests assert per-scheme calibrated
tolerances accordingly. `control.use_deterministic_fft_planning: true` opts a run into
`FFTW_ESTIMATE` planning (fixed heuristic, no runtime timing) in place of the default
`FFTW_MEASURE`, making repeated runs of even the soft-moment Fermi-Dirac case
bit-identical — see [KNOWN-ISSUES.md](KNOWN-ISSUES.md).

**GPU offloading** — Davidson `H|ψ⟩` (FFT local potential, KB nonlocal) + subspace
operations; density/mixing/XC remain on CPU. Selectable via `--precision double|mixed`.

| Backend                      | Target            | Status | Notes                                                                    |
| ---------------------------- | ----------------- | :----: | ------------------------------------------------------------------------ |
| CUDA (cuBLAS/cuSOLVER/cuFFT) | `q3_gpu_cuda`     |   ✅   | Validated on Tesla T4; 6.6× vs serial (GPU-resident Davidson)            |
| CUDA across several GPUs     | `q3_mpi_gpu_cuda` |   ✅   | One rank per device, one k-point pool each; measured on Perlmutter A100s |
| Metal / Accelerate           | `q3_gpu`          |   ✅   | Apple Silicon; energies bit-identical to serial                          |
| HIP / ROCm                   | `q3_gpu_hip`      |   🟡   | Compiles in CI; not yet run on AMD hardware                              |
| SYCL / oneAPI                | `q3_gpu_sycl`     |   🟡   | Runs on CPU-fallback device; FFT not yet on Intel GPU                    |

The GPU-resident Davidson pipeline (device-resident workspace + CUDA
scatter/gather/potential kernels) has shipped — see
[PERFORMANCE.md](PERFORMANCE.md). **Roadmap**: MPI+GPU (NCCL), native
SYCL/Metal FFT, and cluster-scale benchmarking. See [TODOs.md](TODOs.md) §2.

**Benchmark tracking**

| Feature                          | Status | Notes                                                                                                                                                                                                                                                                                                    |
| -------------------------------- | :----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Benchmark harness                |   ✅   | `scripts/benchmark/run_benchmarks.py` runs the decks in `tests/regression/benchmark_matrix.yml` and writes one JSON record per run — per-stage times and work counts from the engine's own counters, wall time, machine, commit and binary digest                                                        |
| Regression comparison            |   ✅   | `scripts/benchmark/compare_benchmarks.py` reduces repeated runs to a median and a spread, then fails on a tracked stage that slowed by more than 10%, or by more than three times its measured spread where that is wider                                                                                |
| `benchmark` workflow             |   ✅   | `.github/workflows/benchmark.yml` runs the CI-tagged decks on x86-64 and arm64 GitHub-hosted runners, each against its own runner class's baseline (`docs/benchmarks/baseline-github-*.json`), and reports on the run summary; non-blocking by design — a shared runner's clock cannot carry a hard gate |
| Generated PERFORMANCE.md section |   ✅   | `scripts/benchmark/generate_performance.py` renders the committed baselines into PERFORMANCE.md between generated markers; the rest of that document stays hand-written                                                                                                                                  |

Timings are only comparable within one machine: every baseline carries the
machine it was taken on and the comparison refuses to cross that line.
A scheduled head-to-head against Quantum ESPRESSO is not part of this — see
[TODOs.md](TODOs.md) §2.4.

---

## 3. Engineering (Developer Experience)

### 3.1 Input & Output Formats

| Feature                                                           | Status | Notes                                                                                                                                                                                                          |
| ----------------------------------------------------------------- | :----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YAML input                                                        |   ✅   | Primary input format                                                                                                                                                                                           |
| YAML manifest unknown-key rejection                               |   ✅   | model/method/control fail fast on a typo instead of silently defaulting                                                                                                                                        |
| Namelist input (established DFT code format)                      |   ✅   | Alternative input format; unknown keys, namelists and cards refused; `HUBBARD` card, `celldm(2..6)`, `A`/`B`/`C`, `&ions`/`&cell` read                                                                         |
| Reference-code input drop-in case set                             |   ✅   | 10 cases where a reference code's _own_ input file is fed to Q3 and its own printed numbers are the reference — `tests/validation/test_reference_input_drop_in.py`                                             |
| `--check` (parse and report, do not run)                          |   ✅   | Answers "will Q3 read this deck, and as what?" without an SCF                                                                                                                                                  |
| `control.verbosity` (`default` / `high`)                          |   ✅   | `high` (or `--log-level debug`) adds the per-iteration and per-phase diagnostics a default run leaves out; QE's `verbosity` maps onto it                                                                       |
| UPF v1 (tag-based) / UPF v2 (XML) readers                         |   ✅   | NC + US; PAW in UPF v2                                                                                                                                                                                         |
| JSON results output                                               |   ✅   | Canonical `<prefix>_results.json` from CLI and bindings, with legacy unprefixed lookup; self-describing units, execution topology, and §3.2's full record                                                      |
| Plane-wave binary output (established code format)                |   ✅   | `wfcN.dat` / `charge-density.dat` in Quantum ESPRESSO's record layout (packed header, interleaved Miller, 1/Bohr); non-spin-polarized only — see [TODOs.md](TODOs.md) §3.1                                     |
| Binary checkpoint save/restore (wfn + ρ)                          |   ✅   | Serial, and MPI with one rank per k-point pool (the automatic topology), each pool writing its own `wfc<N>.dat`; shared pools refuse — [TODOs.md](TODOs.md) §3.1                                               |
| BerkeleyGW mean-field export (WFN + RHO + vxc.dat)                |   ✅   | `mean_field_export: berkeleygw`; ZnO G0W0 through BerkeleyGW 4.0 within 1 meV of QE + `pw2bgw.x`; gaps — [TODOs.md](TODOs.md) §3.1                                                                             |
| BerkeleyGW `VXC`, `VSC` and `VKB` (sigma, parabands)              |   ✅   | `VKB` on request (`mean_field_export_kleinman_bylander_projectors`); `parabands` on Q3's `VSC` + `VKB` reproduces pw.x's eigenvalues to 0.004 meV on silicon; norm-conserving only — [TODOs.md](TODOs.md) §3.1 |
| Mean-field record + standalone converter (`q3_utility convert`)   |   ✅   | `<prefix>.save/mean-field.q3mf` (written when `mean_field_export` is set); converting it reproduces the in-process `WFN`, `RHO`, `vxc.dat` and `VXC` byte for byte                                             |
| Volumetric field export (cube, `.xsf`)                            |   ✅   | `q3_utility volumetric`: ρ, V_eff and a band's \|ψ\|² off a finished run's save set; the written density integrates to the valence electron count                                                              |
| Plane-wave charge-density reader (`charge-density.dat`)           |   ✅   | Reads a density back, Quantum ESPRESSO's layout and the older Q3 one; a saved band (`wfc<k>.dat`) is readable too, in Quantum ESPRESSO's layout                                                                |
| Build-time add-ons: input, dataset and volumetric format plug-ins |   ✅   | `src/addons/<name>/`, compiled in when present and never named by core; listed by `q3 --version` and in `provenance.generator.addons` (note below)                                                             |

**Build-time add-ons** plug into core through registries — input formats, each with
the companion outputs it writes after a run it parsed; pseudopotential formats; PAW
one-center conventions; and volumetric formats — so core names none of them.
`q3 --version` prints a second `add-ons:` line naming each one compiled in, and every
`results.json` lists them in `provenance.generator.addons`, empty for a build without
any ([io_reference.md](docs/io_reference.md) §3 and §4.1,
[architecture.md](docs/architecture.md)).

**The drop-in claim is tested against the reference codes' own files.** Every
other comparison test generates the reference input _from_ Q3's YAML, which
leaves one loophole: a wrong translation in the generator makes both sides
agree on the wrong calculation. In
`tests/validation/test_reference_input_drop_in.py` the reference file is
primary and Q3 consumes it byte for byte -- 10 QE cases (NC, ultrasoft, PAW,
Gamma-only, DFT+U via the `HUBBARD` card, ferromagnetic Fe and Ni,
antiferromagnetic Cr) against the outputs QE itself produced.

Measured: NC 2e-6 Ry, Gamma-only 4e-6, ultrasoft 2e-5, DFT+U total and E_U
within 2e-4, PAW 2.9e-4 to 6.9e-4 (the known one-center residual), Cr AFM
2.5e-4 Ry with |m| 2.5929 against QE's 2.60. PAW and chromium carry their own
tolerances, set from those measurements with the reason recorded beside each.

Its first run found defects in files that had been committed all along:
`mixing_mode = 'plain'` (QE's default) killed Q3 in the mixer factory, and Q3
reported no _absolute_ magnetization -- the only number that separates a
converged antiferromagnet from one whose moments collapsed.

YAML unknown-key rejection covers `model`/`method`/`control` and their sub-objects, where Q3
owns the full key set; inline YAML-embedded `material` (as opposed to an external material
JSON file, which every example uses) is deliberately not covered — see [TODOs.md](TODOs.md) §3.1.
The QE namelist parser refuses unknown keys, unknown namelists and unsupported cards, and
names the reason for anything it recognizes but cannot honour (`qe_input::`'s four tables in
`io/input/parser/qe.hpp`). Keys stating a QE default that switches a feature off — `noinv =
.false.`, `tot_magnetization = -10000` — are accepted, since they ask for what Q3 already
does. 96 of 150 of QE's own `scf` regression inputs parse; every refusal names a real
unimplemented feature (hybrids, noncollinear magnetism, electric fields, explicit
occupations, forced FFT grids).

### 3.2 Schema & Data Model

| Feature                               | Status |
| ------------------------------------- | :----: |
| C++ types generated from JSON schemas |   ✅   |
| ESSE `total_energy` scalar property   |   ✅   |
| Published `results.json` schema       |   ✅   |

**`results.json` carries every number the console report does**, and
[`schemas/q3/results.json`](schemas/q3/results.json) says so: a closed root
that names every key, with `$ref`s into ESSE for the unit enums and array
shapes the two share. The energy breakdown down to the dispersion term, the
per-atom force decomposition beside the total, the scalar pressure, both
magnetizations, DFT+U `Tr[n]` per manifold, and the band edges with the gap
between them are all keys of the file, filled by every flavor into the one
struct generated from the schema ([`io/output/results.hpp`](src/io/output/results.hpp))
and validated against real runs in `tests/regression/test_results_schema.py`.
That is what lets the test suite read results rather than scrape stdout. The
console report remains, and one smoke test keeps it honest, but nothing
depends on its layout any more.

### 3.3 Python Layer

| Feature                                      | Status | Notes                                                                                                       |
| -------------------------------------------- | :----: | ----------------------------------------------------------------------------------------------------------- |
| `q3` Python package                          |   ✅   | `src/py/q3/`, installed via `pip install -e src/py`                                                         |
| `python -m q3 input.yml` CLI                 |   ✅   | Thin wrapper around the `q3` binary (`--binary`, `--no-capture`)                                            |
| `q3.workflows.relax` / `vc_relax`            |   ✅   | ASE BFGS ionic / `FrechetCellFilter` (`ExpCellFilter` fallback) variable-cell                               |
| `q3.workflows.equation_of_state` / `elastic` |   ✅   | `run_*` orchestration over `Q3ASECalculator`; the strain is `q3.structure.strain`, the fits `q3.properties` |
| `q3.calculators.ase_calculator`              |   ✅   | Mode A (subprocess, any cell) + Mode B (nanobind `q3_core`, fixed cell only)                                |
| `q3_core` nanobind module                    |   ✅   | Stateful `SCFSolver` with wavefunction restart                                                              |
| Per-step relaxation audit trail              |   ✅   | `step_NNNN/{input.yml, material.json, output/}` for full reproducibility                                    |
| Unit conversions                             |   ✅   | All factors from `mat3ra.code.constants.Coefficients`; no magic numbers                                     |

### 3.4 Build & Dependencies

Header-heavy library built with CMake. The default target is the serial `q3`
executable; opt-in flags add `q3_mpi` (`Q3_WITH_MPI`) and the GPU executables
`q3_gpu` / `q3_gpu_cuda` / `q3_gpu_hip` / `q3_gpu_sycl` (`Q3_WITH_GPU`,
`Q3_WITH_CUDA`, `Q3_WITH_HIP`, `Q3_WITH_SYCL`). LAPACK comes from the system,
and so does FFTW3 for the default FFT backend; `-DQ3_FFT_BACKEND=pocketfft` swaps
the serial FFT grid to the fetched, header-only, BSD-3-Clause PocketFFT, and then
FFTW is neither searched for nor linked, so the serial `q3`, `q3_utility` and the
tests build without any GPL library (MPI and GPU builds still require the FFTW
backend; CMake refuses the combination); LibXC is used from the system when it is from a tested series (5.2 or
7.0), else 7.0.0 is built from source; the core third-party libraries are fetched
via CMake FetchContent; parallel/GPU dependencies come from the system toolchain
and are required only for their respective opt-in targets.

| Library                         | Purpose            | Provided via                 |
| ------------------------------- | ------------------ | ---------------------------- |
| LibXC                           | XC functionals     | System or CMake FetchContent |
| spglib                          | Symmetry detection | CMake FetchContent           |
| FFTW3                           | FFT                | System (find_library)        |
| PocketFFT                       | FFT (alternative)  | CMake FetchContent           |
| LAPACK                          | Linear algebra     | System (find_package)        |
| spdlog                          | Logging            | CMake FetchContent           |
| yaml-cpp                        | YAML parsing       | CMake FetchContent           |
| nlohmann_json                   | JSON I/O           | CMake FetchContent           |
| GoogleTest                      | Testing            | CMake FetchContent           |
| OpenMP                          | Shared-memory      | System (`Q3_WITH_OPENMP`)    |
| MPI, FFTW3-MPI, ScaLAPACK       | Distributed memory | System (`Q3_WITH_MPI`)       |
| CUDA (cuBLAS/cuSOLVER/cuFFT)    | NVIDIA GPU         | System (`Q3_WITH_CUDA`)      |
| ROCm (hipBLAS/hipSOLVER/hipFFT) | AMD GPU            | System (`Q3_WITH_HIP`)       |
| oneMKL                          | Intel GPU (SYCL)   | System (`Q3_WITH_SYCL`)      |
| Accelerate / Metal / MPS        | Apple Silicon GPU  | System (`Q3_WITH_GPU`)       |

**Release archives.** Every GitHub release carries one archive per tier-1
platform (Linux x86_64, Linux arm64, macOS arm64), built by
`.github/workflows/release.yml` and assembled by `scripts/repo/package_release.py`.

| What a release carries                             | Status |
| -------------------------------------------------- | :----: |
| Native archives, three platforms                   |   ✅   |
| Smoke test of each archive outside its build job   |   ✅   |
| `SHA256SUMS` over every asset                      |   ✅   |
| CycloneDX SBOM, in the archive and beside it       |   ✅   |
| Licenses of everything compiled in, in `licenses/` |   ✅   |
| Build-provenance attestation (public repo only)    |   ✅   |

The SBOM is built from the build's own dependency pins and the binaries' own
link tables, not by a scanner: a C++ executable has no package manifest
inside it, and a generic scan finds almost nothing. That is also how the
packager notices a system LibXC compiled in from Ubuntu's static `libxc.a`,
which `ldd` cannot see. The Linux smoke test runs in a clean `ubuntu:24.04`
container with only the packages the archive's `INSTALL.txt` names, then an
SCF with the unpacked binary, so a wrong install line fails the release
rather than the first user. Two packagings of one build are byte-identical.

### 3.5 Testing

| Suite                                               | Framework          | Status |
| --------------------------------------------------- | ------------------ | :----: |
| Unit & integration (C++)                            | GoogleTest / CTest |   ✅   |
| Regression (binary outputs)                         | pytest             |   ✅   |
| Validation (vs reference)                           | pytest             |   ✅   |
| CPU↔GPU energy parity                              | pytest             |   ✅   |
| LSDA / lattice / supercell-folding self-consistency | pytest             |   ✅   |
| MPI topology / band invariance                      | pytest (`mpirun`)  |   ✅   |
| Type-query ratchet (dynamic casts, `has_*` flags)   | pre-commit         |   ✅   |
| Public-API docstring ratchet (C++ headers, Python)  | pre-commit         |   ✅   |
| A/B parity of a pull request against its merge base | `parity` workflow  |   ✅   |

Two ratchets run in the `lint` job with no C++ build: per-file counts may only
fall, and a fall must be committed to the baseline in the same change.
`scripts/repo/check_cpp_type_queries.py` counts `dynamic_cast`/`typeid`, downcasts to
ultrasoft/PAW types, `as_ultrasoft()`/`as_paw()`, pseudopotential-family `has_*()`
queries, other `has_*()` branch conditions, and string-literal physics switches
(121 at baseline, `--report` for the breakdown).
`scripts/repo/check_docstring_coverage.py` counts public header declarations and
public `q3` Python definitions without a doc comment (50% of 2046 C++ and 93% of
197 Python declarations documented at baseline). Driving both down is tracked in
[TODOs.md](TODOs.md) §3.3.

The `parity` workflow (`.github/workflows/parity.yml`) answers the one
question a tolerance cannot: did a change that claimed to move no numbers move
any? It builds the pull request's merge base and its head in the same runner,
runs the twenty-three cases of `tests/regression/parity_matrix.yml` with both
binaries over the head checkout's inputs
(`scripts/validation/run_parity_matrix.py`), and compares every numeric leaf
of every `*_results.json` plus every byte of every `.save/` checkpoint exactly
(`scripts/validation/compare_results_json.py`). Only wall-clock timings, peak
resident memory and the engine's own version and commit are exempt, because
those differ between two runs of the same binary. The matrix spans the three
pseudopotential families, Γ-only and shifted meshes, the three smearing
schemes, LSDA on Fe/Ni/Cr, DFT+U with both projector schemes on both families,
LDA and GGA, dispersion, a charged cell, two band-path NSCF passes, and the
ortho-atomic projections on norm-conserving and ultrasoft silicon. A change
that is meant to move numbers carries the `numerics-change` label and a
`numerics-change: <reason>` line in its body, which switches the comparison to
1e-12 relative and echoes the reason into the run log. A change that moves
physics further names the matrix rows it moves in a `numerics-change-rows:`
line: those rows are reported rather than failed, every other row is still
held to 1e-12, and a pattern that names no row is refused. The workflow also
runs when a label or the body changes, so the escape takes effect without a
new commit.

CPU↔GPU parity (`tests/validation/test_gpu_energy_parity.py`: FP64 ≤ 1e-10 Ry,
mixed ≤ 1e-5 Ry; NC, Γ-only, US and PAW cases, plus LSDA nspin=2 on Fe/Ni/Cr
and Methfessel-Paxton/Fermi-Dirac/Marzari-Vanderbilt smearing on Cu — ten
cases at both precisions, every run with deterministic FFT planning so that
the comparison sees the backends rather than FFTW's timing-selected plans)
runs in the CUDA workflow and locally against Metal. Machine-validated
internal-consistency suites cover LSDA-vs-unpolarized
(`test_lsda_consistency.py`), lattice-representation equivalence
(`test_lattice_equivalence.py`), supercell folding
(`test_supercell_folding_consistency.py`: a 2×2×2 silicon cell on a shifted
2×2×2 mesh must give eight times the primitive cell on the shifted 4×4×4 mesh
it unfolds onto — the check no external reference can make, since the
primitive cell agrees with QE either way) and screw axes
(`test_screw_axis_wedge_consistency.py`: wurtzite reduced on a mesh odd along
c must keep the pairs its 6₃ screw makes degenerate and match the full mesh's
energy). MPI validation tests
(`tests/validation/test_mpi_*.py`) run in the `mpi-validation` CI job (1/2/4
ranks) and now also cover LSDA + DFT+U (Hubbard) under band parallelism,
plus the committed reproducibility matrix spanning isolated corrections and
norm-conserving/ultrasoft/PAW datasets
reliably — the first real-MPI-hardware confirmation of the
`reduce_within_group` band-parallelism fix in
`accumulate_hubbard_occupation_and_potential` (previously validated only by
code-reading symmetry, no MPI build available at the time). Plain LSDA +
smearing under k-point-pool parallelism is also covered for all four smearing
schemes (formerly `xfail`): writing that case surfaced a real nondeterministic
MPI collective-desync bug, since root-caused by instrumented per-rank tracing
and fixed by synchronizing the SCF convergence residual across pools — see
[TODOs.md](TODOs.md) §2.3 and `compute_residual_between`'s docstring. The
smearing cases assert per-scheme calibrated tolerances (§2 MPI note above).
