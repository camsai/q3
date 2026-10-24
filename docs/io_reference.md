# Q3 Input/Output Reference

This is the field-level reference for everything Q3 reads and writes: the YAML
manifest format, the alternative QE-namelist format, how build-time add-ons plug in
further formats, command-line flags, and every artifact the engine (or the Python workflow layer
built on it) writes to disk. For the high-level feature matrix see [FUNCTIONALITY.md](../FUNCTIONALITY.md);
for the C++ layer architecture see [architecture.md](architecture.md).

Two general input-file conventions apply throughout: a bare value of the form
`include_yaml("path")` anywhere in a YAML manifest splices in another YAML file
recursively (`src/io/input/parser/yaml.hpp:186-228`), and the path selects the
parser — a build-time add-on's format when it recognizes the path (§3), `.in`/`.qe` → QE
namelist, everything else → YAML (`src/io/input/parser/factory.hpp`). `--format` overrides
the detection.

---

## 1. YAML manifest format

The primary, most complete input format. It is described by
[`schemas/q3/input.json`](../schemas/q3/input.json), which is not documentation
of the parser but its input: the schema's per-level key sets are code-generated
into `src/generated/q3/input.hpp` and are what `reject_unknown_keys()`
(`src/io/input/parser/yaml.hpp`) validates a manifest against, so an
unrecognized key throws immediately at parse time rather than being silently
ignored, and a key exists for the parser only once the schema declares it. The
tables below annotate the same keys with type, default, effect, and units.

### 1.1 Root

Schema path: the document root.

`material` is either a path to an external ESSE material JSON (resolved
relative to the manifest's own directory) or an inline map with `lattice`/
`basis` sub-keys mirroring the same ESSE shape. It is **not** covered by
`reject_unknown_keys` — material legitimately carries ESSE fields Q3 doesn't
consume, so the schema deliberately leaves it open.

| Path                                         | Type                        | Default                  | Description                                                       | Units                         |
| -------------------------------------------- | --------------------------- | ------------------------ | ----------------------------------------------------------------- | ----------------------------- |
| `material`                                   | string \| map               | required                 | External material JSON path, or inline `{lattice, basis}`         | —                             |
| `material.lattice.type`                      | string                      | —                        | `FCC`/`BCC`/`SC`/`CUB`/`HEX`/general                              | —                             |
| `material.lattice.a` / `b` / `c`             | float                       | —                        | Lattice parameters                                                | bohr or angstrom, per `units` |
| `material.lattice.alpha` / `beta` / `gamma`  | float                       | 90° (120° for HEX gamma) | Lattice angles                                                    | degrees                       |
| `material.lattice.units`                     | string \| `{length, angle}` | `"bohr"`                 | Length/angle unit for the scalar lattice fields                   | —                             |
| `material.lattice.vectors.a` / `b` / `c`     | float×3                     | —                        | Explicit lattice vector rows                                      | per `vectors.units`           |
| `material.lattice.vectors.alat`              | float                       | —                        | Scalar multiplier for the vector rows                             | —                             |
| `material.lattice.vectors.units`             | string \| `{length}`        | inherits `lattice.units` | Overrides the unit for the vector rows specifically               | —                             |
| `material.basis.elements[].id` / `.value`    | int / string                | —                        | Species index / element symbol                                    | —                             |
| `material.basis.coordinates[].id` / `.value` | int / float×3               | —                        | Species index / fractional (crystal) position                     | crystal coords                |
| `material.basis.units`                       | string                      | —                        | Read but not applied to fractional coordinates in the inline path | —                             |

### 1.2 `model`

Schema path: `model`.

| Path               | Type   | Default  | Description                                                                                      | Units |
| ------------------ | ------ | -------- | ------------------------------------------------------------------------------------------------ | ----- |
| `model.type`       | string | unset    | Accepted, not consumed — DFT/Kohn-Sham is the only supported model                               | —     |
| `model.subtype`    | string | unset    | Accepted, not consumed (self-documents e.g. `gga`)                                               | —     |
| `model.functional` | string | unset    | XC functional name, matched case-insensitively. Unset, the functional the pseudopotentials state | —     |
| `model.dispersion` | string | `"none"` | Dispersion correction, lower-cased on parse                                                      | —     |

**Legal `model.functional` values**, all matched case-insensitively: `PZ`/`SLA+PZ`
(Perdew-Zunger LDA, alias `LDA`), `PBE`, `PBEsol`, `BLYP`, `revPBE`, `RPBE`
(Hammer-Hansen-Norskov exchange with PBE correlation); generic `"libxc:<X>:<C>"` where each half is either a LibXC id or a
LibXC name (`"libxc:117:130"` and `"libxc:gga_x_rpbe:gga_c_pbe"` are the same
functional); or QE's `"XC-iii-jjj-kkk-lll"` form (`src/layer_4_computation/effective_potential/internal/exchange_correlation/factory.hpp:23,37-107`).
Unrecognized values throw.

**With no `model.functional`, the run uses the functional the pseudopotentials
were generated with**, which is pw.x's rule when `input_dft` is absent, and the
same holds for a QE-format deck. Every dataset has to state one and all of them
have to agree, or the input is refused with the dataset named. A UPF header
states it in Quantum ESPRESSO's vocabulary: a short name (`PBE`, `PZ`/`LDA`,
`PBEsol`, `BLYP`, `revPBE`, `RPBE`), the four components it is assembled from
(`SLA PW PBX PBC` is PBE, `SLA PZ NOGX NOGC` is PZ), or QE's LibXC form. A label
Q3 does not map, such as Perdew-Wang LDA (`PW`, which is not PZ) or a nonlocal
term, is refused rather than approximated, and naming the functional in the
input settles it. A named functional always wins, as `input_dft` does in pw.x.
`q3 --check` reports which functional an input resolved to. Q3 used to fall back
on PZ instead, which ran PBE datasets as LDA without a word.

**The functional is recorded lower-cased** (`pbe`, `pbesol`, `libxc:117:130`),
the spelling the example decks and ESSE use, whichever reader it came from and
however the deck wrote it. `--check` and the provenance record in the results file
both show that spelling. `parse_input()` applies it once for
every reader, so `functional: PBE`, `input_dft = 'pbe'` and a dataset stating
`SLA PW PBX PBC` all record `pbe`.

**Legal `model.dispersion` values**: `"none"` and
`"d3bj"` (Grimme D3, Becke-Johnson damping); any other string throws at parse
time, and `create_dispersion_calculator()` (the factory the SCF energy, force,
and stress orchestrators all resolve the scheme through) throws on unknown
names as defense-in-depth.

### 1.3 `method` (+ `cutoffs`)

Schema paths: `method`, `method.cutoffs`.

| Path                          | Type               | Default | Description                                                                                                                       | Units |
| ----------------------------- | ------------------ | ------- | --------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `method.type`                 | string             | unset   | Accepted, not consumed — planewave-pseudopotential is the only method                                                             | —     |
| `method.subtype`              | string             | unset   | Free-form label; only rejected if it names `paw`/`projector-augmented-wave` (PAW guard, not yet implemented)                      | —     |
| `method.pseudopotentials`     | map⟨string,string⟩ | —       | Element symbol → UPF file path (resolved relative to manifest dir); also seeds the pseudopotential directory from the first entry | —     |
| `method.cutoffs.wavefunction` | float              | `30.0`  | Plane-wave kinetic-energy cutoff for wavefunctions                                                                                | Ry    |
| `method.cutoffs.density`      | float              | `0.0`   | Charge-density/potential cutoff; `0.0` = auto (4× wavefunction cutoff)                                                            | Ry    |

### 1.4 `method.scf` (alias `method.self_consistent_field`)

Both spellings are accepted for the container key — they are one `$ref`'d
definition in the schema, so they always accept exactly the same keys; every
committed example uses `scf`. Schema path: `method.scf`.

| Path                                         | Type              | Default      | Description                                                                                                                                                                                                                                                                                                                                                                                       | Units             |
| -------------------------------------------- | ----------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `method.scf.convergence_threshold`           | float             | `1e-6`       | SCF density-residual/energy convergence threshold                                                                                                                                                                                                                                                                                                                                                 | Ry                |
| `method.scf.max_iterations`                  | int               | `100`        | Maximum SCF iterations                                                                                                                                                                                                                                                                                                                                                                            | —                 |
| `method.scf.diagonalization`                 | string            | `"davidson"` | Eigensolver: `"davidson"` (iterative, default) or `"direct"` (dense LAPACK)                                                                                                                                                                                                                                                                                                                       | —                 |
| `method.scf.davidson.basis_expansion_factor` | int               | `2`          | Davidson working basis as a multiple of the band count; the solver holds two arrays of that many vectors. Default 2 (minimum 2); 4 restarts less often at twice the footprint and was the default before 2026-09-13. A non-self-consistent run uses 4 unless this is set, since a single pass at 2 can miss an eigenvalue; a lower value set here is kept there, with a warning (KNOWN-ISSUES.md) | —                 |
| `method.scf.number_of_bands`                 | int               | `0` (auto)   | Number of bands/states; shared field with `method.band_structure.number_of_bands`, which wins if both are set                                                                                                                                                                                                                                                                                     | —                 |
| `method.scf.total_charge`                    | float             | `0.0`        | Net cell charge (`tot_charge` convention: positive = cationic)                                                                                                                                                                                                                                                                                                                                    | electrons         |
| `method.scf.hubbard`                         | list              | `[]`         | DFT+U (Dudarev) specifications — see §1.4.3                                                                                                                                                                                                                                                                                                                                                       | —                 |
| `method.scf.hubbard_projectors`              | string            | `"atomic"`   | Hubbard manifold projector convention: `"atomic"` (raw pseudo-atomic orbitals) or `"ortho-atomic"` (Löwdin-orthogonalized over the full system-wide atomic-orbital set first). Both conventions support norm-conserving, ultrasoft, and PAW datasets                                                                                                                                              | —                 |
| `method.scf.hubbard_starting_eigenvalues`    | list              | `[]`         | First-iteration DFT+U occupation-eigenvalue seeds — see §1.4.4                                                                                                                                                                                                                                                                                                                                    | —                 |
| `method.scf.nspin`                           | —                 | —            | **Rejected** (not in the schema) — throws pointing to `number_of_spin_components`                                                                                                                                                                                                                                                                                                                 | —                 |
| `method.scf.number_of_spin_components`       | int               | `1`          | `1` = unpolarized, `2` = collinear LSDA; any other value throws                                                                                                                                                                                                                                                                                                                                   | —                 |
| `method.scf.starting_magnetization`          | map⟨string,float⟩ | `{}`         | Per-species initial moment in `[-1, 1]` (negative = opposite AFM sublattice)                                                                                                                                                                                                                                                                                                                      | fractional moment |

`diagonalization="direct"` is incompatible with DFT+U (non-empty `hubbard`) —
that combination throws before SCF because the direct path does not apply the
Hubbard Hamiltonian. The GPU applicator likewise refuses DFT+U before backend
initialization. Davidson on serial CPU or MPI supports norm-conserving,
ultrasoft, and PAW datasets, including Gamma-only runs.

#### 1.4.1 `method.scf.mixing`

Schema path: `method.scf.mixing`.

| Path                     | Type   | Default     | Description                                                                                                                                                                                                                                                                                                                                                                                                 | Units  |
| ------------------------ | ------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `mixing.mode`            | string | `"broyden"` | `"broyden"` (G-space quasi-Newton, best for insulators/semiconductors), `"linear"` (plain G-space, recommended for metallic ultrasoft systems), `"anderson"` (real-space Pulay/DIIS)                                                                                                                                                                                                                        | —      |
| `mixing.beta`            | float  | `0.7`       | Mixing step size                                                                                                                                                                                                                                                                                                                                                                                            | —      |
| `mixing.ndim`            | int    | `8`         | History depth (Broyden/Anderson only)                                                                                                                                                                                                                                                                                                                                                                       | —      |
| `mixing.preconditioning` | string | `"default"` | `"default"` resolves to `"kerker"` if smearing is enabled, else `"none"`; or set `"kerker"`, `"local-tf"` or `"none"` explicitly. `"local-tf"` builds the screening from the local density instead of one cell-wide value, which is what makes it differ from `"kerker"` on inhomogeneous cells (slabs, vacuum gaps); it costs one FFT pair per conjugate-gradient step. An unrecognized value is rejected. | —      |
| `mixing.q0`              | float  | `1.0`       | Thomas-Fermi screening wavevector for Kerker preconditioning: `K(G) = G²/(G²+q0²)`. Under `"local-tf"` the same value is the floor on the local screening, so the vacuum region stays at Kerker's behaviour rather than going unscreened.                                                                                                                                                                   | bohr⁻¹ |

#### 1.4.2 `method.scf.smearing`

Schema path: `method.scf.smearing`.

| Path               | Type   | Default  | Description                                                               | Units |
| ------------------ | ------ | -------- | ------------------------------------------------------------------------- | ----- |
| `smearing.type`    | string | `"none"` | Occupation-smearing scheme                                                | —     |
| `smearing.degauss` | float  | `0.0`    | Smearing width σ (QE `degauss` convention)                                | Ry    |
| `smearing.order`   | int    | `1`      | Methfessel-Paxton order N ≥ 1; valid only with a Methfessel-Paxton `type` | —     |

**Legal `smearing.type` values and aliases**: `"fixed"`/`"none"`/empty
(insulating step function), `"gaussian"`/`"gauss"`,
`"methfessel-paxton"`/`"m-p"`/`"mp"`,
`"marzari-vanderbilt"`/`"cold"`/`"m-v"`/`"mv"`,
`"fermi-dirac"`/`"f-d"`/`"fd"`.

#### 1.4.3 `method.scf.hubbard[]`

Schema path: `method.scf.hubbard[]` — `{species, orbital_label, u_ev}`, all
three required.

| Path                      | Type   | Description                                                                                                                                 | Units                                       |
| ------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `hubbard[].species`       | string | Element symbol the U correction applies to                                                                                                  | —                                           |
| `hubbard[].orbital_label` | string | Shell selector: the dataset's own `PP_CHI` label (`"3D"`, case-insensitive), or a bare orbital letter (`"d"`) selecting on angular momentum | —                                           |
| `hubbard[].u_ev`          | float  | Hubbard U (Dudarev rotationally-invariant form)                                                                                             | input in **eV**; converted to Ry internally |

#### 1.4.4 `method.scf.hubbard_starting_eigenvalues[]`

A Hubbard functional at practical U has more than one self-consistent
solution, and which one an unseeded SCF finds depends on the starting guess
and the mixing trajectory rather than on the physics. FeO with U = 4.6 eV on
Fe 3d has an orbitally-ordered minimum 123 mRy below the orbitally-uniform
one. Seeding makes that choice an input instead of an accident, which is what
lets two codes be compared in the _same_ minimum — a cross-code energy
difference is meaningless until the converged occupation eigenvalues agree.

On the first SCF iteration, after the occupation matrix is accumulated,
reduced and symmetrized, it is diagonalized; the named eigenvalue is replaced
and the matrix rebuilt from the same eigenvectors. Seeding the spectrum a run
already has is a no-op by construction.

| Key                                         | Type   | Description                                                                                    | Notes                                            |
| ------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `hubbard_starting_eigenvalues[].species`    | string | Species symbol, matched as written in the structure (so `Fe1`/`Fe2` sublattices are separable) | —                                                |
| `hubbard_starting_eigenvalues[].eigenvalue` | int    | Which eigenvalue, **1-based, ascending** — the largest index is the most occupied orbital      | matches the drop-in `(m, ispin, ityp)` numbering |
| `hubbard_starting_eigenvalues[].spin`       | int    | Spin channel, 1 or 2                                                                           | —                                                |
| `hubbard_starting_eigenvalues[].value`      | float  | Target occupation in [0, 1]                                                                    | per-spin convention                              |

`method.scf.hubbard_fixed_occupation_iterations` (drop-in `mixing_fixed_ns`)
re-applies the seed for that many further iterations while the wavefunctions
adapt. Note this is re-seeding, not freezing: each iteration recomputes the
matrix and the named eigenvalue is re-imposed on it, so the _other_ eigenvalues
stay free to move. Re-seeding cannot manufacture a solution that is not a fixed
point either — seeding FeO toward the orbitally-uniform state and holding it for
15 iterations still relaxes into the ordered minimum, because in Q3 that uniform
state is not stable.

> The bare-letter form exists because UPF makes the `PP_CHI` label optional and
> PseudoDojo's PAW datasets omit it — QE's own `determine_hubbard_occ` refuses
> those files for the same reason. When several shells share an angular
> momentum, the most occupied one is selected, since a Hubbard U applies to the
> partially-filled valence shell; an equal-occupancy tie is refused rather than
> resolved by file order.
>
> Atomic and ortho-atomic projectors support norm-conserving, ultrasoft, and PAW
> datasets. Energy, forces, and stress are implemented for each combination,
> including the generalized `<phi|S|phi>` Löwdin metric and its derivatives for
> ultrasoft and PAW. MPI band groups deliberately refuse stress; GPU and direct
> diagonalization deliberately refuse DFT+U. See the support matrix in
> [FUNCTIONALITY.md](../FUNCTIONALITY.md#dftu-support-matrix).

#### 1.4.4 `method.scf.kgrid`

Schema path: `method.scf.kgrid`.

| Path                  | Type   | Default      | Description                                              | Units |
| --------------------- | ------ | ------------ | -------------------------------------------------------- | ----- |
| `kgrid.dimensions`    | int×3  | `[1,1,1]`    | Monkhorst-Pack grid dimensions                           | —     |
| `kgrid.shifts`        | int×3  | `[0,0,0]`    | Grid shift flags (0 or 1) per direction                  | —     |
| `kgrid.ibz_reduction` | string | `"internal"` | `"internal"` (Q3's own symmetry reduction) or `"spglib"` | —     |

### 1.5 `method.band_structure`

Schema path: `method.band_structure` — `{number_of_bands}` only. No committed
example actually uses this block — real band-structure examples set
`method.scf.number_of_bands` instead and switch calculation type via
`control.calculation`.

### 1.6 `control`

Schema path: `control`.

| Path                                                          | Type                          | Default                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Units                                |
| ------------------------------------------------------------- | ----------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `control.calculation`                                         | string                        | `"self_consistent_field"` | Run type; shorthands `scf`/`nscf`/`bands`/`relax`/`dos` map to the canonical `self_consistent_field`/`non_self_consistent_field`/`band_structure`/`relaxation`/`density_of_states`; `interpolation` marks an input for `q3_utility interpolate` (§2.4), which the engine refuses like the Python-orchestrated types                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | —                                    |
| `control.outdir`                                              | string                        | `"./"`                    | Output directory                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | —                                    |
| `control.prefix`                                              | string                        | `"q3"`                    | Output filename prefix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | —                                    |
| `control.verbosity`                                           | string                        | `"default"`               | How much the run prints. `"default"` reports the physics and the SCF trajectory; `"high"` adds the per-iteration and per-phase diagnostics (Davidson micro-iterations, timing breakdowns, mixer internals), which are logged at debug level. An explicit `--log-level` overrides it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | —                                    |
| `control.disable_symmetry`                                    | bool                          | `false`                   | Use no symmetry at all: no k-point/IBZ reduction, and no symmetrization of the density, forces or stress (QE `nosym` + `noinv`); a shifted mesh then gives the raw-mesh energy rather than the symmetrized-mesh one                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | —                                    |
| `control.nonlocal_stress_derivative_method`                   | string                        | `"analytical"`            | `"analytical"` (differentiate the Lagrange interpolation polynomial, matches QE) or `"numerical"` (finite-difference)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | —                                    |
| `control.number_of_kpoint_pools`                              | int                           | `0`                       | MPI k-point-pool count. Zero selects the count automatically (chosen automatically: the largest divisor of the rank count that does not exceed the irreducible k-point count), and refuses to leave ranks sharing a pool. An explicit `--pools` value overrides this field                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | —                                    |
| `control.use_deterministic_fft_planning`                      | bool                          | `false`                   | `false` = FFTW_MEASURE (faster, per-process-timed, can vary run-to-run); `true` = FFTW_ESTIMATE (bit-reproducible)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | —                                    |
| `control.use_exact_wavefunction_grid`                         | bool                          | `false`                   | `false` = the 2×ecutwfc wavefunction FFT grid: faster, but approximate (V·ψ and the band density alias), which is harmless where the wavefunctions have little weight near the cutoff and up to mRy off pw.x on hard pseudopotentials at low cutoffs; `true` = the exact grid, sized for 4×ecutwfc as pw.x sizes it, at 1.4-2.3× the solver time on the silicon and copper examples. QE-format input always uses the exact grid                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | —                                    |
| `control.band_path`                                           | list                          | `[]`                      | High-symmetry k-path for band structure — see §1.6.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | —                                    |
| `control.band_path_points_per_segment`                        | int                           | `20`                      | k-points sampled per band-path segment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | —                                    |
| `control.kpoints_list`                                        | list of `[kx,ky,kz,weight]`   | `[]`                      | Explicit k-point list for NSCF; takes precedence over `kgrid` when non-empty                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | crystal coords; weight dimensionless |
| `control.dos_broadening`                                      | float                         | `0.02`                    | Gaussian broadening for DOS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Ry                                   |
| `control.dos_energy_min` / `dos_energy_max`                   | float                         | `-100.0` / `100.0`        | DOS energy grid bounds                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Ry                                   |
| `control.dos_energy_steps`                                    | int                           | `2000`                    | DOS energy grid point count                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | —                                    |
| `control.compute_projections`                                 | bool                          | `false`                   | Opt-in atomic-orbital (ortho-atomic/Löwdin) projection pass during NSCF                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | —                                    |
| `control.interpolation`                                       | object                        | —                         | Band-structure interpolation from a small model (tutorial 08). On an NSCF run the block asks the engine to build the model and save it as `<prefix>.save/interpolation.q3ip`; on `calculation: interpolation` it names the saved model that `q3_utility interpolate` evaluates. Every key below belongs to exactly one of those steps and is **refused** on the other rather than ignored: `grid`, `bands`, `projection_centers`, `projection_decay_bohr`, `trial_shells` and `export_formats` are read when the record is captured; `dos_mesh`, `disentanglement`, `disentanglement_maximum_iterations`, `disentanglement_convergence_tolerance`, `localization`, `localization_maximum_iterations`, `projectability_minimum`, `projectability_maximum` and `frozen_window_above_fermi_ev` are read when the model is evaluated                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | —                                    |
| `control.interpolation.method`                                | string                        | (required)                | How the model is built; `wannier` (Wannier functions in the one-shot projection gauge) is the only method today                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | —                                    |
| `control.interpolation.grid`                                  | `[n1, n2, n3]`                | (required on NSCF)        | The full, unreduced k-point grid `kpoints_list` enumerates (Wannier construction needs every k-point's neighbours, so a symmetry-reduced grid cannot be used); stated explicitly because an explicit list carries no grid shape of its own                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | —                                    |
| `control.interpolation.bands`                                 | `[first, last]`               | lowest bands              | The 1-based inclusive band range the model is built from. Absent means the lowest bands, which is what an isolated valence manifold needs. Give a range when the manifold does not start at band 1, as it does not when the pseudopotential carries semicore states below it. The range must hold exactly as many bands as there are trial orbitals, and the run must compute at least `last` bands. This is band SELECTION, not disentanglement: the chosen range must still be separated from what surrounds it at every k-point                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | —                                    |
| `control.interpolation.projection_centers`                    | list of `[c1,c2,c3]`          | `[]`                      | Wannier: centres of the s-type trial orbitals used for the initial gauge, one per Wannier function. Empty falls back to the pseudopotential's atom-centred pseudo-atomic orbitals, whose count then fixes the number of Wannier functions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | crystal coords                       |
| `control.interpolation.projection_decay_bohr`                 | float                         | `1.0`                     | Wannier: decay length _a_ of those hydrogenic 1s trial orbitals, g(r) ∝ exp(−\|r−τ\|/a); Wannier90's `zona` is 1/a                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | bohr                                 |
| `control.interpolation.trial_shells`                          | map of species to list of int | every shell               | Wannier, build step: per species, the 0-based indices of the pseudo-atomic shells (the file's `PP_CHI` entries, in order) to build trial orbitals from. A species not named keeps every shell its pseudopotential does not mark unusable (a negative occupation, or a zero-norm radial function). Refused together with `projection_centers`. `q3.workflows.wannier` always writes it: every occupied shell, and an empty one only where the atom has no occupied valence shell of the same l. That leaves out the second s and p channels a PAW or ultrasoft dataset lists beside its valence (silicon PAW: 8 trial orbitals per atom become 4) and keeps copper's empty 4p beside its semicore 3p. A shell is semicore when its principal quantum number is below the outermost occupied shell's, one less for d; the numbers come from the shell labels, or, on a file without them, from counting the element's Z − z_valence core electrons into closed subshells                                                                                                                                                                                                                                                                                                                                                                                                           | —                                    |
| `control.interpolation.export_formats`                        | list of string                | `[]`                      | Interoperability formats written beside the native record on the NSCF build step; `wannier90` writes `<prefix>.mmn`, `.amn`, `.eig` and `.nnkp` for `wannier90.x` and the tools built on it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | —                                    |
| `control.interpolation.dos_mesh`                              | `[n1, n2, n3]`                | none                      | Evaluation step: the dense uniform k-mesh the interpolated density of states is sampled on, binned with `dos_broadening`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | —                                    |
| `control.interpolation.disentanglement`                       | string                        | `"none"`                  | Evaluation step: how the N states the gauge is built from are chosen out of the band window. `none` requires the window to be exactly those N states, which is what an isolated manifold gives. `projectability` classifies every band by its projectability and energy, then chooses the subspace that varies most smoothly across the zone — what an entangled manifold, such as a transition metal's d and s bands, needs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | —                                    |
| `control.interpolation.localization`                          | string                        | `"auto"`                  | Evaluation step: how the gauge inside the chosen subspace is fixed. Minimising the Marzari-Vanderbilt spread makes the Wannier functions as localised as the mesh allows, so H(R) decays quickly enough to interpolate _between_ mesh points; the one-shot projection gauge is exact _at_ them whatever it does. `auto` minimises when the record carries the neighbour overlaps M_mn(k,b) the spread is defined through, and keeps the projection gauge when it does not — a record imported from another code's projections and eigenvalues alone carries none. `maximal` demands it, so such a record becomes an error. `none` always keeps the projection gauge, which isolates what localisation contributes. Refused on a mesh with a two-point axis, where k+b and k−b are the same point and the finite-difference Berry phase the centres are built on does not exist                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | —                                    |
| `control.interpolation.projectability_minimum`                | float                         | `0.01`                    | Evaluation step: a band whose projectability p = Σ\|A\|² falls below this is discarded from the window entirely, unless the frozen energy window kept it                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | —                                    |
| `control.interpolation.projectability_maximum`                | float                         | `0.95`                    | Evaluation step: a band whose projectability reaches this is frozen — kept exactly, never rotated by the optimisation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | —                                    |
| `control.interpolation.frozen_window_above_fermi_ev`          | float or `"auto"`             | `"auto"`                  | Evaluation step: bands at or below E*F plus this are frozen whatever their projectability, which is what keeps a metal's Fermi surface in the model — those states are the free-electron-like ones with the \_lowest* projectability. `auto` applies the published 2 eV for atomic projectors to metal and insulator alike — it is the window the band distance that judges the model is measured on, and the two must coincide (silicon at 12×12×12: 220 meV with the occupied manifold alone frozen, 14 meV at E_F + 2 eV, all of it the lowest conduction band); for an insulator the engine's Fermi level sits in the gap, so the window reaches the conduction states the distance weighs. Whether the system is a metal (a band crossing E_F), the window applied and the bands crossing E_F are reported under `interpolation.disentanglement` in the results. A number overrides the rule; a value below the whole manifold freezes nothing by energy. The singular-gauge rule stands either way: a band inside the window whose projectability is below `projectability_minimum` is one the trial orbitals cannot represent, so it is excluded from the freeze and listed as `bands_excluded_from_freeze` (band, k-point count) rather than frozen, because freezing it makes the gauge singular — copper's norm-conserving dataset, which has no 4p shell, is the case | eV                                   |
| `control.interpolation.localization_maximum_iterations`       | int                           | `3000`                    | Evaluation step: the spread minimiser's iteration budget. Far more than a well-posed case needs (copper converges in about sixty); the retry ladder in `q3.workflows.wannier` doubles it after a stall, which is the one reason it is a key                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | —                                    |
| `control.interpolation.disentanglement_maximum_iterations`    | int                           | `4000`                    | Evaluation step: the subspace iteration's budget. Exhausting it is an error that prints the last five gauge-invariant spreads, never a subspace returned as if it had converged                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | —                                    |
| `control.interpolation.disentanglement_convergence_tolerance` | float                         | `1.0e-10`                 | Evaluation step: the relative change in the gauge-invariant spread Ω_I, over five iterations, below which the disentangled subspace counts as converged. The retry ladder in `q3.workflows.wannier` loosens it for a disentanglement that exhausted its budget, to three times the relative change among the five spreads that failure printed: the iteration is deterministic, so the retry stops no later than the failure did, and the model it returns is judged by the same gate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | —                                    |
| `control.relax_max_steps`                                     | int                           | `100`                     | Maximum ionic-relaxation steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | —                                    |
| `control.relax_force_threshold`                               | float                         | `1.0e-3`                  | Max-force convergence threshold                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Ry/bohr                              |
| `control.relax_energy_threshold`                              | float                         | `1.0e-4`                  | Energy-change convergence threshold                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Ry                                   |
| `control.isolated_system_correction`                          | string                        | `"none"`                  | `"none"`, `"makov-payne"`, or `"martyna-tuckerman"` — corrects the spurious electrostatic self-interaction of a net-charged/dipolar periodic cell with its own images. `"makov-payne"` (QE's `assume_isolated='makov-payne'` analog): monopole+dipole+quadrupole multipole series, cubic lattices only (throws otherwise). `"martyna-tuckerman"` (QE's `assume_isolated='m-t'` analog, but see the note below): exact reciprocal-space kernel-difference sum, no lattice-type restriction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | —                                    |

**`martyna-tuckerman` is the post-SCF (non-self-consistent) evaluation only** —
it is evaluated once on the density already converged under the ordinary
periodic Coulomb kernel, unlike QE's `assume_isolated='m-t'`, which feeds the
corrected potential back into the SCF loop. See
`MartynaTuckermanCorrectionCalculator`'s class docstring
([layer_5_properties/isolated_system_correction/planewave/martyna_tuckerman.hpp](../src/layer_5_properties/isolated_system_correction/planewave/martyna_tuckerman.hpp))
for the full scope statement and formulas.

No YAML input field is expressed in kbar — that unit
(`RYDBERG_TO_KILOBAR`, `src/layer_0_core/types/constants.hpp:67-68`) is used
only for stress-tensor _output_.

#### 1.6.1 `control.band_path[]`

Two accepted entry forms: a bare scalar string label (e.g. `"Γ"`/`"G"`, `"X"`,
`"W"`, `"K"`, `"L"`), or a map — schema path `control.band_path[]` —
with `{label, coordinates}`:

| Path                      | Type    | Default   | Description                                           | Units          |
| ------------------------- | ------- | --------- | ----------------------------------------------------- | -------------- |
| `band_path[].label`       | string  | —         | High-symmetry point label                             | —              |
| `band_path[].coordinates` | float×3 | `[0,0,0]` | Explicit override instead of an internal label lookup | crystal coords |

Crystal coordinates are read against **Q3's own primitive vectors**, not a
paper's: QE's ibrav 2, 3 and 4 orientations for FCC, BCC and HEX and the
a/b/c/angle construction for the rest, all in
`src/layer_1_structure/crystal/lattice.hpp` and mirrored in
`q3.structure.lattice`. This matters when writing `coordinates` by hand,
because the tabulated fractional triples for a labelled point differ between
primitive sets for the same physical lattice — Setyawan & Curtarolo's BCC
values, for instance, belong to what `lattice.hpp` calls ibrav=-3 and name
interior points when read as ibrav=3. The label table in `q3.workflows.kpath`
is already expressed in Q3's basis.

---

## 2. Command-line flags

Both C++ calculation entry points (and the GPU binaries) share one
hand-rolled parser, `q3::cli::parse_options()` (`src/io/cli.hpp:42-64`) — no
CLI library, plain `argv`/`argc` string comparison. `q3_utility` (§2.4) is the
exception: its subcommands take a save directory or an evaluation manifest and
parse their own arguments, so none of the structural rules below apply to it.

**Structural rules that apply to every calculation binary:**

-   **Flags and the input path may appear in any order.** The first argument
    that is neither a flag nor a flag's value is the input path, so
    `q3 --format qe scf.in` and `q3 scf.in --format qe` are equivalent.
-   **The input path is optional only where a build-time add-on's input format
    recognizes the working directory** as a run directory of its own (§3).
    Otherwise, and always in a build without one, usage prints and the binary
    exits 1.
-   **A flag with no value is an error**, not a silent drop
    (`--log-level needs a value`), and so is an **unrecognized `--flag`** or a
    second positional argument. Swallowing either used to mean `q3 --typo deck/`
    consumed `--typo` as the path and discarded the deck.
-   There is no `--help`/`-h` flag; usage prints on a malformed invocation.
-   Every flag is syntactically accepted by every binary (`parse_options()` is
    shared verbatim), but not every binary _acts on_ every flag — see the
    per-binary tables below.

### 2.1 `q3` (serial)

| Flag                      | Argument                                                    | Effect                                                                                                                                                                                                                                                                                                                               |
| ------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| _(positional)_ input path | optional                                                    | A YAML or QE file, or what a build-time add-on's input format accepts (§3). Omitted, it resolves only where an add-on format recognizes the working directory                                                                                                                                                                        |
| `--format <name>`         | `qe` \| `yaml`, plus any name a build-time add-on registers | Forces the parser instead of detecting from the path; an unrecognized name is an error that lists the supported ones                                                                                                                                                                                                                 |
| `--check`                 | none                                                        | Reads the input, reports what it resolved to (calculation, functional, cutoffs, spin, atoms, k-grid, DFT+U, dispersion, isolated-system correction, symmetry) and exits 0 without running. Exits 1 with the parser's own message on anything it refuses — the cheap way to ask whether Q3 will read someone else's deck, and as what |
| `--material <file>`       | ESSE material JSON with `lattice.vectors`                   | Substitutes the input's structure (cell, positions), as a YAML `material` key would; settings, species, atom order and constraints stay the input's. How `q3.workflows` runs each geometry of a relaxation on any binary. Refused when the atom count differs                                                                        |
| `--log-level <level>`     | one of spdlog's `trace`/`debug`/`info`/`warn`/`error`/`off` | Applied after `log::init()`                                                                                                                                                                                                                                                                                                          |
| `--log-file <path>`       | path                                                        | Also writes the log to this file, as a second sink at `debug` level, so the file keeps records the console drops. Truncated on open                                                                                                                                                                                                  |
| `--version`               | none                                                        | Prints the binary name, version and build commit, then exits 0; a binary built with add-ons adds a second line, `add-ons:` followed by each one's name and version. Answered before the input is resolved, so it works where no deck exists. The commit is empty when the build came from a modified tree or from no repository      |
| `--pools <n>`             | int                                                         | **Parsed, never read** — the serial binary has no pool/topology concept                                                                                                                                                                                                                                                              |
| `--precision <mode>`      | string                                                      | **Parsed, never read**                                                                                                                                                                                                                                                                                                               |

### 2.2 `q3_mpi`

| Flag                      | Argument                                   | Effect                                                                                                                                                   |
| ------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _(positional)_ input file | required                                   | Parsed after MPI init and before pool-topology configuration                                                                                             |
| `--log-level <level>`     | spdlog level                               | Applied after `log::init_mpi()`                                                                                                                          |
| `--log-file <path>`       | path                                       | Rank 0 writes `<path>`; rank N writes `<path>.N`. Silencing a non-root rank's console does not silence its file, which is the point of the per-rank name |
| `--version`               | none                                       | Prints the build identity from the root rank and exits 0                                                                                                 |
| `--pools <n>`             | int, must evenly divide the MPI world size | Sets the k-point pool-group count                                                                                                                        |
| `--precision <mode>`      | string                                     | **Parsed, never read** by `q3_mpi`                                                                                                                       |

Omitting `--pools` honors `control.number_of_kpoint_pools`; zero or an omitted
input value selects the count **automatically** — the largest divisor of the
rank count that does not exceed the irreducible k-point count, so ranks are
spread over k-points rather than sharing a pool. If ranks would still have to
share one, the run stops rather than distributing the FFT by default: that path
has no measured speedup and disables gamma-only, stress, checkpointing and
export, so it has to be asked for. An explicit CLI value takes precedence and is
honored as given. The requested count must divide the MPI world size and is
clamped to the number of available IBZ k-points so no pool is empty. Parsing and topology validation are inside the
entry point's normal exception boundary, so invalid values produce the same
fatal-error path as other input errors.

### 2.3 GPU binaries (`q3_gpu`, `q3_gpu_cuda`, `q3_gpu_hip`, `q3_gpu_sycl`)

All four share `q3::cli::run_gpu_entry_point()` (`src/io/gpu_entry_point.hpp:123-216`).

| Flag                      | Argument                | Effect                                                                                               |
| ------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| _(positional)_ input file | required                | Same dispatch as above                                                                               |
| `--log-level <level>`     | spdlog level            | Applied after logger init                                                                            |
| `--log-file <path>`       | path                    | As for the serial binary                                                                             |
| `--version`               | none                    | Prints the build identity and exits 0                                                                |
| `--precision <mode>`      | `"double"` or `"mixed"` | `"mixed"` enables mixed FP32/FP64 GPU precision; anything else non-empty prints an error and exits 1 |
| `--pools <n>`             | int                     | **Parsed, never read**                                                                               |

### 2.4 `q3_utility` (utility subcommands over saved records)

`src/main_utility.cpp` dispatches; each subcommand is one class under
`src/commands/`. The binary consumes what a finished run left in its save
directory rather than an input manifest, and is serial by construction, so it
carries none of the `_mpi`/`_gpu` variants — those name where the engine runs,
and a science- or format-facing suffix beside them (the former `q3-wannier`
and `q3-convert`) mixed two axes on one name.

```
q3_utility [--log-level <level>] interpolate <input.yml> [--output <file>]
q3_utility [--log-level <level>] convert <save-directory> [--format <name>] [--outdir <directory>]
q3_utility [--log-level <level>] convert --from wannier90 --seedname <s> --grid <nx> <ny> <nz> [--outdir <directory>] [--prefix <p>]
q3_utility [--log-level <level>] volumetric <input.yml> [--field <name>] [--format <name>] [--spin <n>] [--kpoint <k>] [--band <n>] [--outdir <directory>] [--output <file>]
```

**`interpolate`** evaluates a saved interpolation record (§4.9). The input is
an ordinary manifest with `calculation: interpolation` and a
`control.interpolation` block; its `control.outdir`/`prefix` name the run
whose NSCF saved the record, `control.kpoints_list` (or a `band_path` whose
vertices carry coordinates, walked `band_path_points_per_segment` steps per
segment) names the k-points, and `control.interpolation.dos_mesh` with
`dos_broadening`/`dos_energy_steps` asks for a density of states. Labels
without coordinates are refused — the high-symmetry table lives in
`q3.workflows.kpath`, and `q3.workflows.interpolation` writes the explicit
list. Output: `<outdir>/<prefix>_interpolation_results.json`, with
`eigenvalues_ry` shaped `[k-point][spin][band]` like §4.1 so the Python
extractors read it unchanged, plus `kpoints`, `fermi_energy_ry`,
`density_of_states` (`energies_ry`, `dos`, `mesh`, `broadening_ry`), an
`interpolation` block (method, model size, the Wigner-Seitz weight identity as
`total_bloch_sum_weight`, and the two blocks below) and the §4.1 `provenance`
stamp. `fermi_energy_ry` is the Fermi level the NSCF that built the record
computed over its own full-grid k-set — the highest occupied level when every
requested band is occupied, as in the silicon example;
`q3.workflows.interpolation` reports the SCF's value instead, which is the
authoritative one.

`interpolation.spread` is the Marzari-Vanderbilt spread of the model's
Wannier functions at the gauge it was built with, in bohr and bohr squared:
`gauge_invariant_bohr_squared` (Ω_I, always), and when the record carries the
neighbour overlaps that define the rest, `total_bohr_squared`, `centers_bohr`,
one `spreads_bohr_squared` per function (Ω_n = ⟨r²⟩\_n − |⟨r⟩\_n|², summing to
the total), `initial_minimizable_bohr_squared` (Ω_D + Ω_OD at the projection
gauge), the `guiding_centers` the phase branches were fixed by and the
`trial_orbital_centers_bohr` themselves; `localized` says whether the gauge
was minimised, and if so `localization_iterations`, `localization_converged`
(whether it reached a stationary gauge, not merely stopped) and
`localization_gradient_norm`.

`interpolation.quality` is what the model says about itself with no reference
calculation: `spread` (`median_bohr_squared`, `variance_bohr_squared` — the
published no-reference criterion, `maximum_bohr_squared`, the zero-based
`outliers` above five times the median and the functions
`above_ten_angstrom_squared`), `centers` (per function, the distance to the
nearest atom and to the nearest bond midpoint of the manifest's material over
periodic images, and the `shortest_bond_bohr` that defines a bond as anything
within 20 % of it), `hamiltonian_decay` (the largest Bloch-sum term
w_R·|H_mn(R)| on the outermost Wigner-Seitz shell, `outermost_shell_maximum_ry`,
against the `largest_hopping_ry` anywhere — any R ≠ 0, or m ≠ n on site, so a
rigid shift of the energy zero cannot move it — and their `ratio`; a sum that
has not decayed by its last term is the direct cause of error between mesh
points), and `localization_converged` again when the gauge was minimised. The
band distance η against a direct calculation, and the `status` that decides
whether the model is kept, are added to this block by
`q3.workflows.wannier`, which runs the calculation they need.

The file is described once, in `schemas/q3/interpolation_results.json`: the
C++ writer is generated from it (`src/generated/q3/interpolation_results.hpp`,
filled by `io/interpolation/results_document.hpp`), and the Python reader,
`q3.io.interpolation_results`, mirrors it block for block under a parity test
(`tests/python/test_interpolation_results.py`). `required` there is the
producer's guarantee, not a reader's precondition — every Python field defaults
to `None`, and the keys `q3.workflows.wannier` adds afterwards
(`quality.status`, `reason`, `ladder`, `band_distance`) are optional in the
same schema, so one document describes the file before and after the workflow
annotates it. The `control.interpolation` block a manifest carries is typed the
same way on the Python side — `q3.io.interpolation_settings.CaptureSettings`
for the NSCF that builds a record and `EvaluationSettings` for what
`interpolate` reads — each writing only the keys it sets and holding the
schema's bounds, so a block a model wrote is one the parser accepts;
`tests/python/test_interpolation_settings.py` validates those blocks against
the schema itself and holds the schema's defaults to `specification.hpp`'s.

A failed evaluation prints its reason and, on the line after it, a code a
driver can route on without reading the prose:

```
Error: disentanglement did not converge in 4000 iterations. Last gauge-invariant spreads (bohr^2): 32.869717, ...
diagnostic: disentanglement_unconverged
diagnostic_spreads_bohr_squared: 32.869717, 32.869701, 32.869684, 32.869668, 32.869651
```

The codes are the enumerators of `WannierDiagnostic`
(`src/layer_4_computation/interpolator/method/wannier/diagnostic.hpp`):
`disentanglement_unconverged`, `disentanglement_unsupported`,
`frozen_window_overfull`, `kept_bands_underfull`, `gauge_singular`,
`gauge_collapsed`, `overlaps_not_orthonormal`, `localization_unsupported` and
`two_point_axis`; the third line appears only for the unconverged case, with
the same six decimals as the prose. `q3.workflows.wannier.diagnostic.Diagnostic`
spells the same names and a test holds the two lists equal. The retry ladder
routes on the code and on nothing else: a failure without one is unknown to
it, whatever its sentence says, and the spreads it loosens a tolerance from
are read off the third line, never out of the prose. `q3.binary` reads the
code into `Q3RuntimeError.diagnostic` beside the command line and the output
it kept.

**`convert`** is Q3's analogue of QE's `pw2bgw.x` and `pw2wannier90.x` in one
place: it reads a record a finished run left behind and drives the same writer
code the in-process export uses, so a second format, or a re-export after a
format fix, costs a file read instead of a re-run.

| Argument                         | Default                                      | Effect                                                                                                                                                      |
| -------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _(positional)_ save directory    | required (export)                            | A run's `<outdir>/<prefix>.save/`, holding `mean-field.q3mf` (§4.8) and/or `interpolation.q3ip` (§4.9)                                                      |
| `--format <name>`                | the one format the directory's record allows | `berkeleygw` from the mean-field record, `wannier90` from the interpolation record; a directory holding both records needs the flag                         |
| `--outdir <directory>`           | the save directory's parent (`.` on import)  | By default the files land exactly where the in-process export would have put them                                                                           |
| `--from wannier90`               | —                                            | Import direction: `<seedname>.amn/.eig/.nnkp` on `--grid nx ny nz` become `<outdir>/<prefix>.save/interpolation.q3ip` (overlaps are not imported, see §4.9) |
| `--seedname <s>`, `--prefix <p>` | prefix defaults to the seedname's basename   | Import only                                                                                                                                                 |
| `-h` / `--help`                  | —                                            | Prints usage and exits 0 — **unlike** the calculation binaries, which have no help flag                                                                     |

Exits 1 with a named cause, not a stack trace, when a record is missing (the
message names the control key that produces it), when a file is not the record
it claims to be, or when its version is one this build does not know.

**`volumetric`** turns a saved field into a file a visualizer reads — Q3's
analogue of QE's `pp.x`. It takes the manifest the run was launched with rather
than a directory, because the save directory holds the numbers but not the cell
they belong to: `control.outdir`/`prefix` say where to read, and the cell and
cutoffs say what grid the saved arrays are on. The output is §4.10.

| Argument               | Default              | Effect                                                                                                                                      |
| ---------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| _(positional)_ input   | required             | The run's own manifest                                                                                                                      |
| `--field <name>`       | `density`            | `density` (from `charge-density.dat`), `potential` (from `effective-potential.q3cp`), `wavefunction` (from `wfc<k>.dat`, written as \|ψ\|²) |
| `--format <name>`      | `cube`               | `cube` or `xsf`, plus any format a build-time add-on registers (§4.10)                                                                      |
| `--spin <n>`           | 1                    | Which spin channel, 1-based                                                                                                                 |
| `--kpoint <k>`         | 1                    | Which `wfc<k>.dat`, 1-based over the run's irreducible wedge                                                                                |
| `--band <n>`           | 1                    | Which band of it, 1-based                                                                                                                   |
| `--outdir <directory>` | the run's own outdir | Where to write                                                                                                                              |
| `--output <file>`      | —                    | The full path, overriding `--outdir` and the default name                                                                                   |

The log line after a successful write reports the field's integral over the
cell, which for a density is the valence electron count and for a band's
probability density is 1 — the cheapest check that a file is what it says.

CMake targets: `q3` (default on), `q3_utility` (default on),
`q3_mpi` (`Q3_WITH_MPI`), `q3_gpu` (`Q3_WITH_GPU`), `q3_gpu_cuda`
(`Q3_WITH_CUDA`), `q3_gpu_hip` (`Q3_WITH_HIP`), `q3_gpu_sycl`
(`Q3_WITH_SYCL`).

---

## 3. Input formats from build-time add-ons

Beside YAML and the QE namelist, a build-time add-on — a directory under `src/addons/`,
compiled in when it exists ([architecture.md](architecture.md)) — can register an input
format of its own. Each registered format is asked in a fixed priority order whether it
recognizes the path a binary was given; the first that does parses it, YAML takes any file
no other format claims, and `--format` names a format outright. A format may also
recognize a working directory as a run directory of its own, which is what lets a binary
run with no input path at all (§2). Whatever the format, the result is the same
`InputData`, so nothing downstream knows which front end read it: `parse_input()`
(`src/io/input/parser/factory.hpp`) is the one funnel every entry point, the Python
bindings and `q3_utility` read through.

After a run, core hands its companion-output step to the format that parsed the input
(§4.5). A build without add-ons registers no other format: `--format` then accepts `yaml`
and `qe`, and the error for any other name lists exactly those.

---

## 4. Output artifacts

### 4.1 Main results file — `<outdir>/<prefix>_results.json`

Written by `FlavorResult::write_results_json()`
(`src/layer_4_computation/flavors/result.hpp:128`) at the end of every SCF or
NSCF run (converged or not), called from `main.cpp`, `main_mpi.cpp` (root
rank only), and the GPU entry points. Top-level keys:

| Key                                                                                                                                                                                | Contents                                                                                                                                                                                                                                                                                                  | Units                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `flavor`                                                                                                                                                                           | Calculation flavor name                                                                                                                                                                                                                                                                                   | —                                                                                                                                                                                                    |
| `converged`                                                                                                                                                                        | SCF convergence flag                                                                                                                                                                                                                                                                                      | —                                                                                                                                                                                                    |
| `self_consistent_field_iterations`                                                                                                                                                 | SCF iterations the run performed: the one it converged on, or all of `provenance.precision.maximum_self_consistent_field_iterations` when it did not. Present only for flavors that iterate to self-consistency                                                                                           | —                                                                                                                                                                                                    |
| `units`                                                                                                                                                                            | `{"energy": "Ry", "forces": "Ry/bohr", "stress": "kbar", "eigenvalues": "Ry"}` — self-describing unit map for every field below                                                                                                                                                                           | —                                                                                                                                                                                                    |
| `execution`                                                                                                                                                                        | Actual process, pool, and band-group counts after any IBZ clamp, `number_of_threads_per_process` (OpenMP threads each process may use), whether G-vectors are distributed, and `peak_resident_memory_bytes` -- the largest peak resident set over the ranks, or zero where the platform offers no reading | —                                                                                                                                                                                                    |
| `material`                                                                                                                                                                         | The structure the run used, as an ESSE material in the engine's own frame: lattice vectors in angstrom; basis in crystal coordinates, with `constraints` (`true` = free, from `if_pos`). `q3.workflows` starts a relaxation from it                                                                       |
| `total_energy_ry`, `band_energy_ry`, `hartree_energy_ry`, `exchange_correlation_energy_ry`, `ewald_energy_ry`, `double_counting_correction_ry`, `smearing_entropy_contribution_ry` | Energy breakdown                                                                                                                                                                                                                                                                                          | Ry                                                                                                                                                                                                   |
| `hubbard_energy_ry`, `one_center_energy_ry`                                                                                                                                        | Optional nonzero DFT+U and PAW one-center energy components                                                                                                                                                                                                                                               | Ry                                                                                                                                                                                                   |
| `isolated_system_correction`                                                                                                                                                       | Nested object, present only if `control.isolated_system_correction` is not `none`: `method` plus the scheme's labeled terms (Makov-Payne: `first_order_ry`, `second_order_ry`; Martyna-Tuckerman: single `energy_ry` term), `total_ry`, and `corrected_total_energy_ry`                                   | Ry                                                                                                                                                                                                   |
| `fermi_energy_ry`                                                                                                                                                                  | Fermi energy                                                                                                                                                                                                                                                                                              | Ry                                                                                                                                                                                                   |
| `forces_ry_per_bohr`                                                                                                                                                               | Per-atom Cartesian forces, `[N_atoms][3]`                                                                                                                                                                                                                                                                 | Ry/bohr                                                                                                                                                                                              |
| `forces_are_incomplete`, `forces_incomplete_reason`                                                                                                                                | bool, string                                                                                                                                                                                                                                                                                              | Present **only** when Q3 left a term out of the forces (today: never). The ASE calculator refuses to hand back forces carrying this, so a relaxation stops instead of converging to a wrong geometry |
| `stress_is_incomplete`, `stress_incomplete_reason`                                                                                                                                 | bool, string                                                                                                                                                                                                                                                                                              | The same for stress (today: a multi-rank band group whose distributed stress path is not implemented)                                                                                                |
| `stress_kbar`                                                                                                                                                                      | 3×3 stress tensor                                                                                                                                                                                                                                                                                         | kbar                                                                                                                                                                                                 |
| `eigenvalues_ry`                                                                                                                                                                   | `[k][spin][band]`                                                                                                                                                                                                                                                                                         | Ry                                                                                                                                                                                                   |
| `occupations`                                                                                                                                                                      | `[k][spin][band]`, present only if populated                                                                                                                                                                                                                                                              | — (fraction, not spin-degeneracy-multiplied)                                                                                                                                                         |
| `kpoint_weights`                                                                                                                                                                   | Per-k-point IBZ integration weight, normalized to sum to 2.0                                                                                                                                                                                                                                              | —                                                                                                                                                                                                    |
| `orbital_projections`, `orbital_labels`                                                                                                                                            | Ortho-atomic/Löwdin projections, present only if `control.compute_projections: true`                                                                                                                                                                                                                      | —                                                                                                                                                                                                    |
| `provenance`                                                                                                                                                                       | How the result was produced: `generator` (`engine_name`, `engine_version`, `engine_commit`, `linear_algebra_libraries` -- the BLAS/LAPACK the build linked, `addons`), `model`, `method` and `precision`; `provenance.generator.addons` is described below the table                                      | —                                                                                                                                                                                                    |

**`provenance.generator.addons`** lists one `{"name": ..., "version": ...}` object per
build-time add-on compiled into the binary (§3), the same ones `--version` prints on its
second line. It is always present, and an empty list for a build without add-ons. A
parity comparison between two builds treats it as volatile, like the version and commit
beside it.

`FlavorResult::to_json_string()` produces byte-for-byte the same schema for
the nanobind Python binding's in-memory return path.

**Canonical name:** every CLI and nanobind entry point writes
`<outdir>/<prefix>_results.json`. The Python helper
`q3.binary._find_results_json()` prefers that name and accepts the older
unprefixed `<outdir>/results.json` only as a compatibility fallback for files
written by bindings released before this contract was unified.

### 4.2 Binary SCF checkpoint set — `<outdir>/<prefix>.save/`

Written at the end of an SCF run whenever every process holds the k-points it
works on whole: the serial binaries, and `q3_mpi` or `q3_mpi_gpu_cuda` with one
rank per k-point pool, which is what the automatic topology chooses. Under
pools each rank writes the `wfc<N>.dat` files of its own k-points, named by
their index in the global grid, and rank 0 writes the other two files: the
density and potential are reduced across pools, so it holds them whole, and the
eigenvalues are gathered into global order first. The directory is the one a
serial run writes, and a serial NSCF restores it the same way. Ranks that share
a pool divide each k-point's coefficients and the density grid, and the writers
have no gather for either, so that layout refuses before creating
`<prefix>.save/`; the refusal is logged and `results.json` is still written. A
fourth file, `mean-field.q3mf`, lands here only when a mean-field export was
requested — see §4.8.

| File                                 | Format                                     | Contents                                                                                                                                                                                                                                                                                             |
| ------------------------------------ | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `charge-density.dat`                 | QE Fortran-unformatted binary              | `ρ(G)` on the smooth grid: header (gamma_only flag — **always written 0**, G-vector count, spin count), Miller indices, complex ρ(G)                                                                                                                                                                 |
| `wfc<N>.dat` (one per k-point state) | QE Fortran-unformatted binary              | k-point header (index, coordinates, spin, the actual gamma-only flag, scale factor — the aligned coordinates start at byte offset 8), dimensions, reciprocal lattice vectors, Miller indices, one record per band of complex coefficients                                                            |
| `effective-potential.q3cp`           | Q3-native binary (magic `Q3CP`, version 3) | Header + real-space ρ and V_eff per spin channel, then (version 3) the PAW on-site occupations ρ_ij per spin channel, empty for norm-conserving and ultrasoft runs; SCF-iteration/energy/convergence header fields are left at zero for this specific artifact — only the arrays are meaningful here |

`effective-potential.q3cp` is read back by NSCF runs to obtain a frozen
potential without re-running SCF (`non_self_consistent_field.hpp`) — this is
how the band-structure/DOS/PDOS Python workflows work. A PAW run also reads
ρ_ij, which D's one-centre term is built from, and refuses a checkpoint
without them (one written before version 3); rerun the SCF to regenerate it.

The test reader uses the same aligned offsets and has a non-Γ fixture that
would fail if the coordinates or gamma-only flag regressed.

### 4.3 Mid-SCF restart checkpoint (retired)

`control.checkpoint_interval` and its write-only `q3_checkpoint.bin` producer
were removed because no restart path consumed the file. The YAML parser now
rejects the retired key as unknown. A future mid-SCF restart feature must ship
its writer and tested load path together.

### 4.4 Workflow-level outputs (Python layer)

These calculation types have no dedicated on-disk schema of their own — they
reuse the `<prefix>_results.json`/`.save` artifacts from an underlying
SCF/NSCF run and reshape the result in Python, in memory:

-   **Band structure** (`q3.workflows.band_structure`): runs SCF, then an NSCF
    step along the interpolated k-path that **reuses the SCF's
    prefix/outdir** (needed to find `effective-potential.q3cp`) — this means
    the NSCF step's results.json **overwrites** the SCF run's results.json.
    The workflow captures the SCF results in memory first; anyone driving the
    two steps manually should copy the SCF results.json out before running
    NSCF.
-   **Density of states** (`q3.workflows.dos`): Gaussian-broadens eigenvalues
    from an existing results.json, entirely in Python; nothing new written.
-   **PDOS** (`q3.workflows.pdos`): SCF + NSCF-with-`compute_projections`, same
    overwrite caveat as band structure; projection data lives in the results
    JSON's `orbital_projections`/`orbital_labels` keys (§4.1).
-   **Wannier interpolation** (`q3.workflows.interpolation`): SCF, a full-grid
    NSCF that saves the record (§4.9), then `q3_utility interpolate` (§2.4); the
    grid comes from a 0.2 Å⁻¹ spacing when none is given. **The automatic
    protocol** (`q3.workflows.wannier.run_automatic_wannier_workflow`, the
    entry point in `q3.workflows.wannier.driver` over `WannierProtocol` in
    `.protocol`) takes its trial orbitals from the datasets' own shells
    (`q3.io.pseudo_atomic_shells` reads them, `TrialOrbitals` chooses), adds a
    reference NSCF at k-points half a
    mesh step off the mesh (or along a path), writes the band distance η, a
    `status` (`accept` / `reject`), the `reason` and the `ladder` history into
    the results file's `interpolation.quality` block through
    `q3.io.interpolation_results`, and re-runs `q3_utility interpolate` for every
    rung of the retry ladder (`q3.workflows.wannier.rungs`: one class per
    rung, walked in order over a frozen `LadderState`) against the same
    record — so the results file holds the LAST evaluation, and the manifests
    it wrote are temporary files beside the SCF manifest, deleted afterwards.
    The protocol returns a `WorkflowResult`; the entry point returns it as
    a dict of fifteen keys.
    **The campaign** (`scripts/validation/run_wannier_campaign.py`, the
    `wannier_campaign/` package beside it) writes one directory per structure
    (`material.json`, `scf.yml`, `output/`) plus `campaign.json` and
    `campaign.md`, updated after every structure; its full roster is a tier-3
    test the scheduled workflow runs monthly, keeping that table.
-   **Γ-point phonons** (`q3.workflows.phonons`): drives 6N finite-difference
    single-point SCF calculations; uses ASE's own `ase.phonons.Phonons` cache
    (`<workdir>/<cache_name>.*.json`, ASE-native format, not Q3-defined).
-   **Relaxation / variable-cell relaxation** (`q3.workflows.relax`,
    `.vc_relax`): ASE drives the optimizer; Q3 is a single-point
    energy/forces/stress evaluator per step.
    -   ASE trajectory file: `relax.traj` / `vc_relax.traj` (ASE-native ULM
        binary format).
    -   In subprocess mode (forced for vc_relax, since the bindings can't move
        the cell, and for any input that is not YAML), each step runs the
        input itself, in its own directory, with
        `--material <workdir>/step_NNNN/material.json`. Every step writes the input's own
        `<outdir>/<prefix>_results.json` and `.save/`, and starts from the
        density the previous step saved there. A non-YAML input (a pw.x deck,
        any `ibrav`) is first run once as written; its results' `material`
        (§4.1) is the starting structure, and that run is the first step.
    -   An input whose calculation is a relaxation (`relax`, `vc-relax`,
        `relaxation`), run by a binary directly, runs one SCF at its own
        geometry, from a saved density when one fits, and logs that it did.
    -   In bindings mode (the default when `q3_core` is importable), no `.save`
        checkpoint is written at all — the solver keeps density/state cached
        in-process across successive `solve()` calls instead.

### 4.5 Companion outputs from build-time add-ons

An input format a build-time add-on registers (§3) may write companion files next to the
results JSON after a run it parsed: the files that the tools built around that format
expect to find. Core calls that step once per run, on the root rank, after the results
JSON is written, and only on the format that parsed the input, so a YAML or QE-format run
leaves no such files behind. The results JSON stays the authoritative record of the run.

### 4.6 Logs and diagnostics

All SCF progress, convergence, and profiling output is emitted only via the
logger to stdout/stderr; no CSV or structured timing-history file exists
anywhere in the codebase.

What a default run prints is the physics and the SCF trajectory: the setup
summary, one line per SCF iteration, the convergence notice, eigenvalues,
final energies, forces, stress, and a three-line counter close. The
diagnostics — Davidson micro-iterations, the per-iteration and per-phase
timing breakdowns, the memory estimate, the mixer's residual bookkeeping, the
Davidson timing table — go to `log::debug` and appear only under
`control.verbosity: high` or an explicit `--log-level debug`. Neither the
numbers nor the results file change with the setting.

`--log-file <path>` adds a file sink alongside the console. It is given a
lower threshold than the console on purpose: a file is asked for when the
console has already proved too terse, so it carries the debug records whatever
`control.verbosity` says.

### 4.7 BerkeleyGW mean-field export — `<outdir>/`

Opt-in via `control.mean_field_export: berkeleygw` (default `"none"`, a
no-op); written into `outdir` from both the `self_consistent_field` and
`non_self_consistent_field` flavors (BerkeleyGW normally consumes an NSCF
run with many empty conduction bands, but the export works from either).
The same files can be produced after the fact from a saved run with
`q3_utility convert` (§2.4, §4.8) rather than by re-running the SCF. Framing
throughout is QE's Fortran unformatted sequential convention
(`QEBinaryRecordFormat`, `[int32 len][data][int32 len]`), matching QE's own
`pw2bgw.f90` converter byte for byte for the records it implements.

**Not written under multi-rank `q3_mpi`** — refused with a logged error
rather than written wrong. Every multi-rank topology distributes the state
the exporter would serialize (k-point pools each hold only their own
k-points; band groups hold only a G-vector slab of each band), and there is
no gather step, so no rank sees the complete mean field
(`export_and_record_mean_field_under_topology()`,
`io/mean_field_export/record_and_export.hpp`). A single-rank `q3_mpi` run
exports normally. The checkpoint asks less of the topology than the export
does (§4.2), so a pooled SCF still leaves one behind, and a serial
`non_self_consistent_field` run that restores it can write the export. Pinned
by `tests/validation/test_mpi_mean_field_export.py`.

| File      | Format                     | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WFN`     | Fortran-unformatted binary | Header (title/dimensions/lattice/symmetry/atoms) + per-k-point plane-wave counts, weights (normalized to sum to 1), crystal-coordinate k-points, occupied-band bounds, eigenvalues (Ry) and occupations, the charge-density G-vector set, then per k-point: Miller indices and per-band (per-spin) complex coefficients. Always written as flavor 2 ("WFN-Complex") — see `wavefunction.hpp`.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `RHO`     | Fortran-unformatted binary | Same header minus the k-point-specific fields, then the charge-density G-vector set and ρ(G) per spin channel, scaled by the cell volume.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `vxc.dat` | ASCII                      | Exchange-correlation matrix elements `<psi_nk\|Vxc\|psi_nk>` (eV), one header line (`kx ky kz ndiag noffdiag`, crystal coordinates) then `ndiag` diagonal lines (`spin band Re Im`) per k-point — diagonal-only, so `noffdiag = 0` always. Computed as a G-space convolution over this band's own plane-wave coefficients and the already-available G-space Vxc, sidestepping the dense-grid wavefunction transform pw2bgw itself uses; see `exchange_correlation.hpp`'s class comment for the derivation. Supported for spin-unpolarized, non-gamma-only runs only — write_to_file() throws a clear error otherwise (caught and logged by the exporter so WFN/RHO are unaffected); see [TODOs.md](../TODOs.md) §4.1 for the remaining scope gaps (HDF5, spin-polarized, gamma-only, MPI coverage). |

`BerkeleyGWCommonHeader` (`io/mean_field_export/berkeleygw/header.hpp`) writes
the 8 records shared by WFN and RHO. Its `cell_symmetry` follows pw2bgw's
bracketing of the Bravais index (1 for hexagonal and trigonal R, 0 for every
other lattice) when the input named one. A free lattice (`bravais_lattice_index` 0) is classified from its vectors instead (`berkeleygw/cell_symmetry.hpp`).
That covers YAML `lattice.vectors`, a YAML lattice outside the cubic family, and
QE's `CELL_PARAMETERS`: two equal vectors at 120° or 60° with the third
perpendicular is hexagonal, and three equal vectors at one non-cubic angle is
trigonal R. Free lattices used to be refused, which shut out every non-cubic
crystal. A file whose writing fails part-way is removed rather than left
truncated.

The symmetry records follow `pw2bgw`'s conventions, checked operation by
operation against QE 7.5 + `pw2bgw` for wurtzite ZnO, diamond Si and trigonal
Te: `mtrx` is QE's `s`, the transpose of Q3's crystal-basis rotation R, and
`tnp` is 2π·(−R⁻¹τ), the translation of the inverse operation (QE stores
`ft = −τ`). Writing R and (R⁻¹)ᵀτ instead keeps the group but pairs rotations
with the wrong translations, which BerkeleyGW's epsilon reports as "G-vectors
are falling outside of the sphere in gmap".

`kgrid`/`kshift` come from the Monkhorst-Pack input when there is one. An
explicit `kpoints_list` carries no grid, so the header recovers it from the
points, as the numbers `pw2bgw` would be given as `wfng_nk`/`wfng_dk`: from the
list unfolded by the crystal's rotations and time reversal when that fills a
grid exactly (a full grid, or the symmetry-reduced wedge of a grid the rotations
preserve), else from the list itself when it is a full grid -- every point once,
equal weights -- such as the WFNq grid displaced by q0, whose displacement
becomes `kshift` in grid steps. Anything else (a band path, or the wedge of a
grid the rotations do not preserve, such as a shifted FCC grid) leaves 0 0 0
with a warning, since BerkeleyGW's epsilon refuses such a WFN ("kgrid out of
bounds in wfn"); list every point of the grid instead.

Three things are worth knowing when exporting from an NSCF run — the flavor
BerkeleyGW is normally fed from:

-   **`RHO` carries the converged SCF density, restored from the checkpoint**,
    not the density the NSCF pass itself recomputes. An NSCF run executes one
    turn of the shared per-iteration flow, which ends in a mixing step, so the
    iterator's density is a blend of the atomic-superposition initial guess and
    that pass's output — governed by `mixing.beta`, a parameter with no meaning
    for a non-self-consistent run. `NonSelfConsistentFieldFlavor` therefore
    overwrites the exported state's density (both the real- and G-space copies)
    with the checkpoint's before exporting, and fails loudly if the checkpoint
    carries none. This matches `pw2bgw`, whose `write_rhog` writes QE's _saved_
    charge density and treats recomputing it from the bands as a separate opt-in
    diagnostic (`calc_rhog`).
-   **`vxc.dat` is Vxc of that converged density.** The NSCF pass loads V_eff
    from the checkpoint whole and so never evaluates exchange-correlation
    itself; the flavor evaluates Vxc(G) once from the restored density (core
    charge included) for the export, as `pw2bgw` does from QE's saved density.
-   **`kw` is normalized by the actual weight sum**, so `sum(kw) = 1` as
    BerkeleyGW requires whatever scale the input used. `pw2bgw` instead relies on
    QE's invariant that `wk` sums to 2; Q3's generated grids share that
    invariant, but `control.kpoints_list` weights are taken from the input
    verbatim, so a list written on BerkeleyGW's own sum-to-1 convention would
    otherwise land on disk as `sum(kw) = 0.5`.

Note also that `WFN` is not bit-reproducible across runs: eigenvectors are
defined only up to a phase, and up to a unitary rotation within any degenerate
subspace, so coefficients can differ by O(1) between two runs of the same input
while the eigenvalues and the density agree to ~1e-15. For the same reason, an
element-by-element `vxc.dat` comparison against `pw2bgw.x` would need to
compare traces over degenerate subspaces rather than individual diagonal
elements.

HDF5 `WFN.h5`/`RHO.h5` (the format modern BerkeleyGW increasingly expects
instead of the original Fortran binaries) is not implemented — see
[TODOs.md](../TODOs.md) §4.1.

This is not a Q3-invented format: it is BerkeleyGW's own mean-field input
format, the same one Quantum ESPRESSO's `pw2bgw.x` (`PP/src/pw2bgw.f90`)
converter produces, so files written here are meant to be readable directly
by BerkeleyGW's `epsilon`/`sigma`/`kernel`/`absorption` executables and
inspectable with BerkeleyGW's own `wfn_rho_vxc_info.x` utility.

**The machine-checkable spec.** `tests/helpers_berkeleygw.py` is an
independent Python reader for these three files, derived directly from
`pw2bgw.f90` without consulting Q3's own writer — it is the executable
cross-check, not just documentation of one. It validates Fortran record
framing (leading/trailing length markers must agree, or it raises naming
the record), decodes the mixed-type and interleaved records, and returns
every header field under a spelled-out name. `tests/python/test_berkeleygw_format_reader.py`
unit-tests the reader itself against synthetic fixtures;
`tests/validation/test_berkeleygw_export.py` runs it against a real Q3
output directory and checks, among other things:

**Known limitation, stated plainly**: the reader's WFN _body_ parsing (not
its header parsing, which is fully general) only supports
`number_of_spin_components == 1` — `pw2bgw.f90`'s spin-polarized case
writes a structurally different, spin-interleaved coefficient record layout
that is not implemented here (it raises a clear error rather than
mis-parsing). This is expected to match Q3's initial (spin-unpolarized)
export; revisit if/when a spin-polarized mean-field export is added.

### 4.8 Mean-field record — `<outdir>/<prefix>.save/mean-field.q3mf`

Q3-native binary (magic `Q3MF`, version 1), written by
`io/mean_field_record/checkpoint.hpp`. This is the converter seam: it holds
everything a mean-field exporter needs — geometry, symmetry, the G-vector
set, k-points and weights, eigenvalues and occupations, ρ(G), V_xc(G), and
the wavefunction coefficients — as plain data (`MeanFieldRecord`,
`io/mean_field_record/init.hpp`), with no reference to the live SCF state.
Both producers of that data (a finished run, via `build_mean_field_record()`)
and both consumers (the in-process export and `q3_utility convert`) therefore meet
at one struct, and the format writers exist in exactly one copy.

**Written only when `control.mean_field_export` is not `"none"`.** The
record duplicates the wavefunctions that `wfc<N>.dat` already stores, so
writing it unconditionally would roughly double every run's output for users
who never convert anything; tying it to "the user asked for a mean-field
export" costs nothing to everyone else and needs no additional control key.
Failing to write it is logged rather than thrown — the export the user asked
for is already on disk by that point.

Units follow `Crystal`'s own conventions rather than the exported format's,
so that the record is a faithful snapshot and every format conversion
happens in one place (the writer): lattice vectors and Cartesian atom
positions in units of `alat`, reciprocal lattice vectors in `tpiba`,
energies in Rydberg.

Coefficients are written last, as one contiguous block, so a reader can seek
to a given k-point from the running offsets alone; `read_mean_field_record()`
returns a band source that keeps one k-point resident rather than the whole
wavefunction set. Every scalar is written component-by-component (never a
struct `memcpy`), so the file does not depend on host padding.

The record's guarantee — that converting a saved run reproduces the
in-process export **byte for byte**, excluding only the wall-clock timestamps
in record 1 of WFN/RHO — is gated end-to-end by
`tests/validation/test_mean_field_converter.py`; the round-trip itself is
unit-tested field-by-field in `tests/unit/io/mean_field_record/test_checkpoint.cpp`.

A natural follow-up is to teach the converter to read `wfc<N>.dat` and
`charge-density.dat` (§4.2) directly, which would retire the duplication
entirely. Readers for both now exist — `q3_utility volumetric` (§2.4, §4.10) is
built on them — so what remains is teaching the BerkeleyGW writers to source a
`MeanFieldRecord` from them rather than from the duplicated copy.

---

### 4.9 Interpolation record — `<outdir>/<prefix>.save/interpolation.q3ip`

Written by every NSCF run that carries a `control.interpolation` block
(`io/interpolation/record_and_export.hpp`): the model the interpolation step
built — which method, the lattice in bohr, and the method's payload (for
`wannier`, the k-grid, the neighbour shells, the overlaps M_mn(k,b), the
projections A_mn(k), the eigenvalues and projectabilities). It is the primary
persistence of the step: `q3_utility interpolate` evaluates it along any path,
and `q3_utility convert` re-exports it, without re-running the NSCF. Binary,
`"Q3IP"` magic + version, length-prefixed arrays (`io/interpolation/checkpoint.hpp`).

The formats `control.interpolation.export_formats` lists are written beside it
as `<outdir>/<prefix>.*` — `wannier90` gives `.mmn`, `.amn`, `.eig`, `.nnkp`
in the Wannier90 file formats (Å and eV, converted from the record's bohr and
Ry by `io/interpolation/wannier90/exporter.hpp`). The reverse
direction, `q3_utility convert --from wannier90`, imports another code's files
into a record; overlaps are not imported, since nothing in Q3 consumes them
yet, and such a record re-exports without an `.mmn` rather than with an empty
one.

### 4.10 Volumetric field files — cube, `.xsf`

Written on demand by `q3_utility volumetric` (§2.4), never by a calculation run:
they are derived from the save set of §4.2 and nothing in the engine consumes
them. Default names are `<outdir>/<prefix>_<field>.cube` / `.xsf`; a format a
build-time add-on registers names its own file.

Every format samples the same field on the same dense (charge-density) FFT
grid, with the boundary point stored once and point `(i1, i2, i3)` sitting at
`r = i1/n1 · a1 + i2/n2 · a2 + i3/n3 · a3`. What they disagree about is
everything else, and the disagreements are silent, so they are listed rather
than implied:

| Format | Lengths  | Values                            | Fastest axis | Boundary    |
| ------ | -------- | --------------------------------- | ------------ | ----------- |
| cube   | bohr     | the field itself (e/bohr³, or Ry) | third        | stored once |
| `.xsf` | Ångström | the field itself                  | first        | repeated    |

A cube's voxel counts are written positive, which is what declares its lengths
to be bohr (negative would mean Ångström); its per-atom columns are the atomic
number and the species' **valence** charge, which is what the density in the
file actually integrates to. An `.xsf` datagrid is a "general" grid spanning
its vectors including the far end, so it is written `(n+1)³` with the last
plane a copy of the first — writing `n` instead leaves a one-voxel gap that
renders as a hole through every isosurface.

`ρ` is rebuilt from `charge-density.dat`'s ρ(G) by an inverse transform (a
gamma-only file stores a half sphere, whose conjugate half is rebuilt on the
way out); `V_eff` is read from `effective-potential.q3cp`, which stores it in
real space already. A band is written as |ψ_nk(r)|², normalized so its integral
over the cell is 1: the amplitude is complex at a general k-point and has no
single real value to plot, and a phase-dependent quantity would have no number
worth checking.

## 5. QE namelist format (`.in`/`.qe`)

An alternative, line-oriented input format (`src/io/input/parser/qe.hpp`)
reading `&CONTROL`/`&SYSTEM`/`&ELECTRONS`/`&IONS`/`&CELL` namelists and the
`ATOMIC_SPECIES`/`ATOMIC_POSITIONS`/`K_POINTS`/`CELL_PARAMETERS`/`HUBBARD`
cards. `key = value` extraction accepts quoted and bare values, normalizes
Fortran `d`-exponent literals (`1.0d0` → `1.0e0`), and handles several
assignments on one line (`ibrav= 2, celldm(1)=10.2, nat= 2`), which QE's own
examples use freely. Trailing `!` comments are stripped outside quotes.

Line breaks inside a namelist carry no meaning, as in Fortran: assignments may
share the header's line and the closing `/` may follow the last assignment, so
`&SYSTEM ibrav=2, celldm(1)=10.2, nat=2, ntyp=1, ecutwfc=30.0 /` is a whole
namelist. The first `/` outside quotes closes it (`pseudo_dir = './pseudo/'`
keeps its slashes), and text after that `/` is refused: Fortran skips the rest
of the line, so pw.x would drop it silently. `&END` is accepted as the older
spelling of `/`.

The goal of this format is to read real QE input files unchanged; QE's own
regression inputs are the working corpus.

### 5.1 Strictness policy

A drop-in replacement that silently computes different physics than the code it
replaces is worse than one that refuses to run, so every key falls in one class:

| Class                        | Behaviour                                                                |
| ---------------------------- | ------------------------------------------------------------------------ |
| supported                    | mapped onto the Q3 equivalent                                            |
| physics-neutral              | accepted and ignored, with a note in the log                             |
| default-valued               | accepted silently — a feature switched off asks for what Q3 already does |
| recognized but unimplemented | **hard error** naming the key and the reason                             |
| unknown                      | **hard error** naming the key                                            |

Unrecognized _namelists_ (`&FCP`, `&RISM`) are refused too: an unknown header
used to leave the parser scanning the previous namelist's keys, so everything
inside `&IONS` was dropped without a word.

Three tables in `qe_input::` carry the policy —
`physics_neutral_keys()`, `keys_inert_at_default()` and `unimplemented_keys()`
(key → reason) — alongside `honoured_keys_by_namelist()`, which is the one
table kept in step with the dispatch by hand. Drift there is self-announcing in
the safe direction: forgetting to list a newly handled key makes the parser
refuse an input it can run, which fails on the next test run rather than
producing a wrong number.

`keys_inert_at_default()` exists because real inputs state defaults all the
time. `noinv = .false.` and `tot_magnetization = -10000` (QE's "unspecified"
sentinel) ask for exactly what Q3 does, and refusing them would be a false
refusal. Numeric defaults compare numerically, `.false.` through the Fortran
logical spellings.

Two keys are checked rather than assumed: `dftd3_version` must be 4 and
`dftd3_threebody` must be false, which is what Q3's D3 implements. A D3 deck
must state both. pw.x's defaults are `dftd3_version = 3` (zero damping) and
`dftd3_threebody = .true.`, so `vdw_corr = 'dft-d3'` with neither key asks pw.x
for a different calculation from the one Q3 runs. Q3 used to run it anyway and
report success. Now the refusal names each key that is missing, the default pw.x
would take for it, and the line to add. The keys may appear anywhere in
`&SYSTEM`, so the check runs once the whole file has been read.

### 5.2 `&CONTROL`

| Key             | InputData field             | Notes                                                                                                                                                               |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `calculation`   | `calculation`               | QE vocabulary is translated: `scf`→`self_consistent_field`, `nscf`/`bands`→`non_self_consistent_field`, `relax`→`relaxation`, `vc-relax`→`variable_cell_relaxation` |
| `pseudo_dir`    | `pseudopotential_directory` | YAML derives this from the first resolved pseudopotential path                                                                                                      |
| `prefix`        | `prefix`                    |                                                                                                                                                                     |
| `outdir`        | `output_directory`          |                                                                                                                                                                     |
| `verbosity`     | `verbosity`                 | `'high'`/`'debug'` select Q3's `high`, `'low'`/`'default'`/`'medium'`/`'minimal'` its `default`; any other value is refused. Read rather than ignored since 2026-09 |
| `npool`         | `number_of_kpoint_pools`    | **Q3 extension**: real QE only accepts `-npool` on the command line. Ignored under `q3_mpi` — see §2.2                                                              |
| `restart_mode`  | —                           | only `'from_scratch'`; Q3 cannot resume a QE run's saved state                                                                                                      |
| `nstep`         | `relax_maximum_steps`       | inert for `calculation='scf'`, so applied unconditionally rather than deferred until the calculation type is known (QE does not require it to appear first)         |
| `forc_conv_thr` | `relax_force_threshold`     | Ry/bohr in both                                                                                                                                                     |
| `etot_conv_thr` | `relax_energy_threshold`    | Ry in both                                                                                                                                                          |

### 5.3 `&SYSTEM`

| Key                                 | InputData field                   | Notes                                                                                                                                                                                                               |
| ----------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ibrav`                             | `bravais_lattice_index`           | YAML derives this from `lattice.type`                                                                                                                                                                               |
| `celldm(1..6)`                      | `cell_dimension_1..6`             | 2..6 were dropped until 2026-08-08, which built every non-cubic Bravais index with c/a = 0 — a collapsed cell                                                                                                       |
| `A`,`B`,`C`,`cosAB`,`cosAC`,`cosBC` | `cell_dimension_1..6`             | QE's crystallographic form, converted exactly the way QE's own input reader converts them, including which celldm each cosine lands in (ibrav-dependent). Refused alongside `celldm(1)`, as QE refuses the pair too |
| `nat`                               | `number_of_atoms`                 | **not cross-validated** against the `ATOMIC_POSITIONS` line count                                                                                                                                                   |
| `ntyp`                              | `number_of_species`               | used only to bounds-check `starting_magnetization` indices                                                                                                                                                          |
| `ecutwfc`                           | `wavefunction_energy_cutoff`      | `method.cutoffs.wavefunction`                                                                                                                                                                                       |
| `ecutrho`                           | `charge_density_energy_cutoff`    | `method.cutoffs.density`                                                                                                                                                                                            |
| `nbnd`                              | `number_of_bands`                 |                                                                                                                                                                                                                     |
| `input_dft`                         | `exchange_correlation_functional` | `model.functional`. Absent, the functional the pseudopotentials state, as in pw.x — see §1.2                                                                                                                        |
| `nspin` (1 or 2)                    | `number_of_spin_components`       | **inverted vocabulary**: `nspin` is namelist-only (YAML rejects it); `number_of_spin_components` is YAML-only (the namelist rejects it with a message pointing here). `nspin=4` is refused                          |
| `starting_magnetization(i)`         | `starting_magnetization[i-1]`     | 1-based; unindexed means species 1 (Fortran scalar-assignment rule). YAML instead keys a map by element symbol. Fortran's array-continuation form (`= 0.5, 0.3`) is **refused**, not truncated                      |
| `degauss`                           | `smearing_width`                  |                                                                                                                                                                                                                     |
| `smearing`                          | `smearing`                        |                                                                                                                                                                                                                     |
| `occupations = 'smearing'`          | `smearing`                        | conditionally defaults to `"gaussian"` when no explicit `smearing` key set it                                                                                                                                       |
| `tot_charge`                        | `total_charge`                    |                                                                                                                                                                                                                     |
| `vdw_corr`                          | `dispersion_correction`           | `'dft-d3'`/`'grimme-d3'` → `"d3bj"`; other values refused                                                                                                                                                           |
| `dftd3_version`, `dftd3_threebody`  | —                                 | must be stated as `4` and `.false.` when D3 is on, since pw.x defaults to 3 and `.true.`                                                                                                                            |
| `nosym`                             | `disable_symmetry`                | no k-point reduction and no density/force/stress symmetrization; there is no separate `noinv`, so `nosym` alone already means QE's `nosym` + `noinv`                                                                |
| `assume_isolated`                   | `isolated_system_correction`      | `'makov-payne'`/`'m-p'`/`'mp'` and `'martyna-tuckerman'`/`'m-t'`/`'mt'`; `'esm'` and `'2D'` are different physics and are refused                                                                                   |

### 5.4 `&ELECTRONS`

| Key                 | InputData field          | Notes                                                                                                                                                                                                                                        |
| ------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `conv_thr`          | `convergence_threshold`  |                                                                                                                                                                                                                                              |
| `mixing_beta`       | `mixing_beta`            |                                                                                                                                                                                                                                              |
| `mixing_mode`       | `mixing_mode`            | `plain` (pw.x's default, also when the key is absent) → Broyden with preconditioning `none`, as pw.x mixes; `TF` → Broyden + `kerker`; `local-TF` → Broyden + `local-tf`. A QE deck therefore never gets YAML's smearing-dependent `default` |
| `mixing_ndim`       | `mixing_history_size`    |                                                                                                                                                                                                                                              |
| `electron_maxstep`  | `maximum_iterations`     |                                                                                                                                                                                                                                              |
| `diagonalization`   | `diagonalization_method` | `cg`/`ppcg`/`paro` → Davidson with a logged notice: they pick an iteration path, not a fixed point, and every QE PAW reference has to be generated with `cg` (QE's own Davidson aborts in `cegterg` on PAW datasets)                         |
| `scf_must_converge` | —                        | `.false.` refused; Q3 always requires convergence before reporting                                                                                                                                                                           |

### 5.5 `&IONS` and `&CELL`

| Key             | Notes                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------- |
| `ion_dynamics`  | only `'bfgs'`; damped and molecular dynamics follow a different trajectory and stop somewhere else |
| `cell_dynamics` | only `'bfgs'`                                                                                      |

### 5.6 Cards

| Card                                                                                                                               | Format                                        | Notes                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ATOMIC_SPECIES`                                                                                                                   | `symbol mass pseudopotential_file` per line   | `material.basis.elements` + `method.pseudopotentials`                                                                                                                                                                                                                                                         |
| `ATOMIC_POSITIONS {crystal\|alat\|bohr\|angstrom}`                                                                                 | `symbol x y z` per line                       | `alat`/`bohr`/`angstrom` rows are Cartesian, converted to crystal coordinates; no option means `alat`, as in pw.x; `crystal_sg` is refused. Trailing `if_pos` flags become the results' `material` constraints; other trailing text is refused. A symbol not declared in `ATOMIC_SPECIES` is silently dropped |
| `K_POINTS`                                                                                                                         | one following line: `nk1 nk2 nk3 sk1 sk2 sk3` | **gap**: only this `automatic`-style form; `gamma`/`crystal`/`tpiba`/explicit-list variants are not read                                                                                                                                                                                                      |
| `CELL_PARAMETERS {alat\|bohr\|angstrom}`                                                                                           | 3×3 rows                                      | requires `ibrav = 0`; `alat` rows additionally require `celldm(1)` (or `A`). No option means `alat` when `celldm(1)` or `A` is given and `bohr` when neither is, as in pw.x; any other option is refused                                                                                                      |
| `HUBBARD (atomic)` / `(ortho-atomic)`                                                                                              | `U <species>-<shell> <eV>` per line           | the projector flavour comes from the header; QE's `wf`/`pseudo`/`norm-atomic` are refused. Only `U` — QE's `J`, `V` and `B` describe interactions Dudarev's form does not have. eV in the card, Ry internally                                                                                                 |
| `OCCUPATIONS`, `CONSTRAINTS`, `ATOMIC_FORCES`, `ATOMIC_VELOCITIES`, `ADDITIONAL_K_POINTS`, `SOLVENTS`, `HUBBARD_V`, `TOTAL_CHARGE` | —                                             | refused by name                                                                                                                                                                                                                                                                                               |

A card header is distinguished from a namelist key by the absence of `=`.
Without that guard the `occupations` _key_ in `&SYSTEM` matches the
`OCCUPATIONS` _card_ and an ordinary input is rejected.

### 5.7 YAML-only keys with no namelist equivalent

Cross-referencing the YAML allow-lists against `qe.hpp` finds no namelist
equivalent for: `control.nonlocal_stress_derivative_method`,
`.use_deterministic_fft_planning`, `.use_exact_wavefunction_grid`,
`.band_path`/`.band_path_points_per_segment`, `.kpoints_list`,
`.dos_broadening`/`.dos_energy_min`/`.dos_energy_max`/`.dos_energy_steps`,
`.compute_projections`, `mixing.preconditioning`, `mixing.q0`, and
`kgrid.ibz_reduction`. Setting those requires the YAML format.
