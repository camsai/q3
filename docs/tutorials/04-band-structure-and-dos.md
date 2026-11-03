# 04 — Band structure and DOS

**Goal:** get silicon's actual band gap — 0.61 eV, not the 2.28 eV tutorial 02
read off an SCF grid — and understand why the SCF grid could never have given
it.

## Why the SCF grid cannot answer this

An SCF calculation samples the Brillouin zone wherever the _integrals_ need
sampling. It converges the density, and the density is an integral over
occupied states; a Monkhorst-Pack grid is a quadrature rule chosen for that
job. Nothing about it guarantees a point sits at the valence-band maximum or
the conduction-band minimum.

For silicon those two extrema are at different places, and one of them is at no
rational grid point you would pick by accident: the VBM is at Γ, and the CBM is
about 83% of the way from Γ to X. Tutorial 02's 2×2×2 grid held two
inequivalent points, neither of them near that minimum, so the smallest gap it
could see was 2.28 eV.

The fix is to decouple the two jobs. Converge the density on a grid, then
**stop iterating** and diagonalize the resulting fixed Hamiltonian at whatever
k-points you actually care about. That second step is a non-self-consistent
field (NSCF) calculation, and it is cheap: one diagonalization per k-point, no
loop.

## 1. The two-step, by hand

The SCF step is an ordinary calculation — the only requirements are that it be
well converged (tutorial 03) and that you remember its `outdir` and `prefix`,
because that is how the NSCF step finds it. Write `scf.yml`:

```yaml
material: examples/data/materials/silicon.json
model:
    type: dft
    subtype: gga
    functional: pbe
method:
    type: pseudopotential
    subtype: norm-conserving
    cutoffs:
        wavefunction: 48.0
        density: 192.0
    pseudopotentials:
        Si: examples/data/pseudopotentials/nc/Si.upf
    scf:
        convergence_threshold: 1.0e-08
        number_of_bands: 12
        mixing: { mode: broyden, beta: 0.7, ndim: 8 }
        kgrid:
            dimensions: [8, 8, 8]
            shifts: [0, 0, 0]
control:
    calculation: scf
    outdir: ./output
    prefix: si_bands
```

```bash
./build/q3 scf.yml
```

```
     convergence achieved after 6 iterations
    total energy         =  -16.924974846 Ry
```

Now `nscf.yml`. Same material, same method — a different Hamiltonian would make
the reloaded potential meaningless — but `calculation: nscf`, no `kgrid`, and
an explicit `kpoints_list` of `[kx, ky, kz, weight]` rows. The coordinates are
crystal — fractions of the reciprocal lattice vectors, the same convention a
generated `kgrid` uses. These four are Γ, the point 83% along Γ–X, X itself,
and L:

```yaml
material: examples/data/materials/silicon.json
model:
    type: dft
    subtype: gga
    functional: pbe
method:
    type: pseudopotential
    subtype: norm-conserving
    cutoffs:
        wavefunction: 48.0
        density: 192.0
    pseudopotentials:
        Si: examples/data/pseudopotentials/nc/Si.upf
    scf:
        convergence_threshold: 1.0e-08
        number_of_bands: 12
control:
    calculation: nscf
    outdir: ./output
    prefix: si_bands
    kpoints_list:
        - [0.000000, 0.000000, 0.000000, 1.0]
        - [0.416667, 0.000000, 0.416667, 1.0]
        - [0.500000, 0.000000, 0.500000, 1.0]
        - [0.500000, 0.500000, 0.500000, 1.0]
```

```bash
./build/q3 nscf.yml
```

```
NSCF: loaded effective potential from './output/si_bands.save/effective-potential.q3cp'
NSCF: diagonalizing at 4 explicit k-points
```

That first line is the whole mechanism. The converged effective potential — the
`.save/` contents tutorial 02 listed — is read back instead of being
recomputed, and the four k-points are diagonalized against it.

**The NSCF run overwrites `<prefix>_results.json`.** Same `outdir`, same
`prefix`, same filename. The SCF energies in it are replaced by NSCF
eigenvalues at the new k-points. Copy the SCF results aside first if you
need both.

Reading the eigenvalues out — silicon has 8 valence electrons, so bands 0–3 are
occupied and band 4 is the first empty one:

```python
from q3.io.results.classes import Results, eigenvalue_arrays

RYDBERG_TO_EV = 13.605693
results = Results.read("output/si_bands_results.json")
eigenvalues, _, _ = eigenvalue_arrays(results.properties.eigenvalues_at_kpoints)  # Ry, [k-point][spin][band]

labels = ["G", "0.83 G-X", "X", "L"]
for label, kpoint in zip(labels, eigenvalues):
    valence, conduction = kpoint[0][3], kpoint[0][4]
    print(f"{label:<10} {valence:9.5f} {conduction:9.5f} "
          f"{(conduction - valence) * RYDBERG_TO_EV:8.3f}")

valence_maximum = max(kpoint[0][3] for kpoint in eigenvalues)
conduction_minimum = min(kpoint[0][4] for kpoint in eigenvalues)
print(f"indirect gap = {(conduction_minimum - valence_maximum) * RYDBERG_TO_EV:.3f} eV")
```

```
G           0.44198   0.62917    2.547
0.83 G-X    0.24411   0.48657    3.299
X           0.23619   0.49703    3.549
L           0.35508   0.54884    2.636
indirect gap = 0.607 eV
```

**0.607 eV**, from four k-points. Every one of those _direct_ gaps — the
vertical distance at a single k-point — is above 2.5 eV. The fundamental gap is
small only because the maximum and the minimum are at different k, which is
precisely the information a Monkhorst-Pack grid throws away.

You can also use more bands in the NSCF step than the SCF used
(`number_of_bands: 16` works against a 12-band SCF): the extra conduction bands
never entered the density, so they cost one diagonalization and nothing else.

## 2. The full path

Four points prove the mechanism; a plot needs the path filled in. That
bookkeeping — looking up high-symmetry coordinates for the lattice,
interpolating segments, tracking cumulative distance for the x-axis — is what
`q3.workflows.band_structure` does. It needs the Python package
([tutorial 05](05-python-and-ase.md) covers the install):

```python
from q3.workflows.band_structure import run_band_structure_workflow

result = run_band_structure_workflow(
    scf_input_path="examples/band_structure--semiconductor--silicon/nc.yml",
    band_path=["G", "X", "W", "K", "G", "L"],
    lattice_type="FCC",
    points_per_segment=30,
    q3_binary="./build/q3",
)

bands = result["band_structure"]
print(len(bands["eigenvalues_ry"]), "k-points,", len(bands["eigenvalues_ry"][0][0]), "bands")
print("Fermi energy =", bands["fermi_energy_ry"], "Ry")
```

```
151 k-points, 12 bands
Fermi energy = 0.46574566890790314 Ry
```

It runs both steps for you and returns `eigenvalues_ry`, `kpoints`,
`kpath_distances` (the plot's x-axis) and `fermi_energy_ry`. On the same
analysis as above the 151-point path gives:

| Quantity             | Value                            |
| -------------------- | -------------------------------- |
| Indirect gap         | 0.607 eV                         |
| Valence band maximum | Γ                                |
| Conduction minimum   | (0.4167, 0, 0.4167) — 83% of Γ–X |
| Direct gap at Γ      | 2.547 eV                         |

The same 0.607 eV as the four hand-picked points, which is the reassuring
outcome: the path did not find anything lower, so those four points were the
right four. Silicon's experimental gap is 1.17 eV — PBE underestimates it, as
semilocal functionals do, and 0.6 eV is the correct _PBE_ answer. Getting
1.17 eV out of this calculation would mean something was wrong, not right.

**Two committed manifests do not run through the binary directly.**
`examples/band_structure--*/nc.yml` and `examples/density_of_states--*/nc.yml`
declare `calculation: band_structure` and `calculation: density_of_states`,
which the YAML parser accepts but the flavor factory does not implement —
`./build/q3` on either exits with `unsupported calculation type`. They are
inputs for the Python workflows above, which rewrite the calculation type to
`self_consistent_field` before invoking the binary.

## 3. Density of states

The DOS asks a different question: not _where_ the states are in k, but how
many there are at each energy. It comes from the same NSCF machinery — a dense
grid instead of a path, then a broadened histogram of the eigenvalues.

```python
from q3.workflows.dos import run_dos_workflow

result = run_dos_workflow(
    scf_input_path="examples/density_of_states--semiconductor--silicon/nc.yml",
    broadening_ry=0.01,
    energy_steps=3000,
    q3_binary="./build/q3",
)

energies = result["dos"]["energies_ry"]
density = result["dos"]["dos"]
fermi = result["dos"]["fermi_energy_ry"]
```

The check worth running on any DOS — integrate it up to the Fermi level and see
whether you get your electrons back:

```python
electrons = 0.0
for index in range(1, len(energies)):
    if energies[index] > fermi:
        break
    width = energies[index] - energies[index - 1]
    electrons += 0.5 * (density[index] + density[index - 1]) * width

print(f"electrons below E_F = {electrons:.4f}")
```

```
electrons below E_F = 8.0002
```

Eight, as it must be for two silicon atoms at Z=4. This catches a broken
normalization, a wrong Fermi level, or a k-point weight bug in one line — and
it is the reason to trust anything else the DOS tells you.

**Do not read the band gap off this plot.** With `broadening_ry: 0.01` — 0.14 eV
— the Gaussian tails from the band edges fill the gap in, and the width you
measure depends entirely on the threshold you call "zero": 0.27 eV at 2% of
peak height, 0.45 eV at 5%. Neither is the 0.607 eV that the band path gives.
Broadening is a plotting parameter; the gap is not a plotting parameter. Get
gaps from eigenvalues on a path, and use the DOS for what it is good at —
where the weight is, how it is distributed, whether there is a peak at the
Fermi level.

## Next

[05 — Python and ASE](05-python-and-ase.md): installing the package used
above, and what else it drives — relaxation, phonons, and the ASE calculator.
