# 05 — Python and ASE

**Goal:** drive Q3 from Python instead of the command line, and use the layer
that turns single-point calculations into workflows — relaxation, phonons, and
the ASE calculator interface everything else is built on.

The binary computes one thing at a time. Anything that needs a _sequence_ of
calculations — move the atoms, run again, decide where to move next — needs a
driver. That driver is the `q3` Python package, and it is deliberately thin: it
writes manifests, runs the binary, and parses `results.json`.

## 1. Install

**Python 3.12 or newer is required.** The `python3` on your PATH may well be
older; check before you build a virtual environment around it, because the
failure comes at install time with a message that is easy to misread:

```
ERROR: Package 'q3' requires a different Python: 3.10.13 not in '>=3.12'
```

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[test]"
```

ASE, which everything in this lesson needs, is a base dependency; `test`
pulls in pytest, which [tutorial 07](07-contributing.md) needs. Verify:

```bash
.venv/bin/python -c "import q3, ase; print(ase.__version__)"
```

The package is layered the same way the C++ is, and for the same reason — an
upward import is a cycle waiting to happen. `q3.constants` and `q3.errors` at
the bottom; then the
adapters `q3.binary`, `q3.io`, `q3.structure`; then `q3.calculators` and
`q3.flavors`, which own _one_ run of the engine; then `q3.workflows` and
`q3.jax`, which orchestrate many. A module imports from its own tier or below,
never above. `AGENTS.md` documents the rule and a pre-commit hook enforces it.

## 2. The ASE calculator

`Q3ASECalculator` makes Q3 look like any other ASE calculator: give it a
manifest to use as a template, hand it `Atoms`, ask for properties.

```python
from q3.calculators.ase_calculator import Q3ASECalculator
from q3.structure.ase_atoms import material_json_to_ase_atoms

atoms = material_json_to_ase_atoms("examples/data/materials/silicon.json")
atoms.calc = Q3ASECalculator(
    "examples/total_energy--semiconductor--silicon/nc.yml",
    mode="subprocess",
    q3_binary="./build/q3",
)

print(atoms.get_chemical_formula())
print(f"energy = {atoms.get_potential_energy():.7f} eV")
print(f"max force = {abs(atoms.get_forces()).max():.2e} eV/A")
```

```
Si2
energy = -230.1187741 eV
max force = 2.10e-17 eV/A
```

The manifest supplies the physics — cutoffs, k-grid, pseudopotentials,
functional — and the `Atoms` object supplies the geometry. For each property
request the calculator writes a fresh `input.yml` and `material.json` with the
current positions into a numbered step directory, runs the binary, and reads
the results back.

**Units change at this boundary.** ASE speaks eV and eV/Å; Q3 speaks Rydberg
and Ry/bohr. The conversion is exact in intent but not in the last digits,
because the two projects use Ry→eV constants that differ in the ninth
significant figure (13.605693127 in Q3's, 13.605693012 in `ase.units.Ry`).
Converting back gives −16.913418074 Ry, while the `results.json` this run
wrote holds −16.913417932 Ry — the same number lesson 01 produced from the
command line, bit for bit. **When you need Q3's exact value, read
`results.json`**; the ASE value is for ASE's algorithms.

Two practical notes:

-   **Stress costs an extra SCF run.** `get_potential_energy()` and
    `get_forces()` share one calculation, but `get_stress()` triggers a second.
    On a variable-cell relaxation that doubles your cost — worth knowing before
    you attribute it to the optimizer.
-   **The work directory persists.** It defaults to `q3_ase_workdir/` in the
    current directory and accumulates one `step_NNNN/` per calculation, each
    holding `input.yml`, `material.json` and `output/`. That is a feature when
    a relaxation goes wrong at step 40 and you want to see what was submitted,
    and a disk leak when you forget. Pass `workdir=` to control it.

## 3. Relaxation

With an ASE calculator in hand, relaxation is ASE's optimizer driving Q3's
forces. `run_relax_workflow` wires the two together:

```python
from q3.workflows import run_relax_workflow

result = run_relax_workflow(
    scf_input_path="examples/relaxation--semiconductor--silicon/nc.yml",
    fmax=0.01,
    steps=50,
    q3_binary="./build/q3",
)
print(result["converged"], result["steps"], result["final_energy_ry"])
```

That example starts from `silicon_displaced.json` — silicon with the atoms
pushed off their symmetry positions, so there is something to relax:

```
      Step     Time          Energy          fmax
BFGS:    0 02:00:51     -227.753724        0.528032
BFGS:    1 02:00:52     -227.761722        0.074791
BFGS:    2 02:00:53     -227.761887        0.000800
True 2 -16.74018993100038
```

Three force evaluations to go from 0.53 to 0.0008 eV/Å. `fmax` is the
convergence criterion in **eV/Å** (ASE's convention), not Ry/bohr — one of the
places the unit boundary above will bite you if you copy a threshold across
from a manifest.

The returned dict carries `converged`, `steps`, `final_energy_ry` and
`final_forces_ry_per_bohr`. Check `converged` before using the geometry, for
exactly the reason tutorial 02 gave: hitting `steps` without reaching `fmax`
still returns a complete-looking result.

You will also see this on the way past:

```
WARNING:root:FixSymmetry constraint could not be loaded; proceeding without symmetry constraints.
```

It is what it says — the relaxation runs unconstrained, so symmetry is not
preserved by construction and can drift as forces get small. For a displaced
structure relaxing back to a symmetric one that is harmless; if you need
symmetry held exactly, verify the final structure rather than assuming it.

## 4. Phonons

Force constants come from finite differences: displace each atom along each
Cartesian axis, compute forces, assemble the dynamical matrix. That is 6N SCF
calculations for N atoms plus an equilibrium reference — 13 for silicon's
two-atom cell — which is why this is a workflow and not something you do by
hand.

```python
from q3.workflows import run_gamma_phonons_workflow

result = run_gamma_phonons_workflow(
    "examples/total_energy--semiconductor--silicon/nc.yml",
    q3_binary="./build/q3",
)
print([round(float(f), 2) for f in result["frequencies_cm1"]])
```

```
[-0.0, 0.0, 0.0, 507.89, 507.89, 507.89]
```

Six modes for two atoms, and both halves are checks you should make:

-   **Three modes at zero.** These are the acoustic modes — translating the
    whole crystal costs no energy. They come out at zero only if the force
    constants obey the acoustic sum rule, so a nonzero value here means the
    forces are not converged.
-   **Three degenerate at 507.89 cm⁻¹.** Silicon's optical mode at Γ is
    triply degenerate by symmetry, and the measured Raman frequency is
    ~520 cm⁻¹. PBE softening it by a couple of percent is the expected result.

You will also see:

```
WARNING, 1 imaginary frequencies at q = ( 0.00, 0.00, 0.00) ; (omega_q = 4.868e-09*i)
```

That is an acoustic mode landing a hair below zero — 5×10⁻⁹, which is numerical
noise, not an instability. A _real_ imaginary mode is a substantial number and
means the structure is not at a minimum. Read the magnitude before reacting to
the word "imaginary".

This workflow is Γ-only and does no symmetry reduction of the displacement set.
It is the cheapest thing that can answer the question, and it answers it only
at the zone centre.

### Dispersions and thermodynamics, through Phonopy

For frequencies at any q you need force constants that reach beyond one unit
cell, which means displacing atoms in a supercell. `run_phonopy_workflow`
hands that problem to [Phonopy](https://phonopy.github.io/phonopy/): Phonopy
works out which displacements the crystal's symmetry actually requires, Q3
computes the forces on each displaced supercell, and Phonopy turns the force
sets into force constants, a dispersion and harmonic thermodynamics.

Install it alongside Q3 with `pip install q3[phonons]`.

```python
from q3.workflows import run_phonopy_workflow

result = run_phonopy_workflow(
    "examples/total_energy--semiconductor--silicon/nc.yml",
    supercell_matrix=(2, 2, 2),
    band_path=["G", "X", "W", "K", "G", "L"],
    points_per_segment=20,
    q3_binary="./build/q3",
)
dispersions = result["phonon_dispersions"]
print(dispersions["yAxis"], len(dispersions["yDataSeries"]), "branches")
print([round(series[0], 2) for series in dispersions["yDataSeries"]])
```

```
{'label': 'frequency', 'units': 'THz'} 6 branches
[-0.0, -0.0, -0.0, 14.38, 14.38, 14.38]
```

Two SCF calculations for a 16-atom supercell — that is the whole cost, and it
buys the entire Brillouin zone. Silicon's diamond structure leaves exactly one
inequivalent displacement direction; the workflow runs it at `+u` and `-u`
because a one-sided difference on an unconverged k-mesh picks up a systematic
error that the central difference cancels (the workflow's docstring has the
measured numbers).

The result is the ESSE `phonon_dispersions` property: `xDataArray` holds the
q-points along the path, in the material's own crystal coordinates, and
`yDataSeries` one series per branch, so the line above read each branch at its
first point, Γ. At X, one 20-point segment along:

```python
x_point = 20
print([round(series[x_point], 2) for series in dispersions["yDataSeries"]])
```

```
[4.15, 4.15, 11.74, 11.74, 12.81, 12.81]
```

Three pairs, and each degeneracy is required by symmetry at X. The two
transverse acoustic branches come first — low and flat, the signature feature
of silicon's dispersion — then LA and LO, which meet at X, then the two
transverse optical branches. Measured values are 4.49, 12.32 and 13.90 THz.
The shortfall is not the method: Quantum ESPRESSO, through the same Phonopy on
the same nominal k-mesh, puts Γ at 15.22 THz, and the difference is how Q3
samples a shifted mesh on the displaced supercell (FUNCTIONALITY.md has the
comparison).

The thermodynamics are free once the force constants exist, because they come
from Fourier-interpolating them onto a q-mesh rather than from more DFT. ESSE
has no temperature-dependent property for them, so each of free energy, entropy
and heat capacity comes back as ESSE's abstract two-dimensional plot:
temperature in `xDataArray`, the values in `yDataSeries[0]`, in Phonopy's units:

```python
heat_capacity = result["thermal_properties"]["heat_capacity"]
room_temperature = heat_capacity["xDataArray"].index(300.0)
print(round(heat_capacity["yDataSeries"][0][room_temperature], 1))
print(result["zero_point_energy"])
```

```
40.5
{'name': 'zero_point_energy', 'units': 'kJ/mol', 'value': 11.210552030879674}
```

That is per mole of unit cells, and silicon's cell has two atoms, so 20.2
J/K/mol of atoms against a measured 20.0 at room temperature. At high
temperature it climbs to the Dulong–Petit limit of 3 × 2 × R = 49.9; the
zero-point energy, an ESSE property again, is 0.058 eV/atom.

## 5. What else is there

`q3.workflows` exports `run_band_structure_workflow` and `run_dos_workflow`
(both used in [tutorial 04](04-band-structure-and-dos.md)),
`run_gamma_phonons_workflow` and `run_phonopy_workflow`, `run_relax_workflow`,
`run_vc_relax_workflow` for variable-cell relaxation, and `run_pdos_workflow`
for the projected density of states. They share the shape you have now seen twice: point them at an SCF
manifest, pass the workflow's own parameters as keywords, get a dict back.

`q3.jax` is a different thing entirely — it exposes Q3 as a differentiable
primitive. See [docs/differentiability.md](../differentiability.md).

## Next

[06 — Bringing existing inputs](06-existing-inputs.md): running a
Quantum ESPRESSO namelist without converting it first.
