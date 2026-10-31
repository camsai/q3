# 02 — Reading the results

**Goal:** get energies, forces, stress, eigenvalues and occupations out of
`results.json` in a form you can script against.

The terminal log is for humans. `results.json` is the contract — it is what the
regression suite reads, what the Python workflow layer parses, and what you
should build on. Run tutorial 01's silicon example first, then:

```bash
python3 -c "import json; d=json.load(open('output/si_nc_results.json')); print({k: sorted(v) for k, v in d.items()})"
```

```
{'structure': ['material'],
 'provenance': ['generator', 'model', 'precision', ...],
 'properties': ['energy_contributions', 'force_contributions', 'records',
                'stress_contributions'],
 'runtime_log': ['converged', 'execution', 'flavor',
             'self_consistent_field_iterations', 'timings_ms']}
```

The file is grouped by domain: `structure` is the material the run computed,
`provenance` what it was asked to do and by which build, `properties` the
physical results, and `runtime_log` how the calculation itself went.

The headline results -- total energy and its breakdown, Fermi energy, forces,
stress, pressure, band gaps -- are
[ESSE](https://github.com/Exabyte-io/esse) property records in the
`properties.records` list, each carrying its own `units`:

```json
{ "name": "total_energy", "units": "Ry", "value": -16.913417932 }
```

Everything else is an ESSE shape that carries its unit the same way, in a
`units` field beside the value, never in the key's name:

```json
"energy_contributions": {
    "double_counting": { "name": "double_counting", "units": "Ry", "value": 4.18208578 }
}
```

Q3 works in Rydberg atomic units throughout, and every quantity in the file
states its unit in its own `units` field. `q3.io.results.classes.Results` reads the
file into typed blocks; each ESSE record is ESSE's own pydantic model, read by
its name as an attribute of `properties`.

## Check `runtime_log.converged` first — always

```python
from q3.io.results.classes import Results

results = Results.read("output/si_nc_results.json")

if not results.runtime_log.converged:
    raise SystemExit("SCF did not converge — every number below is meaningless")

energy = results.properties.total_energy
print(f"total energy = {energy.value:.9f} {energy.units.value}")
```

```
total energy = -16.913417932 Ry
```

An unconverged run still writes a complete, well-formed results file with a
plausible-looking energy in it. `runtime_log.converged` is the only thing standing between
you and publishing that number. Make this the first line of any script.

When a run did not converge, `runtime_log.self_consistent_field_iterations` says how far it
got, and `provenance.precision.maximum_self_consistent_field_iterations` is the
limit it was given. Equal numbers mean it ran out of iterations; the silicon
example converges on iteration 7 of its 100.

## Energy components

The total is decomposed, which is how you diagnose a suspicious result:

```python
contributions = results.properties.total_energy_contributions
for key, term in contributions.model_dump(by_alias=True, exclude_none=True).items():
    if isinstance(term, dict):
        print(f"{key:30s} {term['value']:16.9f}")
for key, quantity in results.properties.energy_contributions:
    if quantity is not None:
        print(f"{key:30s} {quantity.value:16.9f}")
```

```
eigenvalues                           0.620351315
hartree                               1.133669940
exchange_correlation                 -6.169552421
ewald                               -16.679972546
temperatureEntropy                    0.000000000
double_counting                       4.182085780
```

The contributions are ESSE's `total_energy_contributions` record, keyed as ESSE
keys them: `eigenvalues` is the band energy and `temperatureEntropy` the `-TS`
smearing term. The terms ESSE has no key for yet (the double-counting
correction, DFT+U, dispersion, PAW one-center) are under `properties.energy_contributions`,
each an ESSE energy with its own `units`.

`ewald` is the classical ion–ion electrostatic energy. It depends
only on the geometry and charges, not on the SCF, so it is a fast way to check
that two calculations really used the same structure: if the Ewald energies
differ, the geometries differ, whatever you believed about your inputs.

`temperatureEntropy` is zero here because silicon is an insulator
run with fixed occupations. In a metal it is the `-TS` term, and a large value
means your smearing width is doing more work than it should.

## Forces and stress

```python
forces = sorted(results.properties.atomic_forces.values, key=lambda atom: atom.id)
largest = max(max(abs(component) for component in atom.value) for atom in forces)
print(f"largest force component = {largest:.2e} Ry/bohr")

stress = [row.root for row in results.properties.stress_tensor.value]
pressure = results.properties.pressure.value
print(f"pressure = {pressure:.2f} kbar")
```

```
largest force component = 2.89e-19 Ry/bohr
pressure = 3.71 kbar
```

The forces record lists `{"id", "value"}` pairs, `id` 1-based as in
`structure.material`; sorting by `id` puts them in the input's order. Nothing
converts units on the way: check a record's `units` if your code assumes one.
Stress is a
3×3 tensor; the trace over 3 is the pressure, and a **positive** pressure means
the cell wants to expand.

`properties.stress_contributions` breaks the tensor into `kinetic`, `hartree`,
`exchange_correlation`, `ewald`, `local_pseudopotential`,
`nonlocal_pseudopotential`, `core_correction`, `augmentation`, `dispersion` and
`hubbard`. These individual terms are large and cancel heavily — kinetic is
+2176 kbar against Ewald's −2964 kbar to leave 3.71 kbar — which is exactly why
a small absolute error in one term can matter, and why they are reported
separately. Each is a `stress_tensor` shape (`units`, `value`), so
`results.properties.stress_contributions["ewald"]` reads one;
`force_contributions` holds the force terms the same way.

## Eigenvalues and occupations

They are ESSE's `eigenvalues_at_kpoints` record, `results.properties.eigenvalues_at_kpoints`:
one entry per k-point with its `kpoint`, its `weight`, and the band `energies` and
`occupations` of each spin channel. `eigenvalue_arrays` turns that into arrays
indexed `[k-point][spin][band]`, with the occupations and weights beside them:

```python
from q3.io.results.classes import eigenvalue_arrays

eigenvalues, occupations, weights = eigenvalue_arrays(results.properties.eigenvalues_at_kpoints)  # Ry
print(f"{len(eigenvalues)} k-points, {len(eigenvalues[0])} spin channel(s), "
      f"{len(eigenvalues[0][0])} bands")

RYDBERG_TO_EV = 13.605693
occupied = 4  # 8 electrons / 2 per band

valence_maximum = max(eigenvalues[k][0][occupied - 1] for k in range(len(eigenvalues)))
conduction_minimum = min(eigenvalues[k][0][occupied] for k in range(len(eigenvalues)))
print(f"gap on this grid = {(conduction_minimum - valence_maximum) * RYDBERG_TO_EV:.2f} eV")
```

```
2 k-points, 1 spin channel(s), 8 bands
gap on this grid = 2.28 eV
```

**That is not silicon's band gap.** Silicon's PBE gap is around 0.6 eV. The
2×2×2 grid used here contains only two inequivalent k-points, and neither is
near the conduction-band minimum along Γ–X, so the smallest gap this grid can
_see_ is far too large. The number is arithmetically correct and physically
wrong — a good illustration of why [tutorial 03](03-convergence.md) exists, and
why band gaps come from the k-path workflow in
[tutorial 04](04-band-structure-and-dos.md), not from an SCF grid.

A spin-polarized run lists every k-point twice, the spin-up states first and
then the spin-down ones, each at half the k-point's weight. Its moment is
reported directly, as ESSE's `magnetization_total` and `magnetization_absolute` records:

```python
moment = results.properties.magnetization_total.value  # uB
```

k-point weights are **not** normalized to 1 — in this example they are 0.5 and
1.5, summing to 2. Always divide by the sum when you average over them;
assuming normalization is a common source of factor-of-two errors.

## What else a run leaves behind

Alongside `<prefix>_results.json`, the `outdir` holds a `<prefix>.save/`
directory:

```
output/si_nc.save/
├── charge-density.dat          converged density
├── effective-potential.q3cp    V_eff, for restarts and NSCF
├── wfc1.dat                    wavefunctions, per k-point
└── wfc2.dat
```

These are what a band-structure or DOS run reads back instead of redoing the
SCF, and what a restart resumes from. [docs/io_reference.md](../io_reference.md)
documents the formats.

## Next

[03 — Converging a calculation](03-convergence.md): making these numbers mean
something.
