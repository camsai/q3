# 02 — Reading the results

**Goal:** get energies, forces, stress, eigenvalues and occupations out of
`results.json` in a form you can script against.

The terminal log is for humans. `results.json` is the contract — it is what the
regression suite reads, what the Python workflow layer parses, and what you
should build on. Run tutorial 01's silicon example first, then:

```bash
python3 -c "import json; d=json.load(open('output/si_nc_results.json')); print(sorted(d))"
```

```
['band_energy_ry', 'converged', 'double_counting_correction_ry', 'eigenvalues_ry',
 'ewald_energy_ry', 'exchange_correlation_energy_ry', 'execution', 'fermi_energy_ry',
 'flavor', 'forces_ry_per_bohr', 'hartree_energy_ry', 'kpoint_weights', 'occupations',
 'provenance', 'self_consistent_field_iterations', 'smearing_entropy_contribution_ry',
 'stress_contributions_kbar', 'stress_kbar', 'timings_ms', 'total_energy_ry', 'units']
```

Every key carries its unit in its name, and a `units` block states them again
(`energy: Ry`, `forces: Ry/bohr`, `stress: kbar`, `eigenvalues: Ry`). Q3 works
in Rydberg atomic units throughout.

## Check `converged` first — always

```python
import json

with open("output/si_nc_results.json") as handle:
    results = json.load(handle)

if not results["converged"]:
    raise SystemExit("SCF did not converge — every number below is meaningless")

print(f"total energy = {results['total_energy_ry']:.9f} Ry")
```

```
total energy = -16.913417932 Ry
```

An unconverged run still writes a complete, well-formed results file with a
plausible-looking energy in it. `converged` is the only thing standing between
you and publishing that number. Make this the first line of any script.

When a run did not converge, `self_consistent_field_iterations` says how far it
got, and `provenance.precision.maximum_self_consistent_field_iterations` is the
limit it was given. Equal numbers mean it ran out of iterations; the silicon
example converges on iteration 7 of its 100.

## Energy components

The total is decomposed, which is how you diagnose a suspicious result:

```python
for key in ("total_energy_ry", "band_energy_ry", "hartree_energy_ry",
            "exchange_correlation_energy_ry", "ewald_energy_ry",
            "double_counting_correction_ry", "smearing_entropy_contribution_ry"):
    print(f"{key:38s} {results[key]:16.9f}")
```

```
total_energy_ry                         -16.913417932
band_energy_ry                            0.620351315
hartree_energy_ry                         1.133669940
exchange_correlation_energy_ry           -6.169552421
ewald_energy_ry                         -16.679972546
double_counting_correction_ry             4.182085780
smearing_entropy_contribution_ry          0.000000000
```

`ewald_energy_ry` is the classical ion–ion electrostatic energy. It depends
only on the geometry and charges, not on the SCF, so it is a fast way to check
that two calculations really used the same structure: if the Ewald energies
differ, the geometries differ, whatever you believed about your inputs.

`smearing_entropy_contribution_ry` is zero here because silicon is an insulator
run with fixed occupations. In a metal it is the `-TS` term, and a large value
means your smearing width is doing more work than it should.

## Forces and stress

```python
forces = results["forces_ry_per_bohr"]
largest = max(max(abs(component) for component in atom) for atom in forces)
print(f"largest force component = {largest:.2e} Ry/bohr")

stress = results["stress_kbar"]
pressure = sum(stress[i][i] for i in range(3)) / 3
print(f"pressure = {pressure:.2f} kbar")
```

```
largest force component = 2.89e-19 Ry/bohr
pressure = 3.71 kbar
```

Forces are per atom in the input's order, `[[fx, fy, fz], ...]`. Stress is a
3×3 tensor; the trace over 3 is the pressure, and a **positive** pressure means
the cell wants to expand.

`stress_contributions_kbar` breaks the tensor into `kinetic`, `hartree`,
`exchange_correlation`, `ewald`, `local_pseudopotential`,
`nonlocal_pseudopotential`, `core_correction`, `augmentation`, `dispersion` and
`hubbard`. These individual terms are large and cancel heavily — kinetic is
+2176 kbar against Ewald's −2964 kbar to leave 3.71 kbar — which is exactly why
a small absolute error in one term can matter, and why they are reported
separately.

## Eigenvalues and occupations

Both are indexed `[k-point][spin][band]`:

```python
eigenvalues = results["eigenvalues_ry"]
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

For spin-polarized runs the middle index has two entries, and `occupations`
(same indexing) gives you the moment:

```python
weights = results["kpoint_weights"]
total_weight = sum(weights)
moment = sum(
    weight * (sum(up) - sum(down))
    for weight, (up, down) in zip(weights, results["occupations"])
) / total_weight
```

k-point weights are **not** normalized to 1 — in this example they are 0.5 and
1.5, summing to 2. Always divide by the sum; assuming normalization is a
common source of factor-of-two errors.

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
