# 03 — Converging a calculation

**Goal:** turn the number from tutorial 01 into a result you can defend, by
finding out how much of it is physics and how much is your choice of
approximations.

`converged: true` in `results.json` means the SCF loop reached its density
threshold. It says nothing about whether the _answer_ is converged. Those are
different questions, and this lesson is about the second one.

Two approximations control the error in every plane-wave DFT calculation, and
they are independent:

-   **The basis cutoff** — how many plane waves represent each orbital.
    Variational: more plane waves can only lower the energy, so it converges
    from above, smoothly and monotonically.
-   **The Brillouin-zone sampling** — how many k-points approximate an integral
    over the zone. Not variational: it is a quadrature error, so it can
    converge from either side, and for metals it oscillates.

You converge each one by scanning it and watching a quantity stop moving.

## A scan script

Save this as `scan.py`. It rewrites one field in a manifest, runs each variant,
and prints the energy:

```python
"""Sweep one input field and report the total energy."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml
from q3.io.results.classes import Results

manifest_path, field, *values = sys.argv[1:]
manifest_path = Path(manifest_path).resolve()
base = yaml.safe_load(manifest_path.read_text())
keys = field.split(".")
scratch = Path("scan_output").resolve()
scratch.mkdir(exist_ok=True)

for value in values:
    config = copy.deepcopy(base)
    node = config
    for key in keys[:-1]:
        node = node[key]
    parsed = json.loads(value)
    # A k-grid is three numbers; a cutoff is one.
    node[keys[-1]] = [parsed] * 3 if keys[-1] == "dimensions" else parsed
    if keys[-1] == "wavefunction":
        config["method"]["cutoffs"]["density"] = 4.0 * parsed

    # Paths in the manifest resolve against the manifest's directory, so the
    # copy has to live beside the original.
    config["control"]["outdir"] = str(scratch / f"run_{value}")
    config["control"]["prefix"] = "scan"
    copy_path = manifest_path.parent / f".scan_{value}.yml"
    copy_path.write_text(yaml.safe_dump(config))
    try:
        subprocess.run(["./build/q3", str(copy_path)], check=True, capture_output=True)
    finally:
        copy_path.unlink()

    results = Results.read(scratch / f"run_{value}" / "scan_results.json")
    energy = results.properties.total_energy.value
    print(f"{value:>6}  {energy:.9f} Ry  converged={results.runtime_log.converged}")
```

Two details in it are the ones people get wrong. `outdir` is made absolute
because it resolves against your _current directory_, not the manifest's — see
tutorial 01. The manifest copy is written next to the original because
`material:` and `pseudopotentials:` resolve against the manifest's directory,
so a copy in `/tmp` would not find them.

## 1. The cutoff

`ecutwfc` sets the plane-wave basis: every wave with kinetic energy below it is
included. `ecutrho` is the cutoff for the density and potential, conventionally
4× `ecutwfc` for norm-conserving pseudopotentials — the script keeps that ratio.

```bash
python3 scan.py examples/total_energy--semiconductor--silicon/nc.yml method.cutoffs.wavefunction 12 16 20 24 32 40 48
```

Silicon has 2 atoms per cell, so per-atom errors are half the per-cell
differences:

| ecutwfc (Ry) | Total energy (Ry) | ΔE vs 48 Ry (meV/atom) | G-vectors | FFT grid |
| ------------ | ----------------- | ---------------------- | --------- | -------- |
| 12           | −16.877058954     | 247.34                 | 1471      | 20³      |
| 16           | −16.906029928     | 50.26                  | 2421      | 20³      |
| 20           | −16.911436115     | 13.48                  | 3383      | 24³      |
| 24           | −16.912760563     | 4.47                   | 4381      | 24³      |
| 32           | −16.913286775     | 0.89                   | 6735      | 32³      |
| 40           | −16.913409032     | 0.06                   | 9409      | 32³      |
| 48           | −16.913417932     | 0.00                   | 12339     | 36³      |

Monotonic from above, as variational convergence must be. The example's 48 Ry
is comfortable; 32 Ry would already be inside 1 meV/atom. The G-vector count
grows as `ecutwfc^(3/2)`, and cost grows with it — going from 24 to 48 Ry
triples the basis to buy 4.5 meV/atom.

**This is the cutoff for _this pseudopotential_.** A different Si file — and
certainly an ultrasoft or PAW one — has its own requirement. Ultrasoft
datasets are designed to converge at much lower `ecutwfc` with a much higher
`ecutrho` ratio (8–12× rather than 4×). Never carry a cutoff across
pseudopotential families.

## 2. The k-grid: an insulator

Now hold the cutoff at 48 Ry and scan the sampling:

```bash
python3 scan.py examples/total_energy--semiconductor--silicon/nc.yml method.scf.kgrid.dimensions 2 4 6 8 10
```

| Grid | Irreducible k-points | Total energy (Ry) | ΔE vs 10³ (meV/atom) | Pressure (kbar) | Runtime |
| ---- | -------------------- | ----------------- | -------------------- | --------------- | ------- |
| 2³   | 2                    | −16.913417932     | 80.57                | 3.71            | 0.7 s   |
| 4³   | 10                   | −16.925061210     | 1.36                 | 0.01            | 1.2 s   |
| 6³   | 28                   | −16.925253705     | 0.05                 | 0.07            | 2.4 s   |
| 8³   | 60                   | −16.925260460     | 0.00                 | 0.08            | 4.8 s   |
| 10³  | 110                  | −16.925260799     | 0.00                 | 0.08            | 8.3 s   |

**The example you ran in tutorial 01 is 80 meV/atom from converged.** That is
not a small number — it is larger than most energy differences anyone computes
DFT to obtain. The 2×2×2 grid is in the examples because it runs in a second,
not because it is right.

Note the irreducible count: a 10×10×10 grid is 1000 points, but symmetry
reduces it to 110. This is why the cost grows so much more slowly than the grid
does.

Silicon converges fast because it is an insulator: the occupied states are
separated from the empty ones by a gap everywhere in the zone, so the quantity
being integrated is smooth and a coarse quadrature does well.

## 3. The k-grid: a metal

Copper has no gap. The occupation drops from 1 to 0 across the Fermi surface,
and the integrand is discontinuous there. Same scan, same cutoff discipline:

```bash
python3 scan.py examples/total_energy--metal--copper/nc.yml method.scf.kgrid.dimensions 2 4 6 8 10 12 16
```

Copper's cell has one atom, so these are already per-atom:

| Grid | Irreducible k-points | Total energy (Ry) | ΔE vs 16³ (meV/atom) | Pressure (kbar) | Runtime |
| ---- | -------------------- | ----------------- | -------------------- | --------------- | ------- |
| 2³   | 2                    | −386.544070980    | 581.46               | 99.77           | 1.0 s   |
| 4³   | 10                   | −386.583379146    | 46.64                | 55.61           | 2.3 s   |
| 6³   | 28                   | −386.586517423    | 3.94                 | 39.90           | 5.5 s   |
| 8³   | 60                   | −386.586889239    | −1.12                | 41.05           | 12.1 s  |
| 10³  | 110                  | −386.586868754    | −0.84                | 41.73           | 20.1 s  |
| 12³  | 182                  | −386.586814295    | −0.10                | 41.58           | 34.6 s  |
| 16³  | 408                  | −386.586807249    | 0.00                 | 41.67           | 85.9 s  |

Three things here that silicon did not show you:

-   **It is not monotonic.** The 8³ energy is 1.1 meV/atom _below_ the 16³ one,
    and the sequence approaches from below after crossing. There is no
    variational floor for k-sampling. If you scan 2³ → 4³ → 6³ and stop when
    the change gets small, you can stop on a crossing and mistake it for
    convergence.
-   **It is much slower to converge.** 4³ — the grid the shipped copper example
    uses — is 47 meV/atom out. Silicon reached that accuracy at 4³ too, but
    silicon's 4³ error was 1.4 meV/atom.
-   **The pressure converges more slowly than the energy.** At 6³ the energy is
    within 4 meV/atom but the pressure is still 1.8 kbar from its final value,
    and it is still moving at 10³. Energy is a variational quantity; its
    derivatives are not, and they inherit a larger sampling error. **Converge
    the property you actually want**, not the energy as a proxy for it.

## 4. Smearing: the knob metals add

To integrate across a discontinuity, metals need the step function replaced by
something smooth. That is `smearing.degauss` — a width, in Ry. It is a
numerical device, so it needs converging like the others. Use the committed
8×8×8 copper variant, so the grid is not itself the limiting error:

```bash
python3 scan.py examples/total_energy--metal--copper/nc_888.yml method.scf.smearing.degauss 0.005 0.01 0.02 0.05 0.10
```

| degauss (Ry) | Free energy F (Ry) | −TS (Ry)  | E₀ = F − (−TS)/2 (Ry) | Fermi (Ry) | Pressure (kbar) |
| ------------ | ------------------ | --------- | --------------------- | ---------- | --------------- |
| 0.005        | −386.586552299     | −0.000043 | −386.586530862        | 1.2426     | 38.82           |
| 0.01         | −386.586611145     | −0.000161 | −386.586530424        | 1.2413     | 40.26           |
| 0.02         | −386.586889239     | −0.000791 | −386.586493834        | 1.2430     | 41.05           |
| 0.05         | −386.589032029     | −0.005256 | −386.586404171        | 1.2471     | 41.80           |
| 0.10         | −386.598552897     | −0.029232 | −386.583937122        | 1.2586     | 60.46           |

The `total_energy` record is the **free energy** — it already includes the
`temperatureEntropy` (`-TS`) term of `total_energy_contributions`, which is why it drifts by
163 meV/atom across this range while the physics does not. The zero-width
extrapolation `E₀ = F − (−TS)/2` is the quantity to compare: it moves
0.5 meV/atom between 0.005 and 0.02 Ry, and 1.7 meV/atom out to 0.05 Ry.

At `degauss = 0.10` everything breaks down at once — the entropy term reaches
29 mRy (400 meV), E₀ jumps 35 meV/atom, the Fermi level shifts 16 mRy, and the
pressure goes from 42 to 60 kbar. A smearing wide enough to smooth the integrand is also wide
enough to change the physics. Watch `temperatureEntropy`: when it
grows to the size of the energy differences you care about, it is no longer a
numerical convenience.

Smearing and the k-grid are coupled — a wider smearing makes a coarse grid look
converged by blurring exactly the structure the grid is failing to resolve.
Converge them together, or fix a physically defensible smearing (0.01–0.02 Ry
for a simple metal) and converge the grid at that value.

## 5. What to actually do

Absolute total energies are not observables (tutorial 01). Convergence is
therefore never about the energy itself, but about the _difference_ you are
computing — and differences converge faster than absolutes, because errors
common to both sides cancel.

Concretely:

1.  Decide the target quantity and the accuracy you need. "1 meV/atom in the
    total energy" and "1 kbar in the pressure" are different jobs.
2.  Converge the cutoff at a generous k-grid, then the k-grid at the chosen
    cutoff. One pass each is enough when the two are as weakly coupled as they
    are here; re-check the cutoff at the final grid if the answer is close to
    your tolerance.
3.  Scan the **difference** — the two structures you are comparing, at each
    setting — not one endpoint.
4.  Use identical settings on both sides of any comparison. Identical cutoff,
    identical grid _density_ (not the same integers, if the cells differ in
    size), identical pseudopotentials, identical smearing.
5.  Write down what you converged and to what. That table is what makes the
    result defensible.

The last point deserves emphasis: the difference between a number and a result
is the scan you ran to justify it.

## Next

[04 — Band structure and DOS](04-band-structure-and-dos.md): the calculation
that finally gives silicon's real band gap, and why no SCF grid could.
