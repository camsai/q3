# 01 — Your first calculation

**Goal:** build Q3, run a self-consistent field (SCF) calculation on silicon,
and understand what the log is telling you.

## 1. Build

You need a C++17 compiler (GCC 9 or newer, or Clang), CMake, FFTW3 and LAPACK:

```bash
brew install cmake fftw libomp lapack
```

```bash
sudo apt-get update && sudo apt-get install -y cmake build-essential libfftw3-dev libomp-dev liblapack-dev libblas-dev
```

Then clone with submodules (the JSON schemas Q3 generates C++ from live in one):

```bash
git clone --recurse-submodules <repository-url> q3 && cd q3
```

The build generates C++ from those schemas as it goes, which needs a few Python
modules. Create the environment before configuring — a `.venv` at the repository
root is picked up automatically:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[build]"
```

```bash
cmake -S . -B build -DQ3_BUILD_TESTS=ON && cmake --build build -j
```

That produces `build/q3` (the engine) and `build/q3_tests` (the suite). A good
first sanity check is to run the tests:

```bash
ctest --output-on-failure --test-dir build
```

## 2. Run silicon

```bash
./build/q3 examples/total_energy--semiconductor--silicon/nc.yml
```

It finishes in about a second. Before reading the output, look at the input —
it is short, and every DFT calculation you ever run is a variation on it:

```yaml
material: ../data/materials/silicon.json # the crystal: lattice + atom positions
model:
    type: dft
    subtype: gga
    functional: pbe # the physics: which exchange-correlation approximation
method:
    type: pseudopotential
    subtype: norm-conserving # how core electrons are replaced
    cutoffs:
        wavefunction: 48.0 # the numerics: plane-wave basis size, in Ry
        density: 192.0
    pseudopotentials:
        Si: ../data/pseudopotentials/nc/Si.upf
    scf:
        convergence_threshold: 1.0e-08
        number_of_bands: 8
        kgrid:
            dimensions: [2, 2, 2] # how finely the Brillouin zone is sampled
            shifts: [1, 1, 1]
control:
    calculation: scf
    outdir: ./output
    prefix: si_nc
```

Four blocks, and they answer four different questions: **what** you are
computing (`material`), **which physics** (`model`), **with what numerical
approximations** (`method`), and **what to do with it** (`control`). Paths
inside the manifest resolve relative to the manifest's own directory, so the
example works from anywhere.

> **`outdir` is relative to your current directory, not to the manifest.**
> Running the command above from the repository root writes
> `./output/si_nc_results.json` at the repository root. This trips people up;
> `cd` into the example directory first if you want the output beside it.

## 3. Read the log

### The setup block

```
     Reading PP: .../data/pseudopotentials/nc/Si.upf
     UPFv2: Si (NC) Z=4 mesh=1510 num_beta=6 max_l=2 has_nlcc=true
     Spacegroup: Fd-3m (#227, 48 symmetry ops)
     IBZ reduction (internal): 8 -> 2 k-points (48 symmetry ops + time reversal)
```

Read this every time. Three things here catch most input mistakes before you
waste an hour:

-   **`Z=4`** — the pseudopotential treats 4 electrons per silicon atom as
    valence. With 2 atoms, that is the `number of electrons = 8.0000` printed
    below.
-   **`Fd-3m (#227, 48 symmetry ops)`** — Q3 detected diamond structure. If you
    expected a symmetry and see `P1 (#1, 1 symmetry op)`, your coordinates are
    slightly off and the calculation will be needlessly slow.
-   **`8 -> 2 k-points`** — symmetry reduced the sampling to 2 inequivalent
    points. This is a 4× saving, done automatically.

```
     ecutwfc                   = 48.0000 Ry
     FFT grid (rho)            = 36 x 36 x 36
     number of G-vectors       = 12339
     number of bands           = 8
```

The basis Q3 built from your cutoffs. `number of bands = 8` for 8 electrons
means 4 doubly-occupied bands and 4 empty ones — enough for an insulator.

### The SCF loop

```
     iteration #   1     Etot =  -16.917738544 Ry
     iteration #   2     Etot =  -16.912096733 Ry   delta(E) = 5.64e-03   dr2 = 2.52e-02
     iteration #   3     Etot =  -16.913380766 Ry   delta(E) = -1.28e-03   dr2 = 3.20e-04
     iteration #   4     Etot =  -16.913416221 Ry   delta(E) = -3.55e-05   dr2 = 5.08e-06
     iteration #   5     Etot =  -16.913417809 Ry   delta(E) = -1.59e-06   dr2 = 4.32e-07
     iteration #   6     Etot =  -16.913417917 Ry   delta(E) = -1.08e-07   dr2 = 6.33e-08
     iteration #   7     Etot =  -16.913417932 Ry   delta(E) = -1.45e-08   dr2 = 1.07e-09

     convergence achieved after 7 iterations
```

This is the heart of it. DFT is a fixed-point problem: the electron density
determines the potential, the potential determines the orbitals, and the
orbitals determine the density. Q3 iterates until the input and output
densities agree.

-   **`dr2`** is the density residual — how much the density still moved. It is
    the quantity actually compared against `convergence_threshold` (1e-8 here),
    **not** `delta(E)`. Energy converges faster than density, so stopping on
    energy alone would give you a plausible number from an unconverged density.
-   Each iteration drops `dr2` by roughly an order of magnitude. That steady
    geometric decay is what healthy convergence looks like. Oscillation or
    stagnation means the mixer needs help — see
    [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md).
-   One line per iteration is what a run prints by default. Add
    `verbosity: high` under `control` (or pass `--log-level debug`) to see what
    happens inside each one: the Davidson micro-iterations, the per-phase timing
    breakdowns, and the mixer's residual bookkeeping. It changes nothing but the
    output.

### The results

```
     Final Energies (Ry):
     --------------------
     total energy            =  -16.913417932 Ry
```

```
     Pressure = 3.71 kbar
```

Forces here are zero to machine precision (10⁻¹⁹ Ry/bohr) — silicon's atoms sit
at symmetry-fixed positions, so no other answer is possible, which makes this a
useful correctness check. The residual 3.71 kbar pressure is real: the PBE
equilibrium lattice constant differs slightly from the experimental one in the
input file.

**A caution about the total energy.** −16.913417932 Ry is not a physical
observable. It is referenced to the pseudopotential's own zero, so it is
comparable only against another calculation using the same pseudopotentials,
cutoffs and functional. Energy _differences_ between such calculations are the
physics. Tutorial 03 is about making those differences trustworthy.

## 4. Try a variation

Silicon's ultrasoft and PAW datasets are committed too:

```bash
./build/q3 examples/total_energy--semiconductor--silicon/us.yml
```

The total energy will be completely different (−15.717 Ry). That is expected
and it is the point of the caution above: different pseudopotentials use
different energy zeros. Compare _differences_, never absolutes across families.

## Next

[02 — Reading the results](02-reading-results.md): getting these numbers out of
`results.json` programmatically.
