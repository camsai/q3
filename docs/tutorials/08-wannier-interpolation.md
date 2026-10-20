# 08 — Wannier interpolation

**Goal:** turn a 64-k-point DFT calculation into a model that returns silicon's
valence bands at _any_ k-point for about six microseconds each — and learn the
three checks that tell you whether to believe it.

## Why bother

Tutorial 04 got the band gap by diagonalizing at the k-points that mattered.
That works because a band path is a few dozen points. It stops working when you
need thousands: a well-converged density of states, a Fermi surface, an
electron–phonon average, a transport integral. Those want tens of thousands of
k-points, and DFT charges you a diagonalization for every one.

Here is the actual cost on this laptop, from the runs below:

| Step                               | k-points | Wall clock |
| ---------------------------------- | -------- | ---------- |
| DFT (NSCF, the input to the model) | 64       | 2.47 s     |
| Interpolation on a 24×24×24 mesh   | 13,824   | 0.09 s     |
| Interpolation on a 48×48×48 mesh   | 110,592  | 0.65 s     |

Those 110,592 points would have cost roughly seventy minutes of DFT. They cost
0.65 seconds instead — and the 2.47 s that bought the model is already spent.

The trick is not a cheaper diagonalization. It is a change of basis. Bloch
states are delocalized, so ε(k) is a wildly oscillatory function that you can
only sample. Wannier functions are the localized combinations of those same
states, and in that basis the Hamiltonian is a set of hoppings H(R) between
sites a few lattice vectors apart — which decay fast. Keep the ones that
matter and you can Fourier-transform back to _any_ k, not just the ones you
computed. That is the whole idea; everything below is bookkeeping and checks.

It is also the foundation for things Q3 cannot do yet at all: electron–phonon
coupling, Boltzmann transport, and Landauer device conductance all consume a
Wannier Hamiltonian.

## What you will need

`./build/q3` and `./build/q3_utility`, both from tutorial 01's build. No other
code — this chain is entirely Q3. (Quantum ESPRESSO appears nowhere in this
lesson; if you want a second opinion from an independent implementation, the
matrices below are written in the Wannier90 file formats and `wannier90.x` will
read them.)

## 1. The SCF step

Nothing new — an ordinary well-converged SCF, whose density every later step
reads back. Write `wannier_scf.yml`:

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
        convergence_threshold: 1.0e-10
        number_of_bands: 4
        kgrid:
            dimensions: [4, 4, 4]
            shifts: [0, 0, 0]
control:
    calculation: scf
    outdir: ./wannier_output
    prefix: si
```

```bash
./build/q3 wannier_scf.yml
```

```
     convergence achieved after 8 iterations
    total energy         =  -16.911637481 Ry
```

Four bands, because silicon has four valence bands and those are what we are
going to Wannierize. That choice is not a detail — see §9.

## 2. The NSCF step, on the full grid

Now the same Hamiltonian, diagonalized on **every** k-point of the 4×4×4 grid,
with the Wannier block switched on.

Every point matters here, which is the one structural difference from tutorial 04. Ordinary calculations exploit symmetry and keep only the irreducible wedge
— 8 points instead of 64 for this grid. Wannier construction cannot: it needs
overlaps between each k and its neighbours, so it needs the neighbours to
exist. That is why the grid arrives as an explicit `kpoints_list` rather than a
`kgrid`, which would be symmetry-reduced.

Sixty-four rows is more than anyone types. Write `make_wannier_nscf.py`:

```python
"""Write wannier_nscf.yml: the same calculation on all 4x4x4 = 64 k-points."""

MESH = 4
HEADER = """material: examples/data/materials/silicon.json
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
        convergence_threshold: 1.0e-10
        number_of_bands: 4
control:
    calculation: nscf
    outdir: ./wannier_output
    prefix: si
    interpolation:
        method: wannier
        grid: [{mesh}, {mesh}, {mesh}]
        projection_centers:
            - [0.0, 0.0, 0.0]
            - [0.5, 0.0, 0.0]
            - [0.0, 0.5, 0.0]
            - [0.0, 0.0, 0.5]
        projection_decay_bohr: 1.0
        export_formats: [wannier90]
    kpoints_list:
"""

weight = 1.0 / MESH**3
rows = [
    f"        - [{x / MESH:.6f}, {y / MESH:.6f}, {z / MESH:.6f}, {weight:.8f}]"
    for x in range(MESH)
    for y in range(MESH)
    for z in range(MESH)
]
with open("wannier_nscf.yml", "w") as handle:
    handle.write(HEADER.format(mesh=MESH) + "\n".join(rows) + "\n")
print(f"wrote wannier_nscf.yml with {len(rows)} k-points")
```

`kpoints_list` rows are `[kx, ky, kz, weight]` in **crystal** coordinates —
fractions of the reciprocal lattice vectors, which is why a 4×4×4 grid is
multiples of 0.25.

```bash
python make_wannier_nscf.py
./build/q3 wannier_nscf.yml
```

```
NSCF: loaded effective potential from './wannier_output/si.save/effective-potential.q3cp'
NSCF: diagonalizing at 64 explicit k-points
NSCF: building Wannier interface matrices on the 4x4x4 grid (64 k-points) from 4 hydrogenic s orbitals at explicit centres (decay 1 bohr)
     Wrote interpolation record (wannier): ./wannier_output/si.save/interpolation.q3ip
wrote ./wannier_output/si.mmn, .amn, .eig and .nnkp
```

The record, `si.save/interpolation.q3ip`, is the model: everything the
evaluation step needs, and what Q3 reads back. The four files beside it exist
because the input said `export_formats: [wannier90]` — the same matrices in the
Wannier90 file formats, which is the interface between the plane-wave world and
any Wannier code:

| File    | Holds                                                     | Why it is needed                                                        |
| ------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| `.eig`  | ε_nk, the eigenvalues                                     | what the model has to reproduce                                         |
| `.mmn`  | M_mn(k,b) = ⟨u_mk\|u_n,k+b⟩, overlaps with neighbouring k | how the phase is tracked from one k to the next — the localization data |
| `.amn`  | A_mn(k) = ⟨ψ_mk\|g_n⟩, projections onto trial orbitals    | where the Wannier functions are told to sit                             |
| `.nnkp` | which k-points are neighbours, and the b-vectors          | the finite-difference stencil in reciprocal space                       |

They are written in the Wannier90 formats rather than a private one, so an
independent implementation can read exactly what Q3 produced.

## 3. Evaluating the model

Evaluating the saved model is a separate, cheap step with its own input: a
manifest with `calculation: interpolation` whose `outdir` and `prefix` point at
the record, and which lists the k-points to evaluate. It runs with `q3_utility`,
the utility binary, not with `q3` — nothing here touches plane waves. Start by
evaluating at the 64 mesh points themselves; §4 says why. Write
`make_wannier_interpolate.py`:

```python
"""Write wannier_interpolate.yml: evaluate the saved model at the 4x4x4 mesh points."""

MESH = 4
HEADER = """material: examples/data/materials/silicon.json
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
        number_of_bands: 4
control:
    calculation: interpolation
    outdir: ./wannier_output
    prefix: si
    interpolation:
        method: wannier
    kpoints_list:
"""

rows = [
    f"        - [{x / MESH:.6f}, {y / MESH:.6f}, {z / MESH:.6f}, 1.0]"
    for x in range(MESH)
    for y in range(MESH)
    for z in range(MESH)
]
with open("wannier_interpolate.yml", "w") as handle:
    handle.write(HEADER + "\n".join(rows) + "\n")
print(f"wrote wannier_interpolate.yml with {len(rows)} k-points")
```

```bash
python make_wannier_interpolate.py
./build/q3_utility interpolate wannier_interpolate.yml
```

```
Wigner-Seitz supercell: 93 lattice vectors, total weight 64.000000 (expected 64)
q3_utility interpolate: wannier (projection gauge): 4 orbitals on 93 lattice vectors from ./wannier_output/si.save/interpolation.q3ip
evaluated 64 k-points
Results written to: ./wannier_output/si_interpolation_results.json
```

**Read the first line.** The Hamiltonian is being re-expressed on a set of
real-space lattice vectors R — the Wigner-Seitz cell of the 4×4×4 supercell,
93 of them. Vectors on its boundary are shared between neighbouring cells and
carry fractional weight; get that bookkeeping wrong and the interpolation is
subtly and silently wrong everywhere. The weights must sum to the number of
k-points, and `64.000000 (expected 64)` is the code checking its own arithmetic
in front of you. If those two numbers ever disagree, stop.

## 4. The round trip — the check that catches almost everything

Here is the sharpest test available, and it costs nothing.

At the 64 k-points you actually computed, going k → R → k is an exact
mathematical identity: the forward and backward Fourier transforms are inverses.
So evaluating _at the mesh points_ — which is exactly what §3 just did — must
return the DFT eigenvalues to machine precision. Not approximately. Exactly.

```python
import json
import numpy as np

RYDBERG_TO_EV = 13.605693122994
results = json.load(open("wannier_output/si_interpolation_results.json"))
# eigenvalues_ry is [k-point][spin][band], like every Q3 results file; one spin here.
interpolated = np.array(results["eigenvalues_ry"])[:, 0, :] * RYDBERG_TO_EV

reference = np.zeros_like(interpolated)
for line in open("wannier_output/si.eig"):       # band, kpoint, energy in eV
    band, kpoint, energy = line.split()
    reference[int(kpoint) - 1, int(band) - 1] = float(energy)

print(f"max deviation = {np.abs(interpolated - np.sort(reference, axis=1)).max():.3e} eV")
```

```
max deviation = 9.770e-15 eV
```

Fourteen digits. This one number exercises the Fourier transforms, the
Wigner-Seitz vectors and their degeneracy weights, the gauge matrices and their
unitarity, and the file readers. Almost any bookkeeping error anywhere in that
chain destroys it. What it does _not_ test is physics — it would look exactly
this good with badly chosen Wannier functions, because the identity holds
whatever gauge you picked. For physics, keep reading.

## 5. Bands along a path

A path needs no script: an evaluation manifest may carry `band_path` vertices
with explicit coordinates, walked `band_path_points_per_segment` steps per
segment, and a `dos_mesh` for §6 at the same time. Write `wannier_bands.yml`
— the same `material`, `model` and `method` as before, with this `control`:

```yaml
control:
    calculation: interpolation
    outdir: ./wannier_output
    prefix: si
    interpolation:
        method: wannier
        dos_mesh: [24, 24, 24]
    dos_broadening: 0.01
    band_path:
        - label: G
          coordinates: [0.0, 0.0, 0.0]
        - label: X
          coordinates: [0.0, 0.5, 0.5]
        - label: L
          coordinates: [0.5, 0.5, 0.5]
        - label: G
          coordinates: [0.0, 0.0, 0.0]
    band_path_points_per_segment: 20
```

The labels are annotations here; the coordinates are what is evaluated. (Bare
labels, as tutorial 04 wrote them, are resolved by `q3.workflows.interpolation`
through the same lattice table — `q3_utility` itself has none.)

```bash
./build/q3_utility interpolate wannier_bands.yml --output wannier_output/si_bands.json
```

```
evaluated 61 k-points
evaluated a density of states on a 24x24x24 mesh
Results written to: wannier_output/si_bands.json
```

The eigenvalues land in `wannier_output/si_bands.json` under `eigenvalues_ry`,
one row per k-point. In eV, at the three high-symmetry points:

| Point | Interpolated valence eigenvalues (eV) |
| ----- | ------------------------------------- |
| Γ     | −5.7741, 6.0550, 6.0550, 6.0550       |
| X     | −1.7043, −1.7043, 3.2458, 3.2458      |
| L     | −3.4938, −0.8306, 4.8684, 4.8684      |

Look at the degeneracies. The top of silicon's valence band at Γ is the
three-fold Γ25′ state; at X the four bands pair into two doublets; at L the
upper two are degenerate and the lower two are not. **Nothing in the
construction enforces any of this.** The model is 93 hopping matrices fitted to
64 k-points; it has no idea the crystal has symmetry. Getting the degeneracies
right — to all printed digits — is the interpolation telling you it captured
the actual Hamiltonian rather than something that merely passes through the
sampled points. The Γ valence bandwidth, 11.8291 eV, is likewise the physical
value.

## 6. Density of states

`dos_mesh: [24, 24, 24]` evaluated the model on 13,824 k-points, which is where
the speed argument stops being abstract. The same sanity check tutorial 04 used
applies:

```python
import json
import numpy as np

dos = json.load(open("wannier_output/si_bands.json"))["density_of_states"]
energies = np.array(dos["energies_ry"])
density = np.array(dos["dos"])
print(f"integral = {np.trapezoid(density, energies):.4f}")
```

```
integral = 4.0000
```

Four — the number of bands in the model, one state each. (Eight electrons, once
you count spin: this DOS counts states, not electrons.) A wrong normalization,
a mis-weighted k-mesh or a truncated energy window would all show up here first.

## 7. How wrong is it between the mesh points?

At the mesh points the model is exact by construction, so its error lives
entirely _between_ them, and measuring it needs a direct calculation on the same
path. Run one — same SCF density, `number_of_bands: 4`, the 61 path k-points as
an explicit `kpoints_list`, into a separate `outdir` — and compare:

| Grid  | max deviation | mean | rms      |
| ----- | ------------- | ---- | -------- |
| 4×4×4 | 266.3 meV     | 45.0 | 67.8 meV |
| 8×8×8 | 49.5 meV      | 5.5  | 10.0 meV |

Doubling the grid cut the error by a factor of about seven. **That convergence
is the point** — it is what separates interpolation error from a bug. A
bookkeeping mistake does not improve when you add k-points; it just gets
re-expressed on a finer mesh.

Two traps worth knowing before you measure this yourself.

**The error is exactly zero at Γ, X and L.** All three are 4×4×4 mesh points
(0.5 = 2/4), so the round-trip identity applies and the deviation there is
0.0000 meV. High-symmetry points are the _worst_ possible places to judge
interpolation quality, and they are exactly where the eye is drawn. The error
peaks between them — for this path, in the L→Γ segment.

**Compare against the same density.** The 8×8×8 row above needs a direct
calculation that reloaded the 8×8×8 SCF density. Comparing the 8×8×8 model
against the 4×4×4 direct bands gives 36.6 meV rms instead of 10.0 — nearly four
times too large, and it looks exactly like interpolation failing to converge.
It is not: the two SCF runs converged to different densities (−16.911637 Ry at
4×4×4 against −16.924975 Ry at 8×8×8), and that difference is what the number
measures. Interpolation error and k-grid convergence error are different
quantities; do not let one masquerade as the other.

## 8. Where the trial orbitals go, and why it matters

`projection_centers` in §2 placed four s-type orbitals at the **Si–Si
bond midpoints** — `(0,0,0)`, `(½,0,0)`, `(0,½,0)`, `(0,0,½)` for this
structure, each exactly √3/8 of a lattice parameter from _both_ atoms.

That is the physics: silicon's valence electrons live in the bonds, so its
valence Wannier functions do too. The projections are the only place you tell
the code where to look, and with a one-shot projection gauge and no spread
minimiser, that initial guess is most of the answer. Moving those four orbitals
and changing nothing else:

| Trial-orbital centres             | rms error      |
| --------------------------------- | -------------- |
| bond midpoints                    | 67.8 meV       |
| displaced 0.05 along (111)        | 72.0 meV       |
| a plausible-looking but wrong set | 194.6 meV      |
| two on atoms, two on bonds        | refuses to run |

A small displacement barely matters — the gauge is forgiving about where
exactly the centre sits. Getting the _topology_ wrong costs a factor of three.
And a set that fails to span the manifold at all does not quietly degrade; it
stops:

```
ProjectionGauge::compute: the trial orbitals do not span the target manifold at
this k-point -- smallest singular value of A is 4.155597e-09, below the
tolerance 1.000000e-08. Choose different trial orbitals or a different band
selection; a gauge built from a rank-deficient A is meaningless.
```

The middle row of that table is the one to take seriously. A misplaced centre
produces a model that runs, passes the round-trip check to twelve digits, and
returns bands that are three times worse. **The round trip cannot see it**; only
§7's comparison against a direct calculation can. Derive your centres from the
structure rather than writing them down.

If you omit `projection_centers` entirely, Q3 falls back to the
pseudopotential's own atom-centred orbitals — for silicon, 3s+3p on each of two
atoms, so eight Wannier functions, which then needs at least eight bands. Those
eight lowest bands are _not_ an isolated manifold: the conduction bands mix with
higher ones, so a model built from them by projection alone comes out around
900 meV rms. That case is what the next section is about.

## 9. Entangled windows, and localizing the gauge

This lesson worked because silicon's four valence bands are an **isolated
manifold** — separated by a gap from everything above, at every k-point. Two
further capabilities handle the case where they are not, and both are evaluation-
step keys, so the same captured record serves a run with them and a run without.

**Disentanglement** picks a smooth N-dimensional subspace out of a wider window.
Add `disentanglement: projectability` to the evaluation manifest and give the
capture more bands than you have orbitals:

```yaml
interpolation:
    method: wannier
    disentanglement: projectability
```

Every band is classified by its projectability p = Σ|A|² onto the trial orbitals
and by its energy: frozen (kept exactly), candidate (offered to the iteration),
or discarded. The Souza–Marzari–Vanderbilt iteration then chooses the subspace
that varies most smoothly across the zone. `examples/band_structure--metal--copper--wannier`
is the worked case — a 14-band window down to 10 states, where no band range
works at all because the sp band leaves the d manifold and comes back.

**Localization** then minimizes the Marzari–Vanderbilt spread over the gauge
inside that subspace. It is on by default (`localization: auto`), and it matters
for a reason worth being precise about: the projection gauge is exact _at_ the
mesh points whatever it does, so a model without localization looks perfect
where it was fitted and is only as good as its trial orbitals everywhere else —
which is where an interpolator is actually used. Set `localization: none` to see
the difference for yourself on any of the examples.

How much it buys depends on how good the starting guess was:

| example                                      | off-mesh rms, `none` → `auto` |
| -------------------------------------------- | ----------------------------- |
| silicon, bond-centred orbitals               | 68 → 67 meV                   |
| gallium arsenide                             | 124 → 117 meV                 |
| copper                                       | 112 → 89 meV                  |
| silicon, symmetry-free cell, atomic orbitals | 202 → 147 meV                 |
| Si₃C rattled alloy                           | 71 → 32 meV                   |

The first four start from good trial orbitals on coarse meshes, where the mesh
spacing dominates the gauge. The last is the case localization was built for.

## 10. The protocol around the mechanics

Everything above was driven by hand: you wrote `grid:`, you chose the
projections, and a test — not the run — measured how wrong the model was between
mesh points. The protocol that makes the mechanics unattended is in the code
too, and it is worth knowing what each piece does even when you drive the steps
yourself:

-   **The mesh rule.** `run_interpolation_workflow` no longer needs `grid`: it
    takes n_i = ceil(|b_i| / 0.2 Å⁻¹) per reciprocal vector, rounded up to even
    and never below four, which gives silicon 10×10×10 (its 4×4×4, 6×6×6 and
    8×8×8 are the spacings 0.5, 0.34 and 0.25). The mesh is the lever that
    matters most — on undistorted silicon the automatic recipe gives 826 / 110 /
    12.6 meV at 4×4×4 / 6×6×6 / 8×8×8, against at most 1.4× from the gauge — so
    it is the rule the protocol applies first.
-   **The quality report.** `q3_utility interpolate` writes `interpolation.quality`
    beside `spread`: one spread Ω_n per Wannier function and their variance (the
    published no-reference criterion), the distance of every centre to the
    nearest atom and bond midpoint, and how far H(R) has decayed by the edge of
    its Wigner-Seitz domain — the largest term on the outermost shell against
    the largest hopping, which is 2.0×10⁻³ for the 4×4×4 automatic silicon model
    and 6.8×10⁻⁵ at 8×8×8, the two ends of the 826 → 12.6 meV curve. Read it
    before you trust a model.
-   **The frozen window.** `frozen_window_above_fermi_ev` defaults to `auto`,
    the published E_F + 2 eV for metal and insulator alike — the same window
    the band distance is measured on, which is not a coincidence: a state the
    distance weighs is reproduced only if it is frozen. Silicon at 12×12×12
    gives 220 meV with the occupied manifold alone frozen and 14 meV at
    E_F + 2 eV, the whole difference being the lowest conduction band. Whether
    the system is a metal (a band crossing E_F) is decided and reported, and a
    state inside the window that the trial orbitals cannot represent is never
    frozen — freezing it makes the gauge singular — but reported under
    `interpolation.disentanglement`. Copper's example shows that, and what its
    4p-less dataset costs.
-   **The retry ladder.** `q3.workflows.wannier.run_automatic_wannier_workflow`
    runs the whole chain from an SCF manifest with nothing else said: the mesh,
    the trial orbitals and the band window from the pseudopotentials (every
    occupied shell, and an empty one only where the atom has no valence shell
    of that l, so a PAW or ultrasoft dataset's second s and p channels stay
    out), a reference calculation half a mesh step off every mesh point, the
    band distance η of Vitale et al. (2020) against it, and a ladder that
    retries a disentanglement that ran out of iterations under the tolerance
    its last spreads call for, retries a stalled localisation, backs a metal's
    window off to the occupied manifold
    when its dataset cannot fill the window, lowers a window that freezes too
    much and widens one whose converged model still misses the gate (silicon
    at 12×12×12: 14 meV at E_F + 2 eV, 5 meV at E_F + 3 eV), walks the
    published grid of projectability thresholds, halves the mesh spacing, and
    rejects with the whole history when nothing works. It accepts at
    η ≤ 10 meV and η_max ≤ 100 meV and writes the verdict into the results
    file's `quality`.
-   **The campaign.** `python scripts/validation/run_wannier_campaign.py` runs
    a fixed roster of structures from the eight elements Q3 ships datasets for
    and produces one table; `tests/validation/test_wannier_campaign.py` runs it.
    It resumes from that table by default, rerunning only the rows that ended
    in an error, so an interrupted campaign continues where it stopped.
    `--only` takes a family (`copper`) or one structure's full name
    (`copper--doubled`), comma-separated for several, and `--exact` restricts
    it to the name itself — which is how you reach the pristine case of a
    family, since it carries the family's own name. Narrowing the work does not
    narrow the record: the table keeps every row it already had, which is what
    lets a long roster be continued one family at a time.

Still hand-set: `projection_centers` (silicon's bond-centred orbitals) and
`bands` (gallium arsenide) remain as overrides, because semicore shells are
still trial orbitals — the rule leaves out an augmented dataset's repeated
empty channels, not gallium's and arsenic's 3d; no SCDM fallback
exists, since no shipped dataset lacks atomic wavefunctions; and spin-polarized
capture is unconfirmed. `KNOWN-ISSUES.md`, `FUNCTIONALITY.md` and `TODOs.md`
§1.3 carry the current state.

## Cleaning up

```bash
rm -rf wannier_output wannier_scf.yml wannier_nscf.yml make_wannier_nscf.py
rm -f wannier_interpolate.yml make_wannier_interpolate.py wannier_bands.yml
```

## Next

You have reached the end of the numbered lessons. [07 —
Contributing](07-contributing.md) is the one to read next if you want to change
the code; leaving semicore shells out of the trial set, named at the end of
§10, is the well-defined change left, and the campaign table is where the
protocol's thresholds get re-examined.
