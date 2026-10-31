# Wannier-interpolated band structure and DOS — silicon

Interpolates silicon's valence bands and density of states from a model built
out of Wannier functions. **Every step is Q3's own** — the SCF, the NSCF that
builds the model, and the evaluation.

```bash
cd examples/band_structure--semiconductor--silicon--wannier
../../build/q3 nc_scf.yml
../../build/q3 nc_nscf.yml
../../build/q3_utility interpolate nc_interpolate.yml
```

Needs `build/q3` and `build/q3_utility`, both from the default build. Quantum
ESPRESSO is not required; the validation suite uses it for a cross-check.

The same chain from Python, with the k-path resolved from labels:

```python
from q3.workflows import run_interpolation_workflow

result = run_interpolation_workflow(
    "examples/band_structure--semiconductor--silicon--wannier/nc_scf.yml",
    grid=(4, 4, 4),
    projection_centers=[(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)],
    band_path=["G", "X", "L", "G"],
    dos_mesh=(24, 24, 24),
    dos_broadening_ry=0.01,
)
result["band_structure"]["eigenvalues_ry"]
result["density_of_states"]["dos"]
```

`grid` may be left out: the workflow then chooses it from a 0.2 Å⁻¹ spacing,
which gives silicon 10×10×10 (`result["grid"]` says what it chose). The
bond-centred `projection_centers` stay a choice a person makes — the automatic
recipe uses the pseudopotential's atomic s and p instead, and
`q3.workflows.wannier.run_automatic_wannier_workflow` runs that recipe end to end
with nothing said at all.

## What it does

```
q3 scf  ->  q3 nscf (full grid; builds the model, saves si.save/interpolation.q3ip)
                                  |
                                  v
            q3_utility interpolate  ->  bands along the path + DOS on the mesh
```

`control.interpolation` in `nc_nscf.yml` makes the NSCF run build the model
from its converged states — for `method: wannier`, the trial-orbital
projections A_mn(k), the overlaps M_mn(k,b) and the eigenvalues on the full
4×4×4 grid — and save it as the **interpolation record**,
`output/si.save/interpolation.q3ip`. `export_formats: [wannier90]` additionally
writes the same matrices as `output/si.mmn/.amn/.eig/.nnkp`, so `wannier90.x`
can localize them if you want a second opinion from an independent
implementation.

`nc_interpolate.yml` (`calculation: interpolation`) names that record through
its `outdir`/`prefix`, and `q3_utility interpolate` evaluates it: the projection
gauge U(k) = A(A†A)^(−1/2), the transform to a real-space Hamiltonian H(R) on
the Wigner-Seitz supercell, and a diagonalization at every k-point asked for —
the Γ–X–L–Γ path for bands, a 24×24×24 mesh for the DOS. Evaluating the saved
model is cheap; changing the path or the mesh means re-running only this step.

## What you should see

`output/si_interpolation_results.json` holds, under `properties`, the
`eigenvalues` (one entry per path k-point, listed like the engine's own results
file) and the `density_of_states`, and the model's diagnostics under
`construction` and `quality`:

```
Wigner-Seitz supercell: 93 lattice vectors, total weight 64.000000 (expected 64)
q3_utility interpolate: wannier (projection gauge): 4 orbitals on 93 lattice vectors from ./output/si.save/interpolation.q3ip
evaluated 61 k-points
evaluated a density of states on a 24x24x24 mesh
```

Two numbers mean quite different things.

The **round trip** is an exact identity: at the coarse mesh points the k → R →
k transform is its own inverse, so evaluating there must return the NSCF
eigenvalues to machine precision (9.8e-15 eV measured). It tests the Fourier
transform and the Wigner-Seitz degeneracy weights, and says nothing about
physics. The **band distance** against a direct calculation is real
interpolation error, measured between the mesh points where nothing constrains
the model, and it shrinks with mesh density:

|                    | band distance |
| ------------------ | ------------- |
| 4×4×4 (this input) | ~68 meV       |
| 8×8×8              | ~9.5 meV      |

That convergence — roughly a factor of seven for a doubling — is the point. It
is what distinguishes genuine interpolation error from a bug: a bookkeeping
mistake would not improve on a finer mesh. The 8×8×8 model needs an 8×8×8 grid
in both `nc_scf.yml` and `nc_nscf.yml` (the Python workflow's `grid` argument
does both) and takes a few minutes.

## Why the trial orbitals sit on the bonds

`nc_nscf.yml` places four s orbitals at the Si–Si **bond midpoints** via
`control.interpolation.projection_centers`, rather than using the
pseudopotential's atom-centred orbitals.

Silicon's valence Wannier functions genuinely live on the bonds. With only a
one-shot projection gauge and no minimiser, that initial guess is most of the
answer, and moving the four orbitals while changing nothing else shows it:

| Trial-orbital centres             | rms deviation from a direct calculation |
| --------------------------------- | --------------------------------------- |
| bond midpoints                    | 67.8 meV                                |
| displaced 0.05 along (111)        | 72.0 meV                                |
| a plausible-looking but wrong set | 194.6 meV                               |
| two on atoms, two on bonds        | refuses to run (rank-deficient A)       |

Small displacements barely matter; wrong topology costs a factor of three.

Omitting `projection_centers` falls back to the pseudopotential's atom-centred
orbitals — 3s+3p on two atoms, so eight Wannier functions needing eight bands.
That is not the same calculation with different projections: Si's eight lowest
bands are not an isolated manifold, and without the disentangler the resulting
model is around 900 meV rms.

The centres are derived from the atom positions rather than written down —
`(0,0,0)`, `(½,0,0)`, `(0,½,0)`, `(0,0,½)` for `silicon.json`'s placement,
each exactly √3/8 of a lattice parameter from _both_ atoms. A misplaced
centre does not fail loudly; it quietly degrades the gauge, which is how it
was caught here in the first place (195 meV instead of 68).

## Agreement with Quantum ESPRESSO

`tests/validation/test_wannier_interpolation.py` runs QE's own scf/nscf/bands
and `pw2wannier90.x` on the same cell (`reference/`), imports QE's matrices
with `q3_utility convert --from wannier90`, and compares. The two codes agree on
the underlying DFT:

|                                      |                                                        |
| ------------------------------------ | ------------------------------------------------------ |
| total energy                         | −16.911637 Ry (Q3) vs −16.911639 Ry (QE) — **1.9 µRy** |
| valence eigenvalues, all 64 k-points | max **0.006 meV**, mean 0.004 meV                      |
| Γ valence bandwidth                  | 11.8291 eV in both                                     |

Deviations above 0.1 meV occur only in the topmost requested band, the
expected signature of the highest state in a truncated set converging slowest
with no buffer above it.

The interface matrices themselves are compared one level up: the same path
interpolated from Q3's record and from `pw2wannier90.x`'s files agrees to
**0.6 meV rms** (2.1 meV worst state), a hundredth of the 68 meV both share
against the direct bands. The residual is a convention, not a discrepancy: Q3
Löwdin-orthonormalizes the trial orbitals before projecting, pw2wannier90
projects onto the raw guiding functions, so the two projection gauges differ
by construction. (`reference/si.win` sets `zona = 1.0` to match Q3's 1 bohr
decay — pw2wannier90 applies it in inverse bohr, whatever the Wannier90 guide
says about Ångström; see the comment there.)

**Both sides must describe the same crystal for any of this to mean
anything.** `reference/scf.in` states `celldm(1) = 10.3344` bohr to match
`silicon.json`'s primitive |a| of 3.866976 Å. An earlier version of this
example used 10.2 there — a 1.3 % compressed cell — and produced an apparent
400 meV Q3-vs-QE eigenvalue disagreement that was entirely the mismatch.

## What is missing, and what it would buy

There is **no maximal localization** yet — no spread minimiser, no
disentanglement. Two consequences:

-   The 9.5 meV at 8×8×8 would improve; the target for this fixture is 5 meV,
    and closing that gap is what the minimiser is for.
-   Entangled manifolds are out of reach entirely. This example works because
    silicon's four valence bands are an isolated manifold. A metal, or any
    valence+conduction window, needs the disentangler before it can be
    Wannierized at all.

`TODOs.md` §1.3 tracks the full list, including the other interpolation
methods the same `control.interpolation` block is meant to grow.

## Against Wannier90

The same matrices this example captures — `export_formats: [wannier90]` writes
them as `.amn`, `.mmn`, `.eig` and `.nnkp` — were handed to Wannier90 4.0.1 as
a black box, and its converged result agrees with Q3's: Ω_I to 2×10⁻⁶ Å², the minimised Ω to 2×10⁻⁶ Å², and every centre to 10⁻⁶ Å. That is the
comparison none of the numbers above can make, since they all measure the model
against Q3's own direct calculation; it is what says the minimum is the right
one and not merely a lower number than the projection gauge started from.
`tests/validation/test_wannier90_parity.py` repeats it (tier 2; it skips
without the oracle).

## Files

|                            |                                                          |
| -------------------------- | -------------------------------------------------------- |
| `nc_scf.yml`               | Q3 SCF, 48/192 Ry, norm-conserving, 4×4×4                |
| `nc_nscf.yml`              | Q3 NSCF on the full grid, with the `interpolation` block |
| `nc_interpolate.yml`       | the evaluation manifest for `q3_utility interpolate`     |
| `reference/*.in`, `*.win`  | QE inputs, used by the validation suite's cross-check    |
| `reference/reference.json` | committed gauge-invariant summary of the QE chain        |

The chains' own outputs (the record, `.mmn`/`.amn`/`.eig`/`.nnkp`, `.out`) are
regenerable and are not committed; `reference.json` records the
gauge-invariant quantities a cross-code comparison can legitimately use.
Regenerate it with `tests/validation/refresh_wannier_references.py`.
