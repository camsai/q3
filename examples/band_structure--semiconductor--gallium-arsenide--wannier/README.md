# Gallium arsenide: an isolated manifold that does not start at band 1

Silicon's four valence bands are the lowest four, so a Wannier model of them
needs nothing but the projection gauge. GaAs's four are bands 19 to 22, and this
example is why `control.interpolation.bands` exists.

```bash
q3 nc_scf.yml                          # the density
q3 nc_nscf.yml                         # capture bands 19-22 on a 4x4x4 grid
q3_utility interpolate nc_interpolate.yml # bands on Γ–X–L–Γ and a 16³ DOS
```

## Why the window is offset

The PseudoDojo datasets put the 3d shells of both species in valence: gallium
carries 21 valence electrons and arsenic 23, so the cell has 44 and fills 22
bands. Eighteen of those are semicore, spanning −185 eV to −15 eV. The four sp
valence bands everyone means by "the valence bands of GaAs" sit on top of them.

Measured on this SCF, relative to the valence band maximum:

| Bands | What they are           | Gap to the next band |
| ----- | ----------------------- | -------------------- |
| 1–18  | Ga and As 3s, 3p, 3d    | 2.083 eV above 18    |
| 19–22 | the sp valence manifold | 0.486 eV above 22    |
| 23+   | conduction              |                      |

So the manifold is isolated on both sides, by 2.08 eV below and the PBE band gap
of 0.486 eV above. It needs no disentanglement at all. It was simply unreachable
before, because the capture always took the lowest bands and there was no way to
say otherwise. `bands: [19, 22]` says otherwise.

It is also the one number in this example that the automatic protocol does not
remove. `q3.workflows.wannier` takes every `PP_CHI` shell of a dataset as a trial
orbital, and these datasets carry 3s, 3p and 3d for both atoms — 26 orbitals
spanning the semicore as well as the valence — so the unattended model of GaAs
is the 26-function one, not this 4-function valence model. Selecting the valence
alone needs the semicore rule of the plan, which is not built; until it is,
`bands` stays an override.

That gap of 0.486 eV is PBE doing what PBE does; the experimental gap is 1.42 eV.
Nothing here corrects for that, and nothing needs to: the model reproduces
whatever the underlying calculation says.

## Trial orbitals

Four hydrogenic s orbitals at the Ga–As bond midpoints, the same construction
silicon uses. The Wannier functions of a polar semiconductor actually sit closer
to the anion than the midpoint, and moving the centres 62% and 70% of the way
toward the As was measured here: the largest deviation improves from 495 to 441
meV, while the root mean square does not move at all, staying near 123 meV. On a
4x4x4 grid the mesh spacing dominates the error, not the guess, so the example
keeps the midpoint for simplicity. On a finer grid the anion-shifted guess would
be worth revisiting.

## What comes out

Against a direct calculation on the same density and the same path:

| Band | Root mean square | Largest |
| ---- | ---------------- | ------- |
| 19   | 150 meV          | 383 meV |
| 20   | 186 meV          | 495 meV |
| 21   | 53 meV           | 163 meV |
| 22   | 33 meV           | 102 meV |

124 meV over the manifold, against silicon's 68 meV on an equally coarse grid.
The polar bond is the difference: a symmetric s orbital at the midpoint is a
cruder starting gauge for GaAs than for silicon, and without a spread minimiser
that starting gauge is the final one. The upper two bands, which are the ones a
transport or optical model cares about, do better than the manifold average.

The density of states integrates to exactly 4, one per Wannier function. Note
that `q3_utility` counts Wannier functions rather than electrons; the Python
workflow applies the spin factor of two, this file does not.

## Against Wannier90

The same matrices this example captures — `export_formats: [wannier90]` writes
them as `.amn`, `.mmn`, `.eig` and `.nnkp` — were handed to Wannier90 4.0.1 as
a black box, and its converged result agrees with Q3's: Ω_I to 5×10⁻¹⁰ Å², the minimised Ω to 10⁻⁷ Å², and every centre to 4×10⁻⁵ Å. That is the
comparison none of the numbers above can make, since they all measure the model
against Q3's own direct calculation; it is what says the minimum is the right
one and not merely a lower number than the projection gauge started from.
`tests/validation/test_wannier90_parity.py` repeats it (tier 2; it skips
without the oracle).

## Files

| File                 | What it does                                              |
| -------------------- | --------------------------------------------------------- |
| `nc_scf.yml`         | SCF on a 4x4x4 grid, 22 bands                             |
| `nc_nscf.yml`        | the full unreduced 4x4x4 grid, capturing bands 19–22 only |
| `nc_interpolate.yml` | evaluate on Γ–X–L–Γ and a 16³ DOS mesh                    |
