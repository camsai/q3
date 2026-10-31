# Copper: an entangled manifold, disentangled

Silicon's valence bands are isolated, so a Wannier model of them needs only the
projection gauge. Copper's are not, and this example is the case that motivates
disentanglement.

```bash
q3 nc_scf.yml                          # the density
q3 nc_nscf.yml                         # capture a 14-band window on a 4x4x4 grid
q3_utility interpolate nc_interpolate.yml # choose 10 states from it, then evaluate
```

## Why copper cannot be done the silicon way

The norm-conserving dataset carries 19 valence electrons in four shells, 3s, 3p,
3d and 4s, which render as ten pseudo-atomic trial orbitals. Ten Wannier
functions, then. But the lowest ten bands are not the same ten states everywhere
in the Brillouin zone: the free-electron-like sp band runs from about 9 eV below
the Fermi level up to 6 eV above it and back down, crossing through the d
manifold. Measured on an 8x8x8 grid, at 17 of 512 k-points the tenth band has
**zero** overlap with any atomic orbital, so the projection matrix loses rank and
`q3_utility` refuses the model rather than returning a meaningless one.

No band range fixes that. Every cut between bands 5 and 9 falls inside a set that
is exactly degenerate at some k-point, and a degenerate pair cannot be split.

## What disentanglement does instead

It captures a window wider than the model — fourteen bands here — classifies
every band at every k-point by its projectability, and then chooses the
ten-dimensional subspace that varies most smoothly across the zone. The choice
is made from the neighbour overlaps M_mn(k,b) the capture already stores.

One setting in `nc_interpolate.yml` deserves explanation, and one absence.

`disentanglement: projectability` turns it on. Without it the run is refused,
because a fourteen-band window cannot fill ten Wannier functions on its own.

The frozen window is **not** in the manifest, so the run resolves it itself
(`frozen_window_above_fermi_ev: auto`, the default): one band crosses the Fermi
level, so copper is a metal and gets the published E_F + 2 eV. The log then
says two more things, and both are in the results file under
`construction.disentanglement` and `quality`. Band 10, the
free-electron-like sp band, lies inside that window at 7 of the 64 k-points
with a projectability below the minimum, and is **excluded from the freeze**
there: this dataset has no 4p shell, so nothing in the trial set can represent
that state, and freezing a state the trial orbitals cannot reach makes the
gauge singular by construction. And the localisation **stops short**: at the
other 57 k-points the same band is barely reachable and is frozen, the tenth
Wannier function built on it spreads to 8.1 bohr² against 0.7 to 2.2 for the
other nine (the quality block flags it as an outlier), and the gauge never
becomes stationary. The description says `localization stalled`.

That is an honest limitation of the dataset, reported rather than hidden, and
it is what the automatic workflow's retry ladder acts on:
`q3.workflows.wannier.run_automatic_wannier_workflow` retries a stalled metal
whose window holds such states with the occupied manifold alone frozen
(`frozen_window_above_fermi_ev: 0.0`), which converges in 66 iterations to
Ω = 17.21 bohr² — the same model, to every digit, that this example shipped
with a hand-set `-5.0` before the rule and the ladder existed (stage 4.4 of the
protocol removed the number). The numbers below describe that model; to
reproduce it by hand, add `frozen_window_above_fermi_ev: 0.0` to the manifest.
A dataset carrying 4p, or a trial set mixing atomic orbitals with an s orbital
at the interstitial site, would let the Fermi-surface state be frozen and kept.

## What comes out

With the occupied-manifold window the subspace iteration converges in about
twenty iterations. Against a direct calculation on the same density and the
same path, on this 4x4x4 grid:

| Bands                           | Mean deviation |
| ------------------------------- | -------------- |
| 3p semicore                     | 2 to 3 meV     |
| the 3d manifold                 | 12 to 87 meV   |
| the state above the Fermi level | ~350 meV       |

The d bands, which are what a copper tight-binding model is for, land in the same
range silicon's valence bands do on an equally coarse grid. The worst band is the
one nearest the entanglement, which is where the remaining error should be. A
denser capture grid improves all of them; 4x4x4 is chosen here so the example
runs in a couple of minutes.

The density of states integrates to exactly 10, one per Wannier function. Note
that `q3_utility` counts Wannier functions, not electrons: the Python workflow
applies the spin factor of two, this file does not.

## Against Wannier90

The same matrices this example captures — `export_formats: [wannier90]` writes
them as `.amn`, `.mmn`, `.eig` and `.nnkp` — were handed to Wannier90 4.0.1 as
a black box, and its converged result agrees with Q3's: Ω_I to 2×10⁻¹⁰ Å² — the disentangled subspace itself — the minimised Ω to 6×10⁻⁸ Å², and every centre to 3×10⁻⁶ Å, with both codes freezing by projectability. That is the
comparison none of the numbers above can make, since they all measure the model
against Q3's own direct calculation; it is what says the minimum is the right
one and not merely a lower number than the projection gauge started from.
`tests/validation/test_wannier90_parity.py` repeats it (tier 2; it skips
without the oracle).

## Files

| File                 | What it does                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| `nc_scf.yml`         | SCF on an 8x8x8 shifted grid, gaussian smearing, 14 bands                                                |
| `nc_nscf.yml`        | the full unreduced 4x4x4 grid, capturing the interface matrices                                          |
| `nc_interpolate.yml` | disentangle to 10 states with the window resolved automatically, then bands on Γ–X–W–K–Γ–L and a 16³ DOS |
