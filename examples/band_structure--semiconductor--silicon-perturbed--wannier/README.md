# Silicon with its symmetry removed: what runs without a human

The other Wannier examples each carry a number a person worked out by
looking at a band structure. Silicon and GaAs get their trial orbitals placed
on the bonds; GaAs gets `bands: [19, 22]`; copper gets
`frozen_window_above_fermi_ev: -5.0`. This one carries none. Its
`control.interpolation` block is two keys on the way in and two on the way out:

```yaml
interpolation: # capture
    method: wannier
    grid: [6, 6, 6]
```

```yaml
interpolation: # evaluation
    method: wannier
    disentanglement: projectability
    dos_mesh: [16, 16, 16]
```

Everything else comes from the pseudopotential and the defaults. `Si.upf`
carries a 3S and a 3P shell, so the trial orbitals are one s and three p per
atom and the model holds **8** Wannier functions — nobody chose that. The
12-band window is wider than 8, which is what disentanglement is for, and the
thresholds are the protocol's own (0.01 / 0.95, frozen at or below E_F).

## Why the cell is broken on purpose

The structure is mp-149 — the same silicon — with the cell sheared and both
atoms displaced by about 0.06 Å, which leaves the shortest bond at 2.32 Å
against 2.37 Å unperturbed. No symmetry operation maps the cell to itself, the
two atoms are inequivalent, and every degeneracy the diamond structure enforces
is split.

That matters because a Wannier construction has several places it could
accidentally lean on symmetry: degenerate states are exactly where a gauge
choice is ambiguous, and a k-point mesh of a symmetric cell visits the same few
local environments over and over. A model that only works on the structures
somebody tuned it for is not automated. This one is deliberately a structure
nobody tuned.

## What it gets right, and what it does not

|                               | valence bands 1–4              |
| ----------------------------- | ------------------------------ |
| at the 216 mesh points        | **0.08 meV** rms, 0.31 meV max |
| along Γ–X–W–L–Γ, between them | **147 meV** rms, 592 meV max   |

The first row is the one that has to hold. The model is a Fourier transform of
H(k) over the mesh, so evaluated back at a mesh point it must return the DFT
eigenvalues of the chosen subspace, whatever the gauge and whatever the
localisation. It does, to eight hundredths of a meV, with no symmetry to help
it. The whole chain — capture, classification, the subspace iteration, the
gauge, the transform, the evaluation — is doing what it claims.

The second row is the gap. Between mesh points the model is only as good as the
localisation of its Wannier functions. The spread minimiser now runs by default
and it is what takes that row from 202 meV to 147: the same record, the same
subspace, the same Ω_I of 45.043 bohr², with the gauge inside it rotated until
Ω_D + Ω_OD stops falling. That is a 27% improvement and not an order of
magnitude, which is the honest shape of the thing. The mesh, not the gauge, is
the lever — measured on undistorted silicon with this same automatic recipe:

| mesh  | localisation off | localisation on |
| ----- | ---------------- | --------------- |
| 4×4×4 | 1137 meV         | 1100 meV        |
| 6×6×6 | 118 meV          | **82 meV**      |
| 8×8×8 | 17.5 meV         | 19.8 meV        |

Refining the mesh moves the error by a factor of 65 across that range;
localisation moves it by at most 1.4, and at 8×8×8 it is slightly worse. A
driver with a fixed budget should spend it on k-points. The hand-tuned example still reaches 68 meV
with four bond-centred orbitals on an even coarser mesh, because eight atomic
s and p orbitals over a twelve-band window is a different and harder model than
four orbitals over an isolated four-band manifold.

Only above the frozen window is the comparison to DFT bands meaningless: there
the disentangled subspace is deliberately not the lowest bands but the smoothest
ones, so it is compared against nothing here. The density of states integrates
to exactly 8, one per Wannier function.

## So is this automated Wannier?

The mechanics are, and since the spread minimiser landed the gauge is a real
maximally-localised one rather than whatever the projection happened to give.
Nothing in this directory was chosen by looking at a result,
and the same three files with the element symbols changed run on silicon
carbide — a chemistry none of these examples uses — where they produce eight
Wannier functions from silicon's 3s3p and carbon's 2s2p and need nothing else
said. The protocol around them exists too, in
`q3.workflows.wannier.run_automatic_wannier_workflow`, and this structure is one
of the cases `tests/validation/test_wannier_protocol.py` runs through it:

-   **the mesh rule** — 6×6×6 is still written in `nc_nscf.yml` so the manual
    chain above runs in minutes; the workflow chooses the mesh from a 0.2 Å⁻¹
    spacing, which gives this cell 10×10×10.
-   **the quality report** — `q3_utility interpolate` now writes
    `interpolation.quality`: per-function spreads and their variance, the
    centres against the atoms and the bonds, and how far H(R) has decayed. The
    147 meV above is still measured against a direct calculation — by the
    workflow itself now, as the band distance η, which is what it accepts or
    rejects on.
-   **the retry ladder** — a stalled localisation, an infeasible classification
    or a band distance above the gate is retried, up to a recapture on a finer
    mesh, and a model that survives nothing is rejected with the history.
-   **the frozen window** — the published E_F + 2 eV whether the run detects a
    metal (copper) or an insulator (this cell), because it is the window the
    band distance is measured on; the ladder lowers it when it freezes more
    states than there are Wannier functions and widens it when a converged
    model still misses the gate.

```bash
q3 nc_scf.yml
q3 nc_nscf.yml                          # 216 k-points, 12 bands, no Wannier input
q3_utility interpolate nc_interpolate.yml  # 8 orbitals chosen by the pseudopotential
```

| File                 | What it does                                             |
| -------------------- | -------------------------------------------------------- |
| `nc_scf.yml`         | SCF on a 6×6×6 grid, 12 bands                            |
| `nc_nscf.yml`        | the full unreduced 6×6×6 grid; `method` and `grid` only  |
| `nc_interpolate.yml` | Γ–X–W–L–Γ and a 16³ DOS, disentanglement at its defaults |
