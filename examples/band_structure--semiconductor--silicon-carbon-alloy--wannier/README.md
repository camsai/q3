# A rattled substitutional alloy: four atoms, two elements, no symmetry

The other Wannier examples are one or two atoms of a single element in a cell
with symmetry. This one is deliberately none of those things:

-   a **2×1×1 supercell** of mp-149, so four atoms rather than two;
-   one silicon **substituted by carbon**, so two chemistries in one cell;
-   every atom **displaced by up to 12%** of a bond length, seeded and fixed;
-   the cell **sheared 5%**, so the lattice itself is triclinic.

The shortest contact that leaves is 2.008 Å, between the carbon and a silicon,
against an equilibrium Si–C bond of about 1.89 Å — stretched, not a collision.
No symmetry operation maps the cell to itself and no two atoms are equivalent.

```bash
q3 nc_scf.yml
q3 nc_nscf.yml                          # 64 k-points, 20 bands, no Wannier input
q3_utility interpolate nc_interpolate.yml  # 16 orbitals chosen by the pseudopotentials
```

## What is automatic here

Nothing in these three files was chosen by looking at a result. The
`control.interpolation` block is `method` and `grid` going in, and `method`,
`disentanglement` and `dos_mesh` coming out. `grid` is the one number a person
still writes, for the manual chain; `q3.workflows.wannier` chooses it from a
0.2 Å⁻¹ spacing (this cell gets 4×8×8) and judges the model it builds by the band
distance against a direct calculation, which is what the 32 meV below would be
measured by unattended.

**16** Wannier functions, and nobody wrote 16 anywhere: `Si.upf` carries a 3S
and a 3P shell and `C.upf` a 2S and a 2P, so each atom contributes one s and
three p, and four atoms make sixteen. This is the case where an orbital count
taken per species rather than per structure has to be right — three silicons
and one carbon is not four of anything.

## What it gets right

|                       | eight occupied bands             |
| --------------------- | -------------------------------- |
| at the 64 mesh points | **0.109 meV** rms, 0.584 meV max |
| along a generic path  | **32 meV** rms, 153 meV max      |

The first row is the assertion that carries the chain. Evaluated back at a point
it was built from, the model is a Fourier transform of H(k) and must return the
DFT eigenvalues of the chosen subspace — with no symmetry, no degeneracy and no
tuned parameter anywhere in this structure to make that accidental. It does, to
a tenth of a meV.

The second row is far better than the perturbed-silicon example manages (147
meV), which is worth stating because it is the opposite of what one might guess:
a messier structure interpolating better. The 20-band window over 16 Wannier
functions leaves the disentangler less room to wander than silicon's 12 over 8,
a larger cell has a denser lattice-vector set for the same mesh, and this is the
one example whose gauge needed the guiding centres described below.

The density of states integrates to exactly 16, one per Wannier function.

## What it found: the branch of a logarithm, twice

This example was built to find out what breaks on a structure nobody tuned, and
it found two things the four tuned examples could not.

**First**, the spread minimisation stalled here — 23 iterations, the spread
moved by 0.6%, the gradient left at essentially its starting value. Measured on
this structure's own overlaps at the gauge the minimisation starts from, 8% of
the diagonal phases sit within 0.64 rad of ±π. `Im ln` is a principal value, so
Ω*D jumps by 2π when a diagonal overlap rotates across the negative real axis,
and no descent direction exists across a jump. The fix is guiding centres: take
each phase on the branch nearest the one a known centre implies. Where the
shift is applied matters — folding the \_residual* breaks the analytic gradient
(a central difference gave 246.6 where the gradient predicted 332.4); shifting
the _phase_ keeps the identity Σ_b w_b **b** q_n = 0 that the gradient relies on.

**Second**, and this is the one that needed an outside witness: the first
guiding centres were the projection gauge's _own_ Berry-phase centres, taken at
the start. On this structure those are already wrong — with the branches
inconsistent across k, the average that defines a centre lands 5 Å from any
atom — and guiding to a wrong centre keeps it wrong. The minimisation then
"converged" at Ω = 170.6 Å² with Ω_D near 149, and the number stood in this
file for a day labelled unexplained. Wannier90 4.0.1 on the very same exported
matrices, guiding by its projection centres, started at 36.7 Å² and finished at
25.9. The guides now are the **trial-orbital positions** the record carries
(format version 2), which is what Wannier90 does and is the one thing known
about a Wannier function before it has been minimised.

|              | first guides      | trial-orbital guides         | Wannier90 4.0.1    |
| ------------ | ----------------- | ---------------------------- | ------------------ |
| Ω            | 170.6 Å²          | **26.02 Å²**                 | 26.02 Å²           |
| Ω_I          | 21.24 Å²          | 21.24 Å²                     | 21.24 Å² (to 10⁻⁸) |
| centres      | 5 Å from any atom | within 0.08 Å of Wannier90's |                    |
| off-mesh rms | 32 meV            | 32 meV                       |                    |
| on-mesh rms  | 0.109 meV         | 0.109 meV                    |                    |

Two things worth reading off that table. Ω*I never moved: the disentangler was
right throughout, and Wannier90 agrees with its subspace to 10⁻⁸ Å² once both
codes freeze by projectability. And the off-mesh error did not move either — a
gauge that was not the minimum interpolated the occupied bands about as well
as the one that is, on this mesh. What changed is that the model now \_is* what
it says it is, and a driver reading its spread reads a real number.

## Files

| File                 | What it does                                                   |
| -------------------- | -------------------------------------------------------------- |
| `nc_scf.yml`         | SCF on a 4×4×4 grid, 20 bands, 100 Ry (carbon sets the cutoff) |
| `nc_nscf.yml`        | the full unreduced 4×4×4 grid; `method` and `grid` only        |
| `nc_interpolate.yml` | a generic path — the cell has no special points to name        |
