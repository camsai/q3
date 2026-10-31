# 06 — Bringing existing inputs

**Goal:** run a Quantum ESPRESSO namelist through Q3 without converting it to a
Q3 manifest first.

This lesson stands alone. If you arrived with an input file and want to see
whether Q3 will run it, start here and go back to
[01](01-first-calculation.md) afterwards.

Q3 picks its parser from the file it is handed — a `.in` namelist and a YAML
manifest go to two different front ends that produce the same internal
`InputData`. Nothing downstream knows which one you used. That is the point: a
compatible engine has to answer to the inputs people already have, not ask them
to migrate first.

## Running a namelist

The QE reference inputs used to validate Q3 are committed, so there is a real
one to try:

```bash
cd examples/total_energy--semiconductor--silicon/reference/nc
```

```bash
../../../../build/q3 scf_qe.in
```

```
Calculation type: self_consistent_field
     Reading PP: ../../../data/pseudopotentials/nc/Si.upf
     UPFv2: Si (NC) Z=4 mesh=1510 num_beta=6 max_l=2 has_nlcc=true
     Spacegroup: Fd-3m (#227, 48 symmetry ops)
     IBZ reduction (internal): 8 -> 2 k-points (48 symmetry ops + time reversal)
     convergence achieved after 7 iterations
    total energy         =  -16.913417932 Ry
```

That is the same log, and the same **−16.913417932 Ry**, that
[tutorial 01](01-first-calculation.md) got from `nc.yml`. The two files
describe the same calculation — `ibrav = 2` with `celldm(1)` instead of an
explicit lattice, `K_POINTS automatic` instead of a `kgrid` block — and Q3
resolves both to the same thing.

> **`pseudo_dir` resolves against your current directory, not the input
> file's.** This is the same trap `outdir` sets in YAML manifests. The `cd`
> above is load-bearing: run `build/q3 examples/.../scf_qe.in` from the
> repository root and it fails with `Cannot open PP file`. Run QE inputs from
> the directory they live in.

Metals work the same way — `occupations = 'smearing'`, `smearing = 'gaussian'`
and `degauss` map onto the smearing block from [tutorial 03](03-convergence.md):

```bash
cd examples/total_energy--metal--copper/reference/nc && ../../../../build/q3 scf_qe.in
```

```
    total energy         = -386.583379230 Ry
```

Against `copper/nc.yml`'s −386.583379146 Ry, that is a difference of
8×10⁻⁸ Ry. It is not a format artifact: the two files ask for _different
mixing algorithms_ — `mixing_mode = 'plain'` in the namelist, `mode: broyden`
in the manifest — and two mixers converging to the same fixed point under a
1×10⁻⁸ density threshold land a hair apart in energy. Silicon happens to agree
to all nine digits because it is an easy SCF; copper does not, and the residual
is the size of the convergence threshold, not of the input format.

## Relaxing a namelist

A `relax` or `vc-relax` deck needs something to move the atoms between SCFs.
In Q3 that is Python — ASE's BFGS, through the same
[relaxation workflows](05-python-and-ase.md#3-relaxation) a YAML manifest uses
— and the deck itself is what each step runs. The committed silicon reference
has one atom pushed off its site:

```python
from q3.workflows import run_relax_workflow

result = run_relax_workflow(
    "examples/relaxation--semiconductor--silicon/reference/nc/relax.in",
    q3_binary="build/q3",
)
print(result["converged"], result["steps"], result["final_energy_ry"])
```

```
      Step     Time          Energy          fmax
BFGS:    0 01:04:54     -227.753496        0.527868
BFGS:    1 01:04:55     -227.761497        0.071103
True 1 -16.7401613
```

The deck is `ibrav = 2`, and Python never parses it. The first step runs the
deck as written; the engine writes the structure it built into the results'
`material`, with any `if_pos` flags as constraints, and the optimizer continues
from there, handing each new geometry back with `--material`. Each step runs in
the deck's directory and starts from the density the previous one saved in its
`outdir`. The convergence threshold is the deck's own `forc_conv_thr`
(5e-3 Ry/bohr): pw.x stops one BFGS step later at −16.7401734 Ry, and with
`forc_conv_thr = 1e-4` Q3 lands there too.

The same call runs on a cluster by naming the binary and how to launch it —
`q3_binary="q3_mpi"` with
`execution_policy=ExecutionPolicy(number_of_processes=64, number_of_kpoint_pools=64, launcher="srun -n {number_of_processes}")`
from `q3.binary`, or a GPU binary. A `vc-relax` deck
goes through `run_vc_relax_workflow`; give `press` as
`target_pressure_gpa=press / 10` and take it out of the deck.

Run the deck directly, `build/q3 relax.in`, and you get one SCF at the starting
geometry, with a log line saying that is what happened:

```
     Calculation 'relaxation' runs as one SCF at this geometry, starting from a saved density when one fits: a step of the relaxation q3.workflows.run_relax_workflow / run_vc_relax_workflow drives
```

## What this does and does not buy you

Reading your input is not the same as reproducing your result. The parser
resolves your namelist into Q3's own model, and from there Q3 does what Q3 does
— its own pseudopotential handling, its own mixing, its own XC evaluation
through LibXC. Where those agree with the reference implementation, they agree
because they were validated to; see
[FUNCTIONALITY.md](../../FUNCTIONALITY.md) for what has been checked against
what, and [KNOWN-ISSUES.md](../../KNOWN-ISSUES.md) before trusting a regime
nobody has tested.

The honest summary: the input format costs you nothing, and the physics still
has to be checked.

Every namelist field Q3 reads — and, in its strictness policy, what it
refuses rather than approximating — is documented in
[docs/io_reference.md](../io_reference.md) §5.

## Next

[07 — Contributing a change](07-contributing.md): the layer map, where a change
goes, and how to get it merged.
