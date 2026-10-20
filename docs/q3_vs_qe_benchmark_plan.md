# Benchmarking Q3 against Quantum ESPRESSO on Perlmutter

A plan, not results. Nothing here has been run beyond the incidental timings
quoted as already measured, which came from validating the container images
rather than from a benchmark.

The comparison this describes is a **performance** one. Numerical agreement is
already covered: `tests/validation/` compares Q3 against the committed QE
reference energies, and the container images assert the silicon example's
converged total energy on every architecture they publish.

## What already exists, and what it is missing

[`scripts/validation/run_mpi_reproducibility_matrix.py`](../scripts/validation/run_mpi_reproducibility_matrix.py)
is most of the harness. It sweeps topologies (`mpi_2_ranks`,
`mpi_4_ranks_2_pools`, `mpi_4_ranks_4_pools`, …), repeats each three times so a
same-topology spread is distinguishable from a cross-topology difference,
records `wall_seconds` alongside every energy component, and captures the git
commit. Extending it beats writing a second harness.

Two things it does not do, both needed here:

-   **No QE leg.** It takes `--serial-binary`, `--mpi-binary` and
    `--gpu-binary`, all Q3. A comparison needs `pw.x` invoked on the paired
    input beside each deck, with its own topology flags (`-nk`, and optionally
    `-nb`/`-ndiag`).
-   **It launches natively.** `topology.command` is built as
    `("mpirun", "-np", N, binary, "--pools", P)`. Benchmarking the published
    images instead means a `srun … podman-hpc run --mpi …` prefix. Since the
    command is already a tuple carried on the topology, that is a launcher
    abstraction rather than a rewrite.

## The comparison is not symmetric, and the asymmetry is the finding

Two asymmetries dominate any number this study produces. Both belong next to
the results, not in a footnote.

**The library stacks differ.** The published images link the distribution's
reference BLAS and LAPACK (`libblas3`, `liblapack3`) and the container's FFTW.
The NERSC QE module links Cray LibSci and Cray FFTW, built for the machine.
Reference BLAS is commonly an order of magnitude or more slower than an
optimised one on large matrix work, so a raw head-to-head measures the library
stack at least as much as it measures either code. That is a legitimate result
— it is what somebody pulling the image today experiences — but it is not a
statement about Q3's algorithms, and the writeup should not let it be read as
one. Recording iterations-to-convergence and time-per-iteration keeps a partial
signal. An image built against OpenBLAS would separate the two properly and is
the obvious follow-up.

**The parallel axes differ.** Q3 distributes k-points, one rank per pool, and
refuses a rank count exceeding the irreducible k-point count. QE additionally
distributes plane waves and bands, and can put the subspace diagonalisation on
ScaLAPACK. So Q3's useful rank count has a ceiling set by the input, and QE's
does not; and `q3_gpu_cuda` is one process on one device, where QE's GPU build
runs a rank per GPU over GPU-aware MPI. The honest presentation is not a
speedup curve with Q3's line stopping early, but a statement of where each code
can go at all.

## System ladder

Silicon, norm-conserving, PBE, 48/192 Ry, `conv_thr = 1e-8`. Irreducible
k-point counts are what Q3 reports for its own decks.

| deck                | atoms | bands | k-grid        | irred. k-pts   | max PW/k-pt | Q3 rank ceiling |
| ------------------- | ----- | ----- | ------------- | -------------- | ----------- | --------------- |
| `nc.yml`            | 2     | 8     | 2×2×2 shifted | 2              | 1,557       | 2               |
| `nc_2x2x2.yml`      | 16    | 40    | 2×2×2 shifted | 2              | 12,410      | 2               |
| `nc_3x3x3.yml`      | 54    | 130   | Γ only        | 1              | —           | 1               |
| `nc_kpt_8x8x8.yml`  | 2     | 8     | 8×8×8 shifted | 60             | 1,564       | 60              |
| `nc_2x2x2_k444.yml` | 16    | 40    | 4×4×4 shifted | to be measured | 12,410      | = irred. k-pts  |

Those give a size axis at fixed k-point count (2 → 16 atoms, 1.5k → 12.4k plane
waves per k-point) and a k-point axis at fixed size (2 → 60). `nc_2x2x2_k444`
was added for this plan: every other large-cell deck samples one k-point, so
without it the regime where both codes can actually scale is unrepresented.

Adding copper (a metal, so smearing and slower convergence) and diamond (PAW
rather than norm-conserving) covers the other occupation and pseudopotential
paths. Compare Q3 against QE within each material; the difference between
materials mixes methods and is not a physics result.

## Phases

**Parity.** Confirm the NERSC module reproduces the committed reference
energies, that Q3 and QE agree per deck, and that iteration counts are close
enough that wall-clock is not silently comparing convergence paths. Produce the
missing irreducible k-point counts. Check that Q3's `broyden` and QE's
`mixing_mode = 'plain'` are the same scheme, which the QE documentation implies
and `beta`/`ndim` agreement supports; and that `nc_3x3x3`'s Kerker
preconditioning has a QE analogue in the committed input.

**Serial baseline.** One rank, one thread, both codes, every deck. The reference
point for every speedup, and the cleanest view of the library gap.

**CPU MPI scaling.** Rank sweeps on the k-point-rich decks, to each code's
ceiling. Then the cell Q3 cannot enter — `nc_3x3x3`, Γ-only, above one rank —
which QE covers by distributing plane waves. That contrast is the most
decision-relevant number here. Record QE both with `-nk` alone, the fair
analogue of Q3's single knob, and with one tuned configuration, the honest
answer to what somebody should actually run.

**GPU.** Q3 on one A100 against QE on one, two and four. Confirm rather than
sweep `--precision mixed`: it already showed no speedup and 7.2e-6 Ry of error
on the 54-atom cell.

**Throughput.** Four independent calculations on a four-GPU node, each code, for
the parameter-sweep case where latency does not matter.

## Metrics

Total energy and its deviation from the committed reference; iterations to
convergence; wall time; time per SCF iteration; node-seconds and GPU-seconds;
parallel efficiency against the same code's one-rank baseline.

Two measurement traps, both hit already. Each `podman-hpc run` costs about 1.3 s
of startup and the `srun` step around it roughly 2 s more, so any case finishing
in under a minute is measuring the launch, not the calculation — an earlier rank
sweep on a 2-atom input flattened at about five seconds for exactly that reason.
And the images set
`OMP_NUM_THREADS=1`, which GNU `nproc` honours, so thread counts have to be set
deliberately and read back rather than assumed.

## Already measured

Incidental, from container validation on Perlmutter, 2026-09-10, quoted here so
the plan starts from something:

| case                        | result                                               |
| --------------------------- | ---------------------------------------------------- |
| 2-atom Si, Q3 vs QE 7.5     | agree to 2.6e-7 Ry                                   |
| 54-atom Γ, serial CPU       | 706.8 s                                              |
| 54-atom Γ, one A100         | 132.5 s, 5.3×, energy identical to serial            |
| 2-atom 8×8×8, `q3_mpi` 1→16 | 11.8 s → 5.4 s, energy identical at every rank count |
| 4×54-atom on four A100s     | 160.6 s total, 3.3× throughput                       |
