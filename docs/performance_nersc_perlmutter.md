# Performance on Perlmutter

Measured numbers for the builds described in
[build_nersc_perlmutter.md](build_nersc_perlmutter.md), and what they imply about how to
run Q3 on this machine.

These are **site-specific**. [PERFORMANCE.md](../PERFORMANCE.md) holds the project's
portable benchmarks; this page holds Perlmutter's, because the numbers depend on the
node, the CPE version and the compiler, and because the GPU comparison here is against a
Perlmutter CPU node rather than against a CI runner.

Reproduce with:

```bash
salloc -N 1 -C cpu -q interactive -t 02:00:00 -A m2651 \
    scripts/hpc/nersc_perlmutter/benchmark.sh cpu
salloc -N 1 -C gpu -G 1 -q interactive -t 01:00:00 -A m2651_g \
    scripts/hpc/nersc_perlmutter/benchmark.sh gpu
```

## How to read these numbers

-   **Times are the solver's own reported total, not wall clock.** `srun` startup is
    2-9 s on this machine, which swamps runs of under a second: measured by wall clock
    the Si case appears to stop scaling at 8 ranks, and measured by the solver it scales
    to 32. Wall is in the tables too so the gap stays visible.
-   **Every run's total energy is checked**, because a speedup that changes the answer is
    not a speedup.
-   **The numbers in this document are single-shot**, from before the run-to-run spread
    was measured. That spread is now known, and it is large on short runs: a relative
    standard deviation of 1.5% for serial silicon but 10.7% for its four-rank topology,
    against 0.3-0.4% on the 39-second copper deck. A difference below roughly three
    standard deviations is not distinguishable from the machine.

    What that invalidates here: **the "24% slower" attributed to ScaLAPACK below is about
    two standard deviations and should be read as "no measurable difference"**, not as a
    cost. The larger findings clear the floor comfortably — pool scaling 28.9x, threading
    8 against 32 (46%), affinity (45%), GPU against 32 threads (37%), and the
    reciprocal-space split against serial on copper (42%).

    `scripts/hpc/nersc_perlmutter/compare_benchmark.py` derives a per-entry threshold from measured
    spread; `docs/benchmarks/` holds the protocol.

-   Measured 2026-09-10 on CPE 26.03, `PrgEnv-gnu` (GCC 14.3.0), `cray-fftw/3.3.10.11`,
    `cray-libsci/26.03.0`. CPU nodes are dual AMD EPYC 7763 (128 physical cores, 256
    hardware threads); the GPU node used an A100-SXM4-40GB.

## Two systems, because they measure different things

| System             | Input                                                   | Shape                                      | What it exercises                                               |
| ------------------ | ------------------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------- |
| Cu 3x3x3 supercell | `total_energy--metal--copper/nc_3x3x3_bench.yml`        | gamma-only, 280 bands, 384,455 G-vectors   | One k-point, so pools are unusable: threading, band groups, GPU |
| Si 8x8x8           | `total_energy--semiconductor--silicon/nc_kpt_8x8x8.yml` | 60 irreducible k-points, small per k-point | k-point pools                                                   |

The Cu input sets `max_iterations: 3` and deliberately does **not** converge -- it times
a fixed amount of work. Its energy is therefore its own 3-iteration value
(-10438.720620917 Ry), not the converged -10438.57534212 Ry in `reference/nc/`.

## Cu 3x3x3, gamma-only, 280 bands

### OpenMP threading, one rank

| Threads | Solver   | Speedup   | Energy           |
| ------- | -------- | --------- | ---------------- |
| 1       | 26913 ms | 1.00x     | -10438.720620917 |
| 8       | 6136 ms  | **4.39x** | -10438.720620917 |
| 32      | 8972 ms  | 3.00x     | -10438.720620917 |

**Threading peaks at 8 and regresses beyond it.** 32 threads are 42% slower than 8, and
64 threads are slower than 4. Q3's own per-stage breakdown says where it goes (all times
ms, one rank, unbound):

| Threads | TOTAL    | h_psi | fft      | nonlocal | allreduce | subspace | diagonalize |
| ------- | -------- | ----- | -------- | -------- | --------- | -------- | ----------- |
| 1       | 26833    | 5528  | 4588     | 5214     | 1780      | 2383     | 868         |
| 2       | 15066    | 3050  | 2884     | 2722     | 961       | 1285     | 813         |
| 4       | 8742     | 1850  | 1755     | 1467     | 580       | 739      | 777         |
| 8       | **6306** | 1365  | **1530** | 871      | 387       | 509      | 841         |
| 16      | 6923     | 1722  | 2316     | 702      | 323       | 580      | 1204        |
| 32      | 8925     | 2214  | 3163     | 701      | 278       | 469      | 2344        |
| 64      | 15075    | 3151  | 4049     | 1244     | 465       | 707      | **6262**    |

Two stages account for it, and they fail differently:

-   **`fft` bottoms out at 8 threads and then climbs**, from 1530 ms to 4049 ms at 64 --
    worse than the 4588 ms it took on one thread. `fft_grid_serial.hpp` passes
    `OMP_NUM_THREADS` straight to `fftw_plan_with_nthreads`, so threads go _inside_ each
    transform. The wave grid is 72^3 and there are 280 bands, so a Davidson iteration
    issues thousands of transforms and each one pays thread fork/join. Past 8 threads
    that overhead exceeds the work. The scalable axis for 280 bands would be threading
    _across_ bands with serial transforms, not within each transform.
-   **`diagonalize` degrades monotonically**, 868 ms on one thread to 6262 ms on 64 -- a
    7.2x slowdown from adding threads. This is the dense subspace eigensolve through
    libsci's threaded LAPACK, on a matrix that reaches about 1005x1005. It does not want
    64 threads.

Everything else scales sensibly: `nonlocal` 7.4x by 32, `allreduce` 6.4x, `subspace` 5.1x.
So this is not memory bandwidth saturating the node -- three stages keep scaling while
two reverse.

### Binding threads makes it worse, not better

The usual advice is to bind threads explicitly. Measured here, it costs 45%:

| Threads | unbound  | `OMP_PROC_BIND=spread` + `--cpu-bind=cores` | `OMP_PROC_BIND=close` |
| ------- | -------- | ------------------------------------------- | --------------------- |
| 8       | **6306** | 9124                                        | 9117                  |
| 16      | 6923     | 8896                                        | 8947                  |
| 32      | 8925     | 9928                                        | 9853                  |

The damage is concentrated in `fft`: at 8 threads it goes from 1530 ms unbound to 4328 ms
bound, 2.8x worse. `spread` and `close` come out the same, so it is not the placement
policy -- it is confining the process to `-c` CPUs at all, which the threaded FFTs
evidently do worse under than they do when the scheduler is left alone.

This is a measurement on one workload on one node type, not a general claim about
affinity. But it does mean the default -- no binding, `OMP_NUM_THREADS=8` -- is the
fastest configuration measured here, and adding the affinity flags that usually help will
make Q3 slower.

### MPI band groups (`--pools 1`, one thread per rank)

| Ranks | ScaLAPACK | Solver   | vs 1 thread | Note                    |
| ----- | --------- | -------- | ----------- | ----------------------- |
| 2     | on        | 38151 ms | 0.71x       |                         |
| 4     | on        | 34509 ms | 0.78x       |                         |
| 8     | on        | --       | --          | **crash**, replicated-G |
| 16    | on        | --       | --          | **crash**, replicated-G |
| 4     | off       | 35777 ms | 0.75x       |                         |
| 16    | off       | 69126 ms | 0.39x       | replicated-G            |

**The reciprocal-space split is a net loss on this system at every width that runs.**
This section measures the default split only. `--distribute-bands` is a different mode and
a later measurement found it **2.3x faster than this** and 2.5x faster than serial on the
same deck -- see "Two splits, not one" below. Two ranks are
40% slower than one thread; sixteen are 2.5x slower. A gamma-only calculation has one
k-point, so there is no pool parallelism to fall back on -- which means **MPI does not
help this system at all**, and threading is the only useful CPU parallelism.

Above 4 ranks per pool the band-group slab distribution leaves some ranks with no plane
waves and the code falls back to replicated G-vectors, so every rank repeats the whole
solve. That is what the 16-rank ScaLAPACK-off row is measuring: 16 ranks each doing all
the work.

### The 8-and-above crash (fixed 2026-09-10)

With ScaLAPACK on, 8 and 16 ranks used to abort:

```
MPI_Allreduce(sendbuf=..., recvbuf=..., count=1010025, MPI_DOUBLE_COMPLEX, MPI_SUM,
              comm=0x84000003) failed
srun: error: tasks 0-7: Segmentation fault
```

Neither the size nor the datatype was at fault -- that exact reduction, 1005^2 complex
doubles over 8 ranks, completes in 10 ms in isolation with `MPI_DOUBLE_COMPLEX`,
`MPI_C_DOUBLE_COMPLEX` and `MPI_CXX_DOUBLE_COMPLEX` alike. The trigger was the
**replicated-G fallback meeting the collective ScaLAPACK solver**, and it reproduced at
any problem size: on Si 8x8x8, 8 ranks in one pool (which falls back) crashed with
ScaLAPACK and completed without it, while 4 ranks per pool (which does not fall back)
was fine either way.

`disable_group_distribution()` sets `group_size()` to 1 but deliberately leaves the group
communicator with all its ranks. In that state every rank holds the whole problem and
walks Davidson on its own, so a collective solve had each rank entering group-wide BLACS
and `MPI_Allreduce` calls independently at whatever basis size it had reached. The
subspace solve now branches on `group_size() <= 1` and solves locally, which is what
`MPIExecutionPolicy` documents that callers must do.

Both cases now complete and agree with the no-ScaLAPACK path exactly
(-10438.724181819 Ry at 8 and at 16 ranks). They are not faster -- 69.2 s and 70.4 s
against 69.1 s without ScaLAPACK -- because in the fallback every rank redundantly
repeats the whole solve, so extra ranks buy nothing. The band-group ceiling of 4 ranks
per pool for this system still stands; what changed is that exceeding it degrades instead
of crashing.

## Si 8x8x8, 60 irreducible k-points

| Ranks | Pools | Ranks/pool | ScaLAPACK | Solver  | Speedup   | Energy        |
| ----- | ----- | ---------- | --------- | ------- | --------- | ------------- |
| 1     | 1     | 1          | --        | 2293 ms | 1.00x     | -16.925260460 |
| 8     | 8     | 1          | on        | 304 ms  | 7.55x     | -16.925260460 |
| 32    | 32    | 1          | on        | 79 ms   | **28.9x** | -16.925260460 |
| 8     | 2     | 4          | on        | 1187 ms | 1.93x     | -16.925260460 |
| 8     | 2     | 4          | off       | 957 ms  | 2.39x     | -16.925260460 |

**k-point pools scale nearly linearly**: 28.9x on 32 ranks, 90% efficiency. This is the
parallelism that works.

**Band groups cost, even here.** At a fixed 8 ranks, 8 pools (304 ms) beat 2 pools with
4 ranks each (1187 ms) by 3.9x. Spend ranks on pools first, and only use band groups
when the irreducible k-point count has run out.

**ScaLAPACK does not pay at this size**: 1187 ms with it against 957 ms without, 24%
slower. The subspace is small enough that distributing it costs more than it saves. It
is correctness-equivalent -- the energies match to every digit -- so the reason to enable
it is a large subspace, not a small one.

A wider sweep on the same system, run separately, confirms where pools stop helping:
1 rank 2288 ms, 4 ranks 571 ms, 8 ranks 300 ms, 16 ranks 165 ms, 32 ranks 82 ms, then
64 ranks 93 ms and 128 ranks 111 ms. Past 32 ranks the 60 k-points cap the pool count,
the extra ranks can only go into band groups, and throughput falls. **32 ranks is the
optimum for this system, and 128 ranks is 35% slower than 32.**

## Multi-node: the node boundary costs nothing

Graphite `nc_d3`, a 12x12x6 grid reducing to **76 irreducible k-points** -- the only deck
here with enough of them for pools alone to cross a node boundary. Four repetitions, first
discarded, capped at three SCF iterations:

| Topology                | Nodes | Ranks | Solver   | vs serial | relative sd |
| ----------------------- | ----- | ----- | -------- | --------- | ----------- |
| serial                  | 1     | 1     | 9404 ms  | --        | 14.3%       |
| `mpi_8_ranks_8_pools`   | 1     | 8     | 840.9 ms | 11.2x     | 0.6%        |
| `mpi_2_nodes_8_ranks`   | **2** | 8     | 840.5 ms | 11.2x     | 1.2%        |
| `mpi_16_ranks_16_pools` | 1     | 16    | 459.9 ms | 20.4x     | 4.1%        |
| `mpi_2_nodes_16_ranks`  | **2** | 16    | 428.9 ms | 21.9x     | 0.6%        |

At matched rank counts the two-node runs are **not slower**: 840.5 ms against 840.9 ms at
eight ranks, and 428.9 against 459.9 at sixteen, where the two-node version is if anything
7% faster -- consistent with eight ranks per node having more memory bandwidth each than
sixteen. So k-point pool parallelism crosses the interconnect for free on this deck.

Efficiency above 100% at sixteen ranks (20.4x) is the usual cache effect, helped by the
serial baseline being the noisiest row in the table.

Energies are invariant to all of it. All twenty runs agree to **1.4e-14 Ry**, about three
units in the last place of a double, and each topology is bit-identical across its own
repetitions.

## Multi-GPU: one calculation across many devices

`q3_gpu_cuda` is one process on one device -- `src/main_gpu_cuda.cpp` has no MPI call at
all, and `run_gpu_entry_point()` builds a serial execution policy -- so `srun -n 8
q3_gpu_cuda` runs eight independent duplicate calculations rather than one wider one.

`q3_mpi_gpu_cuda` is the composition that was missing: the MPI setup from `q3_mpi` with
the CUDA backends and the GPU Davidson injected, one device per rank, each rank owning a
k-point pool. Silicon 8x8x8, 60 irreducible k-points:

| Devices | Nodes | Solver   | Speedup   | Energy        |
| ------- | ----- | -------- | --------- | ------------- |
| 1       | 1     | 15580 ms | 1.00x     | -16.925260460 |
| 2       | 1     | 8041 ms  | 1.94x     | -16.925260460 |
| 4       | 1     | 3716 ms  | 4.19x     | -16.925260460 |
| 8       | **2** | 1978 ms  | **7.88x** | -16.925260460 |

Near-linear, 98% efficiency at eight devices across two nodes, and every energy identical
to what the single-device `q3_gpu_cuda` reports for the same deck.

Ranks sharing a node must be given different devices from outside, because
`device/cuda.cpp` calls `cudaSetDevice(0)`:

```bash
srun -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 \
    bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; q3_mpi_gpu_cuda --pools 8 deck.yml'
```

Without it every rank on a node opens GPU 0, and what looks like four devices is four
processes contending for one.

The ceiling is the same as on CPU: pools cannot exceed the irreducible k-point count, so
a gamma-only deck still cannot use a second device this way.

## GPU throughput: independent calculations

Separately from the above, for the parameter-sweep case where latency does not matter.

What can be measured is throughput, which is the parameter-sweep case. Copper 3x3x3, one
calculation per device, `CUDA_VISIBLE_DEVICES` pinned per local rank:

| Devices | Nodes | Wall per calculation | Calculations/hour | Energies            |
| ------- | ----- | -------------------- | ----------------- | ------------------- |
| 1       | 1     | 20 s                 | 180               | --                  |
| 2       | 1     | 20 s                 | 360               | agree               |
| 4       | 1     | 23 s                 | 626               | agree               |
| 8       | 2     | 23 s                 | 1252              | agree to 1e-13 rel. |
| 16      | 4     | 20 s                 | 2880              | agree to 1e-13 rel. |

Throughput is linear in device count -- 16x on sixteen devices -- and a calculation takes
about as long with four sharing a node as it does alone (20 s against 23 s). So the four
A100s in a node are genuinely independent for work of this size, and a sweep of
independent calculations is the thing this build is good at.

Above eight devices two distinct energies appear, differing by one unit in the last
printed digit: 1e-9 Ry absolute, 1e-13 relative, which is the log's printing precision, so
the underlying difference may be smaller. Floating-point accumulation across devices, not
a correctness question.

## The large deck is memory-bound, not core-bound

`nc_6x6x6` -- 432 atoms, 1040 bands, **1,338,865 G-vectors**, 167,155 plane waves per
k-point. Measured on four ranks with `--distribute-bands`:

    resident set per rank    38.9 GB
    four ranks               156 GB of a 503 GB node

Two consequences the deck's own "~45 GB per rank" note implies without spelling out:

-   **Ranks per node are limited by memory to about twelve, not by cores to 128.** A
    Perlmutter CPU node cannot be filled with MPI ranks on this deck; the remaining cores
    have to be reached with threads or with more nodes.
-   **One rank does not comfortably fit a 40 GB A100.** Since `q3_gpu_cuda` is a single
    process on a single device, this deck is effectively out of reach of the GPU build as
    it stands.

`--distribute-bands` did not materially reduce the per-rank footprint here, which is worth
knowing before planning around it. Whether that is expected -- bands distributed while the
basis stays replicated -- or a missed opportunity has not been established.

A cost note, now measured rather than estimated: **four band-distributed ranks did not
finish three SCF iterations in 83 minutes** (the run hit a 5000-second timeout, still
short of completion). So a single data point at that width costs more than 5.5 node-hours
and its true cost is unknown, because it never finished. An earlier estimate of "about a
node-hour" in this file was wrong and is corrected here.

Which means the deck is not benchmarkable at small rank counts at all. Making it
measurable needs one of:

-   Many more ranks -- but memory caps a node at about twelve, so that means several
    nodes.
-   A cheaper variant of the deck: one SCF iteration instead of three, or a smaller cell,
    accepting that it then measures something other than what the 432-atom case does.

Both are decisions about how much machine time a single number is worth, not things to
settle by running it again and hoping.

## GPU: A100 vs a Perlmutter CPU node

Cu 3x3x3, same input. The CPU rows here were measured on the GPU node itself, so the
comparison is within one node.

| Configuration          | Solver   | vs 1 CPU thread | Energy           |
| ---------------------- | -------- | --------------- | ---------------- |
| CPU, 1 thread          | 27110 ms | 1.00x           | -10438.720620917 |
| CPU, 32 threads        | 8973 ms  | 3.02x           | -10438.720620917 |
| A100, double           | 6551 ms  | **4.14x**       | -10438.720620917 |
| A100, mixed (FP32 FFT) | 6017 ms  | 4.51x           | -10438.718710103 |

Double precision on the GPU reproduces the CPU energy exactly. Mixed precision differs
in the fifth decimal, which is expected for FP32 FFTs via cuFFT rather than a fault.

**What the GPU is actually worth here.** Against one CPU core, 4.1x. Against 32 threads
on the same node, 1.37x. Against the best CPU configuration measured -- 8 threads, 6136
ms on a CPU node -- it is **roughly parity**, and mixed precision is only about 2%
ahead.

`PERFORMANCE.md` reports 6.6x for this system, measured on a GitHub Actions runner with
a Tesla T4 against **CPU serial**. That is a different comparison on different hardware,
and it should not be read as what an A100 buys on Perlmutter. Both numbers are correct
about what they measured; neither is a general "GPU speedup".

The honest summary is that 280 bands and 384,455 G-vectors is not enough work to occupy
an A100. A larger cell, more bands, or a k-point set the GPU path could stride over would
change the picture -- and none of that has been measured.

## What to actually run

| If your system...                        | use                                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| has many irreducible k-points            | `q3_mpi` with one rank per pool, up to the IBZ count (5.75x on four ranks). This is the parallelism that scales.                                                   |
| is gamma-only or has few k-points        | `q3_mpi --distribute-bands` (2.59x on four ranks), or `q3` with `OMP_NUM_THREADS=8` and no affinity flags. The default reciprocal-space split is worth only 1.13x. |
| is gamma-only and large                  | `q3_gpu_cuda`, but expect parity with 8 CPU threads on a system this size rather than a large speedup.                                                             |
| needs more ranks than there are k-points | Reconsider. Past the IBZ count the extra ranks go into band groups and throughput falls.                                                                           |

`OMP_NUM_THREADS=8` is the measured optimum, not a round number: 32 threads are 42%
slower. Do not add `--cpu-bind=cores` or `OMP_PROC_BIND` -- both cost about 45% here.

Do not enable ScaLAPACK expecting speed either: it was correctness-equivalent, and the
24% it appeared to cost is about two standard deviations of the run-to-run spread on that
deck, so the honest reading is that no difference was measured. Above four ranks per pool it now
degrades to a redundant per-rank solve rather than crashing, so it is safe, just not
useful there.

## Two splits, not one

Everything above measures Q3's **default** intra-pool split, which distributes reciprocal
space in G-vector slabs. `--distribute-bands` distributes bands instead, and it is a
different mode rather than a variation: `enable_band_distribution()` sets `group_size()`
to 1, each rank then holding whole coefficient vectors for a subset of bands.

Measured 2026-09-10 with four repetitions, first discarded, on four ranks:

| Deck                                | serial   | reciprocal space | `--distribute-bands` | pools              |
| ----------------------------------- | -------- | ---------------- | -------------------- | ------------------ |
| Copper 3x3x3, gamma-only            | 39923 ms | 35342 ms (1.13x) | **15386 ms (2.59x)** | n/a                |
| Silicon 8x8x8, 60 irreducible k-pts | 3771 ms  | 3214 ms (1.17x)  | 1889 ms (2.00x)      | **656 ms (5.75x)** |

Run-to-run spread on those: 0.2-0.5% for the band split, 0.3-0.9% for reciprocal space,
3-4% for serial. The differences are far outside it.

So the guidance is by deck shape, and it is not the same answer in both rows:

-   **Many irreducible k-points**: pools win, and by a wide margin (5.75x on four ranks).
-   **Gamma-only or few k-points**: pools are unavailable and `--distribute-bands` is worth
    2.59x, where the default split is worth 1.13x.

The earlier conclusion that band groups cost more than they return was a statement about
the reciprocal-space split. It does not carry over to the mode Q3 provides for exactly this
case, which was never measured until now.

## Does distributing G-vectors change the answer?

No, on a converged calculation. Si 8x8x8 converges to **-16.925260460 Ry** at 1, 2 and 4
ranks per pool alike, identical to serial and to every pool count tried.

The Cu rows above do differ across widths -- -10438.720620917 serial against
-10438.723744294 at 2 ranks and -10438.724083402 at 4, about 3e-3 Ry. That input stops
after 3 iterations without converging, so the runs are being compared at a point the SCF
has not settled on, and a different summation order over a distributed G-vector set
leaves them at slightly different points. The differences are deterministic (repeat runs
agree to 8 digits), and they do not appear on the converged system. Treat the Cu energies
as timing bookkeeping, not as a correctness comparison.
