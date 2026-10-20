# Known Issues

## MPI parallelization limitations

Band parallelism (multiple ranks per k-point pool, sharing each diagonalization
via G-vector stick distribution) is implemented. The remaining constraints are:

### Sharing a pool is opt-in, and has no measured speedup

**Status:** by design — the run stops rather than choosing this layout for you.

Ranks that share a k-point pool distribute reciprocal space by FFT stick rather
than taking separate k-points. Measured on the silicon 3x3x3 supercell with the
slab layout this replaced, two ranks per pool were slower than one and four were
slower still; with the stick layout the 54-atom Gamma-only cell takes 48, 33, 33
and 25 s at one, two, four and eight ranks against Quantum ESPRESSO's 74, 44, 23
and 15 s on the same machine (PERFORMANCE.md), so the layout now pays from two
ranks and that code pulls ahead from four; the arrangement also disables stress,
checkpointing and mean-field export. Q3
therefore spreads ranks over k-points by default and refuses to leave them
sharing a pool unless `--pools` (or `control.number_of_kpoint_pools`) asks for
it.

The consequence worth planning around: a gamma-only supercell has exactly one
k-point, so sharing the pool is its only multi-rank arrangement. It keeps its
half sphere there -- the stick layout holds every G-vector with its negative,
and the one rank whose plane-wave list begins with G = 0 applies the origin
correction -- and its per-rank memory falls with the rank count. Distributing
bands themselves -- as opposed to reciprocal space -- is also implemented and
opt-in via `--distribute-bands`; it divides the work but not the storage, which
is the subject of the next entry.

### Per-rank memory does not fall as ranks are added

**Status:** open -- the footprint is reported, not yet reduced.
**Affects:** every multi-rank run, and `--distribute-bands` in particular.

Band distribution splits the loops over the Davidson basis and allgathers the
results; every rank still holds the whole basis, the whole G-sphere and the
whole density grid. Adding ranks therefore buys arithmetic and costs aggregate
memory. Measured on the silicon 3x3x3 supercell, peak resident memory per rank
was 3.12 GiB at one rank and 3.12 GiB at two, for identical total energies.

The per-rank cost is `6 x nbnd x npw` complex numbers for the solver and
wavefunctions at the default basis expansion factor of 2 (`10 x nbnd x npw` at
4), plus `nkb x npw` for the cached nonlocal projectors. On the 432-atom silicon
example that is about 41 GB, or 53 GB at
`method.scf.davidson.basis_expansion_factor: 4`. It was 108 GB before the
reductions described below. A non-self-consistent run uses 4 unless the deck
sets the factor; see "A non-self-consistent run could miss eigenvalues at the
default Davidson basis" under Resolved.

Quantum ESPRESSO holds `5 x nbnd x npw` plus the same projector term, so per
rank at the same rank count Q3 is now within about 12% of it. The rest of the
gap was structural -- that code divides its plane-wave terms by the ranks in a pool
because it distributes G-vectors -- and ranks that share a pool without
`--distribute-bands` now do the same, by FFT stick; see the entry above.

Measured on a 128-atom silicon cell, peak resident memory fell from 12.1 GB to
8.3 GB for an unchanged total energy. Every run prints an estimate of the named
arrays before the SCF loop and its measured peak at the end, so the cost is
visible before a job is submitted rather than after it is killed.

### Stress tensor is skipped under band parallelism

**Status:** open — omitted from results (a `log::warn` is emitted).
**Affects:** MPI runs with `band_size > 1` (fewer pools than ranks, including
the default single-pool topology on multi-rank runs).

Stress computation is not yet G-vector-distribution-aware; running it on
distributed wavefunctions/projectors would read out of bounds. The SCF flavor
skips it with a logged warning (`skip_stress` in
`src/layer_4_computation/flavors/self_consistent_field.hpp`) and the results
JSON contains no stress. Use one rank per pool
(`number_of_kpoint_pools` = number of ranks) when stress is required.

### Rank count must be divisible by the pool count

`PlanewaveMPIExecutionPolicy::configure_topology` throws when the number of
MPI ranks is not divisible by `number_of_kpoint_pools` — every pool must have
the same number of ranks. Pool counts exceeding the number of IBZ k-points are
clamped (with a warning), matching QE behaviour.

### Subspace diagonalization without ScaLAPACK is redundant

When Q3 is built with MPI but ScaLAPACK is not found, the subspace eigenproblem
is solved redundantly on the band-group root and broadcast — correct but not
scalable to large band groups.

### Run-to-run energy variation for soft-moment systems under MPI

**Status:** expected behavior, documented — not a correctness bug.
**Affects:** repeated identical MPI runs of metallic LSDA systems with hot
smearing (measured: Ni FCC, Fermi-Dirac at degauss 0.02 Ry, ~1e-4 Ry spread
across identical 4-rank runs; gaussian / Methfessel-Paxton / Marzari-Vanderbilt
on the same system are bit-deterministic).

FFTW plans are timing-selected per process (`FFTW_MEASURE`), so repeated runs
take bit-different SCF trajectories. For stiff systems these converge to the
same answer; for a nearly-degenerate moment valley (hot Fermi-Dirac smearing)
`dr2 < threshold` is satisfied anywhere along the valley, so runs legitimately
stop at slightly different moments/energies.

Set `control.use_deterministic_fft_planning: true` for bit-reproducible runs
(FFTW_ESTIMATE — a fixed heuristic plan, no runtime timing — at some
performance cost); default is `false` (FFTW_MEASURE, the faster choice). This
does not narrow the run-to-run variation reported above when left at its
default; opting into it eliminates the variation entirely (verified: 5
repeated runs of the exact Fermi-Dirac case above give bit-identical energy,
iteration count, and converged moment with the flag set). See
[TODOs.md](TODOs.md) §2.3.

Serial and GPU runs vary the same way, on a smaller scale. On the Metal build,
three serial runs of the chromium antiferromagnet example spread over 5.0e-10
Ry and three GPU runs over 8.4e-10 Ry (2026-10-09). That was enough to fail the
1e-10 Ry CPU↔GPU parity test in 7 of 9 pairings, so
`tests/validation/test_gpu_energy_parity.py` now plans every run
deterministically.

### The MPI failure path races itself, and its test is intermittent

**Status:** open. `tests/validation/test_mpi_topology_invariance.py::test_sharing_a_pool_must_be_asked_for`
fails on roughly one CI leg per run, alternating between the two ScaLAPACK
matrix legs, and passes locally every time.

The test asserts that a refused topology tells the user about `--pools`. The
refusal itself is reliable; delivering its message is not.

`create_kohn_sham_iterator` raises for a rank-uniform reason -- more ranks than
irreducible k-points -- so **every rank throws**, and each reaches
`main_mpi.cpp`'s handler and calls `MPI_Abort`. The first one there kills the
job, rank 0 included, and rank 0 was the only rank whose console is live. When
it had not finished writing the line, the line was gone.

Two things followed. Both were addressed on 2026-09-06, and what remains is
narrower than this entry first recorded:

1.  **The abort was also illegal.** `MPI_Finalize` runs from
    `PlanewaveMPIExecutionPolicy`'s destructor, and unwinding destroys the
    policy before the handler runs, so `MPI_Abort` was called after
    `MPI_Finalize`. The handler now checks `MPI_Finalized()` and returns 1
    rather than aborting into a finalized MPI. An earlier attempt at that guard
    was reverted -- ranks exited on their own, the launcher tore down on the
    first to do so, and the message was lost that way instead -- but the
    message is now written and flushed to stderr _before_ the abort-or-return
    decision, so that teardown can no longer outrun it. The deeper repair named
    here -- that MPI should not be finalized during unwinding -- was made on
    2026-09-10: the destructor now runs no teardown at all while an exception is
    in flight, every step of it being collective and no peer being obliged to
    arrive. That closed a worse failure than the one this entry is about. A
    _rank-local_ error -- an unreadable file, a failed allocation, anything one
    rank sees and the others do not -- used to block that rank in the
    destructor's `MPI_Barrier` while its peers sat in a physics collective, so
    the job hung instead of failing; and because the barrier sits ahead of the
    handler, it hung without printing why. Measured before the change at over
    75 seconds with an empty stderr, and after it at exit 1 in about a second
    with the reason.
2.  **Only the reporting rank should abort.** Still true, and still not a
    patch: it needs a decision about what the other ranks do meanwhile. The
    race it describes is now much less damaging, because a rank losing the
    race has already said its piece. It does, however, apply more widely than
    when it was written: now that the destructor no longer finalizes during
    unwinding, a rank-uniform refusal finalizes on no rank, so every rank takes
    the abort branch rather than the return-1 one. The race is therefore run on
    every such failure rather than none of them. It was measured at that
    setting and stays benign -- 40 consecutive four-rank refusals under a load
    average of 35, all four ranks reporting on stderr every time, exit code 1
    every time -- because each rank writes and flushes before it aborts.

Reporting no longer depends on rank 0 surviving. `init_mpi()` silences every
non-root rank via `spdlog::set_level(off)`, so `log::error()` from a failing
non-root rank emitted nothing at all and no amount of flushing could recover
it. `log::fatal()` bypasses spdlog, writes to stderr tagged with the rank, and
flushes before returning. `log::flush()` also exists for the non-MPI exits that
skip destructors.

Found 2026-09-05 while investigating the same test failing on one pull request
and passing on its child with identical code. Whether the intermittency is
fully gone has not been confirmed over enough CI runs to say.

## PAW, DFT+U and derivatives

### Ortho-atomic DFT+U metric and derivatives (resolved 2026-08-11)

Ortho-atomic DFT+U now uses and differentiates the same joint, system-wide
Löwdin transform in energy, forces, and stress for norm-conserving, ultrasoft,
and PAW datasets. Stress is no longer marked incomplete and is available through
the ASE calculator for every supported dataset family.

The ortho-atomic projector is a joint, system-wide Löwdin transform of every
atomic orbital in the cell, so `dP/dε` carries a `d(O^-1/2)/dε` term over
orbitals belonging to no Hubbard manifold — exactly the structure the
ortho-atomic _force_ needed `lowdin_inverse_square_root_derivative` for. The
stress path reuses that matrix-function derivative, builds `dO/dε` from every
orbital in the cell, and only then extracts Hubbard-manifold rows. It preserves
the full `3x3` tensor until the final symmetry average.

The off-symmetry SiC gate checks all six independent components against Q3's
own central energy derivatives at `h = 0.002` and `0.001`, with
`convergence_threshold: 1e-12`; the largest errors are `0.045` and `0.049 kbar`.
A four-step study stayed below `0.076 kbar`. Removing only `d(O^-1/2)/dε`
changes the Hubbard tensor by `36.25 kbar` and makes the committed gate fail by
`36.2 kbar`. Gamma-only overlaps in the SCF metric, force, and stress now
reconstruct the full sphere: half/full differences are `3.8e-11 Ry` in energy,
`1.5e-7 Ry/bohr` in force, `1.7e-4 kbar` in total stress, and `1.4e-6 kbar` in
Hubbard stress.

For ultrasoft and PAW, the metric is `⟨φ|S|φ⟩`, assembled through the production
overlap operator before the Löwdin diagonalization. Its force and stress
derivatives include both the orbital and `S` derivatives. Ultrasoft silicon
matches LibXC QE within `5.8e-7 Ry` in total energy, `1.1e-8 Ry` in Hubbard
energy, and `5e-5` in occupation traces. On PAW SiC, force differs from Q3's
energy gradient by `2.4e-6 Ry/bohr`; the six stress components remain within
`0.05 kbar` over the committed step-size study. A Gamma-only audit also found
and fixed a pre-existing factor-of-two error in PAW augmentation stress.

### Low-symmetry and Gamma-only stress were wrong (resolved 2026-08-11)

Two independent defects survived because earlier stress gates were cubic or
reported only pressure. First, the reciprocal Ewald shear was assembled in the
lower triangle while the real-space Ewald term was accumulated in the upper
triangle. The final copy from upper to lower therefore discarded every
reciprocal off-diagonal. On the inversion-only triclinic discriminator, Q3's
diagonal stress matched QE while the three shears differed by exactly the
missing QE Ewald shears: `-3.56`, `-2.38`, and `-60.50 kbar`. Both Ewald pieces
now accumulate in the upper triangle before symmetrization.

Second, Gamma-only storage keeps one member of each `+/-G` pair. Kinetic,
Hartree, Ewald, local, and NLCC stress sums did not reconstruct the omitted
member, and the nonlocal calculator contracted complex half-sphere projector
overlaps instead of their real full-sphere values. Each nonzero reciprocal
term now carries multiplicity two; `G=0` remains single-counted. Nonlocal base
and strain-derivative overlaps are completed as `2 Re[half]` with the `G=0`
product subtracted exactly where it exists.

Before the fix, the Gamma half/full kinetic terms were `1442/2885 kbar`, Ewald
`-1560/-2964 kbar`, and the total pressures `929/395 kbar`, despite energies
agreeing. They now agree in every contribution. The triclinic reduced and full
zones are invariant, all seven nonzero contribution tensors match QE to the
reference's printed precision, and the complete analytic tensor matches Q3's
own six-component central energy derivative to `0.006 kbar`. Regression
coverage is `tests/validation/test_stress_baseline.py`.

### The occupation matrix was never averaged over time reversal (resolved 2026-08-09)

Symptom: on zincblende SiC, PAW, U = 3 eV on silicon with carbon displaced, the
analytic force was **1.00305** of Q3's own dE/dR with symmetry enabled, against
0.99987 over the full zone and 0.99987 with no U at all. 0.30% of the total
force, ~7% of the Hubbard contribution. Four candidate causes had been tested
and eliminated (group-averaging dn/dR, unsymmetrized per-term IBZ forces, an
inconsistent wedge across the finite-difference triple, and SCF noise); the one
unexplained measurement left was that the converged _energies_ differed between
the wedge and the full zone by 7.7e-6 Ry on identical geometry.

**Cause.** `reduce_kpoints_internal` folds −Rk into Rk as well as Rk into k, so
the wedge is the quotient by the spatial operations **and time reversal** — the
log says so: `64 -> 24 k-points (2 symmetry ops + time reversal)`. For a
collinear Hamiltonian Θ acts by conjugation, ψ\_{−k} = ψ_k\*, so with real atomic
orbitals

    n^{(−k)} = (n^{(k)})* = (n^{(k)})ᵀ

A full-zone sum contains both members of every ±k pair and its imaginary part
cancels; a wedge sum keeps one and leaves the whole antisymmetric imaginary part
behind. The group average added on 2026-08-08 restores only the _spatial_ half,
and cannot touch this one: D(R) is real, so D·(iB)·Dᵀ is still
antisymmetric-imaginary. Nothing removed it.

Measured on the SiC case: max|Im n| = **6.8e-3** against an off-diagonal real
part of 1.06e-2. It survives into Tr[n²] = Σ|n\_{mm'}|², moving it by 1.5e-4,
and into the D matrix the Hamiltonian applies. The force then disagreed with the
energy for a specific reason: `accumulate_hubbard_force_gradient` contracts
`2·Re[...]`, which _is_ the time-reversal average, against a D matrix built from
a complex `n` that had not had it.

**Fix.** Average over time reversal as well —
`symmetrize_hubbard_occupation_matrices_under_time_reversal`, which is (n + nᵀ)/2
= Re n for Hermitian n. QE arrives at the same place from the other side: its
collinear occupation matrix is real by construction, accumulating only the
squared modulus of each projection (it carries a complex occupation matrix only
in the noncollinear case, where Θ is not a symmetry of a single spin channel).
Unconditional: a full-zone `n` is already real to 1e-11, and every
Monkhorst-Pack mesh is closed under k → −k, so it is a no-op exactly where it is
not needed.

|                            | analytic / dE/dR | Hubbard force vs full zone | E(IBZ) − E(full zone) |
| -------------------------- | ---------------- | -------------------------- | --------------------- |
| symmetry on, before        | 1.003049         | 1.48%                      | −7.7e-6 Ry            |
| symmetry on, after         | **1.000005**     | **0.031%**                 | +4.35e-5 Ry           |
| full zone (unchanged)      | 0.999868         | —                          | —                     |
| no U at all, same geometry | 0.99987          | —                          | +4.21e-5 Ry           |

Read the last two columns together. The wedge and the full zone still differ by
4.35e-5 Ry — _more_ than the 7.7e-6 that was the clue — but the no-U control on
the same geometry differs by 4.21e-5 Ry, so that residual is ordinary wedge
discretization with no DFT+U signature at all. Before the fix the spurious
imaginary part pushed E_U the other way and masked it down to 7.7e-6. The same
reading applies to the force: the wedge-versus-full-zone gap was 0.11% with U
and is now 0.036%, against 0.042% for the no-U control.

Gated by `test_the_wedge_hubbard_force_is_the_gradient_of_q3s_own_energy` (0.1%,
which sits 3x below the defect and 200x above what is left) and
`test_the_wedge_hubbard_force_matches_the_full_zone`, plus two unit tests on the
averaging itself. Both validation gates were confirmed to fail with the fix
disabled and the rest of the branch unchanged.

### DFT+U forces through the overlap operator (resolved 2026-08-09)

For ultrasoft and PAW the Hubbard projector is S|φ⟩ rather than |φ⟩, so the
DFT+U force carries dS/dR terms as well as dφ/dR. Q3 returned **zero** for the
entire Hubbard contribution and warned that total forces were incomplete.

Implements the same DFT+U force QE uses for these datasets. S|φ⟩ is never
built: expanding ⟨φ_m|S|ψ⟩ = ⟨φ_m|ψ⟩ + Σ q_ij⟨φ_m|β_i⟩⟨β_j|ψ⟩ needs only
overlaps the calculator already forms plus ⟨φ|β⟩, and cannot drift from the
SCF's own definition of S.
The structural change is that a Hubbard force now appears on **every** atom
carrying projectors, not only on Hubbard atoms — displacing an ordinary atom
moves the β functions inside S.

The sign convention was fixed by requiring the new path to reduce exactly to the
validated norm-conserving one when q_ij = 0, which is safer than re-deriving it
from the energy.

Measured on silicon ultrasoft with U = 3 eV: the force matches the derivative of
Q3's own energy to **0.044%** (ratio 0.99956). With the Hubbard term omitted, as
before, the ratio is 0.941 — the missing piece was **5.9%** of the total force
on the displaced atom. Gated by
`test_hubbard_force_is_the_gradient_of_q3s_own_energy_for_ultrasoft`, paired
with an assertion that the contribution is actually non-zero so the pair cannot
pass on a regression back to zero.

The DFT+U stress for atomic projectors is covered separately below;
ortho-atomic derivatives are covered by the resolved entry above.

### DFT+U forces with ortho-atomic projectors (resolved 2026-08-09, norm-conserving)

Ortho-atomic forces were refused outright: the gradient needs the position
derivative of the joint, system-wide Löwdin `O^-1/2`, and every atomic orbital
in the cell contributes to it, not just the manifold's own.

Implements the same ortho-atomic DFT+U force derivative QE uses. The
Löwdin transform now returns its eigendecomposition alongside `O^-1/2`, because
`dO` and `O^-1/2` do not commute — the derivative is the solution of a Sylvester
equation, diagonal only in `O`'s own eigenbasis, and the naive
`-½ O^-3/2 dO` is wrong in a way that still looks dimensionally right.
The structural consequence, as with dS/dR above, is that a Hubbard force now
appears on atoms that carry **no** U at all.

Gated on the whole force **vector** rather than one projection: three
crystal-axis central differences of Q3's own total energy determine `F·a_i` and
those are solved for the Cartesian force, so no component goes untested. The
deck states its lattice vectors explicitly, because those rows are the axes the
differences are taken along.

Measured on norm-conserving SiC with U = 3 eV on silicon only and carbon
displaced off-symmetry in all three crystal directions:
`|F_analytic − F_numerical| / |F|` = **4.9e-5**, against **4.2e-2** with the
`d(O^-1/2)` term dropped — a factor of 860, verified by building Q3 without it.
Dropping the term does not zero the Hubbard force on carbon, it reverses its
sign, because the ortho-atomic orbital on silicon already carries weight on
carbon's orbitals through `O^-1/2` itself.

**Ultrasoft and PAW remain refused**, at SCF setup rather than at the force: the
orthogonalization would first have to use the ⟨φ|S|φ⟩ metric instead of ⟨φ|φ⟩
(QE applies the overlap operator before orthogonalizing). Differentiating an
orthogonalization performed in the wrong metric would only make a wrong energy's
gradient consistent with itself.

### The DFT+U stress tensor (resolved 2026-08-10, atomic projectors)

Q3 reported no Hubbard stress at all, for any dataset family. On silicon
ultrasoft at U = 3 eV that omission was **29% of the total pressure**; on copper
ultrasoft at U = 4 eV on the 3d shell it was 22600% of it — the term is
−4353 kbar and the converged total is +19.

E_U depends on strain only through `P_m = ⟨φ_m|S|ψ⟩`, so with ψ and the
occupations held fixed

    dP_m/dε_ab = ⟨dφ_m/dε_ab|S|ψ⟩ + ⟨φ_m|dS/dε_ab|ψ⟩
    dS/dε_ab   = Σ_ij q_ij ( |dβ_i/dε_ab⟩⟨β_j| + |β_i⟩⟨dβ_j/dε_ab| )

and for any `A = f(|k+G|)·Y_lm(k+G)·S_I/√Ω` — both φ and β are —

    dA/dε_ab = −[ f'(q)(k+G)_a(k+G)_b/q·Y_lm + f(q)·(dY_lm/dq_a)(k+G)_b ] − δ_ab·A/2

Implements the same Hubbard stress decomposition Quantum ESPRESSO uses, reusing
the ⟨φ|β⟩ and per-atom q_ij assembly the force built on 2026-08-09,
`NonlocalPseudopotentialStressTensor`'s existing radial/angular β-derivative
construction, and — for the orbitals' `f'(q)` — the derivative of the same
tabulated χ(q) the projectors themselves are interpolated from, rather than a
new analytic radial routine.

**The sign was pinned by reduction, not re-derived.** Substituting the dA/dε
above into `-(1/Ω) dE/dε` reproduces
`NonlocalPseudopotentialStressTensorCalculator`'s own accumulation exactly — the
factor 2 on the derivative overlaps and the factor 1 on the δ_ab diagonal are
what that split produces — which fixes the convention against a validated term.
Setting q_ij = 0 then collapses the expression to the norm-conserving limit.

**One bug, and it was worth 16x.** Q3 carries `i^l` on its atomic orbitals
(`build_atomic_orbitals_g_space`) and `(-i)^l` on its beta projectors
(`compute_angular_phase_factor`). Using the projector convention for the
orbitals' derivative parts multiplies the radial and angular halves by −1
relative to the 1/√Ω half they must be added to, reading **−571.47 kbar**
against −34.89 on silicon 3p — wrong by a factor of 16 while still looking like
a stress.

Measured against QE's own `hubbard stress` line, analytic against analytic at
the same cutoff so no Pulay term enters:

| case                              | dataset   | manifold | Q3 (kbar)    | QE (kbar)    |
| --------------------------------- | --------- | -------- | ------------ | ------------ |
| Si diamond, U = 3 eV on 3p, 4×4×4 | ultrasoft | l = 1    | **−34.89**   | **−34.89**   |
| Si diamond, U = 3 eV on 3p, 4×4×4 | NC        | l = 1    | **−55.96**   | **−55.96**   |
| Si diamond, U = 3 eV on 3p, Γ     | ultrasoft | l = 1    | **−39.43**   | **−39.43**   |
| Cu fcc, U = 4 eV on 3d, 4×4×4     | ultrasoft | l = 2    | **−4353.04** | **−4353.07** |

and the totals follow: silicon −118.907 against −118.91, copper 19.16 against
19.38 (the 0.22 kbar residual is kinetic −0.11, local −0.64 and nonlocal +0.50,
all of them pre-existing; the Hubbard term supplies +0.03).

The IBZ wedge and the full zone agree — −118.907155 kbar over 8 k-points against
−118.904812 over 64 — which is the check that `dn/dε` is consistent with the
Θ-averaged `n` it is differentiating. Gated by `test_qe_hubbard_stress.py`, whose
discrimination was verified by rebuilding twice: with the term disabled 9 of its
10 tests fail (copper's total pressure reads +4372.20), and with only the two
dS/dε terms dropped the two ultrasoft cases fail while norm-conserving still
passes untouched — silicon reads −98.96 and **copper reverses sign to +3934.87**.

**PAW is not gated against QE**, only exercised: every committed PAW dataset in this
repository leaves its PP_CHI shells unlabelled, and QE itself refuses
`U Si-3p` / `U Cu-3d` on such a file rather than guessing. PAW shares the
ultrasoft branch (`has_ultrasoft()`) and produces a non-zero term there
(−6401.47 kbar on copper PAW at U = 4 eV).

The local QE reference build also ships a complete, labelled PSLibrary copper
PAW dataset. LibXC QE accepts it and reports a `-3711.29 kbar` Hubbard stress on
the matched deck, but Q3's current PAW reader diverges on that older
representation before a comparison can be made. The dataset was left
unmodified; its format-compatibility gap is tracked with PAW validation rather
than being hidden by editing the reference data.

**Ortho-atomic stress is covered separately** by the resolved entry above,
including the ultrasoft/PAW generalized metric and its derivatives.

### DFT+U stress is missing — refused rather than warned (2026-08-09)

**Status:** superseded for every currently supported projector/dataset
combination by the entries above. The result-level refusal mechanism remains as
a safety contract for any future projector type with a derivative gap.

No DFT+U stress term existed for any dataset family. It was announced only as a
log warning — invisible to anything reading the results
programmatically. A geometry or cell relaxation driven by them does not fail;
it converges, to the wrong answer.

Q3 records `stress_is_incomplete` with a reason in its results JSON (and
`forces_are_incomplete` for any future gap), and `raise_if_derivative_incomplete`
in the ASE calculator refuses the affected property when a caller asks for it —
so a cell relaxation on a Hubbard system stops instead of converging to a wrong
cell. Energies are correct
and single-point runs on the same system are unaffected, so only the calculation
that would have been wrong is stopped. Covered by four tests in
`tests/python/test_ase_calculator.py`.

The DFT+U derivatives themselves have since shipped for every currently
supported projector/dataset combination, which is what supersedes this entry;
the refusal mechanism above stays as the contract for any future gap.

### Nickel's PAW magnetic stabilization energy is not validated

**Status:** open.

The PAW moment of ferromagnetic nickel is gated, and the absolute totals sit inside the
known per-element one-center residual band, but the _difference_ between the magnetic
and non-magnetic states does not cancel that constant, and it disagrees with an
independent reference code's. The nickel gate therefore checks the moment only, and says
why.

### PAW one-center residual grows with d-character

**Status:** open, characterized, five hypotheses falsified.

PAW absolute totals carry a per-element offset against reference codes, smallest for
light sp elements and largest for the d elements; against QE the drop-in case set
measures 2.9e-4 to 6.9e-4 Ry on its PAW cases. It is a per-element _constant_, so energy
differences — equations of state, relative stabilities — are unaffected. QE sees it, so
it is not an XC-library difference. Five candidate explanations have been tested and
ruled out; do not re-run those.

## Wannier interpolation

### Copper's augmented Wannier models do not reach Wannier90's minimum

**Status:** open. Silicon agrees on both augmented families; copper's two
augmented models are unvalidated.

Q3 and Wannier90 4.0.1 localise Q3's own exported matrices, measured 2026-09-14
once a PAW non-self-consistent run carried D's one-centre term:

| case              | Ω_I, Q3 vs Wannier90 (Å²) | Ω, Q3 vs Wannier90 (Å²) | worst centre (Å) |
| ----------------- | ------------------------- | ----------------------- | ---------------- |
| silicon PAW       | agrees                    | agrees                  | agrees           |
| silicon ultrasoft | agrees                    | agrees                  | agrees           |
| copper ultrasoft  | agrees                    | 9.7409 vs 7.9083        | 0.778            |
| copper PAW        | 8.9993 vs 8.9985          | 51.2714 vs 32.5988      | 0.802            |

Silicon PAW used to disagree (Ω 6.0277 vs 6.2258 on 2026-09-10), which read as
an augmentation term right in magnitude and wrong in phase. It was the capture:
the NSCF had diagonalized without the one-centre term (resolved 2026-09-14,
below), so neither code was localising silicon. On the fixed capture it agrees
on the isolated four-band manifold to every gate, which is the direct check on
the augmentation in A_mn and M_mn(k,b) that this entry had lacked.

The copper rows are localisations that do not finish in the same place, and
the numbers say which side stops. On copper ultrasoft Wannier90's last
iterations move Ω by 4e-6 Å², while Q3's localisation stalls at 9.74 Å² with a
gradient of 0.07 after 863 iterations. Copper PAW's eighteen functions -- every
shell of the file, the pseudo-atomic set before the trial-shell rule -- stall
both: Q3 at a gradient of 3.8 after 905 iterations, Wannier90 still falling
0.02 Å² an iteration at its 1000-iteration cap, so those two Ω compare
unfinished minimisations. Its Ω_I, with both disentanglements reporting
convergence, is 7.7e-4 Å² apart against the 1e-4 gate, where on the defective
capture it had agreed.

Found the day the augmented path first ran. `capture_interpolation_record`
downcast the iterator's by-value base quantum system rather than its
polymorphic snapshot, so the cast could never succeed and `build_ultrasoft` had
never executed outside its own unit tests. The copper cases are `xfail` in
`tests/validation/test_wannier90_parity.py` with these numbers beside them;
every part that agrees stays a strict assertion.

### A state the trial orbitals cannot represent is left out of the model

The trial orbitals come from a dataset's `PP_CHI` shells, and a band that
projects onto none of them cannot be in the model. Copper's norm-conserving dataset has
no 4p shell, and its free-electron-like sp band above the Fermi level has zero
projectability at a handful of k-points (4 to 7 of 64 on the 4×4×4 example,
depending on the SCF; 17 of 512 on an 8×8×8 grid). The classifier keeps such a
state out of the frozen window rather than freezing it — freezing it makes the
projection gauge singular — and reports the band and the k-point count under
`interpolation.disentanglement.bands_excluded_from_freeze`. The model then has
no state of that character at those k-points, and the automatic 2 eV
window, which freezes the same band where it is barely reachable, produces a
tenth Wannier function of 8 bohr² and a localisation that stalls;
`q3.workflows.wannier` retries with the occupied manifold alone frozen and
recovers a stationary model, which is what the example used to reach with a
hand-set window.

**Status:** open by the nature of the dataset — reported, never hidden. A
dataset carrying 4p, or a trial set with an s orbital at the interstitial site,
would let the Fermi-surface state be kept.

### Semicore shells stay trial orbitals, so `bands` stays an override

`control.interpolation.bands` selects a contiguous range of bands at capture,
which is how gallium arsenide's four sp valence bands are reached under the ten
3d bands the pseudopotentials put in valence. The automatic protocol's
trial-shell rule leaves out only the empty channels an augmented dataset
repeats for an l its atom has in valence; it keeps every occupied shell,
semicore included, so its unattended model of GaAs is the 26-function one
spanning semicore and valence together, and the four-function valence model
needs `bands: [19, 22]` by hand. No SCDM fallback
exists either, since no shipped dataset lacks atomic wavefunctions.

**Status:** open — tracked in [TODOs.md](TODOs.md) §1.3.

### The accept thresholds are not yet set from a population

The automatic workflow accepts at η ≤ 10 meV and η_max ≤ 100 meV, numbers the
plan inherited before any automatic model existed. The campaign
(`python scripts/validation/run_wannier_campaign.py`) is the population to set
them from, and it has run: 40 rows at the published 0.2 Å⁻¹ spacing, every one
reaching a verdict — 34 accepted at η between 0.69 and 9.70 meV, 6 rejected:
every norm-conserving copper geometry, at 209–254 meV and 87.6 meV for the
doubled cell (that dataset carries no 4p shell, so η weighs a state no trial
orbital can represent), and copper PAW, whose localisation stalls on all nine
attempts at η 218–302 meV with its thirteen functions. The augmented rows reran
on the trial-shell rule and a PAW NSCF that carries the one-centre term
(resolved 2026-09-14, below): silicon PAW accepts at 9.70 meV and diamond PAW
at 3.70, next to their norm-conserving rows' 8.61 and 3.45. Silicon ultrasoft
had rejected with its disentanglement still moving at 4000 iterations, in a
43-band window the workflow sized for 26 trial orbitals while the engine, which
skips the dataset's zero-norm 4s, built 24; with the rule's 18, which the
engine agrees with, in 31 bands, it accepts on the first rung at 1.91 meV.
Nothing lands between 9.7 and 87.6 meV, so the 10 meV gate is not exercised by
that roster; the 100 meV η_max gate is, with seven accepted rows at 63–82 meV.
The thresholds have not been re-set from the table, and of its augmented
accepts, copper ultrasoft at 0.69 meV is a model whose localisation does not
reach Wannier90's minimum (above). A rejection at the published spacing still
says the model missed those gates, not that it is unusable.

**Status:** open — the table exists; re-setting the gates from it is the item
in [TODOs.md](TODOs.md) §1.3.

### Wannier capture refused ultrasoft and PAW datasets (resolved 2026-09-10)

`control.interpolation` on an NSCF run failed on any dataset carrying
augmentation charges with "the pseudopotentials carry augmentation charges but
the quantum system is not the ultrasoft one". The augmented builder existed and
passed its own unit tests; nothing could reach it. The capture downcast the
iterator's base quantum system — the value member holding the wavefunctions —
rather than the polymorphic snapshot the factory built for the run, so the
downcast was null on every run, norm-conserving ones included; norm-conserving
captures never took that branch, so they were unaffected.

The capture now reaches the snapshot. It carries the pseudopotential tables but
stale electronic state, so the iterator's state is copied into it before the
builder reads wavefunctions out of it. PAW and ultrasoft datasets are
Wannierized — and that is what first exposed the comparison recorded above
under "Copper's augmented Wannier models do not reach Wannier90's minimum".

**Status:** resolved 2026-09-10 — the four augmented cases in
`tests/validation/test_wannier90_parity.py` that found it run; the two copper
cases that still `xfail` do so on their localisation, not on the capture.

The NSCF build step (`control.interpolation` on an NSCF run) additionally
refuses spin-polarized and Γ-only runs, any k-set that is not the full
unreduced grid, and more trial orbitals than bands; explicit trial orbitals
are s-type only. All of it is tracked in [TODOs.md](TODOs.md) §1.3.

## Non-self-consistent calculations

### The highest bands of a single NSCF pass can still be wrong

**Status:** open.
**Affects:** the empty bands of any non-self-consistent run, the highest most.

A single pass holds an occupied band to its tolerance but lets an empty band
stop once its eigenvalue moves by less than 1e-5 Ry an iteration. That is the
self-consistent rule, safe there because the next iteration carries on; a pass
has no next iteration and keeps what it stopped at. On copper's 4×4×4
norm-conserving Wannier mesh with sixteen bands, against direct diagonalization
of the same Hamiltonian, bands 1–15 are within 0.056 meV and band 16 is 1.2 meV
off at one k-point of 64. Holding every band to the pass's tolerance brings all
sixteen within 0.005 meV, for 21% more Hamiltonian applications. Until a pass
does that, request a band or two more than a calculation reads. Tracked in
[TODOs.md](TODOs.md) §2.1. The 68.8 meV and 0.85 meV this entry used to quote
came from a looser re-diagonalization, resolved 2026-10-08 (below).

## Testing coverage gaps

### QE-comparison validation only runs on developer machines

`tests/validation/test_qe_energy_comparison.py`, `test_qe_projwfc_comparison.py`
and `test_qe_graphite_d3_dispersion.py` skip unless a local Quantum ESPRESSO
reference build is present (no CI environment installs QE). The physics they
gate — total-energy parity, PDOS/Löwdin projections, D3(BJ) dispersion — is
therefore validated at development time, not continuously. The LSDA
spin-channel layout assumption in the PDOS extraction is exercised only by
these tests.

### Gamma phonon workflow crashes in some environments

**Status:** open, environment-specific, unresolved.

`tests/validation/test_gamma_phonons.py` exits 133 (SIGTRAP) at the first test
on macOS with ASE 3.29 + NumPy 2.5 — no Python traceback and no crash report.
Nothing in Q3's own phonon path is implicated; the rest of the validation suite
runs with it excluded.

## Resolved

### A smeared metal kept half a micro-electron too many (resolved 2026-10-20)

**Affected:** every run with smearing on a metal since 2026-08-18. The Fermi energy sat
0.5e-6/DOS(E_F) above the value that makes the cell neutral, so the cell held 5e-7 extra
electrons and the total energy came out E_F × 5e-7 Ry high: on the copper NC example, E_F
1.243 Ry and 13.1 states/Ry, a 3.8e-8 Ry shift in E_F and +0.62 μRy in the energy. A gapped
system, whose bracket spans its gap, was not affected.

The fix for a gapped system under smearing (FeO with U on Fe 3d) brackets the flat stretch of
the electron count from both sides and takes its midpoint, so that E_F sits inside the gap
instead of on the valence band edge. The upper bisection stops where the count reaches the
target plus a 1e-6 slack, so a metal has a bracket too: the slack over its density of
states, 7.6e-8 Ry on copper. The midpoint was meant to be skipped on a metal, but the test
that skipped it compared the bracket with the bisection tolerance, 1e-12 Ry, which every
metal's bracket exceeds. The copper smearing gate in `test_qe_energy_comparison.py` caught
it once a separate correction had removed an error that offset it.

**Fix.** The midpoint is taken only for a bracket wider than the smearing width. On FeO the
lower end of the bracket sat 3.9 σ below the E_F an independent implementation reports, so
that bracket is several σ wide; copper's is about 4e-6 σ. Unit tests require a metal's
occupations to sum to its electron count to 1e-9 (5e-7 before), and Marzari-Vanderbilt
smearing across a 12 σ gap, whose bracket spans 8 σ, to keep the midpoint. The four copper
smearing schemes moved from 1.55 to 0.93 μRy above Quantum ESPRESSO 7.5, inside their
1 μRy gate. The 0.93 μRy that remains is the wavefunction grid, which YAML input sizes for
2 × ecutwfc. With `control.use_exact_wavefunction_grid: true`, which sizes it as pw.x
does, all four agree with pw.x to 0.005 μRy; without the fix they sat 0.62 μRy above it.

### A pw.x relax or vc-relax deck could not be relaxed (resolved 2026-10-20)

**Affected:** every Quantum ESPRESSO `relax` / `vc-relax` deck. Reported as camsai/q3#28.
The workflows (`run_relax_workflow`, `run_vc_relax_workflow`) read only YAML manifests, the
binaries refused a relaxation outright, and nothing could run a relaxation step on `q3_mpi`
or a GPU binary.

**Fix.** A relaxation input runs as one SCF at its own geometry, from a saved density when
one fits, on every binary (`parse_input`). `--material` substitutes a step's structure, and
`results.json` carries the ESSE `material` the engine built, `if_pos` as constraints. The
workflows run a deck once as written, start ASE's BFGS from that material, and hand each
geometry back with `--material`, so any `ibrav` the engine reads relaxes and Python reads no
deck. Measured on silicon with one atom displaced: −16.7401734361 Ry at
`forc_conv_thr = 1e-4` against pw.x's −16.7401734, the same for the deck written with
`ibrav = 0` and `ibrav = 2`; −16.7401612799 Ry on `q3_mpi` (2 ranks) and Metal `q3_gpu`
alike; an `if_pos 0 0 0` atom stays put with zero force; restarting from the previous
density takes an SCF from 7 iterations to 2. Not yet measured: a `vc-relax` deck against
pw.x, whose fixed basis differs from Q3's per-step one by a Pulay offset at low cutoff.

### Multi-rank runs wrote no checkpoint, so nothing could restore them (resolved 2026-10-12)

**Affected:** every `q3_mpi` and `q3_mpi_gpu_cuda` run on more than one rank. Reported as
camsai/q3#27 from Perlmutter: a 64-rank LiCoO₂ SCF (150 irreducible k-points, 64 pools)
converged and wrote `results.json` but no `<prefix>.save/`. Both MPI binaries run SCF
only, so the NSCF or BerkeleyGW export that follows has to restore a checkpoint in a serial
run, and the SCF, the one step that could use many ranks, had to run on one as well.

The writers refused unless one process held the complete state, and under MPI that meant
one rank. That rule fits ranks sharing a pool, which split every k-point's coefficients and
the density grid. It was stricter than the automatic topology needs, which gives each rank
its own pool: such a rank holds each of its k-points whole, and the density, the potential
and the PAW ρ_ij are reduced across pools, so every rank holds them whole too. Only the
eigenvalues were per pool.

**Fix.** With one rank per pool, each rank writes the `wfc<N>.dat` files of its own
k-points under their index in the global grid. Rank 0 gathers the eigenvalues into global
order and writes `charge-density.dat` and `effective-potential.q3cp`. The directory is the
one a serial run writes. Ranks sharing a pool still refuse, now saying so; that layout
needs a gather, see TODOs.md §3.1. Measured on silicon, 4×4×4 shifted (10 k-points), 4
ranks against the serial binary:

| Pooled SCF, 4 ranks × 1 per pool                     | Before               | After                                                   |
| ---------------------------------------------------- | -------------------- | ------------------------------------------------------- |
| `si_nc.save/`                                        | not created          | `wfc1.dat`–`wfc10.dat`, density, effective potential    |
| Serial NSCF from it at Γ, X, L and a general k-point | fails: no checkpoint | matches the NSCF from a serial checkpoint to 1.4e-10 Ry |
| Stored eigenvalues against the serial checkpoint's   | —                    | 9e-15 Ry                                                |
| ρ(G) against the serial checkpoint's                 | —                    | 1e-16 of ρ(0)                                           |
| Occupied subspace of each `wfc<N>.dat`               | —                    | complete to 1e-14                                       |

`q3_mpi_gpu_cuda` uses the same execution policy and SCF flavor, so it takes the same path.
It was not run for this change, because the machine has no CUDA device.

`tests/validation/test_mpi_checkpoint.py` runs both SCFs and both NSCFs, and checks every
file: names, k-point headers, Miller indices, occupied subspaces, ρ(G) and the stored
eigenvalues. Against the old binary all 15 tests stop at "no checkpoint directory was
written". The pooled case of `test_mpi_mean_field_export.py` now expects the checkpoint
where it used to forbid one, and the shared-pool case still forbids it. Unit tests pin the
new topology predicate, including a pool whose coefficient distribution was switched off,
and check that the global k-point index names the k-point each pool was dealt.

### A QE-format D3 deck that left its options to pw.x ran a different correction (resolved 2026-10-12)

**Affected:** every QE-format deck with `vdw_corr = 'dft-d3'` or `'grimme-d3'` that did
not state both `dftd3_version` and `dftd3_threebody`, and every deck with any other
`vdw_corr` (DFT-D2, Tkatchenko-Scheffler, MBD, XDM).

pw.x defaults to `dftd3_version = 3`, which is zero damping, and to
`dftd3_threebody = .true.` (`system_defaults` in QE's `Modules/read_namelists.f90`, the
same in 7.2, 7.4 and 7.5). A deck that says `vdw_corr = 'dft-d3'` and nothing more
therefore runs zero-damped D3 with the three-body term in pw.x. Q3 implements only the
two-body term with Becke-Johnson damping, which pw.x runs for `dftd3_version = 4` and
`dftd3_threebody = .false.`. The reader mapped `vdw_corr` to that unconditionally and
checked the two keys only when the deck stated them, so it ran such a deck as two-body
D3(BJ) and reported success. The fix for "The QE input parser ignored what it did not
recognize" below made a stated value count; a value left to pw.x still did not. Any other
`vdw_corr` fell through and ran with no dispersion correction at all, although
[docs/io_reference.md](docs/io_reference.md) §5.3 said it was refused.

**Fix.** The reader notes which of the two keys the deck states and, once the whole file
is read (QE takes a namelist's keys in any order), refuses a D3 deck that leaves either
one out. The message names each missing key, the default pw.x takes for it, and the line
to add. A stated value Q3 does not implement is still refused where it is read, and a
`vdw_corr` other than D3 or `'none'` is refused. The committed `scf_qe_d3.in` decks state
both keys and still parse. Unit tests in `tests/unit/io/input/parser/test_qe.cpp` cover a
deck left at pw.x's defaults in each spelling, a deck missing only one key, the keys in
either order, and the other `vdw_corr` values; three of them fail against the old reader.

### PDOS and Wannier grids inherited a band-structure manifest's path (resolved 2026-10-11)

**Affected:** `run_pdos_workflow`, and the capture NSCF of `run_interpolation_workflow` and
`run_automatic_wannier_workflow`, whenever the SCF manifest they were given carried
`control.band_path`, as every `examples/band_structure--*/nc.yml` does. The interpolation
workflow invites it, since it takes the manifest's band path as the path to evaluate.

`write_nscf_input` copied the template's `control` section whole, `band_path` and
`band_path_points_per_segment` included, so the full-grid NSCF these workflows wrote
carried a path that does not describe it. The engine reads `band_path_points_per_segment`
as the length of each segment of the k-point list. At each multiple of it, up to the
path's length, it stopped seeding a k-point's wavefunctions from the one before, and it
re-solved the points it took for segment edges from their neighbours. On a grid those are
arbitrary points. Converged bands do not depend on where a solve starts, so the occupied
bands did not move. The highest bands did: a single NSCF pass converges them least well
(the open issue _The highest bands of a single NSCF pass can still be wrong_, above), and
what it returns for them depends on the seed.

On silicon, from the band-structure example's 30 points per segment, on the PDOS
workflow's default 4×4×4 grid with 12 bands, bands 1–4 and the Fermi level agree between
the two runs to 1e-14 Ry and bands 5–10 to 1.3e-6 Ry. Bands 11 and 12 moved by up to
3.9 meV, at the k-points just after the false boundaries at indices 30 and 60. Against a
20-band NSCF on the same grid:

| Silicon, 4×4×4 PDOS grid from the band-structure example, 12 bands | Before        | After         |
| ------------------------------------------------------------------ | ------------- | ------------- |
| Band 11 at k-point index 32                                        | 3.92 meV      | 0.019 meV     |
| Largest band 11 / band 12 error anywhere on the grid               | 6.4 / 3.8 meV | 6.4 / 3.8 meV |

The second row is the open issue, not this one, and is the same in both runs.

**Fix.** `write_nscf_input` takes `uniform_grid=True` from the two grid writers,
`run_pdos_workflow` and `write_interpolation_nscf_input`, and then drops both keys. Their
NSCFs now run exactly as they already did from a manifest without a band path, the case
every shipped PDOS and Wannier input is. The band-structure workflow's NSCF keeps the
keys, since its k-points are the path they describe. The Wannier protocol's reference
NSCF still copies them. Its default reference set is an off-mesh sub-grid of up to 125
points, so a band-structure template can give it the same false boundaries.

`tests/python/test_input_writer.py` checks that the option drops both keys and that by
default a path NSCF keeps them. `tests/python/test_interpolation_workflow.py` and
`tests/python/test_pdos.py` write each workflow's NSCF from a manifest carrying a band
path and find neither key in it. Both fail without the fix.

### The QE-format checkpoint writers produced files QE could not read (resolved 2026-10-04)

**Affected:** every `wfcN.dat` and `charge-density.dat` Q3 wrote into `<prefix>.save/`.

The writers in `src/io/wavefunction/kohn_sham/planewave/checkpoint/qe.hpp` and
`src/io/charge_density/kohn_sham/planewave/checkpoint/qe.hpp` followed QE's
`write_wfc` / `write_rhog` record sequence (`Modules/io_base.f90`) but diverged
from it in four ways, none visible to a test whose reader was written to Q3's own
conventions:

1. **The k-point header was 48 bytes, not 44.** It was dumped from a C struct,
   whose alignment put four padding bytes after the leading `int32`, so every
   later field sat four bytes after where QE's packed Fortran list
   `ik, xk, ispin, gamma_only, scalef` puts it. It is now appended field by field
   (`PackedRecordBuffer`): `int32` @0, `xk(3)` @4, `ispin` @28, `gamma_only` @32,
   `scalef` @36.
2. **Miller indices were transposed.** Both writers emitted all-h, all-k, all-l;
   QE's `mill(3, n)` is column-major with the component first, i.e.
   `h0 k0 l0 h1 k1 l1 …`. Both now interleave.
3. **`charge-density.dat` had no reciprocal-vector record.** QE writes `b1, b2, b3`
   between the header and the Miller indices; Q3 skipped it, so QE's reader took
   the Miller record for the vectors. It is now written.
4. **Units.** `xk` was written in fractional (crystal) coordinates and `b1..b3` in
   units of 2π/alat; QE writes both Cartesian in 1/Bohr (it passes `tpiba*xk` and
   `tpiba*bg`). Both now use 1/Bohr.

`tests/utils/qe/binary_readers.py` now reads QE's layout, so the existing Q3
round-trip checks moved with the format; new checks pin every byte offset
(`tests/unit/io/{wavefunction,charge_density}/kohn_sham/planewave/checkpoint/test_qe.cpp`),
place every Miller triplet inside the cutoff sphere, and -- where the pinned QE
build is present (`requires_qe`) -- parse a `wfc1.dat` written by `pw.x` and
compare Q3's `charge-density.dat` to QE's (reciprocal vectors, G sphere, ρ(G=0)).
Checkpoints written by earlier Q3 builds are not readable by the new reader.
Spin-polarized `.save` directories remain unvalidated -- see TODOs.md §3.1.

### A non-self-consistent pass was repeated at a looser tolerance than it had run at (resolved 2026-10-08)

**Affected:** non-self-consistent runs since 2026-06-28, when a pass began
diagonalizing at the 1e-13 Ry floor, whenever the residual dr2 between the
pass's density and the atomic superposition the iterator starts from fell below
1e-2 times the electron count. That depends on the k-point set, not the
physics. A uniform mesh -- a Wannier capture, a density of states -- can meet
it: copper's 4×4×4 Wannier mesh did, and so did silicon's two irreducible
k-points, though either one alone did not. The 151-point band paths in the
parity matrix did not. Self-consistent runs are unaffected.

A non-self-consistent run is one turn of the SCF loop, and that turn ends with
the SCF's first-iteration rule: when the density residual is below what the
first diagonalization could have left, diagonalize again at 0.1 dr2 / N. The
rule took that diagonalization to have run at the SCF's loose start, 1e-2 Ry,
whatever it had used. In a pass, dr2 measures the bands' density against the
atomic superposition, which says nothing about how well the bands converged,
and the "tighter" threshold was far looser than the floor the pass had run at:
7.8e-4 Ry for silicon, 1.5e-4 Ry for copper. The repeat seeds each k-point from
the one before, discarding the vectors that k-point had converged, and stops at
that threshold.

Against direct diagonalization of the same Hamiltonian, copper's sixteen-band
norm-conserving pass on its 4×4×4 Wannier mesh (one SCF checkpoint for all):

| Davidson basis     |        | worst error, bands 1–14 | band 15   | band 16  |
| ------------------ | ------ | ----------------------- | --------- | -------- |
| 2 × the band count | before | 40.8 meV                | 258 meV   | 3.8 eV   |
|                    | after  | 0.099 meV               | 0.43 meV  | 1.18 eV  |
| 4 × the band count | before | 0.85 meV                | 1.4 meV   | 68.8 meV |
|                    | after  | 0.038 meV               | 0.056 meV | 1.2 meV  |

The "before" rows are the numbers recorded under "A non-self-consistent run
could miss eigenvalues at the default Davidson basis" below, and the ones the
open entry on a pass's highest bands quoted -- including the 0.85 meV it judged
not to be convergence. At a factor of 2 the missing eigenvalue in bands 1–14 is
gone, while band 16 still skips a state at one k-point, so a pass keeps 4; the
1.2 meV left at 4 is the empty-band stopping rule, open above. The pass is also
cheaper without its second sweep: 11,994 Hamiltonian applications on copper
instead of 19,447. On silicon (norm-conserving, 30/120 Ry, shifted 2×2×2 SCF),
a pass over both irreducible k-points put the second's occupied bands
2.9e-6–4.4e-5 Ry from a pass over it alone; they now agree to 2.2e-14 Ry.

**Fix.** The error estimate is the threshold the diagonalization actually ran
at, times the electron count, as Quantum ESPRESSO's is, and the pass is
repeated only at a threshold tighter than that one. A
self-consistent run's first diagonalization runs at 1e-2 Ry, so self-consistent
runs are unchanged bit for bit. Gated by
`NonSelfConsistentFieldIntegration.APassOverSeveralKPointsKeepsItsTolerance`,
which fails without the fix (4.4e-5 Ry against a 1e-10 Ry bound). The parity
matrix is byte-identical, its band-structure and projection passes included.
Non-self-consistent results on uniform meshes computed before the fix --
Wannier models and densities of states above all -- should be rerun; a run
whose log has no "Re-diagonalizing" line after its "NSCF:" lines was not
affected.

### Wurtzite's screw-axis degeneracies split on k-meshes odd along c (resolved 2026-10-09)

**Affected:** every symmetric run, the default, of a crystal whose space group is polar
with a screw axis or glide translating along the polar axis, on a mesh holding k-points
whose time-reversal partner is outside their own star. For wurtzite (P6₃mc: AlN, GaN,
InN, ZnO, 2H-SiC) that is any mesh odd along c. Cmc2₁, Pca2₁, Pna2₁ and P2₁ are of the
same kind. The dataset type does not matter. Reported as camsai/q3#24: AlN, GaN and InN
split pairs that symmetry makes degenerate at K, A, L and H by 2–8 meV, where the same
Q3 deck with `control.disable_symmetry` gave 0.00 meV.

The density symmetrizer averages each shell of symmetry-related G-vectors over the space
group, each operation with its fractional-translation phase, and it skipped shells of one
member. In a polar group a G-vector along the polar axis is left in place by every
rotation and is such a shell. When the group's screw axes or glides translate along that
axis, its average is zero: these are crystallography's systematic absences, (0, 0, l) with
l odd for wurtzite. A k-point of the reduced wedge feeds those components whenever its
time-reversal partner lies outside its own star, as it does off wurtzite's k_z = 0 and 1/2
planes. On a mesh odd along c they therefore came through unaveraged. The density's
planar average along c lost its c/2 period, and the potential lost the 6₃ screw. On AlN
6×6×3, the odd components reached 9e-6 in the density and 2e-4 Ry in the potential. The
issue suspected the PAW on-site occupations, but they matched the full mesh's to 4e-5,
and norm-conserving datasets split too.

**Fix.** A one-member shell's phases form a character of the group, so its average is
exactly 1 or exactly 0. The symmetrizer finds the zeros once per cell and sets those
components to zero after every average. Crystals with inversion, symmorphic crystals,
and polar groups with no translation along the axis have none, and their runs are
bit-identical.

| Case                                                            | Before      | After       |
| --------------------------------------------------------------- | ----------- | ----------- |
| AlN as reported, 9×9×5: lowest pair at K                        | 7.94 meV    | 0.0001 meV  |
| The same on 6×6×3                                               | 7.79 meV    | 0.0001 meV  |
| The same on 9×9×5: E(wedge) − E(full mesh)                      | +1.95e-6 Ry | +1.6e-7 Ry  |
| 2H-SiC, norm-conserving, 3×3×3: widest occupied pair at A, L, H | 35.5 meV    | 2.6e-10 meV |
| The same, exact wavefunction grid: E(wedge) − E(full mesh)      | +3.0e-5 Ry  | +2.7e-13 Ry |

The leftover 1.6e-7 Ry on AlN is the default 2 × ecutwfc wavefunction grid aliasing the
band density, not this defect. On 2H-SiC that grid leaves the wedge and the full mesh
3e-7 Ry apart on an even mesh too, before the fix and after it, and the exact grid
removes it. The issue's second point is efficiency, not correctness, and is not taken up
here: on cells whose γ is 120.000006°, Q3's symmetry tolerance finds Cmc2₁ rather than
the hexagonal group. Cmc2₁ is polar too, with its 2₁ screw
along c, so it forbids the same components and the same change covers those runs.

Two tests gate the fix. The unit tests in
`tests/unit/layer_4_computation/charge_density/planewave/test_init.cpp` cover wurtzite's
invariance under all 12 operations, odd (0, 0, l) set to zero and even passed through
exactly, idempotence, and the real-space entry point. A diamond control asserts that a
centrosymmetric cell has no absences. `tests/validation/test_screw_axis_wedge_consistency.py`
runs the 2H-SiC case: an NSCF from the wedge on the k_z = π/c face, where every level is a
pair, and the wedge's energy against the full mesh's. With the fix disabled, the unit test
fails at an invariance violation of 2.13. The validation test fails at 35.5 meV and
3.0e-5 Ry against the unfixed binary.

### Forces were wrong whenever ranks shared a k-point pool (resolved 2026-10-09)

**Affected:** every `q3_mpi` run with more ranks than pools, at any ratio; two ranks per pool
were enough. Reported as camsai/q3#22 from Perlmutter. BaBrCl (Pnma) with 8 ranks per pool
had a largest force of 0.197 Ry/bohr where one rank per pool and pw.x give 2.4e-3. Wurtzite
ZnO carried a net force of 1.4 Ry/bohr along z and nonzero x and y components on atoms
sitting on 3-fold axes. The energies agreed with the one-rank-per-pool runs to 1e-10 Ry. A
relaxation driven by these forces moved uphill at every BFGS step, and nothing marked them
incomplete the way the stress already is in this layout.

When ranks share a pool, each holds only its slab of every k-point's plane waves. The
nonlocal force multiplies two projector overlaps, ⟨β|ψ⟩ and ⟨β|−i(k+G)ψ⟩, and both were
computed from the rank's own slab and multiplied there. That drops every cross term
between two ranks' slabs, and the pool-to-pool sum afterwards cannot restore them. The
energy needs only the projections the SCF had already summed across the pool, which is
why it stayed exact. The DFT+U force had been written the right way, summing its
overlaps before multiplying. Every other contribution (local, Ewald, core correction,
augmentation) sums over G-vectors every rank holds in full, and measured identical
across layouts.

**Fix.** The nonlocal force sums its four overlap matrices across the pool before forming
any product, as the DFT+U force does. Measured on 2H-SiC (norm-conserving, P6₃mc) with
two ranks in one pool, the nonlocal term was 0.449 Ry/bohr off and the net force +0.75
Ry/bohr along z; it now matches one rank to 8e-11 per term and 5e-15 in total, with the
net force back to the serial −2.1e-4. Silicon with an ultrasoft dataset and a displaced
atom was 0.149 Ry/bohr off and now matches to 3e-14. PAW shares that path.
`tests/validation/test_mpi_topology_invariance.py` gains a case on two ranks in one pool
that checks the plane waves were split before comparing forces against one rank. Every
earlier case gave each pool one rank. Against the old binary it fails at 0.447 Ry/bohr.
Stress under a split basis is still refused; see "Stress tensor is skipped under band
parallelism" above.

### The `-mpi` image's FFTW had no SIMD, so its FFTs ran 1.5× slower (resolved 2026-10-09)

**Affected:** every run of the `-mpi` container image. Reported as camsai/q3#20: k-point-pool
runs on Perlmutter were 1.3–1.5× slower than a native build of the same commit at every
rank count, and the whole difference was FFT time. On silicon `nc_kpt_8x8x8.yml` the FFT
stage took 2746.6 ms against 1576.1 ms, with every other stage equal and scaling unchanged.

`Dockerfile.mpi` builds FFTW from source so that it matches the image's MPICH, and FFTW's
`configure` enables no SIMD codelets unless asked. The library it produced held scalar code
only: no instruction on an AVX register, against 54,420 in Ubuntu's `libfftw3`, which the
serial and CUDA images use and which keeps them at parity with native builds.

**Fix.** The FFTW step passes `--enable-sse2 --enable-avx --enable-avx2` on x86_64 and
`--enable-neon` on arm64. FFTW picks its codelets at plan time from the CPU it runs on, so
the image still runs on x86_64 hosts without AVX2. The step now fails the image build if
the codelets go missing again: on x86_64 no instruction on a `ymm` register, on arm64 no
NEON codelet symbol. The arm64 flag set was built natively from FFTW 3.3.10 here and gives
207 NEON codelets. The x86_64 check runs in `container.yml`'s amd64 build. The speedup
itself still wants a rerun of the benchmark on Perlmutter against the next image.

### The BerkeleyGW export refused every non-cubic crystal (resolved 2026-10-09)

**Affected:** `mean_field_export: berkeleygw` on any cell Q3 holds as a free lattice
(Bravais index 0). That is every YAML lattice outside the cubic family, YAML
`lattice.vectors`, and QE's `CELL_PARAMETERS`, which is how Materials Project cells
arrive. Reported as camsai/q3#21: wurtzite ZnO got through its SCF and a 200-band NSCF,
then the export threw. The only input route that gives a hexagonal cell a nonzero
index, QE's `ibrav = 4`, cannot carry the explicit k-point list an NSCF and WFNq need,
so no non-cubic crystal could be exported at all.

The header's `cell_symmetry` (0 cubic family, 1 hexagonal) followed pw2bgw's bracketing
of `ibrav`. For index 0 pw2bgw reads a namelist input Q3 has no counterpart of, and Q3
threw instead, after the WFN file was already open, so a 104-byte truncated `WFN` was
left behind.

**Fix.** A free lattice is classified from its vectors into the same two families.
Two vectors of equal length at 120° or 60° with the third perpendicular is hexagonal
(index 4's family). Three equal vectors at one common angle other than the cubic
primitive cells' 90°, 60° and 109.47° is trigonal R (index 5's). Anything else is the
cubic bracket. The comparison tolerates a database's 120.000006°. A file whose writing
fails part-way is now removed, and a non-self-consistent pass no longer logs "SCF did
not converge after 1 iterations" for the single pass it is by design.

On 2H-SiC (P6₃mc, the reported ZnO's space group, given as a YAML `HEX` lattice) the old
binary exits 1 with the refusal and a 104-byte `WFN`. The fixed one writes `WFN`, `RHO`
and `vxc.dat`, with `cell_symmetry = 1`, 12 symmetry operations and 4 atoms in the
header. Unit tests cover the index mapping, hexagonal cells in three settings, the
rounded angle, trigonal R against the cubic primitive cells, the other families, the
written header for free hexagonal and cubic lattices, and the removal of a failed
`WFN`.

### An input that named no functional ran LDA, whatever its pseudopotentials were (resolved 2026-10-07)

**Affected:** every Q3 YAML deck without `model.functional` and every QE-format deck without
`input_dft`. The BPO4 deck of camsai/q3#18, run at NERSC with SG15 ONCV PBE pseudopotentials
and no `input_dft`, was computed with Perdew-Zunger LDA.

`InputData` carried a default functional of its own, `PZ`, and neither reader compared it with
the datasets. pw.x does the opposite: `input_dft` overrides the functional read from the
pseudopotential files, and without it the files decide. The run converged and reported
success. Only the `XC functional` line of the log and the provenance record in the results
file named PZ. On the silicon example with its `model.functional` removed (`nc.yml`, PBE
`Si.upf`), the total energy came out −16.894765600 Ry against −16.913417930 Ry with
`functional: pbe`: 18.65 mRy for the two-atom cell, 127 meV/atom.

**Fix.** `InputData` no longer has a default functional. `parse_input()`, which every entry
point reads its input through, gives an input that names none the functional its
pseudopotentials state. It reads it from the UPF header, as a short name (`PBE`, `PZ`/`LDA`,
`PBEsol`, `BLYP`, `revPBE`, `RPBE`), the four-component form (`SLA PW PBX PBC`), or QE's LibXC
form. Datasets that disagree, that state a label Q3 does not map (Perdew-Wang LDA, `PW`, is
not PZ), or that cannot be read are refused by name, and naming the functional in the input
settles any of them. A named functional still wins, as `input_dft` does in pw.x. `--check`
and the provenance record report the resolved functional. The Python `write_scf_input` no
longer writes `pbe` when its caller names no functional, so the engine takes the datasets'
own there too.

The recorded name is now lower-cased once, in `parse_input()`, as the example decks and ESSE
spell it. Before, the YAML reader upper-cased it and the QE reader kept `input_dft` as
written, so one functional could be recorded as `PBE` or `pbe` depending on the reader. The exchange-correlation factory still matches case-insensitively,
so what runs is unchanged.

Unit tests pin the label mapping, every bundled dataset across both UPF versions, the three
refusals, and `parse_input` on YAML and QE-format decks with and without a functional and
in any case. `tests/validation/test_functional_from_pseudopotentials.py` runs the silicon
example without its functional and requires the PBE energy and a provenance record of `pbe`,
the same as the deck that names it. Against the old binary it fails with the 1.87e-2 Ry
difference and a provenance record of `PZ`.

### QE-format decks with hard pseudopotentials at low cutoffs disagreed with pw.x by mRy (resolved 2026-10-09)

**Affected:** QE-format decks compared against pw.x, in proportion to how much weight
the wavefunctions carry near the cutoff sphere: negligible on converged decks (at most
16 μRy on the QE-parity decks), mRy on hard pseudopotentials at low cutoffs.

The wavefunction FFT grid is sized for 2 × ecutwfc, a speed choice made in 2026-06
when a converged copper deck agreed with pw.x to 1e-7 Ry on it. A wavefunction reaches
G_max = √ecutwfc, so |ψ|² reaches 2 G_max, and on a grid holding only √2 G_max both
|ψ|² and V·ψ alias. pw.x sizes the grid for 4 × ecutwfc, where neither does, and it
refuses to run on the smaller one. On J. Voss's Ni/CO slab, with a full-core-hole
carbon at 20 Ry, the 2 × grid put Q3 9.5 mRy below pw.x, with the Fermi level 83 meV
low; the gap shrank with cutoff (−0.89, −1.97, −0.09 mRy at 30, 40 and 60 Ry).

**Fix.** QE-format input now runs on the exact grid, sized for 4 × ecutwfc as pw.x
sizes it and capped at the density grid as before. YAML input keeps the 2 × ecutwfc
grid unless it sets `control.use_exact_wavefunction_grid: true`. The exact grid costs
1.4× to 2.3× the solver time on the silicon and copper examples on macOS, depending on
the thread count, and 2.8× to 4× on CI's x86 runners, which is why it is not the YAML
default. Voss's deck, as he sent it, now agrees with pw.x to 0.04 mRy, with the Fermi
level within 5 meV and the same 3 μB moment. Unit tests pin the exact grid to the
smooth grid pw.x 7.6 builds for FCC silicon at 16/128 Ry (20³), the default one to the
2 × ecutwfc rule, and the QE reader to the exact grid.

### Slabs and molecules picked up an energy that grew with the vacuum (resolved 2026-10-07)

**Affected:** every cell with a long direction — slabs, molecules, isolated atoms —
increasingly with the length of that direction: total energies, and through the same
table the local-potential forces and stress. Bulk cells move by under a μRy: 0.74 μRy on
the copper NC example, nearly all of it at its two shortest G-vector shells.

The local pseudopotential V_loc(q) was read from a uniform q-table (spacing 0.01
bohr⁻¹) by 4-point Lagrange interpolation, with its −4πZe²/(Ωq²) divergence included
in the tabulated values. A polynomial fit across that divergence is off by roughly
(dq/q)⁴ of it near the origin, and only a cell with a long direction has G-vectors that
close to it. The error went as the square of the valence charge times c⁵/area. On J.
Voss's Ni/CO slab, against pw.x 7.6 with the atoms held in place, it was −1.02, −10.4
and −58.9 mRy at c = 10, 15 and 20 Å; one CO molecule in a 3.5 × 3.5 × 20 Å cell was
off by −4.25 mRy. At c = 17.5 Å the smallest |G| lands on a grid node, which hid it
there.

**Fix.** Only the short-range part, smooth down to q = 0, is tabulated. The long-range
tail is added analytically at each |G|, and its q-derivative for the stress; G = 0
keeps its α_Z constant. Against pw.x the vacuum dependence is gone: one Ni atom agrees
to 9 μRy in a 10 Å and in a 20 Å cell, and CO to 8 μRy. The slab keeps the separate,
cutoff-dependent wavefunction-grid difference at its 20 Ry; with that grid sized as
pw.x sizes it in a scratch build, the slab agrees to 0.03 mRy at every cell length.
Bulk stress moves too, through the derivative: on the copper DFT+U stress deck the
local term was 0.65 kbar off ESPRESSO 7.6 and is now 0.12 kbar off. That deck's total
pressure used to agree with the stored ESPRESSO 7.5 reference only because the local
error cancelled a nonlocal one, and ESPRESSO 7.6 itself prints a total 0.76 kbar above
7.5's, so `test_qe_hubbard_stress.py` now budgets 6e-6 of the largest contribution
rather than 2e-6.
Unit tests pin the table against the closed-form transform of −Ze² erf(r/r_c)/r at
small q, its derivative, the α_Z constant and the reciprocal-table branch; all but the
α_Z test fail against the old table. A validation test requires one Si atom's energy to
agree in a 20 Å and a 30 Å cell to 10 μRy (0.44 μRy measured; 511 μRy before).

### A QE-format cell card with no units built a cell 1.89 times too small (resolved 2026-10-06)

**Affected:** every QE-format input whose `CELL_PARAMETERS` card names no units while
`&SYSTEM` gives `celldm(1)` or `A`. ase-espresso writes every deck that way — `celldm(1)` set
to one angstrom in bohr, the rows in angstrom, no option — and a slab deck it wrote, sent by
a user, ran it.

pw.x reads a card with no option as `alat` when a lattice parameter is given and as `bohr`
only when none is, both spellings deprecated. The reader took `bohr` regardless, so the
reported 3.5 × 3.5 × 10 Å box became 3.5 × 3.5 × 10 bohr: every axis 1.89 times too short,
the volume 6.75 times too small, and a C–O pair 0.21 Å apart. The Ewald energy came out
251.09 Ry where pw.x has 132.87 Ry, the total energy −50.47 Ry against pw.x's −237.53 Ry,
and the deck's 3 μB moment vanished. The QE derivation audit had recorded the divergence
(README C5, u5 §4.8 and §4.9-1) with a code comment as its only fix.

**Fix.** A card that names no units is resolved once the whole file is read: `alat` when
`celldm(1)` or `A` is given, `bohr` when neither is, with a note in the log; an option that
is none of `alat`, `bohr` and `angstrom` is refused by name. Checked black-box against
pw.x 7.6, which builds a1 = 6.614041 bohr from the reported cell under either `celldm(1)` or
`A`, and 3.5 bohr with neither. Unit tests pin both readings, the written spellings and the
refusal; the alat reading and the refusal fail against the old reader. Run end to end, the
reported deck's Ewald energy now matches pw.x's to 1e-11 Ry, and its total energy comes
within 10.6 mRy of pw.x's — the same residual the deck gives once rewritten with
`CELL_PARAMETERS {alat}`.

### The one-center channel had no first-order SCF correction (resolved 2026-10-04)

**Affected:** every PAW run's total energy at the default
`convergence_threshold`; worst where the on-site D is large (fcc Cu). Forces,
stress and eigenvalues were never affected -- they are evaluated once from the
final state.

The grid part of the total energy was already pulled back from the mixed density
to the output density by `delta_escf`, making its error second order. The
one-center channel had no equivalent: `compute_double_counting_correction`
removed `sum rho_ij D^oc_ij` at the _mixed_ `rho_ij`, while the band energy had
been built from D at the _input_ `rho_ij`, so any residual `rho_ij` error landed
in the total energy at first order. The double counting now removes the band
term at the output `rho_ij` and adds the first-order pull-back
`sum (rho_ij^out - rho_ij^mix) D^oc_ij(rho^mix)`, the one-center counterpart of
`delta_escf` (QE: the PAW term of `delta_escf` in `electrons.f90`).

Measured on fcc Cu, `paw/Cu.upf`, 8x8x8 shifted, gaussian 0.01 Ry, 80 Ry, as the
maximum residual about a smooth fit through an isotropic strain series:

| convergence_threshold           | energy noise   | pressure noise | mean SCF iterations |
| ------------------------------- | -------------- | -------------- | ------------------- |
| `1e-8`, before                  | 44 meV/atom    | 2.2 kbar       | 31.0                |
| `1e-8` + occupancy gate, before | 2.2 meV/atom   | 0.5 kbar       | 36.9                |
| `1e-12`, before                 | 0.18 meV/atom  | 0.6 kbar       | 45.6                |
| `1e-8`, after                   | 0.011 meV/atom | 0.7 kbar       | 37.3                |

On the parity cases at the default threshold the error against the tightly
converged energy fell from 3.4e-7 to 1.1e-10 Ry (Si PAW) and from 3.1e-7 to
1.2e-10 Ry (diamond PAW); only `total_energy_ry` and
`double_counting_correction_ry` moved. PAW no longer needs
`convergence_threshold: 1e-12` for energy-as-product work.

### The Gaunt table was built for a degree nothing could evaluate (resolved 2026-09-21)

**Affected:** every ultrasoft and PAW run. Nothing observable moved by more than
1e-13 Ry, for the reason below, but every such run built a table roughly nine
times larger than its consumers index, with the surplus wrong.

`generate_gaunt_coefficients()` expands products of _projector_ harmonics, so
its argument is the projector degree and its collocation evaluates harmonics up
to twice that. What the ultrasoft basis passed instead was
`augmentation_q_table_.maximum_angular_momentum_q()` — the maximum L of the Q
expansion, which is already `2 * l_max` and so 4 for any d-channel dataset. The
function doubled it again, to 8, and asked `compute_real_spherical_harmonic()`
for degrees 5 through 8. That function had closed forms only to degree 4 and
answered every one of them with `0.0`.

-   **What that cost.** The collocation matrix was singular in exactly the rows
    the missing harmonics should have filled, and the inversion skipped those
    pivots rather than failing. The sparsity pattern the consumers iterate over
    then listed 5403 entries against the 708 that are genuinely non-zero, so
    every consumer summed terms that are not there. No exception was thrown and
    nothing was logged.
-   **Why no result was visibly wrong.** Consumers index product degrees only up
    to `lmaxq = 4`, and every entry at or below that sum was correct. The
    damaged entries had no consumer — which is also why no result-level test
    could ever have caught this, and why the test that does is an assertion
    about the table itself.
-   **Why a bare refusal was not available.** Adding one to
    `compute_real_spherical_harmonic()` first, as this entry used to recommend,
    rejects every ultrasoft and PAW run: the argument mismatch guarantees a
    request at degree 8. The call site had to be corrected before the refusal
    could exist.

**Resolved by** passing the projector degree, checked against the product degree
its consumers will index (`lmaxq <= 2 * l_max`, which holds for every dataset Q3
reads and is now verified rather than assumed). With that in place both
refusals are in force: `compute_real_spherical_harmonic()` throws above degree
4, and `generate_gaunt_coefficients()` throws when `2 * max_degree` would exceed
it — naming the projector degree, since that is the bound a reader needs.

**What moved.** Dropping the spurious sparsity entries changes which terms are
summed and in what order, so ultrasoft and PAW numbers shift in their last bits:
total energies by at most 2.6e-13 Ry, forces by at most 2.9e-18 Ry/bohr, stress
by at most 1.1e-8 kbar, across the parity matrix. Norm-conserving results are
bit-identical, as they must be — they build no Gaunt table.

**Still open:** Q3 supports projector channels up to d. An f-electron dataset is
now refused with a message that says so, instead of running on a silently wrong
table; lifting the limit needs harmonics at degrees 5 and 6, which
[TODOs.md](TODOs.md) carries.

### The DFT+U occupation matrix was rotated with the wrong end of the atom map (resolved 2026-09-16)

**Affected:** DFT+U on a symmetry-reduced mesh, and only where some Hubbard
atom's orbit carries a permutation that is not its own inverse — three or more
equivalent atoms _cycled_ by a three- or four-fold operation. A cell with one or
two Hubbard atoms is exempt whatever its symmetry, because the identity and a
single transposition are both their own inverses; so is any cell with one
Hubbard atom per species. Every DFT+U case Q3 is gated on is in one of those
exempt classes, which is why nothing caught it, and why no shipped number moved.

`symmetrize_hubbard_occupation_matrices` averaged `D(R) · n_{atom_map[a]} ·
D(R)ᵀ`. `atom_map[a]` is the atom `a` is carried _to_, and the relation the
crystal imposes is `n_{atom_map[a]} = D(R) n_a D(R)ᵀ`, so the average that
reproduces an already-symmetric `n` pairs that image with `D(R)ᵀ · n · D(R)` —
the pairing `SymmetryAnalyzer::symmetrize_vector` spells out for forces and says
is the only one an already-symmetric field survives. Reindexing the sum by
R → R⁻¹ turns one into the other with `atom_map` replaced by its inverse, which
is why the two agree exactly on every cell in the exempt classes above.

Found while writing the same group average for the PAW on-site occupations,
where the pairing had to be settled one way or the other.

**Fix.** The contraction is transposed: `D(R)ᵀ · n_{atom_map[a]} · D(R)`, the
same pairing `paw_one_center::OnSiteOccupationSymmetrizer` uses for ρ_ij.

Measured on a cubic perovskite — U on the three oxygen p manifolds, which the
[111] three-fold cycles while each oxygen keeps a four-fold axis of its own, so
its p manifold stays anisotropic:

| quantity                               | before | after   |
| -------------------------------------- | ------ | ------- |
| `max \|n_{R(a)} − D(R) n_a D(R)ᵀ\|`    | 0.272  | 7.8e-16 |
| movement on a second average (P² vs P) | 0.114  | 2.8e-16 |

The second row is the sharper statement: a group average is a projector, and the
shipped contraction was not one.

**What did not move.** The whole DFT+U validation set was run on both binaries
and every printed number is identical digit for digit, as the exempt-class
argument requires — 29 passed, 1 skipped (a GPU binary) either side, across
`test_qe_hubbard_stress.py`, `test_augmented_ortho_atomic_hubbard.py`,
`test_ortho_atomic_hubbard_stress.py`, `test_gamma_only_hubbard.py` and
`test_hubbard_backend_contracts.py`. Unchanged on both: SiC norm-conserving
E_U = 0.274923599 Ry and ultrasoft 0.309694892 Ry, their Γ-versus-full-sphere
energies, and the analytic-force-to-dE/dx ratios 0.999974 and 1.000096.

Gated by four cases in
`tests/unit/layer_4_computation/effective_potential/internal/hubbard/test_hubbard.cpp`
— the crystal's covariance relation and idempotence on the perovskite, both of
which fail against the old contraction with exactly the numbers above; the trace,
which is the invariant either pairing preserves; and the same covariance on a
two-atom cell, which states the exempt class rather than leaving it assumed.

### The PAW on-site occupations were not symmetrized (resolved 2026-09-16)

**Affected:** every PAW run on a symmetry-reduced k-mesh. Levels that symmetry
makes degenerate came out split, and the total energy sat above the full mesh's.
Norm-conserving and ultrasoft runs were unaffected.

A symmetry-reduced mesh samples one k-point per star, and a quantity accumulated
over it stands for the whole zone only once it is symmetrized over the crystal's
operations. The grid density was. The on-site occupations ρ_ij that PAW's
one-centre Hartree and exchange-correlation terms are built from were not: each
projector channel carries an angular index, every operation mixes those indices
and permutes equivalent atoms, and nothing downstream repaired the result
because ρ_ij never passes through a grid. The ultrasoft augmentation charge
built from the same array does reach the grid, where the density symmetrizer
corrects it — which is why ultrasoft and norm-conserving runs were clean and
made the diagnosis. Quantum ESPRESSO applies the same average to its `becsum`.

**Fix.** `OnSiteOccupationSymmetrizer` replaces ρ_ij by the point-group average

    ρ^a  ←  (1/N_R) Σ_R  D(R)ᵀ · ρ^{R(a)} · D(R)

with D(R) the real-harmonic rotation of each channel's l, built by the same
angular quadrature the DFT+U occupation matrices already use, and R(a) the atom
`atom_map` carries a to. Radial indices are untouched, so the average acts
inside each (βᵢ, βⱼ) block on its (2lᵢ+1)×(2lⱼ+1) angular indices. It runs
in `accumulate_projector_overlaps()`, before the damping step, so that the array
compared against the previous iteration — and the one `occupancy_residual_`
measures — is the one the one-centre term is built from; and it runs on the
single becsum the augmentation charge shares, rather than on a private copy, so
the grid and the on-site terms stay functionals of one state. It stands down
exactly where the density symmetrizer does (`control.disable_symmetry`, a
Γ-only basis, a cell with only the identity), asked of the symmetrizer itself
rather than re-derived.

Time reversal needs no separate step, unlike the DFT+U occupation matrix: becsum
is accumulated as `Re[conj(⟨β_i|ψ⟩)⟨β_j|ψ⟩]`, which is already the average over
the ±k pair the wedge folded together.

One SCF each at 30 Ry (240 Ry density for PAW and ultrasoft, 120 for
norm-conserving), 2×2×2 Γ-centred, reduced to 3 k-points against the full 8,
`convergence_threshold: 1e-12`, before and after on the same binary pair:

| dataset                 | Γ valence maximum, reduced (eV) | full mesh (eV) | total energy, reduced − full |
| ----------------------- | ------------------------------- | -------------- | ---------------------------- |
| diamond PAW, before     | 13.7274, 13.7427, 13.7465       | 13.7393 (×3)   | +1.86e-4 Ry                  |
| diamond PAW, after      | 13.73920, 13.73930, 13.73930    | 13.7393 (×3)   | −7.3e-7 Ry                   |
| silicon PAW, before     | 6.3271, 6.3351, 6.3377          | 6.3339 (×3)    | +2.46e-4 Ry                  |
| silicon PAW, after      | 6.33390 (×3)                    | 6.3339 (×3)    | −2e-11 Ry                    |
| silicon ultrasoft       | 6.32380 (×3), unchanged         | 6.3238 (×3)    | −8.8e-8 Ry, unchanged        |
| silicon norm-conserving | 6.31040 (×3), unchanged         | 6.3104 (×3)    | −2.7e-9 Ry, unchanged        |

What is left on diamond is the 30 Ry cutoff rather than the fix. At the
example's own 60/480 Ry the wedge sat +1.87e-4 Ry above the full mesh with a
19.2 meV Γ split (13.71556 / 13.73093 / 13.73480); it now agrees to **2.6e-10
Ry** with the triplet at 13.72753 three times.

Copper PAW, 40/320 Ry, 4×4×4 Γ-centred, 8 irreducible k-points against the full
64 — the case with occupied d channels and the widest splitting:

| level               | reduced, before (eV)            | reduced, after (eV) | full mesh (eV) |
| ------------------- | ------------------------------- | ------------------- | -------------- |
| 3s                  | −92.88774                       | −92.91562           | −92.91562      |
| 3p triplet          | −50.30185, −50.28696, −50.02492 | −50.23189 (×3)      | −50.23189 (×3) |
| bands 6–8           | 16.62830, 16.73008, 16.80480    | 16.69901 (×3)       | 16.69901 (×3)  |
| total energy − full | +1.19e-4 Ry                     | +7.2e-7 Ry          | —              |

The 3p split was 0.277 eV and is gone; the 3s, which is not degenerate and so
showed nothing, was 28 meV off the full mesh and now matches it to the printed
digit.

Quantum ESPRESSO applies the same average to `becsum`, so the shipped
comparison against it moved the right way on all three PAW cases, without their
tolerances being touched (`test_qe_energy_comparison.py`, |Δ| in mRy/atom
against the LibXC build):

| case                   | before | after      |
| ---------------------- | ------ | ---------- |
| Si PAW, 2×2×2 shifted  | 0.3410 | **0.2161** |
| Cu PAW, 4×4×4 shifted  | 0.4723 | **0.4577** |
| C PAW, 4×4×4 Γ-centred | 0.1219 | **0.0541** |

**The pairing is the one thing here that is a choice**, and it is invisible on
every cell above: `atom_map` paired with `D(R)` instead of `D(R)ᵀ` amounts to
replacing each operation by its square, which changes nothing when every atom
permutation is its own inverse — always true of a two-atom cell. It takes an
orbit of three or more equivalent atoms carrying a cycle to see, which is what
the cubic-perovskite cases in
`tests/unit/layer_4_computation/paw/one_center/test_occupation_symmetrization.cpp`
are for: they measure a covariance violation of 0.23 and a loss of idempotence
against the other pairing. The DFT+U occupation matrix had shipped the pairing
those cases reject, and is corrected alongside this — see the entry above.

Gated by `tests/validation/test_paw_irreducible_wedge_consistency.py` (wedge
against full mesh at 5e-6 Ry, and diamond's Γ triplet at 1 meV, with the two
silicon controls in the same parametrization) and by the nine unit tests above.

### Fixed atoms in a QE-format deck were relaxed anyway (resolved 2026-09-14)

**Affected:** QE-format inputs whose `ATOMIC_POSITIONS` rows carry pw.x's `if_pos` flags —
most consequentially `relax` and `vc-relax` decks that hold some coordinates fixed.

pw.x reads three optional integers after each row's coordinates, and a 0 holds that
coordinate fixed while the ions move. The reader stopped after the coordinates and dropped
whatever followed, and Q3 has no fixed-atom constraint to carry the flags onto, so every atom
relaxed, including the ones the deck held in place.

**Fix.** Three 1s ask for exactly what Q3 does and are accepted. A 0 is refused with a message
naming the row, whatever the calculation type; so is any trailing
text that is not three 0/1 flags. Unit tests cover fixed flags under `relax` and `scf`, three
1s, malformed trailing text, and plain and commented rows; the two refusal tests fail against
the old reader.

**Superseded 2026-10-19.** Relaxation is now driven by `q3.workflows`. The engine keeps the
flags on each atom and writes them to the results' `material` as ESSE `basis.constraints`,
and the workflow holds those coordinates fixed (ASE `FixCartesian`), so a 0 is accepted
instead of refused. Measured: a silicon relax deck with one atom at `if_pos 0 0 0` converges
with that atom unmoved and its force zero.

### QE-format positions in angstrom, bohr or alat built a different structure (resolved 2026-09-14)

**Affected:** every QE-format input whose `ATOMIC_POSITIONS` card says anything other
than `crystal` — `angstrom`, `bohr`, `alat`, or no option at all. Reported as
camsai/q3#18.

The reader matched the card header, threw its option away, and stored every row as
crystal coordinates. pw.x reads `angstrom`, `bohr` and `alat` rows as Cartesian
positions, and a card with no option as `alat`, so each of those spellings ran a
different structure from the one the deck describes, without a word. In the reported
deck, BPO4 given in angstrom, the misread cell put two O-O pairs 1.08 Å apart: the
Ewald energy came out −79.87 Ry where pw.x has −93.82 Ry, and the total energy missed
pw.x's by 2.4 Ry. `crystal_sg` was read as `crystal` too, although it lists only the
symmetry-inequivalent atoms.

**Fix.** The reader takes the option from the header and, once the whole file is
read, converts Cartesian rows to crystal coordinates through the cell the Crystal is
built from — the `CELL_PARAMETERS` vectors, or the `ibrav` cell — with alat equal to
celldm(1), or |a1| for a `CELL_PARAMETERS` card in bohr or angstrom. A card with no
option is read as `alat`, with a note in the log; `crystal_sg` and unrecognized
options are refused by name. Unit tests check each spelling against the crystal
coordinates pw.x 7.6 prints for the reported deck, and cover an `ibrav = 4` cell, both
refusals and a deck with no cell; all fail against the old reader. Run end to end with
stand-in pseudopotentials of the same valence, the reported deck's Ewald energy is now
−93.817963626 Ry, and a graphite deck written four ways gives the same total energy to
every printed digit.

### A PAW non-self-consistent run diagonalized without the one-centre term (resolved 2026-09-14)

**Affected:** every non-self-consistent run on a PAW dataset — band structures,
projected densities of states, and anything else built on an NSCF pass.
Norm-conserving and ultrasoft datasets carry no one-centre term.

A PAW D_ij carries a one-centre Hartree and exchange-correlation correction
built from the on-site occupations ρ_ij. The self-consistent iterator fills it
each time it constructs a density, so from its second iteration on every
Hamiltonian includes it. A non-self-consistent run loads V_eff and the density
from `effective-potential.q3cp`, which held no ρ_ij, and diagonalizes before it
constructs any density: the correction was still the empty one the one-centre
engine starts with, and D had no one-centre term at all. On the shipped copper
PAW example (40 Ry, 4×4×4 Γ-centred), SCF against NSCF at Γ on one checkpoint:
3s −92.89 → −196.41 eV, 3p −50.3 → −156.93 eV, the 3d manifold 16.6–17.6 →
−89.7 eV, 4s 10.37 → 4.51 eV. Diamond's lowest band moved 0.9 eV. No test
compared a PAW pass with the SCF it continues.

**Fix.** Checkpoint version 3 stores ρ_ij per spin channel after V_eff. The PAW
iterator restores them as soon as its one-centre engine is built, before the
first Hamiltonian, and keeps the term built from them for the whole pass: the
occupations the pass accumulates come from its own k-points and occupations,
and would otherwise rebuild D for the first-iteration re-diagonalization. The
NSCF flavor refuses a PAW run on a checkpoint without them, which is how a
version-2 file reads, so a PAW checkpoint from before this fix has to be
regenerated by rerunning its SCF. On copper the pass now reproduces the SCF at
Γ in every band but the highest, the one a single pass converges least well.
An integration test on diamond holds bands 1–7 to 1e-6 Ry against a measured
3.4e-7, with the SCF converged to 1e-12, and fails without the fix (0.067 Ry); a
unit test covers the version-3 round trip and a version-2 read.

### A non-self-consistent run could miss eigenvalues at the default Davidson basis (resolved 2026-09-14)

**Affected:** every non-self-consistent run -- band structures, densities of
states, projections, Wannier captures -- from the default Davidson basis
expansion factor of 2 (#211) until this fix: releases `2026.9.14-1` and
`2026.9.14-2`. `2026.9.14-0` and earlier are not affected.

A single NSCF pass seeds each k-point from its neighbour's vectors and keeps
what the solver returns. With a working basis only twice the band count,
Davidson can settle on eigenpairs that skip a state the seed lacks while every
residual test passes. Measured on copper's 4×4×4 norm-conserving example with
sixteen bands, against direct diagonalization of the same Hamiltonian:

| Davidson basis     | worst error, bands 1–14 | band 15 | band 16  |
| ------------------ | ----------------------- | ------- | -------- |
| 2 × the band count | 40.8 meV                | 258 meV | 3.8 eV   |
| 4 × the band count | 0.85 meV                | 1.4 meV | 68.8 meV |

At one k-point every band from 6 to 14 was off by 6–41 meV: one eigenvalue
missing, the bands above it each shifted down an index. The same pass also ran
into the solver's cap of 20 iterations with up to seven of fourteen bands
unconverged at 48 of 140 k-point solves, and said nothing. A self-consistent
run re-diagonalizes every iteration and was unaffected: copper's Wannier Ω_I is
the same to 1e-6 bohr² whichever factor its SCF used, and moves by 4e-3 with
the NSCF's. Found by #175's Wannier tier-2 chains, whose copper Ω_I and gallium
arsenide band errors left their bounds; nothing else checks NSCF eigenvectors
that closely.

**Fix.** A non-self-consistent run uses four times the band count unless the
deck sets the factor -- a deck that sets less keeps it and is warned -- allows
each k-point 200 Davidson iterations, and warns for any k-point whose solve
still ends with bands unconverged. NSCF results computed with the affected
releases, or with `main` after 2026-09-13, should be rerun. Gated by
`SCFIntegration.DavidsonCountsTheBandsACappedSolveLeavesUnconverged`, the
`NonSelfConsistentFieldDavidsonBasis` unit tests, and #175's Wannier tier-2
chains.

### The starting wavefunctions were chosen with the wrong local potential (resolved 2026-09-14)

**Affected:** every run whose datasets carry more atomic orbitals than there are
bands, serial included, since 2026-07-02 — the ultrasoft and PAW silicon examples
among them. Norm-conserving silicon, with exactly as many orbitals as bands, was
not. Separately, runs padding the starting set with random vectors started from a
different guess on every topology.

The starting guess keeps the lowest Ritz vectors of H on the atomic orbitals. When
the grids were first distributed, the conversion of the local potential to the
wavefunction grid, and its 1/N scaling, moved out of the routine that applies that
H, and the caller went on passing the density-grid potential unchanged: read on
the wrong grid, N times too large, and differently on each topology. The subspace
H came out around −2×10⁴ Ry on the diagonal. With as many orbitals as bands
Rayleigh-Ritz keeps the whole span whatever H is, which is why norm-conserving
silicon never showed it; with more, the bands kept were chosen by that H. The SCF
converged to the right answer from a poorer and topology-dependent start: after
one iteration, two ranks in one pool were 3.3e-4 Ry (ultrasoft) and 1.2e-4 Ry
(PAW) from serial, and PAW silicon at Γ took 48, 64 and 79 iterations on one, two
and four ranks.

The random padding had its own dependence on layout: its coefficients came from a
generator stepped along the rank's plane-wave list, so on silicon with sixteen
bands two ranks in one pool started from different vectors than one.

**Fix.** The iterator converts the potential exactly as the Hamiltonian
application does before building the guess. The random coefficients are drawn
from a hash of the k-point's coordinates, the vector and the plane wave's Miller
indices. On 1, 2 and 4 ranks in one pool the ultrasoft, PAW and norm-conserving
silicon examples now take identical SCF paths (9, 37 and 7 iterations), and the
first Davidson step of the ultrasoft and PAW runs starts 10 and 20 times closer to
its answer. A validation test compares the energy after one SCF iteration, serial
against two ranks in one pool, for the ultrasoft and PAW examples; a unit test
checks that the padding follows each plane wave when the list is reversed. Both
fail without the fix.

### Local Thomas-Fermi preconditioning hung when ranks shared a pool (resolved 2026-09-14)

**Affected:** `mixing_mode = 'local-TF'` on more than one rank per pool, from the
stick-distributed FFT of #211 until this fix. The option is reachable only from a
QE-format input; the YAML schema does not offer it.

The preconditioner solves [G² + k_TF²(r)] x = G² F by conjugate gradients, with
the cell average of k_TF² as the diagonal of its inner preconditioner. The
average was taken over the rank's own real-space points. Ranks sharing a pool
hold different planes, so each rank solved with a different diagonal, took
different steps, and left the loop at a different iteration, while every step
runs the same collective transforms. The silicon reference input on two ranks in
one pool hung before its first SCF iteration finished; serial it converged in 11.

**Fix.** The screening sum and the point count are completed within the pool
before the average is taken, so every rank solves the same problem. A rank
holding no real-space planes now reports the same average and runs the same
solve, rather than taking the uniform-screening shortcut the others skip. A unit
test simulates ranks holding halves of an inhomogeneous density, and one holding
none; a validation test runs the silicon input serial and on two ranks in one
pool, where a hang fails through the subprocess timeout.

### Isolated-system corrections placed grid points wrongly when ranks shared a pool (resolved 2026-09-14)

**Affected:** `isolated_system_correction: makov-payne` and `martyna-tuckerman` on
more than one rank per pool, from the stick-distributed FFT of #211 until this
fix.

Both corrections need the position of each real-space grid point — Makov-Payne
for the electronic dipole and quadrupole, Martyna-Tuckerman for its kernel — and
took it by adding `local_offset()` planes to the local index, which is the layout
of a slab along the first dimension. Ranks sharing a pool hold whole planes along
the third dimension instead, so every point was placed at a permutation of its
own coordinates. In a cubic cell with the density centred that permutation
changes nothing, which is why the charged silicon example agreed. A silicon atom
off the centre of its box, at 2 ranks in one pool: the Makov-Payne correction came
out 1.8524767 Ry against 1.7035825 Ry serial, and in a box with unequal sides the
Martyna-Tuckerman correction came out −13.5240469 Ry against 1.4986444 Ry, with a
warning that the density was not localized.

**Fix.** `FFTGrid::global_real_space_index()` names the serial-layout index of a
local real-space point; the stick grid overrides it for its plane layout, and both
corrections use it. A unit test reads the distributed transform through it and
compares with the serial transform on 1 to 4 ranks; it fails without the override.
A validation test runs both corrections on the off-centre atom, serial against two
ranks in one pool, and fails without the fix.

### Equal wavefunction and density grids dropped plane waves from the density when ranks shared a pool (resolved 2026-09-14)

**Affected:** runs on more than one rank per pool whose wavefunction and density
FFT grids have the same dimensions, from the stick-distributed FFT of #211 until
this fix. The wavefunction grid is sized for twice the wavefunction cutoff and
capped at the density grid, so this means a density cutoff at or below about
twice the wavefunction cutoff. The examples use four times, and never took this
path.

With the grids the same size, the band density was accumulated straight onto the
density grid instead of on the wavefunction grid and transferred through
reciprocal space. That scatter is only right when each rank owns the same sticks
on both grids, and ranks sharing a pool split the two grids independently, each
by its own G-vector count per stick. A plane wave whose stick the density grid
gave to another rank was dropped from the density. Silicon at 48/96 Ry on two
ranks in one pool gave −17.702695341 Ry against −16.913417902 Ry serial at
2×2×2 k-points, and −17.209278012 Ry against −15.687763651 Ry at Γ.

**Fix.** The density is accumulated on the wavefunction grid whenever the density
grid is split within the pool, as it already was whenever the grids differ in
size; the transfer through reciprocal space is right for any ownership. Serial,
pool and band-distribution layouts keep the direct path. A validation test runs
the silicon example with the density cutoff at twice the wavefunction cutoff, at
k-points and at Γ, serial against two ranks in one pool; both cases fail without
the fix.

### Ultrasoft and PAW at Gamma gave the wrong energy when ranks shared a pool (resolved 2026-09-14)

**Affected:** ultrasoft and PAW runs at Γ alone (a 1×1×1 unshifted grid) on more
than one rank per pool, from the stick-distributed FFT of #211, which first kept
the half sphere under a shared pool, until this fix. Norm-conserving runs, k-point
pools and band distribution were not affected.

At Γ the stored half sphere is summed as 2 Re, which counts G = 0 twice, and the
rank whose plane-wave list begins with G = 0 removes one copy. With reciprocal
space split by stick that is one rank. The projector overlaps that build the
augmentation charge asked the density G-vector list instead, which is complete
on every rank, so every rank answered yes and subtracted the product of its first
stored plane wave. On the two-atom silicon examples at Γ, two ranks in one pool
gave −14.507838530 Ry against −14.508710182 Ry serial (ultrasoft) and
−92.213130311 Ry against −92.211841450 Ry (PAW). The Γ shared-pool test ran the
norm-conserving example only, which never reaches this code.

**Fix.** The correction asks `gamma_origin_is_local()`, as the Hamiltonian
application and the ultrasoft overlap already did. The Γ shared-pool test now
runs the ultrasoft example on two and four ranks and the PAW example on two,
converged to 1e-12 so that the 1e-8 Ry comparison sees the fixed point (PAW at
the examples' 1e-8 stops 4e-7 Ry short of it), and the ultrasoft and PAW cases
fail without the fix. The same runs also warned that the total charge was about
half the electron count: the check summed only the rank's own real-space planes,
and now completes the sum within the pool first.

### A system LibXC 7.1 sent spin-polarized PBE to NaN (resolved 2026-09-14)

**Affected:** builds that found LibXC 7.1.0, 7.1.1 or 7.1.2 installed, which the
configure step took ahead of the pinned 7.0.0 source build whatever its version.
Homebrew's `libxc` was 7.1.2 when this was found, so a macOS machine with it
installed got it without asking. Spin-polarized GGA runs whose spin gradients
cancel somewhere: the chromium antiferromagnet went NaN at its first SCF step,
reported as a failed subspace eigensolve.

Between the opposite moments of an antiferromagnet the two spin densities are
equal and their gradients opposite, so σ↑↑ = σ↓↓ = −σ↑↓: the total gradient
σ↑↑ + 2σ↑↓ + σ↓↓ vanishes while neither spin gradient does. PBE correlation
depends on the total gradient only and is smooth there. From 7.1.0 through 7.1.2,
LibXC's spin-polarized PBE correlation returns NaN for its σ derivatives at such
a point, while its energy and density derivatives agree with 7.0.0 to every digit;
7.0.0 returns finite derivatives for the same input. Q3's inputs are valid — both
series screen σ↑↓ identically before evaluating — so the fault is the LibXC
version. Q3's share was letting the NaN through: the safety clamps compare
magnitudes, which a NaN never exceeds.

**Fix.** A system LibXC is now used only from a series Q3 is tested against: 5.2,
which Ubuntu 24.04 ships and every CI job and container image builds with, and
7.0, the pinned source build. Any other version is reported at configure time
and 7.0.0 is built from source instead; `-DQ3_ALLOW_UNTESTED_SYSTEM_LIBXC=ON`
takes it anyway. Every LibXC evaluation now refuses a non-finite value at a point
Q3 is about to use, naming the LibXC version, the functional, the grid point and
its inputs, so a forced LibXC 7.1 stops at the first evaluation rather than a
step later. A unit test evaluates spin-polarized PBE at the grid point captured
from the chromium run against LibXC 7.0.0's values; it passes on 5.2.3 and 7.0.0
and fails on 7.1.2.

### GPU Davidson converged to the wrong energy whenever its basis collapsed (resolved 2026-09-14)

**Affected:** every GPU backend, norm-conserving runs, from the identity-overlap
memory reduction of #204 until this fix; visible by default once the Davidson
basis expansion factor became 2.

For a norm-conserving pseudopotential the overlap operator is the identity, and
the CPU workspace stops storing S|V> separately: it hands back the basis array
itself. The subspace collapse relied on that and skipped rotating S|V> whenever
the overlap was the identity. The device workspace still keeps S|V> in a buffer
of its own, so on the GPU that buffer came out of every collapse unrotated, and
the orthogonalisation and subspace update that followed read stale overlap
vectors. On the two-atom silicon deck at expansion factor 2, which collapses on
every iteration, the GPU converged to −18.678851372 Ry against −16.913417930 Ry
on the CPU; at factor 4 small decks usually converge before the basis fills, so
the GPU energy parity suite kept passing.

**Fix.** The collapse skips the overlap rotation only when the overlap array is
the basis array, in both subspace managers. The CPU path is unchanged, since
there the two are the same array; the GPU now matches the CPU to every printed
digit at expansion factors 2, 3 and 4 and at Γ. A unit test holds an identity
overlap in its own buffer and checks it is rotated; it fails without the fix.
No CI job builds the GPU targets, which is how this reached main (TODOs.md §2.3).

### FFT grid factors were read off atom-pair distances, and 0.375 was read as 1/3 (resolved 2026-09-11)

No committed number moved, which is why this stood for so long: a rule with no
grounding that happened to land on the right answer. `compute_fft_factors` asked
for the 2x2x2 silicon supercell's FFT grids to be divisible by 6, where its
symmetry asks for 4, and the 6 came from the 0.375 between its two sublattices.
At the committed cutoffs the dense and smooth grids still came out 72³ and 48³,
exactly twice the primitive cell's 36³ and 24³, so the supercell energy was
untouched and `test_supercell_folding_consistency.py` had nothing to catch. At
other cutoffs the round-up to a {2,3,5}-smooth multiple of an invented factor
need not land on twice the primitive grid, and then two descriptions of one
crystal integrate on real-space grids of different spacing: sweeping 291 cutoffs
from 20 to 600 Ry on these two cells, the old factors broke the doubling at 192
of them — at 28 Ry they put the supercell on 30³ against the primitive cell's
16³ — and the new ones hold it at all 291.

**Cause.** The factors were read from differences between same-species atom
pairs: for each fractional-coordinate difference Δ the code took round(1/|Δ|)
and accepted it whenever it came out 2, 3, 4 or 6, never asking whether |Δ| was
1/n at all, and round(1/0.375) = 3. Both halves of that were wrong. The
denominator was never checked against the difference that produced it, and atom
pairs are not where the constraint lives — two like atoms a fraction of a
lattice vector apart need not be related by any operation of the space group,
and in a supercell most pairs are not.

**Fix.** The factors come from the fractional translations of the symmetry
operations spglib returns, which is what Quantum ESPRESSO's `fft_fact` reads,
and a denominator survives only if |τ| sits within
`fractional_translation_rounding_tolerance` (1e-3) of 1/n — after folding τ into
[−1/2, 1/2], so that spglib's 3/4 is recognized as a quarter rather than as
nothing. Diamond's 48 operations carry halves along the primitive axes, so the
primitive cell asks for 2 (it asked for 4 before, off the 0.25 between its
sublattices); the supercell's 384 carry those same halves as quarters, plus the
primitive lattice's own translations as halves, so it asks for 4. The divisor
now scales with the cell — twice the cell, twice the divisor — which is what
makes the round-up commute with the doubling and the two cells' grid spacings
equal. No committed grid dimension changes, because 36³/24³ and 72³/48³ are
multiples of both the old factors and the new, and every pinned energy, force
and stress in `tests/validation` and `tests/regression` is unchanged.

Gated by nine tests in
`tests/unit/layer_3_representation/basis/planewave/internal/fft/test_grid_sizer.cpp`:
what a single translation component may and may not be read as (0.375 and 0.4
ask for nothing; 1/3 and 5/6 do; 1/5 and 1/8 are denominators `get_good_fft_order`
could not honour anyway), the two silicon cells' factors, that the supercell's
are the primitive cell's doubled, and that their grids are too at the committed
cutoffs.

### The `-mpi` container image aborted whenever two ranks shared a k-point pool (resolved 2026-09-11)

Symptom: inside the published `-mpi` image, two ranks in one pool abort in the
first complex reduction, on any deck and at any rank count:

```
MPI_Allreduce(sendbuf=MPI_IN_PLACE, recvbuf=0x..., count=12339,
              MPI_DATATYPE_NULL, MPI_SUM, comm=0x84000002) failed
```

One rank per pool was unaffected, k-point pools scaled normally, and the same
source built natively ran the same configuration to convergence. Reported from
`podman-hpc --mpi` on Perlmutter and read there as the host-MPI swap failing to
carry a datatype handle across the ABI boundary (camsai/q3#11).

**Cause.** Not the swap: the abort reproduces with the published image under
plain Docker on a laptop, with no Cray MPI involved. Q3 passed
`MPI_DOUBLE_COMPLEX` for `std::complex<double>`, and that is Fortran's
`DOUBLE COMPLEX`. `Dockerfile.mpi` builds MPICH `--disable-fortran`, since a
C++-only image has no other use for Fortran, and an MPI built that way never
initializes the Fortran predefined datatypes -- so the handle arrives at the
collective null. The native build and CI's Open MPI both have Fortran, which is
why neither ever saw it.

**Fix.** `MPI_CXX_DOUBLE_COMPLEX`, which is what the standard defines for
`std::complex<double>` and whose handle is present however the library was
configured, named once in `src/layer_0_core/execution/mpi_datatype.hpp` and used
for all eight complex collectives. (The handle being present is not the same as
`MPI_SUM` accepting it, which is the next entry.) The band-group `Allgatherv` also passed
`MPI_DATATYPE_NULL` as its send type, unused under `MPI_IN_PLACE` but still an
argument an implementation may check; it names the real datatype now.

| 2 ranks, silicon `nc.yml`  | published `2026.9.10-5-mpi` | rebuilt from the fix |
| -------------------------- | --------------------------- | -------------------- |
| `--pools 2` (one per pool) | −16.913417932               | −16.913417932        |
| `--pools 1` (sharing)      | **abort**                   | **−16.913417932**    |

Ranks sharing a pool is still opt-in and still has no measured speedup (see
above) -- but a layout the engine accepts must not abort.

Five releases shipped with it because the only configuration that puts a complex
buffer on the wire is ranks sharing a pool: with one rank per pool every complex
collective returns before it reaches MPI. That is the configuration the
container smoke test excluded by design, so it runs there now, and four unit
tests in `tests/unit/layer_0_core/execution/test_mpi_execution_policy.cpp` cover
the datatype and the collectives that carry it.

### A host MPI swapped into the `-mpi` image refused `MPI_SUM` on the complex datatype (resolved 2026-09-12)

Symptom: the follow-on to the entry above, with the same shape and a different
cause. Same image, same deck, same rank count; the only variable is who provides
MPI:

| how MPI is provided                              | result                      |
| ------------------------------------------------ | --------------------------- |
| the image's own MPICH (`mpiexec` in a container) | converged, −16.925260460 Ry |
| host MPICH injected by `podman-hpc --mpi`        | **abort**                   |

```
MPIR_SUM_check_dtype(83): MPI_Op MPI_SUM operation not defined for this datatype
```

The datatype is no longer null -- it carries a real handle, `0x4c001035` -- and
the image's own library sums it correctly. The injected one does not accept it
for `MPI_SUM` (camsai/q3#17).

**Cause, and what is not the cause.** The reported diagnosis was that building
MPICH `--disable-fortran` shifts the predefined handle numbering, so the fix
would be to build the image's MPICH with Fortran. That was tested and is wrong:
configuring MPICH 4.2.3 both ways yields byte-identical datatype constants, no
value differs, and `MPI_CXX_DOUBLE_COMPLEX` is `0x4c001035` in each. The values
are hardcoded per release rather than derived from what a build enables, and
3.4.3 -- the generation the Cray library derives from -- assigns the same
constant. Rebuilding with Fortran would have cost build time and fixed nothing.

What is established is narrower. MPICH files `MPI_CXX_DOUBLE_COMPLEX` under
`COMPLEX_EXTRA`, commented in its own source as "extra types that are not
required to be supported by the MPI Standard", and gates it on
`HAVE_CXX_COMPLEX` -- a configure result, so whether `MPI_SUM` accepts the type
is a property of how that library was built. Which option the host library was
built with cannot be determined from here, and the fix does not depend on
knowing.

**Fix.** Stop asking any MPI to sum a complex datatype. Complex addition is
componentwise, so reducing N complex numbers as 2N `MPI_DOUBLE` is the same
arithmetic, and `MPI_DOUBLE` is the one datatype observed to survive the
substitution -- it is what the one-rank-per-pool path reduces, which is why that
path never aborted. `sum_complex()` in
`src/layer_0_core/execution/mpi_datatype.hpp` is now the only route for complex
`MPI_SUM`, used by all four reduction sites; `complex_datatype()` remains for
moving data, where no operator is involved.

The equivalence is asserted bitwise rather than approximately, against the
`MPI_CXX_DOUBLE_COMPLEX` reduction on an MPI that accepts it. The failing
configuration itself is not reproducible on any CI runner -- it needs a Cray
system and a host-MPI swap -- so confirmation on Perlmutter is still outstanding.

### A supercell's density was symmetrized with a subgroup of its space group, so it did not fold onto its primitive cell (resolved 2026-09-10)

Symptom: the 2-atom silicon primitive cell on a shifted 4×4×4 mesh and its
2×2×2 supercell on a shifted 2×2×2 mesh sample identical k-points, so
E(16 atoms) must equal 8 × E(2 atoms). Q3 put the supercell **+1.44e-4 Ry**
(9.0e-6 Ry/atom) above that, against 2.4e-7 Ry for QE 7.5 on the same decks
(camsai/q3#6). Ewald folded to 2.9e-7 Ry; every density-dependent term was off
by ~1e-3 Ry and they nearly cancelled — two runs converging to different
densities, not one term computed wrongly. Reduced and full-zone runs were
bit-identical in both cells, Q3's primitive energy matched QE's symmetric
number while its supercell energy matched QE's `nosym` one, and
`control.disable_symmetry` changed the k-point count without moving either.

**Cause.** spglib reports the supercell's whole space group — 384 operations,
the 48 rotations times the 8 translations of the primitive lattice inside the
cell — and the k-point reduction uses all of them. The density symmetrizer did
not: on finding a pure translation it declared the cell a supercell and dropped
every operation carrying a fractional translation, a rule carried over from
Quantum ESPRESSO, which needs it because its own operation search is not
guaranteed to produce a group. spglib's list is a group, so the reason never
applied. What survived were the 12 operations fixing the origin (the
bond-centre site symmetry, −3m), and those happen to map the shifted 2×2×2
mesh onto itself with the same orbits as the full group: averaging the
2-point wedge density over them reproduces the raw 8-point mesh density
exactly. That is why the wedge and full-zone runs agreed to every digit, and
why the answer was QE's `nosym` number. The primitive cell is never flagged,
keeps all 48 operations, and converges to the symmetrized-mesh density — QE's
symmetric number. A shifted Monkhorst-Pack mesh is not invariant under the FCC
point group, so the two densities differ, by 1.8e-5 Ry per 2-atom cell in both
codes. A wedge only stands in for the full mesh when the density is averaged
over the operations that produced it, and the two code paths were reading
different groups.

**Fix.** The symmetrizer averages over every operation spglib returns. Over the
384-element group that average is a projector: the 8 translations zero every
Fourier component the repeated cell cannot carry (any odd Miller index in the
supercell basis) and the rotations act exactly as they do in the primitive
cell, so the supercell's symmetrized density is the primitive cell's and its
energy is eight times the primitive energy. `detect_supercell` and its branch
are deleted; the mesh reduction and the density symmetrization now read the
same operation list.

|                       | 8 × E(2-atom, 4×4×4) | E(16-atom, 2×2×2) | folding error |
| --------------------- | -------------------- | ----------------- | ------------- |
| Q3 before             | −135.400489678       | −135.400345677    | +1.44e-4 Ry   |
| Q3 after              | −135.400489678       | −135.400489400    | +2.8e-7 Ry    |
| QE 7.5 symmetric      | −135.40048768        | −135.40048744     | +2.4e-7 Ry    |
| QE 7.5 `nosym`        | −135.40034400        | −135.40034371     | +2.9e-7 Ry    |
| Q3 after, no symmetry | −135.400345961       | −135.400345674    | +2.9e-7 Ry    |

The primitive cell was never flagged, so no pinned primitive-cell energy
moves. What remains is the Ewald term's own +2.9e-7 Ry, which comes from the
committed supercell coordinates rather than from the density; QE carries the
same residual.

Gated by `test_supercell_energy_folds_onto_primitive_cell`
(`tests/validation/test_supercell_folding_consistency.py`, 1e-6 Ry: 3.6× above
what is left and 140× below the defect) and by three unit tests in
`tests/unit/layer_4_computation/charge_density/planewave/test_init.cpp` —
invariance of the symmetrized density under all 48 / 384 operations,
idempotence, and the 2×2×2 cell folding onto the primitive one component by
component. The folding test and the two supercell unit tests were confirmed to
fail with the fix disabled and the rest of the branch unchanged.

The report's second finding — `control.disable_symmetry` changed the k-point
count but never the energy — was the flag gating only the mesh reduction while
the density was symmetrized regardless, so a full-zone run produced the
symmetrized-mesh density all the same. As of the same date the flag means what
its name says: no mesh reduction, and no symmetrization of the density, forces
or stress (QE's `nosym` with `noinv`). On the shifted 4×4×4 mesh the 2-atom
cell then lands +1.797e-5 Ry above its symmetric energy — the gap QE shows
(+1.796e-5 Ry) — and at the number the supercell used to produce, divided by
eight; on a Γ-centred mesh, which is invariant under the point group, nothing
moves, and the four suites that use the flag for full-zone controls are
unaffected. Gated by `test_disable_symmetry_gives_the_unsymmetrized_mesh` and,
in tier 2, `test_supercell_energy_folds_without_symmetry` in the same module.

### The augmentation force and stress were spin-blind (resolved 2026-08-09)

Their nonlocal partners carried a spin index; these two integrated the
**spin-averaged** potential against the **spin-summed** projector occupancies.
The difference is

    ∫ (V↑ − V↓)(ρ↑ − ρ↓) dQ/dR / 2

which is identically zero without a moment — so every unpolarized test passed —
and non-zero with one. The two halves of the same pseudopotential derivative
therefore disagreed for exactly the magnetic cells.

The cause was plumbing, not missing physics: the iterator had already computed
`total_potential_g_by_spin` and `projector_overlap_coefficients_by_spin`, but
the SCF state handed to layer 5 carried only the averaged and summed forms.
Both are now on the state, and the augmentation force and stress sum over spin
channel against matching channel. An unpolarized run takes the single stored
channel and is bit-identical to before.

Measured on ferromagnetic bcc iron: the spin-blind assembly gave a force
4.1% away from the derivative of Q3's own energy (−0.04543 against −0.04366
Ry/bohr); resolving spin brings it to 0.56% (−0.04390). The finite difference
is identical in both, so the gap was the cross term and nothing else.

Gated by `test_magnetic_augmentation_force_is_the_gradient_of_q3s_own_energy`
in `tests/validation/test_spin_polarized_nonlocal.py` — the analytic force
against a central difference of Q3's own energy on ferromagnetic bcc iron
(ultrasoft, conventional two-atom cell, one atom displaced), with a companion
test asserting the cell actually carries a moment so the gate cannot pass
vacuously.

### Q3 reported no absolute magnetization, so an antiferromagnet could not be checked (resolved 2026-08-08)

Q3 printed ∫(ρ↑−ρ↓) and nothing else. That number is zero for a converged
antiferromagnet **and** zero for one whose moments collapsed to non-magnetic, so
nothing in Q3's own output distinguished the two — on chromium, the case where
it matters most.

Now both of QE's numbers are printed. Chromium's AFM state comes out at
|m| = 2.5929 Bohr mag/cell against QE's 2.60, confirming the state rather than
assuming it, and the drop-in case set asserts the absolute value for exactly
that reason.

### QE mixing keywords were not translated (resolved 2026-08-08)

`mixing_mode = 'plain'` is Quantum ESPRESSO's **default** and appears in most
real QE inputs. Q3 stored the string verbatim and died in the mixer factory
with `Unknown mixer mode: 'plain'` — the same class of failure as
`calculation = 'scf'` before it was mapped: the parser read the file and then
could not run it. `plain` now maps to Broyden. QE's Thomas-Fermi
preconditioners now map onto Q3's own: `TF` -- the average (global) screening --
is Kerker, and is mapped there with a notice saying so; `local-TF` maps to
`mixing.preconditioning: local-tf`, which builds the screening from the local
density. Both are the same preconditioner QE names, not merely the same fixed
point by another route.

It was found by the drop-in case set on its first run, on files that had been
sitting in `examples/` all along.

### The QE input parser ignored what it did not recognize (resolved 2026-08-08)

Any namelist key, namelist or card the parser did not know was dropped without
a word. The sharpest case: a `HUBBARD` card was silently discarded, so a PAW+U
input ran to convergence and reported the **no-U** energy — a converged,
plausible, wrong number that no amount of comparing against QE would attribute
to the parser.

Every key is now classified: honoured, physics-neutral (accepted with a note),
stating a QE default that switches a feature off (accepted silently),
recognized but unimplemented (refused, with the reason), or unknown (refused).
Unknown namelists and unsupported cards are refused too.

Turning the policy on immediately caught Q3 _assuming_ `dftd3_version = 4` and
`dftd3_threebody = .false.` while ignoring what the file said, and `celldm(2..6)`
being dropped — which built every non-cubic Bravais index with c/a = 0, a
collapsed cell. Both fixed here.

96 of 150 of QE's own `scf` regression inputs now parse; every refusal names a
real unimplemented feature. See [docs/io_reference.md](docs/io_reference.md) §5.1.

### DFT+U occupations were not symmetry-adapted (resolved 2026-08-08)

The occupation matrix is accumulated over the irreducible wedge. Weighted IBZ
sums reproduce full-zone sums only for quantities invariant under the point
group, and `n` is not — it carries the manifold's angular indices. Q3 summed the
wedge and stopped, so `n` came out anisotropic where symmetry forbids it; the
_trace_ was right, which is why this looked like a projector-normalization
problem for so long.

Silicon's 3p manifold sits on a tetrahedral site, so symmetry requires n = x·1
and hence Tr[n²] = (Tr n)²/3. QE's numbers satisfied that identity exactly;
Q3's exceeded it by 0.154, and since E_U = (U/2)(Tr n − Tr n²) the surplus went
straight into the energy: E_U = 0.1204 Ry against QE's 0.1918 with Tr[n]
agreeing to 0.5%.

Fixed by averaging over the point group after the k-sum, as QE does. Reading
**QE's own DFT+U input file**, Q3 now gives a total of −15.517022664 Ry against
QE's −15.51702311 (**4.5e-7 Ry**) and E_U 0.191758053 against 0.19175802
(**3e-8 Ry**).

The rotation matrices for the real spherical harmonics are built numerically
from the defining relation on an angular grid exact to the degree involved,
rather than from hard-coded Wigner formulae, so this stays correct for any l a
manifold can carry. Regression coverage:
`SCFIntegration.HubbardOccupationsAreSymmetryAdapted`, which gates the wedge
against the full zone — the invariance whose violation _was_ the bug, and which
needs no reference code.

### DFT+U lowered the total energy instead of raising it (resolved 2026-08-08)

E_U = (U/2)Tr[n(1−n)] is positive semi-definite, so switching U on must raise
the total energy. Q3 lowered it — by ~1 Ry on silicon and ~0.6 Ry on nickel —
on **every** pseudopotential family.

The Hubbard potential is applied inside the Hamiltonian, so every eigenvalue
already contains it; adding E*U on top without removing Tr[V_U n] counts the
on-site interaction twice. Because Tr[V_U n] = U(½Tr n − Tr n²) is negative for
a partially filled shell, the error pushed the total \_down*. The PAW one-center
term has exactly this counterpart and always did
(`one_center_potential_energy()`); DFT+U simply never grew one.

This is almost certainly the "~0.6 Ry unexplained residual" recorded in PR #38
after that PR fixed a k-point weighting bug in the test: the correction moves
the same nickel configuration by **+0.600 Ry**. The regression anchor that PR
left behind was described as QE-cross-checked; it was reproducing the bug.

Verified three ways: U → 0 reproduces the no-U total bit-identically; the total
now rises monotonically with U; and dE/dU → E_U/U to ~1.5% at small U, the
Hellmann-Feynman consistency the broken assembly violated. Regression coverage:
`SCFIntegration.HubbardURaisesTheTotalEnergy`.

### Gamma-only forces were wrong in four terms (resolved 2026-08-08)

At a single unshifted Γ point Q3 stores only the positive half of G-space. A
G-sum over that list is half the full-sphere sum whenever the summand is even
under G → −G, which every force summand is; the **local**, **NLCC**, **Ewald**
and **augmentation** forces each lacked the factor. The **nonlocal** term had a
different bug: `2·Re[⟨β|ψ⟩_half]` double counts G = 0, because Q3's half sphere
includes it, while the derivative overlap needs no correction since its summand
carries G_α. That one touched only l = 0 projectors, which is why it skewed
components unevenly instead of scaling them.

Silicon's Γ force was **0.637** of Q3's own dE/dR with a norm-conserving
dataset and **0.822** with PAW; both are now within 4e-4, and Q3 matches QE
term by term to ~1e-6 Ry/bohr. Energies were correct throughout — only
derivatives were wrong, which a relaxation converges through silently.
Regression coverage:
`tests/validation/test_paw_pressure_and_gamma_forces.py::test_gamma_only_forces_are_the_gradient_of_q3s_own_energy`
on the committed silicon PAW dataset, where restoring the four defects gives 0.716, and
`tests/validation/test_gamma_only_hubbard.py::test_gamma_only_hubbard_force_is_the_gradient_of_q3s_own_energy`,
which holds the half-sphere force to Q3's own energy on ultrasoft and norm-conserving
datasets with DFT+U.

### Spin-polarized ultrasoft and PAW collapsed the moment (resolved 2026-08-08)

The nonlocal D matrix was built once from the spin-averaged effective potential
and shared by both channels, and the compensation charge was split between
channels by the ratio of the _smooth_ densities. Both are exact for a
non-magnetic cell; in a magnetic one they remove the moment, because in a 3d
metal nearly all of it lives inside the augmentation spheres. Norm-conserving
was never affected — its D_ion has no v_eff dependence — which is why every
LSDA validation case passed and said nothing about it.

Ferromagnetic iron with a GBRV ultrasoft dataset, a path that had never been
measured, went from 0.0144 to 2.2482 μB and from 473 to 0.06 meV/atom against QE.
Regression coverage:
`tests/validation/test_spin_polarized_nonlocal.py`.

### SCF convergence decision desynced k-point pools (resolved 2026-07-27)

The SCF convergence residual (`dr2`) was computed independently per rank from
that rank's own `FFTW_MEASURE`-planned FFTs, making it bit-divergent across
pools; ranks near the convergence threshold then decided "stop" vs "continue"
differently, desynchronizing the pools' collective call sequences — observed
as nondeterministic `MPI_ERR_TRUNCATE` crashes and corrupted totals for
LSDA+smearing runs under k-point-pool parallelism. Fixed by `MPI_MAX`-reducing
the residual across pools before any branch consumes it
(`compute_residual_between`,
`src/layer_4_computation/iterator/kohn_sham/planewave/init.hpp`); root-cause
writeup in [TODOs.md](TODOs.md) §2.3, regression coverage in
`tests/validation/test_mpi_topology_invariance.py` (all four smearing
schemes) and `test_mpi_band_parallelism.py`.

### Band parallelism produced wrong energies (PR #20, resolved)

The original `feat/parallelization-2026-7-1` architecture reserved the group
communicator for band parallelism without distributing wavefunction
coefficients, so multi-rank pools double-counted inner products (Si:
-16.5506 Ry vs the correct -16.9134 Ry). Fixed by the G-vector slab
distribution introduced in `feat/parallelization-2026-7-2x`; topology
invariance is validated in `tests/validation/test_mpi_band_parallelism.py`.
