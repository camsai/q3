# Troubleshooting Guide

Known issues, root causes, and resolutions encountered during Q3 development. Written for both humans and AI agents.

---

## Build and environment failures

Every entry below is a failure someone actually hit. For the full list of
configuration variables, see [docs/build_reference.md](docs/build_reference.md).

### `Compiler doesn't support IPO` from spglib or GoogleTest

Configuration stops inside a vendored dependency, not in Q3.

Interprocedural optimization is requested for the whole project, so it reaches
every subproject, and those configure with `CMP0069=NEW`, which turns "this
toolchain cannot do link-time optimization" into a hard error. Some HPC compiler
wrappers fail the probe.

Current versions probe first and skip it. On an older checkout, comment out the
`set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)` line near the top of
`CMakeLists.txt`, delete the build directory and configure again. Passing
`-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF` does **not** work: a plain `set()`
shadows the cache entry.

### `filesystem: No such file or directory`

The compiler predates `<filesystem>`. Q3 needs **GCC 9 or newer**; GCC 8 has the
header but needs `-lstdc++fs`, which Q3 does not link.

Usually this means the build picked up an old system compiler rather than the
one you loaded. **The compiler is cached, so re-running `cmake` in the same
build directory changes nothing** — that is the trap. Check what is actually in
use, then start a fresh directory:

```bash
grep -E "^CMAKE_CXX_COMPILER:" build/CMakeCache.txt
```

```bash
rm -rf build && CC=cc CXX=CC cmake -S . -B build
```

Editing the sources to `<experimental/filesystem>` does not work either: the
qualified names throughout are `std::filesystem::`, so the includes and the call
sites would both have to change.

### `ModuleNotFoundError: No module named 'jsonref'`

The build generates C++ from JSON schemas, and that script needs `jinja2`,
`jsonref` and `pyyaml`.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[build]"
```

A `.venv` at the repository root is picked up automatically. If you meant to use
a different interpreter, pass `-DPython3_EXECUTABLE=...`; configuration reports
which one it chose. Current versions warn and fall back to the committed headers
instead of failing.

### `FFTW3 not found`, with advice to run `brew install fftw`

On a module-based system the message is unhelpful because FFTW is installed, just
not where the search looks.

```bash
module load cray-fftw   # exports FFTW_DIR / FFTW_INC, which discovery reads
```

If discovery still misses it, name the paths. Look first rather than guessing —
the library may be `.a` and the directory may be `lib64`:

```bash
env | grep -i fftw && find "$FFTW_DIR" "$FFTW_ROOT" -name 'libfftw3*' 2>/dev/null
```

Note that `FFTW_ROOT` as an environment variable is read by Q3's discovery, but
plain `find_library` does not consult `<Package>_ROOT` — that is a
`find_package` convention. `-DCMAKE_LIBRARY_PATH=<directory>` works as an
alternative; `-DCMAKE_PREFIX_PATH` is unreliable because it searches
`<prefix>/lib` and misses `lib64`.

### `No rule to make target '/path/a.so /path/b.so /path/c.so'`

Several paths were passed in one variable. CMake lists are semicolon-separated,
so a space-separated string is a single value and the build looks for one file
with that entire name.

Each library has its own variable:

```bash
cmake -S . -B build -DFFTW3_LIB=$FFTW_DIR/libfftw3.so -DFFTW3_THREADS_LIB=$FFTW_DIR/libfftw3_threads.so -DFFTW3_MPI_LIB=$FFTW_DIR/libfftw3_mpi.so
```

Do not add `libfftw3_omp` alongside `libfftw3_threads`: they are alternative
threading backends and Q3 expects the latter.

### `q3_mpi` hangs above a few ranks

If the run stops during the first Davidson iteration and the log contains

```
Band parallelism: some ranks received 0 plane waves. Falling back to replicated G-vectors.
```

then in a build **without ScaLAPACK** the subspace eigenproblem was solved on
one rank while the others waited — fixed in current versions. Workaround on an
older checkout: give each rank its own pool, which avoids the path entirely.

```bash
srun -n 8 ./build/q3_mpi --pools 8 input.yml
```

`--pools` must divide the rank count and must not exceed the irreducible
k-point count (the log reports it as `IBZ reduction`). A gamma-only supercell
has one k-point, so it cannot use pools at all; run it on one rank with
`OMP_NUM_THREADS` instead.

Before concluding it is Q3, check that the ranks can run at all — a login node
caps CPU, and oversubscribed MPI busy-waits in a way that looks identical to a
hang:

```bash
srun --label -n 8 hostname
```

### `q3_mpi` is slower than `q3` on a gamma-only system

Expected, not a misconfiguration. A gamma-only calculation has one k-point, so there are
no k-point pools to spread across and every extra rank goes into a band group, which
distributes G-vectors within the pool. Measured on Perlmutter, band groups are slower
than a single threaded rank at every width: see
[docs/performance_nersc_perlmutter.md](docs/performance_nersc_perlmutter.md).

For such a system, use the serial binary with threads instead:

```bash
OMP_NUM_THREADS=8 ./build/q3 input.yml
```

Threading peaked around 8 threads there and regressed by 32, so more is not better.

### `MPI_Allreduce ... failed` and a segmentation fault with ScaLAPACK enabled

```
MPI_Allreduce(sendbuf=..., recvbuf=..., count=..., MPI_DOUBLE_COMPLEX, MPI_SUM, ...) failed
srun: error: tasks 0-7: Segmentation fault
```

Fixed in current versions. It happened with `Q3_WITH_SCALAPACK` whenever the run had
taken the replicated-G fallback: `group_size()` becomes 1 there while the group
communicator keeps all its ranks, so every rank solved its own copy of the problem while
still entering the collective ScaLAPACK path, at whatever Davidson basis size it had
reached. The subspace solve now branches on `group_size() <= 1` and solves locally.

On an older checkout, either keep to the band-group limit for your system (four ranks per
pool on the cases measured) or build without `SCALAPACK_LIBRARIES`.

Note that the ScaLAPACK path is only reached with more than one rank per pool: with one
rank per pool the subspace eigenproblem is local, so a run with `--pools` equal to the
rank count never touches it.

### `MPI_SUM operation not defined for this datatype` under `podman-hpc --mpi` or Shifter

```
MPIR_SUM_check_dtype(83): MPI_Op MPI_SUM operation not defined for this datatype
```

Fixed in current versions. It appeared only when the host's MPI was swapped into the
`-mpi` container -- `podman-hpc --mpi`, or Shifter's `--module=mpich` -- and only with
more than one rank per pool. The same image running its own MPICH summed the same buffer
correctly, which is what made it look like a code bug rather than an environment one.

Whether an MPI accepts `MPI_SUM` on `MPI_CXX_DOUBLE_COMPLEX` depends on how that library
was configured; MPICH treats the C++ complex types as optional for reductions. Q3 now
reduces complex buffers as pairs of `MPI_DOUBLE` instead, which is the same arithmetic
and asks nothing of the library beyond what the standard requires of every
implementation.

On an older image, use one rank per pool (`--pools` equal to the rank count), which never
reaches a complex reduction, or run without `--mpi` and let the container use its own
MPICH.

### `error while loading shared libraries` after `cmake --install`

The installed binary needs the libraries it was linked against. Keep the same
modules loaded at run time, or configure with `CRAY_ADD_RPATH=yes` exported so
wrapper-injected libraries are recorded too.

---

## Developer notes

The remaining sections are implementation notes about physics conventions,
retained for anyone changing the corresponding code.

## 1. Spherical Harmonic Indexing Mismatch

**Symptom:** Wrong total energies, wrong stress tensors, wrong forces — any physical quantity that depends on KB projectors. The errors can be large (e.g., nonlocal stress off by ~200 kbar) and affect all materials, all pseudopotential types.

**Root Cause:** Q3 has **two different (l,m) → flat-index mappings** for spherical harmonics, and mixing them up produces silently wrong results:

| (l, m) | `l² + l + m` (standard math) | `harmonic_combined_degree_order_index` (Q3 batch Ylm) |
| ------ | :--------------------------: | :---------------------------------------------------: |
| (0, 0) |              0               |                           0                           |
| (1, 0) |            **2**             |                         **1**                         |
| (1, 1) |            **3**             |                         **2**                         |
| (1,−1) |            **1**             |                         **3**                         |
| (2, 0) |            **6**             |                         **4**                         |
| (2, 1) |            **7**             |                         **5**                         |
| (2,−1) |            **5**             |                         **6**                         |
| (2, 2) |            **8**             |                         **7**                         |
| (2,−2) |            **4**             |                         **8**                         |

The `compute_real_spherical_harmonics_batch()` function in `harmonics.hpp` stores values using the **alternating ±m** convention (m = 0, +1, −1, +2, −2, …), which is what `harmonic_combined_degree_order_index(l, m)` returns. The **standard math** mapping `l² + l + m` uses a different order (m = −l, …, 0, …, +l).

Using the wrong index to read from the batch array returns the Y_lm value for a **different (l,m)**, introducing errors up to O(1) — not small numerical noise, but completely wrong values.

**How to Diagnose:**

1. If KB projectors produce wrong energies/stress but the per-point `compute_real_spherical_harmonic(l, m, direction)` gives correct results, suspect an indexing mismatch.
2. Write a quick test comparing `batch_values[g][lm_index]` against `compute_real_spherical_harmonic(l, m, g_vec)` for several random G-vectors. If errors are O(1), it's an indexing bug. If errors are O(1e-15), the indexing is correct.
3. The correct index for accessing batch Ylm values is **always** `harmonic_combined_degree_order_index(l, m)`.

**Resolution:** In any code that calls `compute_real_spherical_harmonics_batch()` and then reads from the result array, use:

```cpp
int lm_index = harmonic_combined_degree_order_index(angular_momentum, magnetic_projection);
Real ylm = spherical_harmonics_values[g_index][lm_index];
```

Never use `l * l + l + m` to index into batch Ylm arrays.

**Files involved:**

-   `src/layer_0_core/math/spherical/harmonics.hpp` — defines both the batch function and `harmonic_combined_degree_order_index`
-   `src/layer_3_representation/pseudopotential/planewave/internal/projector.hpp` — KB projector computation (was the site of the original bug)
-   `src/layer_5_properties/stress_tensor/planewave/pseudopotential/internal/nonlocal.hpp` — nonlocal stress (also uses batch Ylm)
-   `src/layer_3_representation/wavefunction/planewave/norm_conserving/initializer.hpp` — wavefunction initializer (also uses batch Ylm)

**Key lesson:** Whenever you see batch Ylm usage, verify the index mapping. The batch function and per-point function use **different conventions**. The per-point function takes `(l, m)` as arguments directly and handles the mapping internally. The batch function returns a flat array that must be indexed with `harmonic_combined_degree_order_index`.

---

## 2. Gamma-Only Half-Sphere Symmetry Breaking

**Symptom:** Eigenvalue degeneracies at the Γ-point are broken. For example, the triply-degenerate valence band top of FCC silicon (bands 2-4) splits by ~0.06 Ry instead of being degenerate to ~1e-6 Ry.

**Root Cause:** The gamma-only optimization stores only half the G-sphere (G_z > 0, plus the G_z = 0 hemisphere). This breaks the full point-group symmetry of the crystal because the G-vector set is no longer invariant under all symmetry operations of the cubic group.

**Current Status:** The gamma-only optimization is enabled and functional for total energy and stress calculations. The symmetry breaking is small enough that total energies and pressures are accurate, but eigenvalue degeneracies are not preserved. The integration tests use relaxed tolerances to account for this.

**Impact:**

-   Total energies: accurate (regression-tested)
-   Stress tensor: accurate (verified against pre-gamma-only baseline)
-   Forces: accurate **since 2026-08-08** — see the separate half-sphere force
    bug below, which is a different problem from the symmetry breaking described
    here and was fixed
-   Eigenvalue degeneracies: broken (~0.06 Ry splitting for Si FCC at Γ)
-   Band structure plots: may show spurious splittings at high-symmetry points

**Workaround:** If exact eigenvalue degeneracies are needed, disable gamma-only by hardcoding `use_gamma_only = false` in `init.hpp`.

**Proper fix (TODO):** Implement symmetry-aware G-vector selection for the half-sphere that preserves the crystal's point group. QE handles this by carefully choosing which G-vectors to include/exclude at the boundary.

### 2a. Gamma-only forces were wrong until 2026-08-08 (fixed)

A separate half-sphere problem, and a more damaging one, because it produced no
symptom at all. Every G-sum force term is _even_ under G → −G, so summing the
stored half sphere gives half the answer: the **local**, **NLCC**, **Ewald** and
**augmentation** forces all lacked the factor of two. The **nonlocal** term had
a different bug — `2·Re[⟨β|ψ⟩_half]` double counts G = 0, because Q3's half
sphere includes it, while the derivative overlap needs no correction since its
summand carries G_α. That one touched only l = 0 projectors, which is why it
skewed force components unevenly instead of scaling them.

Silicon's Γ force was **0.637** of the derivative of Q3's own energy with a
norm-conserving dataset and **0.822** with PAW. Energies were correct
throughout, so nothing looked wrong — a geometry relaxation at Γ, the classic
molecule-in-a-box case, simply converged somewhere else.

If you are looking at pre-2026-08-08 Γ-only relaxation results, redo them.
Regression coverage:
`tests/validation/test_paw_forces_and_stress.py::test_gamma_only_forces_are_the_gradient_of_q3s_own_energy`.

---

## 3. Batch Ylm vs. Per-Point Ylm: When to Use Which

The codebase provides two ways to compute real spherical harmonics:

### `compute_real_spherical_harmonic(l, m, direction)` — Per-Point

-   Takes a single `(l, m, direction)` and returns one Y_lm value
-   Uses hard-coded Cartesian formulas (l ≤ 4)
-   Numerically robust everywhere (no trigonometric intermediate)
-   **Use for:** debugging, small computations, or when you only need one (l,m) at a time

### `compute_real_spherical_harmonics_batch(...)` — Batch

-   Computes ALL (l,m) components for ALL G-vectors in one call
-   Uses associated Legendre recurrence, matching the alternating ±m ordering Quantum ESPRESSO produces
-   Much faster for large G-vector sets (thousands of G-vectors × multiple lm)
-   **Use for:** KB projectors, stress tensor overlaps, any inner-loop computation
-   **CRITICAL:** Access results using `harmonic_combined_degree_order_index(l, m)`, NOT `l*l + l + m`

### `compute_spherical_harmonic_derivatives_batch(...)` — Batch Derivatives

-   Computes dY_lm/dG_i using finite differences on the batch Ylm function
-   Used in stress tensor computations
-   Same indexing convention as the batch Ylm function

Both functions are numerically equivalent to machine precision (~1e-15). The only pitfall is the indexing convention (see Issue #1 above).
