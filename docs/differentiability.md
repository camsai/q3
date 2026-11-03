# Differentiability

Status: design note (not yet implemented). Captures the intended approach for
making the engine differentiable so it can serve as a data **and gradient**
engine for machine-learned interatomic potentials, inverse design, and learned
exchange-correlation functionals.

The engine is already _partially_ differentiable: forces are `−dE/dR` and the
stress tensor is `dE/dε`, both computed analytically. "Making it differentiable"
means generalizing that existing machinery into a uniform gradient API and
exposing it to ML toolchains.

---

## 1. Two usage regimes (decide before building bindings)

The engineering cost is dominated by _which_ regime you target.

### Regime A — offline data/gradient generation (near-term, dominant)

The engine runs as a batch job: for each structure, return
`(energy, forces, ∂E/∂θ)`. A single host (CPU↔GPU) boundary per _full SCF solve_
is negligible against the cost of the solve itself.

-   No XLA FFI, no DLPack zero-copy gymnastics required.
-   Delivers the ML-training-data value with minimal binding complexity.
-   Binding: a plain `(E, grad)` batch API and/or a PyTorch
    `torch.autograd.Function` backed by a C++/CUDA op.

### Regime B — live differentiable coupling (later, large effort)

The engine is invoked repeatedly _inside_ a training/optimization graph
(inverse design, in-the-loop functional learning). Here fine-grained host
roundtrips are catastrophic and the binding/memory traps in §4 all apply.

**Recommendation:** build the adjoint math (§2) first — it is required in both
regimes — then ship Regime A. Only build the on-device JAX path (§4) when a
concrete Regime B workload justifies it.

---

## 2. The math: don't backprop through the SCF loop

Unrolling the self-consistent iterations and reverse-mode AD-ing through them is
a trap (memory blow-up, numerical noise, differentiates the _solver_ not the
_solution_).

### Implicit differentiation at the fixed point

The SCF solution satisfies `n* = F(n*, θ)`. Differentiate the converged fixed
point, not the iteration:

```
dn*/dθ = (I − ∂F/∂n)^(-1) · ∂F/∂θ
```

Solve one linear system at convergence, reusing the existing mixing / Krylov
machinery to apply `(I − ∂F/∂n)^(-1)`. This is less code and more stable than
differentiating each step.

### Exploit stationarity (first derivatives)

At self-consistency the energy is variational, so wavefunction-response terms
vanish (Hellmann–Feynman):

```
dE/dθ = ⟨ψ| ∂H/∂θ |ψ⟩
```

The plane-wave basis is position-independent, so atomic-position gradients carry
no Pulay/basis-derivative term (already exploited for forces). Generalize this
from "θ = atomic position" to "θ = any parameter".

### Second order / non-variational observables

Needed for phonons, elastic constants, Hessian-based quantities. Solve the
DFPT / Sternheimer response equation:

```
(H − ε_i S) |∂_θ ψ_i⟩ = −(∂_θ H − ∂_θ ε_i · S) |ψ_i⟩
```

projected onto the unoccupied subspace.

---

## 3. Per-term adjoints (the real work)

Each energy term exposes value **and** a hand-coded vector–Jacobian product.
This maps onto the existing calculator decomposition (each `*Calculator` gains a
`backward()` / `vjp()` alongside `compute()`).

Difficulty is **not** uniform:

-   **FFTs are easy.** An FFT is linear and (normalized) unitary; its VJP is the
    conjugate/inverse transform — exact and cheap. Do not treat this as a
    stability hazard.
-   **Exchange-correlation is mostly plumbing.** LibXC already provides `vxc`
    (1st), `fxc` (2nd), `kxc` (3rd) functional derivatives.
-   **The eigensolver adjoint is the hard part.** Eigenpair derivatives contain
    `1/(ε_i − ε_j)` terms; guard against near-degeneracy with degenerate
    perturbation theory.
-   **Occupations are made smooth by smearing.** Integer fillings are
    non-differentiable, but Gaussian / Fermi–Dirac smearing (already implemented)
    makes occupations a smooth function of eigenvalues. Keep `degauss > 0` when
    computing parameter gradients on metals.

Fixed/discrete inputs (k-grid, symmetry reduction, G-vector cutoff) are held
constant across a gradient evaluation.

### Proposed interface

```cpp
struct EnergyTerm {
    Real value() const;                 // forward (primal)
    // reverse-mode: given dL/d(this term's output), accumulate dL/dθ
    void vjp(const Cotangent& seed, Gradients& accum) const;
};
```

Forward vs reverse mode by parameter count:

-   Few params → many outputs (e.g. `dE/dR` on a small cell): forward mode / dual
    numbers — simplest, memory-light.
-   Many params → scalar loss (e.g. ML functional weights): reverse mode / adjoint
    — one response solve, then accumulate.

Build both; route on `dim(θ)`.

---

## 4. Binding to ML frameworks (Regime B specifics)

The C++ core owns the math (forward + adjoints). The Python layer registers the
engine as a custom differentiable op.

### PyTorch (simpler — start here)

`torch.autograd.Function` + a custom C++/CUDA extension works via standard
bindings.

### JAX (the XLA custom-call trap)

`nanobind` + `jax.custom_vjp` wrapping a host function behaves like a
`pure_callback`: JAX pulls tensors off GPU VRAM, runs the C++ on CPU, and copies
back — a catastrophic I/O bottleneck for fine-grained ops.

**Fix:** lower the C++ entry points into XLA via **`jax.ffi`** (the current
foreign-function interface, superseding hand-registered `custom_call` targets)
so the compiled graph stays on the accelerator. Caveat: `jax.ffi` only helps if
the underlying kernel is itself on-device (CUDA/HIP/SYCL) — FFI alone does not
remove a CPU-only kernel's roundtrip.

### Zero-copy memory (non-negotiable on GPU)

When data crosses the C++/Python boundary, copying multi-GB wavefunction arrays
will dominate runtime. Use **DLPack** (supported by `nanobind`) to share GPU
buffers across PyTorch / JAX / CuPy without reallocation; use the buffer /
`ndarray` protocol on CPU.

---

## 5. Disciplines that keep adjoints correct

-   Keep kernels pure / avoid in-place mutation of buffers needed in the backward
    pass; cache the converged state (density, wavefunctions, occupations,
    eigenvalues).
-   Stay in atomic units (already done) so no conversion factors leak into
    gradients.
-   Add **gradient tests**: finite-difference checks against analytic VJPs for
    every term, as a first-class part of the validation suite.

---

## 6. Suggested phasing

1. Refactor existing force/stress derivative code into a uniform `EnergyTerm`
   interface — no new physics, just structure.
2. Implicit-diff at the SCF fixed point, reusing the mixer as the linear solver;
   validate `dE/dR` against current analytic forces (match to machine
   precision).
3. Parameter gradients beyond geometry: XC and pseudopotential parameters via
   LibXC `fxc` + adjoints.
4. Regime A binding: `(E, grad)` batch API and/or PyTorch `autograd.Function`.
5. Second order (DFPT / Sternheimer) for phonons, elastic constants.
6. Regime B (only if justified): `jax.ffi` + DLPack on-device path.

Steps 1–2 are low-risk and also harden forces. Steps 3–4 unlock the
"data _and_ gradient engine for ML potentials" differentiator.

---

## 7. Generalizing across discretizations

The approach above is written for the current plane-wave representation. If the
engine later adds other discretizations (real-space grids, local-orbital / LCAO
DFT), most of the machinery carries over — but it bakes in a few plane-wave
assumptions that must be made explicit.

### What is basis-agnostic (carries over unchanged)

-   Implicit differentiation at the SCF fixed point (§2).
-   Variational stationarity (Hellmann–Feynman) for first derivatives and
    DFPT/Sternheimer for second order (§2).
-   Smearing → smooth occupations.
-   Forward-vs-reverse routing (§3).
-   The entire binding story (§4): regimes A/B, DLPack zero-copy, the JAX/XLA
    trap. These concern data movement and ML interfaces, not the basis.

So layer 4 (`computation`) and the binding layers survive a basis change largely
intact. **The changes live in layer 3 (`representation`) and in how layer 5
(`properties`) assembles gradients.**

### The dominant change: Pulay (basis-derivative) terms become first-class

Plane waves `e^(iG·r)` do **not** move with the atoms, so atomic-position
gradients have no basis-derivative term — pure Hellmann–Feynman. The note relies
on this. The general form is:

```
dE/dθ = ⟨ψ| ∂H/∂θ |ψ⟩        (Hellmann–Feynman)
        + (terms from ∂φ/∂θ)   (Pulay / basis-derivative)
```

Note the plane-wave basis already depends on the **cell** (G-vectors and the
cutoff sphere scale with the reciprocal lattice) — which is exactly why stress
already needs finite-difference `dV/dG` handling. The `EnergyTerm` / operator
interface must therefore treat `∂/∂θ` as a general, possibly-nonzero quantity
supplied by the representation: zero for plane-wave position gradients, nonzero
otherwise.

### Real-space grids (finite-difference / finite-element / wavelets)

-   **Operators change shape.** Kinetic energy is no longer diagonal-in-G; it
    becomes a stencil / sparse operator. Its VJP is still trivial (transpose of a
    linear operator), but the response / implicit-diff solve **can no longer rely
    on FFT diagonalization** — it must be a matrix-free Krylov solve (CG/GMRES)
    against an abstract `operator.apply()`. This is the most important code change.
-   **Hartree/Poisson** moves to multigrid (or FFT); the adjoint of a self-adjoint
    Poisson solve is another Poisson solve.
-   **Discretization noise / "egg-box".** A fixed grid breaks exact translational
    invariance, so forces may not be the exact gradient of the discretized energy
    → noisy/inconsistent gradients. A differentiable design must use a
    variational-preserving discretization, apply double-grid/compensation, or
    explicitly document the gradient tolerance. This concern does not exist for
    plane waves.
-   Uniform grids are atom-position-independent (no position Pulay, only egg-box),
    but cell strain deforms the grid → cell/stress gradients gain grid-deformation
    terms.

### Local-orbital / LCAO DFT (Gaussian or numerical atomic orbitals)

The largest departure: the basis is attached to atoms, `φ_μ(r − R_A)`.

-   **Pulay is mandatory and pervasive.** Overlap `S(θ)` and Hamiltonian `H(θ)`
    depend on atomic positions through the basis functions. Position gradients
    require differentiating the basis integrals (overlap, kinetic, nuclear
    attraction, Hartree/XC matrix elements). Analytic for Gaussians; radial-spline
    derivatives for numerical atomic orbitals.
-   **The hard work shifts.** In plane waves the hard adjoint is the eigensolver
    (degeneracy). In LCAO the eigensolver adjoint is the same, but the bulk of the
    effort moves to **basis-integral derivatives**.
-   **Reuse of ultrasoft infrastructure.** LCAO is intrinsically a generalized
    eigenproblem `Hψ = εSψ` with position-dependent `S` — structurally the same
    machinery already built for ultrasoft. The US "overlap operator + `∂S/∂θ`"
    work is the direct stepping stone; generalize it from US augmentation to the
    basis overlap.
-   The Pulay term has a clean closed form via the **energy-weighted density
    matrix**, `Tr(W ∂S/∂θ)`, which maps directly onto a VJP.
-   Hybrid functionals in a Gaussian basis additionally need two-electron integral
    derivatives (large but analytic).
-   **Silver lining:** LCAO/Gaussian DFT with analytic integrals is markedly more
    amenable to full AD than plane-wave DFT, so a future LCAO track could lean on
    an AD framework for the integral layer while still using implicit diff for the
    SCF fixed point.

### Architectural takeaways (do these now to stay portable)

1. **Differentiability contract in layer 2.** Every representation must supply
   operators _and_ their parameter-derivatives / VJPs, with Pulay allowed to be
   nonzero.
2. **Matrix-free response solver in layer 4.** Depend only on `operator.apply()`
   (plus the mixer as preconditioner); never on FFT-diagonal structure. This is
   what lets real-space and LCAO reuse the same implicit-diff code.
3. **Hellmann–Feynman + Pulay assembly in layer 5.** Sum a representation-supplied
   explicit term and a representation-supplied basis-derivative term; plane waves
   contribute zero to the latter for positions, other bases do not.

In short: §2 and §4 carry over unchanged; the adjustments are (a) make Pulay /
basis-derivatives first-class, (b) make the response solver discretization-agnostic,
and (c) expect the hard adjoint work to migrate from eigensolver degeneracy
(plane waves) toward basis-integral derivatives (LCAO) or discretization-noise
control (real-space). The ultrasoft overlap-derivative work is the natural bridge.
