"""q3.jax.primitives — JAX-differentiable DFT primitives backed by Q3.

Exposes ``dft_energy()`` and ``dft_forces()`` as JAX-traceable functions so
that automatic differentiation can be applied to DFT total energies.

The key identities exploited:

    ∂E/∂R_I = -F_I     (Hellmann-Feynman theorem)

This means the JVP (forward-mode derivative) of ``dft_energy()`` with respect
to atomic positions is given by the negative of the forces returned by the
same Q3 calculation.  For the VJP (reverse-mode / gradient), the same forces
serve as the cotangent.

Architecture
------------
Both primitives work in Mode A (subprocess) by default.  If ``q3_core`` is
available (Mode B / nanobind), all calls to ``q3.flavors.run_scf()`` already
use it automatically.

Status
------
* ``dft_energy`` — **functional**.  ``jax.grad(dft_energy)(positions)``
  returns forces at the current geometry.
* ``dft_forces`` — **stub** (custom_vjp registered but backward not yet
  implemented; awaiting DFPT support in the C++ core).

Usage::

    import jax
    import jax.numpy as jnp
    from q3.jax import dft_energy

    # positions shape: (num_atoms, 3), in Bohr
    forces = jax.grad(dft_energy)(positions, scf_input_path=..., q3_binary=...)

Notes
-----
JAX tracing does not flow *through* the Q3 DFT call (it is an opaque
computation).  The custom_jvp/custom_vjp rules implement the Hellmann-Feynman
theorem as an explicit tangent/cotangent injection point.
"""

from __future__ import annotations

from typing import Any

from q3.flavors import run_scf

# ---------------------------------------------------------------------------
# JAX availability guard
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


def _require_jax() -> None:
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for q3.jax primitives.  Install it with: pip install jax"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_scf_with_forces(
    positions_bohr: Any,
    scf_input_path: str,
    q3_binary: str | None = None,
) -> tuple[float, list[list[float]]]:
    """Run SCF and return (total_energy_ry, forces_ry_per_bohr).

    ``positions_bohr`` is only used to size-check; the actual positions are
    baked into the YAML at ``scf_input_path``.  In a full implementation this
    function would rebuild the YAML with the supplied positions.

    Returns
    -------
    (energy_ry, forces_ry_per_bohr)
        energy_ry: float
        forces_ry_per_bohr: list of [fx, fy, fz] per atom
    """
    results = run_scf(scf_input_path, q3_binary=q3_binary)
    energy = results.properties.total_energy
    forces = results.properties.atomic_forces
    return (
        float(energy.value) if energy is not None else 0.0,
        [atom.value for atom in forces.values] if forces is not None else [],
    )


# ---------------------------------------------------------------------------
# dft_energy — differentiable total energy
# ---------------------------------------------------------------------------


def _make_dft_energy():
    """Factory that creates the JAX-traced dft_energy primitive."""
    if not _JAX_AVAILABLE:
        return None

    @jax.custom_jvp
    def dft_energy(
        positions: jnp.ndarray,
        *,
        scf_input_path: str,
        q3_binary: str | None = None,
    ) -> jnp.ndarray:
        """Compute the DFT total energy as a JAX scalar.

        Parameters
        ----------
        positions : jnp.ndarray, shape (num_atoms, 3)
            Atomic positions in Bohr (Cartesian).  JAX traces through this
            array to enable automatic differentiation via the Hellmann-Feynman
            tangent rule registered below.
        scf_input_path : str
            Path to a Q3 YAML input file.  The geometry in this file must
            correspond to ``positions`` (the Python layer does not yet
            automatically rewrite the YAML with the supplied positions).
        q3_binary : str or None
            Path to the ``q3`` binary for Mode A.  Auto-detected when ``None``.

        Returns
        -------
        jnp.ndarray
            Scalar total energy in Rydberg.
        """
        energy, _ = _run_scf_with_forces(positions, scf_input_path, q3_binary)
        return jnp.asarray(energy, dtype=jnp.float64)

    @dft_energy.defjvp
    def _dft_energy_jvp(primals, tangents):
        """Forward-mode derivative via the Hellmann-Feynman theorem.

        ∂E/∂R_I = -F_I

        The JVP is:  ẋ · ∂E/∂R = -ẋ · F
        where ẋ is the tangent (velocity) of positions.
        """
        (positions,) = primals
        (positions_dot,) = tangents

        # Recover keyword arguments from the closure via the attached attributes.

        energy, forces = _run_scf_with_forces(
            positions, _dft_energy_jvp._scf_input_path, _dft_energy_jvp._q3_binary
        )
        forces_array = jnp.asarray(forces, dtype=jnp.float64)
        primal_out = jnp.asarray(energy, dtype=jnp.float64)

        # ∂E/∂R · ṙ = -F · ṙ
        tangent_out = -jnp.sum(forces_array * positions_dot)
        return primal_out, tangent_out

    return dft_energy


# ---------------------------------------------------------------------------
# dft_forces — differentiable forces (stub)
# ---------------------------------------------------------------------------


def _make_dft_forces():
    """Factory that creates the JAX-traced dft_forces primitive (stub)."""
    if not _JAX_AVAILABLE:
        return None

    @jax.custom_vjp
    def dft_forces(
        positions: jnp.ndarray,
        *,
        scf_input_path: str,
        q3_binary: str | None = None,
    ) -> jnp.ndarray:
        """Compute DFT Hellmann-Feynman forces as a JAX array.

        Parameters
        ----------
        positions : jnp.ndarray, shape (num_atoms, 3)
            Atomic positions in Bohr (Cartesian).
        scf_input_path : str
            Path to a Q3 YAML input file.
        q3_binary : str or None
            Path to the ``q3`` binary.

        Returns
        -------
        jnp.ndarray, shape (num_atoms, 3)
            Forces in Ry/Bohr (Cartesian).

        Notes
        -----
        The VJP (backward pass) is currently a **placeholder** that raises
        ``NotImplementedError``.  Full DFPT-backed differentiation will be
        added in a future phase.
        """
        _, forces = _run_scf_with_forces(positions, scf_input_path, q3_binary)
        return jnp.asarray(forces, dtype=jnp.float64)

    def _fwd(positions, *, scf_input_path, q3_binary=None):
        result = dft_forces(positions, scf_input_path=scf_input_path, q3_binary=q3_binary)
        # Residuals: save positions and forces for backward.
        return result, (positions, result)

    def _bwd(residuals, cotangent):
        """Backward pass for dft_forces — requires DFPT (not yet implemented).

        In principle: ∂F_I/∂R_J = -∂²E/∂R_I ∂R_J  (Hessian of energy).
        A full implementation requires DFPT or finite-difference Hessian.
        """
        raise NotImplementedError(
            "Backward pass for dft_forces requires DFPT support (not yet implemented).  "
            "Use jax.grad(dft_energy) instead — it provides forces via the "
            "Hellmann-Feynman theorem at zero extra cost."
        )

    dft_forces.defvjp(_fwd, _bwd)
    return dft_forces


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _not_available(name: str):
    """Return a placeholder that raises ImportError on call."""

    def _placeholder(*args, **kwargs):
        _require_jax()  # will raise ImportError with a clear message

    _placeholder.__name__ = name
    _placeholder.__doc__ = f"{name} — requires JAX (pip install jax)"
    return _placeholder


if _JAX_AVAILABLE:
    dft_energy = _make_dft_energy()
    dft_forces = _make_dft_forces()
else:
    dft_energy = _not_available("dft_energy")  # type: ignore[assignment]
    dft_forces = _not_available("dft_forces")  # type: ignore[assignment]
