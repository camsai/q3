"""q3.workflows.pdos — Projected Density of States (PDOS) workflow.

Orchestrates a two-step PDOS calculation:

1. Run SCF to get the self-consistent effective potential.
2. Run NSCF over a dense, full (unreduced) Monkhorst-Pack k-grid with
   ``compute_projections: true``, producing ortho-atomic/Löwdin orbital
   projections alongside eigenvalues and occupations.

The raw NSCF results are then post-processed (Gaussian-broadened per-orbital
DOS curves, Löwdin population charges) by
``q3.io.output_writer.extract_pdos`` / ``extract_lowdin_charges``.

Example::

    from q3.workflows import run_pdos_workflow

    result = run_pdos_workflow(
        scf_input_path="examples/total_energy--semiconductor--silicon/nc.yml",
        kgrid_dimensions=(4, 4, 4),
        q3_binary="./build/q3",
    )
    # result["pdos"]["atom_shell_pdos"]      → per-atom, per-shell PDOS curves
    # result["lowdin_charges"]["atom_shell"] → per-atom, per-shell charges
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import yaml

from q3.binary import find_binary, run_q3
from q3.io.input_writer import write_nscf_input
from q3.io.output_writer import extract_lowdin_charges, extract_pdos

# ---------------------------------------------------------------------------
# Monkhorst-Pack grid generation
# ---------------------------------------------------------------------------


def monkhorst_pack_grid(
    dimensions: tuple[int, int, int],
    shifts: tuple[int, int, int] = (0, 0, 0),
) -> list[tuple[float, float, float]]:
    """Generate a full (unreduced) Monkhorst-Pack k-point grid.

    Mirrors ``KPointsGrid::generate()`` in
    ``src/layer_1_structure/reciprocal_space/kpoints_grid.hpp`` exactly (same
    fractional-coordinate convention and [-0.5, 0.5) folding), but without
    symmetry (IBZ) reduction: every k-point in the full grid is listed
    explicitly. This is required because NSCF's ``kpoints_list`` input path
    applies no reduction of its own (see
    ``build_global_kpoint_grid_from_input`` in
    ``src/io/input/system_builder.hpp``) -- it is more expensive than a
    symmetry-reduced grid but correct for any point group.

    Parameters
    ----------
    dimensions:
        Grid dimensions ``(n1, n2, n3)``.
    shifts:
        Grid shifts ``(s1, s2, s3)``, each ``0`` or ``1`` (QE convention).

    Returns
    -------
    list[tuple[float, float, float]]
        Fractional k-point coordinates, ``n1 * n2 * n3`` entries.
    """
    n1, n2, n3 = dimensions
    s1, s2, s3 = shifts

    def fold(value: float) -> float:
        return value - math.floor(value + 0.5)

    kpoints = []
    for x_index in range(n1):
        x = x_index / n1 + (0.5 / n1 if s1 else 0.0)
        for y_index in range(n2):
            y = y_index / n2 + (0.5 / n2 if s2 else 0.0)
            for z_index in range(n3):
                z = z_index / n3 + (0.5 / n3 if s3 else 0.0)
                kpoints.append((fold(x), fold(y), fold(z)))
    return kpoints


def monkhorst_pack_kpoints_list(
    dimensions: tuple[int, int, int],
    shifts: tuple[int, int, int] = (0, 0, 0),
) -> list[tuple[float, float, float, float]]:
    """Full Monkhorst-Pack grid as a ``kpoints_list`` with correct weights.

    Weight per k-point is ``2.0 / (n1*n2*n3)``, matching
    ``KPointsGrid::generate()``'s spin-degeneracy convention (weights sum to
    2.0 over the full grid). This is required for Löwdin charges and PDOS
    integration to come out correctly -- unlike
    ``q3.workflows.kpath.kpoints_list_with_uniform_weight`` (weight 1.0 per
    point), which is only appropriate for band-structure plotting where
    eigenvalues are read individually and weights are never summed.

    Parameters
    ----------
    dimensions:
        Grid dimensions ``(n1, n2, n3)``.
    shifts:
        Grid shifts ``(s1, s2, s3)``, each ``0`` or ``1``.

    Returns
    -------
    list[tuple[float, float, float, float]]
        ``(kx, ky, kz, weight)`` tuples suitable for
        ``write_nscf_input(..., kpoints_list=...)``.
    """
    n1, n2, n3 = dimensions
    weight = 2.0 / (n1 * n2 * n3)
    return [(kx, ky, kz, weight) for kx, ky, kz in monkhorst_pack_grid(dimensions, shifts)]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def run_pdos_workflow(
    scf_input_path: str | Path,
    *,
    kgrid_dimensions: tuple[int, int, int] = (4, 4, 4),
    kgrid_shifts: tuple[int, int, int] = (0, 0, 0),
    number_of_bands: int | None = None,
    broadening_ry: float = 0.01,
    energy_min_ry: float | None = None,
    energy_max_ry: float | None = None,
    energy_steps: int = 1000,
    q3_binary: str | Path | None = None,
    skip_scf: bool = False,
    capture_output: bool = True,
    nscf_input_path: str | Path | None = None,
) -> dict:
    """Run a PDOS workflow (SCF + NSCF with ortho-atomic projections).

    Parameters
    ----------
    scf_input_path:
        Path to a Q3 YAML input file for the SCF step. The SCF prefix and
        outdir from this file are reused for the NSCF step so the checkpoint
        is found.
    kgrid_dimensions:
        Full (unreduced) Monkhorst-Pack grid dimensions ``(n1, n2, n3)`` for
        the PDOS NSCF step. Should generally be at least as dense as the SCF
        grid; PDOS/Löwdin-charge accuracy is limited by this grid's density,
        not the SCF grid's.
    kgrid_shifts:
        MP grid shifts ``(s1, s2, s3)`` (0 or 1).
    number_of_bands:
        Number of bands for the NSCF calculation. If ``None``, the SCF value
        from *scf_input_path* is used.
    broadening_ry, energy_min_ry, energy_max_ry, energy_steps:
        Forwarded to ``extract_pdos`` for the Gaussian-broadened PDOS curves.
    q3_binary:
        Path to the ``q3`` binary. Auto-detected when ``None``.
    skip_scf:
        When ``True``, assume the SCF has already been run and a checkpoint
        exists in the output directory. Skip straight to NSCF.
    capture_output:
        Pass ``False`` to stream binary output to the terminal.
    nscf_input_path:
        Where to write the generated NSCF input YAML. A temporary file is
        used when ``None``.

    Returns
    -------
    dict
        Contains:
        - ``"scf_results"`` — dict from SCF run (or ``None`` if *skip_scf*).
        - ``"nscf_results"`` — raw dict from NSCF ``results.json``.
        - ``"pdos"`` — see ``extract_pdos``.
        - ``"lowdin_charges"`` — see ``extract_lowdin_charges``.
    """
    scf_input_path = Path(scf_input_path).expanduser().resolve()
    binary = find_binary(q3_binary)

    # ── Step 1: SCF ──────────────────────────────────────────────────────────
    # Rewrite the input YAML so the binary sees self_consistent_field and
    # outdir is absolute (avoids CWD-vs-YAML resolution mismatch), mirroring
    # q3.workflows.band_structure.
    with open(scf_input_path) as file_handle:
        scf_config = yaml.safe_load(file_handle)

    scf_config.setdefault("control", {})["calculation"] = "self_consistent_field"

    outdir = scf_config.get("control", {}).get("outdir", "./output")
    if not Path(outdir).is_absolute():
        scf_config["control"]["outdir"] = str((scf_input_path.parent / outdir).resolve())

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yml",
        dir=scf_input_path.parent,
        delete=False,
    ) as tmp_scf_file:
        yaml.dump(scf_config, tmp_scf_file, default_flow_style=False)
        effective_scf_path = Path(tmp_scf_file.name)

    scf_results: dict | None = None
    try:
        if not skip_scf:
            scf_results = run_q3(
                effective_scf_path, q3_binary=binary, capture_output=capture_output
            )

        # ── Step 2: Full-BZ Monkhorst-Pack grid for PDOS ────────────────────────
        kpoints_list = monkhorst_pack_kpoints_list(kgrid_dimensions, kgrid_shifts)

        # ── Step 3: Write NSCF input with projections enabled ───────────────────
        if nscf_input_path is not None:
            nscf_path = Path(nscf_input_path).expanduser().resolve()
            nscf_path.parent.mkdir(parents=True, exist_ok=True)
            cleanup_nscf_path = False
        else:
            # A temporary file, per this parameter's own docstring -- NOT a predictable
            # sibling path, which risks silently overwriting a same-named committed
            # example file (see run_band_structure_workflow's identical fix/comment).
            with tempfile.NamedTemporaryFile(
                suffix="_pdos_nscf.yml", dir=scf_input_path.parent, delete=False
            ) as tmp_nscf_file:
                nscf_path = Path(tmp_nscf_file.name)
            cleanup_nscf_path = True

        write_nscf_input(
            nscf_path,
            scf_input_path=effective_scf_path,
            kpoints_list=kpoints_list,
            number_of_bands=number_of_bands,
            compute_projections=True,
            uniform_grid=True,
        )
    finally:
        effective_scf_path.unlink(missing_ok=True)

    # ── Step 4: NSCF run ─────────────────────────────────────────────────────
    try:
        nscf_results = run_q3(nscf_path, q3_binary=binary, capture_output=capture_output)
    finally:
        if cleanup_nscf_path:
            nscf_path.unlink(missing_ok=True)

    # ── Step 5: Post-process ─────────────────────────────────────────────────
    # Prefer the SCF Fermi energy, same rationale as run_band_structure_workflow:
    # it's the authoritative self-consistent value. Unlike a band path, this
    # workflow's NSCF grid *is* a proper full-BZ Monkhorst-Pack grid, so its own
    # fermi_energy_ry is also meaningful -- used only when skip_scf=True (no SCF
    # results). Löwdin charges don't need this: they're built directly from the
    # NSCF run's own occupations, which already reflect its own Fermi bisection.
    if scf_results is not None:
        fermi_energy_ry = scf_results.get("fermi_energy_ry")
    else:
        fermi_energy_ry = nscf_results.get("fermi_energy_ry")

    lowdin_charges = extract_lowdin_charges(nscf_results)
    pdos = extract_pdos(
        nscf_results,
        energy_min_ry=energy_min_ry,
        energy_max_ry=energy_max_ry,
        energy_steps=energy_steps,
        broadening_ry=broadening_ry,
        fermi_energy_ry=fermi_energy_ry,
    )

    return {
        "scf_results": scf_results,
        "nscf_results": nscf_results,
        "pdos": pdos,
        "lowdin_charges": lowdin_charges,
    }
