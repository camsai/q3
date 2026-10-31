"""q3.workflows.band_structure — Band structure workflow.

Orchestrates a two-step band structure calculation:

1. Run SCF to get the self-consistent charge density and effective potential
   (skip if a valid checkpoint already exists).
2. Run NSCF at explicit k-points along the requested high-symmetry path.

Example::

    from q3.workflows import run_band_structure_workflow

    result = run_band_structure_workflow(
        scf_input_path="examples/total_energy--semiconductor--silicon/nc.yml",
        band_path=["G", "X", "W", "K", "G", "L"],
        lattice_type="FCC",
        points_per_segment=30,
        q3_binary="./build/q3",
    )
    # result["band_structure"]["eigenvalues_ry"]  → eigenvalues
    # result["band_structure"]["kpath_distances"] → x-axis for plotting
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from q3.binary import ExecutionPolicy, find_binary, run_q3
from q3.io.input_writer import write_nscf_input
from q3.io.output_writer import extract_band_structure
from q3.workflows.kpath import (
    band_path_from_input,
    interpolate_kpath,
    kpoints_list_with_uniform_weight,
)


def run_band_structure_workflow(
    scf_input_path: str | Path,
    *,
    band_path: list[str] | None = None,
    lattice_type: str | None = None,
    points_per_segment: int | None = None,
    number_of_bands: int | None = None,
    q3_binary: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    skip_scf: bool = False,
    capture_output: bool = True,
    nscf_input_path: str | Path | None = None,
) -> dict:
    """Run a band structure workflow (SCF + NSCF).

    Parameters
    ----------
    scf_input_path:
        Path to a Q3 YAML input file for the SCF step.  The file must specify
        ``calculation: scf`` (or equivalent).  The SCF prefix and outdir from
        this file are reused for the NSCF step so the checkpoint is found.

        This same file may also carry ``control.band_path`` and
        ``control.band_path_points_per_segment`` (the ``calculation:
        band_structure`` / ``bands`` YAML convention every
        ``examples/band_structure--*/`` file uses) -- when *band_path* or
        *points_per_segment* is not passed explicitly, they default to
        whatever this file states. *lattice_type* likewise defaults to the
        referenced material's own ``lattice.type``. The C++ engine itself
        never reads ``calculation: band_structure`` directly (see
        ``create_flavor()`` in ``src/layer_4_computation/flavors/factory.hpp``
        -- multi-step workflows are Python-orchestrated by design); this is
        what actually consumes those fields.
    band_path:
        Ordered list of high-symmetry point labels, e.g.
        ``["G", "X", "W", "K", "G", "L"]``. Defaults to *scf_input_path*'s
        own ``control.band_path`` if present.
    lattice_type:
        Bravais lattice type string used to look up fractional coordinates for
        labels in *band_path* (e.g. ``"FCC"``). Defaults to the ``lattice.type``
        of the material *scf_input_path* references.
    points_per_segment:
        Number of k-points per path segment. Defaults to *scf_input_path*'s
        own ``control.band_path_points_per_segment`` if present, else 30.
    number_of_bands:
        Number of bands for the NSCF calculation.  If ``None``, the SCF
        value from *scf_input_path* is used.
    q3_binary:
        Path to the ``q3`` binary.  Auto-detected when ``None``.
    execution_policy:
        The execution policy every engine run of the workflow starts the
        binary with: ranks, k-point pools, the launcher (see
        :class:`q3.binary.ExecutionPolicy`).  ``None`` runs one process with
        the binary's defaults.
    skip_scf:
        When ``True``, assume the SCF has already been run and a checkpoint
        exists in the output directory.  Skip straight to NSCF.
    capture_output:
        Pass ``False`` to stream binary output to the terminal.
    nscf_input_path:
        Where to write the generated NSCF input YAML.  A temporary file is
        used when ``None``.

    Returns
    -------
    dict
        Contains:
        - ``"scf_results"`` — dict from SCF run (or ``None`` if *skip_scf*).
        - ``"nscf_results"`` — raw dict from NSCF ``results.json``.
        - ``"band_structure"`` — structured band structure with keys
          ``eigenvalues_ry``, ``kpoints``, ``kpath_distances``,
          ``fermi_energy_ry``.
    """
    scf_input_path = Path(scf_input_path).expanduser().resolve()
    binary = find_binary(q3_binary)

    with open(scf_input_path) as file_handle:
        scf_config = yaml.safe_load(file_handle)

    # ── Defaults from scf_input_path itself ────────────────────────────────────
    # The calculation: band_structure / bands YAML convention (every
    # examples/band_structure--*/ file) states band_path and
    # band_path_points_per_segment directly; explicit keyword arguments win.
    band_path, lattice_type, points_per_segment, lattice = band_path_from_input(
        scf_input_path, scf_config, band_path, lattice_type, points_per_segment
    )

    # ── Step 1: SCF ──────────────────────────────────────────────────────────
    # Rewrite the input YAML so the binary sees self_consistent_field (not
    # band_structure) and outdir is absolute (avoids CWD-vs-YAML resolution mismatch).
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
                effective_scf_path,
                q3_binary=binary,
                execution_policy=execution_policy,
                capture_output=capture_output,
            )

        # ── Step 2: Interpolate k-path ────────────────────────────────────────
        kpoints, kpath_distances = interpolate_kpath(
            band_path, lattice_type, points_per_segment=points_per_segment, lattice=lattice
        )
        kpoints_with_weight = kpoints_list_with_uniform_weight(kpoints)

        # ── Step 3: Write NSCF input ──────────────────────────────────────────
        if nscf_input_path is not None:
            nscf_path = Path(nscf_input_path).expanduser().resolve()
            nscf_path.parent.mkdir(parents=True, exist_ok=True)
            cleanup_nscf_path = False
        else:
            # A temporary file, per this parameter's own docstring -- NOT a predictable
            # sibling path (scf_input_path.stem + "_nscf.yml"), which silently overwrote
            # this example directory's own committed nc_nscf.yml reference file on every
            # test run (same name, by construction, for any example whose SCF input is
            # named nc.yml).
            with tempfile.NamedTemporaryFile(
                suffix="_nscf.yml", dir=scf_input_path.parent, delete=False
            ) as tmp_nscf_file:
                nscf_path = Path(tmp_nscf_file.name)
            cleanup_nscf_path = True

        # Use the rewritten SCF config as template so NSCF inherits the
        # absolute outdir and finds the SCF checkpoint.
        write_nscf_input(
            nscf_path,
            scf_input_path=effective_scf_path,
            kpoints_list=kpoints_with_weight,
            number_of_bands=number_of_bands,
        )

    finally:
        effective_scf_path.unlink(missing_ok=True)

    # ── Step 4: NSCF run ──────────────────────────────────────────────────────
    try:
        nscf_results = run_q3(
            nscf_path,
            q3_binary=binary,
            execution_policy=execution_policy,
            capture_output=capture_output,
        )
    finally:
        if cleanup_nscf_path:
            nscf_path.unlink(missing_ok=True)

    # ── Step 5: Format band structure ─────────────────────────────────────────
    # Always prefer the SCF Fermi energy.  The NSCF step runs smearing bisection
    # on the band-path k-points, which are not a proper Monkhorst-Pack grid.
    # For metals this gives a physically wrong E_F; for insulators both agree,
    # but the SCF value is always the authoritative one.
    # Fall back to the NSCF value only when skip_scf=True (no SCF results).
    fermi_energy = (
        scf_results if scf_results is not None else nscf_results
    ).properties.fermi_energy
    fermi_energy_ry = fermi_energy.value if fermi_energy is not None else None

    band_structure = extract_band_structure(
        nscf_results,
        kpoints=kpoints,
        kpath_distances=kpath_distances,
        fermi_energy_ry=fermi_energy_ry,
    )

    return {
        "scf_results": scf_results,
        "nscf_results": nscf_results,
        "band_structure": band_structure,
    }
