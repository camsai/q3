"""q3.workflows.dos — Density of States workflow.

Computes the DOS from SCF eigenvalues using Gaussian broadening.  Reads
the eigenvalues directly from an existing ``results.json`` (no re-run needed
if SCF has already completed), or runs SCF first if requested.

Example::

    from q3.workflows import run_dos_workflow

    result = run_dos_workflow(
        scf_input_path="examples/total_energy--semiconductor--silicon/nc.yml",
        broadening_ry=0.01,
        energy_steps=500,
        q3_binary="./build/q3",
    )
    # result["dos"]["energies_ry"]  → list of energy grid points
    # result["dos"]["dos"]          → density of states values
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from q3.binary import ExecutionPolicy, run_q3
from q3.io.output_writer import extract_dos


def run_dos_workflow(
    scf_input_path: str | Path,
    *,
    broadening_ry: float | None = None,
    energy_min_ry: float | None = None,
    energy_max_ry: float | None = None,
    energy_steps: int | None = None,
    q3_binary: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    skip_scf: bool = False,
    scf_results: dict | None = None,
    capture_output: bool = True,
) -> dict:
    """Compute the density of states.

    Parameters
    ----------
    scf_input_path:
        Path to a Q3 YAML input file.  If *skip_scf* is ``False``, the SCF
        is run first; otherwise the binary is not called and *scf_results*
        must be provided.

        When *skip_scf* is ``False``, this file may also carry
        ``control.dos_broadening``/``dos_energy_min``/``dos_energy_max``/
        ``dos_energy_steps`` (the ``calculation: density_of_states`` YAML
        convention every ``examples/density_of_states--*/`` file uses) --
        when the matching keyword argument is not passed explicitly, it
        defaults to whatever this file states. The C++ engine itself never
        reads ``calculation: density_of_states`` directly (see
        ``create_flavor()`` in ``src/layer_4_computation/flavors/factory.hpp``
        -- multi-step workflows are Python-orchestrated by design); this is
        what actually consumes those fields. Ignored when *skip_scf* is
        ``True`` (no file is read in that mode; unpassed arguments keep
        their plain defaults below), matching this parameter's existing
        "not read at all" behavior in that mode.
    broadening_ry:
        Gaussian broadening σ in Rydberg. Defaults to *scf_input_path*'s own
        ``control.dos_broadening`` if present, else 0.01.
    energy_min_ry, energy_max_ry:
        Energy window in Rydberg. Defaults to *scf_input_path*'s own
        ``control.dos_energy_min``/``dos_energy_max`` if present, else ±5σ
        beyond the band edges.
    energy_steps:
        Number of points in the DOS energy grid. Defaults to
        *scf_input_path*'s own ``control.dos_energy_steps`` if present,
        else 1000.
    q3_binary:
        Path to the ``q3`` binary.  Auto-detected when ``None``.
    execution_policy:
        The execution policy every engine run of the workflow starts the
        binary with: ranks, k-point pools, the launcher (see
        :class:`q3.binary.ExecutionPolicy`).  ``None`` runs one process with
        the binary's defaults.
    skip_scf:
        When ``True``, skip the SCF run and use *scf_results* directly.
    scf_results:
        Pre-computed SCF results dict.  Required when *skip_scf* is ``True``.
    capture_output:
        Pass ``False`` to stream binary output to the terminal.

    Returns
    -------
    dict
        Contains:
        - ``"scf_results"`` — dict from SCF run (or the supplied *scf_results*).
        - ``"dos"`` — DOS dict with keys ``energies_ry``, ``dos``,
          ``fermi_energy_ry``.
    """
    if skip_scf:
        if scf_results is None:
            raise ValueError("skip_scf=True requires scf_results to be provided.")
        results = scf_results
        input_control: dict = {}
    else:
        scf_input_path = Path(scf_input_path).expanduser().resolve()

        # The DOS example YAMLs use calculation=density_of_states as a label,
        # but the C++ binary only understands self_consistent_field.  Rewrite to
        # SCF in a temporary file so the binary runs correctly.
        with open(scf_input_path) as file_handle:
            config = yaml.safe_load(file_handle)

        # ── Defaults from scf_input_path itself ────────────────────────────
        # The calculation: density_of_states YAML convention (every
        # examples/density_of_states--*/ file) states dos_broadening and
        # dos_energy_steps directly -- read them here (below) so those files
        # are actually runnable through this workflow with no other
        # arguments, rather than carrying fields nothing consumes. Explicit
        # keyword arguments always win over the file. Matches the treatment
        # applied to run_band_structure_workflow's band_path/
        # band_path_points_per_segment.
        input_control = config.get("control", {})

        if input_control.get("calculation") == "density_of_states":
            config.setdefault("control", {})["calculation"] = "self_consistent_field"

        # Make outdir absolute so the binary and _find_results_json agree
        # regardless of the subprocess working directory.
        outdir = input_control.get("outdir", "./output")
        if not Path(outdir).is_absolute():
            config["control"]["outdir"] = str((scf_input_path.parent / outdir).resolve())

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yml",
            dir=scf_input_path.parent,
            delete=False,
        ) as tmp_file:
            yaml.dump(config, tmp_file, default_flow_style=False)
            effective_input_path = Path(tmp_file.name)

        try:
            results = run_q3(
                effective_input_path,
                q3_binary=q3_binary,
                execution_policy=execution_policy,
                capture_output=capture_output,
            )
        finally:
            if effective_input_path != scf_input_path:
                effective_input_path.unlink(missing_ok=True)

    if broadening_ry is None:
        broadening_ry = input_control.get("dos_broadening", 0.01)
    if energy_steps is None:
        energy_steps = input_control.get("dos_energy_steps", 1000)
    if energy_min_ry is None:
        energy_min_ry = input_control.get("dos_energy_min")
    if energy_max_ry is None:
        energy_max_ry = input_control.get("dos_energy_max")

    dos = extract_dos(
        results,
        energy_min_ry=energy_min_ry,
        energy_max_ry=energy_max_ry,
        energy_steps=energy_steps,
        broadening_ry=broadening_ry,
    )

    return {
        "scf_results": results,
        "dos": dos,
    }
