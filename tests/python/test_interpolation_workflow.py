"""tests/python/test_interpolation_workflow.py

Unit tests for q3.workflows.interpolation's manifest writing, argument forming
and error paths, against a stub `q3_utility` rather than the real one.

A stub is the point: these tests are about the workflow's own contract --
which manifests it writes, which arguments it forms, which inputs it rejects
before spending anything, and how it reports a failure -- and they must run
in milliseconds without a build. The physics is validated end to end in
tests/validation/test_wannier_interpolation.py; nothing here says an
interpolation is right.
"""

from __future__ import annotations

import json
import shlex
import stat
import sys
from pathlib import Path

import pytest
import yaml
from q3.errors import Q3RuntimeError
from q3.io.interpolation.results.classes import InterpolationResults
from q3.io.interpolation_settings import CaptureSettings, EvaluationSettings
from q3.io.results.classes import eigenvalue_arrays
from q3.structure.lattice import reciprocal_lattice_vectors_inverse_angstrom
from q3.workflows.interpolation import (
    evaluate_interpolation,
    full_grid_kpoints,
    import_wannier90_record,
    load_material,
    mesh_for_spacing,
    run_interpolation_workflow,
    write_interpolation_evaluation_input,
    write_interpolation_nscf_input,
)

# The examples' silicon: the ESSE shape, where "a" is the PRIMITIVE vector length.
SILICON_MATERIAL = {
    "name": "Si",
    "lattice": {
        "type": "FCC",
        "a": 3.866976,
        "b": 3.866975,
        "c": 3.866975,
        "alpha": 60,
        "beta": 60,
        "gamma": 60,
        "units": {"length": "angstrom", "angle": "degree"},
    },
    "basis": {
        "units": "crystal",
        "elements": [{"id": 0, "value": "Si"}, {"id": 1, "value": "Si"}],
        "coordinates": [
            {"id": 0, "value": [0.875, 0.875, 0.875]},
            {"id": 1, "value": [0.125, 0.125, 0.125]},
        ],
    },
}

# The rattled Si3C alloy: explicit vectors, no symmetry, twice as long along a.
ALLOY_MATERIAL = {
    "name": "Si3C",
    "lattice": {
        "type": "TRI",
        "vectors": {
            "a": [0.054437, 5.362044, 5.579792],
            "b": [2.830724, 0.217748, 2.830724],
            "c": [2.803505, 2.504102, -0.136093],
            "units": "angstrom",
        },
    },
    "basis": {
        "units": "crystal",
        "elements": [{"id": 0, "value": "Si"}],
        "coordinates": [{"id": 0, "value": [0.0, 0.0, 0.0]}],
    },
}


def _write_scf_manifest(directory: Path) -> Path:
    """A minimal SCF manifest with relative file references, as the examples have."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "material": "../data/materials/silicon.json",
        "model": {"type": "dft", "subtype": "gga", "functional": "pbe"},
        "method": {
            "type": "pseudopotential",
            "subtype": "norm-conserving",
            "cutoffs": {"wavefunction": 48.0, "density": 192.0},
            "pseudopotentials": {"Si": "../data/pseudopotentials/nc/Si.upf"},
            "scf": {
                "convergence_threshold": 1.0e-10,
                "number_of_bands": 4,
                "kgrid": {"dimensions": [4, 4, 4], "shifts": [0, 0, 0]},
            },
        },
        "control": {"calculation": "scf", "outdir": "./output", "prefix": "si"},
    }
    path = directory / "nc_scf.yml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def _write_stub_utility(path: Path) -> Path:
    """A fake q3_utility that records its arguments and, for `interpolate`, writes results.

    Runs under this test's own interpreter so it can read the manifest with
    PyYAML; the recorded arguments land beside the stub as `<stub>.calls.json`.
    A one-line shell wrapper execs the interpreter, since a shebang cannot
    carry an interpreter path with a space in it (a home directory with one,
    say) and older Linux kernels truncate long shebang lines.
    """
    body_path = str(path) + ".py"
    wrapper = f"""#!/bin/sh
exec {shlex.quote(sys.executable)} {shlex.quote(body_path)} "$@"
"""
    script = f"""import json, pathlib, sys
import yaml
arguments = sys.argv[1:]
pathlib.Path({str(path) + ".calls.json"!r}).write_text(json.dumps(arguments))
if arguments and arguments[0] == "interpolate":
    with open(arguments[1]) as handle:
        control = yaml.safe_load(handle)["control"]
    outdir = pathlib.Path(control["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    results = {{
        "arguments": arguments,
        "properties": {{
            "eigenvalues_at_kpoints": {{"name": "eigenvalues_at_kpoints", "units": "Ry", "values": [
                {{"kpoint": row[:3], "eigenvalues": [{{"energies": [-0.5, 0.25]}}]}}
                for row in control.get("kpoints_list", [])
            ]}},
            "fermi_energy": {{"name": "fermi_energy", "units": "Ry", "value": 0.125}},
        }},
    }}
    if "dos_mesh" in control["interpolation"]:
        results["properties"]["density_of_states"] = {{
            "name": "density_of_states",
            "xAxis": {{"label": "energy", "units": "Ry"}},
            "yAxis": {{"label": "density of states", "units": "states/unitcell"}},
            "xDataArray": [0.0, 0.1],
            "yDataSeries": [[1.0, 2.0]],
            "legend": [],
        }}
    (outdir / (control["prefix"] + "_interpolation_results.json")).write_text(json.dumps(results))
"""
    Path(body_path).write_text(script)
    path.write_text(wrapper)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _recorded_arguments(stub: Path) -> list[str]:
    return json.loads(Path(str(stub) + ".calls.json").read_text())


def _write_stub_engine(path: Path) -> Path:
    """A fake q3 that writes a plausible results file and keeps every manifest it was given.

    Enough for the workflow to run its three steps: `converged` so run_q3 does
    not raise, a Fermi level, and one eigenvalue row per requested k-point. The
    manifests land in `<stub>.manifests.json`, in order, so a test can read the
    NSCF manifest the workflow wrote and deleted.
    """
    body_path = str(path) + ".py"
    wrapper = f"""#!/bin/sh
exec {shlex.quote(sys.executable)} {shlex.quote(body_path)} "$@"
"""
    script = f"""import json, pathlib, sys
import yaml
manifest_path = pathlib.Path(sys.argv[1])
with open(manifest_path) as handle:
    manifest = yaml.safe_load(handle)
seen_path = pathlib.Path({str(path) + ".manifests.json"!r})
seen = json.loads(seen_path.read_text()) if seen_path.exists() else []
seen.append(manifest)
seen_path.write_text(json.dumps(seen))
control = manifest["control"]
outdir = pathlib.Path(control.get("outdir", "."))
if not outdir.is_absolute():
    outdir = manifest_path.parent / outdir
outdir.mkdir(parents=True, exist_ok=True)
kpoints = control.get("kpoints_list") or [[0.0, 0.0, 0.0, 2.0]]
results = {{
    "runtime_log": {{"converged": True}},
    "properties": {{
        "records": [
            {{"name": "fermi_energy", "units": "Ry", "value": 0.1}},
            {{"name": "eigenvalues_at_kpoints", "units": "Ry", "values": [
                {{"kpoint": row[:3], "weight": row[3], "eigenvalues": [{{"energies": [-0.5, 0.25]}}]}}
                for row in kpoints
            ]}},
        ],
    }},
}}
(outdir / (control.get("prefix", "q3") + "_results.json")).write_text(json.dumps(results))
"""
    Path(body_path).write_text(script)
    path.write_text(wrapper)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _seen_manifests(stub: Path) -> list[dict]:
    return json.loads(Path(str(stub) + ".manifests.json").read_text())


def _norm(row) -> float:
    return sum(value * value for value in row) ** 0.5


def test_the_mesh_rule_reproduces_the_silicon_examples_at_the_spacings_that_name_them():
    """
    Silicon's primitive reciprocal vectors are 1.99 per angstrom long (2 pi
    included), so the examples' 4x4x4, 6x6x6 and 8x8x8 are the spacings 0.5,
    0.34 and 0.25, and the published 0.2 gives 10x10x10 -- the mesh the
    parent plan quoted for silicon before any rule existed.
    """
    silicon = SILICON_MATERIAL["lattice"]
    for row in reciprocal_lattice_vectors_inverse_angstrom(silicon):
        assert _norm(row) == pytest.approx(1.9898, abs=1e-3)
    assert mesh_for_spacing(silicon, 0.5) == (4, 4, 4)
    assert mesh_for_spacing(silicon, 0.34) == (6, 6, 6)
    assert mesh_for_spacing(silicon, 0.25) == (8, 8, 8)
    assert mesh_for_spacing(silicon) == (10, 10, 10)
    # 1.99 / 0.4 = 4.97 rounds up to five, and five up to the next even count.
    assert mesh_for_spacing(silicon, 0.4) == (6, 6, 6)


def test_the_mesh_rule_on_a_triclinic_cell_is_even_anisotropic_and_never_two_points():
    """
    The alloy is a doubled cell with no symmetry: its long axis has the short
    reciprocal vector and gets fewer points, every count is even so a retry can
    halve it, and no axis ever gets the two points the localiser refuses -- a
    cell with a 40 angstrom axis still gets four along it, not one.
    """
    grid = mesh_for_spacing(ALLOY_MATERIAL["lattice"])
    assert all(count % 2 == 0 and count >= 4 for count in grid), grid
    assert grid[0] < grid[1] and grid[0] < grid[2], f"the doubled axis is coarser: {grid}"

    slab = {"vectors": {"a": [3.0, 0, 0], "b": [0, 3.0, 0], "c": [0, 0, 40.0], "units": "angstrom"}}
    assert mesh_for_spacing(slab) == (12, 12, 4)
    with pytest.raises(ValueError, match="positive"):
        mesh_for_spacing(ALLOY_MATERIAL["lattice"], 0.0)


def test_load_material_reads_a_path_relative_to_the_manifest_or_an_inline_map(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "si.json").write_text(json.dumps(SILICON_MATERIAL))
    assert load_material({"material": "data/si.json"}, tmp_path)["name"] == "Si"
    assert load_material({"material": ALLOY_MATERIAL}, tmp_path)["name"] == "Si3C"
    with pytest.raises(ValueError, match="material"):
        load_material({"material": 3}, tmp_path)


def test_the_workflow_chooses_the_grid_from_the_material_when_none_is_given(tmp_path):
    """
    The capture manifest the stub engine receives carries the 10x10x10 the
    spacing rule gives silicon -- a thousand k-points it never had to be told
    -- and the choice comes back in the result so a caller can see it.
    """
    materials = tmp_path / "data" / "materials"
    materials.mkdir(parents=True)
    (materials / "silicon.json").write_text(json.dumps(SILICON_MATERIAL))
    scf = _write_scf_manifest(tmp_path / "run")
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(tmp_path / "stub_util")

    result = run_interpolation_workflow(
        scf, kpoints=[(0.0, 0.0, 0.0)], q3_binary=engine, q3_utility_binary=utility
    )
    assert result["grid"] == [10, 10, 10]
    scf_manifest, nscf_manifest = _seen_manifests(engine)
    assert scf_manifest["control"]["calculation"] == "self_consistent_field"
    assert nscf_manifest["control"]["interpolation"]["grid"] == [10, 10, 10]
    assert len(nscf_manifest["control"]["kpoints_list"]) == 1000

    coarse = run_interpolation_workflow(
        scf,
        kpoint_spacing_inverse_angstrom=0.5,
        kpoints=[(0.0, 0.0, 0.0)],
        q3_binary=engine,
        q3_utility_binary=utility,
        skip_scf=True,
    )
    assert coarse["grid"] == [4, 4, 4]
    assert coarse["scf_results"] is None


def test_full_grid_kpoints_covers_the_grid_with_uniform_weight():
    rows = full_grid_kpoints((2, 3, 1))
    assert len(rows) == 6
    # Weights sum to 2, the engine's own convention: KPointsGrid embeds the spin
    # degeneracy in the weight, and an explicit kpoints_list is taken verbatim.
    assert sum(row[3] for row in rows) == pytest.approx(2.0)
    assert rows[0] == (0.0, 0.0, 0.0, pytest.approx(2 / 6))
    assert rows[-1] == (0.5, 2 / 3, 0.0, pytest.approx(2 / 6))
    assert len({row[:3] for row in rows}) == 6, "every grid point exactly once"


def test_nscf_input_carries_the_full_grid_and_the_interpolation_block(tmp_path):
    scf = _write_scf_manifest(tmp_path)
    path = write_interpolation_nscf_input(
        tmp_path / "nscf.yml",
        scf_input_path=scf,
        settings=CaptureSettings(
            grid=[2, 2, 2],
            projection_centers=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
            projection_decay_bohr=1.0,
            export_formats=["wannier90"],
        ),
    )
    manifest = yaml.safe_load(path.read_text())
    control = manifest["control"]
    assert control["calculation"] == "nscf"
    assert len(control["kpoints_list"]) == 8
    assert sum(row[3] for row in control["kpoints_list"]) == pytest.approx(2.0)
    assert control["interpolation"] == {
        "method": "wannier",
        "grid": [2, 2, 2],
        "projection_centers": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "projection_decay_bohr": 1.0,
        "export_formats": ["wannier90"],
    }
    assert control["prefix"] == "si", "the prefix is kept so the run finds the SCF checkpoint"
    assert "kgrid" not in manifest["method"]["scf"]

    # The pseudo-atomic set instead, with some of its shells named.
    path = write_interpolation_nscf_input(
        tmp_path / "nscf_shells.yml",
        scf_input_path=scf,
        settings=CaptureSettings(grid=[2, 2, 2], trial_shells={"Si": [0, 2]}),
    )
    block = yaml.safe_load(path.read_text())["control"]["interpolation"]
    assert block == {"method": "wannier", "grid": [2, 2, 2], "trial_shells": {"Si": [0, 2]}}


def test_nscf_input_from_a_band_structure_manifest_carries_no_band_path(tmp_path):
    """The capture grid is not a path: the template's band path must not reach it."""
    scf = _write_scf_manifest(tmp_path)
    manifest = yaml.safe_load(scf.read_text())
    manifest["control"]["band_path"] = ["G", "X", "W", "L", "G"]
    manifest["control"]["band_path_points_per_segment"] = 2
    scf.write_text(yaml.safe_dump(manifest, sort_keys=False))

    path = write_interpolation_nscf_input(
        tmp_path / "nscf.yml",
        scf_input_path=scf,
        settings=CaptureSettings(grid=[2, 2, 2]),
    )
    control = yaml.safe_load(path.read_text())["control"]
    assert len(control["kpoints_list"]) == 8
    assert "band_path" not in control
    assert "band_path_points_per_segment" not in control


def test_evaluation_input_names_the_record_and_makes_the_paths_absolute(tmp_path):
    scf = _write_scf_manifest(tmp_path)
    path = write_interpolation_evaluation_input(
        tmp_path / "elsewhere" / "evaluate.yml",
        template_input_path=scf,
        outdir=tmp_path / "output",
        prefix="si",
        settings=EvaluationSettings(dos_mesh=[4, 4, 4]),
        kpoints=[(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)],
        dos_broadening_ry=0.01,
    )
    manifest = yaml.safe_load(path.read_text())
    control = manifest["control"]
    assert control["calculation"] == "interpolation"
    assert control["outdir"] == str((tmp_path / "output").resolve())
    assert control["prefix"] == "si"
    assert control["interpolation"] == {"method": "wannier", "dos_mesh": [4, 4, 4]}
    assert control["dos_broadening"] == 0.01
    assert control["kpoints_list"] == [[0.0, 0.0, 0.0, 1.0], [0.25, 0.0, 0.0, 1.0]]
    # Written into a different directory than the template, so its relative
    # references would have pointed at nothing.
    assert Path(manifest["material"]).is_absolute()
    assert Path(manifest["method"]["pseudopotentials"]["Si"]).is_absolute()
    assert "kgrid" not in manifest["method"]["scf"]


def test_evaluate_runs_the_utility_and_reads_its_results(tmp_path):
    scf = _write_scf_manifest(tmp_path)
    stub = _write_stub_utility(tmp_path / "stub_util")
    manifest = write_interpolation_evaluation_input(
        tmp_path / "evaluate.yml",
        template_input_path=scf,
        outdir=tmp_path / "output",
        prefix="si",
        settings=EvaluationSettings(dos_mesh=[2, 2, 2]),
        kpoints=[(0.0, 0.0, 0.0)],
    )
    results = evaluate_interpolation(manifest, q3_utility_binary=stub)
    assert _recorded_arguments(stub) == ["interpolate", str(manifest)]
    model = InterpolationResults.model_validate(results)
    assert eigenvalue_arrays(model.properties.eigenvalues_at_kpoints)[0] == [[[-0.5, 0.25]]]
    assert model.properties.density_of_states.xDataArray == [0.0, 0.1]
    assert model.properties.density_of_states.yDataSeries == [[1.0, 2.0]]


def test_import_forms_the_convert_arguments_and_names_the_record(tmp_path):
    stub = _write_stub_utility(tmp_path / "stub_util")
    record = import_wannier90_record(
        tmp_path / "si", (4, 4, 4), outdir=tmp_path / "imported", q3_utility_binary=stub
    )
    assert record == tmp_path / "imported" / "si.save" / "interpolation.q3ip"
    arguments = _recorded_arguments(stub)
    assert arguments[:4] == ["convert", "--from", "wannier90", "--seedname"]
    assert arguments[arguments.index("--grid") + 1 : arguments.index("--grid") + 4] == [
        "4",
        "4",
        "4",
    ]
    assert arguments[arguments.index("--prefix") + 1] == "si"
    assert arguments[arguments.index("--outdir") + 1] == str((tmp_path / "imported").resolve())


def test_rejects_an_invalid_grid_before_running_anything(tmp_path):
    scf = _write_scf_manifest(tmp_path)
    with pytest.raises(ValueError, match="three positive integers"):
        write_interpolation_nscf_input(
            tmp_path / "nscf.yml", scf_input_path=scf, settings=CaptureSettings(grid=[4, 0, 4])
        )
    with pytest.raises(ValueError, match="three positive integers"):
        import_wannier90_record("si", (4, 4), q3_utility_binary=tmp_path / "unused")


def test_the_evaluation_input_carries_the_disentanglement_request(tmp_path):
    """
    The thresholds are read when the model is evaluated, not when the record is
    captured, so this is the manifest they have to reach. Without them the
    workflow cannot ask for disentanglement at all and an entangled metal is
    reachable only by hand-written YAML.
    """
    scf = _write_scf_manifest(tmp_path)
    path = write_interpolation_evaluation_input(
        tmp_path / "evaluate.yml",
        template_input_path=scf,
        outdir=tmp_path,
        prefix="si",
        settings=EvaluationSettings(
            disentanglement="projectability",
            disentanglement_convergence_tolerance=7e-6,
            projectability_maximum=0.9,
            frozen_window_above_fermi_ev=-5.0,
        ),
    )
    block = yaml.safe_load(path.read_text())["control"]["interpolation"]
    assert block["disentanglement"] == "projectability"
    assert block["disentanglement_convergence_tolerance"] == 7e-6
    assert block["projectability_maximum"] == 0.9
    assert block["frozen_window_above_fermi_ev"] == -5.0
    assert "projectability_minimum" not in block, "an unset threshold keeps the engine's default"
    assert "disentanglement_maximum_iterations" not in block


def test_reports_a_utility_failure_with_its_output(tmp_path):
    scf = _write_scf_manifest(tmp_path)
    failing = tmp_path / "failing_util"
    failing.write_text("#!/bin/sh\necho 'something went wrong' >&2\nexit 3\n")
    failing.chmod(failing.stat().st_mode | stat.S_IEXEC)
    manifest = write_interpolation_evaluation_input(
        tmp_path / "evaluate.yml", template_input_path=scf, outdir=tmp_path, prefix="si"
    )
    with pytest.raises(Q3RuntimeError, match="code 3") as error:
        evaluate_interpolation(manifest, q3_utility_binary=failing)
    assert "something went wrong" in error.value.stderr


def test_rejects_a_missing_utility_binary(tmp_path):
    scf = _write_scf_manifest(tmp_path)
    manifest = write_interpolation_evaluation_input(
        tmp_path / "evaluate.yml", template_input_path=scf, outdir=tmp_path, prefix="si"
    )
    with pytest.raises(FileNotFoundError):
        evaluate_interpolation(manifest, q3_utility_binary=tmp_path / "does_not_exist")
