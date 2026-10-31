"""tests/python/test_wannier_workflow.py

The automatic Wannier workflow's own decisions, against stub binaries whose
answers are scripted: which mesh and band window it chooses without being
told, which rung fires on which outcome, what it writes back into the results
file, and when it gives up. No DFT runs here and nothing says a model is
right; the physics is in tests/validation. The point is that the ladder does
exactly what the plan says on every outcome it can meet, and that a failure
is never shipped as a success.
"""

from __future__ import annotations

import json
import shlex
import stat
import sys
from pathlib import Path

import pytest
import yaml
from q3.constants import RYDBERG_TO_ELECTRONVOLT
from q3.errors import Q3Error
from q3.workflows.wannier import (
    ACCEPT_ETA_EV,
    PROJECTABILITY_MAXIMUM_LADDER,
    TrialOrbitals,
    band_window,
    last_disentanglement_spreads,
    loosened_disentanglement_tolerance,
    off_mesh_reference_kpoints,
    pseudo_atomic_orbitals_in,
    pseudo_atomic_shells,
    run_automatic_wannier_workflow,
    trial_shells,
)

SILICON_MATERIAL = {
    "name": "Si",
    "lattice": {
        "type": "FCC",
        "a": 3.866976,
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

#: A silicon-like dataset: a 3S and a 3P shell, so four orbitals per atom.
FAKE_UPF = """<UPF version="2.0.1">
<PP_HEADER element="Si" number_of_wfc="2"/>
<PP_PSWFC>
<PP_CHI.1
type="real"
size="4"
columns="4"
index="1"
occupation=" 2.000"
pseudo_energy="   -0.7947291737E+00"
label="3S"
l="0" >
0.0 0.0 0.0 0.0
</PP_CHI.1>
<PP_CHI.2 type="real" size="4" columns="4" index="2" occupation=" 2.000" label="3P" l="1" >
0.0 0.0 0.0 0.0
</PP_CHI.2>
</PP_PSWFC>
</UPF>
"""

#: The same silicon as a PAW file writes it: no labels, and an empty second s
#: and p beside the occupied ones, which is sixteen trial orbitals for two atoms
#: when every shell counts.
FAKE_PAW_UPF = """<UPF version="2.0.1">
<PP_HEADER
    element="Si"
    z_valence=" 4.000"
    number_of_wfc="     4" />
<PP_PSWFC>
<PP_CHI.1 type="real" size="4" l="0" occupation=" 2.0000" columns="4">
0.0 0.0 0.0 0.0
</PP_CHI.1>
<PP_CHI.2 type="real" size="4" l="0" occupation=" 0.0000" columns="4">
0.0 0.0 0.0 0.0
</PP_CHI.2>
<PP_CHI.3 type="real" size="4" l="1" occupation=" 2.0000" columns="4">
0.0 0.0 0.0 0.0
</PP_CHI.3>
<PP_CHI.4 type="real" size="4" l="1" occupation=" 0.0000" columns="4">
0.0 0.0 0.0 0.0
</PP_CHI.4>
</PP_PSWFC>
</UPF>
"""

#: The datasets the examples and the campaign run on.
DATASETS = Path(__file__).resolve().parents[2] / "examples" / "data" / "pseudopotentials"


def _unconverged(*spreads: float) -> str:
    """The engine's words for a disentanglement that ran out of iterations, with these spreads."""
    return (
        "disentanglement did not converge in 4000 iterations. Last gauge-invariant spreads "
        f"(bohr^2): {', '.join(f'{spread:.6f}' for spread in spreads)}. A subspace that has "
        "not converged is not returned as if it had."
    )


def _failure(prose: str, diagnostic: str, spreads: tuple[float, ...] = ()) -> dict:
    """A scripted refusal as the utility prints it: the prose, the code, the spreads if it has any."""
    return {"error": prose, "diagnostic": diagnostic, "spreads": list(spreads)}


def _unconverged_failure(*spreads: float) -> dict:
    """A disentanglement out of iterations, scripted: the code and the spreads route, not the words."""
    return _failure(_unconverged(*spreads), "disentanglement_unconverged", spreads)


def _unconverged_text(*spreads: float) -> str:
    """The same failure as the text a run leaves behind, its own lines after the prose."""
    return (
        f"Error: {_unconverged(*spreads)}\n"
        "diagnostic: disentanglement_unconverged\n"
        f"diagnostic_spreads_bohr_squared: {', '.join(f'{spread:.6f}' for spread in spreads)}\n"
    )


#: Diamond PAW's failure on the campaign: still falling 1.7e-5 bohr^2 an iteration.
DIAMOND_SPREADS = (32.869717, 32.869701, 32.869684, 32.869668, 32.869651)
DIAMOND_UNCONVERGED = _unconverged_failure(*DIAMOND_SPREADS)

#: The eigenvalue both stubs agree on, in rydberg: band n at fractional k.
EIGENVALUE_FORMULA = (
    "lambda n, k: -0.8 + 0.06 * n + 0.02 * (math.cos(2 * math.pi * k[0]) + "
    "math.cos(2 * math.pi * k[1]) + math.cos(2 * math.pi * k[2]))"
)


def _write_case(directory: Path, dataset: str = FAKE_UPF) -> Path:
    """An SCF manifest with its material and a fake dataset beside it."""
    (directory / "data" / "materials").mkdir(parents=True)
    (directory / "data" / "pseudopotentials").mkdir(parents=True)
    (directory / "data" / "materials" / "silicon.json").write_text(json.dumps(SILICON_MATERIAL))
    (directory / "data" / "pseudopotentials" / "Si.upf").write_text(dataset)
    (directory / "run").mkdir()
    manifest = {
        "material": "../data/materials/silicon.json",
        "model": {"type": "dft", "subtype": "gga", "functional": "pbe"},
        "method": {
            "type": "pseudopotential",
            "subtype": "norm-conserving",
            "cutoffs": {"wavefunction": 48.0, "density": 192.0},
            "pseudopotentials": {"Si": "../data/pseudopotentials/Si.upf"},
            "scf": {"number_of_bands": 8, "kgrid": {"dimensions": [4, 4, 4], "shifts": [0, 0, 0]}},
        },
        "control": {"calculation": "scf", "outdir": "./output", "prefix": "si"},
    }
    path = directory / "run" / "nc_scf.yml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def _wrapper(path: Path, body_path: str) -> None:
    path.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(body_path)} "$@"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_stub_engine(path: Path) -> Path:
    """A fake q3: writes eigenvalues from the shared formula, keeps every manifest."""
    body_path = str(path) + ".py"
    script = f"""import json, math, pathlib, sys
import yaml
eigenvalue = {EIGENVALUE_FORMULA}
manifest_path = pathlib.Path(sys.argv[-1])
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
bands = int(manifest.get("method", {{}}).get("scf", {{}}).get("number_of_bands", 8))
results = {{
    "runtime_log": {{"converged": True}},
    "properties": {{
        "records": [
            {{"name": "fermi_energy", "units": "Ry", "value": -0.5}},
            {{"name": "eigenvalues_at_kpoints", "units": "Ry", "values": [
                {{"kpoint": row[:3], "weight": row[3],
                  "eigenvalues": [{{"energies": [eigenvalue(n, row[:3]) for n in range(bands)]}}]}}
                for row in kpoints
            ]}},
        ],
    }},
}}
(outdir / (control.get("prefix", "q3") + "_results.json")).write_text(json.dumps(results))
"""
    Path(body_path).write_text(script)
    _wrapper(path, body_path)
    return path


def _write_stub_utility(path: Path, script: list[dict]) -> Path:
    """A fake q3_utility whose `interpolate` answers follow a script, one entry per call.

    An entry with `error` exits 2 with that text on stderr, with `diagnostic`
    the engine's code on the line after it, and with `spreads` the spreads on
    the line after that; otherwise the
    eigenvalues are the shared formula shifted by `offset_ev` (so the band
    distance IS the offset), with the spread block's `localized` and
    `localization_converged` as scripted, and the model's `number_of_bands` when
    the entry gives one -- which is then also how many eigenvalues it writes
    per k-point, as the engine's model has one band per Wannier function and
    not one per captured band. Every manifest is kept.
    """
    body_path = str(path) + ".py"
    (Path(str(path) + ".script.json")).write_text(json.dumps(script))
    body = f"""import json, math, pathlib, sys
import yaml
eigenvalue = {EIGENVALUE_FORMULA}
arguments = sys.argv[1:]
script_path = pathlib.Path({str(path) + ".script.json"!r})
script = json.loads(script_path.read_text())
entry = script.pop(0) if script else {{"offset_ev": 0.0}}
script_path.write_text(json.dumps(script))
with open(arguments[1]) as handle:
    manifest = yaml.safe_load(handle)
seen_path = pathlib.Path({str(path) + ".manifests.json"!r})
seen = json.loads(seen_path.read_text()) if seen_path.exists() else []
seen.append(manifest)
seen_path.write_text(json.dumps(seen))
if "error" in entry:
    sys.stderr.write("Error: " + entry["error"] + "\\n")
    if "diagnostic" in entry:
        sys.stderr.write("diagnostic: " + entry["diagnostic"] + "\\n")
    if entry.get("spreads"):
        spreads = ", ".join(format(spread, ".6f") for spread in entry["spreads"])
        sys.stderr.write("diagnostic_spreads_bohr_squared: " + spreads + "\\n")
    sys.exit(2)
control = manifest["control"]
outdir = pathlib.Path(control["outdir"])
outdir.mkdir(parents=True, exist_ok=True)
offset = entry.get("offset_ev", 0.0) / {RYDBERG_TO_ELECTRONVOLT}
captured = int(manifest.get("method", {{}}).get("scf", {{}}).get("number_of_bands", 8))
bands = int(entry.get("number_of_bands", captured))
kpoints = control.get("kpoints_list", [])
results = {{
    "properties": {{
        "eigenvalues_at_kpoints": {{"name": "eigenvalues_at_kpoints", "units": "Ry", "values": [
            {{"kpoint": row[:3],
              "eigenvalues": [{{"energies": [eigenvalue(n, row[:3]) + offset for n in range(bands)]}}]}}
            for row in kpoints
        ]}},
        "fermi_energy": {{"name": "fermi_energy", "units": "Ry", "value": -0.5}},
    }},
    "construction": {{
        "spread": {{
            "localized": entry.get("localized", True),
            "localization_converged": entry.get("converged", True),
        }},
        "disentanglement": {{
            "metallic": bool(entry.get("excluded")),
            "frozen_window_automatic": "frozen_window_above_fermi_ev" not in control["interpolation"],
            "frozen_window_above_fermi_ev": control["interpolation"].get("frozen_window_above_fermi_ev", 2.0),
            "bands_excluded_from_freeze": entry.get("excluded", []),
        }},
    }},
    "quality": {{"hamiltonian_decay": {{"ratio": 0.01}}}},
}}
if "number_of_bands" in entry:
    results["construction"]["number_of_bands"] = entry["number_of_bands"]
(outdir / (control["prefix"] + "_interpolation_results.json")).write_text(json.dumps(results))
"""
    Path(body_path).write_text(body)
    _wrapper(path, body_path)
    return path


def _seen(stub: Path) -> list[dict]:
    return json.loads(Path(str(stub) + ".manifests.json").read_text())


def _interpolation_blocks(stub: Path) -> list[dict]:
    return [manifest["control"]["interpolation"] for manifest in _seen(stub)]


def test_the_dataset_decides_the_trial_orbital_count_and_the_window(tmp_path):
    scf = _write_case(tmp_path)
    upf = tmp_path / "data" / "pseudopotentials" / "Si.upf"
    assert pseudo_atomic_orbitals_in(upf) == 4, "3S and 3P: one s and three p"
    manifest = yaml.safe_load(scf.read_text())
    assert TrialOrbitals.of_manifest(manifest, scf.parent).number_of_wannier_functions == 8, (
        "two atoms"
    )
    assert band_window(8) == 16, "ceil(1.5 * 8) + 4"
    assert band_window(10) == 19

    # A shell with a negative occupation is one the dataset says not to use.
    unused = tmp_path / "unused.upf"
    unused.write_text(
        FAKE_UPF.replace('occupation=" 2.000" label="3P"', 'occupation="-1.0" label="3P"')
    )
    assert pseudo_atomic_orbitals_in(unused) == 1

    with pytest.raises(ValueError, match="PP_CHI"):
        pseudo_atomic_orbitals_in(tmp_path / "data" / "materials" / "silicon.json")

    # A PAW file's empty second s and p repeat the occupied ones: four, not eight.
    paw = tmp_path / "paw.upf"
    paw.write_text(FAKE_PAW_UPF)
    assert pseudo_atomic_orbitals_in(paw) == 4

    # UPF version 1, the ultrasoft datasets' format: one header line per shell.
    version_one = tmp_path / "v1.UPF"
    version_one.write_text(
        "<PP_PSWFC>\n3S    0  2.00          Wavefunction\n 0.0 0.0\n"
        "3P    1  2.00          Wavefunction\n 0.0 0.0\n"
        "3D    2 -1.00          Wavefunction\n 0.0 0.0\n</PP_PSWFC>\n"
    )
    assert pseudo_atomic_orbitals_in(version_one) == 4


@pytest.mark.parametrize(
    ("dataset", "shells", "orbitals"),
    [
        # A norm-conserving file lists its valence and nothing else.
        ("nc/Si.upf", [0, 1], 4),
        ("nc/Cu.upf", [0, 1, 2, 3], 10),
        ("nc/Ga.upf", [0, 1, 2, 3, 4], 13),
        # A PAW file's empty second s and p repeat the occupied 3s and 3p (2s and 2p).
        ("paw/Si.upf", [0, 2], 4),
        ("paw/C.upf", [0, 2], 4),
        # Copper's one occupied p shell is the semicore 3p, so its empty p is 4p and
        # stays; its empty second d repeats the valence 3d and goes.
        ("paw/Cu.upf", [0, 1, 2, 3, 4], 13),
        # Silicon ultrasoft's empty 4s and 4p go; its occupied 3d stays.
        ("us/Si_PBE_USPP.UPF", [0, 2, 4], 9),
        # Copper ultrasoft keeps its 4p for the same reason as copper PAW.
        ("us/cu_pbe_v1.2.uspp.F.UPF", [0, 1, 2, 3, 4], 13),
    ],
)
def test_the_rule_leaves_out_the_empty_channels_an_augmented_dataset_repeats(
    dataset, shells, orbitals
):
    path = DATASETS / dataset
    assert trial_shells(pseudo_atomic_shells(path)) == shells
    assert pseudo_atomic_orbitals_in(path) == orbitals


def test_a_shell_without_a_radial_function_is_neither_kept_nor_a_valence_partner(tmp_path):
    """
    The engine skips a shell whose radial function integrates to nothing, as
    Quantum ESPRESSO does with its "zero norm" warning, so the rule has to as
    well: naming such a shell is refused, and counting it sizes the band window
    for a function never built. Silicon ultrasoft lists its 4s that way, and
    before the rule read it the workflow counted 26 functions where the engine
    built 24.
    """
    silicon = pseudo_atomic_shells(DATASETS / "us/Si_PBE_USPP.UPF")
    assert [shell.zero_norm for shell in silicon] == [False, True, False, False, False]

    dataset = tmp_path / "zero_norm.upf"
    dataset.write_text(
        '<UPF version="2.0.1">\n<PP_MESH>\n<PP_RAB type="real" size="3">0.1 0.1 0.1</PP_RAB>\n'
        "</PP_MESH>\n<PP_PSWFC>\n"
        '<PP_CHI.1 type="real" size="3" l="0" occupation="2.0" label="3S">0.0 0.0 0.0</PP_CHI.1>\n'
        '<PP_CHI.2 type="real" size="3" l="0" occupation="0.0" label="4S">0.5 0.5 0.5</PP_CHI.2>\n'
        '<PP_CHI.3 type="real" size="3" l="1" occupation="2.0" label="3P">0.5 0.5 0.5</PP_CHI.3>\n'
        "</PP_PSWFC>\n</UPF>\n"
    )
    shells = pseudo_atomic_shells(dataset)
    assert [shell.zero_norm for shell in shells] == [True, False, False]
    # The occupied 3s has no radial function, so the empty 4s is the s channel there is.
    assert trial_shells(shells) == [1, 2]


def _label_less_dataset(path: Path, element: str, valence: float, shells: list) -> Path:
    """A PAW-style file: each shell's l and occupation, and the element and charge in the header."""
    chi = "".join(
        f'<PP_CHI.{index + 1} type="real" size="2" l="{angular}" occupation="{occupation:.4f}" '
        f'columns="2">\n0.0 0.0\n</PP_CHI.{index + 1}>\n'
        for index, (angular, occupation) in enumerate(shells)
    )
    path.write_text(
        f'<UPF version="2.0.1">\n<PP_HEADER\n    element="{element}"\n'
        f'    z_valence="{valence}"\n    number_of_wfc="{len(shells)}" />\n'
        f"<PP_PSWFC>\n{chi}</PP_PSWFC>\n</UPF>\n"
    )
    return path


def test_a_label_less_dataset_has_its_shells_counted_up_from_the_core(tmp_path):
    """
    Which of copper's shells is semicore decides whether its empty p shell is a
    second 3p or the 4p its sp band needs, and a PAW file says neither: no
    labels, no energies. Nineteen valence electrons out of twenty-nine leave a
    [Ne] core, so the s shells are 3s and 4s and the occupied p is 3p. Without
    an element to count from, file order is all there is, and it reads 3p as
    the valence p -- the one case the rule cannot settle.
    """
    copper = [(0, 2.0), (0, 1.0), (1, 6.0), (1, 0.0), (2, 10.0), (2, 0.0)]
    path = _label_less_dataset(tmp_path / "Cu.upf", "Cu", 19.0, copper)
    shells = pseudo_atomic_shells(path)
    assert [shell.principal_quantum_number for shell in shells] == [3, 4, 3, 4, 3, 4]
    assert trial_shells(shells) == [0, 1, 2, 3, 4]

    # 54 core electrons are [Kr]4d 5s 5p, which filling by n and then l never
    # lands on; 28 are [Ar]3d, which filling by n + l never lands on.
    cesium = _label_less_dataset(tmp_path / "Cs.upf", "Cs", 1.0, [(0, 1.0), (1, 0.0)])
    assert [shell.principal_quantum_number for shell in pseudo_atomic_shells(cesium)] == [6, 6]
    gallium = _label_less_dataset(tmp_path / "Ga.upf", "Ga", 3.0, [(0, 2.0), (1, 1.0)])
    assert [shell.principal_quantum_number for shell in pseudo_atomic_shells(gallium)] == [4, 4]

    unknown = _label_less_dataset(tmp_path / "X.upf", "Xx", 19.0, copper)
    shells = pseudo_atomic_shells(unknown)
    assert all(shell.principal_quantum_number is None for shell in shells)
    assert trial_shells(shells) == [0, 1, 2, 4]


def test_off_mesh_reference_points_sit_half_a_step_from_every_mesh_point():
    for grid in ((4, 4, 4), (10, 10, 10), (4, 8, 8), (6, 6, 12)):
        points = off_mesh_reference_kpoints(grid)
        assert 0 < len(points) <= 125
        for point in points:
            for coordinate, count in zip(point, grid, strict=True):
                assert (coordinate * count) % 1.0 == pytest.approx(0.5)
    assert len(off_mesh_reference_kpoints((4, 4, 4))) == 64
    assert len(off_mesh_reference_kpoints((10, 10, 10))) == 125


def test_the_tolerance_a_disentanglement_that_ran_out_of_iterations_calls_for():
    """
    A margin above the change among the last spreads the failure printed: the
    same deterministic iteration had met that tolerance by the time it gave up,
    so a retry under it stops no later. Diamond PAW's spreads were still
    falling; copper PAW's had stopped moving at the six decimals printed, where
    the print resolution is the change; silicon ultrasoft's fell ten times
    faster than diamond's and still sit under the cap.
    """
    diamond = _unconverged_text(*DIAMOND_SPREADS)
    assert last_disentanglement_spreads(diamond) == list(DIAMOND_SPREADS)
    assert loosened_disentanglement_tolerance(diamond) == 7e-6
    assert loosened_disentanglement_tolerance(_unconverged_text(*[20.683824] * 5)) == 2e-7
    silicon = _unconverged_text(62.956888, 62.956745, 62.956602, 62.956460, 62.956317)
    assert loosened_disentanglement_tolerance(silicon) == 3e-5

    # Nothing looser than the tolerance that failed, nothing for spreads jumping
    # by percent, and nothing without a spreads line to read -- the sentence is
    # for a reader, and is not parsed.
    assert loosened_disentanglement_tolerance(diamond, 7e-6) is None
    assert (
        loosened_disentanglement_tolerance(_unconverged_text(30.1, 29.5, 31.2, 30.8, 29.9)) is None
    )
    assert loosened_disentanglement_tolerance(_unconverged(*DIAMOND_SPREADS)) is None


def test_a_model_that_passes_is_accepted_on_the_first_rung(tmp_path):
    """
    The capture manifest carries the mesh the spacing rule gives silicon and a
    window the dataset decided; the reference is the half-shifted sub-mesh;
    and the verdict, the band distance and the ladder are written back into
    the results file, where a driver reads them.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(tmp_path / "stub_util", [{"offset_ev": 0.002}])

    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "accept"
    assert result["grid"] == [10, 10, 10]
    assert result["number_of_wannier_functions"] == 8
    assert result["number_of_bands"] == 16
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["rung"] == "default"
    assert result["band_distance"]["eta_ev"] == pytest.approx(0.002, abs=1e-9)

    scf_manifest, capture, reference = _seen(engine)
    assert scf_manifest["control"]["calculation"] == "self_consistent_field"
    assert capture["control"]["interpolation"]["grid"] == [10, 10, 10]
    assert "projection_centers" not in capture["control"]["interpolation"]
    assert capture["control"]["interpolation"]["trial_shells"] == {"Si": [0, 1]}
    assert result["trial_shells"] == {"Si": [0, 1]}
    assert capture["method"]["scf"]["number_of_bands"] == 16
    assert len(capture["control"]["kpoints_list"]) == 1000
    assert len(reference["control"]["kpoints_list"]) == 125
    assert reference["method"]["scf"]["number_of_bands"] == 16
    assert "interpolation" not in reference["control"]

    (evaluation,) = _interpolation_blocks(utility)
    assert evaluation["disentanglement"] == "projectability"
    assert evaluation["localization"] == "auto"
    assert "frozen_window_above_fermi_ev" not in evaluation, "left to the classifier's rule"

    written = json.loads(
        (tmp_path / "run" / "output" / "si_interpolation_results.json").read_text()
    )
    quality = written["quality"]
    assert quality["status"] == "accept"
    assert quality["band_distance"]["eta_ev"] == pytest.approx(0.002, abs=1e-9)
    assert quality["band_distance"]["reference"] == "half-shifted mesh"
    assert quality["ladder"][0]["status"] == "accept"


def test_an_augmented_dataset_is_captured_on_its_valence_shells(tmp_path):
    """
    The PAW file's empty channels stay out of the count, the window and the
    capture alike: eight functions, a window of sixteen, and the capture told
    which shells to build them from, since the engine on its own would build
    all four.
    """
    scf = _write_case(tmp_path, FAKE_PAW_UPF)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util", [{"offset_ev": 0.002, "number_of_bands": 8}]
    )

    manifest = yaml.safe_load(scf.read_text())
    assert TrialOrbitals.of_manifest(manifest, scf.parent).shells_by_species == {"Si": [0, 2]}
    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "accept"
    assert result["number_of_wannier_functions"] == 8
    assert result["number_of_bands"] == 16
    assert result["trial_shells"] == {"Si": [0, 2]}
    _, capture, _ = _seen(engine)
    assert capture["control"]["interpolation"]["trial_shells"] == {"Si": [0, 2]}
    # The distance returned is the one the gate measured: over the model's eight
    # bands at every reference k-point, not over the sixteen that were captured,
    # and the same count the results file carries.
    states = result["band_distance"]["number_of_states"]
    assert states == 8 * len(result["reference_kpoints"])
    assert states == result["quality"]["band_distance"]["number_of_states"]


def test_a_model_of_another_size_than_the_rule_counted_is_refused(tmp_path):
    """
    The window and the report follow the rule's count and the engine builds
    what it was told; were they ever to read a dataset differently, the ladder
    would judge one model and report another. That stops the run.
    """
    scf = _write_case(tmp_path, FAKE_PAW_UPF)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util", [{"offset_ev": 0.002, "number_of_bands": 16}]
    )

    with pytest.raises(Q3Error, match="built 16 Wannier functions where the trial-orbital"):
        run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)


def test_a_disentanglement_out_of_iterations_is_retried_under_the_tolerance_it_calls_for(
    tmp_path,
):
    """
    Diamond PAW's failure: retried at once under 7e-6, which then stays with
    the record, so the stalled rung after it disentangles under it too.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util",
        [
            DIAMOND_UNCONVERGED,
            {"offset_ev": 0.001, "converged": False},
            {"offset_ev": 0.001, "converged": True},
        ],
    )

    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "accept"
    rungs = [attempt["rung"] for attempt in result["attempts"]]
    assert rungs == ["default", "unconverged", "stalled"]
    assert result["attempts"][0]["status"] == "error:unconverged"
    assert "32.869651" in result["attempts"][0]["reason"]
    assert result["attempts"][1]["settings"]["disentanglement_convergence_tolerance"] == 7e-6
    first, retried, stalled = _interpolation_blocks(utility)
    assert "disentanglement_convergence_tolerance" not in first
    assert retried["disentanglement_convergence_tolerance"] == 7e-6
    assert stalled["disentanglement_convergence_tolerance"] == 7e-6
    assert stalled["localization"] == "maximal"


def test_the_retry_fires_when_the_engine_output_is_shown(tmp_path):
    """
    A run that shows the engine's output reached the ladder with a bare exit
    code and was rejected as unknown, without a retry -- which is how this
    retry's first measurement on a real failure was lost.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util", [DIAMOND_UNCONVERGED, {"offset_ev": 0.001}]
    )

    result = run_automatic_wannier_workflow(
        scf, q3_binary=engine, q3_utility_binary=utility, capture_output=False
    )

    assert result["status"] == "accept"
    assert [attempt["rung"] for attempt in result["attempts"]] == ["default", "unconverged"]


def test_a_disentanglement_that_does_not_settle_is_rejected_with_its_spreads(tmp_path):
    """
    A retry that runs out of iterations again is not retried a second time,
    and nothing walks thresholds on it -- every attempt would cost as much as
    the one that failed. Spreads jumping by percent are not retried at all.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util", [DIAMOND_UNCONVERGED, DIAMOND_UNCONVERGED]
    )

    result = run_automatic_wannier_workflow(
        scf, q3_binary=engine, q3_utility_binary=utility, maximum_mesh_refinements=0
    )

    assert result["status"] == "reject"
    assert "did not converge" in result["reason"]
    assert [attempt["rung"] for attempt in result["attempts"]] == ["default", "unconverged"]
    assert [attempt["status"] for attempt in result["attempts"]] == ["error:unconverged"] * 2

    wild = _write_stub_utility(
        tmp_path / "stub_util_wild", [_unconverged_failure(30.1, 29.5, 31.2, 30.8, 29.9)]
    )
    result = run_automatic_wannier_workflow(
        scf, q3_binary=engine, q3_utility_binary=wild, maximum_mesh_refinements=0, skip_scf=True
    )
    assert result["status"] == "reject"
    assert [attempt["rung"] for attempt in result["attempts"]] == ["default"]


def test_a_stalled_localisation_is_retried_as_a_demand_with_twice_the_budget(tmp_path):
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util",
        [{"offset_ev": 0.001, "converged": False}, {"offset_ev": 0.001, "converged": True}],
    )

    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "accept"
    rungs = [attempt["rung"] for attempt in result["attempts"]]
    assert rungs == ["default", "stalled"]
    assert result["attempts"][0]["status"] == "retry"
    assert "stalled" in result["attempts"][0]["reason"]
    first, second = _interpolation_blocks(utility)
    assert first["localization"] == "auto"
    assert second["localization"] == "maximal"
    assert second["localization_maximum_iterations"] == 6000


def test_a_metal_window_the_dataset_cannot_fill_falls_back_to_the_occupied_manifold(tmp_path):
    """
    Copper's case: the classifier says it excluded band 10 from the automatic
    2 eV window, the localisation stalls anyway (the state is frozen wherever
    it is barely reachable), the demand rung stalls too, and the window rung
    freezes the occupied manifold alone -- which converges. The threshold
    walk, had it been needed, would have carried that window forward.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    excluded = [{"band": 10, "kpoints": 7}]
    utility = _write_stub_utility(
        tmp_path / "stub_util",
        [
            {"offset_ev": 0.001, "converged": False, "excluded": excluded},
            {"offset_ev": 0.001, "converged": False, "excluded": excluded},
            {"offset_ev": 0.001, "converged": True, "excluded": excluded},
        ],
    )

    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "accept"
    assert [attempt["rung"] for attempt in result["attempts"]] == ["default", "stalled", "window"]
    blocks = _interpolation_blocks(utility)
    assert "frozen_window_above_fermi_ev" not in blocks[0]
    assert "frozen_window_above_fermi_ev" not in blocks[1]
    assert blocks[2]["frozen_window_above_fermi_ev"] == 0.0
    assert blocks[2]["localization"] == "maximal", "the demand and its budget are kept"
    assert result["attempts"][2]["settings"]["frozen_window_above_fermi_ev"] == 0.0


def test_a_window_that_freezes_too_much_is_lowered_before_the_thresholds_walk(tmp_path):
    """
    Too many frozen states is the classifier's own refusal, and lowering the
    window is the answer it names: 1 eV, then the occupied manifold. When
    neither is feasible either, the published grid follows, maximum varying
    fastest from (0.01, 0.99) and skipping the (0.01, 0.95) already tried, at
    the window it started from; the report names the pair it settled on.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    infeasible = (
        "disentanglement: k-point 3 freezes 11 bands but only 10 Wannier functions were asked for"
    )
    utility = _write_stub_utility(
        tmp_path / "stub_util",
        [_failure(infeasible, "frozen_window_overfull")] * 4 + [{"offset_ev": 0.003}],
    )

    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "accept"
    rungs = [attempt["rung"] for attempt in result["attempts"]]
    assert rungs == ["default", "window", "window", "thresholds", "thresholds"]
    assert result["attempts"][0]["status"] == "error:infeasible"
    assert "freezes 11" in result["attempts"][0]["reason"]
    blocks = _interpolation_blocks(utility)
    assert blocks[1]["frozen_window_above_fermi_ev"] == 1.0
    assert blocks[2]["frozen_window_above_fermi_ev"] == 0.0
    assert "frozen_window_above_fermi_ev" not in blocks[3], "neither window helped; back to auto"
    assert blocks[3]["projectability_maximum"] == pytest.approx(0.99)
    assert blocks[4]["projectability_maximum"] == pytest.approx(0.98)
    assert result["attempts"][4]["settings"]["projectability_maximum"] == pytest.approx(0.98)
    # The lowering rung once reused the band count's name as its loop variable,
    # so a run that passed through it reported the last energy tried, 0.0 eV,
    # as its band count. The window is 16 for this fixture's eight functions
    # and must still be after the rung.
    assert result["number_of_bands"] == 16
    assert isinstance(result["number_of_bands"], int)
    # The same leak zeroed the state count the band distance carries: the model's
    # eight bands at every reference k-point, which is what the gate compared.
    states = result["band_distance"]["number_of_states"]
    assert states == 8 * len(result["reference_kpoints"]), states


def test_the_window_rung_fires_on_the_engines_code_whatever_the_prose_says(tmp_path):
    """
    The code is the contract. A refusal whose prose no longer says "freezes"
    but whose diagnostic line says frozen_window_overfull still lowers the
    window; the prose is read only when there is no code to read.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util",
        [
            {
                "error": "the classifier declined this window",
                "diagnostic": "frozen_window_overfull",
            },
            {"offset_ev": 0.003},
        ],
    )

    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "accept"
    assert [attempt["rung"] for attempt in result["attempts"]] == ["default", "window"]
    assert result["attempts"][0]["status"] == "error:infeasible"
    assert _interpolation_blocks(utility)[1]["frozen_window_above_fermi_ev"] == 1.0


def test_a_converged_model_short_of_the_gate_gets_a_wider_window_first(tmp_path):
    """
    Silicon's case: 14 meV at the published window with a stationary gauge,
    5 meV one electronvolt wider. The window is widened by 1 eV, and by 2 eV
    when that helped but not enough; a step that does not help ends it.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util", [{"offset_ev": 0.014}, {"offset_ev": 0.005}]
    )
    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)
    assert result["status"] == "accept"
    assert [attempt["rung"] for attempt in result["attempts"]] == ["default", "window"]
    blocks = _interpolation_blocks(utility)
    assert "frozen_window_above_fermi_ev" not in blocks[0]
    assert blocks[1]["frozen_window_above_fermi_ev"] == pytest.approx(3.0)

    twice = _write_stub_utility(
        tmp_path / "stub_twice",
        [{"offset_ev": 0.014}, {"offset_ev": 0.012}, {"offset_ev": 0.004}],
    )
    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=twice)
    assert result["status"] == "accept"
    assert [attempt["rung"] for attempt in result["attempts"]] == ["default", "window", "window"]
    blocks = _interpolation_blocks(twice)
    assert blocks[1]["frozen_window_above_fermi_ev"] == pytest.approx(3.0)
    assert blocks[2]["frozen_window_above_fermi_ev"] == pytest.approx(4.0)

    no_help = _write_stub_utility(
        tmp_path / "stub_no_help",
        [{"offset_ev": 0.014}, {"offset_ev": 0.016}, {"offset_ev": 0.003}],
    )
    result = run_automatic_wannier_workflow(
        scf, q3_binary=engine, q3_utility_binary=no_help, maximum_threshold_attempts=1
    )
    assert result["status"] == "accept"
    assert [attempt["rung"] for attempt in result["attempts"]] == [
        "default",
        "window",
        "thresholds",
    ]
    blocks = _interpolation_blocks(no_help)
    assert "frozen_window_above_fermi_ev" not in blocks[2], (
        "the wider window did not help, so the walk keeps the published one"
    )


def test_a_band_distance_above_the_gate_exhausts_the_grid_then_refines_the_mesh(tmp_path):
    """
    One wider window that does not help, then twenty-nine threshold pairs
    after the default, all above the gate, then the one expensive rung: halve
    the spacing, recapture on 20x20x20, and start again. The recapture is
    visible as a second capture manifest and a second reference set.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    too_far = [{"offset_ev": 0.050}] * (1 + 1 + 2 * len(PROJECTABILITY_MAXIMUM_LADDER) - 1)
    utility = _write_stub_utility(tmp_path / "stub_util", too_far + [{"offset_ev": 0.001}])

    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "accept"
    assert result["grid"] == [20, 20, 20]
    rungs = [attempt["rung"] for attempt in result["attempts"]]
    assert rungs[1] == "window"
    assert rungs.count("thresholds") == 29
    assert rungs[-2:] == ["mesh", "default"]
    assert result["attempts"][-2]["status"] == "recapture"
    manifests = _seen(engine)
    captures = [m for m in manifests if "interpolation" in m["control"]]
    assert [m["control"]["interpolation"]["grid"] for m in captures] == [[10, 10, 10], [20, 20, 20]]
    assert result["attempts"][0]["eta_ev"] == pytest.approx(0.050, abs=1e-9)
    assert result["attempts"][0]["eta_ev"] > ACCEPT_ETA_EV


def test_a_model_that_survives_nothing_is_rejected_with_the_history(tmp_path):
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(tmp_path / "stub_util", [{"offset_ev": 0.030}] * 80)

    result = run_automatic_wannier_workflow(
        scf, q3_binary=engine, q3_utility_binary=utility, maximum_mesh_refinements=0
    )

    assert result["status"] == "reject"
    assert "30.0 meV" in result["reason"]
    assert len(result["attempts"]) == 31, (
        "the default, one wider window, and the whole grid; no mesh rung"
    )
    assert result["band_distance"]["eta_ev"] == pytest.approx(0.030, abs=1e-9)
    written = json.loads(
        (tmp_path / "run" / "output" / "si_interpolation_results.json").read_text()
    )
    assert written["quality"]["status"] == "reject"
    assert len(written["quality"]["ladder"]) == 31


def test_the_walk_can_be_capped_and_an_explicit_grid_is_doubled_by_the_mesh_rung(tmp_path):
    """
    A campaign would rather report than spend an hour on one structure, so
    the walk takes a cap; and a grid given by hand still has a mesh rung,
    which doubles it -- the 2x2x2 capture of the plan's gate comes back at
    4x4x4.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(
        tmp_path / "stub_util", [{"offset_ev": 0.050}] * 5 + [{"offset_ev": 0.001}]
    )

    result = run_automatic_wannier_workflow(
        scf,
        grid=(2, 2, 2),
        maximum_threshold_attempts=3,
        q3_binary=engine,
        q3_utility_binary=utility,
    )
    assert result["status"] == "accept"
    assert result["grid"] == [4, 4, 4]
    rungs = [attempt["rung"] for attempt in result["attempts"]]
    assert rungs == [
        "default",
        "window",
        "thresholds",
        "thresholds",
        "thresholds",
        "mesh",
        "default",
    ]


def test_a_record_the_localiser_refuses_is_rejected_with_its_reason(tmp_path):
    """
    A record without overlaps is not localised under `auto`; the stalled rung
    demands it, the localiser says why it cannot, and that is the verdict --
    no threshold walk, no recapture, because neither would change the record.
    """
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    refusal = (
        "MaximallyLocalizedGaugeChoice: localization needs the neighbour overlaps M_mn(k,b), and "
        "this record carries none"
    )
    utility = _write_stub_utility(
        tmp_path / "stub_util",
        [
            {"offset_ev": 0.001, "localized": False, "converged": False},
            _failure(refusal, "localization_unsupported"),
        ],
    )

    result = run_automatic_wannier_workflow(scf, q3_binary=engine, q3_utility_binary=utility)

    assert result["status"] == "reject"
    assert "overlaps" in result["reason"]
    assert [attempt["rung"] for attempt in result["attempts"]] == ["default", "stalled"]
    assert result["attempts"][1]["status"] == "error:localization"


def test_an_explicit_grid_and_window_override_the_rules(tmp_path):
    scf = _write_case(tmp_path)
    engine = _write_stub_engine(tmp_path / "stub_q3")
    utility = _write_stub_utility(tmp_path / "stub_util", [{"offset_ev": 0.0}])

    result = run_automatic_wannier_workflow(
        scf,
        grid=(4, 4, 4),
        number_of_bands=12,
        reference_kpoints=[(0.1, 0.2, 0.3)],
        q3_binary=engine,
        q3_utility_binary=utility,
        skip_scf=True,
    )
    assert result["status"] == "accept"
    assert result["grid"] == [4, 4, 4]
    assert result["number_of_bands"] == 12
    assert result["scf_results"] is None
    capture, reference = _seen(engine)
    assert capture["control"]["interpolation"]["grid"] == [4, 4, 4]
    assert reference["control"]["kpoints_list"] == [[0.1, 0.2, 0.3, 1.0]]

    with pytest.raises(ValueError, match="below the 8 trial orbitals"):
        run_automatic_wannier_workflow(
            scf, number_of_bands=4, q3_binary=engine, q3_utility_binary=utility, skip_scf=True
        )
