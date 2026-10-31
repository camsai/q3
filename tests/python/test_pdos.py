"""tests/python/test_pdos.py — Unit tests for q3.workflows.pdos.

Tests the Monkhorst-Pack grid generation helpers without invoking the q3
binary, and the NSCF manifest ``run_pdos_workflow`` writes, against a stand-in
for the engine. The physics of ``run_pdos_workflow`` needs a real SCF+NSCF run
and is covered separately under tests/validation/.
"""

import math
from pathlib import Path

import pytest
import yaml
from q3.io.results.classes import Results
from q3.workflows import pdos
from q3.workflows.pdos import monkhorst_pack_grid, monkhorst_pack_kpoints_list

# ── monkhorst_pack_grid ───────────────────────────────────────────────────────


def test_grid_point_count():
    """Grid has n1*n2*n3 points."""
    kpoints = monkhorst_pack_grid((2, 3, 4))
    assert len(kpoints) == 24


def test_unshifted_grid_includes_gamma():
    """An unshifted grid always includes the Γ point (0,0,0)."""
    kpoints = monkhorst_pack_grid((2, 2, 2), shifts=(0, 0, 0))
    assert (0.0, 0.0, 0.0) in kpoints


def test_shifted_grid_excludes_gamma():
    """A fully shifted even grid never lands exactly on Γ."""
    kpoints = monkhorst_pack_grid((2, 2, 2), shifts=(1, 1, 1))
    assert (0.0, 0.0, 0.0) not in kpoints


def test_coordinates_within_first_brillouin_zone():
    """Every coordinate is folded into [-0.5, 0.5)."""
    kpoints = monkhorst_pack_grid((5, 6, 7), shifts=(0, 1, 0))
    for kx, ky, kz in kpoints:
        for component in (kx, ky, kz):
            assert -0.5 <= component < 0.5


def test_matches_cpp_generate_convention_single_axis():
    """Matches KPointsGrid::generate()'s fold formula along a single axis.

    x_index/n before folding is 0, 0.25, 0.5, 0.75 for n=4; folding via
    x -= floor(x + 0.5) maps 0.5 -> -0.5 and 0.75 -> -0.25.
    """
    kpoints = monkhorst_pack_grid((4, 1, 1), shifts=(0, 0, 0))
    x_coordinates = [kx for kx, _, _ in kpoints]
    assert x_coordinates == pytest.approx([0.0, 0.25, -0.5, -0.25])


def test_shift_offsets_by_half_step():
    """A shifted single-axis grid is offset by 0.5/n before folding."""
    kpoints = monkhorst_pack_grid((4, 1, 1), shifts=(1, 0, 0))
    x_coordinates = [kx for kx, _, _ in kpoints]
    assert x_coordinates == pytest.approx([0.125, 0.375, -0.375, -0.125])


# ── monkhorst_pack_kpoints_list ───────────────────────────────────────────────


def test_kpoints_list_weight_sums_to_two():
    """Weights sum to 2.0 over the full grid (spin-degeneracy convention)."""
    kpoints_list = monkhorst_pack_kpoints_list((3, 3, 3))
    total_weight = sum(entry[3] for entry in kpoints_list)
    assert total_weight == pytest.approx(2.0)


def test_kpoints_list_weight_value():
    """Each point gets weight 2.0/N."""
    kpoints_list = monkhorst_pack_kpoints_list((2, 2, 2))
    assert len(kpoints_list) == 8
    for entry in kpoints_list:
        assert entry[3] == pytest.approx(0.25)


def test_kpoints_list_coordinates_match_grid():
    """The first three entries of each tuple match monkhorst_pack_grid."""
    dimensions = (2, 3, 1)
    grid = monkhorst_pack_grid(dimensions)
    kpoints_list = monkhorst_pack_kpoints_list(dimensions)
    assert [entry[:3] for entry in kpoints_list] == [pytest.approx(kpt) for kpt in grid]


def test_grid_matches_direct_formula():
    """Cross-check against the raw Monkhorst-Pack formula for a small grid."""

    def fold(value):
        return value - math.floor(value + 0.5)

    dimensions = (3, 2, 1)
    expected = []
    for i in range(3):
        for j in range(2):
            for k in range(1):
                expected.append((fold(i / 3), fold(j / 2), fold(k / 1)))

    grid = monkhorst_pack_grid(dimensions)
    assert len(grid) == len(expected)
    for point, expected_point in zip(grid, expected, strict=True):
        assert point == pytest.approx(expected_point)


# ── run_pdos_workflow ─────────────────────────────────────────────────────────


def test_the_pdos_nscf_from_a_band_structure_manifest_carries_no_band_path(
    tmp_path, monkeypatch, examples_dir
):
    """The PDOS grid is not a path: the template's band path must not reach it."""
    band_structure_input = examples_dir / "band_structure--semiconductor--silicon" / "nc.yml"
    scf_input = tmp_path / "nc.yml"
    scf_input.write_text(band_structure_input.read_text())

    written_manifests = []

    def record_the_manifest(input_path, **_):
        written_manifests.append(yaml.safe_load(Path(input_path).read_text()))
        return Results.model_validate(
            {"properties": {"records": [{"name": "fermi_energy", "units": "Ry", "value": 0.0}]}}
        )

    monkeypatch.setattr(pdos, "find_binary", lambda binary: "q3")
    monkeypatch.setattr(pdos, "run_q3", record_the_manifest)
    monkeypatch.setattr(pdos, "extract_lowdin_charges", lambda results: {})
    monkeypatch.setattr(pdos, "extract_pdos", lambda results, **_: {})

    pdos.run_pdos_workflow(scf_input, kgrid_dimensions=(2, 2, 2), skip_scf=True)

    (nscf_manifest,) = written_manifests
    control = nscf_manifest["control"]
    assert control["calculation"] == "nscf"
    assert len(control["kpoints_list"]) == 8
    assert "band_path" not in control
    assert "band_path_points_per_segment" not in control
