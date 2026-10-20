"""tests/python/test_input_writer.py — Unit tests for q3.io.input_writer.

Tests the YAML input generation helpers without invoking the q3 binary.
"""

import json
from pathlib import Path

import pytest
import yaml
from q3.errors import Q3InputError
from q3.io.input_writer import (
    ensure_calculation_type,
    write_material_json,
    write_nscf_input,
    write_scf_input,
    write_supercell_scf_input,
)

# ── write_scf_input ──────────────────────────────────────────────────────────


def test_write_scf_input_creates_file(tmp_path):
    """write_scf_input creates a valid YAML file."""
    output = tmp_path / "scf.yml"
    write_scf_input(
        output,
        material_path="/some/material.json",
        pseudopotentials={"Si": "/some/Si.upf"},
        wavefunction_cutoff=30.0,
        kgrid_dimensions=(4, 4, 4),
        outdir=str(tmp_path / "output"),
        prefix="test",
    )
    assert output.exists()
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)
    assert config["control"]["calculation"] == "scf"
    assert config["method"]["cutoffs"]["wavefunction"] == 30.0
    assert config["method"]["pseudopotentials"]["Si"] == "/some/Si.upf"


def test_write_scf_input_leaves_an_unnamed_functional_to_the_pseudopotentials(tmp_path):
    """No functional from the caller writes none, so the engine takes the datasets' own.

    Writing a fixed default here would repeat, one layer up, the engine bug this
    mirrors: a functional nobody chose, whatever the pseudopotentials were.
    """
    unnamed = tmp_path / "unnamed.yml"
    write_scf_input(
        unnamed, material_path="/some/material.json", pseudopotentials={"Si": "/some/Si.upf"}
    )
    named = tmp_path / "named.yml"
    write_scf_input(
        named,
        material_path="/some/material.json",
        pseudopotentials={"Si": "/some/Si.upf"},
        exchange_correlation_functional="pbesol",
    )
    with open(unnamed) as file_handle:
        assert "functional" not in yaml.safe_load(file_handle)["model"]
    with open(named) as file_handle:
        assert yaml.safe_load(file_handle)["model"]["functional"] == "pbesol"


def test_write_scf_input_writes_an_iteration_budget_only_when_given_one(tmp_path):
    """Left out, the key is absent and the engine's own default applies."""
    arguments = {"material_path": "/some/material.json", "pseudopotentials": {"Si": "/some/Si.upf"}}
    write_scf_input(tmp_path / "default.yml", **arguments)
    write_scf_input(tmp_path / "budget.yml", maximum_iterations=200, **arguments)
    default = yaml.safe_load((tmp_path / "default.yml").read_text())
    budget = yaml.safe_load((tmp_path / "budget.yml").read_text())
    assert "max_iterations" not in default["method"]["scf"]
    assert budget["method"]["scf"]["max_iterations"] == 200


def test_write_scf_input_density_cutoff_defaults_to_4x(tmp_path):
    """Density cutoff defaults to 4× wavefunction cutoff."""
    output = tmp_path / "scf.yml"
    write_scf_input(
        output,
        material_path="/some/material.json",
        pseudopotentials={"Si": "/some/Si.upf"},
        wavefunction_cutoff=40.0,
    )
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)
    assert config["method"]["cutoffs"]["density"] == 160.0


def test_write_scf_input_explicit_density_cutoff(tmp_path):
    """Explicit density cutoff overrides default."""
    output = tmp_path / "scf.yml"
    write_scf_input(
        output,
        material_path="/some/material.json",
        pseudopotentials={"Si": "/some/Si.upf"},
        wavefunction_cutoff=30.0,
        density_cutoff=200.0,
    )
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)
    assert config["method"]["cutoffs"]["density"] == 200.0


def test_write_scf_input_smearing_uses_parser_keys(tmp_path):
    """write_scf_input's smearing block must use the keys the C++ YAML parser
    actually reads (method.scf.smearing.type / .degauss, see
    src/io/input/parser/yaml.hpp) -- not scheme/width, which the parser
    silently ignores, making the smearing/smearing_width parameters a no-op.
    """
    output = tmp_path / "scf.yml"
    write_scf_input(
        output,
        material_path="/some/material.json",
        pseudopotentials={"Si": "/some/Si.upf"},
        wavefunction_cutoff=30.0,
        smearing="gaussian",
        smearing_width=0.02,
    )
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)
    smearing_config = config["method"]["scf"]["smearing"]
    assert smearing_config == {"type": "gaussian", "degauss": 0.02}


# ── write_nscf_input ─────────────────────────────────────────────────────────


def test_write_nscf_input_from_scf_template(tmp_path, si_scf_input):
    """write_nscf_input produces a valid NSCF YAML from an SCF template."""
    kpoints_list = [
        (0.0, 0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0, 1.0),
    ]
    output = tmp_path / "nscf.yml"
    write_nscf_input(output, scf_input_path=si_scf_input, kpoints_list=kpoints_list)

    assert output.exists()
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)

    assert config["control"]["calculation"] == "nscf"
    written_kpoints = config["control"]["kpoints_list"]
    assert len(written_kpoints) == 3
    assert written_kpoints[0] == [0.0, 0.0, 0.0, 1.0]


def test_write_nscf_input_removes_kgrid(tmp_path, si_scf_input):
    """write_nscf_input removes the kgrid section to avoid confusion."""
    output = tmp_path / "nscf.yml"
    write_nscf_input(
        output,
        scf_input_path=si_scf_input,
        kpoints_list=[(0.0, 0.0, 0.0, 1.0)],
    )
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)
    method_scf = config.get("method", {}).get("scf", {})
    assert "kgrid" not in method_scf


def test_write_nscf_input_drops_the_band_path_for_a_uniform_grid(tmp_path, examples_dir):
    """A grid NSCF from a band-structure manifest carries none of its band path."""
    band_structure_input = examples_dir / "band_structure--semiconductor--silicon" / "nc.yml"
    output = tmp_path / "nscf.yml"
    write_nscf_input(
        output,
        scf_input_path=band_structure_input,
        kpoints_list=[(0.0, 0.0, 0.0, 1.0), (0.5, 0.0, 0.0, 1.0)],
        uniform_grid=True,
    )
    control = yaml.safe_load(output.read_text())["control"]
    assert "band_path" not in control
    assert "band_path_points_per_segment" not in control


def test_write_nscf_input_keeps_the_band_path_by_default(tmp_path, examples_dir):
    """A path NSCF keeps the template's band path, which the engine seeds along."""
    band_structure_input = examples_dir / "band_structure--semiconductor--silicon" / "nc.yml"
    template_control = yaml.safe_load(band_structure_input.read_text())["control"]
    output = tmp_path / "nscf.yml"
    write_nscf_input(
        output, scf_input_path=band_structure_input, kpoints_list=[(0.0, 0.0, 0.0, 1.0)]
    )
    control = yaml.safe_load(output.read_text())["control"]
    assert control["band_path"] == template_control["band_path"]
    assert control["band_path_points_per_segment"] == 30


def test_write_nscf_input_compute_projections_default_omitted(tmp_path, si_scf_input):
    """write_nscf_input omits compute_projections from the YAML by default."""
    output = tmp_path / "nscf.yml"
    write_nscf_input(
        output,
        scf_input_path=si_scf_input,
        kpoints_list=[(0.0, 0.0, 0.0, 1.0)],
    )
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)
    assert "compute_projections" not in config["control"]


def test_write_nscf_input_compute_projections_enabled(tmp_path, si_scf_input):
    """write_nscf_input sets control.compute_projections: true when requested."""
    output = tmp_path / "nscf.yml"
    write_nscf_input(
        output,
        scf_input_path=si_scf_input,
        kpoints_list=[(0.0, 0.0, 0.0, 1.0)],
        compute_projections=True,
    )
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)
    assert config["control"]["compute_projections"] is True


def test_write_nscf_input_preserves_prefix(tmp_path, si_scf_input):
    """write_nscf_input keeps the SCF prefix by default."""
    with open(si_scf_input) as file_handle:
        scf_config = yaml.safe_load(file_handle)
    scf_prefix = scf_config["control"]["prefix"]

    output = tmp_path / "nscf.yml"
    write_nscf_input(
        output,
        scf_input_path=si_scf_input,
        kpoints_list=[(0.0, 0.0, 0.0, 1.0)],
    )
    with open(output) as file_handle:
        config = yaml.safe_load(file_handle)
    assert config["control"]["prefix"] == scf_prefix


# ── ensure_calculation_type ───────────────────────────────────────────────────


def test_ensure_calculation_type_accepts_alias(tmp_path, si_scf_input):
    """ensure_calculation_type accepts 'scf' as alias for 'self_consistent_field'."""
    # si_scf_input has calculation: scf — should be accepted.
    result = ensure_calculation_type(
        si_scf_input, expected="self_consistent_field", aliases=("scf",)
    )
    assert result == si_scf_input


def test_ensure_calculation_type_raises_on_mismatch(tmp_path):
    """ensure_calculation_type raises Q3InputError on wrong type."""
    scf_yml = tmp_path / "input.yml"
    scf_yml.write_text("control:\n  calculation: scf\n")
    with pytest.raises(Q3InputError):
        ensure_calculation_type(scf_yml, expected="non_self_consistent_field")


# ── write_material_json ──────────────────────────────────────────────────────


MINIMAL_MATERIAL = {
    "name": "Si",
    "lattice": {"type": "FCC", "a": 5.43},
    "basis": {
        "elements": [{"id": 0, "value": "Si"}],
        "coordinates": [{"id": 0, "value": [0.0, 0.0, 0.0]}],
    },
}


def test_write_material_json_accepts_a_dict(tmp_path):
    """The form q3.structure.ase_atoms.ase_atoms_to_material_json returns."""
    written = write_material_json(MINIMAL_MATERIAL, tmp_path / "material.json")

    assert written.is_file()
    reloaded = json.loads(written.read_text())
    assert reloaded["lattice"]["type"] == "FCC"
    assert reloaded["basis"]["elements"][0]["value"] == "Si"


def test_write_material_json_accepts_the_material_protocol(tmp_path):
    """
    A ``mat3ra.made`` ``Material`` presents itself through ``to_dict()``. The
    protocol is accepted rather than the class so that package stays out of
    this one's dependencies, where nothing else needs it -- a stand-in with the
    same method is therefore the honest test.
    """

    class MaterialLike:
        def to_dict(self):
            return MINIMAL_MATERIAL

    class JsonOnlyMaterialLike:
        def to_json(self):
            return json.dumps(MINIMAL_MATERIAL)

    for material in (MaterialLike(), JsonOnlyMaterialLike(), json.dumps(MINIMAL_MATERIAL)):
        written = write_material_json(material, tmp_path / "m.json")
        assert json.loads(written.read_text())["lattice"]["a"] == 5.43


def test_write_material_json_refuses_a_document_q3_cannot_read(tmp_path):
    """
    Both keys are checked here rather than left to the C++ reader: the caller
    never opened the file, so a parser error naming a missing key in it is a
    long way from the object that lacked one.
    """
    with pytest.raises(Q3InputError) as failure:
        write_material_json({"lattice": {"type": "FCC"}}, tmp_path / "m.json")
    assert "basis" in str(failure.value)

    with pytest.raises(Q3InputError):
        write_material_json(42, tmp_path / "m.json")

    with pytest.raises(Q3InputError):
        write_material_json("{not json", tmp_path / "m.json")


def test_write_material_json_output_feeds_write_scf_input(tmp_path):
    """The point of the helper: an in-memory structure becomes a runnable input."""
    material_path = write_material_json(MINIMAL_MATERIAL, tmp_path / "material.json")
    input_path = write_scf_input(
        tmp_path / "scf.yml",
        material_path=material_path,
        pseudopotentials={"Si": "Si.upf"},
    )

    manifest = yaml.safe_load(input_path.read_text())
    assert Path(manifest["material"]).name == "material.json"


# ── write_supercell_scf_input ────────────────────────────────────────────────


@pytest.fixture
def unit_cell_input(tmp_path):
    """A unit-cell input naming its files relative to itself, as the examples do."""
    config = {
        "material": "materials/silicon.json",
        "method": {
            "pseudopotentials": {"Si": "pseudopotentials/Si.upf"},
            "scf": {
                "number_of_bands": 8,
                "kgrid": {"dimensions": [2, 2, 2], "shifts": [1, 1, 1]},
            },
        },
        "control": {"calculation": "scf"},
    }
    input_path = tmp_path / "nc.yml"
    input_path.write_text(yaml.dump(config))
    return input_path


def supercell_input(unit_cell_input, supercell_matrix, kgrid_dimensions=None):
    if kgrid_dimensions is not None:
        config = yaml.safe_load(unit_cell_input.read_text())
        config["method"]["scf"]["kgrid"]["dimensions"] = kgrid_dimensions
        unit_cell_input.write_text(yaml.dump(config))
    written = write_supercell_scf_input(
        unit_cell_input.parent / "derived" / "supercell.yml",
        scf_input_path=unit_cell_input,
        supercell_matrix=supercell_matrix,
    )
    return yaml.safe_load(written.read_text())


def test_write_supercell_scf_input_leaves_the_band_count_to_the_engine(unit_cell_input):
    """8 bands describe 2 atoms; inherited by 16 they would leave out three
    quarters of the valence states in an SCF that still converges, so the
    count is dropped and the engine sizes it for the supercell's electrons."""
    derived = supercell_input(unit_cell_input, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert "number_of_bands" not in derived["method"]["scf"]


def test_write_supercell_scf_input_divides_the_kgrid_and_keeps_the_shift(unit_cell_input):
    """Silicon's own case: the 2x2x2 shifted mesh of the 2-atom cell is,
    point for point, a 1x1x1 shifted mesh of the 2x2x2 supercell."""
    derived = supercell_input(unit_cell_input, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert derived["method"]["scf"]["kgrid"] == {"dimensions": [1, 1, 1], "shifts": [1, 1, 1]}


@pytest.mark.parametrize(
    ("unit_cell_grid", "diagonal", "supercell_grid"),
    [
        ([3, 3, 3], [2, 2, 2], [2, 2, 2]),
        ([2, 2, 2], [4, 4, 4], [1, 1, 1]),
        ([6, 4, 2], [3, 3, 3], [2, 2, 1]),
    ],
)
def test_write_supercell_scf_input_rounds_the_kgrid_up_and_never_to_zero(
    unit_cell_input, unit_cell_grid, diagonal, supercell_grid
):
    derived = supercell_input(
        unit_cell_input,
        [[diagonal[0], 0, 0], [0, diagonal[1], 0], [0, 0, diagonal[2]]],
        unit_cell_grid,
    )
    assert derived["method"]["scf"]["kgrid"]["dimensions"] == supercell_grid


def test_write_supercell_scf_input_keeps_a_non_diagonal_supercell_kgrid(unit_cell_input):
    """No per-axis divisor exists, so the grid is kept: oversampling, not wrong."""
    derived = supercell_input(unit_cell_input, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert derived["method"]["scf"]["kgrid"]["dimensions"] == [2, 2, 2]


def test_write_supercell_scf_input_resolves_relative_paths(unit_cell_input):
    """The copy lives elsewhere, so a reference relative to the template would
    send the engine looking for the pseudopotential in the wrong directory."""
    derived = supercell_input(unit_cell_input, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert derived["material"] == str(unit_cell_input.parent / "materials" / "silicon.json")
    assert derived["method"]["pseudopotentials"]["Si"] == str(
        unit_cell_input.parent / "pseudopotentials" / "Si.upf"
    )


def test_write_supercell_scf_input_does_not_modify_the_template(unit_cell_input):
    before = unit_cell_input.read_text()
    supercell_input(unit_cell_input, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert unit_cell_input.read_text() == before
