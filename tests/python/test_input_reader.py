"""tests/python/test_input_reader.py — the files a manifest names, found where the engine finds them."""

import json
from pathlib import Path

import pytest
from q3.io.input_reader import (
    load_material,
    make_file_references_absolute,
    pseudopotential_paths,
)

ALLOY = {
    "name": "Si3C",
    "basis": {
        "elements": [
            {"id": 0, "value": "Si"},
            {"id": 1, "value": "C"},
            {"id": 2, "value": "Si"},
            {"id": 3, "value": "Si"},
        ]
    },
}


def test_the_material_is_the_file_the_manifest_names_or_the_map_it_carries(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "alloy.json").write_text(json.dumps(ALLOY))
    assert load_material({"material": "data/alloy.json"}, tmp_path) == ALLOY
    assert load_material({"material": str(tmp_path / "data" / "alloy.json")}, "/nowhere") == ALLOY
    assert load_material({"material": ALLOY}, tmp_path) is ALLOY
    with pytest.raises(ValueError, match="neither a path"):
        load_material({"material": 3}, tmp_path)


def test_each_species_has_the_dataset_the_manifest_names_once_and_in_basis_order(tmp_path):
    manifest = {"method": {"pseudopotentials": {"C": "/datasets/C.upf", "Si": "data/Si.upf"}}}
    paths = pseudopotential_paths(manifest, ALLOY, tmp_path)
    assert list(paths) == ["Si", "C"]
    assert paths == {"Si": tmp_path / "data" / "Si.upf", "C": Path("/datasets/C.upf")}


def test_a_species_without_a_dataset_is_refused_by_name(tmp_path):
    manifest = {"method": {"pseudopotentials": {"Si": "data/Si.upf"}}}
    with pytest.raises(ValueError, match="no pseudopotential for C"):
        pseudopotential_paths(manifest, ALLOY, tmp_path)
    with pytest.raises(ValueError, match="no pseudopotential for Si"):
        pseudopotential_paths({}, ALLOY, tmp_path)


def test_a_manifests_file_references_become_absolute_in_place(tmp_path):
    manifest = {
        "material": "data/alloy.json",
        "method": {"pseudopotentials": {"Si": "data/Si.upf", "C": "/datasets/C.upf"}},
    }
    make_file_references_absolute(manifest, tmp_path)
    assert manifest["material"] == str((tmp_path / "data" / "alloy.json").resolve())
    assert manifest["method"]["pseudopotentials"] == {
        "Si": str((tmp_path / "data" / "Si.upf").resolve()),
        "C": "/datasets/C.upf",
    }

    inline = {"material": ALLOY}
    make_file_references_absolute(inline, tmp_path)
    assert inline == {"material": ALLOY}
