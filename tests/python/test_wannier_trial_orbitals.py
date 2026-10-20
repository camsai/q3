"""tests/python/test_wannier_trial_orbitals.py — what the datasets of a material decide, read once."""

import json
from pathlib import Path

import pytest
from q3.workflows.wannier import TrialOrbitals, band_window, trial_orbitals

DATASETS = Path(__file__).resolve().parents[2] / "examples" / "data" / "pseudopotentials"

COPPER_SILICIDE = {
    "basis": {
        "elements": [{"id": 0, "value": "Cu"}, {"id": 1, "value": "Si"}, {"id": 2, "value": "Cu"}]
    },
}
COPPER_TRIMER = {
    "basis": {"elements": [{"id": index, "value": "Cu"} for index in range(3)]},
}


def test_the_count_is_one_datasets_worth_of_orbitals_per_atom():
    copper = TrialOrbitals.of(COPPER_TRIMER, {"Cu": DATASETS / "paw" / "Cu.upf"})
    assert copper.shells_by_species == {"Cu": [0, 1, 2, 3, 4]}
    assert copper.number_of_wannier_functions == 3 * 13
    assert copper.band_window == band_window(39)


def test_every_species_contributes_its_own_shells():
    datasets = {"Cu": DATASETS / "paw" / "Cu.upf", "Si": DATASETS / "paw" / "Si.upf"}
    orbitals = TrialOrbitals.of(COPPER_SILICIDE, datasets)
    assert orbitals.shells_by_species == {"Cu": [0, 1, 2, 3, 4], "Si": [0, 2]}
    assert orbitals.number_of_wannier_functions == 13 + 4 + 13


def test_each_dataset_is_read_once(monkeypatch):
    read = []
    reader = trial_orbitals.pseudo_atomic_shells

    def counting_reader(path):
        read.append(Path(path).name)
        return reader(path)

    monkeypatch.setattr(trial_orbitals, "pseudo_atomic_shells", counting_reader)
    TrialOrbitals.of(COPPER_TRIMER, {"Cu": DATASETS / "paw" / "Cu.upf"})
    assert read == ["Cu.upf"]


def test_a_dataset_that_leaves_no_usable_shell_is_refused(tmp_path):
    dataset = tmp_path / "unusable.upf"
    dataset.write_text(
        '<UPF version="2.0.1">\n<PP_PSWFC>\n'
        '<PP_CHI.1 type="real" size="2" l="0" occupation="-1.0" label="3S">0.1 0.1</PP_CHI.1>\n'
        "</PP_PSWFC>\n</UPF>\n"
    )
    with pytest.raises(ValueError, match="leaves no shell to build trial orbitals from"):
        TrialOrbitals.of({"basis": {"elements": [{"id": 0, "value": "Si"}]}}, {"Si": dataset})


def test_a_manifest_names_the_material_and_the_datasets(tmp_path):
    (tmp_path / "silicon.json").write_text(
        json.dumps({"basis": {"elements": [{"id": 0, "value": "Si"}, {"id": 1, "value": "Si"}]}})
    )
    manifest = {
        "material": "silicon.json",
        "method": {"pseudopotentials": {"Si": str(DATASETS / "paw" / "Si.upf")}},
    }
    orbitals = TrialOrbitals.of_manifest(manifest, tmp_path)
    assert orbitals.shells_by_species == {"Si": [0, 2]}
    assert orbitals.number_of_wannier_functions == 8
    assert orbitals.band_window == 16
