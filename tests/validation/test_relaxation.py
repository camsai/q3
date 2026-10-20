"""
Validation tests for the Geometry Relaxation workflow in Q3.

Runs the ASE BFGS + Q3ASECalculator relaxation workflow for Si (NC) and
validates key physical properties:

  1. Convergence: BFGS converges within the force threshold.
  2. Step count: at equilibrium geometry, should converge at step 0.
  3. Total energy: close to the reference recorded in the example's own
     `reference/results.json`, which names the code and settings behind it.
  4. Forces: maximum |F| below the convergence threshold.

Note: the Si example starts at the equilibrium geometry (forces ~0 by
symmetry), so BFGS converges immediately without taking any ionic steps.
This validates the SCF + forces pipeline; a separate test with a displaced
geometry would validate the position-update path.
"""

import json
import logging

import pytest
import yaml
from q3.workflows.relax import run_relax_workflow

from helpers import EXAMPLES_DIRECTORY

# ── Constants ────────────────────────────────────────────────────────────────

RY_TO_EV = 13.6058

SI_EXAMPLE_DIR = EXAMPLES_DIRECTORY / "relaxation--semiconductor--silicon"
CU_EXAMPLE_DIR = EXAMPLES_DIRECTORY / "relaxation--metal--copper"


def reference_energies_ry(example_directory):
    """
    The reference total energies recorded beside the example's own inputs.

    Read rather than transcribed: an example's ``reference/results.json``
    records which code produced each number and at which settings, and a copy
    living here instead would carry neither. The copper numbers were literally
    that -- constants annotated "QE+LibXC reference" that were in fact Q3's own
    output, which nothing beside them could contradict.
    """
    with open(example_directory / "reference" / "results.json") as reference_file:
        reference = json.load(reference_file)
    return {flavor: reference[flavor]["equilibrium_total_energy_ry"] for flavor in ("nc", "us")}


SI_REFERENCE_ENERGY_RY = reference_energies_ry(SI_EXAMPLE_DIR)
CU_REFERENCE_ENERGY_RY = reference_energies_ry(CU_EXAMPLE_DIR)

# Tolerance: accept ±5 mRy (same as the QE comparison tests).
SI_ENERGY_TOLERANCE_RY = 5.0e-3
CU_ENERGY_TOLERANCE_RY = 5.0e-3

# Force convergence threshold used in the example inputs (Ry/bohr).
SI_FMAX_RY_PER_BOHR = 5.0e-3
CU_FMAX_RY_PER_BOHR = 5.0e-3


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("flavor", ["nc", "us"])
def test_silicon_relaxation_converges_at_equilibrium(q3_command, tmp_path, flavor):
    """
    Validate Si PBE geometry relaxation starting from the equilibrium geometry.

    At the equilibrium geometry, forces are zero by symmetry. BFGS should
    converge immediately at step 0 (no ionic steps required).
    """
    # Copy files to tmp_path and set coordinates to equilibrium [0.875, 0.875, 0.875]
    workdir = tmp_path / f"equilibrium_{flavor}"
    workdir.mkdir()

    with open(SI_EXAMPLE_DIR / "silicon_displaced.json") as f:
        material = json.load(f)
    material["basis"]["coordinates"][0]["value"] = [0.875, 0.875, 0.875]

    with open(workdir / "material.json", "w") as f:
        json.dump(material, f)

    with open(SI_EXAMPLE_DIR / f"{flavor}.yml") as f:
        config = yaml.safe_load(f)
    config["material"] = str(workdir / "material.json")
    config["method"]["pseudopotentials"]["Si"] = str(
        (SI_EXAMPLE_DIR / config["method"]["pseudopotentials"]["Si"]).resolve()
    )
    config.setdefault("control", {})["outdir"] = str(workdir / "output")

    with open(workdir / f"{flavor}.yml", "w") as f:
        yaml.safe_dump(config, f)

    result = run_relax_workflow(
        workdir / f"{flavor}.yml",
        fmax=SI_FMAX_RY_PER_BOHR * RY_TO_EV / 0.529177,  # convert to eV/Å for ASE
        steps=20,
        q3_binary=q3_command[0],
        workdir=workdir / "relax_workdir",
    )

    # 1. BFGS converged within the force threshold.
    assert result["converged"], "BFGS did not converge within 20 steps"

    # 2. At equilibrium, no ionic steps should be needed.
    assert result["steps"] == 0, f"Expected 0 ionic steps at equilibrium, got {result['steps']}"

    # 3. Total energy is within tolerance of the QE+LibXC reference.
    energy_ry = result["final_energy_ry"]
    ref_energy = SI_REFERENCE_ENERGY_RY[flavor]
    assert abs(energy_ry - ref_energy) < SI_ENERGY_TOLERANCE_RY, (
        f"Si total energy {energy_ry:.6f} Ry is more than {SI_ENERGY_TOLERANCE_RY} Ry "
        f"from the reference {ref_energy:.6f} Ry"
    )

    # 4. Forces at the final geometry are negligibly small (equilibrium).
    forces = result["final_forces_ry_per_bohr"]
    max_force_ry_per_bohr = max(abs(component) for atom in forces for component in atom)
    assert max_force_ry_per_bohr < SI_FMAX_RY_PER_BOHR, (
        f"Max force {max_force_ry_per_bohr:.2e} Ry/bohr exceeds threshold "
        f"{SI_FMAX_RY_PER_BOHR:.2e} Ry/bohr"
    )


@pytest.mark.parametrize("flavor", ["nc", "us"])
def test_copper_relaxation_converges_at_equilibrium(q3_command, tmp_path, flavor):
    """
    Validate Cu PBE geometry relaxation starting from the equilibrium geometry.

    At the equilibrium geometry, forces are zero by symmetry. BFGS should
    converge immediately at step 0 (no ionic steps required).
    """
    # Copy files to tmp_path
    workdir = tmp_path / f"equilibrium_copper_{flavor}"
    workdir.mkdir()

    with open(EXAMPLES_DIRECTORY / "data/materials/copper.json") as f:
        material = json.load(f)

    with open(workdir / "material.json", "w") as f:
        json.dump(material, f)

    with open(CU_EXAMPLE_DIR / f"{flavor}.yml") as f:
        config = yaml.safe_load(f)
    config["material"] = str(workdir / "material.json")
    config["method"]["pseudopotentials"]["Cu"] = str(
        (CU_EXAMPLE_DIR / config["method"]["pseudopotentials"]["Cu"]).resolve()
    )
    config.setdefault("control", {})["outdir"] = str(workdir / "output")

    with open(workdir / f"{flavor}.yml", "w") as f:
        yaml.safe_dump(config, f)

    result = run_relax_workflow(
        workdir / f"{flavor}.yml",
        fmax=CU_FMAX_RY_PER_BOHR * RY_TO_EV / 0.529177,  # convert to eV/Å for ASE
        steps=20,
        q3_binary=q3_command[0],
        workdir=workdir / "relax_workdir",
    )

    # 1. BFGS converged within the force threshold.
    assert result["converged"], "BFGS did not converge within 20 steps"

    # 2. At equilibrium, no ionic steps should be needed.
    assert result["steps"] == 0, f"Expected 0 ionic steps at equilibrium, got {result['steps']}"

    # 3. Total energy is within tolerance of the QE+LibXC reference.
    energy_ry = result["final_energy_ry"]
    ref_energy = CU_REFERENCE_ENERGY_RY[flavor]
    assert abs(energy_ry - ref_energy) < CU_ENERGY_TOLERANCE_RY, (
        f"Cu total energy {energy_ry:.6f} Ry is more than {CU_ENERGY_TOLERANCE_RY} Ry "
        f"from the reference {ref_energy:.6f} Ry"
    )

    # 4. Forces at the final geometry are negligibly small (equilibrium).
    forces = result["final_forces_ry_per_bohr"]
    max_force_ry_per_bohr = max(abs(component) for atom in forces for component in atom)
    assert max_force_ry_per_bohr < CU_FMAX_RY_PER_BOHR, (
        f"Max force {max_force_ry_per_bohr:.2e} Ry/bohr exceeds threshold "
        f"{CU_FMAX_RY_PER_BOHR:.2e} Ry/bohr"
    )


@pytest.mark.parametrize(
    "flavor,expected_min_steps,expected_max_steps",
    [
        ("nc", 1, 3),
        ("us", 1, 3),
    ],
)
def test_silicon_relaxation_converges_displaced(
    q3_command, tmp_path, flavor, expected_min_steps, expected_max_steps
):
    """
    Validate Si PBE geometry relaxation starting from a displaced geometry.

    BFGS should take at least one ionic step to relax the displacement.
    """
    workdir = tmp_path / f"displaced_{flavor}"
    workdir.mkdir()

    with open(SI_EXAMPLE_DIR / "silicon_displaced.json") as f:
        material = json.load(f)

    with open(workdir / "material.json", "w") as f:
        json.dump(material, f)

    with open(SI_EXAMPLE_DIR / f"{flavor}.yml") as f:
        config = yaml.safe_load(f)
    config["material"] = str(workdir / "material.json")
    config["method"]["pseudopotentials"]["Si"] = str(
        (SI_EXAMPLE_DIR / config["method"]["pseudopotentials"]["Si"]).resolve()
    )
    config.setdefault("control", {})["outdir"] = str(workdir / "output")

    with open(workdir / f"{flavor}.yml", "w") as f:
        yaml.safe_dump(config, f)

    result = run_relax_workflow(
        workdir / f"{flavor}.yml",
        fmax=SI_FMAX_RY_PER_BOHR * RY_TO_EV / 0.529177,  # convert to eV/Å for ASE
        steps=20,
        q3_binary=q3_command[0],
        workdir=workdir / "relax_workdir",
    )

    # 1. BFGS converged within the force threshold.
    assert result["converged"], "BFGS did not converge within 20 steps"

    # 2. It should take steps in the expected range.
    assert expected_min_steps <= result["steps"] <= expected_max_steps, (
        f"Expected {expected_min_steps}-{expected_max_steps} ionic steps for displaced structure, got {result['steps']}"
    )


def test_relax_max_steps_and_force_threshold_default_from_the_input_file(
    q3_command, tmp_path, caplog
):
    """run_relax_workflow() must read relax_force_threshold/relax_max_steps
    from scf_input_path's own control section (converting Ry/bohr to eV/Å)
    when fmax/steps are not passed explicitly.

    nc.yml sets control.relax_force_threshold == 5.0e-3 Ry/bohr and
    control.relax_max_steps == 50 directly (the convention every
    examples/relaxation--*/ file uses) -- exactly the values
    test_silicon_relaxation_converges_displaced() above passes explicitly
    (converted to eV/Å), which converges nc in 1-3 steps. Before this fix,
    this workflow ignored those fields entirely and always used its own
    hardcoded fmax=0.02 eV/Å (~5.8x tighter than the file's 5.0e-3 Ry/bohr),
    regardless of what the file stated.

    Also verifies relax_energy_threshold (which nc.yml also sets, and which
    this ASE-BFGS-based workflow has no way to act on) produces a warning
    instead of being silently ignored.
    """
    workdir = tmp_path / "defaults_nc"
    workdir.mkdir()

    with open(SI_EXAMPLE_DIR / "silicon_displaced.json") as f:
        material = json.load(f)

    with open(workdir / "material.json", "w") as f:
        json.dump(material, f)

    with open(SI_EXAMPLE_DIR / "nc.yml") as f:
        config = yaml.safe_load(f)
    config["material"] = str(workdir / "material.json")
    config["method"]["pseudopotentials"]["Si"] = str(
        (SI_EXAMPLE_DIR / config["method"]["pseudopotentials"]["Si"]).resolve()
    )
    config.setdefault("control", {})["outdir"] = str(workdir / "output")
    assert config["control"]["relax_force_threshold"] == 5.0e-3
    assert config["control"]["relax_max_steps"] == 50
    assert config["control"]["relax_energy_threshold"] == 1.0e-5

    with open(workdir / "nc.yml", "w") as f:
        yaml.safe_dump(config, f)

    with caplog.at_level(logging.WARNING):
        result = run_relax_workflow(
            workdir / "nc.yml",
            q3_binary=q3_command[0],
            workdir=workdir / "relax_workdir",
        )

    # Same expectation as test_silicon_relaxation_converges_displaced(): the
    # defaulted fmax/steps produced the identical outcome, not just "some"
    # outcome.
    assert result["converged"], "BFGS did not converge using the file's own defaults"
    assert 1 <= result["steps"] <= 3, (
        f"Expected 1-3 ionic steps using nc.yml's own relax_force_threshold/"
        f"relax_max_steps, got {result['steps']} -- the YAML defaults are not "
        "being read correctly"
    )

    assert any(
        "relax_energy_threshold" in message and "no effect" in message
        for message in caplog.messages
    ), "Expected a warning that relax_energy_threshold has no effect on this workflow"


@pytest.mark.parametrize("mode", ["bindings", "subprocess"])
def test_silicon_relaxation_symmetry_comparison(q3_command, tmp_path, mode):
    """
    Verify that using FixSymmetry constraint works under both bindings and subprocess modes,
    and compare the number of relaxation steps with vs without symmetry.
    """
    if mode == "bindings":
        pytest.importorskip("q3_core", reason="q3_core C++ bindings not built")
    # 1. Run WITH symmetry
    workdir_sym = tmp_path / f"relax_sym_{mode}"
    workdir_sym.mkdir()

    with open(SI_EXAMPLE_DIR / "silicon_displaced.json") as f:
        material = json.load(f)

    with open(workdir_sym / "material.json", "w") as f:
        json.dump(material, f)

    with open(SI_EXAMPLE_DIR / "nc.yml") as f:
        config = yaml.safe_load(f)
    config["material"] = str(workdir_sym / "material.json")
    config["method"]["pseudopotentials"]["Si"] = str(
        (SI_EXAMPLE_DIR / config["method"]["pseudopotentials"]["Si"]).resolve()
    )
    config.setdefault("control", {})["outdir"] = str(workdir_sym / "output")

    with open(workdir_sym / "nc.yml", "w") as f:
        yaml.safe_dump(config, f)

    result_sym = run_relax_workflow(
        workdir_sym / "nc.yml",
        fmax=SI_FMAX_RY_PER_BOHR * RY_TO_EV / 0.529177,
        steps=20,
        use_symmetry=True,
        mode=mode,
        q3_binary=q3_command[0],
        workdir=workdir_sym / "relax_workdir",
    )
    assert result_sym["converged"], f"Symmetric relaxation failed to converge under mode {mode}"

    # 2. Run WITHOUT symmetry
    workdir_nosym = tmp_path / f"relax_nosym_{mode}"
    workdir_nosym.mkdir()

    with open(workdir_nosym / "material.json", "w") as f:
        json.dump(material, f)

    config["material"] = str(workdir_nosym / "material.json")
    config["control"]["outdir"] = str(workdir_nosym / "output")

    with open(workdir_nosym / "nc.yml", "w") as f:
        yaml.safe_dump(config, f)

    result_nosym = run_relax_workflow(
        workdir_nosym / "nc.yml",
        fmax=SI_FMAX_RY_PER_BOHR * RY_TO_EV / 0.529177,
        steps=20,
        use_symmetry=False,
        mode=mode,
        q3_binary=q3_command[0],
        workdir=workdir_nosym / "relax_workdir",
    )
    assert result_nosym["converged"], f"Asymmetric relaxation failed to converge under mode {mode}"

    # Verify that energy and forces match between modes, and compare steps
    assert abs(result_sym["final_energy_ry"] - result_nosym["final_energy_ry"]) < 1e-4
    print(
        f"\n[{mode}] Steps with symmetry: {result_sym['steps']}, without symmetry: {result_nosym['steps']}"
    )
    assert result_sym["steps"] > 0
    assert result_nosym["steps"] > 0


# Reference energies for the supercell configurations at equilibrium
SUPERCELL_REFERENCE_ENERGY_RY = {
    ("silicon", "nc"): -455.77312593,
    ("silicon", "us"): -423.49451417,
    ("copper", "nc"): -10438.57534364,
    ("copper", "us"): -10902.86409824,
}


@pytest.mark.skip(reason="postponed: supercell tests are too slow")
@pytest.mark.parametrize(
    "material,flavor",
    [
        ("silicon", "nc"),
        ("silicon", "us"),
        ("copper", "nc"),
        ("copper", "us"),
    ],
)
def test_supercell_relaxation_at_equilibrium(q3_command, tmp_path, material, flavor):
    """
    Validate that relaxation of supercells at equilibrium converges in 0 steps
    and matches the reference total energy.
    """
    pytest.importorskip("q3_core", reason="q3_core C++ bindings not built")
    workdir = tmp_path / f"supercell_eq_{material}_{flavor}"
    workdir.mkdir()

    # Locate the yml and material files
    example_folder = (
        EXAMPLES_DIRECTORY
        / f"total_energy--{'semiconductor' if material == 'silicon' else 'metal'}--{material}"
    )

    # Read the equilibrium material JSON from total_energy examples
    with open(EXAMPLES_DIRECTORY / f"data/materials/{material}_3x3x3.json") as f:
        material_data = json.load(f)
    with open(workdir / "material.json", "w") as f:
        json.dump(material_data, f)

    # Read the total energy yml as a base config, modify it to be a relaxation
    with open(example_folder / f"{flavor}_3x3x3.yml") as f:
        config = yaml.safe_load(f)

    config["material"] = str(workdir / "material.json")

    # Fix the pseudopotentials relative paths to be absolute
    for species, pp_path in config["method"]["pseudopotentials"].items():
        config["method"]["pseudopotentials"][species] = str((example_folder / pp_path).resolve())

    config["control"]["calculation"] = "relaxation"
    config["control"]["outdir"] = str(workdir / "output")
    config["control"]["relax_max_steps"] = 10
    config["control"]["relax_force_threshold"] = 5.0e-3
    config["control"]["relax_energy_threshold"] = 1.0e-5

    # Relax threshold for convergence in eV/Angstrom for ASE
    fmax_ev_per_angstrom = 5.0e-3 * RY_TO_EV / 0.529177

    with open(workdir / "relax.yml", "w") as f:
        yaml.safe_dump(config, f)

    result = run_relax_workflow(
        workdir / "relax.yml",
        fmax=fmax_ev_per_angstrom,
        steps=5,
        mode="bindings",
        q3_binary=q3_command[0],
        workdir=workdir / "relax_workdir",
    )

    assert result["converged"]
    assert result["steps"] == 0
    ref_energy = SUPERCELL_REFERENCE_ENERGY_RY[(material, flavor)]
    assert abs(result["final_energy_ry"] - ref_energy) < 1.0e-2


@pytest.mark.skip(reason="postponed: supercell tests are too slow")
@pytest.mark.parametrize(
    "material,flavor",
    [
        ("silicon", "nc"),
        ("copper", "us"),
    ],
)
def test_supercell_relaxation_displaced(q3_command, tmp_path, material, flavor):
    """
    Validate that running a single step of relaxation on the displaced supercells
    works without crashes and takes exactly 1 step.
    """
    pytest.importorskip("q3_core", reason="q3_core C++ bindings not built")

    workdir = tmp_path / f"supercell_disp_{material}_{flavor}"
    workdir.mkdir()

    example_folder = (
        EXAMPLES_DIRECTORY
        / f"relaxation--{'semiconductor' if material == 'silicon' else 'metal'}--{material}"
    )

    # Read the config file from relaxation examples folder
    with open(example_folder / f"{flavor}_3x3x3.yml") as f:
        config = yaml.safe_load(f)

    # Use the displaced material JSON we generated
    config["material"] = str(EXAMPLES_DIRECTORY / f"data/materials/{material}_3x3x3_displaced.json")

    # Fix the pseudopotentials relative paths to be absolute
    for species, pp_path in config["method"]["pseudopotentials"].items():
        config["method"]["pseudopotentials"][species] = str((example_folder / pp_path).resolve())

    config["control"]["outdir"] = str(workdir / "output")

    with open(workdir / "relax.yml", "w") as f:
        yaml.safe_dump(config, f)

    result = run_relax_workflow(
        workdir / "relax.yml",
        fmax=5.0e-3 * RY_TO_EV / 0.529177,
        steps=1,
        mode="bindings",
        q3_binary=q3_command[0],
        workdir=workdir / "relax_workdir",
    )

    assert result["steps"] == 1
    assert result["final_energy_ry"] is not None
