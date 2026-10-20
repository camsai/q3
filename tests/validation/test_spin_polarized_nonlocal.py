"""
The nonlocal D matrix, and everything downstream of it, per spin channel.

``D_ij = D_ion + integral(v_eff Q_ij)`` inherits the spin dependence of
``v_eff``, so ultrasoft and PAW need one D per channel. Until 2026-08-08 Q3
built a single D from the spin-averaged potential and shared it, and split the
augmentation charge between the channels by the ratio of the *smooth* densities
rather than building it per channel from each channel's own occupancies. Both
are exact for a non-magnetic cell and destroy the moment in a magnetic one, and
norm-conserving is immune because its D_ion carries no v_eff dependence -- which
is why every LSDA test in this repository passed and said nothing about it.

Measured on this branch, same cell and dataset in each row:

===========================  =================  ==================
case                         before             after
===========================  =================  ==================
Fe ultrasoft/GBRV, moment    0.0144 uB          2.2482 uB
Fe ultrasoft/GBRV, energy    473 meV/atom off   0.06 meV/atom off
===========================  =================  ==================

Two kinds of gate here, and both are needed:

* **collapse** -- a non-magnetic system at ``nspin=2`` must reproduce its
  ``nspin=1`` energy exactly. This is the cheap one, it needs no reference
  code, and it catches any accounting error in the spin plumbing.
* **magnitude** -- a genuinely magnetic system must reproduce a reference
  code's moment. Collapse alone cannot see the bug this file exists for: a
  spin-averaged D collapses perfectly.

Run:
    .venv/bin/pytest tests/validation/test_spin_polarized_nonlocal.py -v --q3-bin ./build/q3
"""

import json
import re
import subprocess
import tarfile
from pathlib import Path

import pytest
from spin_polarization_helpers import (
    COLLAPSE_TOLERANCE_RYDBERG,
    parse_total_magnetization,
    write_input,
    write_material,
)

from helpers import (
    BOHR_RADIUS_ANGSTROM,
    EXAMPLES_DIRECTORY,
    Q3_ROOT,
    run_overlay_and_read_energy,
    run_overlay_and_read_stdout,
)

CU_ULTRASOFT_INPUT = EXAMPLES_DIRECTORY / "total_energy--metal--copper" / "us.yml"
CU_PREFIX = "cu_us"

ULTRASOFT_PSEUDOPOTENTIAL_ARCHIVE = (
    Q3_ROOT.parent / "reference" / "mat3ra" / "pseudos" / "data" / "pseudo_gga_pbe.tar.gz"
)
"""
GBRV ultrasoft datasets, resolved from the workspace reference tree.

Iron is not among the datasets committed under ``examples/data``, and it is the
only magnetic ultrasoft case available, so the magnitude gate for that path
extracts it here and skips when the archive is absent.
"""

IRON_ULTRASOFT_DATASET = "fe_pbe_v1.5.uspp.F.UPF"

IRON_ULTRASOFT_REFERENCE = {
    "description": (
        "Quantum ESPRESSO PWSCF 7.5, build_libxc (LibXC 7.0.0), same GBRV "
        "fe_pbe_v1.5 dataset, ibrav=3 celldm(1)=5.41711845, ecutwfc=40 "
        "ecutrho=320, 6x6x6 shifted, gaussian degauss=0.02 Ry, nspin=2, "
        "starting_magnetization=0.5, conv_thr=1e-8. Measured 2026-08-08."
    ),
    "total_energy_rydberg": -250.34529817,
    "total_magnetization_bohr_magnetons": 2.25,
    "energy_tolerance_rydberg": 1.0e-5,
    "magnetization_tolerance_bohr_magnetons": 0.02,
}
"""
Iron is the magnitude gate for the ultrasoft path.

The energy tolerance is 1e-5 Ry against a measured 4e-6 Ry agreement -- the two
codes run the same functional through the same library on the same dataset, so
there is no physical reason for them to differ, and a loose bound here would
stop being a gate. The moment tolerance is set by QE printing two decimals.
"""


@pytest.fixture(scope="module")
def iron_ultrasoft_dataset(tmp_path_factory):
    if not ULTRASOFT_PSEUDOPOTENTIAL_ARCHIVE.is_file():
        pytest.skip(f"no GBRV archive at {ULTRASOFT_PSEUDOPOTENTIAL_ARCHIVE}")
    destination = tmp_path_factory.mktemp("gbrv") / IRON_ULTRASOFT_DATASET
    with tarfile.open(ULTRASOFT_PSEUDOPOTENTIAL_ARCHIVE) as archive:
        member = next(
            (m for m in archive.getmembers() if Path(m.name).name == IRON_ULTRASOFT_DATASET),
            None,
        )
        if member is None:
            pytest.skip(f"{IRON_ULTRASOFT_DATASET} not in {ULTRASOFT_PSEUDOPOTENTIAL_ARCHIVE}")
        extracted = archive.extractfile(member)
        destination.write_bytes(extracted.read())
    return destination


# ─── Collapse gates: no reference code needed, run everywhere ────────────────


def test_ultrasoft_spin_polarized_collapses_to_unpolarized(q3_command, tmp_path):
    """
    Copper ultrasoft at nspin=2 with zero starting moment equals nspin=1.

    Copper because it is the ultrasoft case already committed as an example, so
    this needs no external dataset and runs in CI.
    """
    unpolarized = run_overlay_and_read_energy(
        q3_command, CU_ULTRASOFT_INPUT, None, CU_PREFIX, tmp_path / "unpolarized"
    )
    polarized = run_overlay_and_read_energy(
        q3_command,
        CU_ULTRASOFT_INPUT,
        {
            "method": {
                "scf": {
                    "number_of_spin_components": 2,
                    "starting_magnetization": {"Cu": 0.0},
                }
            }
        },
        CU_PREFIX,
        tmp_path / "polarized",
    )
    difference = abs(polarized - unpolarized)
    assert difference < COLLAPSE_TOLERANCE_RYDBERG, (
        "ultrasoft nspin=2 does not reproduce nspin=1 for a non-magnetic cell:\n"
        f"  E(nspin=1) = {unpolarized:.12f} Ry\n"
        f"  E(nspin=2) = {polarized:.12f} Ry\n"
        f"  |diff|     = {difference:.2e} Ry"
    )


# ─── Magnitude gates: a real moment against a real reference ────────────────


def test_magnetic_ultrasoft_iron_matches_quantum_espresso(
    q3_command, tmp_path, iron_ultrasoft_dataset
):
    """
    Ferromagnetic bcc iron, ultrasoft, against QE with the same dataset.

    This is the case the previous session could not run: no magnetic ultrasoft
    dataset shipped with Q3, so the error on that path was established from the
    code and never measured. With the spin-averaged D it converged to 0.0144 uB
    and an energy 473 meV/atom away from QE.
    """
    material_path = tmp_path / "iron.json"
    write_material(material_path, "Fe", "BCC", 2.4825, 109.4712206)
    input_path = tmp_path / "iron.yml"
    write_input(
        input_path,
        material_path,
        "ultrasoft",
        iron_ultrasoft_dataset,
        "Fe",
        tmp_path / "output",
        [
            "kgrid: {dimensions: [6, 6, 6], shifts: [1, 1, 1]}",
            "number_of_bands: 20",
            "number_of_spin_components: 2",
            "starting_magnetization: {Fe: 0.5}",
        ],
    )

    standard_output = run_overlay_and_read_stdout(q3_command, input_path, None, tmp_path)
    moment = parse_total_magnetization(standard_output)
    energy = run_overlay_and_read_energy(
        q3_command, input_path, None, "spin_case", tmp_path, timeout=1800
    )

    reference = IRON_ULTRASOFT_REFERENCE
    print(
        f"\n  Fe ultrasoft: Q3 E={energy:.8f} Ry m={moment:.4f} uB, "
        f"QE E={reference['total_energy_rydberg']:.8f} Ry "
        f"m={reference['total_magnetization_bohr_magnetons']:.2f} uB"
    )
    assert (
        abs(moment - reference["total_magnetization_bohr_magnetons"])
        <= reference["magnetization_tolerance_bohr_magnetons"]
    ), f"Fe ultrasoft moment {moment:.4f} uB does not match QE"
    assert (
        abs(energy - reference["total_energy_rydberg"]) <= reference["energy_tolerance_rydberg"]
    ), f"Fe ultrasoft energy {energy:.8f} Ry does not match QE"


def test_recorded_tolerances_bracket_the_measurements():
    """
    The manifests above stay gates rather than formalities.

    Runs without any pseudopotential or reference code, so CI checks it even
    where the licensed and external data are absent.
    """
    for name, reference, measured_deltas in (
        (
            "Fe ultrasoft",
            IRON_ULTRASOFT_REFERENCE,
            {"energy_tolerance_rydberg": 4.1e-6},
        ),
    ):
        for key, measured in measured_deltas.items():
            assert reference[key] >= measured, f"{name}: {key} excludes its own measurement"
            assert reference[key] <= 100.0 * measured, f"{name}: {key} is too loose to be a gate"
        assert (
            reference["magnetization_tolerance_bohr_magnetons"]
            < 0.5 * reference["total_magnetization_bohr_magnetons"]
        ), f"{name}: a moment tolerance this wide would admit a collapsed moment"


# ─── The augmentation force must resolve spin too ────────────────────────────

"""
The augmentation force and stress were built from the *spin-averaged* potential
and the *spin-summed* projector occupancies until 2026-08-09, while their
nonlocal partners -- fixed the day before, and the subject of the tests above --
carried a spin index. The difference is

    integral( (V_up - V_down)(rho_up - rho_down) dQ/dR ) / 2

identically zero without a moment, so every unpolarized test passed, and
non-zero with one. The two halves of the same pseudopotential derivative
therefore disagreed for exactly the magnetic cells this file exists to check.

Two conditions are needed to see it at all, and both are easy to get wrong:
the cell must carry a real moment, and it must have a non-zero force. A
one-atom bcc cell has every force component zero by symmetry; the conventional
two-atom cell with one atom displaced has neither problem.
"""

BASE_DISPLACEMENT_BOHR = 0.30
"""
How far the second atom sits from its lattice site in *every* run, including
the one the analytic force is read from.

Without this the test measures nothing: the body-centred site has full cubic
point symmetry, so E(+delta) = E(-delta) exactly and the centred difference is
identically zero -- the first attempt returned 3e-6 Ry/bohr and the assertion
below is what caught it. The force has to be evaluated somewhere the symmetry
does not already force it to vanish.
"""

AUGMENTATION_DISPLACEMENT_BOHR = 0.02
"""The finite-difference step taken *around* the base displacement. Small
enough for the central difference to stay quadratic, large enough to lift the
force clear of SCF noise -- the value the PAW force tests use."""


def write_two_atom_iron(path, displacement_bohr):
    """Conventional bcc iron with the second atom displaced along x."""
    lattice_constant_angstrom = 2.87
    displacement_fraction = (
        displacement_bohr * BOHR_RADIUS_ANGSTROM / lattice_constant_angstrom
        if displacement_bohr
        else 0.0
    )
    path.write_text(
        json.dumps(
            {
                "name": "Fe",
                "lattice": {
                    "type": "CUB",
                    "a": lattice_constant_angstrom,
                    "b": lattice_constant_angstrom,
                    "c": lattice_constant_angstrom,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                    "units": {"length": "angstrom", "angle": "degree"},
                },
                "basis": {
                    "units": "crystal",
                    "elements": [{"id": 0, "value": "Fe"}, {"id": 1, "value": "Fe"}],
                    "coordinates": [
                        {"id": 0, "value": [0.0, 0.0, 0.0]},
                        {"id": 1, "value": [0.5 + displacement_fraction, 0.5, 0.5]},
                    ],
                },
                "isNonPeriodic": False,
            }
        )
    )


@pytest.fixture(scope="module")
def magnetic_iron_displacement_runs(q3_command, tmp_path_factory, iron_ultrasoft_dataset):
    """The centred displacement triple, run once and shared."""
    outputs = {}
    for label, displacement in (
        ("baseline", BASE_DISPLACEMENT_BOHR),
        ("plus", BASE_DISPLACEMENT_BOHR + AUGMENTATION_DISPLACEMENT_BOHR),
        ("minus", BASE_DISPLACEMENT_BOHR - AUGMENTATION_DISPLACEMENT_BOHR),
    ):
        directory = tmp_path_factory.mktemp(f"fe2_{label}")
        material_path = directory / "iron2.json"
        write_two_atom_iron(material_path, displacement)
        input_path = directory / "iron2.yml"
        write_input(
            input_path,
            material_path,
            "ultrasoft",
            iron_ultrasoft_dataset,
            "Fe",
            directory / "output",
            [
                "kgrid: {dimensions: [4, 4, 4], shifts: [0, 0, 0]}",
                "number_of_spin_components: 2",
                "starting_magnetization: {Fe: 0.5}",
            ],
        )
        completed = subprocess.run(
            list(q3_command) + [str(input_path)],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert completed.returncode == 0, (
            f"q3 exited {completed.returncode} on the {label} run\n{completed.stdout[-3000:]}"
        )
        results = json.loads((directory / "output" / "spin_case_results.json").read_text())
        assert results["converged"], (
            f"the {label} run did not converge; a finite-difference force needs three converged "
            "energies, and an unconverged run reports no forces at all -- which surfaces "
            "downstream as an empty force list rather than as the real cause"
        )
        outputs[label] = (completed.stdout, results)
    return outputs


def test_the_displaced_iron_cell_carries_a_moment(magnetic_iron_displacement_runs):
    """
    The premise of the test below.

    With no moment the spin-resolved and spin-averaged augmentation terms are
    identical, and the finite-difference check would pass whether or not the
    fix were present -- a green test proving nothing.

    "absolute magnetization" has no results.json field (unlike "total
    magnetization", also stdout-only -- see parse_total_magnetization()'s
    docstring), so this stays stdout-based.
    """
    baseline_stdout, _ = magnetic_iron_displacement_runs["baseline"]
    matches = re.findall(r"absolute magnetization\s*=\s*(-?\d+\.\d+)", baseline_stdout)
    assert matches, "no absolute magnetization reported"
    assert float(matches[-1]) > 1.0, (
        f"the cell converged to |m| = {matches[-1]} muB, too small to exercise the spin "
        "resolution this test exists to check"
    )


def test_magnetic_augmentation_force_is_the_gradient_of_q3s_own_energy(
    magnetic_iron_displacement_runs,
):
    """
    The analytic force on the displaced atom against a central difference of
    Q3's own total energy, in a cell with a moment.

    This is the check that fails when the augmentation term is spin-blind: the
    energy is built from the spin-resolved density either way, so a mismatch
    localizes to the force assembly rather than to the energy. The same
    diagnostic found the missing dQ/dR and generalized-eigenvalue terms.

    Measured on this cell: the spin-blind assembly gives -0.04543 Ry/bohr
    against a finite difference of -0.04366, a ratio of 1.0406; resolving spin
    gives -0.04390, ratio 1.0056. The finite difference is *identical* in both
    -- the energy was never in question -- so the 4% is the missing
    (V_up - V_down)(rho_up - rho_down) cross term and nothing else. Verified by
    disabling the spin resolution and re-running, which is the only way to know
    this test discriminates rather than merely passing.
    """
    _, baseline_results = magnetic_iron_displacement_runs["baseline"]
    _, plus_results = magnetic_iron_displacement_runs["plus"]
    _, minus_results = magnetic_iron_displacement_runs["minus"]
    analytic = baseline_results["forces_ry_per_bohr"][1][0]
    numerical = -(plus_results["total_energy_ry"] - minus_results["total_energy_ry"]) / (
        2.0 * AUGMENTATION_DISPLACEMENT_BOHR
    )

    assert abs(numerical) > 1e-3, (
        f"the finite-difference force is only {numerical:.2e} Ry/bohr -- the displacement is "
        "not producing a force this test can resolve"
    )
    ratio = analytic / numerical
    assert abs(ratio - 1.0) < 0.03, (
        f"magnetic ultrasoft force {analytic:.8f} Ry/bohr vs finite difference "
        f"{numerical:.8f} Ry/bohr (ratio {ratio:.6f})"
    )
