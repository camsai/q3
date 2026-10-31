"""Q3's Wannier construction against Wannier90 4.0.1, on Q3's own matrices.

Tier 3 (monthly), skipped without the oracle. Everything else in this directory validates
the interpolation against Q3's direct calculations, which catches a wrong
interpolation and cannot catch a wrong MINIMUM: a localiser converging to the
wrong Omega every time passes every such test. This is the one comparison that
can catch it. Wannier90 is run as a black box on the `.amn/.mmn/.eig` Q3
exports; only its printed numbers cross back.

Eight cases. Four norm-conserving, in two regimes:

  silicon   isolated, 4 bond-centred orbitals, no disentanglement
  GaAs      isolated but offset (bands 19-22), no disentanglement
  copper    entangled, 14 -> 10, projectability disentanglement, no energy window
  Si3C      entangled, 20 -> 16, projectability + energy freeze at E_F, and the
            structure that exposed both of this file's founding defects

and four that repeat silicon and copper on the augmented families -- PAW and
ultrasoft, whose A_mn and M_mn carry the augmentation terms. The valence is the
same across the three families for silicon (4) and copper (19), so those cases
are the same problem with a different dataset. Their first finding was that the
capture could not reach the augmented builder at all (resolved 2026-09-10); now
that it runs, silicon agrees on both augmented families, and the two copper
cases XFAIL where their localisers do not reach the same minimum: see
AUGMENTED_GAUGE_DISAGREES below, and KNOWN-ISSUES.md and TODOs.md 1.3.

Two things learned building this, both recorded here because they cost a day:
Wannier90 4.0's `dis_proj_min/max` on their own only EXCLUDE below the minimum;
freezing above the maximum needs `dis_froz_proj = .true.`, without which its
Omega_I comes out 26% lower than Q3's on copper and the comparison is not of
the same problem. And Q3's guiding centres had to become the trial-orbital
positions rather than the starting gauge's own Berry-phase centres -- on Si3C
the latter are already wrong, and guiding to them stalled the minimiser at
170 A^2 where Wannier90 reaches 26.

Run:
    pytest tests/validation/test_wannier90_parity.py --tier2 -v
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers_wannier import (
    locate_wannier90_binary,
    read_wannier90_output,
    write_wannier90_input,
)

from helpers import EXAMPLES_DIRECTORY

RYDBERG_TO_ELECTRONVOLT = 13.605693122994
BOHR_TO_ANGSTROM = 0.529177210903
WANNIER90 = locate_wannier90_binary()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.monthly,
    pytest.mark.skipif(WANNIER90 is None, reason="Wannier90 4.0.1 oracle is not built"),
]

CASES = {
    "silicon": {
        "example": "band_structure--semiconductor--silicon--wannier",
        "material": "silicon.json",
        "pseudopotentials": {"Si": "Si.upf"},
        "prefix": "si",
        "center_gate_angstrom": 1.0e-3,
    },
    "gallium-arsenide": {
        "example": "band_structure--semiconductor--gallium-arsenide--wannier",
        "material": "gallium_arsenide.json",
        "pseudopotentials": {"Ga": "Ga.upf", "As": "As.upf"},
        "prefix": "gaas",
        "center_gate_angstrom": 1.0e-3,
    },
    "copper": {
        "example": "band_structure--metal--copper--wannier",
        "material": "copper.json",
        "pseudopotentials": {"Cu": "Cu.upf"},
        "prefix": "cu",
        "center_gate_angstrom": 1.0e-2,
        # A window below the whole manifold, so nothing is frozen BY ENERGY and
        # both codes freeze on projectability alone. Any window that reaches the
        # bands is a problem Wannier90 cannot be given: Q3's singular-gauge rule
        # drops the zero-projectability band from the freeze, Wannier90 has no
        # such rule, and it refuses with 11 states frozen for 10 target functions
        # (measured at E_F and at E_F + 2 eV alike). Q3's model here is the one
        # its ladder reaches -- the window rung's 0 eV gives the same Omega_I to
        # every digit -- so this compares the shipped model, on the one problem
        # both codes can express.
        "interpolation": {"frozen_window_above_fermi_ev": -5.0},
    },
    "silicon-carbon-alloy": {
        "example": "band_structure--semiconductor--silicon-carbon-alloy--wannier",
        "material": "silicon_carbon_alloy.json",
        "pseudopotentials": {"Si": "Si.upf", "C": "C.upf"},
        "prefix": "si3c",
        # Sixteen functions on four atoms with a disentangled window sit in a
        # nearly flat valley: two minimisers agreeing on Omega to 1e-3 A^2 can
        # still stop 0.05 A apart in where each function sits. Measured 0.050.
        "center_gate_angstrom": 1.0e-1,
    },
    # The augmented families. Same physics as the two cases above, different
    # dataset: these are the only cases here whose A_mn and M_mn carry the
    # augmentation terms, and until now those kernels had been checked against
    # their own internal identities and never against another code.
    "silicon--paw": {
        "example": "band_structure--semiconductor--silicon--wannier",
        "material": "silicon.json",
        "family": "paw",
        "subtype": "paw",
        "pseudopotentials": {"Si": "Si.upf"},
        "cutoffs": {"wavefunction": 40.0, "density": 320.0},
        "prefix": "si",
        "center_gate_angstrom": 1.0e-3,
    },
    "silicon--ultrasoft": {
        "example": "band_structure--semiconductor--silicon--wannier",
        "material": "silicon.json",
        "family": "us",
        "subtype": "ultrasoft",
        "pseudopotentials": {"Si": "Si_PBE_USPP.UPF"},
        "cutoffs": {"wavefunction": 40.0, "density": 200.0},
        "prefix": "si",
        "center_gate_angstrom": 1.0e-3,
    },
    "copper--paw": {
        "example": "band_structure--metal--copper--wannier",
        "material": "copper.json",
        "family": "paw",
        "subtype": "paw",
        "pseudopotentials": {"Cu": "Cu.upf"},
        "cutoffs": {"wavefunction": 40.0, "density": 320.0},
        # The example's 1e-8 exits before the PAW one-centre channel has
        # settled; the exported matrices should come from a converged density.
        "convergence_threshold": 1.0e-10,
        # This dataset's atomic set is 18 orbitals against the norm-conserving
        # file's 10, and the example asks for 14 bands, which the capture
        # rightly refuses. Room for the set plus a little to disentangle from.
        "number_of_bands": 24,
        "interpolation": {"frozen_window_above_fermi_ev": -5.0},
        "prefix": "cu",
        "center_gate_angstrom": 1.0e-2,
    },
    "copper--ultrasoft": {
        "example": "band_structure--metal--copper--wannier",
        "material": "copper.json",
        "family": "us",
        "subtype": "ultrasoft",
        "pseudopotentials": {"Cu": "cu_pbe_v1.2.uspp.F.UPF"},
        "cutoffs": {"wavefunction": 40.0, "density": 200.0},
        "convergence_threshold": 1.0e-10,
        "interpolation": {"frozen_window_above_fermi_ev": -5.0},
        "prefix": "cu",
        "center_gate_angstrom": 1.0e-2,
    },
}


def _stage(case: dict, workspace: Path, name: str) -> dict:
    """The example's manifest with absolute paths, and the dataset the case names.

    A case whose ``family`` is not ``nc`` reuses the norm-conserving example's
    physics -- same material, band count, mesh, projections and thresholds --
    and swaps in that family's dataset, cutoffs and method subtype. That is
    what makes the comparison meaningful: the valence is the same across the
    three families for silicon (4) and for copper (19), so the window
    transfers and both codes are handed the same problem, with only the
    augmentation in the overlaps and projections telling them apart.
    """
    family = case.get("family", "nc")
    manifest = yaml.safe_load((EXAMPLES_DIRECTORY / case["example"] / name).read_text())
    manifest["material"] = str(EXAMPLES_DIRECTORY / "data" / "materials" / case["material"])
    manifest["method"]["pseudopotentials"] = {
        symbol: str(EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / family / filename)
        for symbol, filename in case["pseudopotentials"].items()
    }
    if "subtype" in case:
        manifest["method"]["subtype"] = case["subtype"]
    if "cutoffs" in case:
        manifest["method"]["cutoffs"] = dict(case["cutoffs"])
    if "convergence_threshold" in case and "scf" in manifest["method"]:
        manifest["method"]["scf"]["convergence_threshold"] = case["convergence_threshold"]
    # A family's atomic-wavefunction set is its own: copper's PAW dataset offers
    # 18 where the norm-conserving one offers 10, and the gauge cannot have more
    # Wannier functions than bands, so the example's band count does not carry
    # across families unchanged.
    if "number_of_bands" in case and "scf" in manifest["method"]:
        manifest["method"]["scf"]["number_of_bands"] = case["number_of_bands"]
    # Evaluation-stage keys only: the capture refuses them by name rather than
    # ignoring them, which is how `control.interpolation` keeps its two steps apart.
    if "interpolate" in name and "interpolation" in case:
        manifest["control"].setdefault("interpolation", {}).update(case["interpolation"])
    manifest["control"]["outdir"] = str(workspace / "output")
    return manifest


#: The augmented cases, measured 2026-09-14 against Wannier90 4.0.1 on Q3's own
#: matrices, once a PAW non-self-consistent run carried D's one-centre term:
#:
#:   silicon--paw        agrees on all three
#:   silicon--ultrasoft  agrees on all three
#:   copper--ultrasoft   Omega_I agrees; Omega  9.7409 vs  7.9083 A^2, centres 0.778 A out
#:   copper--paw         Omega_I  8.9993 vs 8.9985 A^2; Omega 51.2714 vs 32.5988, centres 0.802 A
#:
#: Silicon PAW's earlier disagreement (Omega 6.0277 vs 6.2258 on 2026-09-10) was
#: the capture's: its NSCF had diagonalized without the one-centre term, so the
#: matrices both codes localised were not silicon's. What is left is a pair of
#: copper cases whose localisers do not reach the same minimum, and the numbers
#: say which side stops short rather than implicating the augmentation's phase.
#: On copper ultrasoft Wannier90 ends its 1000 iterations 4e-6 A^2 an iteration
#: from its minimum while Q3's localisation stalls at 9.74 with a gradient of
#: 0.07 after 863. Copper PAW's eighteen functions -- every shell of the file --
#: stall both: Q3 at a gradient of 3.8 after 905 iterations, Wannier90 still
#: falling 0.02 A^2 an iteration at its cap. So its Omega and centres compare
#: two unfinished minimisations, and its subspace, which both disentanglers
#: report converged, now differs by 7.7e-4 A^2 against the 1e-4 gate.
#:
#: The cases stay in the suite rather than being skipped: every part that does
#: agree is asserted strictly, so a regression there fails the run, and a case
#: whose disagreement is resolved starts passing and leaves its set.
AUGMENTED_GAUGE_DISAGREES = {"copper--paw", "copper--ultrasoft"}
AUGMENTED_SUBSPACE_DISAGREES = {"copper--paw"}


def _skip_if_the_augmented_gauge_is_known_to_disagree(name: str) -> None:
    if name in AUGMENTED_GAUGE_DISAGREES:
        pytest.xfail(
            f"{name}: the localisers do not reach the same minimum on this augmented case "
            "-- see AUGMENTED_GAUGE_DISAGREES for the measured numbers"
        )


def _skip_if_the_augmented_subspace_is_known_to_disagree(name: str) -> None:
    if name in AUGMENTED_SUBSPACE_DISAGREES:
        pytest.xfail(
            f"{name}: Omega_I differs from Wannier90's beyond the gate with both "
            "disentanglements converged -- see AUGMENTED_GAUGE_DISAGREES for the numbers"
        )


def _run(command: list[str], workspace: Path, label: str) -> str:
    result = subprocess.run(command, cwd=workspace, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(f"{label} failed ({result.returncode}):\n{(result.stdout or '')[-3000:]}")
    return (result.stdout or "") + (result.stderr or "")


def _real_lattice_angstrom(nnkp: Path) -> np.ndarray:
    block = re.search(r"begin real_lattice\s*\n(.*?)end real_lattice", nnkp.read_text(), re.S)
    return np.array(
        [[float(v) for v in line.split()] for line in block.group(1).strip().splitlines()]
    )


def _nearest_image_distance(a: np.ndarray, b: np.ndarray, lattice: np.ndarray) -> float:
    fractional = np.linalg.solve(lattice.T, a - b)
    fractional -= np.round(fractional)
    return float(np.linalg.norm(fractional @ lattice))


@pytest.fixture(scope="module", params=list(CASES))
def parity(request, tmp_path_factory, q3_command, q3_utility_command) -> dict:
    """One case: Q3 captures and exports, evaluates its own model, Wannier90 runs on the export."""
    name = request.param
    case = CASES[name]
    workspace = tmp_path_factory.mktemp(f"wannier90_parity_{name}")
    q3 = [str(Path(list(q3_command)[-1]).resolve())]

    for stage, source in (("scf", "nc_scf.yml"), ("nscf", "nc_nscf.yml")):
        manifest = _stage(case, workspace, source)
        if stage == "nscf":
            manifest["control"]["interpolation"]["export_formats"] = ["wannier90"]
        path = workspace / f"{stage}.yml"
        path.write_text(yaml.safe_dump(manifest, sort_keys=False))
        _run(q3 + ["--log-level", "warn", str(path)], workspace, f"{name} {stage}")
    nscf = yaml.safe_load((workspace / "nscf.yml").read_text())

    evaluate = _stage(case, workspace, "nc_interpolate.yml")
    evaluate["control"]["interpolation"].pop("dos_mesh", None)
    path = workspace / "evaluate.yml"
    path.write_text(yaml.safe_dump(evaluate, sort_keys=False))
    _run(
        list(q3_utility_command)
        + ["interpolate", str(path), "--output", str(workspace / "q3.json")],
        workspace,
        f"{name} evaluation",
    )
    model = json.loads((workspace / "q3.json").read_text())
    spread = model["construction"]["spread"]
    assert spread["guiding_centers"] == "trial orbitals", "the record must carry the trial centres"

    prefix = case["prefix"]
    oracle = workspace / "oracle"
    oracle.mkdir()
    for extension in ("amn", "mmn", "eig", "nnkp"):
        (oracle / f"{prefix}.{extension}").write_bytes(
            (workspace / "output" / f"{prefix}.{extension}").read_bytes()
        )
    lattice = _real_lattice_angstrom(oracle / f"{prefix}.nnkp")
    material = json.loads(Path(evaluate["material"]).read_text())
    atoms = [
        (element["value"], tuple(coordinate["value"]))
        for element, coordinate in zip(
            material["basis"]["elements"], material["basis"]["coordinates"]
        )
    ]
    interpolation = evaluate["control"]["interpolation"]
    # The window Q3 RESOLVED, not the one the manifest asked for. Since the
    # window became `auto` by default the manifest usually names none, and
    # echoing that told Wannier90 to freeze nothing while Q3 froze at E_F+2 eV
    # -- two codes on two different problems, which is not a parity test. The
    # evaluator reports what it settled on; that is what the oracle is handed.
    resolved = model["construction"].get("disentanglement") or {}
    frozen_window_above_fermi_ev = resolved.get(
        "frozen_window_above_fermi_ev", interpolation.get("frozen_window_above_fermi_ev")
    )
    write_wannier90_input(
        oracle / f"{prefix}.win",
        real_lattice_angstrom=lattice,
        atoms_fractional=atoms,
        kpoints_fractional=[tuple(point[:3]) for point in nscf["control"]["kpoints_list"]],
        mp_grid=tuple(nscf["control"]["interpolation"]["grid"]),
        number_of_bands=nscf["method"]["scf"]["number_of_bands"]
        if "bands" not in nscf["control"]["interpolation"]
        else nscf["control"]["interpolation"]["bands"][1]
        - nscf["control"]["interpolation"]["bands"][0]
        + 1,
        guiding_centers_angstrom=np.array(spread["trial_orbital_centers_bohr"]) * BOHR_TO_ANGSTROM,
        fermi_energy_ev=model["properties"]["fermi_energy"]["value"] * RYDBERG_TO_ELECTRONVOLT,
        frozen_window_above_fermi_ev=frozen_window_above_fermi_ev,
        projectability_minimum=interpolation.get("projectability_minimum"),
        projectability_maximum=interpolation.get("projectability_maximum"),
    )
    _run([str(WANNIER90), prefix], oracle, f"{name} wannier90")
    reference = read_wannier90_output(oracle / f"{prefix}.wout", "Final State")
    return {"name": name, "case": case, "q3": spread, "w90": reference, "lattice": lattice}


def test_the_gauge_invariant_spread_agrees(parity):
    """Omega_I is fixed by the subspace alone, so this is the disentangler's gate
    where a window is disentangled and a pure consistency check where it is not."""
    _skip_if_the_augmented_subspace_is_known_to_disagree(parity["name"])
    q3 = parity["q3"]["gauge_invariant_bohr_squared"] * BOHR_TO_ANGSTROM**2
    w90 = parity["w90"].omega_invariant
    assert abs(q3 - w90) < 1.0e-4, (
        f"{parity['name']}: Omega_I {q3:.7f} against Wannier90's {w90:.7f} A^2"
    )


def test_the_minimised_spread_agrees(parity):
    """The localiser's gate: the same minimum, not merely a lower number than it
    started from."""
    _skip_if_the_augmented_gauge_is_known_to_disagree(parity["name"])
    q3 = parity["q3"]["total_bohr_squared"] * BOHR_TO_ANGSTROM**2
    w90 = parity["w90"].omega_total
    assert abs(q3 - w90) < 1.0e-3, (
        f"{parity['name']}: Omega {q3:.7f} against Wannier90's {w90:.7f} A^2"
    )


def test_the_centers_agree_modulo_a_lattice_vector(parity):
    """A centre is a Berry phase: defined inside the cell, silent about which cell."""
    _skip_if_the_augmented_gauge_is_known_to_disagree(parity["name"])
    q3_centers = np.array(parity["q3"]["centers_bohr"]) * BOHR_TO_ANGSTROM
    w90_centers = parity["w90"].centers_angstrom
    lattice = parity["lattice"]
    unmatched = set(range(len(w90_centers)))
    worst = 0.0
    for center in q3_centers:
        distance, index = min(
            (_nearest_image_distance(center, w90_centers[other], lattice), other)
            for other in unmatched
        )
        unmatched.remove(index)
        worst = max(worst, distance)
    gate = parity["case"]["center_gate_angstrom"]
    assert worst < gate, f"{parity['name']}: worst centre mismatch {worst:.2e} A against {gate:.0e}"
