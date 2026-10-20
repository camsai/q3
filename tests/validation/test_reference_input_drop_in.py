"""
The drop-in claim, stated as a table: Q3 reads the reference codes' *own* input
files and reproduces the numbers those codes printed.

Every other comparison test in this suite generates the reference code's input
*from* Q3's YAML. That checks the physics but leaves one loophole open: if the
generator encodes a wrong translation, both sides agree on the wrong
calculation and no test notices. Here the reference code's file is primary and
Q3 consumes it byte for byte -- so a mis-read key is a failure rather than a
shared assumption.

The QE cases are the committed inputs that already sit in `examples/`, checked
against the numbers QE printed for them, recorded in a `reference.json` beside
each. Those runs used the LibXC build (`input_dft = 'XC-...'`) that AGENTS.md
requires for apples-to-apples comparison.

The reference numbers are read from JSON rather than from QE's `.out` files
because `*.out` is gitignored: a test reading them directly passes on the
machine that ran QE and fails on a fresh clone with a missing file. Where a
`.out` *is* present the test re-reads it and asserts the JSON still matches, so
a stale record cannot survive on any machine able to notice.
`refresh_qe_references.py` regenerates them.

Input files are staged into a temporary tree that reproduces their *relative*
layout rather than being rewritten, so the file Q3 reads is byte-identical to
the one QE ran -- rewriting `pseudo_dir` to an absolute path would quietly
weaken the very claim being tested.

**"Exactly the same" needs defining.** Bit-identical across codes is
impossible: different FFT libraries, different XC implementations, different
iteration paths. The defensible claim is matched-input agreement within stated
tolerances, and each case below carries its own measured number and the reason
it is where it is.

Run:
    .venv/bin/pytest tests/validation/test_reference_input_drop_in.py -v \
        --q3-bin ./build/q3
"""

import json
import re
import shutil
import subprocess

import pytest

from helpers import (
    EXAMPLES_DIRECTORY,
    RYDBERG_TO_ELECTRONVOLT,
    parse_q3_total_energy,
)

# ── Tolerances ────────────────────────────────────────────────────────

TOLERANCE_SAME_CODE_RYDBERG = 2e-4
"""
Q3 against QE on the same input file, for norm-conserving and ultrasoft.

Both codes solve the same equations with the same pseudopotential, cutoffs and
k-grid, so the residual is their different FFT grids, mixing histories and
diagonalization paths. Measured: silicon NC agrees to 2e-6 Ry, ultrasoft to
2e-5, Gamma-only to 4e-6. A failure at 1e-3 or above is a mapping difference,
not numerics -- the DFT+U double-counting bug was 0.9 Ry and the historical
functional-default bug was 6.6e-2 Ry.
"""

TOLERANCE_PAW_RYDBERG = 1e-3
"""
PAW is looser, and the reason is a known open gap rather than noise.

Q3's one-center terms carry a per-element residual against both reference
codes -- measured here at 6.9e-4 Ry on silicon (2 atoms), 2.9e-4 on diamond and
4.8e-4 on copper, all with Q3 *above* QE on the insulators and below on the
metal. The copper pair uses a 1e-12 density threshold because its oscillatory
late SCF trajectory can cross 1e-8 at platform-dependent energies. It is
tracked in KNOWN-ISSUES.md; pinning the gate at the measured one-center scale
makes a change visible instead of absorbing it as convergence noise.

Set from the measurements, not from what would be nice: tightening this without
fixing the one-center terms would only mean deleting the PAW cases.
"""

TOLERANCE_MOMENT_BOHR_MAGNETON = 0.02
"""
Magnetic moment against QE's reported magnetization.

QE prints it to two decimals, which sets the floor; the measured spread is
within that printing precision on iron, nickel and chromium.

Both of QE's numbers are compared where it has them. The *absolute*
magnetization is the one that matters for an antiferromagnet: the total is zero
for a converged AFM state and equally zero for one that collapsed to
non-magnetic, so a test that checked only the total would pass on a chromium
run that had lost its moments entirely.
"""

# ── Case tables ───────────────────────────────────────────────────────


def qe_case(example, family, stem="scf_qe", **expectations):
    """One committed QE input, identified by where it lives."""
    directory = EXAMPLES_DIRECTORY / example / "reference" / family
    return pytest.param(
        directory / f"{stem}.in",
        expectations,
        id=f"{example.split('--')[-1]}-{family}" + (f"-{stem}" if stem != "scf_qe" else ""),
    )


QE_CASES = [
    # Baseline: the simplest possible case, and the tightest agreement.
    qe_case("total_energy--semiconductor--silicon", "nc"),
    # A single k-point at Gamma. Q3 stores half the G sphere here and QE stores
    # all of it, so the two codes take genuinely different paths to the same
    # number -- which is why four Q3 terms could be wrong at once in this mode
    # while every denser grid stayed right.
    qe_case("total_energy--semiconductor--silicon", "nc", stem="scf_qe_gamma"),
    # Ultrasoft: augmentation charges, a second grid, and the S operator.
    qe_case("total_energy--semiconductor--silicon", "us"),
    # DFT+U through QE's own HUBBARD card, which the parser silently dropped
    # until 2026-08-08 -- a PAW+U input ran and reported the no-U energy. The
    # Hubbard energy is compared as well as the total, because a wrong E_U can
    # hide inside a total that has other terms to trade against.
    qe_case("total_energy--semiconductor--silicon", "us_hubbard", hubbard_energy=0.19679615),
    # PAW from a UPF dataset. The input states diagonalization='cg' because
    # QE's own Davidson aborts in cegterg on PAW datasets; Q3 maps it to
    # Davidson with a logged notice, which is a different path to the same
    # fixed point and is exactly the kind of substitution worth having a
    # reference case for.
    qe_case("total_energy--semiconductor--silicon", "paw"),
    qe_case("total_energy--insulator--diamond", "paw"),
    # Metal: smearing, partial occupations, a d shell.
    qe_case("total_energy--metal--copper", "paw"),
    # Collinear magnetism, ferromagnetic. The moment is checked as well as the
    # energy -- an unpolarized run reproduces neither, but a run whose spin
    # splitting is subtly wrong can still land near the right energy.
    qe_case("total_energy--metal--magnetic--iron", "nc", moment=2.28),
    qe_case("total_energy--metal--magnetic--nickel", "nc", moment=0.62),
    # Antiferromagnetic: two sublattices with opposite starting_magnetization,
    # which exercises the per-species indexed form of that key.
    #
    # The looser energy tolerance is measured, not chosen: Q3 lands 2.5e-4 Ry
    # (1.7 meV/atom) above QE with an absolute magnetization of 2.5929 against
    # QE's 2.60, so the two codes agree on the magnetic state and differ
    # slightly on how much moment survives self-consistency. Both runs are
    # converged well past their own conv_thr, so this is not an unfinished SCF.
    qe_case(
        "total_energy--metal--magnetic--chromium",
        "nc",
        moment=-0.00,
        absolute_moment=2.60,
        tolerance=5e-4,
    ),
]

# ── Reference lookup ──────────────────────────────────────────────────


def qe_reference(input_path):
    """
    What QE printed for this input, from the committed `reference.json`.

    Cross-checked against the `.out` beside it when that file is present: the
    JSON is the committed record, the `.out` is the artifact it was taken from,
    and letting them drift silently would defeat the point of committing either.
    """
    reference_path = input_path.parent / "reference.json"
    assert reference_path.is_file(), (
        f"no reference.json beside {input_path.name}; run tests/validation/refresh_qe_references.py"
    )
    inputs = json.loads(reference_path.read_text())["inputs"]
    assert input_path.name in inputs, (
        f"{reference_path} has no entry for {input_path.name}; "
        "run tests/validation/refresh_qe_references.py"
    )
    record = inputs[input_path.name]

    output_path = input_path.with_suffix(".out")
    if output_path.is_file():
        text = output_path.read_text()
        for name, pattern in (
            ("total_energy_rydberg", r"^!\s+total energy\s+=\s+(-?\d+\.\d+)"),
            (
                "total_magnetization_bohr_magneton",
                r"^\s+total magnetization\s+=\s+(-?\d+\.\d+)",
            ),
            (
                "absolute_magnetization_bohr_magneton",
                r"^\s+absolute magnetization\s+=\s+(-?\d+\.\d+)",
            ),
            ("hubbard_energy_rydberg", r"^\s+Hubbard energy\s+=\s+(-?\d+\.\d+)"),
        ):
            matches = re.findall(pattern, text, re.M)
            if not matches:
                continue
            assert record.get(name) == pytest.approx(float(matches[-1]), abs=1e-8), (
                f"{reference_path} is stale for {input_path.name}: {name} records "
                f"{record.get(name)} but {output_path.name} says {matches[-1]} -- "
                "run tests/validation/refresh_qe_references.py"
            )
    return record


def q3_total_magnetization(standard_output):
    matches = re.findall(r"[Tt]otal magnetization\s*=\s*(-?\d+\.\d+)", standard_output)
    assert matches, "no total magnetization in Q3 output"
    return float(matches[-1])


def q3_absolute_magnetization(standard_output):
    matches = re.findall(r"absolute magnetization\s*=\s*(-?\d+\.\d+)", standard_output)
    assert matches, "no absolute magnetization in Q3 output"
    return float(matches[-1])


def q3_hubbard_energy(standard_output):
    matches = re.findall(r"DFT\+U \(Hubbard\)\s*=\s*(-?\d+\.\d+)", standard_output)
    assert matches, "no DFT+U energy in Q3 output"
    return float(matches[-1])


# ── Staging ───────────────────────────────────────────────────────────


def stage_qe_input(input_path, tmp_path):
    """
    Copy a committed QE input into a temporary tree that preserves its relative
    layout, and return the directory to run in.

    The inputs reference `../../../data/pseudopotentials/<family>/` and write to
    `./tmp`, both relative to the directory they are run from. Reproducing that
    shape -- rather than rewriting the paths -- keeps the file byte-identical
    and keeps Q3's output out of the committed example tree, which is shared
    with other tests in this suite.
    """
    relative = input_path.relative_to(EXAMPLES_DIRECTORY)
    staged_input = tmp_path / relative
    staged_input.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input_path, staged_input)

    staged_data = tmp_path / "data"
    if not staged_data.exists():
        staged_data.symlink_to(EXAMPLES_DIRECTORY / "data", target_is_directory=True)

    assert staged_input.read_bytes() == input_path.read_bytes()
    return staged_input


def run_q3_on(q3_bin, input_path, timeout=1800):
    completed = subprocess.run(
        [q3_bin, input_path.name],
        cwd=str(input_path.parent),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        f"q3 exited {completed.returncode} on {input_path.name}\n"
        f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-2000:]}"
    )
    return completed.stdout


# ── QE cases ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("input_path,expectations", QE_CASES)
def test_q3_reproduces_quantum_espresso_from_its_own_input(
    q3_bin, tmp_path, input_path, expectations
):
    """Q3 reads QE's input file unchanged and lands on QE's numbers."""
    reference = qe_reference(input_path)
    staged = stage_qe_input(input_path, tmp_path)

    standard_output = run_q3_on(q3_bin, staged)
    energy = parse_q3_total_energy(standard_output)

    reference_energy = reference["total_energy_rydberg"]
    default_tolerance = (
        TOLERANCE_PAW_RYDBERG if input_path.parent.name == "paw" else TOLERANCE_SAME_CODE_RYDBERG
    )
    tolerance = expectations.get("tolerance", default_tolerance)
    difference = abs(energy - reference_energy)
    assert difference < tolerance, (
        f"{input_path.parent.name}: Q3 {energy:.8f} Ry vs QE {reference_energy:.8f} Ry, "
        f"difference {difference:.2e} Ry "
        f"({difference * RYDBERG_TO_ELECTRONVOLT * 1000:.2f} meV)"
    )

    if "moment" in expectations:
        reference_moment = reference["total_magnetization_bohr_magneton"]
        assert reference_moment == pytest.approx(expectations["moment"], abs=1e-8), (
            "the case table's moment no longer matches the committed reference"
        )
        moment = q3_total_magnetization(standard_output)
        assert abs(moment - reference_moment) < TOLERANCE_MOMENT_BOHR_MAGNETON, (
            f"{input_path.parent.name}: Q3 moment {moment:.4f} vs QE {reference_moment:.4f} muB"
        )

    if "absolute_moment" in expectations:
        reference_absolute = reference["absolute_magnetization_bohr_magneton"]
        assert reference_absolute == pytest.approx(expectations["absolute_moment"], abs=1e-8), (
            "the case table's absolute moment no longer matches the committed reference"
        )
        absolute = q3_absolute_magnetization(standard_output)
        assert abs(absolute - reference_absolute) < TOLERANCE_MOMENT_BOHR_MAGNETON, (
            f"{input_path.parent.name}: Q3 |m| {absolute:.4f} vs QE {reference_absolute:.4f} muB "
            "-- the magnetic state differs, which a zero total magnetization would have hidden"
        )

    if "hubbard_energy" in expectations:
        reference_hubbard = reference["hubbard_energy_rydberg"]
        assert reference_hubbard == pytest.approx(expectations["hubbard_energy"], abs=1e-8), (
            "the case table's E_U no longer matches the committed reference"
        )
        hubbard = q3_hubbard_energy(standard_output)
        assert abs(hubbard - reference_hubbard) < TOLERANCE_SAME_CODE_RYDBERG, (
            f"{input_path.parent.name}: Q3 E_U {hubbard:.8f} vs QE {reference_hubbard:.8f} Ry"
        )


def test_every_committed_qe_input_is_readable(q3_bin, tmp_path):
    """
    Nothing in `examples/` may be a file Q3 refuses.

    Broader than the SCF cases above -- it covers the supercell, dispersion and
    relaxation inputs those skip for runtime -- and far cheaper, because
    ``--check`` stops after the parse. This is the shape of test that catches a
    strictness rule refusing something it should not: when the namelist policy
    went in, it immediately rejected the three `scf_qe_d3.in` files over
    `dftd3_version`, a key Q3 had been silently assuming the value of.
    """
    inputs = sorted(EXAMPLES_DIRECTORY.glob("*/reference/*/*.in"))
    assert len(inputs) >= 12, f"expected the committed QE corpus, found {len(inputs)}"

    refused = []
    for path in inputs:
        staged = stage_qe_input(path, tmp_path)
        completed = subprocess.run(
            [q3_bin, staged.name, "--check"],
            cwd=str(staged.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            refused.append(f"{path.relative_to(EXAMPLES_DIRECTORY)}: {completed.stdout.strip()}")
    assert not refused, "Q3 refused committed QE inputs:\n" + "\n".join(refused)
