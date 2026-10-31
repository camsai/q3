"""
The DFT+U stress tensor, against Quantum ESPRESSO's own Hubbard stress term.

Until 2026-08-10 no DFT+U stress existed in Q3 for any dataset family. The
reported stress simply omitted it, and because that omission was announced only
as a log warning and an incomplete-stress flag, the calculation it would
have ruined -- a variable-cell relaxation on a Hubbard system -- was refused
rather than allowed to converge somewhere wrong.

**Why this compares against ESPRESSO rather than a finite difference.** Forces
carry no Pulay term, so the DFT+U force could be gated on a central difference
of Q3's own energy (``test_paw_forces_and_stress.py``). Stress cannot: at the
cutoffs these decks use, the analytic pressure and -dE/dV differ by hundreds of
kbar from basis-set incompleteness alone, which would swamp a Hubbard term worth
tens. Comparing analytic against analytic at the same cutoff sidesteps Pulay
entirely -- both codes are differentiating the same incomplete basis, and the
term is isolated because ESPRESSO reports its stress decomposition per
contribution.

**The reference.** ``reference/*/scf_qe_stress.in`` beside each case, whose
numbers are recorded in the ``reference.json`` next to it (QE's ``.out`` is
gitignored). Those decks carry ``tstress = .true.`` *and* ``verbosity = 'high'``
-- ESPRESSO prints the per-term decomposition only when ``iverbosity > 0``,
so without the second key there is a total and no ``hubbard stress`` line at
all. ESPRESSO documents that term as
``-(1/omega) dE_h/depsilon``, which is Q3's own stress convention, and
Quantum ESPRESSO adds it to the total unmodified, so the printed number is
directly comparable with no sign or factor in between.

**The three cases, and what each one alone would miss.**

===============================  =========  =========  ====================================
case                             dataset    manifold   what it exercises
===============================  =========  =========  ====================================
silicon diamond, U = 3 eV on 3p  ultrasoft  l = 1      dphi/deps *and* dS/deps together
silicon diamond, U = 3 eV on 3p  NC         l = 1      dphi/deps alone; dS/deps is zero
copper fcc, U = 4 eV on 3d       ultrasoft  l = 2      d = 2 spherical-harmonic derivatives
===============================  =========  =========  ====================================

The norm-conserving row is the reduction the sign was pinned by: with q_ij = 0
the overlap operator's strain derivative vanishes identically and the expression
collapses to ``<dphi/deps|psi>``. Copper is the sharpest of the three by a wide
margin -- its Hubbard stress is -4353 kbar against a *total* pressure of 19,
so an error of one part in a thousand in the term moves the total by 23%.

Measured, and what the term is worth:

=========================  ==============  ==============  ================
case                       Q3 (kbar)       ESPRESSO (kbar)  share of total P
=========================  ==============  ==============  ================
silicon ultrasoft          -34.89          -34.89           29%
silicon norm-conserving    -55.96          -55.96           58%
copper ultrasoft           -4353.04        -4353.07         22600%
=========================  ==============  ==============  ================

Run:
    .venv/bin/pytest tests/validation/test_qe_hubbard_stress.py -v --q3-bin ./build/q3
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]

CASES = {
    "silicon_ultrasoft": (
        REPOSITORY
        / "examples"
        / "total_energy--semiconductor--silicon"
        / "reference"
        / "us_hubbard",
        "us",
    ),
    "silicon_norm_conserving": (
        REPOSITORY
        / "examples"
        / "total_energy--semiconductor--silicon"
        / "reference"
        / "nc_hubbard",
        "nc",
    ),
    "copper_ultrasoft": (
        REPOSITORY / "examples" / "total_energy--metal--copper" / "reference" / "us_hubbard",
        "us",
    ),
}

STRESS_TOLERANCE_KBAR = 0.02
"""
Absolute agreement demanded of the small cases.

ESPRESSO prints the decomposition to two decimals, so 0.005 kbar of the budget
is its own rounding on each side. What is left is far below the size of every
defect this gate exists to catch: the term was zero before this branch, and the
first wrong version of it -- the beta projectors' ``(-i)^l`` phase used on the
atomic orbitals, which carry ``i^l`` -- read -571 kbar against -34.89.
"""

RELATIVE_STRESS_TOLERANCE = 5.0e-5
"""
What the copper case is held to instead, its term being two orders larger.

5e-5 of -4353 kbar is 0.22 kbar, still 20000x below the whole term and 90x
below the 19.38 kbar total pressure it has to land inside. The measured gap is
0.03 kbar, seven times inside it.
"""

CONTRIBUTION_RELATIVE_TOLERANCE = 6.0e-6
"""
How closely the two codes agree on a single stress contribution.

Not a property of the DFT+U term -- it is what kinetic, local and nonlocal
already do on these decks, and it is what limits the *total*, which is a
cancellation between contributions much larger than itself.

It cannot be tighter than the reference itself is reproducible. On the copper
deck ESPRESSO 7.6 prints the same total energy as the 7.5 reference stored here
but a total pressure of 20.14 kbar, 0.76 kbar above it -- 5e-6 of the largest
contribution. The budget used to be 2e-6, which only held on copper because Q3's
local stress was 0.64 kbar low and cancelled its nonlocal term's +0.50; the local
term was off because V_loc's Coulomb tail was interpolated, and with it added
analytically it agrees with ESPRESSO 7.6 to 0.12 kbar.
"""


def _q3_input(directory, case_directory, pseudopotential_family):
    """The committed QE deck with `pseudo_dir` made absolute.

    Rewritten rather than run in place because the deck resolves its datasets
    relative to itself and the run has to happen in a scratch directory; the
    drop-in tests, whose subject *is* the parsing, keep their copies
    byte-identical instead.
    """
    text = (case_directory / "scf_qe_stress.in").read_text()
    pseudopotentials = REPOSITORY / "examples" / "data" / "pseudopotentials"
    text = text.replace(
        f"pseudo_dir = '../../../data/pseudopotentials/{pseudopotential_family}/'",
        f"pseudo_dir = '{pseudopotentials / pseudopotential_family}/'",
    )
    path = directory / "scf.in"
    path.write_text(text)
    return path


def _run(q3_bin, directory, input_path):
    completed = subprocess.run(
        [q3_bin, input_path.name],
        cwd=str(directory),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert completed.returncode == 0, (
        f"q3 exited {completed.returncode}\n{completed.stdout[-3000:]}"
    )
    assert (
        "SCF converged" in completed.stdout or "convergence achieved" in completed.stdout.lower()
    ), (
        "the run did not converge, so it reports no stress at all and any comparison "
        f"below would be against a default-initialized tensor\n{completed.stdout[-2000:]}"
    )
    return completed.stdout


def hubbard_stress_diagonal(output):
    """The three diagonal components of the reported DFT+U stress, in kbar.

    Anchored on the contributions block's own row rather than on the string
    "DFT+U", which also names the *energy* line a few hundred lines earlier.
    """
    match = re.search(
        r"^\s+DFT\+U\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\(P=",
        output,
        re.M,
    )
    assert match, "no DFT+U row in the stress contributions block"
    return [float(value) for value in match.groups()]


def total_pressure(output):
    match = re.search(r"^\s+Pressure = (-?\d+\.\d+) kbar", output, re.M)
    assert match, "no total pressure reported"
    return float(match.group(1))


def largest_contribution(output):
    """The biggest single stress contribution, in kbar.

    The total is a cancellation between terms far larger than itself -- on
    copper, 148000 kbar of Ewald against 90000 of Hartree for a total of 19 --
    so what the two codes can agree on absolutely is set by that scale, not by
    the total. Silicon's largest term is 3000 kbar and copper's is 148000, a
    factor of 50, which is why the total's tolerance is derived rather than
    fixed.
    """
    values = re.findall(
        r"^\s+\S[^:]*:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\(P=", output, re.M
    )
    assert values, "no stress contributions block"
    return max(abs(float(component)) for row in values for component in row)


@pytest.fixture(scope="module", params=sorted(CASES))
def hubbard_stress_case(request, q3_bin, tmp_path_factory):
    case_directory, pseudopotential_family = CASES[request.param]
    directory = tmp_path_factory.mktemp(f"hubbard_stress_{request.param}")
    input_path = _q3_input(directory, case_directory, pseudopotential_family)
    reference = json.loads((case_directory / "reference.json").read_text())
    return {
        "name": request.param,
        "output": _run(q3_bin, directory, input_path),
        "reference": reference["inputs"]["scf_qe_stress.in"],
    }


def test_hubbard_stress_matches_quantum_espresso(hubbard_stress_case):
    """The whole point: Q3's Hubbard stress against ESPRESSO's own."""
    expected = hubbard_stress_case["reference"]["hubbard_stress_kbar"]
    tolerance = max(STRESS_TOLERANCE_KBAR, RELATIVE_STRESS_TOLERANCE * abs(expected))
    for component, value in enumerate(hubbard_stress_diagonal(hubbard_stress_case["output"])):
        assert abs(value - expected) < tolerance, (
            f"{hubbard_stress_case['name']}: Q3 Hubbard stress component {component} is "
            f"{value:.2f} kbar against ESPRESSO's {expected:.2f} kbar"
        )


def test_the_hubbard_stress_is_not_negligible(hubbard_stress_case):
    """
    The premise of the comparison above.

    Every deck here is cubic, and a symmetric site is exactly where a stress
    term can vanish identically and make an agreement prove nothing. It does not
    vanish -- the smallest of the three is 29% of its total pressure -- and
    asserting so is what stops the pair from passing on a regression to zero,
    which is precisely the state Q3 was in before this term existed.
    """
    expected = hubbard_stress_case["reference"]["hubbard_stress_kbar"]
    assert abs(expected) > 10.0, "the reference term is too small for the comparison to bite"
    components = hubbard_stress_diagonal(hubbard_stress_case["output"])
    assert min(abs(value) for value in components) > 10.0, (
        f"{hubbard_stress_case['name']}: the reported Hubbard stress is effectively zero"
    )


def test_total_pressure_matches_quantum_espresso(hubbard_stress_case):
    """
    The number a variable-cell relaxation actually steers on.

    Budgeted against the largest single contribution rather than against the
    Hubbard term, because the residual is not the Hubbard term's: on copper Q3
    and the stored ESPRESSO 7.5 reference differ by 0.58 kbar in the total, of
    which the DFT+U term supplies 0.04 and the kinetic, local and nonlocal terms
    supply -0.10, +0.12 and +0.50 -- none of them the Hubbard term's.
    Tying the total to a Hubbard-sized budget would make this test fail for
    reasons it does not measure.

    It still bites hard on what it does measure. Dropping the term moves the
    total by 34.89 kbar on silicon and 4353 on copper, against tolerances of
    0.02 and 0.89.
    """
    expected = hubbard_stress_case["reference"]["total_pressure_kbar"]
    tolerance = max(
        STRESS_TOLERANCE_KBAR,
        CONTRIBUTION_RELATIVE_TOLERANCE * largest_contribution(hubbard_stress_case["output"]),
    )
    measured = total_pressure(hubbard_stress_case["output"])
    assert abs(measured - expected) < tolerance, (
        f"{hubbard_stress_case['name']}: Q3 total pressure {measured:.2f} kbar against "
        f"ESPRESSO's {expected:.2f} kbar (tolerance {tolerance:.2f})"
    )


def test_the_hubbard_stress_does_not_depend_on_symmetry(q3_bin, tmp_path_factory):
    """
    The symmetry-reduced wedge and the full zone must give the same term.

    The irreducible wedge folds by the spatial group **and** time reversal, and
    the occupation matrix it is built from is Theta-averaged for that reason
    (n(-k) = n(k)^T, and no real D(R) expresses an antiunitary operation). A
    dn/d(strain) inconsistent with how n is actually built reproduces exactly the
    defect that put the DFT+U *force* 0.3% off its own energy gradient on the
    wedge while the full zone was right (KNOWN-ISSUES.md, 2026-08-09) -- an error
    invisible to any test that samples one of the two.

    Both runs are the same deck; `nosym` is the only difference, taking silicon
    from 8 k-points in the wedge to all 64.
    """
    case_directory, family = CASES["silicon_ultrasoft"]
    outputs = {}
    for label in ("symmetry", "no_symmetry"):
        directory = tmp_path_factory.mktemp(f"hubbard_stress_{label}")
        input_path = _q3_input(directory, case_directory, family)
        if label == "no_symmetry":
            input_path.write_text(
                input_path.read_text().replace("  ibrav = 2", "  ibrav = 2\n  nosym = .true.")
            )
        outputs[label] = _run(q3_bin, directory, input_path)

    assert "64 -> 8 k-points" in outputs["symmetry"], (
        "the symmetric run is not actually using a reduced wedge, so this proves nothing"
    )
    wedge = hubbard_stress_diagonal(outputs["symmetry"])
    full_zone = hubbard_stress_diagonal(outputs["no_symmetry"])
    assert min(abs(value) for value in wedge + full_zone) > 10.0, (
        "both runs report an effectively zero Hubbard stress, which would agree with "
        "itself while measuring nothing"
    )
    for component, (reduced, complete) in enumerate(zip(wedge, full_zone)):
        assert abs(reduced - complete) < STRESS_TOLERANCE_KBAR, (
            f"Hubbard stress component {component} is {reduced:.2f} kbar on the wedge "
            f"against {complete:.2f} kbar over the full zone"
        )
