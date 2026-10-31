"""
DFT+U forces on ultrasoft and PAW datasets, against Q3's own total energy.

Each gate compares an analytic force with central finite differences of the
energy the same code reports. A force that disagrees with the numerical
derivative of Q3's *own* energy can only be the force term, which is what
makes such a gate able to localize a defect rather than merely detect one.

The cases: the PAW nickel Hubbard occupations and derivatives; DFT+U forces
through the overlap operator on an ultrasoft dataset; DFT+U forces on PAW,
with the dS/dR term reaching a non-Hubbard atom, over the full zone and the
irreducible wedge; and ortho-atomic projectors, whose Löwdin transform carries
a force of its own.

Run:
    .venv/bin/pytest tests/validation/test_paw_forces_and_stress.py -v --q3-bin ./build/q3
"""

import json
import math
import re
import subprocess
import tarfile
from pathlib import Path

import pytest
from force_helpers import BOHR_TO_ANGSTROM, Q3Run, force_contribution_on_atom, force_on_atom
from q3.io.results.classes import Incomplete, Results

from helpers import stress_contributions

# ─── DFT+U with ultrasoft and PAW ───────────────────────────────────────────

NICKEL_MATERIAL = {
    "name": "Ni",
    "lattice": {
        "type": "FCC",
        "a": 2.4890158697766473,
        "b": 2.4890158697766473,
        "c": 2.4890158697766473,
        "alpha": 60,
        "beta": 60,
        "gamma": 60,
        "units": {"length": "angstrom", "angle": "degree"},
    },
    "basis": {
        "units": "crystal",
        "elements": [{"id": 0, "value": "Ni"}],
        "coordinates": [{"id": 0, "value": [0, 0, 0]}],
    },
    "isNonPeriodic": False,
}

PAW_UPF_ARCHIVE = (
    Path(__file__).resolve().parents[2].parent
    / "reference"
    / "mat3ra"
    / "pseudos"
    / "data"
    / "paw-sr-11_pbe_stringent_upf.tgz"
)

NICKEL_D_ELECTRON_COUNT = 8.83
"""
Tr[n] for the nickel 3d manifold with the overlap operator applied.

Nickel is 3d(8)4s(2) atomically and its d occupancy rises toward 9 in the solid,
so this is the physically right neighbourhood. Without the overlap operator the
same run gives 2.71 -- under a third of the shell -- because the pseudo orbital
alone misses the charge the augmentation carries. That is the failure this gates:
the SCF converges either way and only Tr[n] says which one is real.
"""


@pytest.fixture(scope="module")
def nickel_paw_dataset(tmp_path_factory):
    if not PAW_UPF_ARCHIVE.is_file():
        pytest.skip(f"no PAW UPF archive at {PAW_UPF_ARCHIVE}")
    destination = tmp_path_factory.mktemp("paw_upf") / "Ni.upf"
    with tarfile.open(PAW_UPF_ARCHIVE) as archive:
        member = next(
            (m for m in archive.getmembers() if Path(m.name).name in ("Ni.upf", "Ni.UPF")),
            None,
        )
        if member is None:
            pytest.skip("no Ni PAW UPF in the archive")
        destination.write_bytes(archive.extractfile(member).read())
    return destination


def write_hubbard_input(directory, dataset, hubbard):
    material_path = directory / "nickel.json"
    material_path.write_text(json.dumps(NICKEL_MATERIAL))
    hubbard_block = (
        "    hubbard:\n      - {species: Ni, orbital_label: d, u_ev: 4.0}\n" if hubbard else ""
    )
    input_path = directory / "nickel.yml"
    input_path.write_text(
        f"""material: {material_path}
model: {{type: dft, subtype: gga, functional: pbe}}
method:
  type: pseudopotential
  subtype: paw
  cutoffs: {{wavefunction: 40.0, density: 320.0}}
  pseudopotentials: {{Ni: {dataset}}}
  scf:
    convergence_threshold: 1.0e-06
    max_iterations: 60
    mixing: {{mode: broyden, beta: 0.1}}
    smearing: {{type: gaussian, degauss: 0.02}}
    kgrid: {{dimensions: [4, 4, 4], shifts: [1, 1, 1]}}
    number_of_bands: 20
{hubbard_block}    number_of_spin_components: 2
    starting_magnetization: {{Ni: 0.6}}
control: {{calculation: scf, outdir: {directory / "output"}, prefix: ni}}
"""
    )
    return input_path


def test_paw_hubbard_occupations_include_the_augmentation_charge(
    q3_bin, tmp_path, nickel_paw_dataset
):
    """
    DFT+U on a PAW dataset must project with the all-electron inner product.

    Ultrasoft and PAW were refused outright before 2026-08-08. They now work by
    caching S|phi> as the Hubbard projector, which makes the occupation matrix,
    the Hamiltonian term and the projector overlaps all correct at once.

    Gated on Tr[n] rather than on a total energy because Tr[n] is the quantity
    that says whether the projection found the shell, and because no reference
    is available for this dataset: its PP_CHI carry no labels, so Quantum
    ESPRESSO's own Hubbard-shell detection refuses it too ("Ni pseudopotential
    contains the orbitals:" -- blank). Q3 reaches it by selecting on angular
    momentum and occupancy instead.
    """
    input_path = write_hubbard_input(tmp_path, nickel_paw_dataset, hubbard=True)
    completed = subprocess.run(
        [str(q3_bin), str(input_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert completed.returncode == 0, completed.stdout[-3000:]

    results = Results.read(tmp_path / "output" / "ni_results.json")
    occupations = results.properties.hubbard_occupations
    assert occupations, "no Tr[n] reported for any Hubbard manifold"
    traces = occupations[-1].occupation_trace
    total_trace = sum(traces)
    print(f"\n  Tr[n] = {' + '.join(f'{trace:.4f}' for trace in traces)} = {total_trace:.4f}")
    assert abs(total_trace - NICKEL_D_ELECTRON_COUNT) < 0.5, (
        f"the nickel 3d manifold holds {total_trace:.3f} electrons; without the overlap "
        "operator this comes out near 2.7, which is the signature of projecting with the "
        "bare pseudo orbital"
    )


def test_paw_hubbard_derivatives_are_both_computed(q3_bin, tmp_path, nickel_paw_dataset):
    """
    A PAW DFT+U run computes both of its Hubbard derivatives.

    This test has now been inverted twice, and the history is the point. It
    first asserted that the force was reported as **zero** with a warning; that
    became wrong when the dS/dR terms landed on 2026-08-09. It then asserted
    that the force was present but the **stress** gap was still announced; that
    became wrong when the dS/d(strain) terms landed on 2026-08-10. What it
    guards now is that neither warning comes back.

    PAW is not gated against ESPRESSO the way ultrasoft and norm-conserving are
    in `test_qe_hubbard_stress.py`: every PAW dataset in this repository leaves
    its PP_CHI shells unlabelled, and ESPRESSO's own Hubbard-shell detection
    refuses `U Ni-3d` on such a file rather than guessing. So the magnitude is
    not checked here -- PAW and ultrasoft share one code path
    (`has_ultrasoft()`), and that path is validated to 7e-6 on copper -- only
    that a PAW run reaches both terms and that neither is silently dropped.
    """
    input_path = write_hubbard_input(tmp_path, nickel_paw_dataset, hubbard=True)
    completed = subprocess.run(
        [str(q3_bin), str(input_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert completed.returncode == 0, completed.stdout[-2000:]

    # The incompleteness flags are the machine-readable form of the two warnings
    # this used to grep for: a run that leaves a derivative term out says so in
    # its results, which is where the relaxation workflows look, and a flag
    # cannot start passing merely because the wording of a warning drifted.
    results = Results.read(tmp_path / "output" / "ni_results.json")
    incomplete = results.runtime_log.incomplete or Incomplete()
    assert incomplete.forces is None, (
        f"the PAW DFT+U force is implemented, but the run reports it incomplete: "
        f"{incomplete.forces}"
    )
    assert incomplete.stress is None, (
        f"the PAW DFT+U stress is implemented, but the run reports it incomplete: "
        f"{incomplete.stress}"
    )
    hubbard_stress = stress_contributions(results)["hubbard"]
    diagonal = [hubbard_stress[index][index] for index in range(3)]
    assert max(abs(value) for value in diagonal) > 1.0, (
        f"the PAW Hubbard stress diagonal is {diagonal} kbar, effectively zero -- which "
        "is what the term returned before it existed"
    )
    # The run still produces an energy: nothing about the derivatives changed it.
    assert math.isfinite(results.properties.total_energy.value)


# ── DFT+U forces through the overlap operator ────────────────────────────────

"""
For ultrasoft and PAW the Hubbard projector is S|phi>, not |phi>, so the DFT+U
force carries dS/dR terms as well as dphi/dR. Q3 returned zero for the whole
Hubbard contribution until 2026-08-09 and warned that total forces were
incomplete.

The check is the one that found every other force defect here: the analytic
force against a central difference of Q3's *own* total energy. It localizes
cleanly, because the energy is correct either way -- only the force assembly
changed.

Measured on this case: with the Hubbard term the ratio is 0.9996; with it
omitted, as it was before, 0.941. The missing piece was 5.9% of the total force
on the displaced atom, far too large to relax with and far too small to look
obviously wrong.
"""

HUBBARD_ULTRASOFT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "total_energy--semiconductor--silicon"
    / "reference"
    / "us_hubbard"
    / "scf_qe.in"
)

HUBBARD_BASE_DISPLACEMENT = 0.010
"""
Fractional displacement of the second atom in *every* run of the triple.

The diamond site is symmetric, so a difference centred on the ideal position
would vanish identically and the test would measure nothing -- the same trap
the magnetic augmentation gate hit.
"""

HUBBARD_FRACTIONAL_STEP = 0.002
"""The finite-difference step taken around that base displacement."""


def write_displaced_hubbard_input(directory, fractional_displacement):
    """The committed DFT+U deck with atom 2 moved along the first cell vector.

    `pseudo_dir` is rewritten to an absolute path, unlike the drop-in tests
    which deliberately keep the file byte-identical: here the point is the
    derivative, not the parsing.
    """
    text = HUBBARD_ULTRASOFT_INPUT.read_text()
    pseudopotential_directory = HUBBARD_ULTRASOFT_INPUT.parents[3] / "data" / "pseudopotentials"
    text = text.replace(
        "pseudo_dir = '../../../data/pseudopotentials/us/'",
        f"pseudo_dir = '{pseudopotential_directory / 'us'}/'",
    )
    text = text.replace(
        "  Si 0.25 0.25 0.25",
        f"  Si {0.25 + fractional_displacement:.6f} 0.25 0.25",
    )
    path = directory / "scf.in"
    path.write_text(text)
    return path


@pytest.fixture(scope="module")
def hubbard_ultrasoft_runs(q3_bin, tmp_path_factory):
    outputs = {}
    for label, offset in (
        ("baseline", 0.0),
        ("plus", HUBBARD_FRACTIONAL_STEP),
        ("minus", -HUBBARD_FRACTIONAL_STEP),
    ):
        directory = tmp_path_factory.mktemp(f"hubbard_us_{label}")
        input_path = write_displaced_hubbard_input(directory, HUBBARD_BASE_DISPLACEMENT + offset)
        completed = subprocess.run(
            [q3_bin, input_path.name],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert completed.returncode == 0, (
            f"q3 exited {completed.returncode} on the {label} run\n{completed.stdout[-3000:]}"
        )
        results = Results.read(directory / "tmp" / "qe_us_hubbard_results.json")
        outputs[label] = Q3Run(results, completed.stdout)
    return outputs


def _first_cell_vector_bohr():
    """a1 of the committed deck, in bohr.

    Read from the deck rather than assumed: it states `ibrav = 2`, whose a1 is
    a/2(-1,0,1) -- projecting the force onto the wrong axis makes a correct
    force look 1.9x wrong, which cost a full diagnostic cycle to notice.
    """
    text = HUBBARD_ULTRASOFT_INPUT.read_text()
    celldm = float(re.search(r"celldm\(1\)\s*=\s*([\d.]+)", text).group(1))
    return (-0.5 * celldm, 0.0, 0.5 * celldm)


def test_hubbard_force_is_the_gradient_of_q3s_own_energy_for_ultrasoft(hubbard_ultrasoft_runs):
    """The DFT+U force with its dS/dR terms, against Q3's own energy."""
    axis = _first_cell_vector_bohr()
    length = math.sqrt(sum(component * component for component in axis))
    unit_axis = [component / length for component in axis]
    step_bohr = HUBBARD_FRACTIONAL_STEP * length

    force = force_on_atom(hubbard_ultrasoft_runs["baseline"], 1)
    analytic = sum(force[i] * unit_axis[i] for i in range(3))
    numerical = -(
        hubbard_ultrasoft_runs["plus"].results.properties.total_energy.value
        - hubbard_ultrasoft_runs["minus"].results.properties.total_energy.value
    ) / (2.0 * step_bohr)

    assert abs(numerical) > 1e-3, "the displacement is not producing a resolvable force"
    ratio = analytic / numerical
    assert abs(ratio - 1.0) < 0.01, (
        f"DFT+U ultrasoft force {analytic:.8f} Ry/bohr vs finite difference "
        f"{numerical:.8f} Ry/bohr (ratio {ratio:.6f})"
    )


def test_the_hubbard_contribution_is_not_zero(hubbard_ultrasoft_runs):
    """
    The premise of the test above.

    A zero Hubbard force -- what Q3 reported before this term existed -- would
    still leave the total within a few percent of its own gradient. Asserting
    the contribution is actually present is what stops this pair from passing
    on a regression back to zero.
    """
    components = force_contribution_on_atom(hubbard_ultrasoft_runs["baseline"], "hubbard", 1)
    magnitude = math.sqrt(sum(component * component for component in components))
    assert magnitude > 1e-4, f"the Hubbard force is {magnitude:.2e} Ry/bohr, effectively zero"


# ── DFT+U forces on PAW, and the dS/dR term in isolation ─────────────────────

"""
SiC in zincblende with U on silicon only.

This case exists to separate two things the silicon-ultrasoft gate above can
only test together. The DFT+U force has a term from the manifold's own orbital
moving and a term from the beta functions inside S moving; with U on both atoms
of a two-atom cell, a wrong sign or scale in one could hide inside the other.

**Carbon carries no Hubbard manifold**, so its own-orbital term is identically
zero and every bit of its Hubbard force comes from dS/dR. Disabling that term
drops the Hubbard force on carbon to exactly 0.00000000 -- verified by building
it that way -- which makes `test_the_dsdr_term_reaches_a_non_hubbard_atom` the
sharpest discriminator available for it. It is also the only PAW case that is
finite-differenced; the silicon gate is ultrasoft.

Symmetry is disabled here so that the derivative is tested over an unreduced
k-sum, with nothing between the force and the energy. The irreducible wedge is
exercised separately by `sic_hubbard_symmetry_runs` below, which is the gate on
the occupation matrix's own symmetrization.
"""

SIC_MATERIAL = (
    Path(__file__).resolve().parents[2] / "examples" / "data" / "materials" / "silicon_carbide.json"
)
SIC_CONVENTIONAL_ALAT_ANGSTROM = 4.3596

SIC_BASE_DISPLACEMENT = 0.020
"""
Carbon's offset from its lattice site in every run of the triple.

A zincblende site has full tetrahedral symmetry, so a difference centred on the
ideal position vanishes identically.
"""

SIC_FRACTIONAL_STEP = 0.001

SIC_CONVERGENCE_THRESHOLD = "1.0e-12"
"""
Much tighter than a normal SCF needs, and the reason is worth stating.

`convergence_threshold` bounds the *density* residual, not the energy. At 1e-8
the converged energies still scatter by ~5e-6 Ry between neighbouring
geometries, which is larger than the energy difference across a small
displacement: the finite difference then wanders with step size instead of
converging (measured ratios 1.044, 0.971, 1.007, 1.002 at successively larger
steps -- the *small* steps being the bad ones, which is the opposite of
truncation error and the tell that it is noise). At 1e-12 the same three steps
give 1.0035, 1.0030, 1.0029: smooth, and step-size independent to 0.05%.
"""


def write_sic_hubbard_input(directory, fractional_displacement, disable_symmetry=True):
    """Zincblende SiC, PAW, U on silicon only, carbon displaced along a1."""
    material = json.loads(SIC_MATERIAL.read_text())
    material["basis"]["coordinates"][1]["value"] = [0.25 + fractional_displacement, 0.25, 0.25]
    material_path = directory / "sic.json"
    material_path.write_text(json.dumps(material))

    pseudopotentials = SIC_MATERIAL.parents[1] / "pseudopotentials" / "paw"
    symmetry = "disable_symmetry: true, " if disable_symmetry else ""
    input_path = directory / "sic.yml"
    input_path.write_text(
        f"""material: {material_path}
model: {{type: dft, subtype: gga, functional: pbe}}
method:
  type: pseudopotential
  subtype: paw
  cutoffs: {{wavefunction: 45.0, density: 360.0}}
  pseudopotentials:
    Si: {pseudopotentials / "Si.upf"}
    C: {pseudopotentials / "C.upf"}
  scf:
    convergence_threshold: {SIC_CONVERGENCE_THRESHOLD}
    max_iterations: 300
    mixing: {{mode: broyden, beta: 0.3}}
    kgrid: {{dimensions: [4, 4, 4], shifts: [0, 0, 0]}}
    number_of_bands: 12
    hubbard:
      - {{species: Si, orbital_label: p, u_ev: 3.0}}
control: {{{symmetry}calculation: scf, outdir: {directory / "out"}, prefix: sic}}
"""
    )
    return input_path


@pytest.fixture(scope="module")
def sic_hubbard_runs(q3_bin, tmp_path_factory):
    outputs = {}
    for label, offset in (
        ("baseline", 0.0),
        ("plus", SIC_FRACTIONAL_STEP),
        ("minus", -SIC_FRACTIONAL_STEP),
    ):
        directory = tmp_path_factory.mktemp(f"sic_u_{label}")
        input_path = write_sic_hubbard_input(directory, SIC_BASE_DISPLACEMENT + offset)
        completed = subprocess.run(
            [q3_bin, input_path.name],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert completed.returncode == 0, (
            f"q3 exited {completed.returncode} on the {label} run\n{completed.stdout[-3000:]}"
        )
        results = Results.read(directory / "out" / "sic_results.json")
        assert results.runtime_log.converged, f"the {label} run did not converge"
        outputs[label] = Q3Run(results, completed.stdout)
    return outputs


def _sic_displacement_axis():
    """The unit vector along a1 and the finite-difference step along it, in bohr.

    Read from what the deck defines rather than assumed: `ibrav = 2` puts a1 at
    alat/2(-1, 0, 1), and projecting onto the wrong axis makes a correct force
    look 1.9x wrong -- a mistake that has already cost one diagnostic cycle here.
    """
    alat_bohr = SIC_CONVENTIONAL_ALAT_ANGSTROM / BOHR_TO_ANGSTROM
    axis = (-0.5 * alat_bohr, 0.0, 0.5 * alat_bohr)
    length = math.sqrt(sum(component * component for component in axis))
    return [component / length for component in axis], SIC_FRACTIONAL_STEP * length


def _force_against_its_own_gradient(runs):
    """(analytic, numerical) force along a1 on the displaced carbon, in Ry/bohr."""
    unit_axis, step_bohr = _sic_displacement_axis()
    force = force_on_atom(runs["baseline"], 1)
    analytic = sum(force[i] * unit_axis[i] for i in range(3))
    numerical = -(
        runs["plus"].results.properties.total_energy.value
        - runs["minus"].results.properties.total_energy.value
    ) / (2.0 * step_bohr)
    return analytic, numerical


def _hubbard_force_on_carbon(run):
    """The Hubbard term's own contribution to the force on carbon, in Ry/bohr."""
    return force_contribution_on_atom(run, "hubbard", 1)


def _hubbard_force_along_a1(run):
    """That contribution projected onto the displacement axis."""
    unit_axis, _ = _sic_displacement_axis()
    components = _hubbard_force_on_carbon(run)
    return sum(components[i] * unit_axis[i] for i in range(3))


def test_paw_hubbard_force_is_the_gradient_of_q3s_own_energy(sic_hubbard_runs):
    """The DFT+U force on a PAW dataset, against Q3's own energy."""
    analytic, numerical = _force_against_its_own_gradient(sic_hubbard_runs)

    assert abs(numerical) > 1e-3, "the displacement is not producing a resolvable force"
    ratio = analytic / numerical
    assert abs(ratio - 1.0) < 0.005, (
        f"PAW DFT+U force {analytic:.8f} Ry/bohr vs finite difference "
        f"{numerical:.8f} Ry/bohr (ratio {ratio:.6f})"
    )


def test_the_console_report_still_carries_its_force_and_stress_blocks(sic_hubbard_runs):
    """
    The one check in this file that reads the console rather than results.json.

    Every other assertion here used to parse that console output; they now read
    the file the run leaves behind, which is written even when a run ends badly
    and cannot be broken by a change of layout. What that move loses is any
    notice if the human-facing report simply stops printing -- it is nobody's
    input any more, so nothing would fail. This is that notice, and it is
    deliberately the only one: it asserts the blocks exist, not what is in them,
    because what is in them is now checked from the JSON.

    It hangs off the SiC fixture rather than a licensed-dataset deck, because
    such a deck skips wherever its dataset is absent, which is
    everywhere this repository is built. A smoke test that never runs is not one.
    """
    standard_output = sic_hubbard_runs["baseline"].standard_output
    for block in (
        "Forces (Ry/bohr):",
        "Force contributions per atom (Ry/bohr):",
        "Force contributions (diagonal RMS, Ry/bohr):",
        "Stress tensor (kbar):",
        "Pressure =",
        "Final Energies (Ry):",
    ):
        assert block in standard_output, (
            f"the console report no longer prints {block!r}:\n" + standard_output[-3000:]
        )


def test_the_dsdr_term_reaches_a_non_hubbard_atom(sic_hubbard_runs):
    """
    Carbon carries no U, so its Hubbard force is entirely dS/dR.

    Removing that term makes this exactly zero -- confirmed by building Q3
    without it -- so this is the one assertion that cannot be satisfied by the
    own-orbital half of the derivative.
    """
    components = _hubbard_force_on_carbon(sic_hubbard_runs["baseline"])
    magnitude = math.sqrt(sum(component * component for component in components))
    assert magnitude > 1e-4, (
        f"the Hubbard force on the non-Hubbard species is {magnitude:.2e} Ry/bohr, "
        "effectively zero -- the dS/dR term is not reaching it"
    )


# ── The same case over the irreducible wedge ─────────────────────────────────

"""
The wedge run of the identical triple, which is a gate on the occupation matrix.

`reduce_kpoints_internal` folds -Rk into Rk as well as Rk into k, so the wedge
is the quotient by the spatial operations *and* time reversal. `n` is not
invariant under the latter -- n(-k) = n(k)* -- so a wedge sum keeps an
antisymmetric imaginary part that a full-zone sum cancels, and the spatial group
average cannot remove it because D(R) is real.

Until 2026-08-09 nothing did, and this case measured the consequence: the wedge
left max|Im n| = 6.8e-3 against an off-diagonal real part of 1.06e-2, and
because the Hubbard force contracts a real part (`2*Re[...]`, itself the
time-reversal average) against a D matrix built from a complex `n`, the two
sides disagreed. Fixed by averaging over time reversal too, as QE does by
accumulating only the real part of the occupation matrix.
"""


@pytest.fixture(scope="module")
def sic_hubbard_symmetry_runs(q3_bin, tmp_path_factory):
    outputs = {}
    for label, offset in (
        ("baseline", 0.0),
        ("plus", SIC_FRACTIONAL_STEP),
        ("minus", -SIC_FRACTIONAL_STEP),
    ):
        directory = tmp_path_factory.mktemp(f"sic_u_wedge_{label}")
        input_path = write_sic_hubbard_input(
            directory, SIC_BASE_DISPLACEMENT + offset, disable_symmetry=False
        )
        completed = subprocess.run(
            [q3_bin, input_path.name],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert completed.returncode == 0, (
            f"q3 exited {completed.returncode} on the {label} run\n{completed.stdout[-3000:]}"
        )
        results = Results.read(directory / "out" / "sic_results.json")
        assert results.runtime_log.converged, f"the {label} run did not converge"
        reduction = re.search(r"IBZ reduction[^:]*:\s*(\d+) -> (\d+) k-points", completed.stdout)
        assert reduction and int(reduction.group(2)) < int(reduction.group(1)), (
            "the k-point set was not reduced, so this fixture is not testing the wedge"
        )
        outputs[label] = Q3Run(results, completed.stdout)
    return outputs


def test_the_wedge_hubbard_force_is_the_gradient_of_q3s_own_energy(sic_hubbard_symmetry_runs):
    """
    The same derivative as above, with the irreducible wedge in play.

    This is the assertion the unsymmetrized occupation matrix failed: it gave
    1.003049 against the 1.000005 measured with the fix, and 0.999868 over the
    full zone. The 0.1% bound sits three times below the defect and two hundred
    times above what is left, so it discriminates in both directions.
    """
    analytic, numerical = _force_against_its_own_gradient(sic_hubbard_symmetry_runs)

    assert abs(numerical) > 1e-3, "the displacement is not producing a resolvable force"
    ratio = analytic / numerical
    assert abs(ratio - 1.0) < 0.001, (
        f"over the irreducible wedge the PAW DFT+U force is {analytic:.8f} Ry/bohr against "
        f"a finite difference of {numerical:.8f} Ry/bohr (ratio {ratio:.6f}) -- the occupation "
        "matrix is not symmetrized consistently with the force"
    )


def test_the_wedge_hubbard_force_matches_the_full_zone(sic_hubbard_runs, sic_hubbard_symmetry_runs):
    """
    Symmetry on and symmetry off must give the same Hubbard force.

    Taken on the Hubbard contribution alone rather than the total, because that
    is where the defect lived and the total dilutes it: with the wedge
    unsymmetrized under time reversal the Hubbard force was 1.5% from its
    full-zone value while the total moved only 0.11%. It is now 0.03%, against
    a 0.04% wedge-versus-full-zone difference measured on the *same geometry
    with no U at all* -- so what is left is ordinary k-sum discretization and
    carries no DFT+U signature.
    """
    wedge = _hubbard_force_along_a1(sic_hubbard_symmetry_runs["baseline"])
    full_zone = _hubbard_force_along_a1(sic_hubbard_runs["baseline"])

    assert abs(full_zone) > 1e-5, "the full-zone Hubbard force is too small to compare against"
    relative_difference = abs(wedge - full_zone) / abs(full_zone)
    assert relative_difference < 0.005, (
        f"the Hubbard force on carbon is {wedge:.9f} Ry/bohr over the wedge against "
        f"{full_zone:.9f} Ry/bohr over the full zone ({100 * relative_difference:.2f}% apart)"
    )


# ── DFT+U forces with ortho-atomic projectors ────────────────────────────────

"""
Ortho-atomic projectors, norm-conserving, gated on the whole force *vector*.

The ortho-atomic projector is not a property of its own atom: the entire
system-wide atomic-orbital set is Loewdin-orthogonalized jointly, so one
manifold's occupation matrix depends on where *every* atom sits, through
O_IJ = <phi_I|phi_J>. Its gradient therefore carries a term the atomic
convention has no analogue of, built from dO/dR and the eigendecomposition of
O. Q3 refused ortho-atomic forces outright until 2026-08-09.

Two things differ from the gates above, both deliberate.

**The whole vector, not one projection.** Three independent crystal-axis
central differences give F.a1, F.a2 and F.a3, and those three numbers are
solved for the Cartesian force. A single-axis projection leaves two components
untested, and this term is a matrix contraction over every orbital in the cell
-- exactly the shape of error that shows up in one component and not another.

**The lattice is written out explicitly** rather than named by Bravais type,
because the axes the differences are taken along are the deck's own rows. A
force projected onto the wrong axis already looked 1.9x wrong once here.

Measured on SiC (norm-conserving, U on silicon only, carbon displaced
off-symmetry in all three crystal directions):

    |F_analytic - F_numerical| / |F| = 4.9e-5   with the dO^-1/2 term
                                       4.2e-2   with it dropped

-- a factor of 860, verified by building Q3 without the term. Dropping it
reverses the sign of the Hubbard force on carbon rather than zeroing it, which
is why the companion assertion below is written against the *atomic*
convention (where it is identically zero) and not against this term alone.
"""

ORTHO_ATOMIC_PSEUDOPOTENTIALS = (
    Path(__file__).resolve().parents[2] / "examples" / "data" / "pseudopotentials" / "nc"
)

ORTHO_ATOMIC_BASE_DISPLACEMENT = (0.020, 0.013, 0.007)
"""
The displaced atom's offset from its lattice site, in every run of the set.

Distinct in all three crystal directions on purpose: a zincblende site has full
tetrahedral symmetry, so a difference centred on the ideal position vanishes
identically, and equal offsets would leave a mirror plane tying two of the
three differences together.
"""

ORTHO_ATOMIC_STEP = 0.001
"""
The finite-difference step, in crystal coordinates.

Measured step-independent: 0.001 and 0.002 give 4.9e-5 and 4.5e-5 on the SiC
case. `convergence_threshold` is 1e-12 for the reason recorded above -- it
bounds the density residual, not the energy, and at 1e-8 a small-step
difference wanders.
"""


def _cubic_face_centred_vectors(conventional_lattice_parameter_bohr):
    """The three FCC primitive vectors, in bohr, written out for the deck."""
    half = 0.5 * conventional_lattice_parameter_bohr
    return [[-half, 0.0, half], [0.0, half, half], [-half, half, 0.0]]


SILICON_CARBIDE_LATTICE = _cubic_face_centred_vectors(4.3596 / BOHR_TO_ANGSTROM)
SILICON_LATTICE = _cubic_face_centred_vectors(10.334401769190510)


def write_ortho_atomic_input(directory, lattice_vectors, elements, hubbard_species, position):
    """A two-atom norm-conserving DFT+U deck with ortho-atomic projectors."""
    material = {
        "lattice": {"vectors": dict(zip(("a", "b", "c"), lattice_vectors))},
        "basis": {
            "units": "crystal",
            "elements": [{"id": index, "value": element} for index, element in enumerate(elements)],
            "coordinates": [
                {"id": 0, "value": [0.0, 0.0, 0.0]},
                {"id": 1, "value": list(position)},
            ],
        },
    }
    material_path = directory / "material.json"
    material_path.write_text(json.dumps(material))

    pseudopotential_lines = "\n".join(
        f"    {element}: {ORTHO_ATOMIC_PSEUDOPOTENTIALS / (element + '.upf')}"
        for element in dict.fromkeys(elements)
    )
    input_path = directory / "scf.yml"
    input_path.write_text(
        f"""material: {material_path}
model: {{type: dft, subtype: gga, functional: pbe}}
method:
  type: pseudopotential
  subtype: norm-conserving
  cutoffs: {{wavefunction: 45.0, density: 180.0}}
  pseudopotentials:
{pseudopotential_lines}
  scf:
    convergence_threshold: 1.0e-12
    max_iterations: 300
    mixing: {{mode: broyden, beta: 0.3}}
    kgrid: {{dimensions: [2, 2, 2], shifts: [0, 0, 0]}}
    number_of_bands: 8
    hubbard_projectors: ortho-atomic
    hubbard:
      - {{species: {hubbard_species}, orbital_label: p, u_ev: 3.0}}
control: {{disable_symmetry: true, calculation: scf, outdir: {directory / "out"}, prefix: scf}}
"""
    )
    return input_path


def run_ortho_atomic_force_set(q3_bin, tmp_path_factory, label, lattice_vectors, elements, species):
    """The baseline run plus the six displaced runs of a three-axis difference."""
    geometries = {"baseline": ORTHO_ATOMIC_BASE_DISPLACEMENT}
    for axis in range(3):
        for direction, sign in (("plus", 1.0), ("minus", -1.0)):
            offset = list(ORTHO_ATOMIC_BASE_DISPLACEMENT)
            offset[axis] += sign * ORTHO_ATOMIC_STEP
            geometries[f"{direction}{axis}"] = offset

    outputs = {}
    for name, offset in geometries.items():
        directory = tmp_path_factory.mktemp(f"ortho_{label}_{name}")
        input_path = write_ortho_atomic_input(
            directory, lattice_vectors, elements, species, [0.25 + value for value in offset]
        )
        completed = subprocess.run(
            [q3_bin, input_path.name],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert completed.returncode == 0, (
            f"q3 exited {completed.returncode} on the {name} run\n{completed.stdout[-3000:]}"
        )
        results = Results.read(directory / "out" / "scf_results.json")
        assert results.runtime_log.converged, f"the {name} run did not converge"
        outputs[name] = Q3Run(results, completed.stdout)
    return outputs


def numerical_force_from_axis_differences(outputs, lattice_vectors):
    """Solve F . a_i = -dE/dq_i for the Cartesian force on the displaced atom."""
    projections = [
        -(
            outputs[f"plus{axis}"].results.properties.total_energy.value
            - outputs[f"minus{axis}"].results.properties.total_energy.value
        )
        / (2.0 * ORTHO_ATOMIC_STEP)
        for axis in range(3)
    ]
    rows = [list(vector) + [projections[index]] for index, vector in enumerate(lattice_vectors)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(3):
            if row == column:
                continue
            factor = rows[row][column] / rows[column][column]
            for entry in range(column, 4):
                rows[row][entry] -= factor * rows[column][entry]
    return [rows[index][3] / rows[index][index] for index in range(3)]


def relative_force_deviation(analytic, numerical):
    magnitude = math.sqrt(sum(value * value for value in analytic))
    assert magnitude > 1e-3, "the displacement is not producing a resolvable force"
    deviation = math.sqrt(sum((analytic[index] - numerical[index]) ** 2 for index in range(3)))
    return deviation / magnitude


@pytest.fixture(scope="module")
def ortho_atomic_silicon_carbide_runs(q3_bin, tmp_path_factory):
    return run_ortho_atomic_force_set(
        q3_bin, tmp_path_factory, "sic", SILICON_CARBIDE_LATTICE, ["Si", "C"], "Si"
    )


@pytest.fixture(scope="module")
def ortho_atomic_silicon_runs(q3_bin, tmp_path_factory):
    return run_ortho_atomic_force_set(
        q3_bin, tmp_path_factory, "si", SILICON_LATTICE, ["Si", "Si"], "Si"
    )


def test_ortho_atomic_hubbard_force_is_the_gradient_of_q3s_own_energy(
    ortho_atomic_silicon_carbide_runs,
):
    """SiC, U on silicon only, carbon displaced -- the full force vector."""
    analytic = force_on_atom(ortho_atomic_silicon_carbide_runs["baseline"], 1)
    numerical = numerical_force_from_axis_differences(
        ortho_atomic_silicon_carbide_runs, SILICON_CARBIDE_LATTICE
    )
    deviation = relative_force_deviation(analytic, numerical)
    assert deviation < 1e-3, (
        f"ortho-atomic DFT+U force {analytic} Ry/bohr vs finite difference {numerical} "
        f"Ry/bohr (relative deviation {deviation:.6f})"
    )


def test_the_lowdin_transform_carries_a_force_to_a_non_hubbard_atom(
    ortho_atomic_silicon_carbide_runs,
):
    """
    Carbon carries no U, and in the atomic convention that makes its Hubbard
    force identically zero -- the atomic gradient only ever writes to the
    manifold's own atom. Ortho-atomic couples it in through O, so a non-zero
    value here is the assertion that the system-wide transform is being
    differentiated at all, rather than the manifold's own orbital alone.
    """
    components = force_contribution_on_atom(
        ortho_atomic_silicon_carbide_runs["baseline"], "hubbard", 1
    )
    magnitude = math.sqrt(sum(component * component for component in components))
    assert magnitude > 1e-4, (
        f"the Hubbard force on the non-Hubbard species is {magnitude:.2e} Ry/bohr, "
        "effectively zero -- the joint Loewdin transform is not reaching it"
    )


def test_ortho_atomic_hubbard_force_is_the_gradient_with_two_manifolds(ortho_atomic_silicon_runs):
    """
    Two Hubbard atoms, so the displaced atom feels its own manifold moving and
    the other's transform-mediated term at once.

    Silicon's ortho-atomic Hubbard force is a much smaller share of the total
    than SiC's (2.4e-5 against 0.074 Ry/bohr, versus 1.6e-3 against 0.064), so
    this is the weaker of the two gates -- 1.1e-5 relative deviation with the
    dO^-1/2 term and 1.7e-3 without. It is here for the manifold sum, which the
    SiC case does not exercise.
    """
    analytic = force_on_atom(ortho_atomic_silicon_runs["baseline"], 1)
    numerical = numerical_force_from_axis_differences(ortho_atomic_silicon_runs, SILICON_LATTICE)
    deviation = relative_force_deviation(analytic, numerical)
    assert deviation < 2e-4, (
        f"ortho-atomic DFT+U force {analytic} Ry/bohr vs finite difference {numerical} "
        f"Ry/bohr (relative deviation {deviation:.6f})"
    )
