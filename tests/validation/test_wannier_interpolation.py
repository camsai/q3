"""End-to-end validation of Wannier interpolation against Quantum ESPRESSO.

Runs the full reference chain (scf, full-grid nosym nscf, ``wannier90 -pp``,
``pw2wannier90``), imports the resulting matrices into an interpolation record
with ``q3_utility convert --from wannier90``, evaluates it with ``q3_utility
interpolate``, and checks the interpolated bands and density of states.

The chain is exercised on Quantum ESPRESSO's matrices ON PURPOSE. Q3's own
overlap kernels are not in the loop here, so a disagreement can only come from
the interpolation engine -- gauge, Wigner-Seitz set, Fourier transforms --
and not from the plane-wave side. Keeping the two separable is what makes a
failure here diagnostic rather than merely alarming. The second half of the
module then runs the fully native chain and compares the two.

Two very different checks:

- **At the coarse mesh points** the k -> R -> k round trip is an exact
  identity, so the interpolated eigenvalues must reproduce the input `.eig` to
  machine precision. This tests the transform and says nothing about physics.
- **Between the mesh points** there is nothing to be exact about: the
  interpolation error is real, and its size depends on how localized the gauge
  is and how dense the mesh is. The band-distance thresholds below are
  calibrated on measured values, with headroom, and are deliberately loose
  enough to pass with the one-shot projection gauge -- see the note on the
  path test.

QE runs take minutes, so everything here is marked slow (tier 2) and skipped
unless the reference binaries are present.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers_wannier import (
    OverlapFile,
    band_distance,
    compute_b_vectors,
    gauge_invariant_spread,
    locate_reference_binary_directory,
    read_eigenvalue_file,
    read_neighbor_list_file,
    read_overlap_file,
    read_projection_file,
    solve_finite_difference_weights,
)
from q3.io.interpolation_settings import EvaluationSettings
from q3.workflows.interpolation import (
    evaluate_interpolation,
    import_wannier90_record,
    write_interpolation_evaluation_input,
)

from helpers import RYDBERG_TO_ELECTRONVOLT

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "band_structure--semiconductor--silicon--wannier"
REFERENCE_INPUTS = EXAMPLE / "reference"
REFERENCE_BINARIES = locate_reference_binary_directory()
Q3_BINARY = REPOSITORY_ROOT / "build" / "q3"
Q3_UTIL = REPOSITORY_ROOT / "build" / "q3_utility"
GRID = (4, 4, 4)
PATH_CORNERS = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)]

REQUIRED_BINARIES = ("pw.x", "wannier90.x", "pw2wannier90.x")

pytestmark = pytest.mark.slow


def _missing_requirements() -> str | None:
    if REFERENCE_BINARIES is None:
        return "the QE reference build was not found"
    for name in REQUIRED_BINARIES:
        if not (REFERENCE_BINARIES / name).is_file():
            return f"{name} is not built in {REFERENCE_BINARIES}"
    if not Q3_UTIL.is_file():
        return f"{Q3_UTIL} is not built"
    return None


def _run(binary: str, arguments: list[str], working_directory: Path, log_name: str) -> None:
    log_path = working_directory / log_name
    with log_path.open("w") as log:
        result = subprocess.run(
            [str(REFERENCE_BINARIES / binary), *arguments],
            cwd=working_directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-15:])
        raise AssertionError(f"{binary} failed (exit {result.returncode}):\n{tail}")


def _band_path(corners, per_segment: int = 20) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for start, end in zip(corners[:-1], corners[1:]):
        for step in range(per_segment):
            fraction = step / per_segment
            points.append(tuple(s + fraction * (e - s) for s, e in zip(start, end)))
    points.append(tuple(corners[-1]))
    return points


def _coarse_mesh() -> list[tuple[float, float, float]]:
    return [
        (x / GRID[0], y / GRID[1], z / GRID[2])
        for x in range(GRID[0])
        for y in range(GRID[1])
        for z in range(GRID[2])
    ]


def _evaluate(
    directory: Path,
    *,
    kpoints=None,
    dos_mesh=None,
    broadening: float | None = None,
    localization: str | None = None,
) -> dict:
    """Evaluate the record `directory/si.save/interpolation.q3ip` with q3_utility.

    The evaluation manifest borrows material/model/method from the example's
    SCF input; only the record's location and what to evaluate are this
    test's own.
    """
    manifest = write_interpolation_evaluation_input(
        directory / "evaluate.yml",
        template_input_path=EXAMPLE / "nc_scf.yml",
        outdir=directory,
        prefix="si",
        settings=EvaluationSettings(dos_mesh=dos_mesh, localization=localization),
        kpoints=kpoints,
        dos_broadening_ry=broadening,
    )
    return evaluate_interpolation(manifest, q3_utility_binary=Q3_UTIL)


def _bands_ev(results: dict) -> np.ndarray:
    """[k-point][band] in eV, ascending within each row."""
    return np.sort(np.array(results["eigenvalues_ry"])[:, 0, :], axis=1) * RYDBERG_TO_ELECTRONVOLT


@pytest.fixture(scope="module")
def reference_chain(tmp_path_factory):
    """Run the QE chain once, import its matrices, and share it across this module."""
    reason = _missing_requirements()
    if reason is not None:
        pytest.skip(reason)

    workspace = tmp_path_factory.mktemp("wannier_interpolation")
    for name in ("scf.in", "nscf.in", "si.win", "si.pw2wan"):
        shutil.copy(REFERENCE_INPUTS / name, workspace / name)
    shutil.copy(
        REPOSITORY_ROOT / "examples" / "data" / "pseudopotentials" / "nc" / "Si.upf",
        workspace / "Si.upf",
    )

    _run("pw.x", ["-i", "scf.in"], workspace, "scf.out")
    _run("pw.x", ["-i", "nscf.in"], workspace, "nscf.out")
    _run("wannier90.x", ["-pp", "si"], workspace, "w90pp.out")
    _run("pw2wannier90.x", ["-i", "si.pw2wan"], workspace, "pw2wan.out")
    import_wannier90_record(
        workspace / "si", GRID, outdir=workspace, prefix="si", q3_utility_binary=Q3_UTIL
    )
    return workspace


def test_interpolation_reproduces_the_input_eigenvalues_at_the_coarse_mesh(reference_chain):
    """The k -> R -> k round trip is an identity at the mesh points.

    Machine precision is the right expectation here -- this is testing the
    Fourier transform and the Wigner-Seitz degeneracy weights, not physics.
    """
    interpolated = _bands_ev(_evaluate(reference_chain, kpoints=_coarse_mesh()))
    reference = np.sort(read_eigenvalue_file(reference_chain / "si.eig").eigenvalues, axis=1)
    assert interpolated.shape == reference.shape
    deviation = np.abs(interpolated - reference).max()
    assert deviation < 1e-9, f"round trip is not exact at the mesh points: {deviation:.3e} eV"


def test_wigner_seitz_weight_matches_the_kpoint_count(reference_chain):
    """On a real fcc lattice, not only the cubic ones the unit tests use."""
    results = _evaluate(reference_chain)
    assert results["interpolation"]["method"] == "wannier"
    assert results["interpolation"]["total_bloch_sum_weight"] == pytest.approx(64.0, abs=1e-9)


def test_interpolated_bands_track_the_direct_calculation_along_a_path(reference_chain):
    """Interpolation between mesh points, against QE's own band calculation.

    Both codes are handed the identical crystal-coordinate k-point list, so
    the comparison is not sensitive to path conventions.

    The threshold is 150 meV, which sounds loose and is: this is a 4x4x4
    development mesh with the ONE-SHOT PROJECTION GAUGE and no maximal
    localization, and the measured band distance there is ~68 meV. The same
    engine on an 8x8x8 mesh measures ~9.5 meV, and the plan's 5 meV gate
    assumes a localized gauge -- so this test is a regression guard on the
    development configuration, deliberately not the accuracy gate. Tightening
    it belongs with the localizer, not here.

    The bands run gets its OWN outdir: a QE 'bands' calculation writes
    wavefunctions for its path k-points, and sharing outdir with the nscf
    would overwrite the 64-k-point set that pw2wannier90 needs, breaking the
    other tests in this module.
    """
    workspace = reference_chain
    points = _band_path(PATH_CORNERS)

    bands_directory = workspace / "qe_bands"
    if bands_directory.exists():
        shutil.rmtree(bands_directory)
    shutil.copytree(workspace / "qe_output", bands_directory)
    header = (workspace / "nscf.in").read_text().split("K_POINTS")[0]
    header = header.replace("calculation = 'nscf'", "calculation = 'bands'")
    header = header.replace("outdir = './qe_output'", "outdir = './qe_bands'")
    listing = ["K_POINTS crystal", f"  {len(points)}"]
    listing += [f"  {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}  1.0" for p in points]
    (workspace / "bands.in").write_text(header + "\n".join(listing) + "\n")
    _run("pw.x", ["-i", "bands.in"], workspace, "bands.out")

    blocks = re.findall(
        r"bands \(ev\):\s*\n\n((?:\s*-?\d+\.\d+)+)", (workspace / "bands.out").read_text()
    )
    assert len(blocks) == len(points), "QE did not report bands at every path k-point"
    direct = np.sort(np.array([[float(v) for v in b.split()] for b in blocks]), axis=1)

    interpolated = _bands_ev(_evaluate(workspace, kpoints=points))
    direct = direct[:, : interpolated.shape[1]]

    result = band_distance(direct, interpolated, fermi_energy=direct.max())
    assert result.eta < 0.150, (
        f"band distance {result.eta * 1000:.1f} meV exceeds the development-configuration "
        f"guard of 150 meV"
    )


def _neighbor_alignment(reference: OverlapFile, other: OverlapFile) -> np.ndarray:
    """For each k-point, where each of `other`'s neighbours sits in `reference`.

    A neighbour is identified by the k-point index it points at and the
    reciprocal-lattice shift that carries that image to k+b -- what it is,
    not where `wannier90 -pp` happened to list it. Every neighbour of `other`
    must appear exactly once in `reference`; otherwise the two files describe
    different stencils and comparing their overlaps would be meaningless.
    """
    number_of_kpoints, number_of_neighbors = other.neighbor_kpoint.shape
    assert reference.neighbor_kpoint.shape == (number_of_kpoints, number_of_neighbors), (
        "the two runs disagree on the neighbour count"
    )
    alignment = np.empty((number_of_kpoints, number_of_neighbors), dtype=int)
    for kpoint_index in range(number_of_kpoints):
        positions = {
            (
                int(reference.neighbor_kpoint[kpoint_index, n]),
                *map(int, reference.periodic_image_shift[kpoint_index, n]),
            ): n
            for n in range(number_of_neighbors)
        }
        assert len(positions) == number_of_neighbors, (
            f"k-point {kpoint_index} lists a neighbour twice"
        )
        for n in range(number_of_neighbors):
            key = (
                int(other.neighbor_kpoint[kpoint_index, n]),
                *map(int, other.periodic_image_shift[kpoint_index, n]),
            )
            assert key in positions, f"k-point {kpoint_index}: neighbour {key} is in one run only"
            alignment[kpoint_index, n] = positions[key]
    return alignment


def test_overlaps_and_omega_i_do_not_depend_on_the_trial_orbitals(reference_chain):
    """Ω_I is a property of the manifold, not of the guess -- proven, not assumed.

    The committed `si.win` uses bond-centred s projections, which interpolate
    measurably better than atom-centred sp3 (about 68 meV against 106 meV on
    this grid). It would be easy to assume that improvement came with some
    change to the gauge-invariant spread too. It cannot: `.mmn` holds
    <u_mk|u_n,k+b>, which never mentions the trial orbitals, so a different
    projection choice must leave the overlaps -- and therefore Ω_I -- exactly
    alone while changing `.amn` completely.

    That asymmetry is what licenses comparing Ω_I across codes at all, so it
    is worth checking on real data rather than reasoning about. Re-runs
    `wannier90 -pp` and `pw2wannier90` with sp3 projections in place of the
    bond-centred ones and compares both files.
    """
    workspace = reference_chain
    bond_centred = (workspace / "si.win").read_text()
    projection_block = re.search(
        r"begin projections\n.*?end projections", bond_centred, flags=re.DOTALL
    )
    assert projection_block is not None
    alternative = bond_centred.replace(
        projection_block.group(0),
        "begin projections\nf=0.00,0.00,0.00:sp3\nend projections",
    )
    (workspace / "alt.win").write_text(alternative)

    # Its own outdir. This reruns pw2wannier90 against the saved
    # wavefunctions, and another test in this module runs a QE 'bands'
    # calculation that writes wavefunctions of its own; sharing a directory
    # would make this test depend on which ran first, and a wrong-wavefunction
    # read here would look like a physics disagreement rather than a clash.
    alternative_directory = workspace / "qe_alt"
    if alternative_directory.exists():
        shutil.rmtree(alternative_directory)
    shutil.copytree(workspace / "qe_output", alternative_directory)
    (workspace / "alt.pw2wan").write_text(
        (workspace / "si.pw2wan")
        .read_text()
        .replace("'si'", "'alt'")
        .replace("'./qe_output'", "'./qe_alt'")
    )

    _run("wannier90.x", ["-pp", "alt"], workspace, "w90pp_alt.out")
    _run("pw2wannier90.x", ["-i", "alt.pw2wan"], workspace, "pw2wan_alt.out")

    # The projections really did change -- otherwise the rest proves nothing.
    original_projections = read_projection_file(workspace / "si.amn").projections
    alternative_projections = read_projection_file(workspace / "alt.amn").projections
    assert not np.allclose(original_projections, alternative_projections), (
        "the two projection sets are identical; the test is not exercising what it claims"
    )

    # The overlaps did not -- elementwise, once the neighbours are aligned.
    # Two independent `wannier90 -pp` runs need not enumerate the shell in the
    # same sequence, so the blocks are matched by what each neighbour IS (the
    # k-point it points at and the periodic-image shift) rather than by its
    # position. Comparing raw arrays by position was failing intermittently
    # on exactly that ordering; comparing sorted per-k sums of |M|^2 instead
    # was order-proof but blind to any change that preserves those sums. The
    # aligned comparison is both.
    original = read_overlap_file(workspace / "si.mmn")
    alternative_overlaps = read_overlap_file(workspace / "alt.mmn")
    assert alternative_overlaps.number_of_bands == original.number_of_bands
    alignment = _neighbor_alignment(original, alternative_overlaps)
    aligned_original = np.take_along_axis(
        original.overlaps, alignment[:, :, np.newaxis, np.newaxis], axis=1
    )
    np.testing.assert_allclose(
        alternative_overlaps.overlaps, aligned_original, rtol=0.0, atol=1e-12
    )

    # Omega_I itself, which is the quantity the cross-code comparisons rest on.
    # Each is computed with its OWN neighbour list and weights, so this holds
    # whatever order the shells came out in.
    original_weights = solve_finite_difference_weights(
        compute_b_vectors(read_neighbor_list_file(workspace / "si.nnkp"))
    )
    alternative_weights = solve_finite_difference_weights(
        compute_b_vectors(read_neighbor_list_file(workspace / "alt.nnkp"))
    )
    spread = gauge_invariant_spread(original.overlaps, original_weights, 4)
    alternative_spread = gauge_invariant_spread(
        alternative_overlaps.overlaps, alternative_weights, 4
    )
    assert alternative_spread == pytest.approx(spread, abs=1e-12), (
        f"Omega_I changed with the trial orbitals: {spread} vs {alternative_spread}"
    )


def test_interpolated_density_of_states_integrates_to_the_band_count(reference_chain):
    """Normalisation, on real data rather than the analytic unit-test model.

    A DOS whose integral drifts is wrong by a scale factor that inspecting its
    shape would never reveal, so this is the check that makes the DOS
    comparable to anything else.
    """
    results = _evaluate(reference_chain, dos_mesh=(10, 10, 10), broadening=0.01)
    density = results["density_of_states"]
    energies = np.array(density["energies_ry"])
    values = np.array(density["dos"])

    assert np.all(values >= 0.0)
    integral = float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(energies)))
    assert integral == pytest.approx(4.0, abs=0.05), (
        f"the DOS integrates to {integral:.4f}, not the 4 Wannier functions in the model"
    )


# ── The fully native path: Q3 produces the record as well as evaluating it ──


@pytest.fixture(scope="module")
def native_chain(tmp_path_factory):
    """Q3 SCF -> Q3 NSCF building the record. No other code."""
    if not Q3_BINARY.is_file() or not Q3_UTIL.is_file():
        pytest.skip("q3 and q3_utility must both be built")

    workspace = tmp_path_factory.mktemp("wannier_native")
    for name in ("nc_scf.yml", "nc_nscf.yml"):
        document = yaml.safe_load((EXAMPLE / name).read_text())
        document["material"] = str(REPOSITORY_ROOT / "examples/data/materials/silicon.json")
        document["method"]["pseudopotentials"]["Si"] = str(
            REPOSITORY_ROOT / "examples/data/pseudopotentials/nc/Si.upf"
        )
        (workspace / name.replace("nc_", "")).write_text(yaml.safe_dump(document, sort_keys=False))

    for stage in ("scf.yml", "nscf.yml"):
        result = subprocess.run(
            [str(Q3_BINARY), stage], cwd=workspace, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"q3 {stage} failed:\n{result.stdout[-2000:]}"
    return workspace / "output"


def test_native_nscf_writes_the_record_and_the_exported_files(native_chain):
    """The record is the output; the Wannier90 files are its opt-in export."""
    record = native_chain / "si.save" / "interpolation.q3ip"
    assert record.is_file() and record.stat().st_size > 0, f"the NSCF run did not write {record}"
    for extension in (".mmn", ".amn", ".eig", ".nnkp"):
        produced = native_chain / f"si{extension}"
        assert produced.is_file(), f"the NSCF run did not export {produced}"
        assert produced.stat().st_size > 0


def test_native_record_round_trips_through_the_engine(native_chain):
    """Q3's own record, evaluated at the coarse mesh, reproduces its .eig.

    Nothing but Q3 appears anywhere in this path, and the transform is an
    exact identity at the mesh points, so machine precision is the only
    acceptable answer. This is what makes the pipeline native rather than
    merely present.
    """
    interpolated = _bands_ev(_evaluate(native_chain, kpoints=_coarse_mesh()))
    reference = np.sort(read_eigenvalue_file(native_chain / "si.eig").eigenvalues, axis=1)
    deviation = np.abs(interpolated - reference).max()
    assert deviation < 1e-9, f"native round trip is not exact: {deviation:.3e} eV"


def test_native_eigenvalues_match_quantum_espresso(native_chain, reference_chain):
    """The two codes' NSCF eigenvalues, on the same cell and grid.

    This is the check that a cross-code Wannier comparison rests on: if the
    inputs to the interpolation already differ, no statement about the
    interpolation is attributable. An earlier version of this example had the
    QE side at a 1.3 % compressed cell, which showed up here as a ~400 meV
    disagreement and nothing else.

    The valence manifold is compared, not every band: deviations above
    0.1 meV occur only in the topmost requested band, where a truncated
    Davidson set converges slowest for want of a buffer state above it.

    The gate is 0.5 meV against a measured maximum of 0.006 meV over all 64
    k-points -- sixty-fold headroom, and still a hundredth of the smallest
    interpolation effect anything downstream measures. An earlier version of
    this test compared against 0.5 eV while printing meV, which the cell
    mismatch above would have passed with room to spare.
    """
    native = np.sort(read_eigenvalue_file(native_chain / "si.eig").eigenvalues, axis=1)
    espresso = np.sort(read_eigenvalue_file(reference_chain / "si.eig").eigenvalues, axis=1)
    assert native.shape == espresso.shape

    valence = min(4, native.shape[1])
    deviation = np.abs(native[:, :valence] - espresso[:, :valence])
    assert deviation.max() < 0.0005, (
        f"Q3 and QE disagree on the valence eigenvalues by {deviation.max() * 1000:.3f} meV "
        f"(gate 0.5 meV); check that both sides use the same cell"
    )


def test_native_interpolation_matches_the_quantum_espresso_sourced_interpolation(
    native_chain, reference_chain
):
    """The same path, evaluated once from Q3's record and once from QE's matrices.

    This is the guard on Q3's interface kernels as a whole. `q3_utility
    interpolate` is the same binary on both sides, so whatever separates the
    two band sets was in the records it read: the eigenvalues, and the
    trial-orbital projections that fix the gauge. The comparison is
    interpolation against interpolation rather than each against a direct
    calculation, because both share the ~68 meV of interpolation error of
    this coarse grid, and a kernel discrepancy two orders of magnitude smaller
    would vanish under it.

    The two cannot agree exactly, and the gate says by how much they may not.
    `si.win` and `nc_nscf.yml` place s orbitals at the same four bond centres
    with the same 1 bohr decay (pw2wannier90 applies `zona` in inverse bohr,
    whatever the user guide says -- see the comment in `si.win`), but Q3
    Loewdin-orthonormalizes the trial orbitals in the S metric before
    projecting while pw2wannier90 projects onto the raw guiding functions, so
    the two projection gauges differ by construction, and pw2wannier90's
    radial transform is numerical where Q3's is analytic. Measured: 0.60 meV
    rms, 2.1 meV worst state. The gate is 2 meV rms -- three times that, and
    still a hundredth of the shared interpolation error. A kernel error is
    not subtle at this scale: a misplaced centre alone measures ~195 meV
    against a direct calculation, and a mismatched decay 3.7 meV here.
    """
    points = _band_path(PATH_CORNERS)
    # Like with like. This test is about where the MATRICES came from, and the
    # QE-sourced record carries projections and eigenvalues but no overlaps, so
    # it cannot be localised at all -- under the default that record would keep
    # its projection gauge while the native one was minimised, and the gap
    # between two different gauges would be scored as a difference in provenance.
    native = _bands_ev(_evaluate(native_chain, kpoints=points, localization="none"))
    espresso = _bands_ev(_evaluate(reference_chain, kpoints=points, localization="none"))
    assert native.shape == espresso.shape

    result = band_distance(espresso, native, fermi_energy=espresso.max())
    assert result.eta < 0.002, (
        f"the natively sourced interpolation is {result.eta * 1000:.3f} meV of band distance "
        f"from the QE-sourced one (gate 2 meV, worst weighted state "
        f"{result.eta_maximum * 1000:.3f} meV)"
    )
