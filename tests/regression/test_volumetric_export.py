"""End-to-end checks for `q3_utility volumetric` — cube and xsf output.

A volumetric writer is easy to get plausibly wrong: a transposed grid, a
missing factor of the cell volume or a length in the wrong unit all produce a
file that opens in a viewer and draws something crystal-shaped. So nothing
here inspects the writer's input. Every assertion is made on the bytes that
reached disk, and the quantities checked are the ones a mistake moves:

*   **The integral of the density over the cell is the valence electron
    count.** Silicon's norm-conserving dataset carries four valence electrons
    per atom and the cell holds two atoms, so the answer is 8 and no other
    number. A transposed grid leaves it unchanged, but a missing volume factor,
    a wrong point count or a half-filled gamma-only sphere do not.
*   **The written samples equal the engine's own real-space density**, read
    straight out of `effective-potential.q3cp`. That is what pins the grid
    ordering: the cube is rebuilt from rho(G) by an inverse transform, and the
    checkpoint holds rho(r) as the SCF itself stored it, so agreeing point for
    point means the transform and the file layout both place r where the engine
    does.
*   **The two formats describe the same field**, each in its own convention:
    the cube runs its third axis fastest and stores the density; the xsf runs
    its first axis fastest and repeats the boundary plane.
*   **A band's probability density integrates to one**, which is the
    wavefunction path's equivalent of the electron count.

The gamma-only case is run separately because it is the one that stores half
the G-sphere: the conjugate half has to be rebuilt on the way out, and a
reader that forgets gives exactly half the electrons.
"""

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from volumetric_helpers import VALENCE_ELECTRONS, export, read_cube, run_scf

GAMMA_OVERLAY = {
    "method": {"scf": {"kgrid": {"dimensions": [1, 1, 1], "shifts": [0, 0, 0]}}},
    "control": {"prefix": "si_volumetric_gamma"},
}


# ── Readers ───────────────────────────────────────────────────────────


def cube_integral(cube):
    """`Omega / N * sum(values)` — the electron count for a density."""
    volume = abs(np.linalg.det(cube["lattice_bohr"]))
    return cube["values"].sum() * volume / cube["values"].size


def read_xsf_datagrid(path):
    """The first BEGIN_DATAGRID_3D block of an xsf, as a (n1, n2, n3) array."""
    lines = Path(path).read_text().split("\n")
    start = next(index for index, line in enumerate(lines) if "BEGIN_DATAGRID_3D" in line)
    dimensions = [int(value) for value in lines[start + 1].split()]
    span = np.array(
        [[float(value) for value in lines[start + 3 + axis].split()] for axis in range(3)]
    )

    values = []
    for line in lines[start + 6 :]:
        if "END_DATAGRID_3D" in line:
            break
        values.extend(float(value) for value in line.split())
    expected = dimensions[0] * dimensions[1] * dimensions[2]
    assert len(values) == expected, f"{path} holds {len(values)} values, expected {expected}"
    # The xsf's own ordering: first axis fastest, so the array is read in reverse.
    array = np.array(values).reshape(dimensions[2], dimensions[1], dimensions[0])
    return {"dimensions": dimensions, "span_angstrom": span, "values": array.transpose(2, 1, 0)}


def read_checkpoint_charge_density(path):
    """The real-space charge density out of `effective-potential.q3cp`.

    The layout is the one `Q3PlanewaveKohnShamQuantumSystemCheckpoint` writes:
    magic, version, four ints, a double, two bytes of flags, then the density
    section as a spin count followed by a length and that many doubles.
    """
    data = Path(path).read_bytes()
    magic, version = struct.unpack_from("<II", data, 0)
    assert magic == 0x51334350, f"{path} is not a Q3 checkpoint"
    assert version >= 2, f"{path} is version {version}; the density section needs 2 or later"
    offset = 8 + 4 * 4 + 8 + 1 + 1

    (number_of_spin_components,) = struct.unpack_from("<i", data, offset)
    offset += 4
    channels = []
    for _ in range(number_of_spin_components):
        (number_of_grid_points,) = struct.unpack_from("<i", data, offset)
        offset += 4
        channels.append(
            np.frombuffer(data, dtype=np.float64, count=number_of_grid_points, offset=offset)
        )
        offset += 8 * number_of_grid_points
    return channels


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gamma_run(q3_command, tmp_path_factory):
    """A gamma-only silicon SCF, whose saved density is a half sphere."""
    return run_scf(q3_command, tmp_path_factory, GAMMA_OVERLAY, "volumetric_gamma")


# ── The electron count ────────────────────────────────────────────────


def test_density_cube_integrates_to_the_valence_electron_count(density_cube):
    """The one number that says the file is right, taken from the file itself."""
    cube = read_cube(density_cube)
    assert cube_integral(cube) == pytest.approx(VALENCE_ELECTRONS, abs=1e-6)


def test_gamma_only_density_cube_integrates_to_the_same_count(q3_utility_binary, gamma_run):
    """A half-sphere file whose conjugate half was not rebuilt gives half the electrons."""
    export(q3_utility_binary, gamma_run, "--field", "density", "--format", "cube")
    path = gamma_run["output_directory"] / f"{gamma_run['prefix']}_density.cube"
    assert cube_integral(read_cube(path)) == pytest.approx(VALENCE_ELECTRONS, abs=1e-6)


# ── Units and axis order ──────────────────────────────────────────────


def test_cube_is_in_bohr_with_the_voxel_vectors_spanning_the_cell(density_cube):
    cube = read_cube(density_cube)
    # Positive voxel counts are the format's way of saying the lengths are bohr.
    assert all(dimension > 0 for dimension in cube["grid_dimensions"])
    # The example's primitive FCC cell: |a_i| = 3.866976 A = 7.3075 bohr. A
    # length written in Angstrom instead would be smaller by 1.89.
    lengths = np.linalg.norm(cube["lattice_bohr"], axis=1)
    assert lengths == pytest.approx([7.3075, 7.3075, 7.3075], abs=1e-3)
    assert len(cube["atoms"]) == 2
    for atom in cube["atoms"]:
        assert atom["atomic_number"] == 14
        assert atom["charge"] == pytest.approx(4.0)


def test_cube_samples_equal_the_engines_own_real_space_density(density_cube, shifted_run):
    """Pins the grid ordering against the array the SCF itself stored."""
    checkpoint = shifted_run["output_directory"] / f"{shifted_run['prefix']}.save"
    channels = read_checkpoint_charge_density(checkpoint / "effective-potential.q3cp")
    cube = read_cube(density_cube)

    stored = channels[0].reshape(cube["grid_dimensions"])
    assert stored.shape == cube["values"].shape
    scale = np.abs(stored).max()
    # The cube is rebuilt from rho(G), the checkpoint holds rho(r) as the SCF
    # stored it, and the two agree to the precision the file carries.
    assert np.abs(cube["values"] - stored).max() < 1e-6 * scale


def test_xsf_repeats_the_boundary_and_holds_the_same_field(
    q3_utility_binary, shifted_run, density_cube
):
    export(q3_utility_binary, shifted_run, "--field", "density", "--format", "xsf")
    path = shifted_run["output_directory"] / f"{shifted_run['prefix']}_density.xsf"
    datagrid = read_xsf_datagrid(path)
    cube = read_cube(density_cube)

    # An xsf general datagrid spans its vectors including the far end.
    assert datagrid["dimensions"] == [dimension + 1 for dimension in cube["grid_dimensions"]]
    # And its lengths are Angstrom, unlike the cube's bohr.
    bohr_to_angstrom = 0.529177210903
    assert np.allclose(
        datagrid["span_angstrom"], cube["lattice_bohr"] * bohr_to_angstrom, atol=1e-6
    )

    interior = datagrid["values"][:-1, :-1, :-1]
    assert np.allclose(interior, cube["values"], rtol=1e-8, atol=1e-12)
    # The repeated planes are copies of the first, not a continuation.
    assert np.allclose(datagrid["values"][-1, :-1, :-1], cube["values"][0, :, :])


# ── The other two fields ──────────────────────────────────────────────


def test_potential_cube_is_on_the_same_grid_and_is_not_volume_weighted(
    q3_utility_binary, shifted_run, density_cube
):
    export(q3_utility_binary, shifted_run, "--field", "potential", "--format", "cube")
    path = shifted_run["output_directory"] / f"{shifted_run['prefix']}_potential.cube"
    potential = read_cube(path)
    cube = read_cube(density_cube)

    assert list(potential["grid_dimensions"]) == list(cube["grid_dimensions"])
    # V_eff is attractive near the ions and of order a Rydberg, never a density.
    assert potential["values"].min() < -0.1
    assert np.isfinite(potential["values"]).all()


def test_band_probability_density_integrates_to_one(q3_utility_binary, shifted_run):
    """The wavefunction path's equivalent of the electron count."""
    export(
        q3_utility_binary,
        shifted_run,
        "--field",
        "wavefunction",
        "--kpoint",
        "1",
        "--band",
        "1",
        "--format",
        "cube",
    )
    path = shifted_run["output_directory"] / f"{shifted_run['prefix']}_wavefunction.cube"
    cube = read_cube(path)
    assert (cube["values"] >= 0.0).all()
    assert cube_integral(cube) == pytest.approx(1.0, abs=1e-6)


def test_a_band_that_is_not_there_is_refused(q3_utility_binary, shifted_run):
    result = subprocess.run(
        [
            q3_utility_binary,
            "volumetric",
            str(shifted_run["input"]),
            "--field",
            "wavefunction",
            "--band",
            "999",
        ],
        capture_output=True,
        text=True,
        cwd=shifted_run["directory"],
        timeout=300,
    )
    assert result.returncode != 0
    assert "band" in (result.stderr + result.stdout)
