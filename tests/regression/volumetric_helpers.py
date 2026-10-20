"""
What the `q3_utility volumetric` regression tests share: the silicon run they
export from, the electron count every density must integrate to, a cube
reader, and the export call itself. The fixtures built on these live in
tests/regression/conftest.py.
"""

import subprocess
from pathlib import Path

import numpy as np

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, run_q3

BASE_EXAMPLE = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"


# The example's norm-conserving silicon dataset carries four valence electrons
# and the cell holds two atoms.
VALENCE_ELECTRONS = 8.0


# Both runs keep the example's own cutoffs and dataset and change only the
# k-mesh and the prefix, so the density being integrated is one a published
# example produced rather than one tuned to make the check pass.
SHIFTED_OVERLAY = {
    "control": {"prefix": "si_volumetric"},
}


def read_cube(path):
    """Parse a Gaussian cube into its cell, atoms and samples.

    Deliberately a parser of its own rather than a call into anything Q3
    ships: a test that re-uses the writer's own reader cannot catch the two of
    them agreeing on a wrong convention.
    """
    tokens = Path(path).read_text().split("\n")
    header = tokens[2].split()
    number_of_atoms = int(header[0])

    grid_dimensions = []
    voxel_vectors = []
    for axis in range(3):
        fields = tokens[3 + axis].split()
        grid_dimensions.append(int(fields[0]))
        voxel_vectors.append([float(value) for value in fields[1:4]])

    atoms = []
    for atom_index in range(number_of_atoms):
        fields = tokens[6 + atom_index].split()
        atoms.append(
            {
                "atomic_number": int(fields[0]),
                "charge": float(fields[1]),
                "position": [float(value) for value in fields[2:5]],
            }
        )

    values = []
    for line in tokens[6 + number_of_atoms :]:
        values.extend(float(value) for value in line.split())

    grid_dimensions = np.array(grid_dimensions)
    lattice = np.array(voxel_vectors) * grid_dimensions[:, None]
    expected = int(np.prod(grid_dimensions))
    assert len(values) == expected, f"{path} holds {len(values)} values, expected {expected}"
    return {
        "grid_dimensions": grid_dimensions,
        "voxel_vectors": np.array(voxel_vectors),
        "lattice_bohr": lattice,
        "atoms": atoms,
        # The cube's own ordering: first axis outermost, third innermost.
        "values": np.array(values).reshape(grid_dimensions),
    }


def run_scf(q3_command, tmp_path_factory, overlay, name):
    directory = tmp_path_factory.mktemp(name)
    merged_yaml = apply_config_overlay(BASE_EXAMPLE, overlay, directory)
    result = run_q3(q3_command, merged_yaml, working_directory=directory)
    assert result.returncode == 0, f"q3 failed:\n{result.stderr}\n{result.stdout}"
    return {
        "input": merged_yaml,
        "directory": directory,
        "output_directory": directory / "output",
        "prefix": overlay["control"]["prefix"],
    }


def export(q3_utility_binary, run, *arguments):
    """Run `q3_utility volumetric` on a finished run and return the written path."""
    result = subprocess.run(
        [q3_utility_binary, "volumetric", str(run["input"]), *arguments],
        capture_output=True,
        text=True,
        cwd=run["directory"],
        timeout=300,
    )
    assert result.returncode == 0, f"q3_utility failed:\n{result.stderr}\n{result.stdout}"
    return result
