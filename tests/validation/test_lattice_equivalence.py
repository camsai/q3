"""
General-lattice round-trip test (machine-validated, no reference code needed).

The same physical FCC silicon crystal is fed to q3 two ways:

  1. the Bravais-index path — ESSE lattice ``type: FCC`` + primitive length ``a``
  2. the free-lattice path — explicit ``lattice.vectors`` rows

The two runs must converge to the same total energy: every derived quantity
(reciprocal lattice, G-vectors, volume, symmetry reduction) has to be invariant
under the input representation, including the different alat conventions the
two paths produce (conventional lattice constant vs |a1|).

Run:
    .venv/bin/pytest tests/validation/test_lattice_equivalence.py -v --q3-bin ./build/q3
"""

import math

from helpers import EXAMPLES_DIRECTORY, run_overlay_and_read_energy

SI_NC_INPUT = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml"
SI_NC_PREFIX = "si_nc"

# The committed silicon.json: primitive FCC, a = 3.866976 Å (60° angles).
PRIMITIVE_LENGTH_ANGSTROM = 3.866976
# Conventional cube edge = primitive length × √2.
CONVENTIONAL_ANGSTROM = PRIMITIVE_LENGTH_ANGSTROM * math.sqrt(2.0)

SILICON_BASIS = {
    "elements": [
        {"id": 0, "value": "Si"},
        {"id": 1, "value": "Si"},
    ],
    "coordinates": [
        {"id": 0, "value": [0.875, 0.875, 0.875]},
        {"id": 1, "value": [0.125, 0.125, 0.125]},
    ],
}

# QE ibrav=2 primitive vectors in units of the conventional lattice constant:
# a1 = a/2 (-1,0,1), a2 = a/2 (0,1,1), a3 = a/2 (-1,1,0) — the same rows
# lattice.hpp generates, here given explicitly in angstrom.
HALF = CONVENTIONAL_ANGSTROM / 2.0
EXPLICIT_VECTORS_MATERIAL = {
    "lattice": {
        "vectors": {
            "a": [-HALF, 0.0, HALF],
            "b": [0.0, HALF, HALF],
            "c": [-HALF, HALF, 0.0],
            "units": "angstrom",
        }
    },
    "basis": SILICON_BASIS,
}

TYPE_MATERIAL = {
    "lattice": {
        "type": "FCC",
        "a": PRIMITIVE_LENGTH_ANGSTROM,
        "units": {"length": "angstrom", "angle": "degree"},
    },
    "basis": SILICON_BASIS,
}

REPRESENTATION_EQUIVALENCE_TOLERANCE_RYDBERG = 1e-8
"""
The two representations describe bit-comparable cells; remaining differences
come from alat-convention-dependent floating-point rounding in derived
quantities (measured ~1e-10 Ry).
"""


def test_explicit_vectors_match_bravais_type(q3_command, tmp_path_factory):
    """FCC Si via lattice type and via explicit vectors give the same energy."""
    energy_by_type = run_overlay_and_read_energy(
        q3_command,
        SI_NC_INPUT,
        {"material": TYPE_MATERIAL},
        SI_NC_PREFIX,
        tmp_path_factory.mktemp("lattice_by_type"),
    )
    energy_by_vectors = run_overlay_and_read_energy(
        q3_command,
        SI_NC_INPUT,
        {"material": EXPLICIT_VECTORS_MATERIAL},
        SI_NC_PREFIX,
        tmp_path_factory.mktemp("lattice_by_vectors"),
    )
    difference = abs(energy_by_type - energy_by_vectors)
    assert difference < REPRESENTATION_EQUIVALENCE_TOLERANCE_RYDBERG, (
        f"Lattice-representation equivalence violated:\n"
        f"  E(type: FCC)        = {energy_by_type:.12f} Ry\n"
        f"  E(explicit vectors) = {energy_by_vectors:.12f} Ry\n"
        f"  |diff|              = {difference:.2e} Ry"
    )
