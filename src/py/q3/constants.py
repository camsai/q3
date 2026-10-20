"""q3.constants — the unit conversions, in one place.

The engine works in rydberg and bohr; a report is read in electronvolt and
angstrom. The factors between them come from ``mat3ra.utils.constants``, the
table the mat3ra stack generates from one shared source for every language it
runs in, rather than from a literal written out per module: a copied constant
is one that can drift.

The table agrees with the engine's own CODATA constants (``RY_TO_EV`` and
``BOHR_RADIUS_ANGSTROM`` in ``src/layer_0_core/types/constants.hpp``) to
7e-10, so a number converted here and one converted in the engine are the same
number to every digit a report prints; ``tests/python/test_constants.py`` holds
the two to that.
"""

from __future__ import annotations

from mat3ra.utils.constants import COEFFICIENTS

#: Rydberg to electronvolt, and back.
RYDBERG_TO_ELECTRONVOLT = 1.0 / COEFFICIENTS["EV_TO_RY"]
ELECTRONVOLT_TO_RYDBERG = COEFFICIENTS["EV_TO_RY"]

#: Bohr to angstrom, and back. The inverse is taken here rather than read from
#: the table, so that a length converted there and back is the length it was.
BOHR_TO_ANGSTROM = COEFFICIENTS["BOHR_TO_ANGSTROM"]
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
