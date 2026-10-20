"""q3.constants reads the stack's generated table, and that table agrees with the engine.

The engine derives its constants from CODATA SI values at compile time
(src/layer_0_core/types/constants.hpp); the Python side reads the mat3ra
stack's generated table. A report converted on one side and read on the other
must be the same number, so the two are pinned against each other here rather
than discovered to differ in a comparison later.
"""

import pytest
from q3.constants import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    ELECTRONVOLT_TO_RYDBERG,
    RYDBERG_TO_ELECTRONVOLT,
)

#: CODATA 2018, as the engine derives them.
RYDBERG_ELECTRONVOLT_CODATA = 13.605693122994
BOHR_RADIUS_ANGSTROM_CODATA = 0.529177210903


def test_the_conversions_agree_with_the_engines_constants():
    assert RYDBERG_TO_ELECTRONVOLT == pytest.approx(RYDBERG_ELECTRONVOLT_CODATA, abs=1.0e-8)
    assert BOHR_TO_ANGSTROM == pytest.approx(BOHR_RADIUS_ANGSTROM_CODATA, abs=1.0e-8)


def test_each_pair_inverts_exactly():
    assert RYDBERG_TO_ELECTRONVOLT * ELECTRONVOLT_TO_RYDBERG == pytest.approx(1.0, abs=1.0e-15)
    assert BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR == pytest.approx(1.0, abs=1.0e-15)
