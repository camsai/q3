"""
End-to-end validation of Methfessel-Paxton smearing of order N > 1.

Quantum ESPRESSO's pw.x exposes only order 1, so there is no external reference
to compare higher orders against. What can be checked end to end on a real
metal (NC Cu, PBE, the shipped nc_mp.yml example at a reduced cutoff):

- `order: 1` stated explicitly is bit-identical to the default, so existing
  inputs are untouched by the new key.
- Order 2 converges and changes the result, but only by a small amount: both
  orders approximate the same σ → 0 total energy, which bounds their difference
  at fixed σ well below the smearing width itself.
- Higher order weakens the σ dependence of the total energy (MP89: the error
  falls as O(σ^(2N+2))), so E(2σ) − E(σ) shrinks going from order 1 to 2.
"""

import pytest

from helpers import EXAMPLES_DIRECTORY, run_overlay_and_read_energy

COPPER_MP_INPUT = EXAMPLES_DIRECTORY / "total_energy--metal--copper" / "nc_mp.yml"
PREFIX = "cu_nc_mp"

# Reduced cutoff: the checks compare runs with each other, not with an
# external reference, so a cheap basis keeps the module to a few short SCF runs. The
# k-grid stays dense: at 6x6x6 sampling noise hides the order-1 vs order-2 width
# dependence (2.6e-4 vs 2.2e-4 Ry), at 10x10x10 it is clear (9.7e-5 vs 2.6e-5 Ry).
BASE_OVERLAY = {
    "method": {
        "cutoffs": {"wavefunction": 40.0, "density": 160.0},
        "scf": {"kgrid": {"dimensions": [10, 10, 10]}},
    }
}

# |E(order 2) − E(order 1)| at degauss 0.02 Ry; both converge to the same σ → 0 limit.
ORDER_DIFFERENCE_BOUND_RYDBERG = 2.0e-3


def _smearing_overlay(degauss, order=None):
    smearing = {"type": "methfessel-paxton", "degauss": degauss}
    if order is not None:
        smearing["order"] = order
    overlay = {"method": {"scf": {"smearing": smearing}}}
    merged = {"method": {**BASE_OVERLAY["method"]}}
    merged["method"]["scf"] = {**BASE_OVERLAY["method"]["scf"], **overlay["method"]["scf"]}
    return merged


def _energy(q3_command, tmp_path, label, degauss, order=None):
    run_directory = tmp_path / label
    run_directory.mkdir()
    return run_overlay_and_read_energy(
        q3_command, COPPER_MP_INPUT, _smearing_overlay(degauss, order), PREFIX, run_directory
    )


@pytest.fixture(scope="module")
def energies(q3_command, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("methfessel_paxton_order")
    return {
        "default_0.02": _energy(q3_command, tmp_path, "default", 0.02),
        "order1_0.02": _energy(q3_command, tmp_path, "order1", 0.02, 1),
        "order1_0.04": _energy(q3_command, tmp_path, "order1_wide", 0.04, 1),
        "order2_0.02": _energy(q3_command, tmp_path, "order2", 0.02, 2),
        "order2_0.04": _energy(q3_command, tmp_path, "order2_wide", 0.04, 2),
    }


def test_explicit_order_one_is_bit_identical_to_default(energies):
    assert energies["order1_0.02"] == energies["default_0.02"]


def test_order_two_changes_the_energy_within_the_smearing_bound(energies):
    difference = abs(energies["order2_0.02"] - energies["order1_0.02"])
    assert 0.0 < difference < ORDER_DIFFERENCE_BOUND_RYDBERG, (
        f"|E(N=2) - E(N=1)| = {difference:.3e} Ry at degauss 0.02 Ry"
    )


def test_order_two_weakens_the_smearing_width_dependence(energies):
    order_one_spread = abs(energies["order1_0.04"] - energies["order1_0.02"])
    order_two_spread = abs(energies["order2_0.04"] - energies["order2_0.02"])
    assert order_two_spread < order_one_spread, (
        f"E(0.04) - E(0.02): order 1 {order_one_spread:.3e} Ry, order 2 {order_two_spread:.3e} Ry"
    )
