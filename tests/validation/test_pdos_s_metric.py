"""Löwdin charge conservation for ultrasoft and PAW PDOS projections.

Ultrasoft and PAW pseudopotentials give up norm conservation, so the
Kohn-Sham states solve ``H|psi> = eps S|psi>`` with a non-trivial overlap
``S = 1 + sum_ij q_ij |beta_i><beta_j|`` and are S-orthonormal rather than
orthonormal (Vanderbilt, PRB 41, 7892 (1990); Blöchl, PRB 50, 17953 (1994);
Kresse & Joubert, PRB 59, 1758 (1999)). The projection that conserves charge
is therefore ``<phi'_i|S|psi_n>``, with the atomic orbitals Löwdin-
orthogonalized in the same metric.

The observable consequence, and what these tests check, is the Löwdin charge
sum. ``sum_i charge_i = sum_nk wg(n,k) sum_i |<phi'_i|S|psi_nk>|^2`` is
bounded above by ``sum_nk wg(n,k)``, the valence electron count, and reaches
it in the limit of a complete atomic basis -- the shortfall is the spilling
parameter, a basis-set deficiency of a few percent. Projected in the plain
metric instead, the sum is not bounded by that count at all and misses it by
far more than any basis-set deficiency explains, which is the failure these
cases pin down.

Nothing here needs a reference code: the sum rule is Q3's own. The
cross-code gate for the same projections, against QE's ``projwfc.x``, lives
in ``test_qe_projwfc_comparison.py`` and self-skips without the reference
build.

Run:
    .venv/bin/pytest tests/validation/test_pdos_s_metric.py -v \
        --q3-bin ./build/q3
"""

import shutil
from pathlib import Path

import pytest
import yaml
from q3.workflows import run_pdos_workflow

from helpers import EXAMPLES_DIRECTORY

# Silicon: two atoms, both datasets carrying Z_valence = 4.
SILICON_VALENCE_ELECTRONS = 8.0

# The Fermi bisection's own electron count must land on the dataset's valence
# charge essentially exactly -- this is not the physics under test, it is the
# denominator the charge sum is measured against.
ELECTRON_COUNT_TOLERANCE = 1.0e-6

# How far below the valence electron count the Löwdin charges may fall. This
# is the spilling parameter: atomic orbitals are an incomplete basis, so some
# valence charge is genuinely unrepresentable and the sum is always slightly
# short. Both silicon cases measure well under this; projected in the plain
# metric the same cases miss by an order of magnitude more.
MAXIMUM_SPILLING = 0.05

# The sum can only exceed the electron count through numerical noise, never
# through physics: each band's total projection is bounded by 1.
CHARGE_SUM_OVERSHOOT_TOLERANCE = 1.0e-6

PDOS_KGRID_DIMENSIONS = (4, 4, 4)


def copy_example_with_absolute_paths(example_yaml: Path, destination_directory: Path) -> Path:
    """Copy *example_yaml* into *destination_directory*, resolving its paths.

    ``write_nscf_input`` copies ``material:`` and ``pseudopotentials:``
    verbatim from the SCF template, and those are relative to the template's
    own directory -- so a companion NSCF input written anywhere else cannot
    resolve them. Making them absolute up front lets both inputs, and every
    file the run produces, live under the test's temporary directory instead
    of beside the committed example.
    """
    with open(example_yaml) as file_handle:
        configuration = yaml.safe_load(file_handle)

    configuration["material"] = str((example_yaml.parent / configuration["material"]).resolve())
    pseudopotentials = configuration["method"]["pseudopotentials"]
    for species, pseudopotential_path in pseudopotentials.items():
        pseudopotentials[species] = str((example_yaml.parent / pseudopotential_path).resolve())
    configuration.setdefault("control", {})["outdir"] = str(destination_directory / "output")

    destination = destination_directory / example_yaml.name
    with open(destination, "w") as file_handle:
        yaml.dump(configuration, file_handle, default_flow_style=False)
    return destination


@pytest.mark.parametrize(
    "example_name, subtype",
    [
        ("us.yml", "ultrasoft"),
        ("paw.yml", "PAW"),
    ],
)
def test_lowdin_charges_sum_to_the_valence_electron_count(
    q3_command, tmp_path, example_name, subtype
):
    """The S-metric Löwdin charges account for every valence electron but the spilling."""
    example_yaml = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / example_name
    scf_input = copy_example_with_absolute_paths(example_yaml, tmp_path)

    result = run_pdos_workflow(
        scf_input,
        kgrid_dimensions=PDOS_KGRID_DIMENSIONS,
        kgrid_shifts=(0, 0, 0),
        q3_binary=q3_command[0],
        nscf_input_path=tmp_path / "pdos_nscf.yml",
    )
    charges = result["lowdin_charges"]

    number_of_electrons = charges["number_of_electrons"]
    total_charge = charges["total_charge"]
    spilling = charges["spilling_parameter"]
    print(
        f"\n  Si {subtype}: electrons={number_of_electrons:.6f} "
        f"Lowdin charge sum={total_charge:.6f} spilling={spilling:.6f}"
    )

    assert number_of_electrons == pytest.approx(
        SILICON_VALENCE_ELECTRONS, abs=ELECTRON_COUNT_TOLERANCE
    ), (
        f"Si {subtype}: the run's own occupations integrate to {number_of_electrons} electrons, "
        f"not the dataset's {SILICON_VALENCE_ELECTRONS} valence electrons -- the charge sum below "
        "would be measured against the wrong number"
    )
    assert total_charge <= number_of_electrons + CHARGE_SUM_OVERSHOOT_TOLERANCE, (
        f"Si {subtype}: Löwdin charges sum to {total_charge:.6f}, above the "
        f"{number_of_electrons:.6f} valence electrons; each band's projections are bounded by 1 "
        "in the S metric, so a sum over the count means the metric is wrong"
    )
    assert total_charge >= number_of_electrons * (1.0 - MAXIMUM_SPILLING), (
        f"Si {subtype}: Löwdin charges sum to {total_charge:.6f} of {number_of_electrons:.6f} "
        f"valence electrons (spilling {spilling:.4f}, limit {MAXIMUM_SPILLING}) -- more charge is "
        "missing than an incomplete atomic basis accounts for"
    )

    shutil.rmtree(tmp_path / "output", ignore_errors=True)
