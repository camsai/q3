"""The campaign's fixed choices: datasets, cutoffs, the cells, the SCF, the seed and the variants.

Every number that decides what the roster is or how a row is run lives here, so
a row of the table traces to one place and a change to any of it is a visible
edit to this file.
"""

from __future__ import annotations

from pathlib import Path

#: This package is scripts/validation/wannier_campaign/, three levels under the root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIRECTORY = REPOSITORY_ROOT / "examples" / "data"

# ── the datasets and their cutoffs ───────────────────────────────────────────
#: The shipped file per family and element. Cutoffs come from the datasets' own
#: metadata.yml rather than being restated here, because two descriptions drift.
DATASET_FILES: dict[str, dict[str, str]] = {
    "nc": {
        "Si": "nc/Si.upf",
        "C": "nc/C.upf",
        "Cu": "nc/Cu.upf",
        "Ga": "nc/Ga.upf",
        "As": "nc/As.upf",
    },
    "paw": {"Si": "paw/Si.upf", "C": "paw/C.upf", "Cu": "paw/Cu.upf"},
    "us": {"Si": "us/Si_PBE_USPP.UPF", "Cu": "us/cu_pbe_v1.2.uspp.F.UPF"},
}
#: The families, in the order the roster lists them: every variant as
#: norm-conserving, then the pristine cells of each augmented family.
FAMILIES = tuple(DATASET_FILES)
NORM_CONSERVING_FAMILY = "nc"
AUGMENTED_FAMILIES = ("paw", "us")
#: Density over wavefunction cutoff: four where the density is |psi|^2, more where
#: an augmentation charge has structure the smooth part lacks (the examples' duals).
DENSITY_DUAL_BY_FAMILY = {"nc": 4.0, "paw": 8.0, "us": 5.0}
#: Where the campaign runs a different wavefunction cutoff from the metadata's
#: hint, and why: the shipped examples' values, so the rows compare with them.
CUTOFF_OVERRIDES_RY: dict[tuple[str, str], tuple[float, str]] = {
    ("nc", "Si"): (
        48.0,
        "the silicon examples and tutorial 08 run at 48 Ry, validated against QE there; "
        "PseudoDojo's stringent hint of 120 is a recommendation, not a requirement",
    ),
    ("paw", "Si"): (
        40.0,
        "40 Ry rather than the hint's 30: the PAW one-centre channel needs the denser "
        "grid to settle, see KNOWN-ISSUES on PAW convergence",
    ),
}
METHOD_SUBTYPE = {"nc": "norm-conserving", "paw": "paw", "us": "ultrasoft"}
COPPER_NOTE = (
    "copper's norm-conserving dataset is known to be wrong against published PBE; "
    "the augmented files carry the physical claim"
)

# ── the cells ────────────────────────────────────────────────────────────────
#: Cubic lattice constants in angstrom. Silicon's is the shipped PBE constant: the
#: experimental 5.431 puts |b| / 0.2 one hair past ten points.
LATTICE_CONSTANTS_ANGSTROM = {"Si": 5.4687, "GaAs": 5.653, "SiC": 4.358, "C": 3.567, "Cu": 3.615}
#: alpha-gallium: Cmca (number 64), the 8f site, in a primitive cell of four atoms.
GALLIUM_SPACE_GROUP = 64
GALLIUM_SITE = (0.0, 0.1549, 0.0810)
GALLIUM_CELL_PARAMETERS = (4.5197, 7.6633, 4.5260, 90, 90, 90)
ALLOY_MATERIAL = Path("materials") / "silicon_carbon_alloy.json"

# ── the variants ─────────────────────────────────────────────────────────────
RATTLE_FRACTION = 0.12  # of the shortest bond
STRAIN = 0.03
ROSTER_SEED = 20260908
PRISTINE, RATTLED, STRAINED_PLUS, STRAINED_MINUS, DOUBLED = (
    "",
    "--rattled",
    "--strained-plus",
    "--strained-minus",
    "--doubled",
)
#: Each variant by the suffix that names its case, with the words the roster lists it in.
VARIANT_LABELS = {
    PRISTINE: "pristine",
    RATTLED: f"rattled {RATTLE_FRACTION:.0%}",
    STRAINED_PLUS: f"strained {STRAIN:+.0%}",
    STRAINED_MINUS: f"strained {-STRAIN:+.0%}",
    DOUBLED: "doubled along a",
}
#: Which variants a base structure gets; the alloy is already rattled and
#: sheared, and gallium's four-atom cell gets the rattle only.
ALL_VARIANTS = tuple(VARIANT_LABELS)
VARIANTS_BY_FORMULA = {
    "Si3C": (PRISTINE, STRAINED_PLUS, STRAINED_MINUS),
    "Ga": (PRISTINE, RATTLED),
}

# ── the SCF every case starts from ───────────────────────────────────────────
#: The functional every row is run with. Every dataset in DATASET_FILES states it
#: too, but the manifest names it rather than leaving it to the datasets, so the
#: table stays a PBE table whatever a dataset added to it later says.
EXCHANGE_CORRELATION_FUNCTIONAL = "pbe"
SMEARED = {"Cu", "Ga", "As"}  # metals and semimetals
SMEARING_TYPE = "gaussian"
SMEARING_RYDBERG = 0.02
SCF_KPOINT_SPACING_INVERSE_ANGSTROM = 0.3  # the capture's rule, at a coarser spacing
SCF_CONVERGENCE_THRESHOLD_RY = 1.0e-10
SCF_MAXIMUM_ITERATIONS = 200
SCF_MIXING_BETA = 0.3
#: Where a case's run lands under its own directory, and the prefix of its files.
OUTPUT_DIRECTORY = "output"
RUN_PREFIX = "campaign"

# ── the run and its table ────────────────────────────────────────────────────
#: Every threshold pair re-runs the localisation on the full mesh, minutes per
#: structure, so a campaign walks six of the thirty pairs; None walks them all.
DEFAULT_THRESHOLD_ATTEMPTS = 6
#: The published regime the table is read against: 96.5 % of 200 materials within
#: this band distance for projectability disentanglement.
PUBLISHED_ETA_MEV = 20.0
#: How much of a failure that names no ``Error:`` line a row keeps: its end.
FAILURE_TAIL_CHARACTERS = 500
