"""The campaign's finding, as one Markdown table with its caveat in the same breath."""

from __future__ import annotations

from .settings import DEFAULT_THRESHOLD_ATTEMPTS, PUBLISHED_ETA_MEV
from .utils import table_cell

#: The table's columns: header, Markdown alignment, and the cell from a row.
COLUMNS = (
    ("structure", "---", lambda row: row["name"]),
    ("family", "---", lambda row: row["family"]),
    ("atoms", "---:", lambda row: str(row["atoms"])),
    ("mesh", "---", lambda row: row.get("grid", "—")),
    ("N_w", "---:", lambda row: table_cell(row.get("wannier_functions"))),
    ("bands", "---:", lambda row: table_cell(row.get("bands"))),
    ("rung", "---", lambda row: row.get("rung") or "—"),
    ("tries", "---:", lambda row: str(row.get("attempts", 0))),
    ("η₀ (meV)", "---:", lambda row: table_cell(row.get("eta_first_mev"), 4)),
    ("η (meV)", "---:", lambda row: table_cell(row.get("eta_mev"), 4)),
    ("η_max (meV)", "---:", lambda row: table_cell(row.get("eta_max_mev"), 4)),
    (
        "spread var. (bohr²)",
        "---:",
        lambda row: table_cell(row.get("spread_variance_bohr_squared")),
    ),
    ("decay", "---:", lambda row: table_cell(row.get("decay_ratio"))),
    ("status", "---", lambda row: row["status"]),
    ("s", "---:", lambda row: table_cell(row["seconds"], 4)),
)


def campaign_table(
    rows: list[dict], spacing: float, threshold_attempts: int | None = DEFAULT_THRESHOLD_ATTEMPTS
) -> str:
    """One Markdown table, with the caveat in the same breath as the numbers."""
    statuses = [row["status"] for row in rows]
    first_try = sum(row["status"] == "accept" and row.get("rung") == "default" for row in rows)
    in_the_published_regime = sum(
        row.get("eta_first_mev") is not None and row["eta_first_mev"] <= PUBLISHED_ETA_MEV
        for row in rows
    )
    walk = (
        "the full thirty-pair threshold walk"
        if threshold_attempts is None
        else f"the threshold walk capped at {threshold_attempts} pairs"
    )
    lines = [
        f"## Wannier campaign — {len(rows)} structures from eight elements, spacing {spacing} Å⁻¹",
        "",
        f"Accepted {statuses.count('accept')} of {len(rows)} ({first_try} without a retry), "
        f"rejected {statuses.count('reject')}, errors {statuses.count('error')}; "
        f"{in_the_published_regime} of {len(rows)} within {PUBLISHED_ETA_MEV:g} meV on the "
        "first rung. "
        f"Run with {walk} and no mesh refinement. "
        "Eight elements cannot reproduce a two-hundred-material distribution; the table "
        "says whether the machinery is in the published regime (96.5 % of 200 materials "
        f"within {PUBLISHED_ETA_MEV:g} meV for projectability disentanglement) on the chemistry "
        "Q3 ships "
        "datasets for. η₀ is the band distance of the protocol's defaults before any retry; "
        "η is the accepted attempt's, or the last one's. Magnetic elements are excluded "
        "until a spin-polarized capture is confirmed. Copper's norm-conserving dataset is a "
        "consistency case only.",
        "",
        "| " + " | ".join(header for header, _, _ in COLUMNS) + " |",
        "| " + " | ".join(align for _, align, _ in COLUMNS) + " |",
        *("| " + " | ".join(column(row) for _, _, column in COLUMNS) + " |" for row in rows),
    ]
    failures = [row for row in rows if row["status"] != "accept"]
    if failures:
        lines += ["", "Reasons for every row that did not accept:", ""]
        lines += [f"- `{row['name']}`: {row.get('reason', '')}" for row in failures]
    return "\n".join(lines) + "\n"
