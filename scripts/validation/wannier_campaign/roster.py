"""The roster: every case, built once from the shipped datasets, and how a run selects from it."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from ase import Atoms
from q3.structure.ase_atoms import ase_atoms_to_material_json

from .case import CampaignCase
from .settings import (
    ALL_VARIANTS,
    AUGMENTED_FAMILIES,
    COPPER_NOTE,
    CUTOFF_OVERRIDES_RY,
    DATASET_FILES,
    DEFAULT_DATA_DIRECTORY,
    DENSITY_DUAL_BY_FAMILY,
    FAMILIES,
    NORM_CONSERVING_FAMILY,
    PRISTINE,
    ROSTER_SEED,
    SMEARED,
    VARIANT_LABELS,
    VARIANTS_BY_FORMULA,
)
from .structures import base_structures, variants

#: How far apart the seeds of two base structures sit, so their variants never share one.
SEED_STRIDE = 100


@cache
def cutoffs(data_directory: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """(wavefunction, density) per (family, element): the hint or its override, times the dual."""
    metadata_path = Path(data_directory) / "pseudopotentials" / "metadata.yml"
    metadata = yaml.safe_load(metadata_path.read_text())
    table = {}
    for family, files in DATASET_FILES.items():
        described = (metadata.get(family) or {}).get("files") or {}
        for element, relative in files.items():
            override = CUTOFF_OVERRIDES_RY.get((family, element))
            hint = (described.get(relative.split("/", 1)[1]) or {}).get("cutoff_hint_ry")
            wavefunction = override[0] if override else hint
            if wavefunction is None:
                raise ValueError(
                    f"{metadata_path} has no cutoff_hint_ry for {relative}; add one there, or "
                    "record an override in CUTOFF_OVERRIDES_RY with the reason"
                )
            table[(family, element)] = (
                float(wavefunction),
                float(wavefunction) * DENSITY_DUAL_BY_FAMILY[family],
            )
    return table


def build_case(
    name: str,
    family: str,
    formula: str,
    variant: str,
    atoms: Atoms,
    data_directory: Path,
    note: str = "",
) -> CampaignCase:
    """One roster entry: the structure, its datasets, and the largest cutoffs any species asks for."""
    species = sorted(set(atoms.get_chemical_symbols()))
    limits = [cutoffs(data_directory)[(family, symbol)] for symbol in species]
    return CampaignCase(
        name=name,
        family=family,
        formula=formula,
        variant=variant,
        number_of_atoms=len(atoms),
        material=ase_atoms_to_material_json(atoms, {"name": name}),
        pseudopotentials={
            symbol: str(data_directory / "pseudopotentials" / DATASET_FILES[family][symbol])
            for symbol in species
        },
        cutoffs=(
            max(wavefunction for wavefunction, _ in limits),
            max(density for _, density in limits),
        ),
        smeared=bool(SMEARED & set(species)),
        note=note,
    )


def campaign_roster(
    data_directory: str | Path = DEFAULT_DATA_DIRECTORY,
    families: tuple[str, ...] = FAMILIES,
    seed: int = ROSTER_SEED,
) -> list[CampaignCase]:
    """The fixed roster: the base structures with their variants as norm-conserving, then
    the pristine elementals of each augmented family that has a dataset for them."""
    data_directory = Path(data_directory)
    bases = base_structures(data_directory)
    cases = []
    if NORM_CONSERVING_FAMILY in families:
        for base in bases:
            wanted = VARIANTS_BY_FORMULA.get(base.formula, ALL_VARIANTS)
            structures = variants(base.atoms, seed + SEED_STRIDE * base.seed_offset)
            cases += [
                build_case(
                    base.key + suffix,
                    NORM_CONSERVING_FAMILY,
                    base.formula,
                    VARIANT_LABELS[suffix],
                    structures[suffix],
                    data_directory,
                )
                for suffix in ALL_VARIANTS
                if suffix in wanted
            ]
    for family in AUGMENTED_FAMILIES:
        for base in bases if family in families else []:
            if base.formula in DATASET_FILES[family]:
                cases.append(
                    build_case(
                        f"{base.key}--{family}",
                        family,
                        base.formula,
                        VARIANT_LABELS[PRISTINE],
                        base.atoms,
                        data_directory,
                        COPPER_NOTE if base.formula == "Cu" else "",
                    )
                )
    return cases


def select_cases(
    cases: list[CampaignCase], selector: str, *, exact: bool = False
) -> list[CampaignCase]:
    """The cases ``--only`` names -- full names, or families with their ``<family>--*``
    variants -- comma-separated, in roster order.

    A family stops at the ``--`` boundary, since one family's name sits inside
    another's (``gallium`` in ``gallium_arsenide``); ``exact`` takes the name alone,
    which is how a family's pristine member, named after the family, is reached.
    """
    wanted = [part.strip() for part in selector.split(",") if part.strip()]
    if not wanted:
        raise ValueError("--only was given nothing to select")
    chosen: set[str] = set()
    for part in wanted:
        matched = {
            case.name
            for case in cases
            if case.name == part or (not exact and case.name.startswith(part + "--"))
        }
        if not matched:
            families = sorted({case.name.split("--")[0] for case in cases})
            raise ValueError(
                f"--only {part!r} matched no case; give a structure name, or a family "
                f"(one of {families}){'' if exact else ', or add --exact for that name alone'}"
            )
        chosen |= matched
    return [case for case in cases if case.name in chosen]
