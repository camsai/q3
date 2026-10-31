"""q3.io.results.base — the hand-written bases of the generated results models.

``q3.generated`` is generated from the JSON schemas (``schemas/manifest.yaml``,
``scripts/repo/generate_python_from_schemas.py``): the data is all there. What
a reader needs beyond the data -- reading a file, a record by its name, the
eigenvalues as arrays -- is written once here, and the manifest names the
block each base is for. The interpolation results file's own base is in
``q3.io.interpolation.results.base``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ResultsDocument(BaseModel):
    """A whole results file: read from a path, written back as it was read.

    A key the file carries is written back, nulls included (a ladder attempt's
    null band distance is data); a key it does not carry stays absent.
    """

    @classmethod
    def read(cls, path: str | Path):
        """The file at ``path``, whatever subset of the blocks it carries."""
        return cls.model_validate(json.loads(Path(path).read_text()))

    def to_document(self) -> dict:
        """The file's JSON: what was read or set, and nothing else."""
        return self.model_dump(mode="json", exclude_unset=True, by_alias=True)

    def write(self, path: str | Path) -> None:
        """Write the file as the engine formats it: four-space indents, a trailing newline."""
        Path(path).write_text(json.dumps(self.to_document(), indent=4) + "\n")


def _record_name(record: Any) -> str | None:
    """A record's ``name``, whether it arrives as JSON or as ESSE's model."""
    name = record.get("name") if isinstance(record, dict) else getattr(record, "name", None)
    return getattr(name, "value", name)


class RecordsBlock(BaseModel):
    """A block listing ESSE records under ``records``, each read by its name.

    ``properties.total_energy`` is the ``total_energy`` record as ESSE's model,
    or ``None`` when the run listed none.
    """

    def record(self, name: str):
        """The record named ``name``, as ESSE's model of it, or ``None`` when the run listed none."""
        records = getattr(self, "records", None) or []
        return next((record for record in records if _record_name(record) == name), None)

    @property
    def total_energy(self):
        """ESSE's ``total_energy`` record."""
        return self.record("total_energy")

    @property
    def total_energy_contributions(self):
        """ESSE's ``total_energy_contributions`` record."""
        return self.record("total_energy_contributions")

    @property
    def fermi_energy(self):
        """ESSE's ``fermi_energy`` record."""
        return self.record("fermi_energy")

    @property
    def atomic_forces(self):
        """ESSE's ``atomic_forces`` record: one vector per atom, by 1-based ``id``."""
        return self.record("atomic_forces")

    @property
    def stress_tensor(self):
        """ESSE's ``stress_tensor`` record: rows of the 3x3 tensor, each a ``.root`` list."""
        return self.record("stress_tensor")

    @property
    def pressure(self):
        """ESSE's ``pressure`` record."""
        return self.record("pressure")

    @property
    def band_gaps(self):
        """ESSE's ``band_gaps`` record: a ``direct`` and an ``indirect`` entry."""
        return self.record("band_gaps")

    @property
    def convergence_electronic(self):
        """ESSE's ``convergence_electronic`` record: the total energy at each SCF iteration."""
        return self.record("convergence_electronic")

    @property
    def eigenvalues_at_kpoints(self):
        """ESSE's ``eigenvalues_at_kpoints`` record: one entry per k-point state."""
        return self.record("eigenvalues_at_kpoints")

    @property
    def band_edge_valence_max(self):
        """ESSE's ``band_edge_valence_max`` record: the highest level at or below the Fermi energy."""
        return self.record("band_edge_valence_max")

    @property
    def band_edge_conduction_min(self):
        """ESSE's ``band_edge_conduction_min`` record: the lowest level above the Fermi energy."""
        return self.record("band_edge_conduction_min")

    @property
    def magnetization_total(self):
        """ESSE's ``magnetization_total`` record, for a spin-polarized run."""
        return self.record("magnetization_total")

    @property
    def magnetization_absolute(self):
        """ESSE's ``magnetization_absolute`` record, for a spin-polarized run."""
        return self.record("magnetization_absolute")


def eigenvalue_arrays(record):
    """``(eigenvalues, occupations, kpoint_weights)`` of an ``eigenvalues_at_kpoints`` record.

    The first two are indexed [k-point][spin][band]; ``occupations`` is
    ``None`` when the run recorded none.
    """
    eigenvalues = []
    occupations = []
    weights = []
    for kpoint in getattr(record, "values", None) or []:
        channels = kpoint.eigenvalues or []
        eigenvalues.append([list(channel.energies or []) for channel in channels])
        occupations.append([list(channel.occupations or []) for channel in channels])
        weights.append(kpoint.weight)
    has_occupations = any(channel for kpoint in occupations for channel in kpoint)
    return eigenvalues, (occupations if has_occupations else None), weights
