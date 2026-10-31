"""q3.io.results.classes — a run's ``<prefix>_results.json``, typed.

The models are generated from ``schemas/q3/results.json`` into
``q3.generated.results``, one module per schema file (``scripts/repo/generate_python_from_schemas.py``),
and every ESSE record and quantity is ESSE's own model
(``mat3ra.esse.models``), so nothing here re-describes one. A value is read
straight off its record::

    results = Results.read(path)
    results.properties.total_energy.value  # Ry
    results.properties.total_energy.units.value  # "Ry"
    eigenvalues, occupations, weights = eigenvalue_arrays(results.properties.eigenvalues_at_kpoints)

Units are the record's own: the engine writes energies in Ry, forces in
Ry/a.u., stress and pressure in kbar and magnetization in uB, and each record
says so in its ``units``. Nothing here converts. Every field is optional and
unknown keys are kept: a reader takes a subset or a newer file as it comes.
"""

from q3.generated.results import Results
from q3.generated.results.properties import HubbardOccupation, Properties
from q3.generated.results.properties.energy_contributions import EnergyContributions
from q3.generated.results.properties.isolated_system_correction import IsolatedSystemCorrection
from q3.generated.results.properties.orbital_projections import Label as OrbitalProjectionLabel
from q3.generated.results.properties.orbital_projections import OrbitalProjections
from q3.generated.results.runtime_log import RuntimeLog
from q3.generated.results.runtime_log.execution import Execution
from q3.generated.results.runtime_log.incomplete import Incomplete
from q3.generated.results.structure import Structure
from q3.io.results.base import eigenvalue_arrays

__all__ = [
    "RuntimeLog",
    "EnergyContributions",
    "Execution",
    "HubbardOccupation",
    "Incomplete",
    "IsolatedSystemCorrection",
    "OrbitalProjectionLabel",
    "OrbitalProjections",
    "Properties",
    "Results",
    "Structure",
    "eigenvalue_arrays",
]
