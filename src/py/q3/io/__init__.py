"""q3.io — Input/output utilities for Q3.

Nothing here re-exports a module that reads results.json (``q3.io.results``,
``q3.io.interpolation.results``, ``q3.io.output_writer``): the generated results
models derive from bases in this package, so importing one of those here would
import the generated models while they are still being built. Import them from
their own modules.
"""

from q3.io.input_reader import (
    load_material,
    make_file_references_absolute,
    pseudopotential_paths,
)
from q3.io.input_writer import (
    ensure_calculation_type,
    write_material_json,
    write_nscf_input,
    write_scf_input,
    write_supercell_scf_input,
)
from q3.io.interpolation_settings import CaptureSettings, EvaluationSettings
from q3.io.pseudo_atomic_shells import PseudoAtomicShell, pseudo_atomic_shells

__all__ = [
    "CaptureSettings",
    "EvaluationSettings",
    "PseudoAtomicShell",
    "ensure_calculation_type",
    "load_material",
    "make_file_references_absolute",
    "pseudo_atomic_shells",
    "pseudopotential_paths",
    "write_material_json",
    "write_nscf_input",
    "write_scf_input",
    "write_supercell_scf_input",
]
