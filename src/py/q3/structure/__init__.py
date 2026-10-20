"""q3.structure — Crystal-structure adapters between Q3 and third-party formats.

Mirrors the C++ ``layer_1_structure`` in scope: it owns structure
representation and conversion, and knows nothing about calculators, flavors,
or workflows.  Anything that needs to turn a Q3 material file into another
library's structure object imports from here, downward.
"""

from q3.structure.ase_atoms import material_json_to_ase_atoms

__all__ = ["material_json_to_ase_atoms"]
