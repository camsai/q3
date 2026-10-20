"""q3.structure.strain — homogeneous strain of a cell, in the cell's own Cartesian frame.

**Upstream candidate (mat3ra-made).**  This is the piece of Q3 that mat3ra-made
should own, and holds only what mat3ra-made cannot do for Q3 today.  made has
the operation -- ``create_strain(material, strain_matrix=..., scale_factor=...)``
and ``get_isotropic_strain_matrix`` in
``mat3ra.made.tools.build_components.operations.core.modifications.strain`` --
but not in a form Q3 can call:

* It lives under ``mat3ra.made.tools``, which imports pymatgen at module scope
  and so needs made's ``tools`` extra (``pymatgen==2024.4.13``,
  ``numpy<=1.26.4``): a numpy-1 pin Q3's dependency floor exists to avoid.
* It applies the strain to made's ``Lattice``, which keeps only a, b, c and the
  angles and rebuilds the vectors in a frame of its own (see
  ``q3.structure.lattice``).  For silicon's FCC cell that frame does not have the
  cubic axes along x, y and z, so a strain written along Q3's Cartesian axes
  lands on different crystal directions: a yz shear comes back as a mixed
  deformation that changes all three cell angles.  A volume scan does not care;
  an elastic constant does.

Everything here keeps fractional coordinates fixed, which is made's convention
for ``strain`` too.  Once made offers a frame-preserving strain without the
``tools`` extra, these functions become calls to it.
"""

from __future__ import annotations

import numpy as np
from ase.stress import voigt_6_to_full_3x3_strain


def isotropic_strain_matrix(linear_scale_factor: float) -> np.ndarray:
    """The deformation that multiplies every lattice vector by *linear_scale_factor*.

    The volume scales as its cube.  made's ``get_isotropic_strain_matrix``.
    """
    if linear_scale_factor <= 0.0:
        raise ValueError(f"linear_scale_factor must be positive, got {linear_scale_factor}")
    return linear_scale_factor * np.eye(3)


def strain_matrix_from_voigt(voigt_strain) -> np.ndarray:
    """The deformation ``I + eps`` of a six-component Voigt strain.

    Shear components are engineering strains (eps_4 = 2 eps_yz), as in ASE,
    whose ``voigt_6_to_full_3x3_strain`` this is.
    """
    return np.asarray(voigt_6_to_full_3x3_strain(np.asarray(voigt_strain, dtype=float)))


def strain_atoms(atoms, reference_cell, strain_matrix) -> None:
    """Give *atoms* the cell ``reference_cell @ strain_matrix``, fractional coordinates fixed.

    Rows of the cell are lattice vectors, so for a symmetric strain matrix this
    is h' = h (I + eps) in the frame *reference_cell* is written in -- made's
    ``strain`` convention, without the change of frame.  *atoms* is an ASE
    ``Atoms``; *reference_cell* is kept separate so that a scan of many strains
    always starts from the same cell rather than compounding them.
    """
    atoms.set_cell(np.asarray(reference_cell) @ np.asarray(strain_matrix), scale_atoms=True)
