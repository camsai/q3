"""q3.io.pseudo_atomic_shells — the ``PP_CHI`` shells a pseudopotential dataset lists.

``pseudo_atomic_shell`` is one shell as the file states it, ``upf_reader``
reads every shell of a UPF dataset in either version of the format, and
``principal_quantum_numbers`` counts a shell's n up from the element's core
where the file's labels do not say it. What is done with the shells -- which of
them become trial orbitals -- is the Wannier protocol's
(:mod:`q3.workflows.wannier.trial_shells`), not the reader's.
"""

from q3.io.pseudo_atomic_shells.pseudo_atomic_shell import PseudoAtomicShell
from q3.io.pseudo_atomic_shells.upf_reader import ZERO_NORM_THRESHOLD, pseudo_atomic_shells

__all__ = ["PseudoAtomicShell", "ZERO_NORM_THRESHOLD", "pseudo_atomic_shells"]
