"""q3.io.pseudo_atomic_shells.pseudo_atomic_shell — :class:`PseudoAtomicShell`, one ``PP_CHI`` shell."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PseudoAtomicShell(BaseModel):
    """One ``PP_CHI`` shell of a dataset, as the file states it.

    ``index`` is the shell's 0-based position in the file, which is what
    ``control.interpolation.trial_shells`` names. ``label`` is the file's own
    name for it ("3S"), where it gives one. ``principal_quantum_number`` is
    None when neither the labels nor the element and its valence charge
    determine it. ``zero_norm`` is a radial function that integrates to nothing,
    which a dataset can list for a channel it has no wavefunction for (silicon
    ultrasoft's 4s).
    """

    model_config = ConfigDict(frozen=True)

    index: int
    angular_momentum: int
    occupation: float
    label: str | None = None
    principal_quantum_number: int | None = None
    zero_norm: bool = False

    @property
    def orbitals(self) -> int:
        """2l + 1: the orbitals the shell becomes on each atom."""
        return 2 * self.angular_momentum + 1

    @property
    def occupied(self) -> bool:
        return self.occupation > 0.0

    @property
    def usable(self) -> bool:
        """Whether the engine will build orbitals from it at all.

        A negative occupation is the UPF convention for a shell not to be used,
        and a zero-norm radial function cannot be one; the engine skips both.
        """
        return self.occupation >= 0.0 and not self.zero_norm
