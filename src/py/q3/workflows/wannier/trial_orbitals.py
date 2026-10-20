"""q3.workflows.wannier.trial_orbitals — :class:`TrialOrbitals`, what the datasets decide."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from q3.io.input_reader import load_material, pseudopotential_paths
from q3.io.pseudo_atomic_shells import pseudo_atomic_shells
from q3.workflows.wannier.trial_shells import number_of_trial_orbitals, trial_shells
from q3.workflows.wannier.utils import band_window


class TrialOrbitals(BaseModel):
    """The trial orbitals of one material: the shells of each species, and the count over its atoms.

    The protocol takes its trial orbitals, and with them the number of Wannier
    functions and the band window, from the pseudopotentials themselves.
    ``shells_by_species`` is what the capture manifest carries as
    ``control.interpolation.trial_shells``; ``number_of_wannier_functions`` is
    one dataset's worth of orbitals per atom. Each dataset is read once.
    """

    model_config = ConfigDict(frozen=True)

    shells_by_species: dict[str, list[int]]
    number_of_wannier_functions: int

    @classmethod
    def of(cls, material: dict, datasets: dict[str, Path]) -> Self:
        """For *material*, whose species' datasets are *datasets*."""
        shells_by_species: dict[str, list[int]] = {}
        orbitals_per_atom: dict[str, int] = {}
        for symbol, path in datasets.items():
            shells = pseudo_atomic_shells(path)
            kept = trial_shells(shells)
            if not kept:
                raise ValueError(
                    f"{path} leaves no shell to build trial orbitals from: every PP_CHI shell "
                    "carries a negative occupation or a radial function of zero norm"
                )
            shells_by_species[symbol] = kept
            orbitals_per_atom[symbol] = number_of_trial_orbitals(shells)
        return cls(
            shells_by_species=shells_by_species,
            number_of_wannier_functions=sum(
                orbitals_per_atom[element["value"]] for element in material["basis"]["elements"]
            ),
        )

    @classmethod
    def of_manifest(cls, manifest: dict, base_directory: str | Path) -> Self:
        """For the material and the datasets *manifest* names, resolved against *base_directory*."""
        material = load_material(manifest, base_directory)
        return cls.of(material, pseudopotential_paths(manifest, material, base_directory))

    @property
    def band_window(self) -> int:
        """How many bands the capture keeps: room above the manifold for the disentangler."""
        return band_window(self.number_of_wannier_functions)
