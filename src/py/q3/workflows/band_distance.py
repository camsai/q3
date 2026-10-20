"""q3.workflows.band_distance — the band distance eta between two band structures.

The quality gate of the automated Wannier protocol (Vitale, Pizzi, Marrazzo,
Yates, Marzari, Mostofi, npj Comput. Mater. 6, 66 (2020)): a Fermi-Dirac
weighted root-mean-square difference between a model's bands and the direct
calculation's, over the states an interpolation is expected to reproduce, with
the worst weighted deviation beside it. Shared by the workflow that accepts or
rejects a model on it and by the validation suite that checks the workflow, so
that there is one definition rather than two that drift.

All energies in electronvolt. With mu = ``fermi_energy + energy_offset`` and
sigma = ``smearing``,

    f~_nk  = sqrt( f(e^ref_nk; mu, sigma) * f(e^cmp_nk; mu, sigma) )
    eta    = sqrt( sum f~ (e^ref - e^cmp)^2 / sum f~ )
    eta^max = max f~ |e^ref - e^cmp|

The published defaults are an offset of 2 eV and a smearing of 0.1 eV, and
the Fermi level of an insulator is its valence-band maximum -- which is what
Q3's SCF reports for a system with integer occupations, so the SCF's value
serves both cases.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from q3.constants import RYDBERG_TO_ELECTRONVOLT
from q3.io.interpolation_results import BandDistanceReport, InterpolationResults

#: The published window above the Fermi level, in eV, and the smearing of its edge.
DEFAULT_ENERGY_OFFSET_EV = 2.0
DEFAULT_SMEARING_EV = 0.1
#: The largest exponent the weight is evaluated at: past it the occupation is
#: 0 or 1 to every digit, and `np.exp` overflows.
LARGEST_EXPONENT = 500.0


def _fermi_dirac(energies: np.ndarray, chemical_potential: float, smearing: float) -> np.ndarray:
    """Occupations at `smearing`, used as the band distance's weight, not as physics.

    The exponent is clipped because the metric evaluates far into both tails,
    where the true occupation is 0 or 1 to every digit but `np.exp` overflows
    and would put an infinity into the sum.
    """
    exponent = np.clip(
        (energies - chemical_potential) / smearing, -LARGEST_EXPONENT, LARGEST_EXPONENT
    )
    return 1.0 / (1.0 + np.exp(exponent))


class BandDistance(BaseModel):
    """Result of :func:`band_distance`: both the RMS and worst-case measures."""

    model_config = ConfigDict(frozen=True)

    eta: float  # eV
    eta_maximum: float  # eV
    number_of_states: int

    def as_dict(self) -> dict:
        """The distance as the workflow returns it, units carried in the key names."""
        return {
            "eta_ev": self.eta,
            "eta_maximum_ev": self.eta_maximum,
            "number_of_states": self.number_of_states,
        }

    def to_report(self, reference: str, number_of_kpoints: int) -> BandDistanceReport:
        """The distance as the results file's ``quality.band_distance`` block.

        ``reference`` names the k-points it was measured on ("path" or
        "half-shifted mesh") and ``number_of_kpoints`` how many there were.
        """
        return BandDistanceReport(
            **self.as_dict(), reference=reference, number_of_kpoints=number_of_kpoints
        )


def band_distance(
    reference_eigenvalues: np.ndarray,
    comparison_eigenvalues: np.ndarray,
    fermi_energy: float,
    energy_offset: float = DEFAULT_ENERGY_OFFSET_EV,
    smearing: float = DEFAULT_SMEARING_EV,
) -> BandDistance:
    """Fermi-Dirac-weighted band distance eta, following Vitale et al. (2020).

    All energies in eV; the formula is in the module docstring. The weight
    suppresses states well above mu, where an interpolation is not expected
    (or required) to reproduce the DFT bands. Bands are compared **after
    sorting each k-point's eigenvalues in ascending energy**, over the lowest
    ``min(num_bands)`` of the two inputs: the two chains being compared need
    not label bands identically, only span the same manifold.

    eta^max is the worst weighted deviation: a *weighted* maximum, so a large
    deviation far above mu is damped rather than dominating.
    """
    reference = np.asarray(reference_eigenvalues, dtype=float)
    comparison = np.asarray(comparison_eigenvalues, dtype=float)
    if reference.ndim != 2 or comparison.ndim != 2:
        raise ValueError("eigenvalue arrays must be 2-D, (num_kpts, num_bands)")
    if reference.shape[0] != comparison.shape[0]:
        raise ValueError(f"k-point counts differ: {reference.shape[0]} vs {comparison.shape[0]}")
    if smearing <= 0.0:
        raise ValueError("smearing must be positive")

    band_count = min(reference.shape[1], comparison.shape[1])
    reference_sorted = np.sort(reference, axis=1)[:, :band_count]
    comparison_sorted = np.sort(comparison, axis=1)[:, :band_count]

    chemical_potential = fermi_energy + energy_offset
    weight = np.sqrt(
        _fermi_dirac(reference_sorted, chemical_potential, smearing)
        * _fermi_dirac(comparison_sorted, chemical_potential, smearing)
    )
    difference = reference_sorted - comparison_sorted
    total_weight = float(weight.sum())
    if total_weight <= 0.0:
        raise ValueError(
            "every state carries zero Fermi-Dirac weight; check fermi_energy/energy_offset "
            "against the eigenvalue range (energies must be in eV)"
        )
    eta = float(np.sqrt((weight * difference**2).sum() / total_weight))
    eta_maximum = float(np.max(weight * np.abs(difference)))
    return BandDistance(eta=eta, eta_maximum=eta_maximum, number_of_states=int(weight.size))


def eigenvalues_electronvolt(results: dict | InterpolationResults, spin: int = 0) -> np.ndarray:
    """The ``eigenvalues_ry`` of a Q3 results file as a (k-point, band) array in eV.

    Both the engine's ``<prefix>_results.json`` and ``q3_utility interpolate``'s
    results file shape them ``[k-point][spin][band]``; one spin channel is
    taken, since the interpolation is spin-unpolarized today. The file may be
    the document or, for the utility's, the model of it.
    """
    stored = (
        results.eigenvalues_ry
        if isinstance(results, InterpolationResults)
        else results.get("eigenvalues_ry")
    )
    if stored is None:
        raise ValueError("the results carry no eigenvalues_ry")
    eigenvalues = np.asarray(stored, dtype=float)
    if eigenvalues.ndim != 3:
        raise ValueError("eigenvalues_ry must be shaped [k-point][spin][band]")
    return eigenvalues[:, spin, :] * RYDBERG_TO_ELECTRONVOLT
