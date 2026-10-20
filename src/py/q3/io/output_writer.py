"""q3.io.output_writer — Extract structured results from Q3 output.

Converts the flat ``results.json`` dict (written by the C++ binary) into
structured Python objects ready for use by workflows and ESSE serialisation.

All energy values are in Rydberg; forces in Ry/Bohr; stress in kbar.

Public functions
----------------
extract_total_energy(results)
    Return the total energy in Ry.

extract_forces(results)
    Return the per-atom Cartesian forces in Ry/Bohr.

extract_stress(results)
    Return the 3x3 stress tensor in kbar.

raise_if_derivative_incomplete(results, properties)
    Refuse a force or stress Q3 has reported as incomplete.

raise_if_not_converged(results, *, returncode)
    Refuse a result whose SCF did not converge, saying how far it got.

extract_band_structure(results, kpoints, kpath_distances)
    Convert eigenvalues + k-path distances into a band-structure dict.

extract_dos(results, *, energy_min, energy_max, energy_steps, broadening)
    Compute a Gaussian-broadened DOS from the eigenvalues in *results*.

extract_lowdin_charges(results)
    Compute per-orbital Löwdin population charges from NSCF orbital
    projections.

extract_pdos(results, *, energy_min, energy_max, energy_steps, broadening)
    Compute Gaussian-broadened per-orbital PDOS curves from NSCF orbital
    projections.
"""

from __future__ import annotations

import math

import numpy as np
from mat3ra.code.constants import Coefficients as _Coefficients

from q3.errors import IncompleteDerivativeError, SCFNotConvergedError

# ---------------------------------------------------------------------------
# Total energy
# ---------------------------------------------------------------------------


def extract_total_energy(results: dict) -> float:
    """Return the total energy in Rydberg.

    Accepts both the new flat schema (``total_energy_ry``) and the legacy
    nested schema (``total_energy.value``).

    Parameters
    ----------
    results:
        Dict returned by ``run_q3()`` / parsed from ``results.json``.

    Returns
    -------
    float
        Total DFT energy in Rydberg.

    Raises
    ------
    KeyError
        If neither key is present in *results*.
    """
    if "total_energy_ry" in results:
        return float(results["total_energy_ry"])
    # Legacy nested format.
    return float(results["total_energy"]["value"])


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


def extract_forces(results: dict) -> list[list[float]]:
    """Return the per-atom Cartesian forces in Ry/Bohr.

    Parameters
    ----------
    results:
        Dict returned by ``run_q3()`` / parsed from ``results.json``.

    Returns
    -------
    list[list[float]]
        One ``[fx, fy, fz]`` triple per atom, in Q3's native Ry/Bohr.  Empty
        when the run computed no forces, which is not an error: a single-point
        energy is a complete result.

    Raises
    ------
    IncompleteDerivativeError
        If Q3 reported the forces as incomplete.  Reading a derivative Q3 has
        disclaimed is the failure this guard exists for, so it fires here as
        well as in the ASE calculator.
    """
    raise_if_derivative_incomplete(results, ("forces",))
    return [
        [float(component) for component in force] for force in results.get("forces_ry_per_bohr", [])
    ]


def extract_stress(results: dict) -> list[list[float]]:
    """Return the 3x3 stress tensor in kbar.

    The sign is Q3's own Nielsen-Martin convention, under which a compressed
    cell has positive diagonal stress -- *not* ASE's tensile-positive one.  Use
    ``q3.calculators.ase_calculator.stress_kbar_to_ase_voigt`` to convert.

    Parameters
    ----------
    results:
        Dict returned by ``run_q3()`` / parsed from ``results.json``.

    Returns
    -------
    list[list[float]]
        3x3 stress tensor in kbar, or an empty list when the run computed no
        stress.

    Raises
    ------
    IncompleteDerivativeError
        If Q3 reported the stress as incomplete.
    """
    raise_if_derivative_incomplete(results, ("stress",))
    return [[float(component) for component in row] for row in results.get("stress_kbar", [])]


def raise_if_derivative_incomplete(results: dict, properties) -> None:
    """Refuse a force or stress that Q3 has reported as incomplete.

    Q3 knows which derivative terms it cannot yet compute and says so in its
    results JSON. A geometry optimization driven by one of them does not fail --
    it converges, to the wrong geometry, with nothing in its output to say why.

    Only the properties actually asked for are checked: the energy of the same
    system is correct, and a plain single-point run must keep working.

    This lives here, beside the other readers of ``results.json``, rather than
    in the ASE calculator that used to own it: every consumer of a Q3 result
    needs the guard, not only the ones that go through ASE.

    Parameters
    ----------
    results:
        Dict returned by ``run_q3()`` / parsed from ``results.json``.
    properties:
        Which quantities are actually being asked for; only these are checked.

    Raises
    ------
    IncompleteDerivativeError
        If a requested derivative is flagged incomplete.
    """
    for quantity, flag, reason_key in (
        ("forces", "forces_are_incomplete", "forces_incomplete_reason"),
        ("stress", "stress_is_incomplete", "stress_incomplete_reason"),
    ):
        if quantity in properties and results.get(flag):
            reason = results.get(reason_key, "")
            raise IncompleteDerivativeError(
                f"Q3 reports its {quantity} are incomplete for this system, so a calculation "
                f"driven by them would converge to a wrong answer without failing. "
                f"Reason: {reason}.",
                quantity=quantity,
                reason=reason,
            )


def raise_if_not_converged(results: dict, *, returncode: int = -1) -> None:
    """Refuse a result whose SCF did not converge, saying how far it got.

    The binary exits 0 on non-convergence, so ``converged`` in the results is
    the only signal. The message reports the iterations Q3 says it performed
    (``self_consistent_field_iterations``). A binary too old to write that key
    still records the limit it was given, in its provenance, and the message
    then states only that limit -- never a count the result does not carry.
    Supplying one is how every failed run came to be reported as stopping
    "after 0 iterations".

    Shared by every reader of a Q3 result, so that the subprocess runner and
    the ASE calculator describe the same failure in the same words.

    Parameters
    ----------
    results:
        Dict returned by ``run_q3()`` / parsed from ``results.json``. A result
        with no ``converged`` key is accepted as converged, as before.
    returncode:
        Exit status carried on the error, as on any ``Q3RuntimeError``; the
        default -1 means it is not known.

    Raises
    ------
    SCFNotConvergedError
        If *results* reports ``converged: false``.
    """
    if results.get("converged", True):
        return

    iterations = results.get("self_consistent_field_iterations")
    precision = (results.get("provenance") or {}).get("precision") or {}
    maximum_iterations = precision.get("maximum_self_consistent_field_iterations")

    if iterations is not None and maximum_iterations is not None:
        message = (
            f"SCF did not converge after {iterations} iterations; "
            f"its limit was {maximum_iterations}."
        )
    elif iterations is not None:
        message = f"SCF did not converge after {iterations} iterations."
    elif maximum_iterations is not None:
        message = f"SCF did not converge within its {maximum_iterations}-iteration limit."
    else:
        message = "SCF did not converge."

    raise SCFNotConvergedError(
        message,
        iterations=iterations,
        maximum_iterations=maximum_iterations,
        returncode=returncode,
    )


# ---------------------------------------------------------------------------
# Band structure
# ---------------------------------------------------------------------------


def extract_band_structure(
    results: dict,
    kpoints: list[tuple[float, float, float]],
    kpath_distances: list[float],
    fermi_energy_ry: float | None = None,
) -> dict:
    """Build a band-structure dict from NSCF *results*.

    Parameters
    ----------
    results:
        Dict returned by ``run_nscf()`` / parsed from ``results.json``.
    kpoints:
        Ordered list of k-point coordinates (crystal units) used in the NSCF
        run.  Same order as ``results["eigenvalues_ry"]``.
    kpath_distances:
        Cumulative path distance for each k-point (used as the x-axis).
        Typically computed by ``band_path_distances()``.
    fermi_energy_ry:
        Fermi energy in Ry.  Extracted from *results* if not supplied.

    Returns
    -------
    dict
        Band structure with keys:
        ``eigenvalues_ry``, ``kpoints``, ``kpath_distances``, ``fermi_energy_ry``.
        Eigenvalues are shaped ``[num_kpoints][num_spin][num_bands]``.
    """
    # Support both new and legacy schemas.
    if "eigenvalues_ry" in results:
        eigenvalues = results["eigenvalues_ry"]  # [kpt][spin][band]
    else:
        # Legacy: results["eigenvalues"]["kpoints"][i]["bands"]  (spin-0 only)
        raw_kpoints = results["eigenvalues"]["kpoints"]
        eigenvalues = [[kpt_data["bands"]] for kpt_data in raw_kpoints]

    if fermi_energy_ry is None:
        fermi_energy_ry = float(results.get("fermi_energy_ry", 0.0))

    return {
        "eigenvalues_ry": eigenvalues,
        "kpoints": kpoints,
        "kpath_distances": kpath_distances,
        "fermi_energy_ry": fermi_energy_ry,
    }


# ---------------------------------------------------------------------------
# DOS
# ---------------------------------------------------------------------------

_RY_TO_EV = _Coefficients.RYDBERG


def extract_dos(
    results: dict,
    *,
    energy_min_ry: float | None = None,
    energy_max_ry: float | None = None,
    energy_steps: int = 1000,
    broadening_ry: float = 0.01,
) -> dict:
    """Compute a Gaussian-broadened DOS from SCF eigenvalues.

    Parameters
    ----------
    results:
        Dict returned by ``run_scf()`` / parsed from ``results.json``.
    energy_min_ry:
        Lower energy limit of the DOS grid in Ry.  Defaults to
        ``min(eigenvalues) - 5 * broadening``.
    energy_max_ry:
        Upper energy limit in Ry.  Defaults to
        ``max(eigenvalues) + 5 * broadening``.
    energy_steps:
        Number of energy grid points.
    broadening_ry:
        Gaussian broadening width (σ) in Ry.

    Returns
    -------
    dict
        DOS with keys: ``energies_ry``, ``dos``, ``fermi_energy_ry``.
        ``dos[i]`` is the density of states at ``energies_ry[i]``.
    """
    # Collect eigenvalues and their k-point weights from the results.
    # Each eigenvalue at k-point k gets weight w_k.
    all_eigenvalues: list[float] = []
    all_weights: list[float] = []

    kpoint_weights = results.get("kpoint_weights", None)

    if "eigenvalues_ry" in results:
        for kpoint_index, kpt_data in enumerate(results["eigenvalues_ry"]):
            # Determine weight for this k-point.
            if kpoint_weights is not None and kpoint_index < len(kpoint_weights):
                weight = kpoint_weights[kpoint_index]
            else:
                # Fallback: equal weight (sum to 2.0 for spin degeneracy).
                weight = 2.0 / len(results["eigenvalues_ry"])
            for spin_data in kpt_data:
                for eigenvalue in spin_data:
                    all_eigenvalues.append(eigenvalue)
                    all_weights.append(weight)
    else:
        # Legacy format.
        for kpt_data in results["eigenvalues"]["kpoints"]:
            weight = kpt_data.get("weight", 1.0)
            for eigenvalue in kpt_data["bands"]:
                all_eigenvalues.append(eigenvalue)
                all_weights.append(weight)

    eigenvalues = np.array(all_eigenvalues, dtype=np.float64)
    weights = np.array(all_weights, dtype=np.float64)

    effective_energy_min = (
        energy_min_ry if energy_min_ry is not None else eigenvalues.min() - 5.0 * broadening_ry
    )
    effective_energy_max = (
        energy_max_ry if energy_max_ry is not None else eigenvalues.max() + 5.0 * broadening_ry
    )

    energies = np.linspace(effective_energy_min, effective_energy_max, energy_steps)

    # Gaussian broadening: DOS(E) = Σ_nk w_k * δ_σ(E - ε_nk)
    # where δ_σ is a Gaussian of width σ and w_k is the k-point integration weight.
    sigma = broadening_ry
    dos = np.zeros_like(energies)
    for eigenvalue, weight in zip(eigenvalues, weights):
        dos += (
            weight
            * np.exp(-0.5 * ((energies - eigenvalue) / sigma) ** 2)
            / (sigma * math.sqrt(2.0 * math.pi))
        )

    fermi_energy_ry = float(results.get("fermi_energy_ry", 0.0))

    return {
        "energies_ry": energies.tolist(),
        "dos": dos.tolist(),
        "fermi_energy_ry": fermi_energy_ry,
    }


# ---------------------------------------------------------------------------
# PDOS / Löwdin charges
# ---------------------------------------------------------------------------


def _require_orbital_projections(results: dict, function_name: str) -> None:
    if not results.get("orbital_projections"):
        raise KeyError(
            f"{function_name}: 'orbital_projections' not present in results -- "
            "re-run NSCF with control.compute_projections: true."
        )


def _group_key(label: dict) -> tuple[int, str]:
    return (label["atom_index"], label["shell_label"])


def _new_atom_shell_entry(label: dict) -> dict:
    return {
        "atom_index": label["atom_index"],
        "species_type_index": label["species_type_index"],
        "shell_label": label["shell_label"],
        "angular_momentum": label["angular_momentum"],
    }


def extract_lowdin_charges(results: dict) -> dict:
    """Compute Löwdin (ortho-atomic) population charges from NSCF *results*.

    Implements the same formula as QE's ``projwfc.x``:
    ``charge_i = Σ_{k,n} wg(n,k) · proj_i(n,k)``, where
    ``wg(n,k) = occupations[k][0][n] · kpoint_weights[k]`` (Sanchez-Portal
    et al., Solid State Commun. 95, 685 (1995)).

    Parameters
    ----------
    results:
        Dict returned by an NSCF run with ``compute_projections: true`` over
        a proper (dense, full-BZ) k-point grid -- see
        ``q3.workflows.pdos.run_pdos_workflow``.  Requires
        ``orbital_projections``, ``orbital_labels``, ``occupations``, and
        ``kpoint_weights``.

    Returns
    -------
    dict
        - ``"orbital"`` — list of per-orbital dicts (label fields plus
          ``"charge"``), same order as ``results["orbital_labels"]``.
        - ``"atom_shell"`` — the same charges grouped by (atom, shell_label),
          i.e. summed over the magnetic quantum number m -- matches QE's
          per-atom ``pdos_atm#N(elem)_wfc#M(l)`` grouping.
        - ``"total_charge"`` — Σ over all orbitals.
        - ``"number_of_electrons"`` — Σ_{k,n} wg(n,k), re-derived from the
          occupations/weights actually used for this run (not read from the
          input), matching QE's own convention. The denominator for
          ``spilling_parameter``.
        - ``"spilling_parameter"`` — ``1 - total_charge / number_of_electrons``,
          the finite-basis-incompleteness diagnostic QE reports. Zero only in
          the limit of a complete atomic basis.

    Raises
    ------
    KeyError
        If *results* does not contain orbital projections (i.e. the NSCF run
        did not set ``compute_projections: true``).
    """
    _require_orbital_projections(results, "extract_lowdin_charges")

    orbital_labels = results["orbital_labels"]
    projections = results["orbital_projections"]  # [kpt][band][orbital]
    occupations = results["occupations"]  # [kpt][spin][band]
    kpoint_weights = results["kpoint_weights"]  # [kpt]

    charges = [0.0] * len(orbital_labels)
    number_of_electrons = 0.0

    for kpoint_index, kpoint_projections in enumerate(projections):
        kweight = kpoint_weights[kpoint_index]
        kpoint_occupations = occupations[kpoint_index][0]
        for band_index, band_projections in enumerate(kpoint_projections):
            wg = kpoint_occupations[band_index] * kweight
            number_of_electrons += wg
            for orbital_index, projection in enumerate(band_projections):
                charges[orbital_index] += wg * projection

    orbital_charges = [{**label, "charge": charges[i]} for i, label in enumerate(orbital_labels)]

    atom_shell: dict[tuple[int, str], dict] = {}
    for i, label in enumerate(orbital_labels):
        key = _group_key(label)
        entry = atom_shell.setdefault(key, {**_new_atom_shell_entry(label), "charge": 0.0})
        entry["charge"] += charges[i]

    total_charge = sum(charges)

    return {
        "orbital": orbital_charges,
        "atom_shell": list(atom_shell.values()),
        "total_charge": total_charge,
        "number_of_electrons": number_of_electrons,
        "spilling_parameter": (
            1.0 - total_charge / number_of_electrons if number_of_electrons != 0.0 else 0.0
        ),
    }


def extract_pdos(
    results: dict,
    *,
    energy_min_ry: float | None = None,
    energy_max_ry: float | None = None,
    energy_steps: int = 1000,
    broadening_ry: float = 0.01,
    fermi_energy_ry: float | None = None,
) -> dict:
    """Compute Gaussian-broadened projected DOS curves from NSCF *results*.

    Same broadening formula as ``extract_dos`` --
    ``PDOS_i(E) = Σ_nk w_k · proj_i(n,k) · δ_σ(E - ε_nk)`` -- but each
    band's contribution is weighted by its ortho-atomic projection onto
    orbital *i* instead of counted once per eigenvalue, so
    ``Σ_i PDOS_i(E) ≈ DOS(E)`` up to the same basis-set "spilling" deficit
    reported by ``extract_lowdin_charges``.

    Parameters
    ----------
    results:
        Dict returned by an NSCF run with ``compute_projections: true``.
    energy_min_ry, energy_max_ry:
        Energy window in Ry.  Defaults to ±5σ beyond the eigenvalue range,
        matching ``extract_dos``.
    energy_steps:
        Number of energy grid points.
    broadening_ry:
        Gaussian broadening σ in Ry.
    fermi_energy_ry:
        Fermi energy in Ry, for the returned ``"fermi_energy_ry"`` field
        only (does not affect the PDOS curves themselves). Extracted from
        *results* if not supplied -- pass this explicitly to prefer an SCF
        run's Fermi energy, mirroring ``extract_band_structure``.

    Returns
    -------
    dict
        - ``"energies_ry"`` — energy grid.
        - ``"orbital_pdos"`` — list of per-orbital dicts (label fields plus
          ``"dos"`` curve), same order as ``results["orbital_labels"]``.
        - ``"atom_shell_pdos"`` — the same curves grouped by (atom,
          shell_label), matching QE's ``pdos_atm#N(elem)_wfc#M(l)`` files.
        - ``"summed_pdos"`` — Σ over all orbitals at each energy, comparable
          to the total DOS from ``extract_dos`` (their difference is the
          spilling deficit, in the energy domain).
        - ``"fermi_energy_ry"``.

    Raises
    ------
    KeyError
        If *results* does not contain orbital projections.
    """
    _require_orbital_projections(results, "extract_pdos")

    orbital_labels = results["orbital_labels"]
    projections = results["orbital_projections"]  # [kpt][band][orbital]
    eigenvalues = results["eigenvalues_ry"]  # [kpt][spin][band]
    kpoint_weights = results["kpoint_weights"]  # [kpt]

    all_eigenvalues: list[float] = []
    all_band_weights: list[float] = []
    all_projections: list[list[float]] = []
    for kpoint_index, kpoint_projections in enumerate(projections):
        kweight = kpoint_weights[kpoint_index]
        kpoint_eigenvalues = eigenvalues[kpoint_index][0]
        for band_index, band_projections in enumerate(kpoint_projections):
            all_eigenvalues.append(kpoint_eigenvalues[band_index])
            all_band_weights.append(kweight)
            all_projections.append(band_projections)

    eigenvalue_array = np.array(all_eigenvalues, dtype=np.float64)
    weight_array = np.array(all_band_weights, dtype=np.float64)
    projection_array = np.array(all_projections, dtype=np.float64)  # [band*, orbital]

    effective_energy_min = (
        energy_min_ry if energy_min_ry is not None else eigenvalue_array.min() - 5.0 * broadening_ry
    )
    effective_energy_max = (
        energy_max_ry if energy_max_ry is not None else eigenvalue_array.max() + 5.0 * broadening_ry
    )
    energies = np.linspace(effective_energy_min, effective_energy_max, energy_steps)

    sigma = broadening_ry
    # gaussian[b, e] = δ_σ(energies[e] - eigenvalue_array[b])
    gaussian = np.exp(-0.5 * ((energies[None, :] - eigenvalue_array[:, None]) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi)
    )
    weighted_gaussian = gaussian * weight_array[:, None]  # [band*, energy]

    # orbital_pdos_array[i, e] = Σ_b projection_array[b, i] * weighted_gaussian[b, e]
    orbital_pdos_array = projection_array.T @ weighted_gaussian  # [orbital, energy]

    orbital_pdos = [
        {**orbital_labels[i], "dos": orbital_pdos_array[i].tolist()}
        for i in range(len(orbital_labels))
    ]

    atom_shell: dict[tuple[int, str], dict] = {}
    for i, label in enumerate(orbital_labels):
        key = _group_key(label)
        if key not in atom_shell:
            atom_shell[key] = {**_new_atom_shell_entry(label), "dos": np.zeros_like(energies)}
        atom_shell[key]["dos"] = atom_shell[key]["dos"] + orbital_pdos_array[i]

    atom_shell_pdos = []
    for entry in atom_shell.values():
        entry["dos"] = entry["dos"].tolist()
        atom_shell_pdos.append(entry)

    if fermi_energy_ry is None:
        fermi_energy_ry = float(results.get("fermi_energy_ry", 0.0))

    return {
        "energies_ry": energies.tolist(),
        "orbital_pdos": orbital_pdos,
        "atom_shell_pdos": atom_shell_pdos,
        "summed_pdos": orbital_pdos_array.sum(axis=0).tolist(),
        "fermi_energy_ry": fermi_energy_ry,
    }
