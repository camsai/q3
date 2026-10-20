"""q3.io.pseudo_atomic_shells.upf_reader — every ``PP_CHI`` shell of a UPF dataset.

Version 2 states each shell's l, occupation and usually its label as
attributes; version 1 (the ultrasoft datasets) states all three on one header
line per shell inside ``PP_PSWFC``. A shell's principal quantum number comes
from its label, and a file whose shells carry none has them counted from its
element's core (:mod:`q3.io.pseudo_atomic_shells.principal_quantum_numbers`).
"""

from __future__ import annotations

import re
from pathlib import Path

from q3.io.pseudo_atomic_shells.principal_quantum_numbers import (
    principal_quantum_numbers_from_the_core,
)
from q3.io.pseudo_atomic_shells.pseudo_atomic_shell import PseudoAtomicShell

#: Below this, the integral of chi^2 says a shell has no radial function: the
#: engine's threshold (the UPF reader's ``zero_norm_threshold``), under which it
#: skips the shell with a "zero norm" warning.
ZERO_NORM_THRESHOLD = 1.0e-10

_SHELL_BLOCK = re.compile(r"<PP_CHI\.(\d+)\b([^>]*)>(.*?)</PP_CHI\.\1>", re.DOTALL)
_RADIAL_STEPS = re.compile(r"<PP_RAB\b[^>]*>(.*?)</PP_RAB>", re.DOTALL)
_VALUE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
_ANGULAR_MOMENTUM = re.compile(r'\bl\s*=\s*"\s*(\d+)\s*"')
_OCCUPATION = re.compile(r'\boccupation\s*=\s*"\s*([-+0-9.eEdD]+)\s*"')
_LABEL = re.compile(r'\blabel\s*=\s*"\s*(\S*?)\s*"')
_HEADER_TAG = re.compile(r"<PP_HEADER\b([^>]*)>", re.DOTALL)
_ELEMENT = re.compile(r'\belement\s*=\s*"\s*([A-Za-z]+)\s*"')
_VALENCE_CHARGE = re.compile(r'\bz_valence\s*=\s*"\s*([-+0-9.eEdD]+)\s*"')
#: UPF version 1: a header line per shell inside <PP_PSWFC>, "3S  0  2.00  Wavefunction".
_VERSION_ONE_BLOCK = re.compile(r"<PP_PSWFC>(.*?)</PP_PSWFC>", re.DOTALL)
_VERSION_ONE_SHELL = re.compile(
    r"^\s*(\d[A-Za-z])\s+(\d+)\s+([-+0-9.eEdD]+)\s+Wavefunction", re.MULTILINE
)
_LABELLED_SHELL = re.compile(r"^(\d+)[A-Za-z]$")


def pseudo_atomic_shells(upf_path: str | Path) -> list[PseudoAtomicShell]:
    """Every ``PP_CHI`` shell of a dataset, in the order the file lists them."""
    text = Path(upf_path).read_text(errors="replace")
    radial_steps = _radial_steps(text)
    shells = _version_two_shells(text, radial_steps, upf_path) or _version_one_shells(
        text, radial_steps
    )
    if not shells:
        raise ValueError(
            f"{upf_path} carries no PP_CHI shells (no pseudo-atomic orbitals), so it cannot "
            "supply trial orbitals; that takes a dataset with PP_PSWFC"
        )
    numbers = _principal_quantum_numbers(text, shells)
    if numbers is None:
        return shells
    return [
        shell.model_copy(update={"principal_quantum_number": number})
        for shell, number in zip(shells, numbers, strict=True)
    ]


def _number(text: str) -> float:
    """A Fortran-written real, whose exponent may be spelt with a D."""
    return float(text.replace("D", "E").replace("d", "e"))


def _radial_steps(text: str) -> list[float]:
    """``PP_RAB``, the mesh's integration weights; empty when the file carries none."""
    block = _RADIAL_STEPS.search(text)
    return [_number(value) for value in _VALUE.findall(block.group(1))] if block else []


def _version_two_shells(
    text: str, radial_steps: list[float], upf_path: str | Path
) -> list[PseudoAtomicShell]:
    shells = []
    for index, block in enumerate(_SHELL_BLOCK.finditer(text)):
        attributes = block.group(2)
        angular_momentum = _ANGULAR_MOMENTUM.search(attributes)
        if angular_momentum is None:
            raise ValueError(f"{upf_path}: a PP_CHI shell carries no angular momentum l")
        occupation = _OCCUPATION.search(attributes)
        label = _LABEL.search(attributes)
        shells.append(
            PseudoAtomicShell(
                index=index,
                angular_momentum=int(angular_momentum.group(1)),
                occupation=_number(occupation.group(1)) if occupation else 0.0,
                label=label.group(1) if label else None,
                zero_norm=_has_zero_norm(block.group(3), radial_steps),
            )
        )
    return shells


def _version_one_shells(text: str, radial_steps: list[float]) -> list[PseudoAtomicShell]:
    block = _VERSION_ONE_BLOCK.search(text)
    body = block.group(1) if block else ""
    headers = list(_VERSION_ONE_SHELL.finditer(body))
    if not headers:
        return []
    # A shell's radial function runs from its header line to the next one's.
    ends = [header.start() for header in headers[1:]] + [len(body)]
    return [
        PseudoAtomicShell(
            index=index,
            angular_momentum=int(header.group(2)),
            occupation=_number(header.group(3)),
            label=header.group(1),
            zero_norm=_has_zero_norm(body[header.end() : end], radial_steps),
        )
        for index, (header, end) in enumerate(zip(headers, ends, strict=True))
    ]


def _has_zero_norm(values: str, radial_steps: list[float]) -> bool:
    """Whether a shell's radial function integrates to nothing over the file's mesh.

    No mesh to integrate on means no verdict, and the shell is kept.
    """
    if not radial_steps:
        return False
    chi = [_number(value) for value in _VALUE.findall(values)]
    norm = sum(value * value * step for value, step in zip(chi, radial_steps, strict=False))
    return norm < ZERO_NORM_THRESHOLD


def _principal_quantum_numbers(text: str, shells: list[PseudoAtomicShell]) -> list[int] | None:
    """Each shell's n: from the labels when every shell has one, otherwise from the header."""
    labelled = [_LABELLED_SHELL.match(shell.label or "") for shell in shells]
    if all(labelled):
        return [int(match.group(1)) for match in labelled]
    header = _HEADER_TAG.search(text)
    element = _ELEMENT.search(header.group(1)) if header else None
    valence_charge = _VALENCE_CHARGE.search(header.group(1)) if header else None
    if element is None or valence_charge is None:
        return None
    return principal_quantum_numbers_from_the_core(
        element.group(1).capitalize(),
        _number(valence_charge.group(1)),
        [shell.angular_momentum for shell in shells],
    )
