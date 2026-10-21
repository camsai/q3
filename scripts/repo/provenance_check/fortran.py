"""
Fortran: content for references written in it, such as Quantum ESPRESSO and BerkeleyGW.

The core (``core.py``) knows only the abstract ``ForeignLanguage``. This file is
the one piece of reference-specific code; a reference written in another
language adds a file like it, registered in ``check_provenance.py``.
"""

from __future__ import annotations

import re

from .core import ForeignConstruct, ForeignLanguage, ScreenedPattern

OPERATORS = ("and", "or", "not", "eqv", "neqv", "eq", "ne", "lt", "le", "gt", "ge")
INTRINSICS = ("dble", "dcmplx", "conjg")
#: ``x**2``, ``x ** (y)``, ``2**n``, ``(a + b)**n`` -- but not ``int** p``,
#: ``char **argv`` or ``**iterator``, which are pointers.
POWER = re.compile(
    r"[\w)\]]\s*\*\*\s*(?:\d|\.\d|\()"
    r"|(?:(?<![\w.])\d+(?:\.\d*)?|[)\]])\s*\*\*\s*[a-z_]"
)

BOTH_HOSTS = frozenset({"cpp", "python"})
#: In Python, ``**`` is a power, and ``end if`` can be a variable named
#: ``end`` followed by a conditional expression.
CPP_ONLY = frozenset({"cpp"})


class FortranLanguage(ForeignLanguage):
    """
    Fortran: dotted operators, ``dble(`` and its kin, block ends, slices, ``**``.

    Matched against lowercased code, since Fortran is case-insensitive.
    """

    name = "fortran"

    def constructs(self) -> tuple[ForeignConstruct, ...]:
        """Every Fortran construct that cannot be C++ (or Python) code."""
        operators = tuple(
            ForeignConstruct(
                ScreenedPattern(
                    f".{operator}.", (f".{operator}.",), re.compile(re.escape(f".{operator}."))
                ),
                BOTH_HOSTS,
            )
            for operator in OPERATORS
        )
        intrinsics = tuple(
            ForeignConstruct(
                ScreenedPattern(f"{name}(", (name,), re.compile(rf"\b{name}\s*\(")), BOTH_HOSTS
            )
            for name in INTRINSICS
        )
        return (
            *operators,
            *intrinsics,
            ForeignConstruct(
                ScreenedPattern("enddo", ("enddo",), re.compile(r"\benddo\b")), BOTH_HOSTS
            ),
            ForeignConstruct(
                ScreenedPattern("endif", ("endif",), re.compile(r"\bendif\b")), BOTH_HOSTS
            ),
            ForeignConstruct(
                ScreenedPattern("end do", ("end",), re.compile(r"\bend[ \t]+do\b")), BOTH_HOSTS
            ),
            ForeignConstruct(
                ScreenedPattern("end if", ("end",), re.compile(r"\bend[ \t]+if\b")), CPP_ONLY
            ),
            ForeignConstruct(
                ScreenedPattern("(:) slice", (":",), re.compile(r"[(,]\s*:\s*[),]")), CPP_ONLY
            ),
            ForeignConstruct(ScreenedPattern("** power", ("**",), POWER), CPP_ONLY),
        )
