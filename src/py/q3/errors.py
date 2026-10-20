"""Custom exception hierarchy for the Q3 Python layer.

All exceptions raised by q3 Python code are instances of ``Q3Error`` or
one of its subclasses so that callers can catch them uniformly::

    try:
        result = run_scf(...)
    except q3.SCFNotConvergedError as exc:
        print(f"SCF failed: {exc}")  # says how far the run got, when the result does
    except q3.Q3Error:
        ...  # any other q3 error
"""

from __future__ import annotations

import re

# ── what a binary's output says ──────────────────────────────────────────────

_DIAGNOSTIC_LINE = re.compile(r"(?m)^.*?\bdiagnostic:\s*([a-z_]+)\s*$")


def first_error_line(text: str) -> str | None:
    """The first ``Error:`` line of a binary's output, from ``Error:`` on; None without one.

    The engine logs through spdlog, so in its output the line reads
    ``[time] [error] Error: ...``; a stub writes ``Error: ...`` to stderr. Either
    way the sentence, led by ``Error:``, is what a ladder's attempt and a
    campaign's table want -- not the timestamp before it, and not the timing
    footer the engine prints after it.
    """
    for line in text.splitlines():
        index = line.find("Error:")
        if index >= 0:
            return line[index:].strip()
    return None


def diagnostic_code(text: str) -> str | None:
    """The code on a binary's ``diagnostic:`` line, when it printed one; None without."""
    match = _DIAGNOSTIC_LINE.search(text)
    return match.group(1) if match else None


# ── the hierarchy ────────────────────────────────────────────────────────────


class Q3Error(Exception):
    """Base class for all Q3 Python exceptions."""


class Q3InputError(Q3Error):
    """Raised when the supplied input is invalid or incomplete.

    Examples: missing required field in YAML, pseudopotential file not found,
    incompatible calculation type / pseudopotential combination.
    """


class Q3RuntimeError(Q3Error):
    """Raised when a Q3 binary exits with a non-zero status.

    Attributes:
        returncode: Exit code returned by the binary.
        stderr: What the binary wrote to stderr -- or, when its output was
            shown as it ran, the tail of the one merged stream.
        stdout: What the binary wrote to stdout, when it was kept; the engine
            logs through stdout, so its own words are usually here.
        command: The command line that was run.
        diagnostic: The code on the binary's ``diagnostic:`` line, when it
            printed one after its ``Error:`` -- what a driver routes on
            without reading the prose.
    """

    def __init__(
        self,
        message: str,
        returncode: int = -1,
        stderr: str = "",
        *,
        stdout: str = "",
        command: list[str] | None = None,
        diagnostic: str | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout
        self.command = list(command) if command else []
        self.diagnostic = diagnostic

    @property
    def text(self) -> str:
        """Both streams of the failed run, because the binary's reason may be on either.

        The message carries the command line and ``stderr`` the binary's own
        words -- except that the engine logs through stdout, so what a driver
        routes on can arrive by either route.
        """
        return ((self.stderr or "") + "\n" + str(self)).strip()

    @property
    def error_line(self) -> str:
        """The binary's own ``Error:`` line, or the first line of :attr:`text` without one."""
        spoken = first_error_line(self.text)
        if spoken is not None:
            return spoken
        return self.text.splitlines()[0] if self.text else "the binary failed"


class Q3VersionMismatchError(Q3Error):
    """Raised when the checkpoint was written by an incompatible binary version.

    The NSCF flavor requires a v2 checkpoint (written by a binary that supports
    the new checkpoint format).  If an older v1 checkpoint is found and V_eff
    reconstruction is not possible, this error is raised.
    """


class SCFNotConvergedError(Q3RuntimeError):
    """Raised when the binary reports convergence failure.

    Attributes:
        iterations: Number of SCF iterations that were performed before giving
            up, or ``None`` when the result does not say -- binaries older than
            the ``self_consistent_field_iterations`` key never did.
        maximum_iterations: The iteration limit the run was given, or ``None``
            when the result carries no provenance to read it from.
    """

    def __init__(
        self,
        message: str,
        iterations: int | None = None,
        maximum_iterations: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.iterations = iterations
        self.maximum_iterations = maximum_iterations


class IncompleteDerivativeError(Q3RuntimeError):
    """Raised when Q3 reports that a force or stress term was left out.

    Q3 knows about terms it cannot yet compute -- today the DFT+U stress with
    ortho-atomic projectors, whose strain derivative runs through d(O^-1/2) over
    every atomic orbital in the cell -- and says so in its results JSON.

    A geometry optimization driven by such a derivative does not fail: it
    converges, to the wrong geometry, with nothing in its output to say why.
    Refusing is the whole point, so this is raised where the derivative is
    consumed rather than where it is produced -- a plain energy calculation on
    the same system is unaffected and still runs.

    Attributes:
        quantity: "forces" or "stress".
        reason: Q3's own description of the missing term.
    """

    def __init__(self, message: str, quantity: str = "", reason: str = "", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.quantity = quantity
        self.reason = reason
