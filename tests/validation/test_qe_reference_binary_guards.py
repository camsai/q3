"""Every validation test that can launch Quantum ESPRESSO must self-skip.

The QE comparison tests run against a reference build (QE 7.5 + LibXC 7.0.0)
that lives outside the checkout and is never present on a GitHub runner --
see the resolution note in tests/helpers.py. A test that reaches `pw.x`
without a guard therefore does not fail on the author's machine, where the
build exists; it fails on CI, weeks later.

That is not hypothetical. `test_qe_graphite_d3_dispersion.py` re-derived the
binary path locally instead of importing it, lost its guard, and turned the
scheduled tier-2 job red for two consecutive cycles -- runs 30676195249
(2026-08-01, 2 failed / 213 passed / 58 skipped, 1:37:12) and 31853121200
(2026-08-15, 1 failed / 320 passed / 122 skipped, 1:47:34) -- each time with a
FileNotFoundError raised after the rest of the suite had already run. The
scheduled workflow is the only thing that runs tier 2, so nothing caught it
in between.

Two hand-written predecessors of this file each pinned one named test's
marker. This one derives the list instead, so a *new* unguarded test is
caught too, and deliberately carries no `@pytest.mark.slow`: it must run in
the per-pull-request `tests/validation/` job, not on the bi-monthly schedule
whose blind spot it exists to cover. It needs no QE build and no q3 binary --
it only reads source.

Reachability is resolved per module with a call graph over the module's own
functions, because the launch is usually one or two hops from the test: a
test calls a module-level helper (`_qe_gamma_phonon_frequencies_thz`), or
requests a fixture that launches QE (`ultrasoft_reference_runs`), rather than
calling `subprocess.run` itself.
"""

import ast
from pathlib import Path

import pytest

VALIDATION_DIRECTORY = Path(__file__).parent

# Names that stand for the QE reference binaries. A test that mentions one of
# these, directly or through a helper or fixture, can launch QE.
QE_BINARY_NAMES = frozenset({"QE_PW", "QE_PW2BGW", "QE_PROJWFC", "QE_BIN_DIR", "QE_BINARY"})

# Names that stand for "QE is present". Referencing one is what a guard looks
# like from the outside, whether it is used in a `skipif` condition, an `if
# not ...: pytest.skip(...)` at the top of a fixture, or the shared marker.
QE_GUARD_NAMES = frozenset({"requires_qe", "QE_AVAILABLE"})


def _module_paths():
    return sorted(
        p for p in VALIDATION_DIRECTORY.glob("test_*.py") if p.name != Path(__file__).name
    )


class _FunctionIndex:
    """Per-module map of function name -> (names it mentions, names it calls).

    Nested functions are folded into their enclosing definition: a closure like
    `forces_for` inside `_qe_gamma_phonon_frequencies_thz` runs only when its
    parent does, so attributing its QE launch to the parent is what makes the
    reachability walk correct.
    """

    def __init__(self, tree):
        self.mentions = {}
        self.calls = {}
        self.decorators = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._record(node)

    def _record(self, node):
        mentioned, called = set(), set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                mentioned.add(child.id)
            elif isinstance(child, ast.Attribute):
                mentioned.add(child.attr)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                called.add(child.func.id)
        self.mentions[node.name] = mentioned
        self.calls[node.name] = called
        # A fixture is requested by parameter name, so a test's parameters are
        # edges in the call graph just as much as its calls are.
        self.calls[node.name] |= {argument.arg for argument in node.args.args}
        self.decorators[node.name] = {
            name for decorator in node.decorator_list for name in _decorator_names(decorator)
        }

    def reaches(self, name, targets, _seen=None):
        """Whether `name`, or anything it can reach, mentions one of `targets`."""
        seen = _seen if _seen is not None else set()
        if name in seen or name not in self.mentions:
            return False
        seen.add(name)
        if self.mentions[name] & targets:
            return True
        return any(self.reaches(callee, targets, seen) for callee in self.calls[name])


def _decorator_names(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            yield child.id
        elif isinstance(child, ast.Attribute):
            yield child.attr


def _unguarded_tests(source, filename="<source>"):
    index = _FunctionIndex(ast.parse(source, filename=filename))
    unguarded = []
    for name in index.mentions:
        if not name.startswith("test_"):
            continue
        if not index.reaches(name, QE_BINARY_NAMES):
            continue
        # A guard counts wherever it sits on the path to the launch: on the
        # test itself, or inside a helper or fixture it reaches.
        if index.decorators[name] & QE_GUARD_NAMES or index.reaches(name, QE_GUARD_NAMES):
            continue
        unguarded.append(name)
    return unguarded


@pytest.mark.parametrize("path", _module_paths(), ids=lambda p: p.name)
def test_every_live_qe_test_is_guarded(path):
    """No validation test reaches a QE binary without a missing-QE skip."""
    unguarded = _unguarded_tests(path.read_text(), filename=str(path))
    assert not unguarded, (
        f"{path.name}: {', '.join(unguarded)} can launch Quantum ESPRESSO but never checks that "
        f"the reference build exists. On a runner that has no QE this raises FileNotFoundError "
        f"instead of skipping. Decorate the test with `requires_qe` from tests/helpers.py, or "
        f"guard the fixture it uses with `if not QE_AVAILABLE: pytest.skip(...)`."
    )


def test_the_guard_check_can_actually_fail():
    """The check above is only worth having if it detects the real regression.

    Without this, an `_unguarded_tests` that silently returned `[]` -- a
    renamed constant, a broken parse -- would report the whole suite as clean
    and reproduce exactly the blind spot this file exists to close.

    The fixture below is the shape the graphite test had in runs 30676195249
    and 31853121200: guarded module, unguarded test, launch one hop away.
    """
    source = """
from helpers import QE_PW, requires_qe

def _run_qe(path):
    return subprocess.run([QE_PW, "-in", path])

@requires_qe
def test_guarded():
    _run_qe("in")

def test_unguarded():
    _run_qe("in")

def test_no_qe_at_all():
    assert True
"""
    assert _unguarded_tests(source) == ["test_unguarded"]
