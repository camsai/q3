"""tests/python/test_check_test_mirror.py — Unit tests for the test-mirror checker.

Covers ``scripts/repo/check_test_mirror.py``: placement detection (including the
ambiguous-``init.hpp`` case that must not false-positive), CMake registration
in both directions (missing and stale entries), the ``KNOWN_MIRROR_EXCEPTIONS``
allowlist, and the guarantee that the real ``tests/`` tree and
``CMakeLists.txt`` are clean under the current checks.
"""

import importlib.util
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_test_mirror.py"


def _load_checker():
    """Import the checker script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("check_test_mirror", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _write_test_file(tmp_path, monkeypatch, relative_path, source):
    """Write a fake test file under a temporary tests/unit/ root and point the checker at it."""
    tests_unit_root = tmp_path / "tests" / "unit"
    target = tests_unit_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    monkeypatch.setattr(checker, "TESTS_UNIT_ROOT", tests_unit_root)
    return target


# ── Placement ─────────────────────────────────────────────────────────────────


def test_misplaced_test_is_reported(tmp_path, monkeypatch):
    """A test one directory too shallow relative to the header it includes is a violation."""
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "diagonalizer" / "test_direct.cpp",
        '#include "layer_4_computation/diagonalizer/planewave/direct.hpp"\n',
    )
    violation = checker.check_placement(path)
    assert violation is not None
    assert "planewave/test_direct.cpp" in violation.message


def test_correctly_placed_test_is_not_reported(tmp_path, monkeypatch):
    """A test whose directory matches its tested header's directory passes."""
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "diagonalizer" / "planewave" / "test_direct.cpp",
        '#include "layer_4_computation/diagonalizer/planewave/direct.hpp"\n',
    )
    assert checker.check_placement(path) is None


def test_test_name_matching_no_include_is_silently_skipped(tmp_path, monkeypatch):
    """A test named after the class it produces, not the header it tests, is not checked.

    Regression case: tests/unit/layer_4_computation/diagonalizer/planewave/subspace/
    test_subspace_diagonalizer.cpp tests SubspaceDiagonalizerFactory (in
    factory.hpp) but is named after the class, not the file -- this checker
    cannot ground a placement claim in that case (see the module docstring)
    and must not guess.
    """
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "diagonalizer" / "test_subspace_diagonalizer.cpp",
        '#include "layer_4_computation/diagonalizer/planewave/subspace/factory.hpp"\n',
    )
    assert checker.check_placement(path) is None


def test_ambiguous_shared_basename_is_not_a_false_positive(tmp_path, monkeypatch):
    """A test including two same-named headers from different directories is fine

    as long as one of them is its own. ``init.hpp`` alone names dozens of
    files in this codebase (the blessed package-entry convention), so this is
    not a rare edge case: a test's own, correctly-placed ``init.hpp`` can
    easily sit alongside an *earlier* same-named include from a dependency's
    directory. Regression test for a real bug caught while writing this
    checker: a naive "first matching include wins" implementation reported
    this as misplaced.
    """
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("layer_5_properties") / "projections" / "planewave" / "test_init.cpp",
        '#include "layer_3_representation/basis/planewave/init.hpp"\n'
        '#include "layer_5_properties/projections/planewave/init.hpp"\n',
    )
    assert checker.check_placement(path) is None


def test_ambiguous_shared_basename_true_violation_still_reported(tmp_path, monkeypatch):
    """If *none* of several same-named candidates match, it is still a real violation."""
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("layer_5_properties") / "projections" / "test_init.cpp",
        '#include "layer_3_representation/basis/planewave/init.hpp"\n'
        '#include "layer_5_properties/projections/planewave/init.hpp"\n',
    )
    violation = checker.check_placement(path)
    assert violation is not None
    assert "test_init.cpp" in violation.message


def test_known_mirror_exception_placement_is_honored(tmp_path, monkeypatch):
    """A placement allowlist entry suppresses an otherwise-real violation."""
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "diagonalizer" / "test_direct.cpp",
        '#include "layer_4_computation/diagonalizer/planewave/direct.hpp"\n',
    )
    monkeypatch.setattr(
        checker,
        "KNOWN_MIRROR_EXCEPTIONS",
        {
            (
                "tests/unit/layer_4_computation/diagonalizer/test_direct.cpp",
                "placement",
            ): "test allowlist entry"
        },
    )
    assert checker.check_placement(path) is None


# ── CMake registration ──────────────────────────────────────────────────────


def _write_cmake_lists(tmp_path, monkeypatch, q3_tests_block, q3_tests_mpi_block=""):
    cmake_lists = tmp_path / "CMakeLists.txt"
    cmake_lists.write_text(
        f"add_executable(q3_tests\n{q3_tests_block}\n)\n"
        f"add_executable(q3_tests_mpi\n{q3_tests_mpi_block}\n)\n"
    )
    monkeypatch.setattr(checker, "CMAKE_LISTS_PATH", cmake_lists)
    return cmake_lists


def _make_test_trees(tmp_path, monkeypatch):
    tests_unit_root = tmp_path / "tests" / "unit"
    tests_integration_root = tmp_path / "tests" / "integration"
    tests_unit_root.mkdir(parents=True)
    tests_integration_root.mkdir(parents=True)
    monkeypatch.setattr(checker, "TESTS_UNIT_ROOT", tests_unit_root)
    monkeypatch.setattr(checker, "TESTS_INTEGRATION_ROOT", tests_integration_root)
    monkeypatch.setattr(checker, "REPOSITORY_ROOT", tmp_path)
    # The fake tree's own src/, so the real tree's add-on test lists are not read.
    monkeypatch.setattr(checker, "SOURCE_ROOT", tmp_path / "src")
    return tests_unit_root, tests_integration_root


def test_unregistered_file_is_reported(tmp_path, monkeypatch):
    """An on-disk test file absent from every CMake test-source list is a violation."""
    tests_unit_root, _ = _make_test_trees(tmp_path, monkeypatch)
    test_file = tests_unit_root / "layer_0_core" / "test_something.cpp"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    _write_cmake_lists(tmp_path, monkeypatch, "")  # nothing registered at all

    violations = checker.check_registration()
    assert len(violations) == 1
    assert "not listed" in violations[0].message


def test_stale_registration_is_reported(tmp_path, monkeypatch):
    """A CMake entry with no file on disk is a violation, the other direction."""
    _make_test_trees(tmp_path, monkeypatch)
    _write_cmake_lists(tmp_path, monkeypatch, "        tests/unit/layer_0_core/test_deleted.cpp")

    violations = checker.check_registration()
    assert len(violations) == 1
    assert "stale" in violations[0].message


def test_registered_existing_file_is_clean(tmp_path, monkeypatch):
    """A file that exists and is registered produces no violation."""
    tests_unit_root, _ = _make_test_trees(tmp_path, monkeypatch)
    test_file = tests_unit_root / "layer_0_core" / "test_something.cpp"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    _write_cmake_lists(tmp_path, monkeypatch, "        tests/unit/layer_0_core/test_something.cpp")

    assert checker.check_registration() == []


def test_integration_tests_are_also_checked(tmp_path, monkeypatch):
    """tests/integration/ is covered by registration, same as tests/unit/."""
    _, tests_integration_root = _make_test_trees(tmp_path, monkeypatch)
    test_file = tests_integration_root / "test_something.cpp"
    test_file.write_text("")
    _write_cmake_lists(tmp_path, monkeypatch, "")

    violations = checker.check_registration()
    assert len(violations) == 1
    assert "tests/integration/test_something.cpp" in str(violations[0].path)


def test_known_mirror_exception_registration_is_honored(tmp_path, monkeypatch):
    """A registration allowlist entry suppresses an otherwise-real 'unregistered' violation."""
    tests_unit_root, _ = _make_test_trees(tmp_path, monkeypatch)
    test_file = tests_unit_root / "layer_0_core" / "test_fixture_helper.cpp"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    _write_cmake_lists(tmp_path, monkeypatch, "")
    monkeypatch.setattr(
        checker,
        "KNOWN_MIRROR_EXCEPTIONS",
        {
            (
                "tests/unit/layer_0_core/test_fixture_helper.cpp",
                "registration",
            ): "test allowlist entry"
        },
    )

    assert checker.check_registration() == []


# ── Build-time add-ons ──────────────────────────────────────────────────────


def test_addon_test_of_its_own_header_mirrors_the_addon_tree(tmp_path, monkeypatch):
    """tests/unit/addons/<name>/<path>/test_X.cpp tests src/addons/<name>/<path>/X.hpp."""
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("addons") / "widget" / "io" / "test_widget_format.cpp",
        '#include "addons/widget/io/widget_format.hpp"\n',
    )
    assert checker.check_placement(path) is None


def test_addon_test_of_a_core_header_mirrors_its_core_path(tmp_path, monkeypatch):
    """An add-on's test of a core header sits at the core path under the add-on prefix."""
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("addons") / "widget" / "layer_4_computation" / "paw" / "test_densities.cpp",
        '#include "layer_4_computation/paw/densities.hpp"\n',
    )
    assert checker.check_placement(path) is None


def test_addon_test_of_a_core_header_at_the_wrong_core_path_is_reported(tmp_path, monkeypatch):
    """The add-on allowance still requires the core path to match."""
    path = _write_test_file(
        tmp_path,
        monkeypatch,
        Path("addons") / "widget" / "layer_4_computation" / "test_densities.cpp",
        '#include "layer_4_computation/paw/densities.hpp"\n',
    )
    assert checker.check_placement(path) is not None


def test_addon_tests_are_registered_through_the_addon_cmake_call(tmp_path, monkeypatch):
    """A test listed in q3_register_addon(... UNIT_TESTS ...) counts as registered."""
    tests_unit_root, _ = _make_test_trees(tmp_path, monkeypatch)
    test_file = tests_unit_root / "addons" / "widget" / "test_registration.cpp"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    _write_cmake_lists(tmp_path, monkeypatch, "")
    source_root = tmp_path / "src"
    addon_cmake_lists = source_root / "addons" / "widget" / "CMakeLists.txt"
    addon_cmake_lists.parent.mkdir(parents=True)
    addon_cmake_lists.write_text(
        "q3_register_addon(\n    NAME widget\n    UNIT_TESTS\n"
        "        tests/unit/addons/widget/test_registration.cpp\n)\n"
    )
    monkeypatch.setattr(checker, "SOURCE_ROOT", source_root)

    assert checker.check_registration() == []


def test_a_tree_without_addons_needs_no_addon_registration(tmp_path, monkeypatch):
    """With src/addons/ absent, as in the public export, registration still checks cleanly."""
    tests_unit_root, _ = _make_test_trees(tmp_path, monkeypatch)
    test_file = tests_unit_root / "layer_0_core" / "test_something.cpp"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("")
    _write_cmake_lists(tmp_path, monkeypatch, "        tests/unit/layer_0_core/test_something.cpp")
    monkeypatch.setattr(checker, "SOURCE_ROOT", tmp_path / "src")

    assert checker.check_registration() == []


def test_multiline_regex_matches_every_line_not_just_the_first():
    """Regression test for a real bug: CMAKE_TEST_SOURCE_RE without re.MULTILINE

    matched at most one line in a whole multi-hundred-line block (``^``/``$``
    anchor to the string, not the line, without the flag), which made almost
    every registered file look unregistered.
    """
    assert checker.CMAKE_TEST_SOURCE_RE.flags & __import__("re").MULTILINE


# ── The real repository tree ─────────────────────────────────────────────────


def test_known_mirror_exceptions_are_documented():
    """Every allowlist entry must carry a non-empty reason."""
    for key, reason in checker.KNOWN_MIRROR_EXCEPTIONS.items():
        assert reason.strip(), f"KNOWN_MIRROR_EXCEPTIONS entry {key!r} has an empty reason."


def test_repository_tests_are_clean():
    """The checked-in tests/ tree and CMakeLists.txt satisfy this checker's own rules."""
    violations = []
    for test_path in checker.collect_unit_test_files([]):
        violation = checker.check_placement(test_path)
        if violation is not None:
            violations.append(violation)
    violations.extend(checker.check_registration())
    assert violations == [], "\n".join(str(violation) for violation in violations)
