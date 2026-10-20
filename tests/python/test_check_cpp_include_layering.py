"""tests/python/test_check_cpp_include_layering.py — Unit tests for the C++ layering checker.

Covers ``scripts/repo/check_cpp_include_layering.py``: layer extraction from paths,
upward-include detection, downward/same-layer includes being allowed, the
exempt-path categories, the ``KNOWN_UPWARD_INCLUDES`` allowlist (both that it
covers what it should, and that it covers *only* what it should), and the
guarantee that the real ``src`` tree is clean under the current allowlist.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_cpp_include_layering.py"


def _load_checker():
    """Import the checker script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("check_cpp_include_layering", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


# ── Layer extraction from paths ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("relative_path", "expected_layer"),
    [
        (Path("layer_0_core/types/init.hpp"), 0),
        (Path("layer_1_structure/symmetry/symmetry.hpp"), 1),
        (Path("layer_2_quantum_system/band/init.hpp"), 2),
        (Path("layer_3_representation/basis/planewave/init.hpp"), 3),
        (Path("layer_4_computation/flavors/self_consistent_field.hpp"), 4),
        (Path("layer_5_properties/forces/planewave/init.hpp"), 5),
        (Path("io/logger.hpp"), None),
        (Path("bindings/q3_core_api.hpp"), None),
        (Path("main.cpp"), None),
        (Path("utils/something.hpp"), None),
        (Path("generated/esse/schema.hpp"), None),
    ],
)
def test_layer_of(relative_path, expected_layer):
    """layer_of() reads the layer number directly off the directory name."""
    assert checker.layer_of(relative_path) == expected_layer


# ── Exempt-path categories ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("io/logger.hpp"),
        Path("io/input/parser/yaml.hpp"),
        Path("bindings/q3_core_api.hpp"),
        Path("utils/something.hpp"),
        Path("generated/esse/schema.hpp"),
        Path("py/some_module.hpp"),
        Path("main.cpp"),
        Path("main_mpi.cpp"),
        Path("main_gpu.cpp"),
        Path("main_gpu_cuda.cpp"),
        Path("main_gpu_hip.cpp"),
        Path("main_gpu_sycl.cpp"),
    ],
)
def test_exempt_paths_are_exempt(relative_path):
    """io/, bindings/, utils/, generated/, py/, and the main*.cpp entry points are exempt."""
    assert checker.is_exempt(relative_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("layer_0_core/types/init.hpp"),
        Path("layer_4_computation/flavors/self_consistent_field.hpp"),
        Path("layer_5_properties/forces/planewave/init.hpp"),
    ],
)
def test_layer_paths_are_not_exempt(relative_path):
    """A layer_N_*/ path itself must never be treated as exempt."""
    assert not checker.is_exempt(relative_path)


# ── Upward-include detection ─────────────────────────────────────────────────


def _write_source_file(tmp_path, monkeypatch, relative_path, source):
    """Write a fake source file under a temporary src/ root and point the checker at it."""
    source_root = tmp_path / "src"
    target = source_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    monkeypatch.setattr(checker, "SOURCE_ROOT", source_root)
    return target


def test_upward_include_is_reported(tmp_path, monkeypatch):
    """A layer-1 file including a layer-3 header is a violation."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_1_structure") / "symmetry" / "symmetry.hpp",
        '#include "layer_3_representation/basis/planewave/init.hpp"\n',
    )
    violations = checker.check_file(path)
    assert len(violations) == 1
    assert "upward include" in violations[0].message
    assert violations[0].line == 1


def test_downward_include_is_allowed(tmp_path, monkeypatch):
    """A layer-4 file including layer-0..3 headers is fine."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "iterator" / "init.hpp",
        '#include "layer_0_core/types/init.hpp"\n'
        '#include "layer_1_structure/crystal/crystal.hpp"\n'
        '#include "layer_3_representation/basis/planewave/init.hpp"\n',
    )
    assert checker.check_file(path) == []


def test_same_layer_include_is_allowed(tmp_path, monkeypatch):
    """One layer_2 file including a sibling layer_2 header is fine."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_2_quantum_system") / "band" / "occupations.hpp",
        '#include "layer_2_quantum_system/band/init.hpp"\n',
    )
    assert checker.check_file(path) == []


def test_multiple_upward_includes_are_all_reported(tmp_path, monkeypatch):
    """Every offending line is reported, not just the first."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_0_core") / "sneaky.hpp",
        '#include "layer_0_core/types/init.hpp"\n'
        '#include "layer_2_quantum_system/band/init.hpp"\n'
        '#include "layer_5_properties/forces/planewave/init.hpp"\n',
    )
    violations = checker.check_file(path)
    assert len(violations) == 2
    assert violations[0].line == 2
    assert violations[1].line == 3


def test_include_inside_doc_comment_is_not_misdetected(tmp_path, monkeypatch):
    """A usage-example #include quoted inside a /** */ docstring is not a real include.

    Regression test: src/layer_0_core/types/init.hpp and src/io/logger.hpp both
    document their own header with a `` *   #include "..." `` example line
    inside a docstring. Those lines are indented under a `*` continuation, not
    a leading `#`, so a checker that searches for "#include" anywhere on the
    line (rather than anchoring to line-start) would misread them.
    """
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_0_core") / "types" / "init.hpp",
        "/**\n"
        " * Example:\n"
        ' *   #include "layer_5_properties/forces/planewave/init.hpp"\n'
        " */\n"
        '#include "layer_0_core/types/type_aliases.hpp"\n',
    )
    assert checker.check_file(path) == []


def test_plain_string_paths_outside_src_are_ignored(tmp_path, monkeypatch):
    """A file resolved outside SOURCE_ROOT entirely is not this checker's business."""
    source_root = tmp_path / "src"
    source_root.mkdir()
    monkeypatch.setattr(checker, "SOURCE_ROOT", source_root)
    outside_file = tmp_path / "elsewhere" / "not_in_src.hpp"
    outside_file.parent.mkdir(parents=True)
    outside_file.write_text('#include "layer_5_properties/forces/planewave/init.hpp"\n')
    assert checker.check_file(outside_file) == []


# ── Exemptions honored end-to-end through check_file ─────────────────────────


def test_io_file_with_upward_include_is_not_flagged(tmp_path, monkeypatch):
    """io/ is a vertical slice (AGENTS.md): it may reference any layer."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("io") / "input" / "system_builder.hpp",
        '#include "layer_5_properties/forces/planewave/init.hpp"\n'
        '#include "layer_0_core/types/init.hpp"\n',
    )
    assert checker.check_file(path) == []


def test_bindings_file_with_upward_include_is_not_flagged(tmp_path, monkeypatch):
    """bindings/ must expose every layer to Python, so it is exempt."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("bindings") / "q3_core_api.hpp",
        '#include "layer_5_properties/forces/planewave/init.hpp"\n',
    )
    assert checker.check_file(path) == []


def test_entry_point_file_with_upward_include_is_not_flagged(tmp_path, monkeypatch):
    """main*.cpp entry points assemble every layer, so they are exempt."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("main.cpp"),
        '#include "layer_5_properties/forces/planewave/init.hpp"\n'
        '#include "layer_0_core/types/init.hpp"\n',
    )
    assert checker.check_file(path) == []


# ── KNOWN_UPWARD_INCLUDES allowlist ───────────────────────────────────────────


def test_core_file_including_an_addon_header_is_reported(tmp_path, monkeypatch):
    """The core names no add-on, so a layer file may not include one."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "flavors" / "self_consistent_field.hpp",
        '#include "addons/widget/registration.hpp"\n',
    )
    violations = checker.check_file(path)
    assert len(violations) == 1
    assert "add-on header" in violations[0].message


def test_exempt_core_file_including_an_addon_header_is_reported(tmp_path, monkeypatch):
    """io/ may include any layer but still not an add-on, in either include form."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("io") / "input" / "parser" / "factory.hpp",
        '#include "addons/widget/io/format.hpp"\n#include <addons/widget/other.hpp>\n',
    )
    assert len(checker.check_file(path)) == 2


def test_entry_point_including_an_addon_header_is_reported(tmp_path, monkeypatch):
    """Entry points reach add-ons through the generated registration header, never directly."""
    path = _write_source_file(
        tmp_path, monkeypatch, Path("main.cpp"), '#include "addons/widget/registration.hpp"\n'
    )
    assert len(checker.check_file(path)) == 1


def test_addon_file_may_include_core_and_its_own_headers(tmp_path, monkeypatch):
    """An add-on is a vertical slice: core layers and its own headers are fine."""
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("addons") / "widget" / "registration.hpp",
        '#include "addons/widget/io/format.hpp"\n'
        '#include "layer_5_properties/forces/planewave/init.hpp"\n'
        '#include "io/input/parser/input_format_registry.hpp"\n',
    )
    assert checker.check_file(path) == []


def test_known_upward_include_allowlist_is_honored(tmp_path, monkeypatch):
    """A directory on the allowlist for the matching layer is not flagged."""
    monkeypatch.setattr(
        checker,
        "KNOWN_UPWARD_INCLUDES",
        {("layer_4_computation/flavors", 5): "test allowlist entry"},
    )
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "flavors" / "self_consistent_field.hpp",
        '#include "layer_5_properties/forces/planewave/init.hpp"\n',
    )
    assert checker.check_file(path) == []


def test_allowlist_directory_match_is_prefix_scoped(tmp_path, monkeypatch):
    """The allowlist directory key must not accidentally cover a same-named sibling.

    ``layer_4_computation/flavors`` on the allowlist must not also cover
    ``layer_4_computation/flavors_extra`` -- prefix matching must respect the
    path separator, not just the string prefix.
    """
    monkeypatch.setattr(
        checker,
        "KNOWN_UPWARD_INCLUDES",
        {("layer_4_computation/flavors", 5): "test allowlist entry"},
    )
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "flavors_extra" / "sneaky.hpp",
        '#include "layer_5_properties/forces/planewave/init.hpp"\n',
    )
    violations = checker.check_file(path)
    assert len(violations) == 1


def test_upward_include_outside_allowlisted_directory_still_fails(tmp_path, monkeypatch):
    """A layer_4 file *outside* flavors/ including layer_5 is a real violation.

    This is the "allowlist misses fail" guarantee: only the exact allowlisted
    (path, layer) pair is forgiven, nothing else.
    """
    monkeypatch.setattr(
        checker,
        "KNOWN_UPWARD_INCLUDES",
        {("layer_4_computation/flavors", 5): "test allowlist entry"},
    )
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "iterator" / "sneaky.hpp",
        '#include "layer_5_properties/forces/planewave/init.hpp"\n',
    )
    violations = checker.check_file(path)
    assert len(violations) == 1
    assert "upward include" in violations[0].message


def test_allowlist_entry_for_one_layer_does_not_cover_another(tmp_path, monkeypatch):
    """An allowlist entry keyed to layer 5 must not forgive an upward include to a different layer."""
    monkeypatch.setattr(
        checker,
        "KNOWN_UPWARD_INCLUDES",
        {("layer_1_structure/symmetry", 5): "test allowlist entry, layer 5 only"},
    )
    path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_1_structure") / "symmetry" / "sneaky.hpp",
        '#include "layer_3_representation/basis/planewave/init.hpp"\n',
    )
    violations = checker.check_file(path)
    assert len(violations) == 1


def test_allowlist_file_key_matches_only_that_file(tmp_path, monkeypatch):
    """A file-level (not directory-level) allowlist key does not cover a sibling file."""
    monkeypatch.setattr(
        checker,
        "KNOWN_UPWARD_INCLUDES",
        {("layer_4_computation/flavors/self_consistent_field.hpp", 5): "test file-level entry"},
    )
    allowed_path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "flavors" / "self_consistent_field.hpp",
        '#include "layer_5_properties/forces/planewave/init.hpp"\n',
    )
    assert checker.check_file(allowed_path) == []

    sibling_path = _write_source_file(
        tmp_path,
        monkeypatch,
        Path("layer_4_computation") / "flavors" / "non_self_consistent_field.hpp",
        '#include "layer_5_properties/projections/planewave/init.hpp"\n',
    )
    violations = checker.check_file(sibling_path)
    assert len(violations) == 1


# ── The real repository tree ─────────────────────────────────────────────────


def test_every_top_level_source_entry_is_classified():
    """Every top-level src/ entry must be a layer directory, exempt, or an entry point.

    A new top-level directory that is none of these would otherwise fall
    through ``layer_of()`` returning None and ``is_exempt()`` returning
    False being read by ``check_file`` as "nothing to check" -- silently
    unenforced rather than a decision someone made. This is the same
    failure mode `test_every_subpackage_has_a_tier` guards against on the
    Python side.
    """
    unclassified = []
    for entry in checker.SOURCE_ROOT.iterdir():
        if entry.is_dir():
            if entry.name in checker.EXEMPT_TOP_LEVEL_DIRECTORIES:
                continue
            if checker.LAYER_DIRECTORY_RE.match(entry.name):
                continue
            unclassified.append(entry.name)
        elif entry.is_file() and entry.suffix in checker.SOURCE_EXTENSIONS:
            if checker.EXEMPT_ENTRY_POINT_RE.match(entry.name):
                continue
            unclassified.append(entry.name)

    assert not unclassified, (
        f"src/ top-level entr(y/ies) {sorted(unclassified)} are neither a layer_N_* "
        "directory nor declared exempt. Add them to EXEMPT_TOP_LEVEL_DIRECTORIES or "
        "EXEMPT_ENTRY_POINT_RE in scripts/repo/check_cpp_include_layering.py (and document "
        "the decision in AGENTS.md), or confirm they belong in the layer stack."
    )


def test_known_upward_includes_are_documented_with_a_todos_reference():
    """Every allowlist entry must point at the TODOs.md item that tracks its cleanup."""
    for (path_or_directory, layer), reason in checker.KNOWN_UPWARD_INCLUDES.items():
        assert "TODOs.md" in reason, (
            f"KNOWN_UPWARD_INCLUDES entry ({path_or_directory!r}, {layer}) has no "
            "TODOs.md reference in its reason string."
        )


def test_repository_sources_are_clean():
    """The checked-in src/ tree must satisfy its own layering rules (allowlist included)."""
    violations = []
    for path in checker.collect_source_files([]):
        violations.extend(checker.check_file(path))
    assert violations == [], "\n".join(str(violation) for violation in violations)
