"""tests/python/test_check_python_import_layering.py — Unit tests for the layering checker.

Covers ``scripts/repo/check_python_import_layering.py``: the tier table, upward-import
detection, same-tier cycle detection, and the guarantee that the real ``q3``
package is clean.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_python_import_layering.py"


def _load_checker():
    """Import the checker script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("check_python_import_layering", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


# ── Tier table ───────────────────────────────────────────────────────────────


def test_every_subpackage_has_a_tier():
    """Every subpackage/module directly under q3/ must be assigned a tier.

    A new subpackage that nobody added to TIERS would be silently unchecked,
    which is exactly the failure mode this test exists to prevent.
    """
    package_root = checker.PACKAGE_ROOT
    top_level_names = set()
    for entry in package_root.iterdir():
        if entry.name.startswith("_"):
            continue  # __init__.py, __main__.py, __pycache__
        if entry.is_dir() and (entry / "__init__.py").exists():
            top_level_names.add(entry.name)
        elif entry.suffix == ".py":
            top_level_names.add(entry.stem)

    unassigned = top_level_names - set(checker.TIERS)
    assert not unassigned, (
        f"q3 subpackage(s) {sorted(unassigned)} have no tier. Add them to TIERS in "
        f"{SCRIPT_PATH.name} and document them in AGENTS.md § Python Package Layering."
    )


def test_tier_descriptions_cover_all_tiers():
    """Diagnostics interpolate TIER_DESCRIPTIONS, so every tier in use needs one."""
    assert set(checker.TIERS.values()) <= set(checker.TIER_DESCRIPTIONS)


@pytest.mark.parametrize(
    ("module_name", "expected_tier"),
    [
        ("q3.errors", 0),
        ("q3.binary", 1),
        ("q3.io.output_writer", 1),
        ("q3.structure.ase_atoms", 1),
        ("q3.calculators.ase_calculator", 2),
        ("q3.flavors.self_consistent_field", 2),
        ("q3.workflows.relax", 3),
        ("q3.jax.primitives", 3),
        ("q3", None),
        ("q3.__main__", None),
    ],
)
def test_tier_of(module_name, expected_tier):
    """tier_of() maps dotted module names onto the documented tiers."""
    assert checker.tier_of(module_name) == expected_tier


# ── Upward-import detection ──────────────────────────────────────────────────


def _write_module(tmp_path, monkeypatch, relative_path, source):
    """Write a fake q3 module under a temporary package root and point the checker at it."""
    package_root = tmp_path / "q3"
    target = package_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    monkeypatch.setattr(checker, "PACKAGE_ROOT", package_root)
    return target


def test_upward_import_is_reported(tmp_path, monkeypatch):
    """A tier-2 module importing a tier-3 module is a violation."""
    path = _write_module(
        tmp_path,
        monkeypatch,
        Path("calculators") / "ase_calculator.py",
        "from q3.workflows.relax import helper\n",
    )
    violations = checker.check_file(path)
    assert len(violations) == 1
    assert "upward import" in violations[0].message
    assert violations[0].line == 1


def test_downward_import_is_allowed(tmp_path, monkeypatch):
    """A tier-2 module importing tier-1 and tier-0 modules is fine."""
    path = _write_module(
        tmp_path,
        monkeypatch,
        Path("calculators") / "ase_calculator.py",
        "from q3.binary import find_binary\nfrom q3.errors import Q3Error\n"
        "from q3.structure.ase_atoms import material_json_to_ase_atoms\n",
    )
    assert checker.check_file(path) == []


def test_same_tier_import_is_allowed(tmp_path, monkeypatch):
    """One-way imports within a tier are fine — only mutual ones are cycles."""
    path = _write_module(
        tmp_path,
        monkeypatch,
        Path("workflows") / "band_structure.py",
        "from q3.workflows.kpath import interpolate_kpath\n",
    )
    assert checker.check_file(path) == []


def test_plain_import_statement_is_detected(tmp_path, monkeypatch):
    """``import q3.workflows.relax`` is caught, not just ``from ... import ...``."""
    path = _write_module(
        tmp_path,
        monkeypatch,
        Path("io") / "output_writer.py",
        "import q3.workflows.relax\n",
    )
    violations = checker.check_file(path)
    assert len(violations) == 1
    assert "q3.workflows.relax" in violations[0].message


def test_function_level_import_is_still_reported(tmp_path, monkeypatch):
    """Burying an upward import inside a function must not hide it.

    This is the exact evasion the checker exists to close: the pre-fix code
    deferred ``q3.calculators`` -> ``q3.workflows.relax`` into a method body
    so Python's own cycle detection never fired.
    """
    path = _write_module(
        tmp_path,
        monkeypatch,
        Path("calculators") / "ase_calculator.py",
        "def build():\n    from q3.workflows.relax import helper\n    return helper\n",
    )
    violations = checker.check_file(path)
    assert len(violations) == 1
    assert violations[0].line == 2


def test_third_party_imports_are_ignored(tmp_path, monkeypatch):
    """Imports outside the q3 package are none of this checker's business."""
    path = _write_module(
        tmp_path,
        monkeypatch,
        Path("io") / "output_writer.py",
        "import numpy as np\nfrom ase import Atoms\nimport q3lib.other\n",
    )
    assert checker.check_file(path) == []


def test_untiered_entry_point_is_skipped(tmp_path, monkeypatch):
    """``q3/__main__.py`` sits above every tier and may import anything."""
    path = _write_module(
        tmp_path,
        monkeypatch,
        Path("__main__.py"),
        "from q3.workflows.relax import run_relax_workflow\n",
    )
    assert checker.check_file(path) == []


# ── Same-tier cycle detection ────────────────────────────────────────────────


def test_same_tier_cycle_is_reported(tmp_path, monkeypatch):
    """Two tier-3 modules importing each other are reported once, not twice."""
    package_root = tmp_path / "q3"
    (package_root / "workflows").mkdir(parents=True)
    (package_root / "workflows" / "relax.py").write_text("from q3.workflows.vc_relax import a\n")
    (package_root / "workflows" / "vc_relax.py").write_text("from q3.workflows.relax import b\n")
    monkeypatch.setattr(checker, "PACKAGE_ROOT", package_root)

    violations = checker.check_same_tier_cycles(sorted(package_root.rglob("*.py")))
    assert len(violations) == 1
    assert "same-tier import cycle" in violations[0].message


def test_package_init_importing_submodules_is_not_a_cycle(tmp_path, monkeypatch):
    """``q3.workflows/__init__`` importing its own submodules is normal assembly."""
    package_root = tmp_path / "q3"
    (package_root / "workflows").mkdir(parents=True)
    (package_root / "workflows" / "__init__.py").write_text("from q3.workflows.relax import run\n")
    (package_root / "workflows" / "relax.py").write_text("from q3.workflows import shared\n")
    monkeypatch.setattr(checker, "PACKAGE_ROOT", package_root)

    assert checker.check_same_tier_cycles(sorted(package_root.rglob("*.py"))) == []


# ── The real package ─────────────────────────────────────────────────────────


def test_q3_package_layering_is_clean():
    """The checked-in q3 package must satisfy its own layering rules."""
    violations = []
    for path in sorted(checker.PACKAGE_ROOT.rglob("*.py")):
        violations.extend(checker.check_file(path))
    violations.extend(checker.check_same_tier_cycles(sorted(checker.PACKAGE_ROOT.rglob("*.py"))))
    assert violations == [], "\n".join(str(v) for v in violations)
