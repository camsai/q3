"""tests/python/test_check_cpp_type_queries.py — Unit tests for the type-query ratchet.

Covers ``scripts/repo/check_cpp_type_queries.py``: per-category counting (comments
and declarations ignored, standard ``has_value()`` not a capability query),
the baseline comparison in both directions, the ``--write-baseline`` refusal
to raise counts, and the guarantee that the real tree matches its baseline.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_cpp_type_queries.py"


def _load_checker():
    """Import the checker script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("check_cpp_type_queries", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_counts_each_category():
    source = """
    auto* us = dynamic_cast<const UltrasoftPseudopotential*>(p);
    auto sp = std::dynamic_pointer_cast<Paw>(q);
    if (typeid(*p) == typeid(Q)) {}
    auto& paw = static_cast<const PawPseudopotential&>(*p);
    const auto& set = pseudopotentials.as_ultrasoft();
    if (system.has_nonlinear_core_correction()) {}
    if (applicator.has_hamiltonian_applicator()) {}
    if (projector_type == "ortho-atomic") {}
    """
    counts = checker.count_type_queries(source)
    assert counts == {
        "dynamic_cast": 2,
        "typeid": 2,
        "pseudopotential_downcast": 1,
        "family_accessor": 1,
        "family_flag": 1,
        "capability_branch": 1,
        "string_switch": 1,
    }


def test_ignores_comments_declarations_and_standard_presence_checks():
    source = """
    // dynamic_cast<Foo*>(bar) in a comment
    /* if (x.has_ultrasoft()) */
    bool has_ultrasoft() const { return flag_; }
    virtual const UltrasoftSet& as_ultrasoft() const = 0;
    if (value.has_value()) {}
    auto x = static_cast<double>(n);
    log("dynamic_cast<X>");
    """
    assert not checker.count_type_queries(source)


def test_has_call_outside_branch_is_not_capability_branch():
    assert not checker.count_type_queries("bool ok = obj.has_feature();")


def test_compare_reports_increase_and_decrease():
    increases, decreases = checker.compare(
        {"src/a.hpp": {"dynamic_cast": 2}, "src/new.hpp": {"typeid": 1}},
        {"src/a.hpp": {"dynamic_cast": 3}},
    )
    assert increases == ["src/new.hpp: typeid 0 -> 1"]
    assert decreases == ["src/a.hpp: dynamic_cast 3 -> 2"]


def _point_at(tmp_path, monkeypatch, source, baseline):
    """Point the checker at a one-file fake src/ tree and a temporary baseline."""
    source_root = tmp_path / "src"
    (source_root / "layer_0_core").mkdir(parents=True)
    (source_root / "layer_0_core" / "a.hpp").write_text(source)
    (source_root / "generated").mkdir()
    (source_root / "generated" / "g.hpp").write_text("dynamic_cast<A*>(b);")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    monkeypatch.setattr(checker, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(checker, "BASELINE_PATH", baseline_path)
    return baseline_path


def test_main_fails_on_new_query_and_skips_generated(tmp_path, monkeypatch, capsys):
    _point_at(tmp_path, monkeypatch, "auto p = dynamic_cast<A*>(b);", {})
    assert checker.main([]) == 1
    output = capsys.readouterr().out
    assert "src/layer_0_core/a.hpp: dynamic_cast 0 -> 1" in output
    assert "generated" not in output


def test_main_requires_tightening_after_removal(tmp_path, monkeypatch):
    baseline = {"src/layer_0_core/a.hpp": {"dynamic_cast": 1}}
    baseline_path = _point_at(tmp_path, monkeypatch, "int x;", baseline)
    assert checker.main([]) == 1
    assert checker.main(["--write-baseline"]) == 0
    assert json.loads(baseline_path.read_text()) == {}
    assert checker.main([]) == 0


def test_write_baseline_refuses_increase_without_flag(tmp_path, monkeypatch):
    baseline_path = _point_at(tmp_path, monkeypatch, "dynamic_cast<A*>(b);", {})
    assert checker.main(["--write-baseline"]) == 1
    assert json.loads(baseline_path.read_text()) == {}
    assert checker.main(["--write-baseline", "--allow-increase"]) == 0
    assert json.loads(baseline_path.read_text()) == {"src/layer_0_core/a.hpp": {"dynamic_cast": 1}}


def test_an_addons_entries_live_in_its_own_baseline(tmp_path, monkeypatch):
    """An add-on's files are ratcheted beside its code, never in the core baseline."""
    baseline_path = _point_at(tmp_path, monkeypatch, "dynamic_cast<A*>(b);", {})
    addon_directory = checker.SOURCE_ROOT / "addons" / "widget"
    addon_directory.mkdir(parents=True)
    (addon_directory / "w.hpp").write_text("dynamic_cast<A*>(b);")
    assert checker.main(["--write-baseline", "--allow-increase"]) == 0
    assert json.loads(baseline_path.read_text()) == {"src/layer_0_core/a.hpp": {"dynamic_cast": 1}}
    addon_baseline = addon_directory / checker.BASELINE_PATH.name
    assert json.loads(addon_baseline.read_text()) == {
        "src/addons/widget/w.hpp": {"dynamic_cast": 1}
    }
    assert checker.main([]) == 0


def test_real_tree_matches_baseline():
    increases, decreases = checker.compare(checker.measure(), checker.load_baseline())
    assert increases == []
    assert decreases == []
