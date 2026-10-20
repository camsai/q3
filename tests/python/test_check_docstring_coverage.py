"""tests/python/test_check_docstring_coverage.py — Unit tests for the docstring ratchet.

Covers ``scripts/repo/check_docstring_coverage.py``: which C++ declarations count
as public (access specifiers, struct vs class, internal namespaces, special
members, function bodies), which doc-comment forms count, the Python AST
side, the baseline ratchet, and the guarantee that the real tree matches its
baseline.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_docstring_coverage.py"


def _load_checker():
    """Import the checker script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("check_docstring_coverage", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()

HEADER = """
#pragma once
namespace q3 {
/**
 * Documented class.
 */
class Foo : public Bar {
public:
    Foo(int x) : x_(x) {}
    ~Foo() = default;
    /**
     * Documented, with a multi-line body.
     */
    double energy() const { return compute(
        1); }
    int undocumented(int a,
                     int b) const;
    template <typename T>
    T also_undocumented() {
        if (x) { return T{}; }
        return T{};
    }
    /// Single-line doc.
    static std::vector<int> listed();
    bool operator==(const Foo&) const = default;
private:
    int hidden();
    int x_;
};
struct Plain {
    int value;
    int get() const;
};
namespace internal {
int secret();
}
inline double free_function(double x) { return x; }
}
"""


def test_cpp_header_counts_public_declarations():
    # Public: Foo, energy, undocumented, also_undocumented, listed, Plain,
    # get, free_function. Undocumented: the last five except listed.
    assert checker.scan_cpp_header(HEADER) == (8, 5)


def test_plain_line_comment_is_not_a_doc_comment():
    assert checker.scan_cpp_header("// note\nint f();\n") == (1, 1)
    assert checker.scan_cpp_header("/// doc\nint f();\n") == (1, 0)


def test_python_module_counts_public_definitions():
    source = '''
def documented():
    """Doc."""

def undocumented():
    pass

def _private():
    pass

class Public:
    def method(self):
        pass

    def _helper(self):
        pass

class _Hidden:
    def method(self):
        pass
'''
    assert checker.scan_python_module(source) == (4, 3)


def _point_at(tmp_path, monkeypatch, header, baseline):
    """Point the checker at a one-file fake src/ tree and a temporary baseline."""
    source_root = tmp_path / "src"
    (source_root / "layer_0_core").mkdir(parents=True)
    (source_root / "layer_0_core" / "a.hpp").write_text(header)
    (source_root / "generated").mkdir()
    (source_root / "generated" / "g.hpp").write_text("int g();\n")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    monkeypatch.setattr(checker, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(checker, "BASELINE_PATH", baseline_path)
    return baseline_path


def test_main_fails_on_new_undocumented_api(tmp_path, monkeypatch, capsys):
    _point_at(tmp_path, monkeypatch, "int f();\n", {})
    assert checker.main([]) == 1
    output = capsys.readouterr().out
    assert "src/layer_0_core/a.hpp" in output
    assert "generated" not in output


def test_main_requires_tightening_after_documenting(tmp_path, monkeypatch):
    baseline_path = _point_at(
        tmp_path, monkeypatch, "/// doc\nint f();\n", {"src/layer_0_core/a.hpp": 1}
    )
    assert checker.main([]) == 1
    assert checker.main(["--write-baseline"]) == 0
    assert json.loads(baseline_path.read_text()) == {}
    assert checker.main([]) == 0


def test_write_baseline_refuses_increase_without_flag(tmp_path, monkeypatch):
    baseline_path = _point_at(tmp_path, monkeypatch, "int f();\n", {})
    assert checker.main(["--write-baseline"]) == 1
    assert checker.main(["--write-baseline", "--allow-increase"]) == 0
    assert json.loads(baseline_path.read_text()) == {"src/layer_0_core/a.hpp": 1}


def test_an_addons_entries_live_in_its_own_baseline(tmp_path, monkeypatch):
    """An add-on's files are ratcheted beside its code, never in the core baseline."""
    baseline_path = _point_at(tmp_path, monkeypatch, "int f();\n", {})
    addon_directory = checker.SOURCE_ROOT / "addons" / "widget"
    addon_directory.mkdir(parents=True)
    (addon_directory / "w.hpp").write_text("int w();\n")
    assert checker.main(["--write-baseline", "--allow-increase"]) == 0
    assert json.loads(baseline_path.read_text()) == {"src/layer_0_core/a.hpp": 1}
    addon_baseline = addon_directory / checker.BASELINE_PATH.name
    assert json.loads(addon_baseline.read_text()) == {"src/addons/widget/w.hpp": 1}
    assert checker.main([]) == 0


def test_real_tree_matches_baseline():
    current = checker.undocumented_counts(checker.measure_all())
    assert checker.compare(current, checker.load_baseline()) == ([], [])
