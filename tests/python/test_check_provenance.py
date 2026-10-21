"""tests/python/test_check_provenance.py — Unit tests for the provenance-signal ratchet.

Covers ``scripts/repo/check_provenance.py`` and its ``provenance_check`` core:
the core is extensible (a new reference is only a data file, a new foreign
language only a subclass), each rule fires on a fixture that
spells its signal, and stays quiet where the signal is legitimate (comments and
strings for code rules, pointers for ``**``, ``#endif``, ordinary English for
the phrases); the allowlist and its staleness check; the export scope
(``.releaseignore``, vendored text, vault regions); the baseline comparison in
both directions, including stale entries; add-on baselines; and the guarantee
that the real tree matches its baseline.

This file spells forbidden text on purpose and is allowlisted for every rule
(PROVENANCE.md § 5.4). Do not "clean" it.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "scripts" / "repo"
SCRIPT_PATH = SCRIPT_DIRECTORY / "check_provenance.py"

# The core is a package beside its command line under scripts/repo, not part
# of q3: put that directory on the path and import it as itself.
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from provenance_check import core  # noqa: E402
from provenance_check.core import (  # noqa: E402
    ForeignConstruct,
    ForeignLanguage,
    ForeignSyntaxRule,
    ScreenedPattern,
    SourceFile,
    count,
    identifier_words,
    stale_allowlist_entries,
)


def _load_checker():
    """Import the checker script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("check_provenance", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()
REFERENCES = core.load_reference_sources(checker.REFERENCES_DIRECTORY, checker.FOREIGN_LANGUAGES)
RULES = core.build_rules(REFERENCES)
RULE_NAMES = {rule.name for rule in RULES}


def _scan_text(relative_path, text, allowlist=(), rules=None):
    """Every finding in one fixture text."""
    scanner = core.Scanner(rules or RULES, list(allowlist))
    return scanner.scan_file(SourceFile(relative_path, text))


def _keys(relative_path, text, allowlist=(), rules=None):
    """The baseline keys found in one fixture text, with their counts."""
    return count(_scan_text(relative_path, text, allowlist, rules)).get(relative_path, {})


def _real_allowlist():
    return core.load_allowlist(checker.ALLOWLIST_PATH, RULE_NAMES)


# --------------------------------------------------------------------------
# The core: references are data, languages are subclasses
# --------------------------------------------------------------------------


def test_rules_come_from_the_references():
    """One phrase rule, one syntax rule per language (shared by both references), one identifier rule per listed set."""
    assert [reference.key for reference in REFERENCES] == ["berkeleygw", "quantum_espresso"]
    assert [rule.name for rule in RULES] == [
        "red_flag_phrase",
        "fortran_syntax",
        "quantum_espresso_identifier",
    ]


def _write_reference(directory, key, **overrides):
    data = {
        "key": key,
        "names_in_prose": ["ToyCode"],
        "languages": ["fortran"],
        "identifiers": ["zork_table", "zork"],
    }
    data.update(overrides)
    (directory / f"{key}.json").write_text(json.dumps(data))


def test_a_new_reference_is_only_a_data_file(tmp_path):
    _write_reference(tmp_path, "toy_code")
    references = core.load_reference_sources(tmp_path, checker.FOREIGN_LANGUAGES)
    rules = core.build_rules(references)
    assert [rule.name for rule in rules] == [
        "red_flag_phrase",
        "fortran_syntax",
        "toy_code_identifier",
    ]
    source = "// Ported from ToyCode.\nint zork_table_;\nint zork;\nint becsum;\n"
    assert _keys("src/a.hpp", source, rules=rules) == {
        "red_flag_phrase: ported from": 1,
        "toy_code_identifier: zork": 1,
        "toy_code_identifier: zork_table": 1,
    }


def test_a_reference_file_must_be_named_after_its_key(tmp_path):
    _write_reference(tmp_path, "toy_code")
    (tmp_path / "toy_code.json").rename(tmp_path / "other.json")
    with pytest.raises(ValueError, match="key"):
        core.load_reference_sources(tmp_path, checker.FOREIGN_LANGUAGES)


def test_a_reference_naming_an_unknown_language_is_refused(tmp_path):
    _write_reference(tmp_path, "toy_code", languages=["cobol"])
    with pytest.raises(ValueError, match="cobol"):
        core.load_reference_sources(tmp_path, checker.FOREIGN_LANGUAGES)


class _ToyLanguage(ForeignLanguage):
    """A made-up foreign language whose ``:=`` would be a translation in Python, not in C++."""

    name = "toy"

    def constructs(self):
        assignment = ScreenedPattern(":=", (":=",), re.compile(r":="))
        return (ForeignConstruct(assignment, frozenset({"python"})),)


def test_a_new_foreign_language_is_only_a_subclass():
    rule = ForeignSyntaxRule(_ToyLanguage())
    assert rule.name == "toy_syntax"
    assert _keys("src/py/q3/a.py", "x := 1\n", rules=[rule]) == {"toy_syntax: :=": 1}
    assert _keys("src/a.cpp", "x := 1;\n", rules=[rule]) == {}
    assert _keys("docs/a.md", "x := 1\n", rules=[rule]) == {}


# --------------------------------------------------------------------------
# Rule 1: red-flag phrases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "signal"),
    [
        ("// Transcribed from PW/src/addusdens.f90.", "transcribed from"),
        ("# ported from upflib/qvan2.f90", "ported from"),
        ("/* Translated from the QE routine init_us_1.f90 */", "translated from"),
        ("Adapted from Quantum ESPRESSO's stress code.", "adapted from"),
        ("* copied from QE", "copied from"),
        ("This loop follows newd.f90 line by line.", "follows <file> line by line"),
        ("Based on the implementation in sym_rho.f90.", "based on the implementation in"),
    ],
)
def test_red_flag_phrase_fires(text, signal):
    assert _keys("docs/notes.md", text) == {f"red_flag_phrase: {signal}": 1}


def test_red_flag_phrase_wrapped_across_comment_lines_is_one_finding_at_its_first_line():
    source = "int x;\n// The overlap term is ported\n// from PW/src/s_psi.f90 verbatim.\n"
    findings = _scan_text("src/a.hpp", source)
    assert [(finding.line, finding.signal) for finding in findings] == [(2, "ported from")]


@pytest.mark.parametrize(
    "text",
    [
        "Energies are translated from Hartree to Rydberg on output.",
        "A record imported from x.f90 projections carries no overlaps.",
        "Supported from QE 7.0 on.",
        "Implements Eq. (11); cross-checked numerically against QE 7.5.",
        "Adapted from the paper; validated against QE.",
        "Bessel recurrence transcribed from <file> is forbidden.",
    ],
)
def test_red_flag_phrase_ignores_ordinary_wording(text):
    assert _keys("docs/notes.md", text) == {}


# --------------------------------------------------------------------------
# Rule 2: Fortran syntax in code
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "signal"),
    [
        ("if (a .and. b) {}", ".and."),
        ("if (a .OR. b) {}", ".or."),
        ("if (a .eq. b) {}", ".eq."),
        ("if (a .ne. b) {}", ".ne."),
        ("if (a .lt. b) {}", ".lt."),
        ("double y = dble(n);", "dble("),
        ("enddo;", "enddo"),
        ("end do;", "end do"),
        ("endif;", "endif"),
        ("end if;", "end if"),
        ("auto column = a(:);", "(:) slice"),
        ("auto row = a(:, 2);", "(:) slice"),
        ("double square = x**2;", "** power"),
        ("double root = x ** (1.0 / 3.0);", "** power"),
        ("double scaled = 2**n;", "** power"),
        ("double sum_squared = (a + b)**n;", "** power"),
    ],
)
def test_fortran_syntax_fires_in_cpp_code(line, signal):
    assert _keys("src/a.cpp", f"void f() {{\n{line}\n}}\n") == {f"fortran_syntax: {signal}": 1}


def test_fortran_syntax_ignores_cpp_comments_strings_preprocessor_and_pointers():
    source = """
    #ifdef Q3_WITH_MPI
    #endif
    // if (a .and. b) dble(x) enddo a(:) x**2
    /* end do
       end if */
    const char* text = "a .and. b, dble(x), a(:), x**2";
    const char* raw = R"delimiter(a .or. b
    enddo)delimiter";
    int main(int argc, char** argv) { return **argv; }
    void g(double **matrix, int (**handlers)(int));
    auto it = ::std::begin(v);
    """
    assert _keys("src/a.cpp", source) == {}


@pytest.mark.parametrize(
    ("line", "signal"),
    [
        ("ok = a .and. b", ".and."),
        ("y = dble(n)", "dble("),
        ("enddo = 1", "enddo"),
    ],
)
def test_fortran_syntax_fires_in_python_code(line, signal):
    assert _keys("src/py/q3/a.py", line + "\n") == {f"fortran_syntax: {signal}": 1}


def test_fortran_syntax_in_python_ignores_power_end_if_comments_and_docstrings():
    source = '''
"""Docstring: a .and. b, dble(x), enddo."""
square = x**2
stop = end if end is not None else len(values)
# if (a .and. b) dble(x) enddo
message = "a .or. b"
'''
    assert _keys("src/py/q3/a.py", source) == {}


# --------------------------------------------------------------------------
# Rule 3: Quantum ESPRESSO identifiers in C++ code
# --------------------------------------------------------------------------


def test_quantum_espresso_identifier_fires_on_declarations_parameters_and_members():
    source = """
    class BecsumTable {
        std::vector<double> becsum_;
        void accumulate(const Matrix& vkb, int nkb);
        double compute_becsum() const;
        std::vector<int> igk_k;
    };
    """
    assert _keys("src/a.hpp", source) == {
        "quantum_espresso_identifier: becsum": 3,
        "quantum_espresso_identifier: igk_k": 1,
        "quantum_espresso_identifier: nkb": 1,
        "quantum_espresso_identifier: vkb": 1,
    }


def test_quantum_espresso_identifier_ignores_comments_strings_python_and_longer_words():
    cpp_source = """
    // becsum, vkb and evc are the Quantum ESPRESSO names for these arrays.
    const char* key = "nosym_evc";
    double goerigk_dispersion = 0.0;
    """
    assert _keys("src/a.hpp", cpp_source) == {}
    assert _keys("tests/utils/qe/reader.py", "becsum = read_record(stream)\n") == {}


def test_identifier_words_split_at_underscores_and_camel_case():
    assert identifier_words("BecsumTable_igk_kHTTPValue2") == [
        "becsum",
        "table",
        "igk",
        "k",
        "http",
        "value",
        "2",
    ]


# --------------------------------------------------------------------------
# Allowlist
# --------------------------------------------------------------------------


def test_allowlist_excuses_only_the_named_path_and_rule():
    entry = core.AllowlistEntry(
        "src/io/input/parser/", frozenset({"quantum_espresso_identifier"}), "format"
    )
    source = "int nkb;\n// ported from PW/src/a.f90\n"
    assert _keys("src/io/input/parser/qe.hpp", source, [entry]) == {
        "red_flag_phrase: ported from": 1
    }
    assert "quantum_espresso_identifier: nkb" in _keys("src/io/other.hpp", source, [entry])


def test_allowlist_rejects_an_entry_without_a_reason(tmp_path):
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps([{"path": "a.hpp", "rules": ["fortran_syntax"], "reason": " "}]))
    with pytest.raises(ValueError):
        core.load_allowlist(path, RULE_NAMES)


def test_real_allowlist_entries_all_give_a_reason_and_exist():
    allowlist = _real_allowlist()
    assert allowlist
    assert stale_allowlist_entries(allowlist, REPOSITORY_ROOT) == []


# --------------------------------------------------------------------------
# Scope, baseline and the command line
# --------------------------------------------------------------------------


def _point_at(tmp_path, monkeypatch, files, baseline, allowlist=()):
    """Point the checker at a fake tree (not a git work tree), baseline and allowlist."""
    for relative_path, text in files.items():
        path = tmp_path / "tree" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(json.dumps(list(allowlist)))
    monkeypatch.setattr(checker, "REPOSITORY_ROOT", tmp_path / "tree")
    monkeypatch.setattr(checker, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(checker, "ALLOWLIST_PATH", allowlist_path)
    return baseline_path


def test_scan_skips_what_the_export_leaves_out(tmp_path, monkeypatch):
    signal = "int becsum;\n"
    _point_at(
        tmp_path,
        monkeypatch,
        {
            ".releaseignore": "private/\ntools/unshipped_tool.py\nexamples/drafts/\n",
            "src/private/notes.cpp": signal,
            "tools/unshipped_tool.py": "# ported from a/b.f90\n",
            "examples/drafts/x.cpp": signal,
            "schemas/esse/x.md": "ported from a/b.f90\n",
            "src/kept.hpp": signal,
            "src/vaulted.hpp": '// vault:begin reason="x"\n' + signal + "// vault:end\n",
            "src/binary.cpp": "int becsum;\0",
        },
        {},
    )
    assert sorted(
        {finding.path for finding in core.Scanner(RULES, []).scan(checker.REPOSITORY_ROOT)}
    ) == ["src/kept.hpp"]


def test_main_fails_on_a_new_finding_and_prints_its_line(tmp_path, monkeypatch, capsys):
    _point_at(tmp_path, monkeypatch, {"src/a.cpp": "int x;\ndouble y = dble(x);\n"}, {})
    assert checker.main(["--check"]) == 1
    output = capsys.readouterr().out
    assert "src/a.cpp: fortran_syntax: dble( 0 -> 1" in output
    assert "src/a.cpp:2: fortran_syntax: dble(" in output


def test_main_fails_on_a_stale_baseline_entry(tmp_path, monkeypatch, capsys):
    baseline = {"src/a.hpp": {"quantum_espresso_identifier: becsum": 1}}
    baseline_path = _point_at(tmp_path, monkeypatch, {"src/a.hpp": "int x;\n"}, baseline)
    assert checker.main(["--check"]) == 1
    assert "Stale baseline entries" in capsys.readouterr().out
    assert checker.main(["--write-baseline"]) == 0
    assert json.loads(baseline_path.read_text()) == {}
    assert checker.main(["--check"]) == 0


def test_main_fails_on_a_stale_allowlist_entry(tmp_path, monkeypatch, capsys):
    allowlist = [{"path": "src/gone.hpp", "rules": ["fortran_syntax"], "reason": "removed"}]
    _point_at(tmp_path, monkeypatch, {"src/a.hpp": "int x;\n"}, {}, allowlist)
    assert checker.main([]) == 1
    assert "src/gone.hpp" in capsys.readouterr().out


def test_write_baseline_refuses_increase_without_flag(tmp_path, monkeypatch):
    baseline_path = _point_at(tmp_path, monkeypatch, {"src/a.hpp": "int nkb;\n"}, {})
    assert checker.main(["--write-baseline"]) == 1
    assert json.loads(baseline_path.read_text()) == {}
    assert checker.main(["--write-baseline", "--allow-increase"]) == 0
    assert json.loads(baseline_path.read_text()) == {
        "src/a.hpp": {"quantum_espresso_identifier: nkb": 1}
    }


def test_an_addons_entries_live_in_its_own_baseline(tmp_path, monkeypatch):
    """An add-on's findings are ratcheted beside its code, never in the core baseline."""
    baseline_path = _point_at(
        tmp_path,
        monkeypatch,
        {
            ".releaseignore": "addons/\n",
            "src/a.hpp": "int nkb;\n",
            "src/addons/widget/w.hpp": "int nkb;\n",
            "src/addons/README.md": "Ported from a/b.f90.\n",
            "tests/unit/addons/widget/test_w.cpp": "int nkb;\n",
        },
        {},
    )
    assert checker.main(["--write-baseline", "--allow-increase"]) == 0
    assert json.loads(baseline_path.read_text()) == {
        "src/a.hpp": {"quantum_espresso_identifier: nkb": 1}
    }
    addon_baseline = checker.REPOSITORY_ROOT / "src/addons/widget" / checker.BASELINE_PATH.name
    assert sorted(json.loads(addon_baseline.read_text())) == [
        "src/addons/widget/w.hpp",
        "tests/unit/addons/widget/test_w.cpp",
    ]
    assert checker.main([]) == 0


def test_compare_reports_increase_and_decrease():
    increases, decreases = core.Baseline.compare(
        {"src/a.hpp": {"fortran_syntax: enddo": 2}, "src/new.hpp": {"red_flag_phrase: x": 1}},
        {"src/a.hpp": {"fortran_syntax: enddo": 3}},
    )
    assert increases == ["src/new.hpp: red_flag_phrase: x 0 -> 1"]
    assert decreases == ["src/a.hpp: fortran_syntax: enddo 3 -> 2"]


def test_real_tree_matches_baseline():
    findings = core.Scanner(RULES, _real_allowlist()).scan(REPOSITORY_ROOT)
    baseline = core.Baseline(checker.BASELINE_PATH, REPOSITORY_ROOT).load()
    increases, decreases = core.Baseline.compare(count(findings), baseline)
    assert increases == []
    assert decreases == []
