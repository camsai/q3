"""
Tests for scripts/repo/check_doc_comment_style.py.

This script gates every C++ commit through pre-commit, and its --fix mode
rewrites source files in place, so both the detection and the rewriting are
covered here: a false positive blocks unrelated work, and a bad fix silently
mangles documentation.

Rules under test (AGENTS-generic.md § Comment Style):
  1. /** alone on its line
  2. */ alone on its line
  3. multiline doc comments use /** */, not a run of ///
  4. bare /* */ is not documentation (report-only; the inline
     parameter-name idiom is explicitly allowed)
  5. multiline comments use /** */, not a run of full-line // lines
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "repo" / "check_doc_comment_style.py"
)

_spec = importlib.util.spec_from_file_location("check_doc_comment_style", SCRIPT_PATH)
checker = importlib.util.module_from_spec(_spec)
sys.modules["check_doc_comment_style"] = checker
_spec.loader.exec_module(checker)


def check(tmp_path, source):
    """Run the checker (no fix) on `source`, returning the violation kinds found."""
    path = tmp_path / "sample.hpp"
    path.write_text(source, encoding="utf-8")
    return [violation.kind for violation in checker.process_file(str(path), fix=False)]


def fix(tmp_path, source):
    """Run the checker in --fix mode on `source`, returning the rewritten text."""
    path = tmp_path / "sample.hpp"
    path.write_text(source, encoding="utf-8")
    checker.process_file(str(path), fix=True)
    return path.read_text(encoding="utf-8")


# ── Rule 3: multiline /// runs ──────────────────────────────────────


def test_multiline_slash_run_is_flagged(tmp_path):
    assert check(tmp_path, "/// One line\n/// Second line\nint x;\n") == ["slash_block"]


def test_single_slash_line_is_allowed(tmp_path):
    assert check(tmp_path, "/// Just one line.\nint x;\n") == []


def test_separate_single_slash_lines_are_allowed(tmp_path):
    # Two single-line doc comments on different declarations must not be
    # merged into one "multiline" run by the blank line between them.
    source = "/// Docs for x.\nint x;\n\n/// Docs for y.\nint y;\n"
    assert check(tmp_path, source) == []


def test_trailing_member_docs_are_allowed(tmp_path):
    # ///< on consecutive struct members is the Doxygen trailing form, not a
    # multiline block -- these appear back-to-back in real headers.
    source = "struct S {\n    int a; ///< first\n    int b; ///< second\n};\n"
    assert check(tmp_path, source) == []


def test_multiline_slash_run_is_converted(tmp_path):
    result = fix(tmp_path, "    /// One line\n    /// Second line\n    int x;\n")
    assert result == ("    /**\n     * One line\n     * Second line\n     */\n    int x;\n")


def test_conversion_preserves_blank_doc_lines_and_inner_indentation(tmp_path):
    source = "/// Summary.\n///\n///   indented sample\nint x;\n"
    assert fix(tmp_path, source) == ("/**\n * Summary.\n *\n *   indented sample\n */\nint x;\n")


def test_converted_output_is_itself_clean(tmp_path):
    # The fixer must not produce output that the checker then rejects.
    converted = fix(tmp_path, "/// One\n/// Two\nint x;\n")
    assert check(tmp_path, converted) == []


# ── Rule 5: multiline // runs ───────────────────────────────────────


def test_multiline_full_line_comment_run_is_flagged(tmp_path):
    assert check(tmp_path, "    // One line\n    // Second line\nint x;\n") == ["slash_slash_block"]


def test_single_full_line_comment_is_allowed(tmp_path):
    assert check(tmp_path, "    // Just one inline note.\nint x;\n") == []


def test_trailing_comments_never_group(tmp_path):
    # Consecutive statements each carrying a trailing // note are not a
    # multiline comment -- the comments follow code, they do not open lines.
    source = "compute();  // first note\nreduce();  // second note\n"
    assert check(tmp_path, source) == []


def test_differently_indented_neighbors_are_separate_comments(tmp_path):
    # A comment about an outer block followed by a deeper-indented comment
    # about inner code is two separate single-line comments, not one
    # paragraph -- grouping is by identical indentation.
    assert check(tmp_path, "// outer\n    // inner\nint x;\n") == []


def test_line_containing_block_close_never_joins_a_run(tmp_path):
    # Wrapping a line whose text contains */ inside /** ... */ would end the
    # block early and break compilation, so such a line breaks the run.
    source = "// mentions a */ closer\n// second line\nint x;\n"
    assert check(tmp_path, source) == []


def test_full_line_run_fix_produces_block_comment(tmp_path):
    fixed = fix(tmp_path, "    // One\n    // Two\nint x;\n")
    assert fixed == "    /**\n     * One\n     * Two\n     */\nint x;\n"


def test_full_line_run_fix_preserves_blank_comment_lines(tmp_path):
    fixed = fix(tmp_path, "// One\n//\n// Three\nint x;\n")
    assert fixed == "/**\n * One\n *\n * Three\n */\nint x;\n"


def test_full_line_comments_inside_block_comment_are_text(tmp_path):
    # // lines inside an existing /** ... */ block are comment text, not
    # comment syntax, and must not be re-flagged or rewritten.
    source = "/**\n * Header.\n */\nint x;\n"
    assert check(tmp_path, source) == []


def test_slash_run_still_reports_rule_3_not_rule_5(tmp_path):
    # /// runs stay rule 3's finding; the rules are disjoint by pattern.
    assert check(tmp_path, "/// One\n/// Two\nint x;\n") == ["slash_block"]


# ── Rule 4: bare /* */ ──────────────────────────────────────────────


def test_unnamed_parameter_idiom_is_allowed(tmp_path):
    source = "void f(int /*count*/, double* /*data*/) {}\n"
    assert check(tmp_path, source) == []


def test_named_argument_idiom_is_allowed(tmp_path):
    source = "    save(path, /*save_wavefunctions=*/false);\n"
    assert check(tmp_path, source) == []


def test_bare_block_comment_as_documentation_is_flagged(tmp_path):
    assert check(tmp_path, "/* Describes the thing. */\nint x;\n") == ["bare_block"]


def test_multiline_bare_block_comment_is_flagged(tmp_path):
    assert check(tmp_path, "/* Describes\n * the thing.\n */\nint x;\n") == ["bare_block"]


def test_bare_block_is_reported_but_not_rewritten(tmp_path):
    # Rule 4 has no auto-fix: converting a bare block needs a human to confirm
    # it is documentation and not commented-out code.
    source = "/* Describes the thing. */\nint x;\n"
    assert fix(tmp_path, source) == source


# ── Rules 1-2: /** ... */ internal formatting (pre-existing behavior) ──


def test_content_on_open_line_is_flagged(tmp_path):
    assert check(tmp_path, "/** Description.\n */\nint x;\n") == ["open"]


def test_single_line_doc_block_is_flagged(tmp_path):
    assert check(tmp_path, "/** Description. */\nint x;\n") == ["open"]


def test_content_before_closing_is_flagged(tmp_path):
    assert check(tmp_path, "/**\n * Description. */\nint x;\n") == ["close"]


def test_well_formed_block_is_allowed(tmp_path):
    assert check(tmp_path, "/**\n * Description.\n */\nint x;\n") == []


@pytest.mark.parametrize(
    "source,expected",
    [
        ("/** Description. */\nint x;\n", "/**\n * Description.\n */\nint x;\n"),
        ("/**\n * Description. */\nint x;\n", "/**\n * Description.\n */\nint x;\n"),
    ],
)
def test_block_formatting_is_fixed(tmp_path, source, expected):
    assert fix(tmp_path, source) == expected


# ── Comment bodies are not scanned as code ──────────────────────────


def test_rules_do_not_fire_on_text_inside_a_doc_block(tmp_path):
    # A doc block that *mentions* /* or /// in prose is describing syntax,
    # not violating the rules.
    source = "/**\n * Do not use /* here, and /// is single-line only.\n */\nint x;\n"
    assert check(tmp_path, source) == []


# ── Real codebase invariant ─────────────────────────────────────────


def test_repository_sources_are_clean():
    """
    Every tracked C++ source must satisfy all four rules.

    This is the regression guard that makes the rule real: it fails in plain
    pytest (not only under pre-commit), so a violation cannot reach main via a
    developer who has not installed the hooks.
    """
    # From this test file, not from the script: the script moving must not move the tree it scans.
    repository_root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(
        ["git", "ls-files", "*.hpp", "*.cpp", "*.h", "*.cc", "*.cu", "*.mm"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    # src/generated/ is emitted by scripts/repo/generate_cpp_from_schemas.py; its
    # style is the generator's responsibility, not a reviewer's.
    sources = [path for path in tracked if not path.startswith("src/generated/")]
    assert sources, "expected to find tracked C++ sources"

    offenders = {}
    for relative_path in sources:
        violations = checker.process_file(str(repository_root / relative_path), fix=False)
        if violations:
            offenders[relative_path] = [violation.kind for violation in violations]
    assert not offenders, f"doc-comment style violations: {offenders}"
