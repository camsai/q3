#!/usr/bin/env python3
"""
Doc-comment style checker and reformatter for Q3 C++ sources.

Enforces the rules in AGENTS-generic.md § Comment Style:

  1. The opening /** must appear alone on its line — no content after it.
  2. The closing */ must appear alone on its line — no content before it.
  3. Multiline doc comments must use /** ... */, not a run of /// lines
     (/// is for single-line doc comments only).
  4. Bare /* ... */ must not be used for documentation; use /** ... */.
     The inline parameter-name idiom (`f(/*data*/ nullptr)`,
     `g(/*save_wavefunctions=*/false)`) is NOT documentation and is allowed.
  5. Multiline comments must use /** ... */, not a run of full-line // lines
     (// is for single-line inline implementation comments only; a trailing
     // after code is always fine and never grouped).

Allowed:
    /**
     * Description.
     */

    /// Single-line description.

    // Single inline implementation note.

    compute();  // trailing notes are fine, even on consecutive lines

    reduce(ComplexValue* /*data*/, int /*count*/) {}

Not allowed:
    /** Description.        <- rule 1: content on the /** line
     */

    /** Description. */     <- rule 1: single-line /** block

    /**
     * Description. */      <- rule 2: closing */ not on its own line

    /// Description that    <- rule 3: multiline /// run
    /// wraps to two lines.

    /* Description. */      <- rule 4: bare block comment as documentation

    // Description that     <- rule 5: multiline // run
    // wraps to two lines.

Rule 5 only groups consecutive full-line // comments at the SAME indentation
(differently indented neighbors are separate comments about different code),
and a line whose text contains */ never joins a run (wrapping it in a block
comment would end the block early and break compilation).

Usage:
    # Check only (default):
    python scripts/repo/check_doc_comment_style.py src/**/*.hpp src/**/*.cpp

    # Auto-fix violations in-place:
    python scripts/repo/check_doc_comment_style.py --fix src/**/*.hpp src/**/*.cpp

Rules 1-3 and 5 are auto-fixable. Rule 4 is reported only: converting a bare
block comment safely needs a human to confirm it is documentation rather than
commented-out code.

Exit code 0 means no violations (or all fixed); non-zero means violations remain.
"""

import re
import sys

# Rule 1: /** followed immediately by non-whitespace content on the same line.
# Ignores: /**/  /***  /**\n  (all valid or caught by rule 2)
OPEN_VIOLATION_PATTERN = re.compile(r"/\*\*[^*/\n]")

# Rule 2: a doc-comment line that has content before the closing */.
# Matches lines like:  " * Description. */"  or  " * text */"
# Does NOT match:      " */"  (correct — only whitespace before */)
CLOSE_VIOLATION_PATTERN = re.compile(r"^\s+\*\s+.+\*/\s*$")

# Rule 3: a line whose first non-whitespace content is a /// doc comment.
# Excludes ///< (Doxygen trailing member docs), which are single-line by
# construction and legitimately appear on consecutive lines for adjacent
# struct members.
SLASH_DOC_PATTERN = re.compile(r"^(\s*)///(?!<)(.*)$")

# Rule 4: a bare /* that is neither /** (doc) nor part of */.
BARE_BLOCK_PATTERN = re.compile(r"(?<!/)/\*(?!\*)")

# Rule 5: a line whose first non-whitespace content is a plain // comment.
# Excludes /// (rule 3's territory) and //! (Doxygen, unused here but distinct).
#   Group 1: leading whitespace (the indent runs are grouped by)
#   Group 2: comment body, without the single space after //
FULL_LINE_COMMENT_PATTERN = re.compile(r"^(\s*)//(?![/!])\s?(.*)$")

# The inline parameter-name / named-argument idiom that rule 4 must not flag:
# an identifier, optionally followed by '=', as the entire comment body.
PARAMETER_NAME_IDIOM_PATTERN = re.compile(r"^\s*[A-Za-z_]\w*\s*=?\s*$")

# Used by the fixer to extract the parts of a /** ... */ opening line.
#   Group 1: leading whitespace
#   Group 2: content after /**
#   Group 3: closing */ if on the same line (single-line form)
OPEN_FIX_PATTERN = re.compile(r"^(\s*)/\*\*\s*(.*?)(\s*\*/)?\s*$")

# Used by the fixer to split a content+closing line.
#   Group 1: leading whitespace + " * "
#   Group 2: the content text (without trailing " */")
CLOSE_FIX_PATTERN = re.compile(r"^(\s+\*\s+)(.*?)\s*\*/\s*$")

RULE_DESCRIPTIONS = {
    "open": "/** must be followed by a newline, not content",
    "close": "*/ must be on its own line, not after content",
    "slash_block": "multiline doc comment must use /** ... */, not a run of ///",
    "bare_block": "bare /* ... */ must not be used for documentation; use /** ... */",
    "slash_slash_block": "multiline comment must use /** ... */, not a run of // lines",
}


class Violation:
    """
    A single doc-comment style violation.
    """

    def __init__(self, line_number: int, line_content: str, kind: str) -> None:
        self.line_number = line_number
        self.line_content = line_content
        self.kind = kind  # one of RULE_DESCRIPTIONS


def fix_open_line(line: str) -> list[str]:
    """
    Reformat a malformed opening line.

    '    /** Description. */' -> ['    /**\\n', '     * Description.\\n', '     */\\n']
    '    /** Description.'    -> ['    /**\\n', '     * Description.\\n']
    """
    match = OPEN_FIX_PATTERN.match(line.rstrip())
    if not match:
        return [line]

    indent = match.group(1)
    content = match.group(2).strip()
    closing = match.group(3)

    result = [f"{indent}/**\n"]
    if content:
        result.append(f"{indent} * {content}\n")
    if closing:
        result.append(f"{indent} */\n")
    return result


def fix_close_line(line: str) -> list[str]:
    """
    Reformat a line where content and closing */ share a line.

    '     * Description. */' -> ['     * Description.\\n', '     */\\n']
    """
    match = CLOSE_FIX_PATTERN.match(line.rstrip())
    if not match:
        return [line]

    prefix = match.group(1)  # e.g. "     * "
    content = match.group(2)  # e.g. "Description."
    # Derive indent from prefix (everything before "* ")
    indent = prefix[: prefix.index("*")]

    return [f"{prefix}{content}\n", f"{indent}*/\n"]


def slash_doc_content(line: str) -> str | None:
    """
    Body text of a /// doc-comment line, or None if the line is not one.

    Exactly one space after /// is consumed, so deeper indentation inside the
    comment (code samples, nested bullets) survives the conversion.
    """
    match = SLASH_DOC_PATTERN.match(line.rstrip("\n"))
    if not match:
        return None
    content = match.group(2)
    return content[1:] if content.startswith(" ") else content


def fix_slash_block(block_lines: list[str]) -> list[str]:
    """
    Convert a run of /// lines into an equivalent /** ... */ block.

    ['    /// One\\n', '    /// Two\\n']
        -> ['    /**\\n', '     * One\\n', '     * Two\\n', '     */\\n']
    """
    indent = SLASH_DOC_PATTERN.match(block_lines[0].rstrip("\n")).group(1)
    result = [f"{indent}/**\n"]
    for line in block_lines:
        content = slash_doc_content(line)
        result.append(f"{indent} * {content}\n" if content else f"{indent} *\n")
    result.append(f"{indent} */\n")
    return result


def full_line_comment_parts(line: str) -> tuple[str, str] | None:
    """
    (indent, body) of a full-line // comment, or None if the line is not one.

    A line whose body contains */ is deliberately not a full-line comment for
    rule 5's purposes: wrapping it inside a /** ... */ block would terminate
    the block early and break compilation, so such a line never joins a run.
    """
    match = FULL_LINE_COMMENT_PATTERN.match(line.rstrip("\n"))
    if not match or "*/" in match.group(2):
        return None
    return match.group(1), match.group(2)


def fix_slash_slash_block(block_lines: list[str]) -> list[str]:
    """
    Convert a run of full-line // comments into an equivalent /** ... */ block.

    ['    // One\\n', '    // Two\\n']
        -> ['    /**\\n', '     * One\\n', '     * Two\\n', '     */\\n']
    """
    indent = full_line_comment_parts(block_lines[0])[0]
    result = [f"{indent}/**\n"]
    for line in block_lines:
        content = full_line_comment_parts(line)[1]
        result.append(f"{indent} * {content}\n" if content else f"{indent} *\n")
    result.append(f"{indent} */\n")
    return result


def is_documentation_block_comment(line: str, match_start: int) -> bool:
    """
    Whether a bare /* at the given position is documentation (rule 4) rather
    than the allowed inline parameter-name idiom.

    A comment that closes on the same line and whose entire body is an
    identifier (optionally with a trailing '=') is the idiom used for unnamed
    parameters and named arguments — `f(int /*count*/)`, `g(/*flag=*/true)` —
    and is not documentation.
    """
    remainder = line[match_start + 2 :]
    closing_index = remainder.find("*/")
    if closing_index < 0:
        return True  # spans lines: a prose block, not the inline idiom
    return not PARAMETER_NAME_IDIOM_PATTERN.match(remainder[:closing_index])


def process_file(path: str, fix: bool) -> list[Violation]:
    """
    Check (and optionally fix) a single file. Returns the violations found.
    """
    try:
        with open(path, encoding="utf-8") as source_file:
            original_lines = source_file.readlines()
    except (OSError, UnicodeDecodeError) as error:
        print(f"error: could not read {path}: {error}", file=sys.stderr)
        return []

    violations: list[Violation] = []
    new_lines: list[str] = []
    inside_block_comment = False
    line_index = 0

    while line_index < len(original_lines):
        line = original_lines[line_index]
        line_number = line_index + 1

        # Inside a /* ... */ or /** ... */ block only rule 2 applies; nothing
        # else on these lines is code, so /* and /// occurrences are text.
        if inside_block_comment:
            if CLOSE_VIOLATION_PATTERN.search(line):
                violations.append(Violation(line_number, line.rstrip(), "close"))
                new_lines.extend(fix_close_line(line) if fix else [line])
            else:
                new_lines.append(line)
            if "*/" in line:
                inside_block_comment = False
            line_index += 1
            continue

        # Rule 3: a run of two or more /// lines is a multiline doc comment.
        if slash_doc_content(line) is not None:
            block_end = line_index
            while (
                block_end < len(original_lines)
                and slash_doc_content(original_lines[block_end]) is not None
            ):
                block_end += 1
            block_lines = original_lines[line_index:block_end]
            if len(block_lines) > 1:
                violations.append(Violation(line_number, line.rstrip(), "slash_block"))
                new_lines.extend(fix_slash_block(block_lines) if fix else block_lines)
            else:
                new_lines.extend(block_lines)
            line_index = block_end
            continue

        # Rule 5: a run of two or more full-line // comments at the same
        # indentation is a multiline comment. Same indentation only:
        # differently indented neighbors are separate comments about
        # different code, not one paragraph.
        comment_parts = full_line_comment_parts(line)
        if comment_parts is not None:
            indent = comment_parts[0]
            block_end = line_index
            while block_end < len(original_lines):
                next_parts = full_line_comment_parts(original_lines[block_end])
                if next_parts is None or next_parts[0] != indent:
                    break
                block_end += 1
            block_lines = original_lines[line_index:block_end]
            if len(block_lines) > 1:
                violations.append(Violation(line_number, line.rstrip(), "slash_slash_block"))
                new_lines.extend(fix_slash_slash_block(block_lines) if fix else block_lines)
            else:
                new_lines.extend(block_lines)
            line_index = block_end
            continue

        if OPEN_VIOLATION_PATTERN.search(line):
            violations.append(Violation(line_number, line.rstrip(), "open"))
            new_lines.extend(fix_open_line(line) if fix else [line])
        else:
            bare_match = BARE_BLOCK_PATTERN.search(line)
            if bare_match and is_documentation_block_comment(line, bare_match.start()):
                violations.append(Violation(line_number, line.rstrip(), "bare_block"))
            new_lines.append(line)

        # A /** or /* that does not close on this line opens a block comment.
        # Rule 1 fixes rewrite the line, but the open/closed state is the same
        # either way, so this test is valid for both branches above.
        last_open = max(line.rfind("/*"), 0) if "/*" in line else -1
        if last_open >= 0 and "*/" not in line[last_open:]:
            inside_block_comment = True
        line_index += 1

    if not fix:
        for violation in violations:
            print(f"{path}:{violation.line_number}: error: {RULE_DESCRIPTIONS[violation.kind]}")
            print(f"  {violation.line_content}")
            print()

    fixable = [violation for violation in violations if violation.kind != "bare_block"]
    if fix and fixable:
        with open(path, "w", encoding="utf-8") as destination_file:
            destination_file.writelines(new_lines)
        print(f"fixed {len(fixable)} violation(s) in {path}")

    return violations


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: check_doc_comment_style.py [--fix] <file> [<file> ...]",
            file=sys.stderr,
        )
        return 1

    fix_mode = "--fix" in args
    paths = [argument for argument in args if argument != "--fix"]

    all_violations: list[Violation] = []
    for path in paths:
        all_violations.extend(process_file(path, fix=fix_mode))

    # Rule 4 has no auto-fix, so it still fails the run in --fix mode.
    remaining = [
        violation for violation in all_violations if not fix_mode or violation.kind == "bare_block"
    ]
    if remaining:
        print(
            f"\nFound {len(remaining)} doc-comment style violation(s). "
            "Run with --fix to reformat rules 1-3, or see "
            "AGENTS-generic.md § Comment Style for the rules."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
