#!/usr/bin/env python3
"""Doxygen input filter for the Markdown pages of the docs site (FILTER_PATTERNS).

Doxygen calls it with one file path and reads the page from its output. It changes
only how a page renders on the site, never the file, and keeps every line where it
is, so Doxygen's warnings still name the file's own lines.

- Every page gets a label named after its path from the repository root, lower case,
  with `.md` dropped and every other character that is not a letter or digit written
  as `_`: `FUNCTIONALITY.md` is `functionality`, `docs/io_reference.md` is
  `docs_io_reference`, `docs/tutorials/README.md` is `docs_tutorials_readme`. The
  landing page (docs/mainpage.md) links its pages with `\\subpage <label>`. Doxygen's
  own page names escape upper-case letters on a case-insensitive file system, so they
  differ between a macOS and a Linux build, and a `\\subpage` by file name links the
  page without placing it in the sidebar.
- Every other heading gets the anchor GitHub gives it, behind the page's label
  (`## 2.2 Build from Source` in the README is `readme-22-build-from-source`), and
  a link to a section, `#22-build-from-source` or `../README.md#22-build-from-source`,
  is written as a `\\ref` to that anchor. Doxygen generates no GitHub anchors of its own.
- The repository README is shown as the "User Guide" page. Its GitHub front matter
  is dropped: the centred logo, title and badges, the intro and "If you want to..."
  table that the landing page already gives, and the Contents list, whose
  GitHub-style anchors Doxygen does not generate. The page starts at the first
  numbered section. Any other README.md keeps its front matter.
- In the tutorials index (docs/tutorials/README.md), the lesson links in its table
  become `\\subpage` links, so the lessons are listed under "Tutorials" in the sidebar.
- A `---` rule (after a blank line, outside code blocks) is written as `***`. Doxygen
  1.9 reads a `---` that follows a block quote as a heading underline, and shows a
  heading "</blockquote>".
- A link to a repository file that is not a page (a source file, a schema, a
  diagram) is written as a link to that file on GitHub, in the public repository the
  site is built from.
- GitHub's math, `$...$` inline and `$$...$$` displayed, is written as Doxygen's
  `\\f$...\\f$` and `\\f[...\\f]` (rendered by MathJax), outside code blocks and
  code spans. Doxygen reads a bare `\\frac` as an unknown command.
"""

import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
README_TITLE = "# User Guide"
TUTORIALS_INDEX = "docs/tutorials/README.md"
LANDING_PAGE = "docs/mainpage.md"
LESSON_LINK = re.compile(r"\[([^\]]+)\]\(([0-9]{2}-[A-Za-z0-9-]+\.md)\)")
HEADING = re.compile(r"^(#{1,6}) (.*?)\s*#*\s*$")
SECTION_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(((?:[^()\s#:]*\.md)?)#([^()\s]+)\)")
# Spelled with the clean org, which the site is published from (as docs.yml does).
REPOSITORY_FILE_URL = "https://github.com/camsai/q3/blob/main/"
FILE_LINK = re.compile(r"(?<!!)\]\(([^()\s#:]+)\)")
CODE_SPAN = re.compile(r"(`+).*?\1")
DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$")
# Like GitHub: no space just inside either dollar sign, and no digit after the closing one.
INLINE_MATH = re.compile(r"(?<![\\$])\$(?=\S)([^$]*?\S)\$(?![$0-9])")


def page_label(relative_path: str) -> str:
    stem = relative_path[: -len(".md")] if relative_path.endswith(".md") else relative_path
    return re.sub(r"[^a-z0-9]", "_", stem.lower())


def github_slug(heading_text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading_text)
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def outside_code_blocks(lines: list[str]) -> list[bool]:
    in_code_block = False
    outside = []
    for line in lines:
        fence = line.lstrip().startswith(("```", "~~~"))
        if fence:
            in_code_block = not in_code_block
        outside.append(not in_code_block and not fence)
    return outside


def anchor_section_headings(lines: list[str], label: str) -> list[str]:
    slug_counts: dict[str, int] = {}
    anchored = []
    for line, is_text in zip(lines, outside_code_blocks(lines)):
        heading = HEADING.match(line) if is_text else None
        if heading is None or "{#" in line:
            anchored.append(line)
            continue
        slug = github_slug(heading.group(2))
        # GitHub numbers a repeated heading's anchor: slug, slug-1, slug-2, ...
        count = slug_counts.get(slug, 0)
        slug_counts[slug] = count + 1
        if count:
            slug = f"{slug}-{count}"
        anchored.append(f"{line.rstrip()} {{#{label}-{slug}}}")
    return anchored


def section_links_as_anchors(lines: list[str], relative_path: str) -> list[str]:
    page_directory = (REPOSITORY_ROOT / relative_path).parent

    def as_anchor(match: re.Match) -> str:
        text, target_file, slug = match.groups()
        if not target_file:
            target_path = relative_path
        else:
            target = (page_directory / target_file).resolve()
            if not target.is_relative_to(REPOSITORY_ROOT):
                return match.group(0)
            target_path = target.relative_to(REPOSITORY_ROOT).as_posix()
        anchor = f"{page_label(target_path)}-{slug}"
        # Doxygen 1.17 writes the text of a Markdown link to an anchor twice; \ref does not.
        if '"' in text or "`" in text:
            return f"[{text}](#{anchor})"
        return f'\\ref {anchor} "{text}"'

    return [
        SECTION_LINK.sub(as_anchor, line) if is_text else line
        for line, is_text in zip(lines, outside_code_blocks(lines))
    ]


def label_title_heading(lines: list[str], label: str) -> list[str]:
    # A leading HTML comment (a note on the page for its editors) is not its title.
    in_comment = False
    first_text = None
    for index, line in enumerate(lines):
        if not in_comment and line.lstrip().startswith("<!--"):
            in_comment = True
        if in_comment:
            in_comment = "-->" not in line
            continue
        if line.strip():
            first_text = index
            break
    if first_text is None:
        return lines
    title = lines[first_text]
    if "{#" in title:
        return lines
    # A page without a title heading (LICENSE.md) is titled by its first line.
    if not title.startswith("# "):
        title = f"# {title.strip()}"
    return lines[:first_text] + [f"{title.rstrip()} {{#{label}}}"] + lines[first_text + 1 :]


def drop_readme_front_matter(lines: list[str]) -> list[str]:
    if not lines or not lines[0].startswith('<p align="center">'):
        return lines
    first_section = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("## ") and line.strip() != "## Contents"
        ),
        None,
    )
    if first_section is None:
        return lines
    return [README_TITLE] + [""] * (first_section - 1) + lines[first_section:]


def lesson_links_as_subpages(lines: list[str]) -> list[str]:
    lessons_directory = TUTORIALS_INDEX.rsplit("/", 1)[0]

    def as_subpage(match: re.Match) -> str:
        text, lesson_file = match.groups()
        return f'\\subpage {page_label(f"{lessons_directory}/{lesson_file}")} "{text}"'

    return [
        LESSON_LINK.sub(as_subpage, line) if line.lstrip().startswith("|") else line
        for line in lines
    ]


def file_links_to_repository(lines: list[str], relative_path: str) -> list[str]:
    page_directory = (REPOSITORY_ROOT / relative_path).parent

    def as_repository_link(match: re.Match) -> str:
        target_file = match.group(1)
        if target_file.endswith((".md", ".html")):
            return match.group(0)
        target = (page_directory / target_file).resolve()
        if not target.is_relative_to(REPOSITORY_ROOT) or not target.exists():
            return match.group(0)
        return f"]({REPOSITORY_FILE_URL}{target.relative_to(REPOSITORY_ROOT).as_posix()})"

    return [
        FILE_LINK.sub(as_repository_link, line) if is_text else line
        for line, is_text in zip(lines, outside_code_blocks(lines))
    ]


def math_as_formulas(lines: list[str]) -> list[str]:
    def outside_code_spans(line: str) -> str:
        pieces = []
        position = 0
        for span in CODE_SPAN.finditer(line):
            pieces.append(as_formulas(line[position : span.start()]))
            pieces.append(span.group(0))
            position = span.end()
        pieces.append(as_formulas(line[position:]))
        return "".join(pieces)

    def as_formulas(text: str) -> str:
        text = DISPLAY_MATH.sub(lambda match: f"\\f[{match.group(1)}\\f]", text)
        return INLINE_MATH.sub(lambda match: f"\\f${match.group(1)}\\f$", text)

    return [
        outside_code_spans(line) if is_text else line
        for line, is_text in zip(lines, outside_code_blocks(lines))
    ]


def rules_as_asterisks(lines: list[str]) -> list[str]:
    in_code_block = False
    filtered = []
    for line in lines:
        if line.lstrip().startswith(("```", "~~~")):
            in_code_block = not in_code_block
        # After a blank line `---` is a rule; after text it underlines a heading.
        is_rule = (
            line.rstrip() == "---"
            and not in_code_block
            and (not filtered or not filtered[-1].strip())
        )
        filtered.append("***" if is_rule else line)
    return filtered


def main() -> None:
    path = Path(sys.argv[1]).resolve()
    relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()

    if relative_path == "README.md":
        lines = drop_readme_front_matter(lines)
    if relative_path == TUTORIALS_INDEX:
        lines = lesson_links_as_subpages(lines)
    label = page_label(relative_path)
    # The landing page is the main page, titled by the project name.
    if relative_path != LANDING_PAGE:
        lines = label_title_heading(lines, label)
    lines = anchor_section_headings(lines, label)
    lines = section_links_as_anchors(lines, relative_path)
    lines = file_links_to_repository(lines, relative_path)
    lines = math_as_formulas(lines)
    print("\n".join(rules_as_asterisks(lines)))


if __name__ == "__main__":
    main()
