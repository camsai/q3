"""
The abstract core of the provenance check: nothing here names a reference or a foreign language.

PROVENANCE.md forbids translating a reference implementation's source into Q3,
and a translation leaves marks a search can find. Three ``Rule`` s search for
them, each parameterized by content that is unique to one reference:

- ``RedFlagPhraseRule``: the wording of § 4.2, with every reference's names as targets;
- ``ForeignSyntaxRule``: a reference's ``ForeignLanguage`` spelled inside Q3's code;
- ``ReferenceIdentifierRule``: a reference's variable names used as Q3 identifiers.

A ``ReferenceSource`` is that content, read from ``references/<key>.json``. A
``HostLanguage`` is one of Q3's own languages and how it separates code from
comments and strings. ``Scanner`` runs the rules over the files the export
ships (``ExportScope``), and ``Baseline`` ratchets what they find.
"""

from __future__ import annotations

import bisect
import functools
import io
import json
import re
import string
import subprocess
import tokenize
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import NamedTuple

# --------------------------------------------------------------------------
# Findings and text
# --------------------------------------------------------------------------

#: Lowercases ASCII letters only, for the one text that ``str.lower`` would lengthen.
ASCII_LOWERCASE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def lowercase_same_length(text: str) -> str:
    """
    *text* lowercased, with every offset unchanged.

    Offsets into the lowered text are reported as offsets into the original, so
    lowering must not change its length. ``str.lower`` changes the length of
    exactly one character, U+0130 (dotted capital I); a text containing it is
    lowered ASCII-only instead, which is slower.
    """
    return text.translate(ASCII_LOWERCASE) if "İ" in text else text.lower()


def line_numbers(text: str) -> Callable[[int], int]:
    """A function from an offset in *text* to its 1-based line number."""
    starts = [0] + [match.end() for match in re.finditer("\n", text)]
    return lambda offset: bisect.bisect_right(starts, offset)


class ScreenedPattern(NamedTuple):
    """
    A pattern matched against lowercased text, behind a cheap screen.

    Python's ``re`` scans fastest for a pattern without case folding, and not
    at all when no *screen* substring occurs in the text -- which, for every
    pattern here, is almost every file.
    """

    signal: str
    screen: tuple[str, ...]
    pattern: re.Pattern[str]

    def finditer(self, lowered: str) -> Iterator[re.Match[str]]:
        """The matches in *lowered* (lowercased text), or none when the screen fails."""
        if any(literal in lowered for literal in self.screen):
            return self.pattern.finditer(lowered)
        return iter(())


class Finding(NamedTuple):
    """One signal at one line of one file."""

    path: str
    line: int
    rule: str
    signal: str

    @property
    def key(self) -> str:
        """The baseline key: ``"<rule>: <signal>"``."""
        return f"{self.rule}: {self.signal}"


# --------------------------------------------------------------------------
# Host languages: how Q3's own files separate code from prose
# --------------------------------------------------------------------------


class HostLanguage(ABC):
    """
    A language of Q3's own files, and how to tell its code from its prose.

    The syntax and identifier rules read code only: a comment may quote a
    reference and a string may name its format, so each host language removes
    both before a rule looks. Line count is preserved, so a finding's line is
    the line in the file.
    """

    #: The name rules use to say which hosts a signal applies to.
    name: str = ""
    #: The file suffixes written in this language.
    suffixes: frozenset[str] = frozenset()

    @abstractmethod
    def code_lines(self, text: str) -> list[str]:
        """The code of *text*, one entry per line, with comments and string contents removed."""


#: What is not code in a C++ source: raw strings, comments, then string and
#: character literals (an unterminated one ends at the line break).
NON_CODE = re.compile(
    r'(?<![\w])(?:u8|[uUL])?R"(?P<delimiter>[^()\\\s]{0,16})\(.*?\)(?P=delimiter)"'
    r"|//[^\n]*"
    r"|/\*.*?(?:\*/|\Z)"
    r'|"(?:\\.|[^"\\\n])*"?'
    r"|'(?:\\.|[^'\\\n])*'?",
    re.DOTALL,
)


def _blank_non_code(match: re.Match[str]) -> str:
    """A comment becomes nothing and a literal becomes empty quotes; line breaks stay."""
    text = match.group(0)
    breaks = "\n" * text.count("\n")
    if text.startswith(("//", "/*")):
        return breaks
    quote = "'" if text.startswith("'") else '"'
    return quote * 2 + breaks


class CppHostLanguage(HostLanguage):
    """
    C++, CUDA and Objective-C++ sources.

    Comments are removed, string and character literals (raw strings too) are
    emptied, and preprocessor directives -- with their continuation lines --
    are blanked, so ``#endif`` is not Fortran and a quoted path is not code.
    """

    name = "cpp"
    suffixes = frozenset({".hpp", ".cpp", ".h", ".cc", ".cxx", ".c", ".cu", ".mm"})

    def code_lines(self, text: str) -> list[str]:
        """The code of *text*, one entry per line."""
        lines = NON_CODE.sub(_blank_non_code, text).split("\n")
        in_directive = False
        for number, line in enumerate(lines):
            if in_directive or line.lstrip().startswith("#"):
                in_directive = line.rstrip().endswith("\\")
                lines[number] = ""
        return lines


#: Token types that are prose, not code: comments, strings and f-string text.
PROSE_TOKEN_TYPES = frozenset(
    {tokenize.STRING, tokenize.COMMENT}
    | {
        getattr(tokenize, name)
        for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
        if hasattr(tokenize, name)
    }
)


class PythonHostLanguage(HostLanguage):
    """
    Python sources: comments, strings and docstrings are blanked by the tokenizer.

    A file the tokenizer rejects falls back to cutting each line at its first ``#``.
    """

    name = "python"
    suffixes = frozenset({".py"})

    def code_lines(self, text: str) -> list[str]:
        """The code of *text*, one entry per line."""
        lines = text.split("\n")
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, SyntaxError, IndentationError):
            return [line.split("#", 1)[0] for line in lines]
        blanked = [list(line) for line in lines]
        for token in tokens:
            if token.type not in PROSE_TOKEN_TYPES:
                continue
            (start_row, start_column), (end_row, end_column) = token.start, token.end
            for row in range(start_row, end_row + 1):
                characters = blanked[row - 1]
                first = start_column if row == start_row else 0
                last = end_column if row == end_row else len(characters)
                for column in range(first, min(last, len(characters))):
                    characters[column] = " "
        return ["".join(characters) for characters in blanked]


class PlainTextHostLanguage(HostLanguage):
    """
    Documentation, configuration and data: no code at all.

    Only the phrase rule, which reads prose, applies to these files; the code
    rules see no lines, so they need not ask what kind of file they were given.
    """

    name = "text"

    def code_lines(self, text: str) -> list[str]:
        """No code."""
        return []


#: Every host language with code; a new one is added here and nowhere else.
HOST_LANGUAGES: tuple[HostLanguage, ...] = (CppHostLanguage(), PythonHostLanguage())
PLAIN_TEXT = PlainTextHostLanguage()
_BY_SUFFIX = {suffix: language for language in HOST_LANGUAGES for suffix in language.suffixes}


def host_language_for(relative_path: str) -> HostLanguage:
    """The language *relative_path* is written in, by suffix; plain text for anything else."""
    return _BY_SUFFIX.get(PurePosixPath(relative_path).suffix, PLAIN_TEXT)


class SourceFile:
    """
    One file of the export: its path, its shipped text, and its code.

    The code is computed once, on first use, and shared by every rule that
    reads it.
    """

    def __init__(self, relative_path: str, text: str, host_language: HostLanguage = None):
        self.relative_path = relative_path
        self.text = text
        self.host_language = host_language or host_language_for(relative_path)

    @functools.cached_property
    def code_lines(self) -> list[str]:
        """The file's code, one entry per line; empty for a file with no code."""
        return self.host_language.code_lines(self.text)


# --------------------------------------------------------------------------
# References: the content one reference implementation contributes
# --------------------------------------------------------------------------


class ForeignConstruct(NamedTuple):
    """One recognisable construct of a foreign language, and the hosts where it is unambiguous."""

    pattern: ScreenedPattern
    host_names: frozenset[str]


class ForeignLanguage(ABC):
    """
    A language Q3 is not written in, whose syntax inside Q3's code means a translation.

    A subclass lists its constructs. A construct names the host languages in
    which it cannot be legitimate code: a pattern that is ordinary syntax in a
    host (``**`` is a power in Python) leaves that host out.
    """

    #: The rule this language produces is ``<name>_syntax``.
    name: str = ""

    @abstractmethod
    def constructs(self) -> tuple[ForeignConstruct, ...]:
        """Every construct of this language that a search can recognise."""

    def constructs_in(self, host_language: HostLanguage) -> tuple[ScreenedPattern, ...]:
        """The patterns that signal this language inside *host_language* code."""
        return tuple(
            construct.pattern
            for construct in self.constructs()
            if host_language.name in construct.host_names
        )


class ReferenceSource(NamedTuple):
    """
    What is unique to one reference implementation.

    ``key``              names its rules: ``<key>_identifier``;
    ``names_in_prose``   how text names it, so that "ported from <name>" is a red flag;
    ``languages``        what its source is written in, so that its syntax in Q3's code is one;
    ``identifiers``      its variable names, which Q3's identifiers must not reuse
                         (``igk_k`` matches the consecutive words ``igk``, ``k``).
    """

    key: str
    names_in_prose: tuple[str, ...]
    languages: tuple[ForeignLanguage, ...]
    identifiers: tuple[tuple[str, ...], ...]

    @classmethod
    def from_file(cls, path: Path, languages: Mapping[str, ForeignLanguage]) -> ReferenceSource:
        """
        Read one ``references/<key>.json``; the file's stem must be its key.

        *languages* maps each known foreign language's name to it; a reference
        naming any other is refused.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["key"] != path.stem:
            raise ValueError(f"{path.name}: key {data['key']!r} is not the file's name")
        unknown = sorted(set(data["languages"]) - set(languages))
        if unknown:
            raise ValueError(
                f"{path.name}: unknown languages {unknown}; known: {sorted(languages)}"
            )
        return cls(
            key=data["key"],
            names_in_prose=tuple(data["names_in_prose"]),
            languages=tuple(languages[name] for name in data["languages"]),
            identifiers=tuple(tuple(name.lower().split("_")) for name in data["identifiers"]),
        )


def load_reference_sources(
    directory: Path, languages: Mapping[str, ForeignLanguage]
) -> list[ReferenceSource]:
    """Every reference described in *directory*, in file-name order."""
    return [ReferenceSource.from_file(path, languages) for path in sorted(directory.glob("*.json"))]


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


class Rule(ABC):
    """
    One kind of signal: a rule reads a ``SourceFile`` and returns its ``Finding`` s.

    A rule's ``name`` is the first half of every baseline key it produces and
    what the allowlist names to excuse a path from it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The rule's name in the baseline and the allowlist."""

    @abstractmethod
    def find(self, source_file: SourceFile) -> list[Finding]:
        """Every signal of this rule in *source_file*."""


VERBS = ("transcribed", "ported", "translated", "adapted", "copied")
SOURCE_FILE_EXTENSIONS = "f90 f95 f03 f08 fpp for f c cc cpp cxx h hpp cu py jl m".split()
#: A source file (matched lowercased): an optional directory path, then a name
#: starting with a letter and a source extension (``PW/src/addusdens.f90``, ``x.cpp``).
FILE_TARGET = r"(?:[\w.+-]+/)*[a-z_][\w+-]*\.(?:" + "|".join(SOURCE_FILE_EXTENSIONS) + r")\b"
#: Up to three words may sit between "from" and the target ("from the QE
#: routine x.f90"); punctuation ends the run, so a target in the next clause
#: does not count.
INTERVENING_WORDS = r"(?:[\w'’-]+\s+){0,3}?"
#: Comment leaders at the start of each line, removed before joining lines.
COMMENT_LEADER = re.compile(r"^[ \t]*(?:/{2,}!?|/\*+|\*+/?|#+|!+|--|>+|<!--)?[ \t]*", re.MULTILINE)


def joined_prose(text: str) -> tuple[str, list[int]]:
    """*text* as one line with comment leaders removed; return it and each line's start offset."""
    stripped = COMMENT_LEADER.sub("", text)
    starts = [0] + [match.end() for match in re.finditer("\n", stripped)]
    return stripped.replace("\n", " "), starts


class RedFlagPhraseRule(Rule):
    """
    "transcribed / ported / translated / adapted / copied from <target>", where
    the target is a source file or a reference's name in prose; "follows
    <anything> line by line"; and "based on the implementation in".

    Read in every file, prose and code alike. Comment leaders and line breaks
    between the words are ignored, so a phrase wrapped across two comment lines
    is still one phrase, reported at the line where it starts.
    """

    def __init__(self, reference_names: list[str]):
        names = "|".join(
            re.escape(name.lower()) for name in sorted(set(reference_names), key=len, reverse=True)
        )
        target = f"(?:{FILE_TARGET}|\\b(?:{names})\\b)" if names else FILE_TARGET
        self.patterns = [
            ScreenedPattern(
                f"{verb} from",
                (verb,),
                re.compile(rf"\b{verb}\s+from\s+{INTERVENING_WORDS}{target}"),
            )
            for verb in VERBS
        ]
        self.patterns.append(
            ScreenedPattern(
                "follows <file> line by line",
                ("follows",),
                re.compile(r"\bfollows\s+(?:\S+\s+){1,6}?line[\s-]+by[\s-]+line\b"),
            )
        )
        self.patterns.append(
            ScreenedPattern(
                "based on the implementation in",
                ("implementation",),
                re.compile(r"\bbased\s+on\s+the\s+implementation\s+in\b"),
            )
        )

    @property
    def name(self) -> str:
        """``red_flag_phrase``."""
        return "red_flag_phrase"

    def find(self, source_file: SourceFile) -> list[Finding]:
        """Every § 4.2 phrase in the file's text."""
        joined, starts = joined_prose(source_file.text)
        lowered = lowercase_same_length(joined)
        findings = []
        for phrase in self.patterns:
            for match in phrase.finditer(lowered):
                line_index = max(0, bisect.bisect_right(starts, match.start()) - 1)
                findings.append(
                    Finding(source_file.relative_path, line_index + 1, self.name, phrase.signal)
                )
        return findings


class ForeignSyntaxRule(Rule):
    """
    The constructs of one foreign language inside a host file's code.

    Comments and strings are not code, so a citation that quotes the
    reference is not a finding; which constructs count depends on the host
    (see ``ForeignLanguage.constructs_in``).
    """

    def __init__(self, language: ForeignLanguage):
        self.language = language

    @property
    def name(self) -> str:
        """``<language>_syntax``, e.g. ``fortran_syntax``."""
        return f"{self.language.name}_syntax"

    def find(self, source_file: SourceFile) -> list[Finding]:
        """Every construct of the language in the file's code."""
        lowered = lowercase_same_length("\n".join(source_file.code_lines))
        line_of = line_numbers(lowered)
        return sorted(
            Finding(source_file.relative_path, line_of(match.start()), self.name, pattern.signal)
            for pattern in self.language.constructs_in(source_file.host_language)
            for match in pattern.finditer(lowered)
        )


WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b|_)|[A-Z]?[a-z]+|[A-Z]+|\d+")


def identifier_words(identifier: str) -> list[str]:
    """Split *identifier* at underscores and camelCase boundaries, lowercased."""
    words = []
    for part in identifier.split("_"):
        words.extend(word.lower() for word in WORD.findall(part))
    return words


class ReferenceIdentifierRule(Rule):
    """
    One reference's variable names as words of an identifier in host code.

    Identifiers are split into words at underscores and camelCase boundaries,
    so ``becsum_``, ``compute_becsum`` and ``BecsumTable`` all name ``becsum``;
    a multi-word name (``igk_k``) matches consecutive words, longest first.
    Comments and strings are not code: a format reader's keys and a citation
    may spell the names.
    """

    #: The hosts whose identifiers are checked. Python is left out: the test
    #: helpers that read a reference's binary files name their records after it.
    host_names = frozenset({"cpp"})

    def __init__(self, reference: ReferenceSource):
        self.reference = reference
        self.names = tuple(sorted(reference.identifiers, key=len, reverse=True))
        self.spellings = {
            spelling for name in self.names for spelling in ("_".join(name), "".join(name))
        }

    @property
    def name(self) -> str:
        """``<reference>_identifier``, e.g. ``quantum_espresso_identifier``."""
        return f"{self.reference.key}_identifier"

    def find(self, source_file: SourceFile) -> list[Finding]:
        """Every one of the reference's names used in an identifier of the file's code."""
        if source_file.host_language.name not in self.host_names:
            return []
        code = "\n".join(source_file.code_lines)
        line_of = line_numbers(code)
        findings = []
        for start, identifier in sorted(self._candidates(code).items()):
            words = identifier_words(identifier)
            position = 0
            while position < len(words):
                for name in self.names:
                    if tuple(words[position : position + len(name)]) == name:
                        findings.append(
                            Finding(
                                source_file.relative_path, line_of(start), self.name, "_".join(name)
                            )
                        )
                        position += len(name)
                        break
                else:
                    position += 1
        return findings

    def _candidates(self, code: str) -> dict[int, str]:
        """
        The identifiers of *code* that contain a name, keyed by their offset.

        Splitting identifiers into words is the slow part, so only those that
        contain one of the names, with or without its underscores, are split.
        """
        lowered = lowercase_same_length(code)
        candidates: dict[int, str] = {}
        for literal in self.spellings:
            found = lowered.find(literal)
            while found != -1:
                start, end = found, found + len(literal)
                found = lowered.find(literal, found + 1)
                while start > 0 and (code[start - 1].isalnum() or code[start - 1] == "_"):
                    start -= 1
                while end < len(code) and (code[end].isalnum() or code[end] == "_"):
                    end += 1
                if not code[start].isdigit():
                    candidates[start] = code[start:end]
        return candidates


def build_rules(references: list[ReferenceSource]) -> list[Rule]:
    """
    One phrase rule over every reference's names, one syntax rule per foreign
    language any reference is written in, and one identifier rule per reference
    that lists identifiers.
    """
    rules: list[Rule] = [
        RedFlagPhraseRule([name for reference in references for name in reference.names_in_prose])
    ]
    languages = {
        language.name: language for reference in references for language in reference.languages
    }
    rules += [ForeignSyntaxRule(languages[name]) for name in sorted(languages)]
    rules += [
        ReferenceIdentifierRule(reference) for reference in references if reference.identifiers
    ]
    return rules


# --------------------------------------------------------------------------
# Scope, allowlist and baseline
# --------------------------------------------------------------------------


class AllowlistEntry(NamedTuple):
    """One allowlisted file, or directory (a trailing ``/``), the rules it is excused from, and why."""

    path: str
    rules: frozenset[str]
    reason: str

    def covers(self, relative_path: str, rule_name: str) -> bool:
        """Whether this entry excuses *relative_path* from the rule called *rule_name*."""
        if rule_name not in self.rules:
            return False
        if self.path.endswith("/"):
            return relative_path.startswith(self.path)
        return relative_path == self.path


def load_allowlist(path: Path, rule_names: set[str]) -> list[AllowlistEntry]:
    """Read an allowlist; every entry must name rules in *rule_names* and give a reason."""
    entries = []
    for raw in json.loads(path.read_text(encoding="utf-8")):
        rules = frozenset(raw["rules"])
        if rules - rule_names or not raw.get("reason", "").strip():
            raise ValueError(f"allowlist entry {raw['path']!r}: unknown rules or no reason")
        entries.append(AllowlistEntry(raw["path"], rules, raw["reason"]))
    return entries


def stale_allowlist_entries(allowlist: list[AllowlistEntry], root: Path) -> list[str]:
    """The allowlist paths that no longer exist under *root*."""
    return [entry.path for entry in allowlist if not (root / entry.path).exists()]


#: The directory segment every build-time add-on path carries.
ADDON_SEGMENT = "addons"


def addon_name(relative_path: str) -> str | None:
    """The add-on a path belongs to (the segment after ``addons``), or None for core."""
    parts = relative_path.split("/")
    for index, part in enumerate(parts[:-2]):
        if part == ADDON_SEGMENT:
            return parts[index + 1]
    return None


RELEASE_IGNORE = ".releaseignore"


class ReleaseIgnore(NamedTuple):
    """The ``.releaseignore`` syntax: ``name/`` segment, ``a/b/`` prefix, or exact file."""

    segments: frozenset[str]
    prefixes: tuple[str, ...]
    files: frozenset[str]

    @classmethod
    def read(cls, root: Path) -> ReleaseIgnore:
        """The root's export exclusions; none when it has no ``.releaseignore`` (the public tree)."""
        segments: set[str] = set()
        prefixes: list[str] = []
        files: set[str] = set()
        path = root / RELEASE_IGNORE
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.endswith("/") and "/" not in line[:-1]:
                segments.add(line[:-1])
            elif line.endswith("/"):
                prefixes.append(line)
            else:
                files.add(line)
        return cls(frozenset(segments), tuple(prefixes), frozenset(files))

    def excludes(self, relative_path: str) -> bool:
        """
        Whether this scan leaves out *relative_path*.

        Everything the export leaves out, except a named add-on's files: those
        are scanned and ratcheted in the add-on's own baseline. A file directly
        in an ``addons`` directory belongs to no add-on and is left out.
        """
        if relative_path in self.files or relative_path.startswith(self.prefixes):
            return True
        excluding = {name for name in relative_path.split("/")[:-1] if name in self.segments}
        return bool(excluding - {ADDON_SEGMENT}) or (
            bool(excluding) and addon_name(relative_path) is None
        )


#: Vendored upstream text, exported unchanged and so never reworded here.
VENDORED_PREFIXES = ("schemas/esse/",)
VAULT_BEGIN = re.compile(r"^[ \t]*(?:<!--|//|#|\*)\s*vault:begin\b")
VAULT_END = re.compile(r"^[ \t]*(?:<!--|//|#|\*)\s*vault:end\b")
FENCE = re.compile(r"^[ \t]*(?:```|~~~)")


def listed_files(root: Path) -> list[str]:
    """Tracked and unignored files under *root* when it is a git work tree's top, else every file."""
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if Path(top).resolve() == root.resolve():
            command = [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ]
            output = subprocess.run(command, capture_output=True, check=True).stdout
            return sorted({name for name in output.decode("utf-8").split("\0") if name})
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def shipped_text(text: str) -> str:
    """*text* with ``vault:begin`` ... ``vault:end`` regions blanked, line count preserved."""
    lines = text.split("\n")
    inside = fenced = False
    for number, line in enumerate(lines):
        if FENCE.match(line):
            fenced = not fenced
        if fenced or FENCE.match(line):
            if inside:
                lines[number] = ""
            continue
        if not inside and VAULT_BEGIN.match(line):
            inside = True
        if inside:
            ended = VAULT_END.match(line) is not None
            lines[number] = ""
            inside = not ended
    return "\n".join(lines)


class ExportScope:
    """
    The files of *root* the export ships, as the packer would leave them.

    Paths excluded by the root's ``.releaseignore`` are skipped (a named
    add-on's files excepted, see ``ReleaseIgnore.excludes``), as are
    ``schemas/esse/``, binary files, and lines inside ``vault:`` regions. The
    public tree has no ``.releaseignore``, so the same scan there sees the same
    files and the same lines.
    """

    def __init__(self, root: Path):
        self.root = root
        self.release_ignore = ReleaseIgnore.read(root)

    def source_files(self) -> Iterator[SourceFile]:
        """Every shipped text file, in path order."""
        for relative_path in listed_files(self.root):
            if relative_path.startswith(VENDORED_PREFIXES):
                continue
            if self.release_ignore.excludes(relative_path):
                continue
            path = self.root / relative_path
            if not path.is_file():
                continue
            data = path.read_bytes()
            if b"\0" in data[:8192]:
                continue
            yield SourceFile(relative_path, shipped_text(data.decode("utf-8", errors="replace")))


class Scanner:
    """Runs *rules* over source files, skipping each (path, rule) the allowlist excuses."""

    def __init__(self, rules: list[Rule], allowlist: list[AllowlistEntry]):
        self.rules = rules
        self.allowlist = allowlist

    def scan_file(self, source_file: SourceFile) -> list[Finding]:
        """Every finding in one file."""
        findings: list[Finding] = []
        for rule in self.rules:
            if not any(
                entry.covers(source_file.relative_path, rule.name) for entry in self.allowlist
            ):
                findings += rule.find(source_file)
        return findings

    def scan(self, root: Path) -> list[Finding]:
        """Every finding in the files the export ships from *root*."""
        findings: list[Finding] = []
        for source_file in ExportScope(root).source_files():
            findings += self.scan_file(source_file)
        return findings


def count(findings: list[Finding]) -> dict[str, dict[str, int]]:
    """{path: {"<rule>: <signal>": count}} for every path with a finding."""
    counts: dict[str, Counter[str]] = {}
    for finding in findings:
        counts.setdefault(finding.path, Counter())[finding.key] += 1
    return {path: dict(sorted(counts[path].items())) for path in sorted(counts)}


Counts = dict[str, dict[str, int]]


def total(counts: Counts) -> int:
    """Sum every count in a {file: {key: count}} mapping."""
    return sum(sum(per_file.values()) for per_file in counts.values())


class Baseline:
    """
    The committed per-file counts, keyed ``"<rule>: <signal>"``.

    The core baseline is *core_path*. An add-on's findings live in a file of the
    same name in ``src/addons/<name>/``, so the core baseline never names an
    add-on path and a tree without ``src/addons/`` has no keys for files it
    lacks. A missing file is an empty baseline.
    """

    def __init__(self, core_path: Path, root: Path):
        self.core_path = core_path
        self.root = root

    def _addon_directories(self) -> list[Path]:
        return sorted(
            directory
            for directory in (self.root / "src" / ADDON_SEGMENT).glob("*")
            if directory.is_dir()
        )

    @staticmethod
    def _read(path: Path) -> Counts:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def _write(counts: Counts, path: Path) -> None:
        path.write_text(json.dumps(counts, indent=4, sort_keys=True) + "\n", encoding="utf-8")

    def load(self) -> Counts:
        """The core baseline plus every add-on's own."""
        counts = self._read(self.core_path)
        for directory in self._addon_directories():
            counts.update(self._read(directory / self.core_path.name))
        return counts

    def write(self, measured: Counts) -> None:
        """Write *measured*: each add-on's entries beside its code, the rest to the core file."""
        self._write(
            {file: value for file, value in measured.items() if addon_name(file) is None},
            self.core_path,
        )
        for directory in self._addon_directories():
            entries = {
                file: value
                for file, value in measured.items()
                if addon_name(file) == directory.name
            }
            target = directory / self.core_path.name
            if entries or target.exists():
                self._write(entries, target)

    @staticmethod
    def compare(measured: Counts, baseline: Counts) -> tuple[list[str], list[str]]:
        """Return (increases, decreases) as human-readable lines."""
        increases: list[str] = []
        decreases: list[str] = []
        for file in sorted(set(measured) | set(baseline)):
            now = measured.get(file, {})
            then = baseline.get(file, {})
            for key in sorted(set(now) | set(then)):
                current, allowed = now.get(key, 0), then.get(key, 0)
                line = f"{file}: {key} {allowed} -> {current}"
                if current > allowed:
                    increases.append(line)
                elif current < allowed:
                    decreases.append(line)
        return increases, decreases
