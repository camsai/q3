"""Tests for ``scripts/repo/check_no_addon_leak.py``.

The check keeps each add-on's engine-specific text out of every exported path,
and a matcher that silently stopped matching would read exactly like a clean
tree. So every rule is pinned here with a match and a near miss.

The fixtures hash neutral words (zebra, decade, QUOKKA, purple giraffe) with a
test salt and never use the real terms: this file ships, and the real terms must
not. PROVENANCE.md § 5.4 is why a guard's fixtures may otherwise hold the very
strings it rejects.
"""

import base64
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_no_addon_leak.py"
PACKER_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "pack_public.py"

spec = importlib.util.spec_from_file_location("check_no_addon_leak", SCRIPT_PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

SALT = "fixture-salt"
#: (mode, term). "decade" is spelled with hexadecimal letters only, which is what
#: lets a digest test show that masking, not luck, keeps a digest clean.
NEUTRAL_TERMS = (
    ("sub-word", "zebra"),
    ("sub-word", "decade"),
    ("whole-word", "QUOKKA"),
    ("phrase", "purple giraffe"),
)
NEUTRAL_WORDS = ("zebra", "decade", "quokka", "purple", "giraffe")


def neutral_digest(mode: str, term: str) -> str:
    return checker.term_digest(SALT, mode, checker.normalize_term(mode, term))


def neutral_denylist():
    entries = sorted(
        checker.DenylistEntry(neutral_digest(mode, term), mode) for mode, term in NEUTRAL_TERMS
    )
    return checker.Denylist(SALT, tuple(entries))


MATCHER = checker.TermMatcher(neutral_denylist())
TERM_BY_DIGEST = {neutral_digest(mode, term): term for mode, term in NEUTRAL_TERMS}


def matched_terms(text: str) -> set[str]:
    """The neutral terms *text* contains, recovered through this file's own table."""
    return {TERM_BY_DIGEST[entry.digest] for _, entry in MATCHER.matches(text)}


def make_tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_neutral_denylist(path: Path) -> Path:
    path.write_text(checker.denylist_text(neutral_denylist()))
    return path


def write_term_list(root: Path, addon: str, text: str) -> Path:
    path = root / "src" / "addons" / addon / "leak_terms.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def random_base64(seed: bytes, length: int) -> str:
    """Deterministic base64 text that reads as random data, as an embedded image does."""
    blocks = b"".join(hashlib.sha512(seed + bytes([index])).digest() for index in range(16))
    return base64.b64encode(blocks[:length]).decode()


# ── Tokens ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("token", "pieces"),
    [
        ("FooBARReader", ["Foo", "BAR", "Reader"]),
        ("read_foo_file", ["read", "foo", "file"]),
        ("readFooFile", ["read", "Foo", "File"]),
        ("Bar2Reader", ["Bar2", "Reader"]),
        ("BAR2Reader", ["BAR2", "Reader"]),
        ("HTTPServer", ["HTTP", "Server"]),
        ("EXPECTED_NUMBER_OF_FOO", ["EXPECTED", "NUMBER", "OF", "FOO"]),
        ("x86_64", ["x86", "64"]),
        ("ZEBRAsReader", ["ZEBRAs", "Reader"]),
        ("readFOXesFile", ["read", "FOXes", "File"]),
        ("IDsList", ["IDs", "List"]),
        # Only a lone plural ending stays with the acronym, not a word that starts with s.
        ("ABCsize", ["AB", "Csize"]),
    ],
)
def test_sub_words_split_at_underscores_case_changes_and_acronyms(token, pieces):
    assert checker.sub_words(token) == pieces


def test_a_plural_ending_gives_every_stem_it_could_have():
    assert checker.plural_stems("zebras") == ("zebra",)
    assert checker.plural_stems("foxes") == ("foxe", "fox")
    assert checker.plural_stems("zebra") == ()
    assert checker.plural_stems("s") == (), "a stem keeps a letter"


def test_a_trailing_number_also_stands_for_the_word_before_it():
    assert checker.sub_word_forms("Bar2") == ("bar2", "bar")
    assert checker.sub_word_forms("BAR") == ("bar",)
    assert checker.sub_word_forms("64") == (), "a bare number can match nothing"


# ── Match modes ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "ZebraReader",
        "ZEBRAReader",
        "read_zebra_file",
        "readZebra",
        "ZEBRA",
        "zebra2",
        "zebra-crossing",
        "a zebra.",
        "docs/zebra/notes.md",
    ],
)
def test_a_sub_word_term_matches_any_piece_in_any_case(text):
    assert matched_terms(text) == {"zebra"}


@pytest.mark.parametrize(
    "text",
    [
        "zebras",
        "Zebras",
        "ZEBRAS",
        "ZEBRAs",
        "readZEBRAsFile",
        "ZebrasReader",
        "list_of_zebras",
        "two zebras.",
        "zebra's",
        "ZEBRA's",
        "zebras'",
        "the zebra\u2019s stripes",
    ],
)
def test_a_sub_word_term_matches_its_plural_and_possessive(text):
    assert matched_terms(text) == {"zebra"}


#: A sub-word term whose plural takes ``es``, kept apart from the shared fixtures.
FOX_MATCHER = checker.TermMatcher(
    checker.Denylist(SALT, (checker.DenylistEntry(neutral_digest("sub-word", "fox"), "sub-word"),))
)


@pytest.mark.parametrize("text", ["foxes", "Foxes", "FOXES", "FOXes", "readFOXesFile", "FOXs"])
def test_a_sub_word_term_matches_its_es_plural(text):
    assert [entry.mode for _, entry in FOX_MATCHER.matches(text)] == ["sub-word"]


@pytest.mark.parametrize(
    "text", ["zebrafish", "Zebrafish", "subzebra", "zebrass", "zebrasque", "ZEBRAsize"]
)
def test_a_sub_word_term_does_not_match_inside_a_longer_word(text):
    assert matched_terms(text) == set()


@pytest.mark.parametrize(
    "text", ["QUOKKA", "the QUOKKA file", "data/QUOKKA", "QUOKKA.txt", "`QUOKKA`", "QUOKKA = 4"]
)
def test_a_whole_word_term_matches_the_bare_token(text):
    assert matched_terms(text) == {"QUOKKA"}


@pytest.mark.parametrize(
    "text",
    [
        "EXPECTED_NUMBER_OF_QUOKKA",
        "FULL_ZONE_QUOKKA",
        "NSCF_QUOKKA",
        "number_of_quokka",
        "quokka_list",
        "MAXIMUM_REFERENCE_QUOKKA",
        "QUOKKA2",
        "QUOKKAS",
        "QUOKKAs",
        "Quokka",
        "quokka",
    ],
)
def test_a_whole_word_term_leaves_longer_names_and_other_cases_alone(text):
    assert matched_terms(text) == set()


@pytest.mark.parametrize(
    "text",
    [
        "Purple Giraffe 1",
        "a purple giraffe",
        "PURPLE GIRAFFE",
        "purple_giraffe",
        "PurpleGiraffe",
        "PurpleGiraffe2",
        "purple-giraffe",
        "purple\n// giraffe",
        "purple\n * giraffe",
        '"the purple "\n    "giraffe"',
    ],
)
def test_a_phrase_matches_two_adjacent_words(text):
    assert matched_terms(text) == {"purple giraffe"}


@pytest.mark.parametrize(
    "text",
    [
        "purple. Giraffe",
        "purple, giraffe",
        "purple: giraffe",
        "purple 3 giraffe",
        "purple big giraffe",
        "purple giraffes",
        "giraffe purple",
    ],
)
def test_a_phrase_needs_both_words_next_to_each_other(text):
    assert matched_terms(text) == set()


def test_a_phrase_across_a_line_break_is_reported_on_its_first_line():
    findings = checker.scan_text("notes.md", "intro\nsee the purple\ngiraffe here\n", MATCHER)
    assert [(finding.line, finding.entry.mode) for finding in findings] == [(2, "phrase")]


def test_each_entry_is_reported_once_per_line():
    findings = checker.scan_text("notes.md", "zebra ZEBRA zebra\nzebra\n", MATCHER)
    assert [finding.line for finding in findings] == [1, 2]


# ── What is not read as words ────────────────────────────────────────


def test_a_term_inside_base64_data_is_not_a_finding():
    blob = random_base64(b"image", 300)
    image = f'<img src="data:image/png;base64,{blob[:150]}Zebra{blob[150:]}">'
    assert matched_terms(image) == set()
    assert matched_terms(f"{blob[:150]} Zebra {blob[150:]}") == {"zebra"}, (
        "outside the run, the same word is a finding"
    )


def test_a_compressed_diagram_is_masked_while_a_plain_label_is_read():
    payload = random_base64(b"diagram", 600)
    compressed = (
        '<mxfile><diagram id="a" name="Page-1">'
        f"{payload[:300]}9zebra9{payload[300:]}</diagram></mxfile>"
    )
    plain = '<mxfile><diagram id="a"><mxCell id="2" value="zebra reader"/></diagram></mxfile>'
    assert matched_terms(compressed) == set()
    assert matched_terms(plain) == {"zebra"}


def test_a_hexadecimal_digest_is_not_read_as_words():
    digest = "0123456789decade" + "0123456789abcdef" * 3
    assert len(digest) == 64
    assert matched_terms(f"sha256 = {digest}") == set()
    assert matched_terms(f"oid sha256:{digest.upper()}") == set()
    assert matched_terms("a decade later") == {"decade"}, "the word itself is still a finding"


def test_a_long_test_name_with_digits_is_read_rather_than_masked():
    """A camelCase name mixes cases and digits too, but its words are long."""
    name = "ZebraLexicon42IsRefusedWhileTheDeckIsReadNamingTheTag"
    assert len(name) >= 40
    assert matched_terms(f"TEST(ParserTest, {name})") == {"zebra"}


# ── Paths, binaries and links ────────────────────────────────────────


def test_a_term_in_a_path_is_a_finding_even_when_the_text_is_clean(tmp_path):
    tree = make_tree(tmp_path, {"docs/zebra_notes.md": "nothing to see\n"})
    findings = checker.scan(tree, ["docs/zebra_notes.md"], MATCHER, None)
    assert [(finding.path, finding.line) for finding in findings] == [("docs/zebra_notes.md", 0)]
    assert findings[0].render().endswith("in the path")


def test_binary_contents_are_not_read_but_their_paths_are(tmp_path):
    tree = make_tree(tmp_path, {})
    (tree / "figures").mkdir()
    (tree / "figures" / "plot.png").write_bytes(b"\x89PNG\0\0 zebra QUOKKA")
    (tree / "figures" / "zebra.png").write_bytes(b"\x89PNG\0\0")
    findings = checker.scan(tree, ["figures/plot.png", "figures/zebra.png"], MATCHER, None)
    assert [(finding.path, finding.line) for finding in findings] == [("figures/zebra.png", 0)]


def test_a_symbolic_link_is_read_as_its_stored_target_and_never_followed(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("zebra\n")
    tree = make_tree(tmp_path / "tree", {})
    (tree / "link.md").symlink_to(outside)
    (tree / "named.md").symlink_to("zebra.md")
    findings = checker.scan(tree, ["link.md", "named.md"], MATCHER, None)
    assert [(finding.path, finding.line) for finding in findings] == [("named.md", 1)]


def test_a_directory_entry_has_only_its_path_checked(tmp_path):
    tree = make_tree(tmp_path, {"vendor/zebra/README.md": "zebra\n"})
    findings = checker.scan(tree, ["vendor/zebra"], MATCHER, None)
    assert [(finding.path, finding.line) for finding in findings] == [("vendor/zebra", 0)]


# ── Exemptions ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "src/addons/alpha/reader.hpp",
        "src/addons/alpha/leak_terms.txt",
        "tests/unit/addons/alpha/io/test_reader.cpp",
        "tests/python/addons/alpha/test_terms.py",
        "examples/addons/alpha/case/input.txt",
        "addons/alpha/README.md",
        "schemas/esse",
        "schemas/esse/schema/software.json",
    ],
)
def test_addon_and_vendored_paths_are_exempt(path):
    assert checker.is_exempt(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/io/reader.hpp",
        # A file named like the segment is not an add-on, nor is a longer name.
        "docs/addons",
        "docs/addons.md",
        "src/addonsx/reader.hpp",
        "src/my_addons/reader.hpp",
        "schemas/essence/schema.json",
        "schemas/README.md",
    ],
)
def test_other_paths_are_checked(path):
    assert not checker.is_exempt(path)


def test_an_addon_may_name_its_engine_in_its_paths_and_text(tmp_path):
    tree = make_tree(tmp_path, {"src/addons/zebra/zebra_reader.hpp": "QUOKKA zebra\n"})
    assert checker.scan(tree, ["src/addons/zebra/zebra_reader.hpp"], MATCHER, None) == []


def test_paths_the_release_ignore_excludes_are_exempt(tmp_path):
    tree = make_tree(
        tmp_path,
        {
            ".releaseignore": "private/\ndocs/internal/\nNOTES.md\n",
            "private/a.md": "zebra\n",
            "examples/private/b.md": "zebra\n",
            "docs/internal/c.md": "zebra\n",
            "NOTES.md": "zebra\n",
            "docs/international.md": "zebra\n",
        },
    )
    paths = ["private/a.md", "examples/private/b.md", "docs/internal/c.md", "NOTES.md"]
    paths.append("docs/international.md")
    findings = checker.scan(tree, paths, MATCHER, checker.read_release_ignore(tree))
    assert [finding.path for finding in findings] == ["docs/international.md"]


# ── The .releaseignore reader ────────────────────────────────────────


def test_parse_release_ignore_classifies_entry_forms():
    rules = checker.parse_release_ignore(
        "# comment\n\nprivate/\ndrafts/\ndocs/internal/\ntools/build.py\n.hidden\n"
    )
    assert rules.segments == {"private", "drafts"}
    assert rules.files == {"tools/build.py", ".hidden"}
    assert rules.prefixes == ("docs/internal/",)


def test_comments_and_blank_lines_are_not_entries():
    rules = checker.parse_release_ignore(
        "private/   # never ships\n  \n\tNOTES.md # nor this\n#all/\n"
    )
    assert rules.segments == {"private"}
    assert rules.files == {"NOTES.md"}
    assert rules.prefixes == ()


def test_a_segment_entry_matches_a_directory_at_any_depth_but_not_a_file():
    rules = checker.parse_release_ignore("private/\n")
    assert rules.excludes("private/notes.md")
    assert rules.excludes("examples/private/case/input.yml")
    assert not rules.excludes("docs/private")
    assert not rules.excludes("docs/privateer/notes.md")


def test_a_prefix_entry_matches_only_from_the_root():
    rules = checker.parse_release_ignore("docs/internal/\n")
    assert rules.excludes("docs/internal/notes.md")
    assert not rules.excludes("examples/docs/internal/notes.md")
    assert not rules.excludes("docs/international.md")


def test_a_file_entry_matches_exactly_that_file():
    rules = checker.parse_release_ignore("docs/notes.md\n")
    assert rules.excludes("docs/notes.md")
    assert not rules.excludes("docs/notes.md.bak")
    assert not rules.excludes("other/docs/notes.md")


def test_a_tree_without_a_release_ignore_has_no_rules(tmp_path):
    assert checker.read_release_ignore(tmp_path) is None


def test_the_reader_agrees_with_the_packer_on_every_tracked_path():
    """One reading of the manifest, until the packer imports this reader instead."""
    if not (PACKER_PATH.is_file() and (REPOSITORY_ROOT / ".releaseignore").is_file()):
        pytest.skip("the packer and its manifest stay in the raw repository")
    packer_spec = importlib.util.spec_from_file_location("pack_public", PACKER_PATH)
    pack_public = importlib.util.module_from_spec(packer_spec)
    packer_spec.loader.exec_module(pack_public)
    rules = checker.read_release_ignore(REPOSITORY_ROOT)
    packer_rules = pack_public.load_release_ignore(REPOSITORY_ROOT)
    assert tuple(rules) == tuple(packer_rules)
    for path in checker.tracked_files(REPOSITORY_ROOT):
        assert rules.excludes(path) == pack_public.is_excluded(path, packer_rules), path


# ── The denylist ─────────────────────────────────────────────────────


def test_the_shipped_denylist_holds_only_digests_and_modes():
    text = checker.DENYLIST_PATH.read_text()
    document = json.loads(text)
    assert set(document) == {"description", "salt", "entries"}
    assert re.fullmatch("[0-9a-f]{32}", document["salt"])
    assert document["entries"]
    for item in document["entries"]:
        assert set(item) == {"digest", "mode"}
        assert re.fullmatch("[0-9a-f]{64}", item["digest"])
        assert item["mode"] in checker.MODES


def test_the_shipped_denylist_is_in_the_form_regenerating_writes():
    """Four-space indent, one key per line, a trailing newline: regenerating changes nothing."""
    text = checker.DENYLIST_PATH.read_text()
    assert checker.denylist_text(checker.load_denylist(checker.DENYLIST_PATH)) == text


def test_the_check_names_nothing_it_checks():
    shipped = [
        "scripts/repo/check_no_addon_leak.py",
        "scripts/repo/check_no_addon_leak_denylist.json",
        "tests/python/test_check_no_addon_leak.py",
        ".github/workflows/addon-leak-check.yml",
    ]
    for path in shipped:
        assert (REPOSITORY_ROOT / path).is_file(), f"{path} moved; this list must follow it"
    matcher = checker.TermMatcher(checker.load_denylist(checker.DENYLIST_PATH))
    assert checker.scan(REPOSITORY_ROOT, shipped, matcher, None) == []


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        '{"entries": []}',
        '{"salt": "abc", "entries": []}',
        '{"salt": "abc", "entries": [{"digest": "xyz", "mode": "sub-word"}]}',
        '{"salt": "abc", "entries": [{"digest": "' + "0" * 64 + '", "mode": "anywhere"}]}',
    ],
)
def test_a_missing_empty_or_malformed_denylist_stops_the_check(tmp_path, capsys, document):
    tree = make_tree(tmp_path / "tree", {"notes.md": "zebra\n"})
    denylist = tmp_path / "denylist.json"
    denylist.write_text(document)
    assert checker.main(["--root", str(tree), "--denylist", str(denylist)]) == 2
    assert "denylist" in capsys.readouterr().err
    assert checker.main(["--root", str(tree), "--denylist", str(tmp_path / "absent.json")]) == 2


# ── --regenerate ─────────────────────────────────────────────────────


def regenerate(root: Path, denylist: Path) -> int:
    return checker.main(["--root", str(root), "--denylist", str(denylist), "--regenerate"])


def test_regenerate_hashes_every_addon_term_list(tmp_path):
    write_term_list(tmp_path, "alpha", "# a comment\nsub-word zebra\nwhole-word QUOKKA  # note\n")
    write_term_list(tmp_path, "beta", "phrase Purple Giraffe\nsub-word ZEBRA\n")
    denylist_path = tmp_path / "denylist.json"
    assert regenerate(tmp_path, denylist_path) == 0
    text = denylist_path.read_text()
    for word in NEUTRAL_WORDS:
        assert word not in text.lower(), "the shipped file must not spell out a term"
    denylist = checker.load_denylist(denylist_path)
    expected = {
        checker.term_digest(denylist.salt, "sub-word", "zebra"),
        checker.term_digest(denylist.salt, "whole-word", "QUOKKA"),
        checker.term_digest(denylist.salt, "phrase", "purple giraffe"),
    }
    assert {entry.digest for entry in denylist.entries} == expected, "duplicates collapse"
    assert text == checker.denylist_text(denylist)


def test_regenerating_again_keeps_the_salt_and_the_bytes(tmp_path):
    write_term_list(tmp_path, "alpha", "sub-word zebra\n")
    denylist_path = tmp_path / "denylist.json"
    assert regenerate(tmp_path, denylist_path) == 0
    first = denylist_path.read_text()
    assert regenerate(tmp_path, denylist_path) == 0
    assert denylist_path.read_text() == first


def test_a_first_generation_draws_a_fresh_salt(tmp_path):
    salts = set()
    for name in ("one", "two"):
        root = tmp_path / name
        write_term_list(root, "alpha", "sub-word zebra\n")
        assert regenerate(root, root / "denylist.json") == 0
        salts.add(checker.load_denylist(root / "denylist.json").salt)
    assert len(salts) == 2


def test_regenerate_refuses_to_empty_the_denylist(tmp_path):
    denylist_path = tmp_path / "denylist.json"
    denylist_path.write_text("left alone\n")
    assert regenerate(tmp_path, denylist_path) == 2
    write_term_list(tmp_path, "alpha", "# only comments\n")
    assert regenerate(tmp_path, denylist_path) == 2
    assert denylist_path.read_text() == "left alone\n"


@pytest.mark.parametrize(
    "line",
    [
        "zebra",
        "colour zebra",
        "sub-word zebra_stripes",
        "sub-word ZebraStripes",
        "sub-word 42",
        "whole-word two words",
        "whole-word 9LIVES",
        "phrase purple",
        "phrase very purple giraffe",
        "phrase purple giraffe_neck",
    ],
)
def test_a_term_that_could_never_match_is_refused_without_being_echoed(tmp_path, capsys, line):
    write_term_list(tmp_path, "alpha", f"sub-word quokka\n{line}\n")
    assert regenerate(tmp_path, tmp_path / "denylist.json") == 2
    error = capsys.readouterr().err.replace(str(tmp_path.resolve()), "").replace(str(tmp_path), "")
    assert "leak_terms.txt:2:" in error
    for word in line.split()[1:] or line.split():
        assert word not in error
    assert not (tmp_path / "denylist.json").exists()


# ── Inputs and output ────────────────────────────────────────────────


def test_root_alone_checks_every_file_of_a_packed_tree(tmp_path, capsys):
    tree = make_tree(
        tmp_path / "pack",
        {
            "README.md": "zebra\n",
            "docs/guide.md": "fine\n",
            "src/QUOKKA": "\n",
            ".git/HEAD": "zebra\n",
        },
    )
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    assert checker.main(["--root", str(tree), "--denylist", str(denylist)]) == 1
    sub_word = checker.DenylistEntry(neutral_digest("sub-word", "zebra"), "sub-word")
    whole_word = checker.DenylistEntry(neutral_digest("whole-word", "QUOKKA"), "whole-word")
    assert capsys.readouterr().out.splitlines() == [
        f"README.md:1: add-on term, entry {sub_word.identifier} (sub-word)",
        f"src/QUOKKA: add-on term, entry {whole_word.identifier} (whole-word) in the path",
    ]


def test_a_root_with_a_release_ignore_honours_it(tmp_path, capsys):
    tree = make_tree(tmp_path / "tree", {".releaseignore": "private/\n", "private/a.md": "zebra\n"})
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    assert checker.main(["--root", str(tree), "--denylist", str(denylist)]) == 0
    assert capsys.readouterr().out == ""


def test_all_checks_the_tracked_files_only(tmp_path, capsys):
    tree = make_tree(tmp_path / "repository", {"tracked.md": "zebra\n", "untracked.md": "zebra\n"})
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-C", str(tree), "add", "tracked.md"], check=True)
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    assert checker.main(["--root", str(tree), "--all", "--denylist", str(denylist)]) == 1
    printed = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in printed] == ["tracked.md"]


def test_all_outside_a_repository_is_refused_rather_than_passing(tmp_path, capsys):
    tree = make_tree(tmp_path / "pack", {"notes.md": "zebra\n"})
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    assert checker.main(["--root", str(tree), "--all", "--denylist", str(denylist)]) == 2
    assert "git repository" in capsys.readouterr().err


def test_named_files_are_read_relative_to_the_root(tmp_path, capsys):
    tree = make_tree(
        tmp_path / "tree",
        {"a.md": "zebra\n", "b.md": "zebra\n", "src/addons/alpha/c.md": "zebra\n"},
    )
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    arguments = ["a.md", "src/addons/alpha/c.md", str(tree / "b.md")]
    assert checker.main(["--root", str(tree), "--denylist", str(denylist), *arguments]) == 1
    printed = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in printed] == ["a.md", "b.md"]


def test_a_named_file_outside_the_root_is_refused(tmp_path):
    tree = make_tree(tmp_path / "tree", {"a.md": "\n"})
    outside = tmp_path / "elsewhere.md"
    outside.write_text("\n")
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    assert checker.main(["--root", str(tree), "--denylist", str(denylist), str(outside)]) == 2


def test_nothing_to_check_and_all_with_names_are_usage_errors():
    with pytest.raises(SystemExit) as nothing:
        checker.main([])
    with pytest.raises(SystemExit) as both:
        checker.main(["--all", "README.md"])
    assert nothing.value.code == both.value.code == 2


def test_report_only_prints_the_findings_and_passes(tmp_path, capsys):
    tree = make_tree(tmp_path / "tree", {"notes.md": "zebra\n"})
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    assert checker.main(["--root", str(tree), "--denylist", str(denylist), "--report-only"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("notes.md:1: add-on term")
    assert "report-only" in captured.err


def test_a_clean_tree_passes_silently(tmp_path, capsys):
    tree = make_tree(tmp_path / "tree", {"notes.md": "nothing here\n"})
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    assert checker.main(["--root", str(tree), "--denylist", str(denylist)]) == 0
    assert capsys.readouterr() == ("", "")


def test_a_finding_prints_the_entry_id_never_the_term(tmp_path, capsys):
    tree = make_tree(
        tmp_path / "tree", {"notes.md": "Zebra and QUOKKA meet a purple giraffe within a decade.\n"}
    )
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    assert checker.main(["--root", str(tree), "--denylist", str(denylist)]) == 1
    captured = capsys.readouterr()
    output = (captured.out + captured.err).lower()
    for word in NEUTRAL_WORDS:
        assert word not in output
    lines = captured.out.splitlines()
    assert len(lines) == len(NEUTRAL_TERMS)
    for line in lines:
        assert re.fullmatch(
            r"notes\.md:1: add-on term, entry [0-9a-f]{12} \((sub-word|whole-word|phrase)\)", line
        )


def test_summary_counts_findings_per_top_level_directory(tmp_path, capsys):
    tree = make_tree(
        tmp_path / "tree",
        {
            "README.md": "zebra\nzebra\n",
            "docs/a.md": "zebra\n",
            "docs/b.md": "QUOKKA\n",
            "src/zebra.hpp": "\n",
        },
    )
    denylist = write_neutral_denylist(tmp_path / "denylist.json")
    arguments = ["--root", str(tree), "--denylist", str(denylist), "--summary", "--report-only"]
    assert checker.main(arguments) == 0
    assert capsys.readouterr().out.splitlines() == [
        "| Top-level directory | Findings | Files |",
        "| --- | ---: | ---: |",
        "| (top level) | 2 | 1 |",
        "| `docs/` | 2 | 2 |",
        "| `src/` | 1 | 1 |",
        "| Total | 5 | 4 |",
    ]
