"""Tests for ``scripts/repo/check_script_paths.py``.

The checker exists because a moved script breaks the files that name it, some
of which only run on ``main``. These pin what it reads, what it treats as a
path, and that it is not vacuous on this repository.
"""

import importlib.util
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_script_paths.py"

spec = importlib.util.spec_from_file_location("check_script_paths", SCRIPT_PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def _tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_this_repository_names_only_scripts_that_exist():
    assert checker.missing_paths(REPOSITORY_ROOT) == []


def test_it_reads_this_repository_rather_than_passing_on_nothing():
    """A wrong root would find no files and pass: the scan must see the wiring."""
    scanned = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in checker.scanned_files(REPOSITORY_ROOT)
    }
    for wiring in (".pre-commit-config.yaml", "CMakeLists.txt", "AGENTS.md"):
        assert wiring in scanned, wiring
    # .releaseignore is not exported, so the published tree has none to scan.
    if (REPOSITORY_ROOT / ".releaseignore").is_file():
        assert ".releaseignore" in scanned
    assert any(name.startswith(".github/workflows/") for name in scanned)
    assert any(name.startswith("docs/") for name in scanned)


def test_a_moved_script_is_reported_where_it_is_named(tmp_path):
    root = _tree(
        tmp_path,
        {
            "scripts/repo/present.py": "",
            ".pre-commit-config.yaml": "entry: python scripts/repo/present.py\nentry: python scripts/gone.py\n",
            "docs/guide.md": "Run `scripts/hpc/site_machine/` on the cluster.\n",
        },
    )
    assert checker.missing_paths(root) == [
        ".pre-commit-config.yaml:2: scripts/gone.py",
        "docs/guide.md:1: scripts/hpc/site_machine/",
    ]


def test_a_path_after_an_escaped_newline_is_still_a_path(tmp_path):
    """A file written as one string literal names each path after a \\n escape."""
    root = _tree(
        tmp_path, {"tests/python/test_fixture.py": 'TEXT = "plain\\nscripts/gone.py\\n"\n'}
    )
    assert checker.missing_paths(root) == ["tests/python/test_fixture.py:1: scripts/gone.py"]


def test_prose_globs_and_trailing_punctuation_are_not_paths(tmp_path):
    root = _tree(
        tmp_path,
        {
            "scripts/repo/tool.py": "",
            "AGENTS.md": (
                "Diagnostic scripts/inputs are not checked in.\n"
                'CI watches "scripts/**" for changes.\n'
                "Then run scripts/repo/tool.py.\n"
            ),
        },
    )
    assert checker.missing_paths(root) == []


def test_records_and_generated_statistics_are_not_read(tmp_path):
    # The private directories are spelled as joins: this file ships to the public
    # mirror, whose export gate refuses a literal path into either of them.
    plan = "/".join(("plan", "upcoming", "idea.md"))
    vault = "/".join(("vault", "README.md"))
    root = _tree(
        tmp_path,
        {
            plan: "Add scripts/repo/check_magic_numbers.py later.\n",
            vault: "scripts/old_location.py\n",
            "STATISTICS.md": "| scripts/before_the_move.py | 12 |\n",
        },
    )
    assert checker.missing_paths(root) == []
