"""tests/python/test_check_statistics_untouched.py — Unit tests for the STATISTICS.md guard.

Covers ``scripts/repo/check_statistics_untouched.py``: both modes (staged index and
pull-request range), the generator-changed exemption and the freshness condition
attached to it, and the property that makes the pre-commit hook usable —
``pre-commit run --all-files`` passes every file in the tree, so a guard that
read its argument list instead of the index would fail every lint run.

Each case builds a throwaway repository, because what the checker reads is git's
answer, not the filesystem's.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "check_statistics_untouched.py"
GENERATOR_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "generate_statistics.py"


def _load_checker():
    """Import the checker script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("check_statistics_untouched", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def git(repository, *arguments):
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True, capture_output=True, text=True
    )


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """A repository with one commit, a real generator, and the checker pointed at it."""
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")

    (tmp_path / "scripts" / "repo").mkdir(parents=True)
    (tmp_path / "scripts" / "repo" / "generate_statistics.py").write_text(
        GENERATOR_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "thing.hpp").write_text("int thing;\n")
    subprocess.run(
        [sys.executable, "scripts/repo/generate_statistics.py"], cwd=tmp_path, check=True
    )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-m", "base")

    monkeypatch.setattr(checker, "PROJECT_ROOT", tmp_path)
    return tmp_path


def run_staged():
    return checker.check(staged=True, range_specifier=None)


def run_range(range_specifier):
    return checker.check(staged=False, range_specifier=range_specifier)


# ── The staged-index mode ─────────────────────────────────────────────────────


def test_a_commit_that_does_not_touch_the_file_passes(repository):
    (repository / "src" / "other.hpp").write_text("int other;\n")
    git(repository, "add", "src/other.hpp")

    assert run_staged() == 0


def test_staging_the_generated_file_alone_is_refused(repository):
    """The whole point: this is the change that makes two branches conflict."""
    (repository / "STATISTICS.md").write_text("hand written\n")
    git(repository, "add", "STATISTICS.md")

    assert run_staged() == 1


def test_nothing_staged_is_nothing_to_say(repository):
    """`pre-commit run --all-files` stages nothing, and must not fail on that."""
    (repository / "STATISTICS.md").write_text("drifted\n")

    assert run_staged() == 0, "an unstaged change is not this check's business"


# ── The generator-changed exemption ───────────────────────────────────────────


def test_the_file_may_move_with_the_generator_that_writes_it(repository):
    generator = repository / "scripts" / "repo" / "generate_statistics.py"
    generator.write_text(
        generator.read_text().replace("| C++ standard | C++17 |", "| C++ standard | C++20 |")
    )
    subprocess.run(
        [sys.executable, "scripts/repo/generate_statistics.py"], cwd=repository, check=True
    )
    git(repository, "add", "STATISTICS.md", "scripts/repo/generate_statistics.py")

    assert run_staged() == 0


def test_the_exemption_still_requires_the_file_to_match_the_new_generator(repository):
    """Staging the generator is not a licence to hand-write the table."""
    generator = repository / "scripts" / "repo" / "generate_statistics.py"
    generator.write_text(
        generator.read_text().replace("| C++ standard | C++17 |", "| C++ standard | C++20 |")
    )
    (repository / "STATISTICS.md").write_text("whatever I like\n")
    git(repository, "add", "STATISTICS.md", "scripts/repo/generate_statistics.py")

    assert run_staged() == 1


# ── The pull-request range mode ───────────────────────────────────────────────


def test_a_branch_carrying_the_file_is_refused_over_a_range(repository):
    git(repository, "checkout", "-b", "feature")
    (repository / "STATISTICS.md").write_text("carried\n")
    git(repository, "add", "STATISTICS.md")
    git(repository, "commit", "-m", "carry the generated file")

    assert run_range("main...feature") == 1


def test_a_branch_that_leaves_it_alone_passes_over_a_range(repository):
    git(repository, "checkout", "-b", "feature")
    (repository / "src" / "other.hpp").write_text("int other;\n")
    git(repository, "add", "src/other.hpp")
    git(repository, "commit", "-m", "add source, leave the table alone")

    assert run_range("main...feature") == 0


def test_the_range_ignores_what_the_base_branch_did_to_the_file(repository):
    """Three-dot semantics: only the branch's own changes are the branch's problem."""
    git(repository, "checkout", "-b", "feature")
    (repository / "src" / "other.hpp").write_text("int other;\n")
    git(repository, "add", "src/other.hpp")
    git(repository, "commit", "-m", "add source")

    git(repository, "checkout", "main")
    (repository / "STATISTICS.md").write_text("refreshed on main by the bot\n")
    git(repository, "add", "STATISTICS.md")
    git(repository, "commit", "-m", "chore(stats): refresh STATISTICS.md")

    assert run_range("main...feature") == 0, "the refresh commit on main is not the branch's doing"


# ── Each add-on's own file ────────────────────────────────────────────────────


def test_an_addons_statistics_file_alone_is_refused(repository):
    """An add-on's STATISTICS.md is generated too, and held to the same rule."""
    (repository / "src" / "addons" / "alpha").mkdir(parents=True)
    (repository / "src" / "addons" / "alpha" / "STATISTICS.md").write_text("hand written\n")
    git(repository, "add", "src/addons/alpha/STATISTICS.md")

    assert run_staged() == 1


def test_an_addons_file_may_move_with_the_generator_when_current(repository):
    (repository / "src" / "addons" / "alpha").mkdir(parents=True)
    (repository / "src" / "addons" / "alpha" / "reader.hpp").write_text("int alpha;\n")
    generator = repository / "scripts" / "repo" / "generate_statistics.py"
    generator.write_text(
        generator.read_text().replace("| C++ standard | C++17 |", "| C++ standard | C++20 |")
    )
    subprocess.run(
        [sys.executable, "scripts/repo/generate_statistics.py"], cwd=repository, check=True
    )
    git(repository, "add", "-A")

    assert run_staged() == 0


# ── The wiring ────────────────────────────────────────────────────────────────


def test_the_pre_commit_hook_is_wired_the_way_the_checker_needs():
    """`pass_filenames: false` is load-bearing: see test_nothing_staged_is_nothing_to_say."""
    configuration = (REPOSITORY_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    hook = configuration.split("- id: statistics-untouched", 1)
    assert len(hook) == 2, "the statistics-untouched hook is gone from .pre-commit-config.yaml"
    body = hook[1].split("- id: ", 1)[0]
    assert "check_statistics_untouched.py --staged" in body
    assert "pass_filenames: false" in body
    assert "always_run: true" in body


def test_the_guard_workflow_checks_the_pull_requests_own_changes():
    """Two-dot semantics here would blame a branch for what main did to the file."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/statistics-guard.yml").read_text(
        encoding="utf-8"
    )
    assert "check_statistics_untouched.py" in workflow
    range_argument = workflow.split("--range", 1)[1].splitlines()[0]
    assert "..." in range_argument, f"two-dot range in the guard workflow: {range_argument}"
