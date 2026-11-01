"""tests/python/test_unique_test_basenames.py

Every ``test_*.py`` under ``tests/`` has a basename no other test file has.

The test directories are not packages, so pytest's default import mode puts
each file's directory on ``sys.path`` and imports it under its bare basename.
Two files with the same name collide: the second one collected fails with
"import file mismatch" and the whole session stops at collection. A pull
request's CI runs its test directories one at a time and never sees it; only
``pytest tests/``, which the scheduled tier-2 job runs, collects both. That is
how ``tests/python/test_wannier_campaign.py`` and
``tests/validation/test_wannier_campaign.py`` kept tier 2 from running at all
from 2026-10-15 to 2026-11-01. This makes the same collision fail on a pull
request instead.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

TESTS_DIRECTORY = Path(__file__).resolve().parents[1]


def test_no_two_test_files_share_a_basename():
    paths_by_name: dict[str, list[str]] = defaultdict(list)
    for path in TESTS_DIRECTORY.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        paths_by_name[path.name].append(str(path.relative_to(TESTS_DIRECTORY)))
    duplicates = {name: sorted(paths) for name, paths in paths_by_name.items() if len(paths) > 1}
    assert not duplicates, f"test files that share a basename (rename one of each): {duplicates}"
