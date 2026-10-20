"""Check that commits carry a Developer Certificate of Origin sign-off.

Two modes:

- ``--range BASE..HEAD``: verify that every non-merge commit in the range
  carries a ``Signed-off-by`` trailer. Used by the DCO workflow
  (.github/workflows/dco.yml) on pull requests.
- ``--commit-msg-file PATH``: verify a single commit-message file. Used by
  the ``dco-signoff`` pre-commit hook at the commit-msg stage, so a missing
  sign-off is caught at commit time rather than in CI.

CONTRIBUTING.md explains the policy; sign off with ``git commit -s``.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

SIGN_OFF_PATTERN = re.compile(r"^Signed-off-by: .+ <.+@.+>", re.MULTILINE)


def commit_message_has_sign_off(message: str) -> bool:
    return SIGN_OFF_PATTERN.search(message) is not None


def check_commit_message_file(path: Path) -> int:
    message = path.read_text(encoding="utf-8")
    # Strip the comment lines git appends to the message template.
    message = "\n".join(line for line in message.splitlines() if not line.startswith("#"))
    if commit_message_has_sign_off(message):
        return 0
    print(
        "Commit message lacks a 'Signed-off-by' trailer.\n"
        "Sign off with `git commit -s` (see CONTRIBUTING.md, DCO section)."
    )
    return 1


def check_range(range_expression: str) -> int:
    # Git expands %x00/%x1e to the raw bytes in its output; the escapes are
    # spelled out here because argv strings cannot themselves contain nulls.
    record_separator = "\x1e"
    unit_separator = "\x00"
    log_output = subprocess.run(
        [
            "git",
            "log",
            "--no-merges",
            "--format=%H%x00%B%x1e",
            range_expression,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    missing = []
    for record in log_output.split(record_separator):
        record = record.strip()
        if not record:
            continue
        commit_hash, _, message = record.partition(unit_separator)
        if not commit_message_has_sign_off(message):
            missing.append(commit_hash.strip())
    if missing:
        print(f"Commits in {range_expression} missing a 'Signed-off-by' trailer:")
        for commit_hash in missing:
            print(f"  {commit_hash}")
        print(
            f"\n{len(missing)} commit(s) missing sign-off. "
            "Sign off with `git commit -s` and rewrite the branch "
            "(e.g. `git rebase --exec 'git commit --amend --no-edit --signoff' <base>`)."
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--range", dest="range_expression", help="git revision range")
    group.add_argument(
        "--commit-msg-file",
        type=Path,
        help="path to a commit message file (pre-commit commit-msg stage)",
    )
    arguments = parser.parse_args()
    if arguments.range_expression:
        return check_range(arguments.range_expression)
    return check_commit_message_file(arguments.commit_msg_file)


if __name__ == "__main__":
    sys.exit(main())
