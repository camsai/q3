#!/usr/bin/env python3
"""Ratchet the textual signals that code was derived from a reference implementation (PROVENANCE.md).

Q3 is Apache-2.0, and reference codes such as Quantum ESPRESSO are GPL: their
source may be read and cited, never translated (AGENTS.md § Licensed Material and
Provenance). A translation leaves marks that a search can find. The
``provenance_check`` package beside this script searches every file the public
export ships for three kinds:

``red_flag_phrase``
    The wording PROVENANCE.md § 4.2 forbids, about any reference: "transcribed
    / ported / translated / adapted / copied from" a source file or a
    reference's name, "follows <anything> line by line", "based on the
    implementation in".
``<language>_syntax``
    A reference's language spelled inside Q3's C++ or Python *code* (comments,
    strings and preprocessor lines removed), e.g. ``fortran_syntax``.
``<reference>_identifier``
    A reference's variable names used as words of a C++ identifier, e.g.
    ``quantum_espresso_identifier``.

The rules are abstract (``provenance_check/core.py``); what is unique to one
reference -- its names in prose, its languages, its identifiers -- is one file in
``provenance_check/references/``. Adding a reference adds a file there and
changes nothing else; a reference in a new language also adds a
``ForeignLanguage`` subclass beside ``provenance_check/fortran.py`` and registers
it in ``FOREIGN_LANGUAGES`` below.

None of this proves a file is clean. PROVENANCE.md § 4.4 says why: carried-over
*structure* has no wording to search for. This is a backstop for § 3's process,
not a substitute for it.

**Allowlist.** ``check_provenance_allowlist.json`` names files or directories (a
trailing ``/``) excused from some rules, each with a reason: format readers and
writers where a reference's name *is* the format (§ 5.1), and the files that
spell the signals on purpose (§ 5.4). An entry whose path no longer exists fails
the check.

**Baseline.** ``check_provenance_baseline.json`` holds today's findings as per-file
counts keyed ``"<rule>: <signal>"``, the same ratchet as
``check_cpp_type_queries.py``, with an add-on's findings in a baseline beside its
code. The check fails on any count above its baseline (a new finding) and on any
count below it (a stale entry: remove it with ``--write-baseline`` in the change
that earned it). ``--write-baseline`` refuses to raise a count unless
``--allow-increase`` is also given.

Usage::

    python scripts/repo/check_provenance.py                  # check (same as --check)
    python scripts/repo/check_provenance.py --write-baseline # tighten after a fix
    python scripts/repo/check_provenance.py --report         # totals by rule and signal
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from provenance_check.core import (
    Baseline,
    Counts,
    ForeignLanguage,
    Scanner,
    build_rules,
    count,
    load_allowlist,
    load_reference_sources,
    stale_allowlist_entries,
    total,
)
from provenance_check.fortran import FortranLanguage

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REFERENCES_DIRECTORY = SCRIPT_DIRECTORY / "provenance_check" / "references"
ALLOWLIST_PATH = SCRIPT_DIRECTORY / "check_provenance_allowlist.json"
BASELINE_PATH = SCRIPT_DIRECTORY / "check_provenance_baseline.json"

#: The foreign languages a reference may declare, by name. The core names none:
#: a reference written in a new language adds its subclass here.
FOREIGN_LANGUAGES: dict[str, ForeignLanguage] = {
    language.name: language for language in (FortranLanguage(),)
}


def report(measured: Counts, baseline: Counts, rule_names: list[str]) -> None:
    """Print totals per rule and per signal, then each file under its key."""
    per_key: Counter[str] = Counter()
    files_per_key: dict[str, list[str]] = {}
    for file, per_file in measured.items():
        per_key.update(per_file)
        for key, value in per_file.items():
            files_per_key.setdefault(key, []).append(f"{file} ({value})")
    for rule_name in rule_names:
        keys = sorted(key for key in per_key if key.startswith(rule_name + ": "))
        print(f"{rule_name:30s} {sum(per_key[key] for key in keys)}")
        for key in keys:
            print(f"  {key.split(': ', 1)[1]:28s} {per_key[key]}")
            for entry in files_per_key[key]:
                print(f"      {entry}")
    print(f"{'total':30s} {total(measured)} (baseline {total(baseline)})")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="compare against the baseline (the default)"
    )
    parser.add_argument("--write-baseline", action="store_true", help="rewrite the baseline")
    parser.add_argument(
        "--allow-increase",
        action="store_true",
        help="with --write-baseline, permit raising counts (needs review justification)",
    )
    parser.add_argument("--report", action="store_true", help="print totals by rule and signal")
    arguments = parser.parse_args(argv)

    rules = build_rules(load_reference_sources(REFERENCES_DIRECTORY, FOREIGN_LANGUAGES))
    rule_names = [rule.name for rule in rules]
    allowlist = load_allowlist(ALLOWLIST_PATH, set(rule_names))
    findings = Scanner(rules, allowlist).scan(REPOSITORY_ROOT)
    measured = count(findings)
    baseline_store = Baseline(BASELINE_PATH, REPOSITORY_ROOT)
    baseline = baseline_store.load()
    increases, decreases = Baseline.compare(measured, baseline)
    stale_allowlist = stale_allowlist_entries(allowlist, REPOSITORY_ROOT)

    if arguments.report:
        report(measured, baseline, rule_names)

    if arguments.write_baseline:
        if increases and not arguments.allow_increase:
            print("Refusing to raise the provenance baseline without --allow-increase:")
            for line in increases:
                print(f"  {line}")
            return 1
        baseline_store.write(measured)
        print(f"Wrote {BASELINE_PATH.name}: {total(measured)} provenance findings.")
        return 0

    if increases:
        print("New provenance signals (PROVENANCE.md § 3, § 4.2) -- rewrite the code or wording:")
        increased_keys = {tuple(line.rsplit(" ", 3)[0].split(": ", 1)) for line in increases}
        for line in increases:
            print(f"  {line}")
        for finding in findings:
            if (finding.path, finding.key) in increased_keys:
                print(f"    {finding.path}:{finding.line}: {finding.key}")
    if decreases:
        print(
            "Stale baseline entries -- tighten the ratchet with "
            "`python scripts/repo/check_provenance.py --write-baseline`:"
        )
        for line in decreases:
            print(f"  {line}")
    if stale_allowlist:
        print(f"Allowlist entries for paths that no longer exist ({ALLOWLIST_PATH.name}):")
        for path in stale_allowlist:
            print(f"  {path}")
    return 1 if increases or decreases or stale_allowlist else 0


if __name__ == "__main__":
    sys.exit(main())
