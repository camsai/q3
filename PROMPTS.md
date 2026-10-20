# Prompts

Q3 is written primarily by AI coding agents, working from the conventions in
[AGENTS.md](AGENTS.md) and [AGENTS-generic.md](AGENTS-generic.md) and the plans
that precede each change. This file keeps the prompt that started the current
codebase, as a record of where the design came from.

## 1. Initial prompt

### 1.1. Problem statement

An earlier Q3 implementation had gone too far without reaching a minimal viable
phase-0 product — a working code that can run a single self-consistent
calculation for a crystal and produce energy, charge density and forces.
Further implementation would have required many files that were not yet part of
the core functionality, complicating debugging and slowing development.

### 1.2. Solution

Re-implement a minimal version of the code with the following goals:

-   produce a code that can run a single self-consistent calculation for a
    crystal and produce energy, charge density and forces — see the specific
    set of functionalities in [FUNCTIONALITY.md](FUNCTIONALITY.md)
-   the code should be well-documented and easy to understand — follow and
    expand the `AGENTS*.md` conventions
-   the code should be well-tested and easy to contribute to
-   the code should retain the original goals as outlined in
    [SUMMARY.md](SUMMARY.md)

### 1.3. Task

Draft an implementation plan explaining what to retain from the earlier
implementation and what to discard or defer to phase 1+.

## 2. Since then

That plan produced the layered architecture described in
[docs/architecture.md](docs/architecture.md). Subsequent work follows the same
pattern: a plan document first, then implementation against it, then the
handoff rules in [AGENTS.md](AGENTS.md) § Documentation.
