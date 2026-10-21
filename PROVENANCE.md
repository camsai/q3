# Provenance: how to write code that stays permissively licensed

Q3 ships under Apache-2.0. That is only true if nothing in the tree is a
derivative work of code under an incompatible license. This document explains
the mechanism, draws the line between what is and is not a derivative work,
and gives a procedure that keeps you on the right side of it.

It binds humans and AI coding agents equally. `CONTRIBUTING.md` § "Provenance
policy" states the rules; this file is the reasoning and the recipe behind
them.

> Engineering guidance, not legal advice. When a specific case is genuinely
> unclear, stop and escalate rather than deciding alone.

---

## 1. Why this matters

Computational materials science is a small field with a few reference
implementations, and the
useful ones are either **strong-copyleft open source** (GPLv2-or-later) or
**proprietary and licensed**. Both create problems, for opposite reasons.

**Copyleft.** If a work is a derivative of GPL code, the whole work must be
distributed under the GPL. The incompatibility runs one way: Apache-2.0 code
can be absorbed into a GPL project, but GPL code cannot be absorbed into an
Apache one. So a single GPL-derived file makes two things true at once:

1. Distributing it under Apache-2.0 infringes the original author's copyright.
2. Every downstream user was granted rights — permissive reuse, a patent
   grant, no copyleft — that we had no authority to give.

The second is the one that ends adoption. "This Apache project may be
GPL-encumbered" is the single sentence that makes an industrial legal team say
no, and it is not repairable by apologising later.

**Proprietary.** Licensed source usually carries confidentiality obligations
as well as copyright. Consulting it while implementing a competing feature
creates exposure that no amount of later rewriting removes, because the
problem is the act, not the artifact. See § 6.

---

## 2. What is and is not a derivative work

Courts filter a program into layers, discard everything unprotectable, and
compare only what is left: ideas, anything dictated by external constraints or
efficiency, and public-domain material all fall out first.

### Not derivative — all of this is fine

-   **Implementing the physics from the published paper.** Mathematics is not
    copyrightable. Equation (11) of a 1999 PRL is free to implement.
-   **Physical constants and reference data.** Facts. Cite the standard body
    (CODATA) or the paper the parametrisation came from.
-   **Matching a numerical convention** — unit systems, harmonic ordering,
    sign choices, energy zeros — so results are comparable.
-   **Reading another code to understand an algorithm**, then writing your
    own. Reading is not infringing.
-   **Producing identical numerical output.** Behaviour is not expression.
-   **Implementing a file format**, including from the reference
    implementation that defines it (§ 5.1).

### Potentially derivative — this is where the risk is

Copying the **expression**, meaning:

-   Function decomposition that the mathematics did not force
-   Variable and routine naming carried across
-   Control-flow shape: the same loop nesting, the same branch order, the same
    intermediate temporaries in the same sequence
-   The specific ordering of operations where real alternatives existed

### The translation trap

**Translating code is explicitly a derivative work**, exactly as translating a
novel is. Rewriting a Fortran routine in C++ launders nothing. This is the
single most likely way to create a problem in a codebase like this one, and
the reason the red-flag phrases in § 4.2 are forbidden: they describe the act
that creates a derivative.

### Where the line actually falls

For a **closed-form expression**, there is essentially one reasonable way to
write it — the expression merges with the idea, and it is not protectable. A
smearing function or a Bessel recurrence transcribed from anywhere is almost
certainly fine on the merits.

For a **200-line routine with a distinctive internal structure**, it is not
fine. The more scaffolding a piece of code has — ordering, decomposition,
bookkeeping — the more of it is protectable expression rather than
mathematics.

---

## 3. The recipe

Follow this whenever you implement anything that another computational
materials science code also does.

### Step 1 — Start from the literature, not from code

Find the paper. Nearly every algorithm here has one: a smearing scheme, a
mixing method, a correction term, a PAW construction. Implement from the
mathematical statement. If you cannot find a published description, that is
itself a signal — ask before proceeding.

### Step 2 — Decide whether you may consult another implementation at all

| Source                               | May you read it?                                   |
| ------------------------------------ | -------------------------------------------------- |
| Public, permissively licensed        | Yes, freely                                        |
| Public, strong copyleft (GPL)        | In a research session only (below) — never to copy |
| Proprietary / licensed source        | **No.** Do not open it. See § 6                    |
| Public documentation, manuals, wikis | Yes                                                |
| Another engine's **output** files    | Yes — outputs are data, not expression             |

### Step 3 — Separate reading from writing

If you consulted another implementation: **close it before you write.** Do not
have it open in a second buffer, and do not paste it into a scratch file to
"work from". Write from your understanding plus the paper.

For an AI agent this is a hard rule with a specific failure mode: never place
license-incompatible source into your context and then generate the
corresponding implementation in the same session. That is machine-assisted
translation, and it produces a derivative work. The two kinds of session below
make this rule operational.

### Step 4 — Write it in this codebase's own idiom

Use the naming conventions in `AGENTS.md` (no abbreviations), the layer
structure, and the decomposition that fits Q3 — not the one that fitted the
other code. **If your implementation's structure closely mirrors theirs,
treat that as a signal to restructure**, not as confirmation that you got it
right.

### Step 5 — Validate black-box

Compare **numbers**, not code. Q3's cross-code gates already do this: same
inputs, compare energies, forces, stresses to a tolerance. Numerical agreement
is the evidence that the implementation is correct, and it says nothing about
how either side is written. This is why validating against a reference code
is not only safe but desirable.

### Step 6 — Cite honestly, in the approved form

Provenance comments are an asset. Keep writing them — just describe what
actually happened (§ 4).

### Research sessions and implementation sessions

Step 3 separates reading from writing. For AI-assisted work that separation is
a property of the **session**: one conversation context, including everything
carried into it — summaries of earlier context, forks of another session, and
files or notes it was handed. A person's session is a stretch of work on one
task. Every session is exactly one of two kinds, decided before it starts.

-   **Research session.** May read GPL source, published papers, manuals, and
    other engines' outputs. It produces **prose notes only**: the equations,
    conventions, and numbers it found, each with its citation (a GPL file or
    routine name is fine as a citation). It writes no code and no
    pseudo-code that follows the source's structure, and it quotes no source.
    It never opens proprietary source (§ 6).
-   **Implementation session.** Writes Q3 code from papers, public
    documentation, other engines' outputs, Q3's own code, and research notes.
    It **never fetches or opens GPL or proprietary source** — not by reading a
    local checkout, not by fetching a web page that shows it, and not by
    accepting it pasted from elsewhere. When it needs something only the source
    can answer, it stops and asks for a research session; it does not look
    itself.

A session does not change kind partway through. A research session that finds
a defect reports it; the fix is written in a separate implementation session.
A fork or continuation of a research session is a research session.

Research notes pass to an implementation session as prose. If a note quotes
source code, it is not a research note: remove the quotation before it is
handed over.

### Side-by-side reviews

The check § 4.4 says nothing cheaper can replace — reading an implementation
next to the code it might derive from — is a **research session**, done by
someone who **did not write the code under review**: a person, or an AI
session run by someone other than the module's author. The reviewer compares
structure (decomposition, naming, control flow, the order of operations where
alternatives existed; § 2), not numbers. Reviews are recorded per module in
the [provenance register](docs/provenance_register.md), which also lists each
module's published sources and validation gates; § 7 says when to update it.

### AI tools and licensed data

-   **Never place licensed dataset contents in an AI session's context.** The
    format-level identity facts that may be committed — a dataset's title line,
    its valence charge, its SHA-256 digest — are the only parts of a licensed
    dataset an AI session may see. Point tools at the dataset through the
    environment variable that locates it; do not read the file into the
    session.
-   **Use AI tools for engine-adjacent work only under no-training and
    no-retention terms.** Engine-adjacent work is any work on a module that
    another computational materials science code also implements, or on parity
    with such a code. Check the tool's terms before the session starts, not
    after.

---

## 4. Comment wording

### 4.1 Approved forms

```
Implements Eq. (11) of Author, Journal Vol, Page (Year).
Cross-checked numerically against <Engine> <version>.
Matches <Engine>'s convention, which <states the convention explicitly>.
Constants are CODATA <year> values.
Derived from the dataset file itself: <the observable fact>.
Established empirically by <the gate or experiment that pinned it>.
```

Naming another engine as a **reference you validated against**, or whose
**convention you matched**, is accurate, useful, and safe. Keep doing it.

### 4.2 Red-flag phrases — never write these

```
transcribed from <file>          ported from <file>
translated from <file>           adapted from <file>
copied from <file>               follows <file> line by line
based on the implementation in <file>
```

Each describes creating a derivative work. If one of these is the honest
description of what you did, the problem is the code, not the comment — go
back to § 3.

### 4.4 What a search cannot catch

The red-flag list is a **wording** check, and wording is the easy half. A
mechanical search finds phrases and recognisable foreign syntax. It does not
find carried-over _structure_ — and structure is what makes a work derivative.

A concrete case from this codebase: a verbatim line of upstream Fortran
survived in a symmetry routine as an **array-index expression**. It contained
no red-flag phrase and no distinctive foreign syntax, so both a
transcription-wording search and a Fortran-syntax search were structurally
blind to it. It was found by reading the implementation against the upstream
source, and nothing cheaper would have found it.

So treat § 7's checklist as a floor, not a proof. The only reliable check is
reading the implementation next to the thing it might derive from — which is
why § 3's discipline (write from the paper, close the source before writing)
matters far more than any audit run afterwards. An audit is a backstop for
process failures, not a substitute for the process.

### 4.3 Do not "fix" this by deleting citations

Removing a provenance comment does not change whether the code is a
derivative. It removes the evidence that provenance was tracked in good faith,
and if the question is ever raised, a deletion reads as concealment rather
than diligence. **Rewrite the code if it needs rewriting; never quietly strip
the trail.**

---

## 5. Special cases

### 5.1 File formats

Citing the reference implementation of an on-disk format — including specific
line numbers — is **allowed**. A format is a specification, not an algorithm:
implementing it is interoperability, and removing such citations would make
the readers and writers unmaintainable with no offsetting benefit. Existing
citations of this kind stay by owner decision.

**Cite the published specification first, and the converter only for what it
leaves open.** A published spec is strictly better provenance than a
converter's source: it removes the licensing question entirely, and it does
not rot the way a citation pinned to a dated checkout of a moving branch does.
Where a format has no spec, or the spec is silent on a detail, the program that
writes the files is the authority. Cite it by release tag, subroutine and
record, never by line number on a development branch, and state what the code
computes in this codebase's notation, never as a quoted expression in the
other code's variable names.

The BerkeleyGW export resolved this question on 2026-10-20 as follows:

-   **The spec.** BerkeleyGW publishes the format of its WFN, RHO and VXC files
    and of `vxc.dat` in J. Deslippe et al., Comput. Phys. Commun. 183, 1269
    (2012), Appendix 11.1 ([arXiv:1111.4429](https://arxiv.org/abs/1111.4429)).
    It gives each header record's contents, order and units, the body layout,
    the normalization of RHO's G = 0 component, and `vxc.dat`'s line contents
    in eV. The BerkeleyGW 4.0 manual specifies only the HDF5 variant
    ([wfn_h5_spec](https://manual.berkeleygw.org/4.0/wfn_h5_spec/)). The
    readers and writers in `src/io/mean_field_export/berkeleygw/` and
    `tests/helpers_berkeleygw.py` cite the appendix as their authority.
-   **What the spec leaves open.** It does not give the Fortran record framing,
    the integer and real kinds, the 32-byte character fields and their date
    format, the count repeated before each body block, `vxc.dat`'s column
    widths, or the exact transposition and pairing of the symmetry records.
    For these the citation is Quantum ESPRESSO's `pw2bgw.x` at release 7.5 (tag
    `qe-7.5`, the version of the local reference build), named by subroutine
    (`write_wfng`, `write_rhog`, `write_vxc_g`, `write_vxc_r`) and record.
    The symmetry conventions are also pinned by unit tests against that
    release's output.
-   **What was retired.** About a hundred line-number citations pinned to a
    2026-07-27 `develop` checkout, and the Fortran expressions quoted with QE's
    variable names, were replaced by the record they describe and the
    mathematics it carries. No citation was removed without a replacement.

### 5.2 Reproduction commands

Instructions for running another engine to regenerate reference data —
executable names, input decks, build paths — reference **programs, not
source**. Keep them: without them the validation data cannot be reproduced.

### 5.3 Dependencies: linking versus bundling

A GPL **dependency** is not a provenance problem for our source, but it is a
distribution problem for our binaries:

-   **Source release** — unaffected. Users link the dependency themselves.
-   **Binaries and containers that bundle a GPL library** — the shipped
    artifact is a combined work and falls under that library's terms,
    regardless of what our source license says.

So a permissively licensed backend must be the default in anything we publish
as a binary. Check this before adding a dependency, not after.

### 5.4 Test fixtures

Tests that deliberately contain forbidden strings — to prove a guard rejects
them — are correct and must not be "cleaned". They are the guard's evidence
that it still works.

---

## 6. Proprietary source: a stricter rule

For licensed, non-public source, the rule is not "be careful", it is **do not
open it**. Ground the work instead in:

-   Published papers describing the method
-   Public documentation, manuals, and wikis
-   Facts observable in the data files themselves
-   The engine's **output** files
-   Our own experiments, recorded so the derivation can be reconstructed

This is stricter than the GPL rule because the obligation is different: a
confidentiality term is breached by the access, not by the artifact, so there
is no later rewrite that undoes it.

---

## 7. Checklist before you open a pull request

-   [ ] Every non-obvious formula cites a paper, a standard, or an experiment
-   [ ] No red-flag phrase from § 4.2 appears in the diff
-   [ ] No file, routine, or variable name from proprietary source appears
-   [ ] Structure follows Q3's conventions, not another code's
-   [ ] Cross-code validation compares numbers, not implementations
-   [ ] Any new dependency's license is compatible with how we distribute
-   [ ] If you consulted another implementation, you closed it before writing
-   [ ] The pull request states its session type (implementation or research)
        and the sources consulted
-   [ ] No implementation session behind the change fetched or opened GPL or
        proprietary source; any research session behind it left prose notes
        only, with no code
-   [ ] No licensed dataset contents entered an AI session's context
-   [ ] AI tools used for engine-adjacent work ran under no-training and
        no-retention terms
-   [ ] If the change adds or rewrites a module in the
        [provenance register](docs/provenance_register.md), its row is added
        or updated, and a rewritten module's review is reset to "not yet
        reviewed"

This list is a floor, not a proof — see § 4.4 for what it cannot catch.

---

## 8. If you think a line was crossed

Do not delete the comment, and do not quietly rewrite the file.

1. Stop.
2. Leave the provenance comment in place — it is the record.
3. Flag it to a maintainer with the specific file and what you consulted.

An honest report is cheap to fix. A concealed one is not, and concealment is
the part that turns a technical question into a credibility problem.
