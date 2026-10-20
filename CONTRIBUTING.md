# Contributing to Q3

Thank you for contributing. This document covers the legal and provenance
requirements for contributions; engineering conventions live in
[AGENTS.md](AGENTS.md), and build/test instructions in the
[README](README.md).

## License and sign-off

Q3 is licensed under Apache-2.0 (see [LICENSE.md](LICENSE.md); copyright and
licensing for every file, including vendored third-party data, are declared
in [REUSE.toml](REUSE.toml)). By contributing, you agree that your
contributions are licensed under the same terms (inbound = outbound).

Every commit must be signed off under the **Developer Certificate of Origin**
(DCO):

```bash
git commit -s
```

This appends a `Signed-off-by: Your Name <you@example.com>` trailer certifying
the statement below. Commits without a sign-off will not be merged: every
pull request runs the DCO check (`.github/workflows/dco.yml`), and
`pre-commit install` also installs a commit-msg hook that catches a missing
sign-off locally at commit time.

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Provenance policy (hard rule)

[PROVENANCE.md](PROVENANCE.md) explains the reasoning behind these rules —
what counts as a derivative work, what does not, and a procedure for staying
on the right side of the line. Read it before implementing anything another
computational materials science code also does.

Q3 is an independent, from-scratch implementation. Its ability to remain permissively
licensed depends on every contribution honoring these rules:

1. **Never copy, translate, or closely paraphrase code** from GPL-licensed or
   proprietary electronic-structure codes, or from any source whose license is
   incompatible with Apache-2.0.
2. **Acceptable sources** are published papers, textbooks, standards
   documents, public algorithm descriptions, and permissively licensed code
   whose license and attribution requirements are satisfied.
3. **Physical-constant and reference-data tables** must be re-derived from
   primary literature and must cite that literature in the file header. The
   documented-provenance pattern in
   `src/layer_4_computation/effective_potential/internal/dispersion/d3bj_reference_data.hpp`
   is the house standard.
4. **Cross-code validation is encouraged; cross-code copying is not.**
   Comparing Q3's numerical output against another code's output is fine.
   Reading incompatible source code to write the corresponding Q3 feature is
   not.

## AI-assisted contributions

AI coding agents are first-class contributors to Q3 (see
[AGENTS.md](AGENTS.md)). Two rules keep that compatible with the provenance
policy:

-   The human who submits the change signs off under the DCO and is responsible
    for its provenance, exactly as for hand-written code.
-   These provenance rules bind the AI workflow too: do not provide
    license-incompatible source code to a model as reference material for
    implementing a Q3 feature.

## Mechanical requirements

-   New first-party files need no license boilerplate: copyright and licensing
    are declared for the whole tree in [REUSE.toml](REUSE.toml). Vendoring
    third-party files is the exception — add an annotation block naming their
    real copyright holder and license.
-   Install the hooks (`pre-commit install`) and keep them green; CI runs
    `pre-commit run --all-files`.
-   Do not include `STATISTICS.md` in a pull request. It is generated, it counts
    the whole tree, and so it moves on the same few lines in every branch at
    once — carrying it is what made concurrent pull requests conflict on a file
    no one authors. It is refreshed automatically on `main` instead, and the
    numbers will be right one push after your change merges.
-   Changes touching physics must keep the validation suite green
    (`pytest tests/ --q3-bin ./build/q3`); see the README for tiers.
-   Every pull request compiles and runs the C++ suite
    (`ctest --test-dir build`), the same way a merge to `main` does. A green
    tick on a pull request therefore means the tests passed, not that they
    were skipped — run `ctest` locally before pushing rather than finding out
    on the runner.
