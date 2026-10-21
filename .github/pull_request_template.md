## What & why

<!-- What changes, and what problem it solves. Link the issue if one exists. -->

## How it was tested

<!-- Commands run and platforms covered, e.g.:
     ctest --output-on-failure --test-dir build
     pytest tests/ -v --q3-bin ./build/q3 -->

## Provenance

<!-- See PROVENANCE.md § 3. A session is either an implementation session (never
     fetched or opened GPL or proprietary source) or a research session (may
     read GPL source; produced prose notes only, no code). -->

**Session type (implementation/research):**

**Sources consulted:**

<!-- Papers with equation numbers, public manuals, other engines' output files,
     research notes this change was written from. Name GPL files or routines
     only if a research session read them. Write "Q3 only" if nothing else. -->

## Checklist

-   [ ] Every commit is signed off under the DCO (`git commit -s`; see [CONTRIBUTING.md](CONTRIBUTING.md))
-   [ ] The change honors the provenance policy in [CONTRIBUTING.md](CONTRIBUTING.md): nothing copied, translated, or closely paraphrased from license-incompatible sources
-   [ ] The checklist in [PROVENANCE.md](PROVENANCE.md) § 7 holds, including the [provenance register](docs/provenance_register.md) row of any module this change adds or rewrites
-   [ ] `pre-commit run --all-files` is green
-   [ ] C++ tests pass (`ctest --test-dir build`); physics changes keep the validation suite green (`pytest tests/ --q3-bin ./build/q3`)
-   [ ] Documentation updated where behavior changed (README, FUNCTIONALITY.md, docs/io_reference.md)
