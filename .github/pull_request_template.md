## What & why

<!-- What changes, and what problem it solves. Link the issue if one exists. -->

## How it was tested

<!-- Commands run and platforms covered, e.g.:
     ctest --output-on-failure --test-dir build
     pytest tests/ -v --q3-bin ./build/q3 -->

## Checklist

-   [ ] Every commit is signed off under the DCO (`git commit -s`; see [CONTRIBUTING.md](CONTRIBUTING.md))
-   [ ] The change honors the provenance policy in [CONTRIBUTING.md](CONTRIBUTING.md): nothing copied, translated, or closely paraphrased from license-incompatible sources
-   [ ] `pre-commit run --all-files` is green
-   [ ] C++ tests pass (`ctest --test-dir build`); physics changes keep the validation suite green (`pytest tests/ --q3-bin ./build/q3`)
-   [ ] Documentation updated where behavior changed (README, FUNCTIONALITY.md, docs/io_reference.md)
