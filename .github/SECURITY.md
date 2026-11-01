# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately
through GitHub's private vulnerability reporting: on
[camsai/q3](https://github.com/camsai/q3), open the **Security** tab and choose
**Report a vulnerability**. The report is visible only to the maintainers.

Include what you found, the version or commit it affects (`q3 --version` prints
both), and the steps or input that reproduce it. We aim to acknowledge a report
within five working days and to agree a disclosure date with you once a fix is
ready.

## Supported versions

Q3 is released continuously from `main`, and each release is tagged by date.
Fixes land on `main` and ship in the next release; older releases are not
patched. The container images (`ghcr.io/camsai/q3`) are rebuilt with every
release.

## Scope

Q3 is a simulation engine that reads input decks, pseudopotentials and
structure files you supply. Reports are welcome for anything that lets a
crafted input do more than produce a wrong or failed calculation: reading or
writing files outside the paths the input names, executing code, or crashing in
a way that corrupts memory. Problems in the published container images'
third-party packages are best reported to those packages' maintainers;
`THIRD_PARTY_NOTICES.md` lists them.
