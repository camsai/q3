"""
Find the textual signals that code was derived from a reference implementation.

- ``core.py``          the abstract logic: rules, Q3's host languages, the scan, the baseline.
  It names no reference and no foreign language.
- ``fortran.py``       the syntax of Fortran, the language the current references are written in.
- ``references/``      one ``<key>.json`` per reference: its names in prose, its languages, its
  identifiers.

Adding a reference is adding a data file. A reference written in a new language
also adds a ``ForeignLanguage`` subclass beside ``fortran.py`` and registers it in
``scripts/repo/check_provenance.py``, the command line that wires content into the core.
"""
