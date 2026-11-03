<!--
Landing page of the documentation site (https://camsai.github.io/q3/), built by
Doxygen (USE_MDFILE_AS_MAINPAGE in the Doxyfile). It is written for that site, so it
uses Doxygen's \subpage, which also sets the order of the pages in the sidebar:
Functionality, Tutorials, How-to Guides, Summary, Reference and Explanation, then
Doxygen's Namespaces, Classes and Files. The section pages (docs/how_to_guides.md,
docs/reference.md, docs/explanation.md) list the pages of their sections.
A page's label is its path, lower case, with every character that is not a letter
or digit written as "_" (docs/doxygen_markdown_filter.py gives it).
The repository README is shown there as the "User Guide" page, under How-to Guides.
-->

An AI-native engine for next-generation quantum materials modeling, written from
scratch by AI coding agents under the direction and supervision of domain scientists.

Q3 is a self-contained, from-scratch quantum simulation engine. Its first
implementation is plane-wave density functional theory (DFT): it computes the
ground-state electronic structure of periodic crystals — total energies, forces,
stress, and band structures — in modern, readable C++ with a Python layer on top.

| If you want to…                                    | Go here                                                                                                                 |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Know what Q3 can do today                          | \subpage functionality "Functionality" — capabilities by area, each with its validation status                          |
| Learn Q3 step by step                              | \subpage docs_tutorials_readme "Tutorials" — eight runnable lessons, from a first calculation to a contribution         |
| Get a job done: install, build, run, fix a failure | \subpage docs_how_to_guides "How-to Guides" — start with the \ref readme "User Guide"                                   |
| Know what Q3 is, and why it exists                 | \subpage summary "Summary"                                                                                              |
| Look up a field, a flag, a limitation or a number  | \subpage docs_reference "Reference" — inputs and outputs, build variables, \ref known_issues "Known Issues", benchmarks |
| Understand how Q3 is designed                      | \subpage docs_explanation "Explanation" — architecture, data model, differentiability, provenance                       |
| Read the code                                      | the [namespaces](namespaces.html), [classes](annotated.html) and [files](files.html)                                    |

The source is on [GitHub](https://github.com/camsai/q3), under the
\ref license "Apache License 2.0".
