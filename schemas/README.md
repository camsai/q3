# Schema Code Generation

This directory contains JSON schemas that are automatically converted into C++
structs with [nlohmann/json](https://github.com/nlohmann/json) serialization,
and, for the results files, into pydantic models (see [Python models](#python-models)).

## Overview

```
schemas/manifest.yaml          ← declares which schemas to generate
    ├── esse/                  ← resolved ESSE schemas, written from mat3ra-esse (untracked)
    └── q3/                    ← Q3's own schemas, one per file; they $ref ../esse
                ↓
scripts/repo/generate_cpp_from_schemas.py   ← Python generator (Jinja2 + jsonref)
scripts/repo/templates/struct.hpp.j2        ← Jinja2 template for C++ headers
                ↓
src/generated/                 ← output (auto-generated, DO NOT EDIT)
    ├── esse/properties/scalar/total_energy.hpp
    ├── q3/results.hpp
    └── q3/results/properties.hpp  (one header per Q3 schema file)
```

## How It Works

1. **Manifest** (`schemas/manifest.yaml`) lists every schema to generate, grouped
   by source (`esse` or `q3`). Each entry specifies:

    - `schema` — relative path to the JSON Schema file
    - `cpp_name` — the PascalCase name for the generated C++ struct
    - `output` — relative path under `src/generated/` for the output header
    - `key_registry` — optional; also emit the schema's per-object key
      allow-lists (see [Key registries](#key-registries) below)
    - `opaque` — optional; top-level properties generated as `nlohmann::json`
      rather than typed, for blocks another component builds whole and the
      schema only validates (`structure.material`, `provenance`,
      `runtime_log.timings_ms` in `results.json`); a dotted path reaches a nested
      property
    - `python` — optional; also generate pydantic models into
      `src/py/q3/generated/` (see [Python models](#python-models) below):
      `module` names the output, and `bases` gives a block, by its schema
      `$id`, the hand-written base carrying what a reader needs beyond the data

2. **Generator** (`scripts/repo/generate_cpp_from_schemas.py`) reads the manifest,
   loads each JSON Schema (optionally resolving `$ref` links via `jsonref`), maps
   JSON types to C++ types, and renders the Jinja2 template.

3. **Template** (`scripts/repo/templates/struct.hpp.j2`) produces a header-only C++
   file in namespace `q3::generated` containing:

    - Enum classes for `enum` fields (with `to_json`/`from_json`)
    - Nested structs for inline `object` definitions
    - The top-level struct with typed fields
    - `to_json`/`from_json` ADL overloads for nlohmann/json round-tripping
    - A key registry, for schemas that ask for one

4. **CMake integration** — the `generate_schemas` custom target re-runs the
   generator at build time whenever the manifest, the generator script, the
   Jinja2 template, or any schema file changes. Both `q3` and `q3_tests` targets
   depend on it, so headers are always up to date.

## Type Mapping

| JSON Schema type                                                       | C++ type                        |
| ---------------------------------------------------------------------- | ------------------------------- |
| `"type": "number"`                                                     | `double`                        |
| `"type": "integer"`                                                    | `int64_t`                       |
| `"type": "string"`                                                     | `std::string`                   |
| `"type": "boolean"`                                                    | `bool`                          |
| `"type": "array"`                                                      | `std::vector<T>`                |
| `"type": "object"` (nested)                                            | named struct                    |
| `"type": "object"` (open map)                                          | `std::map<std::string, T>`      |
| `"type": "object"` (open, `additionalProperties: true`, no properties) | `nlohmann::json`                |
| `"type": ["number", "null"]` (nullable)                                | `nlohmann::json`                |
| `"enum": [...]`                                                        | `enum class` + string/int serde |
| `"const": x`                                                           | one-value `enum class`          |
| `"allOf": [...]`                                                       | the branches' properties merged |
| no keyword beyond `description`/`title`                                | `nlohmann::json`                |

An "open map" is an object with no fixed `properties` and an
`additionalProperties` schema — `method.pseudopotentials` (element → file
path) is one. Generating a fieldless struct for those would silently drop
every entry.

Properties **not** listed in the schema's `required` array are wrapped in
`std::optional<T>`.

Each Q3 schema file gets its own header, at the same path under `q3/`
(`results/properties.json` is `q3/results/properties.hpp`), holding the enums
and structs generated from that file; an untitled object nested inside it stays
in its header. A header includes the ones it uses, so including the document's
header (`q3/results.hpp`) still brings in every struct of the document. A
schema file that generates no struct of its own (an open map of ESSE records,
an opaque block) has no header.

A nested object with a `title` is named after it, prefixed by the document's
root struct (`ResultsEnergyContributions` in `q3/results/properties/energy_contributions.hpp`), so a shape used by
several keys is one struct, and two headers sharing `q3::generated` never
declare the same name. A title that names two differently shaped objects in one
schema (ESSE titles every energy contribution "scalar schema") is ignored, and
those objects are named by their path instead.

A Q3 schema that `$ref`s an ESSE schema listed in the manifest's `esse` group
(matched by its `$id`) gets that ESSE struct, not a copy under a Q3 name:
`results.json`'s energies are `generated::Energy`, its force and stress
contributions `generated::AtomicForces` and `generated::StressTensor`, and the
generated header includes the ESSE one.

`allOf` is how a Q3 definition builds on an ESSE piece ESSE has no full schema
for yet: `results.json`'s magnetization value is `core/primitive/scalar`
plus a name and ESSE's magnetic unit, and the generator merges the two, the
schema's own keys last.

## Schema Sources

### ESSE schemas (`esse/`)

These come from the [ESSE](https://github.com/mat3ra/esse) project's Python
package, `mat3ra-esse`, pinned in `pyproject.toml`. Its
`mat3ra.esse.data.schemas.SCHEMAS` holds every schema **resolved**: all `$ref`
links inlined and every `allOf` merged. `scripts/repo/schemas.py` writes
them to `esse/`, one file per schema at the path ESSE derives from its `$id`,
with a `.mat3ra-esse-version` stamp. The directory is not committed: the code
generators, the schema tests and CMake call it, and it rewrites the directory
only when it is missing or holds another version's schemas. The generator
loads them as-is (`resolve_refs: false`). To move to a newer ESSE, bump the pin in
`pyproject.toml` (the Dockerfiles and `release.yml` install `".[build]"`, which
carries it; `.pre-commit-config.yaml` repeats it, and
`tests/python/test_pyproject_pins.py` keeps the two equal)
and regenerate.

The same script gives any Q3 schema resolved: `resolved("results.json")`
returns it with every `$ref` inlined, ESSE's included, which is what the tests
validate results files against, and
`python scripts/repo/schemas.py --out <directory>` writes every schema under
`q3/` that way, for a consumer that wants self-contained files.

### Q3 schemas (`q3/`)

Q3's own schemas follow the layout of the platform's web application: one
schema per file, a document's blocks in a directory named after it
(`results.json` → `results/runtime_log.json`, `results/properties/energy_contributions.json`),
and ESSE extended locally rather than copied. A Q3 schema reaches ESSE by a
relative path to `esse/` (`"$ref": "../esse/material.json"` from `q3/`), and builds on an
ESSE schema with `allOf` where it needs more than ESSE states; a change ESSE
itself should make goes to ESSE first (AGENTS.md, "Output schemas"). The
manifest sets `resolve_refs: true` for this group, and the generator uses
`jsonref` to resolve the relative `$ref`s before code generation.

`q3/input.json` is the YAML input manifest format — the file a user writes,
key for key, not the flattened `q3::InputData` the parsers populate. It is the
single source of truth for which keys a manifest may carry: see
[Key registries](#key-registries).

`q3/results.json` is the other half of that pair: the `<prefix>_results.json` a
finished run writes, described key for key with closed objects, so a consumer
outside this repository knows exactly what it may find. It is grouped by
domain: `structure` (the material), `provenance`, `properties` (the physical
results) and `runtime_log` (flavor, convergence, iterations, timings, the parallel
topology and any incomplete derivative). It `$ref`s ESSE — the same
pre-resolved subset the generator reads — wherever ESSE already defines what
Q3 emits: the material, and `properties.records`, whose every record is one of
ESSE's own property schemas (`total_energy`, `total_energy_contributions`,
`fermi_energy`, `atomic_forces`, `stress_tensor`, `pressure`, `band_gaps`,
`convergence_electronic`). Every other quantity `$ref`s the ESSE schema for it
directly (`core/reusable/energy`, `atomic_forces`, `stress_tensor`), or, where
ESSE has none yet, a file of its own under `results/properties/` built from
ESSE pieces, carrying its unit in its own `units` field. No key names a unit; see AGENTS.md, "Output schemas (ESSE
first)".

It is in the manifest, and so are the ESSE property schemas it lists:
Every flavor returns `generated::Results`, filled with the generated ESSE
structs, so the writer cannot emit a key the schema does not declare.
`tests/regression/test_results_schema.py` checks that, and that a real run's
output validates against the schema. `q3/interpolation/results.json`, the file
`q3_utility interpolate` writes, is generated the same way and shares the
same ESSE schemas.

## Key registries

A schema entry with `key_registry: true` also gets its own key allow-lists
emitted into the same header, under
`q3::generated::<snake_case_cpp_name>_schema`:

```cpp
const std::vector<std::string>& keys =
    q3::generated::input_schema::allowed_keys("method.scf.mixing");
```

One entry is emitted per object the schema **closes** with
`"additionalProperties": false`, keyed by the dotted path to it — `""` is the
document root, and `[]` marks a step through an array's `items`
(`method.scf.hubbard[]`). Objects the schema deliberately leaves open produce
no entry, and asking for one throws.

`src/io/input/parser/yaml.hpp` validates every manifest level against these
lists rather than against hand-written copies, which is what keeps the schema
and the parser from drifting apart: a new input key is added to `q3/input.json`
and reaches the parser by regeneration. `tests/python/test_input_schema.py`
closes the loop from the other side, checking every committed example manifest
against the schema.

## Adding a New Schema

1. Create the JSON Schema file under the appropriate directory:

    - For ESSE properties: the schema likely already exists in `esse/`
    - For Q3-specific data: add a `.json` file under `q3/`, one schema per
      file, extending ESSE (a `$ref` into `esse/`) where ESSE has the piece

2. Add an entry to `schemas/manifest.yaml`:

    ```yaml
    q3:
        schemas:
            - schema: my_new_type.json
              cpp_name: MyNewType
              output: q3/my_new_type.hpp
    ```

3. Regenerate:

    ```bash
    source .venv/bin/activate
    python scripts/repo/generate_cpp_from_schemas.py
    ```

    Or simply build — CMake runs it automatically.

4. Include the generated header in your C++ code:

    ```cpp
    #include "generated/q3/my_new_type.hpp"

    q3::generated::MyNewType data;
    nlohmann::json j = data;            // serialize
    auto parsed = j.get<q3::generated::MyNewType>();  // deserialize
    ```

## Python models

`scripts/repo/generate_python_from_schemas.py` generates the Python side of the
same schemas: every `q3` entry with a `python` block becomes a package under
`src/py/q3/generated/` (`q3.generated.results`,
`q3.generated.interpolation.results`), with one module per schema file that is
an object with properties, at the schema's path
(`results/properties/energy_contributions.json` is
`q3.generated.results.properties.energy_contributions`), written by
[datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator)
with the flags ESSE generates `mat3ra.esse.models` with. Before generating, the
schema is prepared so that:

-   a `$ref` into ESSE is ESSE's own model (`customTypePath`), never a copy;
-   a `$ref` to another schema file with a module is that module's class, and
    any other Q3 `$ref` (an array, an open map) is inlined where it is used;
-   an `opaque` path is a plain JSON object, as it is in C++;
-   every field is optional and unknown keys are kept: `required` and
    `additionalProperties: false` are the writer's guarantee, checked by the
    schema tests against real runs, not a reader's precondition;
-   a block the manifest names under `bases` derives from that hand-written
    class in `q3.io.results.base`, or `q3.io.interpolation.results.base` for
    the interpolation file (reading a file, a record by its name, the
    eigenvalues as arrays).

`q3.io.results` and `q3.io.interpolation.results` re-export the generated
models under the names the package reads them by. Regenerate with
`python scripts/repo/generate_python_from_schemas.py`; `--check` (a pre-commit
hook and `tests/python/test_generated_models.py`) fails on a stale module. It
needs the dev extra (`pip install -e ".[dev]"`), which pins the generator and
the formatters it runs.

## Generator CLI

```
python scripts/repo/generate_cpp_from_schemas.py [OPTIONS]

Options:
  --manifest PATH     Path to manifest.yaml      (default: schemas/manifest.yaml)
  --output-dir PATH   Output directory            (default: src/generated)
  --check             Verify generated files are up-to-date (exit 1 if stale)
  --strict            Error on any unmappable schema construct
```

The `--check` flag is useful in CI to ensure generated headers are committed and
not stale.

## Dependencies

The generator requires Python 3.8+ with the following packages (installed in
`.venv`):

-   `jinja2` — template rendering
-   `jsonref` — `$ref` resolution in JSON Schemas
-   `pyyaml` — manifest parsing
