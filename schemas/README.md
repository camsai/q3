# Schema-to-C++ Code Generation

This directory contains JSON schemas that are automatically converted into C++
structs with [nlohmann/json](https://github.com/nlohmann/json) serialization.

## Overview

```
schemas/manifest.yaml          ← declares which schemas to generate
    ├── esse/dist/js/schema/   ← pre-resolved ESSE schemas (read-only)
    └── q3/                    ← Q3-specific schemas
                ↓
scripts/repo/generate_cpp_from_schemas.py   ← Python generator (Jinja2 + jsonref)
scripts/repo/templates/struct.hpp.j2        ← Jinja2 template for C++ headers
                ↓
src/generated/                 ← output (auto-generated, DO NOT EDIT)
    ├── esse/properties/scalar/total_energy.hpp
    └── q3/input.hpp
```

## How It Works

1. **Manifest** (`schemas/manifest.yaml`) lists every schema to generate, grouped
   by source (`esse` or `q3`). Each entry specifies:

    - `schema` — relative path to the JSON Schema file
    - `cpp_name` — the PascalCase name for the generated C++ struct
    - `output` — relative path under `src/generated/` for the output header
    - `key_registry` — optional; also emit the schema's per-object key
      allow-lists (see [Key registries](#key-registries) below)

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

An "open map" is an object with no fixed `properties` and an
`additionalProperties` schema — `method.pseudopotentials` (element → file
path) is one. Generating a fieldless struct for those would silently drop
every entry.

Properties **not** listed in the schema's `required` array are wrapped in
`std::optional<T>`.

## Schema Sources

### ESSE schemas (`esse/`)

These come from the [ESSE](https://github.com/Exabyte-io/esse) project and live
under `esse/dist/js/schema/`. They are **pre-resolved** — all `$ref` links are
already dereferenced — so the generator loads them as-is (`resolve_refs: false`).

### Q3 schemas (`q3/`)

Q3-specific schemas may use `$ref` for local cross-references (e.g.
`#/definitions/self_consistent_field`). The manifest sets `resolve_refs: true`
for this group, and the generator uses `jsonref` to resolve them before code
generation.

`q3/input.json` is the YAML input manifest format — the file a user writes,
key for key, not the flattened `q3::InputData` the parsers populate. It is the
single source of truth for which keys a manifest may carry: see
[Key registries](#key-registries).

`q3/results.json` is the other half of that pair: the `<prefix>_results.json` a
finished run writes, described key for key with a closed root, so a consumer
outside this repository knows exactly what it may find. It `$ref`s into
`esse/dist/js/schema/` — the same pre-resolved subset the generator reads —
wherever ESSE already defines what Q3 emits, which today is the unit enums and
the 3-vector/3×3 array shapes.

Unlike `input.json` it is **not** in the manifest, so no C++ is generated from
it: nothing in the engine reads the results back, and a generated struct that
no code constructed would be a second declaration of the writer's fields to keep
in step with the first. It is validated instead, from both ends, by
`tests/regression/test_results_schema.py`: every key `FlavorResult::results_json()`
can emit must be declared here, and a real run's output must validate against it.

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

    - For ESSE properties: the schema likely already exists in `esse/dist/js/schema/`
    - For Q3-specific data: add a `.json` file under `q3/`

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
