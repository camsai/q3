#!/usr/bin/env python3
"""
JSON Schema -> C++ code generator.

Reads schemas listed in schemas/manifest.yaml and generates C++ headers with
nlohmann/json serialization under src/generated/: one header per ESSE entry,
and for a Q3 document one header per schema file it is made of, at the same
path (schemas/q3/results/properties.json is q3/results/properties.hpp).

Usage:
    python scripts/repo/generate_cpp_from_schemas.py [--manifest PATH] [--output-dir PATH] [--check]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import jsonref
import yaml
from jinja2 import Environment, FileSystemLoader

import schemas

# Module-level strict mode flag
_strict = False


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


# The struct a header is generated for; titled nested structs take it as prefix
_current_root = ""
# Titles the schema gives to objects that differ, which can name none of them
_ambiguous_titles = set()
# ESSE schemas the manifest generates a struct for, by their $id: a Q3 schema
# that refers to one gets that struct, not a copy of it under a Q3 name
_esse_structs = {}
# Off while ESSE's own schemas are generated
_reuse_esse = False
# The document's own schema files, by $id: each gets its own header, with the
# enums and structs generated from it
_schema_files = {}
# The same files' paths relative to the schema group's base path
_schema_paths = {}
# The $id of the schema file the type being generated comes from
_origin = ""


def _titles_naming_different_objects(schema):
    """
    Titles shared by objects that are not the same object. ESSE titles every
    total_energy_contributions term "scalar schema", each with its own name
    enum; one struct named by the title would serve one term and mistype the
    rest, so those objects are named by their paths instead.
    """
    signatures_by_title = {}
    _collect_object_signatures(schema, signatures_by_title)
    return {title for title, signatures in signatures_by_title.items() if len(signatures) > 1}


def _collect_object_signatures(node, signatures_by_title):
    """Add the signature of every titled object under *node* to *signatures_by_title*."""
    if isinstance(node, dict):
        node = _merge_all_of(node)
        if node.get("title") and node.get("type") == "object":
            signature = json.dumps(node, sort_keys=True, default=str)
            signatures_by_title.setdefault(node["title"], set()).add(signature)
        for key, value in node.items():
            # A definition is visited where it is used, not where it is kept
            if key not in ("definitions", "$defs"):
                _collect_object_signatures(value, signatures_by_title)
    elif isinstance(node, list):
        for value in node:
            _collect_object_signatures(value, signatures_by_title)


def json_type_to_cpp(schema, all_enums, all_nested_structs, parent_name=""):
    """
    Map a JSON Schema type to a C++ type string.

    Returns (cpp_type, inner_type_for_optional, extra_enums, extra_nested_structs).
    """
    global _origin
    identifier = schema.get("$id") if isinstance(schema, dict) else None
    esse_struct = _reuse_esse and _esse_structs.get(identifier)
    if esse_struct:
        cpp_name, _ = esse_struct
        return (cpp_name, cpp_name, [], [])

    if identifier not in _schema_files:
        return _schema_type_to_cpp(schema, all_enums, all_nested_structs, parent_name)
    # A schema file of the document's own: what it generates goes in its header
    enclosing, _origin = _origin, identifier
    try:
        return _schema_type_to_cpp(schema, all_enums, all_nested_structs, parent_name)
    finally:
        _origin = enclosing


def _schema_type_to_cpp(schema, all_enums, all_nested_structs, parent_name):
    """``json_type_to_cpp`` for a schema that is not an ESSE struct."""
    schema = _merge_all_of(schema)

    # Handle $ref that wasn't resolved (shouldn't happen with jsonref, but safety)
    if "$ref" in schema:
        if _strict:
            raise ValueError(
                f"Unresolved $ref in '{parent_name}': {schema['$ref']}. "
                f"Use pre-resolved schemas or enable ref resolution."
            )
        return ("nlohmann::json", "nlohmann::json", [], [])

    # A const is an enum of one value: a Q3 schema pins an ESSE unit with it
    if "const" in schema and "enum" not in schema:
        schema = {**schema, "enum": [schema["const"]]}

    # An enum of non-integer numbers (ESSE's spin: 0.5 or -0.5) is the number
    # itself: enum constants cannot be named or matched by a float, and the
    # schema validates the value
    if (
        "enum" in schema
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in schema["enum"]
        )
        and any(isinstance(value, float) for value in schema["enum"])
    ):
        return ("double", "double", [], [])

    # Enum field — generate an enum class
    if "enum" in schema and not schema.get("type") == "object":
        return _handle_enum(schema, parent_name)

    # oneOf with enums — merge all enum values
    if "oneOf" in schema:
        return _handle_one_of_enum(schema, parent_name)

    schema_type = schema.get("type")

    # A nullable value (`"type": ["number", "null"]`) is carried through as
    # json: the null is data a reader must see, and std::optional would read
    # it as absence.
    if isinstance(schema_type, list) and "null" in schema_type:
        return ("nlohmann::json", "nlohmann::json", [], [])

    if schema_type == "number":
        return ("double", "double", [], [])
    elif schema_type == "integer":
        return ("int64_t", "int64_t", [], [])
    elif schema_type == "string":
        if "enum" in schema:
            return _handle_enum(schema, parent_name)
        return ("std::string", "std::string", [], [])
    elif schema_type == "boolean":
        return ("bool", "bool", [], [])
    elif schema_type == "array":
        return _handle_array(schema, all_enums, all_nested_structs, parent_name)
    elif schema_type == "object":
        return _handle_object(schema, all_enums, all_nested_structs, parent_name)

    # A schema that says nothing about its type (an opaque block) is json
    if not set(schema) - {"description", "title"}:
        return ("nlohmann::json", "nlohmann::json", [], [])

    # Fallback
    if _strict:
        raise ValueError(
            f"Unsupported schema construct in '{parent_name}': "
            f"type={schema.get('type')}, keys={list(schema.keys())}"
        )
    return ("nlohmann::json", "nlohmann::json", [], [])


def _merge_all_of(schema):
    """
    Fold an ``allOf`` into one object schema: the union of the branches'
    ``properties`` and ``required``, the schema's own keys last.

    This is how a Q3 schema reuses an ESSE one and narrows it -- ESSE's
    ``core/reusable/energy`` with ``units`` pinned to ``"Ry"``, say -- so a
    property the schema itself declares replaces the branch's; the narrowing
    is the point. ESSE's pre-resolved schemas have no ``allOf`` left in them.
    """
    if not isinstance(schema, dict) or "allOf" not in schema:
        return schema
    merged_properties = {}
    merged_required = []
    merged = {}
    for branch in [*schema["allOf"], {k: v for k, v in schema.items() if k != "allOf"}]:
        branch = _merge_all_of(branch)
        for key, value in branch.items():
            if key == "properties":
                for property_name, property_schema in value.items():
                    previous = merged_properties.get(property_name, {})
                    merged_properties[property_name] = {**previous, **property_schema}
            elif key == "required":
                merged_required += [name for name in value if name not in merged_required]
            else:
                merged[key] = value
    if merged_properties:
        merged["properties"] = merged_properties
        merged.setdefault("type", "object")
    if merged_required:
        merged["required"] = merged_required
    return merged


def _handle_enum(schema, parent_name):
    """Generate an enum class from a JSON Schema enum."""
    enum_values = schema["enum"]
    enum_name = parent_name + "Enum" if parent_name else "UnnamedEnum"

    cpp_values = []
    for value in enum_values:
        cpp_name = _enum_value_to_cpp(value)
        cpp_values.append({"cpp_name": cpp_name, "json_value": str(value)})

    enum_info = {
        "name": enum_name,
        "origin": _origin,
        "entries": cpp_values,
        "is_string": isinstance(enum_values[0], str),
        "is_integer": isinstance(enum_values[0], int),
    }
    return (enum_name, enum_name, [enum_info], [])


def _handle_one_of_enum(schema, parent_name):
    """Handle oneOf where each branch is an enum — merge values.

    A oneOf with a branch that is not an enum (a number or the word "auto",
    say) has no single C++ enum to become; it is carried as raw JSON, and the
    hand-written parser decides what it means.
    """
    if any("enum" not in branch for branch in schema["oneOf"]):
        return ("nlohmann::json", "nlohmann::json", [], [])

    merged_values = []
    is_string = True

    for branch in schema["oneOf"]:
        if "enum" in branch:
            for value in branch["enum"]:
                if value not in [v["json_value"] for v in merged_values]:
                    merged_values.append(
                        {"cpp_name": _enum_value_to_cpp(value), "json_value": str(value)}
                    )
                    if not isinstance(value, str):
                        is_string = False

    if not merged_values:
        if _strict:
            raise ValueError(
                f"oneOf in '{parent_name}' contains no enum values — "
                f"cannot generate a typed C++ enum."
            )
        return ("nlohmann::json", "nlohmann::json", [], [])

    enum_name = parent_name + "Enum" if parent_name else "UnnamedEnum"
    enum_info = {
        "name": enum_name,
        "origin": _origin,
        "entries": merged_values,
        "is_string": is_string,
        "is_integer": not is_string,
    }
    return (enum_name, enum_name, [enum_info], [])


def _handle_array(schema, all_enums, all_nested_structs, parent_name):
    """Map JSON array to C++ vector."""
    items = schema.get("items", {})

    if isinstance(items, dict) and items:
        inner_type, _, extra_enums, extra_nested = json_type_to_cpp(
            items, all_enums, all_nested_structs, parent_name + "Item"
        )
        cpp_type = f"std::vector<{inner_type}>"
        return (cpp_type, cpp_type, extra_enums, extra_nested)

    # Fallback for untyped arrays — warn in strict mode but don't crash,
    # since ESSE has known schemas with bare "type": "array" (no items).
    if _strict:
        print(
            f"  WARNING [strict]: Array in '{parent_name}' has no typed 'items' — "
            f"falling back to std::vector<nlohmann::json>.",
            file=sys.stderr,
        )
    return ("std::vector<nlohmann::json>", "std::vector<nlohmann::json>", [], [])


def _handle_object(schema, all_enums, all_nested_structs, parent_name):
    """
    Map a JSON object to C++: either a named struct or, for an open-ended
    map of homogeneous values (``additionalProperties`` carrying a schema and
    no fixed ``properties``, as ``method.pseudopotentials`` and
    ``method.scf.starting_magnetization`` do), a ``std::map``. Generating a
    fieldless struct for those would silently drop every entry.
    """
    value_schema = schema.get("additionalProperties")
    # An open object the schema deliberately does not describe
    # (`"additionalProperties": true` and no properties) is carried through
    # as json; a fieldless struct would silently drop every key.
    if not schema.get("properties") and value_schema is True:
        return ("nlohmann::json", "nlohmann::json", [], [])
    if not schema.get("properties") and isinstance(value_schema, dict):
        value_type, _, extra_enums, extra_nested = json_type_to_cpp(
            value_schema, all_enums, all_nested_structs, parent_name + "Value"
        )
        cpp_type = f"std::map<std::string, {value_type}>"
        return (cpp_type, cpp_type, extra_enums, extra_nested)

    return _handle_nested_object(schema, all_enums, all_nested_structs, parent_name)


def _handle_nested_object(schema, all_enums, all_nested_structs, parent_name):
    """
    Handle a nested object schema.

    If it has a title, generate a named struct. Otherwise inline as json.
    """
    title = schema.get("title", "")
    struct_name = parent_name if parent_name else "NestedObject"

    if title and title not in _ambiguous_titles:
        # Prefixed by the header's own struct: every header shares
        # q3::generated, and ESSE titles reused pieces ("atomic vector",
        # "scalar") identically in every schema that includes them.
        struct_name = _title_to_pascal_case(title)
        if not struct_name.startswith(_current_root):
            struct_name = _current_root + struct_name

    # Avoid generating the same struct twice
    existing_names = [s["name"] for s in all_nested_structs]
    if struct_name in existing_names:
        return (struct_name, struct_name, [], [])

    # Generate nested struct
    fields, nested_enums, nested_structs = _extract_fields(
        schema, all_enums, all_nested_structs, struct_name
    )

    nested_struct_info = {
        "name": struct_name,
        "origin": _origin,
        "description": schema.get("description", ""),
        "fields": fields,
    }

    return (
        struct_name,
        struct_name,
        nested_enums,
        nested_structs + [nested_struct_info],
    )


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------


def _extract_fields(schema, all_enums, all_nested_structs, parent_name):
    """Extract field definitions from a JSON Schema object."""
    schema = _merge_all_of(schema)
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = []
    collected_enums = []
    collected_nested = []

    for prop_name, prop_schema in properties.items():
        field_parent_name = parent_name + _to_pascal_case(prop_name)

        cpp_type, inner_type, extra_enums, extra_nested = json_type_to_cpp(
            prop_schema,
            all_enums + collected_enums,
            all_nested_structs + collected_nested,
            field_parent_name,
        )

        collected_enums.extend(extra_enums)
        collected_nested.extend(extra_nested)

        is_optional = prop_name not in required
        if is_optional:
            display_type = f"std::optional<{cpp_type}>"
            default_value = ""
        else:
            default_value = _default_for_type(cpp_type)
            display_type = cpp_type

        fields.append(
            {
                "json_name": prop_name,
                "cpp_name": _to_snake_case(prop_name),
                "cpp_type": display_type,
                "inner_type": cpp_type,
                "is_optional": is_optional,
                "default_value": default_value,
                "description": prop_schema.get("description", ""),
            }
        )

    return fields, collected_enums, collected_nested


def _default_for_type(cpp_type):
    """Return a default initializer for a C++ type, or empty string."""
    if cpp_type == "double":
        return "{}"
    elif cpp_type == "int64_t":
        return "{}"
    elif cpp_type == "bool":
        return "{}"
    else:
        return ""


# ---------------------------------------------------------------------------
# Name conversion helpers
# ---------------------------------------------------------------------------


def _title_to_pascal_case(title):
    """Convert a schema title like 'Total energy property schema' to 'TotalEnergyProperty'."""
    # Remove common suffixes like " schema"
    title = re.sub(r"\s+schema$", "", title, flags=re.IGNORECASE)
    words = re.split(r"[\s_\-]+", title)
    return "".join(_capitalize_first(word) for word in words if word)


def _to_pascal_case(name):
    """Convert snake_case or camelCase to PascalCase."""
    # Split on underscores, hyphens, or camelCase boundaries
    parts = re.split(r"[_\-]", name)
    return "".join(_capitalize_first(part) for part in parts if part)


def _capitalize_first(word):
    """Capitalize only the first letter, preserving the rest."""
    if not word:
        return word
    return word[0].upper() + word[1:]


def _to_snake_case(name):
    """Convert camelCase or PascalCase to snake_case."""
    # Already snake_case
    if "_" in name and name == name.lower():
        return name
    # Handle camelCase
    result = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    # Clean up double underscores
    result = re.sub(r"_+", "_", result)
    return result


def _enum_value_to_cpp(value):
    """Convert a JSON enum value to a valid C++ enum constant name."""
    value_str = str(value)
    # Replace special characters
    result = value_str.upper()
    result = re.sub(r"[^A-Z0-9]", "_", result)
    result = re.sub(r"_+", "_", result)
    result = result.strip("_")
    # Ensure it doesn't start with a digit
    if result and result[0].isdigit():
        result = "V_" + result
    if not result:
        result = "UNKNOWN"
    return result


# ---------------------------------------------------------------------------
# Key registry (schema-driven allow-lists)
# ---------------------------------------------------------------------------

# Guards against a self-referential schema turning the walk below into
# unbounded recursion. Nothing legitimate nests anywhere near this deep.
MAXIMUM_KEY_REGISTRY_DEPTH = 32


def build_key_registry(schema):
    """
    Collect the allow-list of keys for every *closed* object in a schema.

    An object is closed when it says ``"additionalProperties": false`` — i.e.
    the schema, not some hand-maintained list elsewhere, owns the full set of
    keys that may appear there. Each such object is recorded under a dotted
    path built from the property names leading to it, with ``[]`` marking a
    step through an array's ``items`` (e.g. ``method.scf.hubbard[]``). The
    root object itself is recorded under the empty path.

    Objects that stay open — ``material``, which legitimately carries ESSE
    fields Q3 does not consume — are skipped, so nothing here can make the
    parser reject a key the schema never claimed to own.

    Returns a list of ``{"path": ..., "allowed_keys": [...]}`` sorted by path.
    """
    entries = {}

    def visit(node, path, depth):
        if not isinstance(node, dict):
            return
        if depth > MAXIMUM_KEY_REGISTRY_DEPTH:
            raise ValueError(
                f"Schema nests deeper than {MAXIMUM_KEY_REGISTRY_DEPTH} levels at '{path}' — "
                f"is a $ref cyclic?"
            )

        # A branch (oneOf/anyOf/allOf) describes the same location, so it
        # contributes to the same path rather than a nested one.
        for branch_keyword in ("oneOf", "anyOf", "allOf"):
            for branch in node.get(branch_keyword, []):
                visit(branch, path, depth + 1)

        if node.get("additionalProperties") is False:
            keys = sorted(node.get("properties", {}))
            previous_keys = entries.get(path)
            if previous_keys is not None and previous_keys != keys:
                raise ValueError(
                    f"Two closed objects at path '{path}' disagree on their keys: "
                    f"{previous_keys} vs {keys}"
                )
            entries[path] = keys

        for property_name, property_schema in node.get("properties", {}).items():
            child_path = f"{path}.{property_name}" if path else property_name
            visit(property_schema, child_path, depth + 1)

        items_schema = node.get("items")
        if isinstance(items_schema, dict):
            visit(items_schema, path + "[]", depth + 1)

    visit(schema, "", 0)
    return [{"path": path, "allowed_keys": keys} for path, keys in sorted(entries.items())]


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def _make_opaque(schema, property_paths):
    """
    Carry the named properties as ``nlohmann::json`` rather than as generated
    structs: a block the writer already builds as JSON elsewhere (an ESSE
    material, an open provenance stamp), which the schema describes for
    validation but a struct would only round-trip, and lossily for an object
    the schema leaves open. A path names a nested property by its parents,
    dot-separated (``structure.material``); the description kept is the
    property's own.
    """
    for path in property_paths:
        schema = _make_property_opaque(schema, path.split("."))
    return schema


def _make_property_opaque(schema, names):
    """*schema* with the property at the path *names* replaced by an opaque one."""
    schema = _merge_all_of(schema)
    properties = dict(schema["properties"])
    first, rest = names[0], names[1:]
    if rest:
        properties[first] = _make_property_opaque(properties[first], rest)
    else:
        properties[first] = {"description": properties[first].get("description", "")}
    return {**schema, "properties": properties}


def load_schema(schema_path, resolve_refs=False):
    """Load a JSON schema, optionally resolving $ref links.

    A ``$ref`` is relative to the file that holds it, as JSON Schema has it,
    so a schema in a subdirectory (``interpolation/results.json``) resolves
    its references from there.
    """
    with open(schema_path) as file_handle:
        raw = json.load(file_handle)

    if resolve_refs:
        base_uri = Path(os.path.dirname(schema_path)).resolve().as_uri() + "/"
        resolved = jsonref.replace_refs(raw, base_uri=base_uri, lazy_load=False)
        return resolved
    else:
        return raw


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _includes_for(fields_lists):
    """The standard headers the given fields need."""
    fields = [field for field_list in fields_lists for field in field_list]
    types = [field["cpp_type"] + " " + field.get("inner_type", "") for field in fields]
    return {
        "needs_optional": any(field["is_optional"] for field in fields),
        "needs_string": any("std::string" in text for text in types),
        "needs_vector": any("std::vector" in text for text in types),
        "needs_map": any("std::map" in text for text in types),
        "needs_cstdint": any("int64_t" in text for text in types),
    }


def generate_headers(schema, cpp_name, schema_path, template_env, emit_key_registry=False):
    """Generate the C++ headers of a resolved JSON schema, one per schema file.

    Returns ``{$id: (schema file, header text)}``. The document's own header
    holds its struct and what its file alone generates; each schema file it
    refers to (``_schema_files``) gets a header of its own with the enums and
    structs that come from it, and a header includes the ones it uses.
    """
    global _current_root, _ambiguous_titles, _origin
    _current_root = cpp_name
    _ambiguous_titles = _titles_naming_different_objects(schema)
    document_id = schema.get("$id", "")
    _origin = document_id

    fields, all_enums, all_nested_structs = _extract_fields(schema, [], [], cpp_name)

    owner = {item["name"]: item["origin"] for item in all_enums + all_nested_structs}
    owner[cpp_name] = document_id
    origins = [document_id] + [
        origin
        for origin in dict.fromkeys(item["origin"] for item in all_enums + all_nested_structs)
        if origin != document_id
    ]

    headers = {}
    template = template_env.get_template("struct.hpp.j2")
    for origin in origins:
        is_document = origin == document_id
        enums = [item for item in all_enums if item["origin"] == origin]
        nested_structs = [item for item in all_nested_structs if item["origin"] == origin]
        own_fields = [nested["fields"] for nested in nested_structs]
        if is_document:
            own_fields.append(fields)
        used = {
            name
            for field_list in own_fields
            for field in field_list
            for name in re.findall(r"\w+", field["cpp_type"])
        }
        esse_headers = dict(_esse_structs.values()) if _reuse_esse else {}
        esse_includes = sorted({esse_headers[name] for name in used if name in esse_headers})
        local_includes = sorted(
            {_schema_files[owner[name]] for name in used if owner.get(name, origin) != origin}
        )
        key_registry = build_key_registry(schema) if is_document and emit_key_registry else []
        includes = _includes_for(own_fields)
        if key_registry:
            includes.update(needs_string=True, needs_vector=True, needs_map=True)
        has_optional = any(field["is_optional"] for field in fields)
        file_path = schema_path if is_document else _schema_paths[origin]
        headers[origin] = (
            file_path,
            template.render(
                schema_path=file_path,
                struct_name=cpp_name if is_document else None,
                description=schema.get("description", ""),
                fields=fields,
                enums=enums,
                nested_structs=nested_structs,
                use_macro=not has_optional and not all_enums,
                key_registry=key_registry,
                esse_includes=esse_includes,
                local_includes=local_includes,
                key_registry_namespace=_to_snake_case(cpp_name) + "_schema",
                **includes,
            ),
        )
    return headers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate C++ from JSON schemas")
    parser.add_argument(
        "--manifest",
        default="schemas/manifest.yaml",
        help="Path to manifest.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="src/generated",
        help="Output directory for generated headers",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that generated files are up-to-date (exit 1 if stale)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Error on any schema construct that cannot be mapped to a concrete C++ type",
    )
    args = parser.parse_args()

    global _strict, _reuse_esse
    _strict = args.strict

    project_root = Path(__file__).resolve().parents[2]
    # Resolved after the relative/absolute branch, not before: a relative path is
    # relative to the project root, and resolving it first would silently
    # reinterpret it against the caller's working directory. Both must end up
    # canonical because output_dir is later reported relative_to(project_root),
    # which raises ValueError when the two name the same directory by different
    # paths -- a shared filesystem reachable as both /global/cfs/cdirs/... and
    # /global/cfs/projectdirs/..., say, where CMake passes whichever spelling it
    # was invoked with.
    manifest_arg = Path(args.manifest)
    output_arg = Path(args.output_dir)
    manifest_path = (
        manifest_arg if manifest_arg.is_absolute() else project_root / manifest_arg
    ).resolve()
    output_dir = (output_arg if output_arg.is_absolute() else project_root / output_arg).resolve()

    schemas.ensure_esse()
    # Load manifest
    with open(manifest_path) as file_handle:
        manifest = yaml.safe_load(file_handle)

    # Setup Jinja2 environment
    template_dir = Path(__file__).resolve().parent / "templates"
    template_env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )

    esse_config = manifest.get("esse", {})
    for entry in esse_config.get("schemas", []):
        with open(project_root / esse_config["base_path"] / entry["schema"]) as file_handle:
            schema_id = json.load(file_handle).get("$id")
        if schema_id:
            _esse_structs[schema_id] = (entry["cpp_name"], entry["output"])

    stale_files = []
    generated_count = 0

    # Process each schema group (esse, q3)
    for group_name, group_config in manifest.items():
        base_path = project_root / group_config["base_path"]
        resolve_refs = group_config.get("resolve_refs", False)
        # Only a schema that refers to ESSE's reuses its structs; an ESSE
        # schema's own header is the struct
        _reuse_esse = group_name != "esse"
        # A Q3 document's schema files each get a header, beside the
        # document's own: schemas/q3/results/properties.json is
        # q3/results/properties.hpp when results.json is q3/results.hpp.
        _schema_files.clear()
        _schema_paths.clear()
        if resolve_refs:
            entry = group_config["schemas"][0]
            prefix = entry["output"][: -len(str(Path(entry["schema"]).with_suffix(".hpp")))]
            for path in sorted(base_path.rglob("*.json")):
                identifier = json.loads(path.read_text()).get("$id")
                relative = path.relative_to(base_path)
                if identifier:
                    _schema_files[identifier] = prefix + str(relative.with_suffix(".hpp"))
                    _schema_paths[identifier] = str(relative)

        for entry in group_config["schemas"]:
            schema_file = base_path / entry["schema"]
            cpp_name = entry["cpp_name"]

            if not schema_file.exists():
                print(f"WARNING: Schema not found: {schema_file}", file=sys.stderr)
                continue

            # Load schema
            schema = load_schema(str(schema_file), resolve_refs=resolve_refs)
            schema = _make_opaque(schema, entry.get("opaque", []))

            # Generate the C++ headers: the document's, then its schema files'
            headers = generate_headers(
                schema,
                cpp_name,
                entry["schema"],
                template_env,
                emit_key_registry=entry.get("key_registry", False),
            )
            document_id = schema.get("$id", "")
            for identifier, (_, header_content) in headers.items():
                relative_output = (
                    entry["output"] if identifier == document_id else _schema_files[identifier]
                )
                output_path = output_dir / relative_output
                if args.check:
                    # Compare with existing file
                    if not output_path.exists() or output_path.read_text() != header_content:
                        stale_files.append(str(output_path))
                else:
                    # Write output
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(header_content)
                    generated_count += 1
                    print(f"  Generated: {output_path.relative_to(project_root)}")

    if args.check:
        if stale_files:
            print("Stale generated files:", file=sys.stderr)
            for stale_file in stale_files:
                print(f"  {stale_file}", file=sys.stderr)
            print(
                "\nRun 'python scripts/repo/generate_cpp_from_schemas.py' to regenerate.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print("All generated files are up-to-date.")
    else:
        print(f"\nGenerated {generated_count} C++ headers in {output_dir}")


if __name__ == "__main__":
    main()
