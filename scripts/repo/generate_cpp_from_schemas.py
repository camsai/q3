#!/usr/bin/env python3
"""
JSON Schema -> C++ code generator.

Reads schemas listed in schemas/manifest.yaml and generates C++ headers with
nlohmann/json serialization under src/generated/.

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

# Module-level strict mode flag
_strict = False


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


def json_type_to_cpp(schema, all_enums, all_nested_structs, parent_name=""):
    """
    Map a JSON Schema type to a C++ type string.

    Returns (cpp_type, inner_type_for_optional, extra_enums, extra_nested_structs).
    """
    # Handle $ref that wasn't resolved (shouldn't happen with jsonref, but safety)
    if "$ref" in schema:
        if _strict:
            raise ValueError(
                f"Unresolved $ref in '{parent_name}': {schema['$ref']}. "
                f"Use pre-resolved schemas or enable ref resolution."
            )
        return ("nlohmann::json", "nlohmann::json", [], [])

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

    # Fallback
    if _strict:
        raise ValueError(
            f"Unsupported schema construct in '{parent_name}': "
            f"type={schema.get('type')}, keys={list(schema.keys())}"
        )
    return ("nlohmann::json", "nlohmann::json", [], [])


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

    if title:
        struct_name = _title_to_pascal_case(title)

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


def load_schema(schema_path, resolve_refs=False, base_dir=None):
    """Load a JSON schema, optionally resolving $ref links."""
    with open(schema_path) as file_handle:
        raw = json.load(file_handle)

    if resolve_refs:
        base_uri = Path(base_dir or os.path.dirname(schema_path)).resolve().as_uri() + "/"
        resolved = jsonref.replace_refs(raw, base_uri=base_uri, lazy_load=False)
        return resolved
    else:
        return raw


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def generate_header(schema, cpp_name, schema_path, template_env, emit_key_registry=False):
    """Generate a C++ header string from a resolved JSON schema."""
    all_enums = []
    all_nested_structs = []

    # Extract fields from the top-level object
    fields, enums, nested_structs = _extract_fields(schema, all_enums, all_nested_structs, cpp_name)
    all_enums.extend(enums)
    all_nested_structs.extend(nested_structs)

    # Determine if we can use the simple NLOHMANN_DEFINE macro
    has_optional = any(f["is_optional"] for f in fields)
    has_enums = len(all_enums) > 0
    use_macro = not has_optional and not has_enums

    # Determine required includes
    needs_optional = has_optional
    needs_string = any(
        "std::string" in f["cpp_type"] or "std::string" in f.get("inner_type", "") for f in fields
    )
    needs_vector = any("std::vector" in f["cpp_type"] for f in fields)
    needs_map = any("std::map" in f["cpp_type"] for f in fields)
    needs_cstdint = any(
        "int64_t" in f["cpp_type"] or "int64_t" in f.get("inner_type", "") for f in fields
    )

    # Also check nested structs for includes
    for nested in all_nested_structs:
        for field in nested.get("fields", []):
            if "std::string" in field["cpp_type"]:
                needs_string = True
            if "std::vector" in field["cpp_type"]:
                needs_vector = True
            if "std::map" in field["cpp_type"]:
                needs_map = True
            if "int64_t" in field["cpp_type"]:
                needs_cstdint = True
            if field["is_optional"]:
                needs_optional = True

    key_registry = build_key_registry(schema) if emit_key_registry else []
    if key_registry:
        needs_string = True
        needs_vector = True
        needs_map = True

    template = template_env.get_template("struct.hpp.j2")
    return template.render(
        schema_path=schema_path,
        struct_name=cpp_name,
        description=schema.get("description", ""),
        fields=fields,
        enums=all_enums,
        nested_structs=all_nested_structs,
        use_macro=use_macro,
        needs_optional=needs_optional,
        needs_string=needs_string,
        needs_vector=needs_vector,
        needs_map=needs_map,
        needs_cstdint=needs_cstdint,
        key_registry=key_registry,
        key_registry_namespace=_to_snake_case(cpp_name) + "_schema",
    )


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

    global _strict
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

    stale_files = []
    generated_count = 0

    # Process each schema group (esse, q3)
    for group_name, group_config in manifest.items():
        base_path = project_root / group_config["base_path"]
        resolve_refs = group_config.get("resolve_refs", False)

        for entry in group_config["schemas"]:
            schema_file = base_path / entry["schema"]
            cpp_name = entry["cpp_name"]
            output_path = output_dir / entry["output"]

            if not schema_file.exists():
                print(f"WARNING: Schema not found: {schema_file}", file=sys.stderr)
                continue

            # Load schema
            schema = load_schema(
                str(schema_file),
                resolve_refs=resolve_refs,
                base_dir=str(base_path),
            )

            # Generate C++ header
            header_content = generate_header(
                schema,
                cpp_name,
                entry["schema"],
                template_env,
                emit_key_registry=entry.get("key_registry", False),
            )

            if args.check:
                # Compare with existing file
                if output_path.exists():
                    existing = output_path.read_text()
                    if existing != header_content:
                        stale_files.append(str(output_path))
                else:
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
