"""q3.io.input_reader — the files an input manifest names, read the way the engine reads them.

A manifest refers to its material and its pseudopotentials by path, and the
engine resolves a relative one against the directory the manifest was read
from. These are that rule, for the Python that has to find the same files: the
material itself, the dataset of each species, and the rewrite that makes a
manifest safe to copy somewhere else.
"""

from __future__ import annotations

import json
from pathlib import Path


def _resolved(reference: str, base_directory: str | Path) -> Path:
    path = Path(reference).expanduser()
    return path if path.is_absolute() else Path(base_directory) / path


def load_material(manifest: dict, base_directory: str | Path) -> dict:
    """The manifest's material as a dict: the JSON file it names, or the inline map.

    A relative path is resolved against *base_directory*, the directory the
    manifest was read from, which is how the engine resolves it too.
    """
    material = manifest.get("material")
    if isinstance(material, dict):
        return material
    if isinstance(material, str):
        with open(_resolved(material, base_directory)) as handle:
            return json.load(handle)
    raise ValueError(
        "the manifest's material is neither a path to a material JSON file nor an inline "
        "material map, so its lattice cannot be read"
    )


def pseudopotential_paths(
    manifest: dict, material: dict, base_directory: str | Path
) -> dict[str, Path]:
    """Each species of *material* and the dataset the manifest names for it, in basis order."""
    named = manifest.get("method", {}).get("pseudopotentials", {})
    paths: dict[str, Path] = {}
    for element in material["basis"]["elements"]:
        symbol = element["value"]
        if symbol in paths:
            continue
        if symbol not in named:
            raise ValueError(f"the manifest names no pseudopotential for {symbol}")
        paths[symbol] = _resolved(named[symbol], base_directory)
    return paths


def make_file_references_absolute(manifest: dict, base_directory: Path) -> None:
    """Resolve *manifest*'s file references, in place, against the directory it came from.

    A manifest written into a temporary directory would otherwise resolve its
    relative ``material`` and pseudopotential paths against the wrong place.
    """
    material = manifest.get("material")
    if isinstance(material, str) and not Path(material).is_absolute():
        manifest["material"] = str((base_directory / material).resolve())
    pseudopotentials = manifest.get("method", {}).get("pseudopotentials")
    if isinstance(pseudopotentials, dict):
        for species, path in pseudopotentials.items():
            if isinstance(path, str) and not Path(path).is_absolute():
                pseudopotentials[species] = str((base_directory / path).resolve())
