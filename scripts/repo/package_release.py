#!/usr/bin/env python3
"""Package a built Q3 tree as a release archive, with its licenses and an SBOM.

The release workflow calls this once per platform, after the build, so that
everything a release *claims* about an archive is computed from the archive
itself rather than written down by hand:

* **The binaries** come from ``cmake --install``, so the archive carries
  whatever the project's ``install()`` rules name. Enumerating binaries here
  would couple this script to names that change.
* **Third-party licenses** for everything linked *statically* -- the
  FetchContent tree is built with ``BUILD_SHARED_LIBS OFF`` -- are copied out of
  the very source directories the build used. MIT and BSD both require the
  notice to travel with a binary distribution, so a dependency whose license
  file cannot be found stops the packaging rather than shipping without it.
* **Dynamically linked libraries** are read from the binaries with ``ldd`` (or
  ``otool -L`` on macOS) and, on Debian-family systems, resolved to the package
  that owns each one. Those libraries are *not* in the archive; they are what
  the user's system must provide, and ``INSTALL.txt`` says how.
* **The SBOM** is CycloneDX 1.5 JSON built from the two lists above. A generic
  SBOM scanner run over a C++ binary finds almost nothing -- there is no
  package manifest inside an ELF file to read -- so the components are taken
  from the build's own dependency pins and the binaries' own link tables.

Usage::

    python scripts/repo/package_release.py --build-dir build --version 2026.10.4-0 \\
        --platform linux-x86_64 --out dist

Prints the paths of the archive and the SBOM it wrote.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform as platform_module
import re
import shutil
import subprocess
import sys
import tarfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CMAKE_LISTS_PATH = REPOSITORY_ROOT / "CMakeLists.txt"

# Files from the repository root that every archive carries.
TOP_LEVEL_DOCUMENTS = ("LICENSE.md", "README.md", "PRERELEASE-NOTICE.md")

# License-file names, tried in order, inside a dependency's source tree.
LICENSE_FILE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.MIT", "COPYING")


@dataclass(frozen=True)
class StaticDependency:
    """A FetchContent dependency that ends up inside the q3 binaries.

    ``content_name`` is the name passed to ``FetchContent_Declare``; the
    source directory and the pinned version are both looked up by it.
    ``required`` is False for a dependency the build may take from the system
    instead -- LibXC -- in which case it is linked dynamically and appears in
    the other list.
    """

    content_name: str
    license_id: str
    purl_type: str
    purl_name: str
    required: bool = True


# googletest and nanobind are fetched too, but neither is linked into a
# shipped binary: the first builds only the test executable and the second
# only the Python module, and the release archive carries neither.
STATIC_DEPENDENCIES = (
    StaticDependency("nlohmann_json", "MIT", "github", "nlohmann/json"),
    StaticDependency("yaml-cpp", "MIT", "github", "jbeder/yaml-cpp"),
    StaticDependency("spdlog", "MIT", "github", "gabime/spdlog"),
    StaticDependency("spglib", "BSD-3-Clause", "github", "spglib/spglib"),
    StaticDependency("Libxc", "MPL-2.0", "gitlab", "libxc/libxc", required=False),
)

# Shared libraries every system already has. They are still listed in the
# SBOM -- they are linked -- but leaving them out of the install line keeps it
# to the packages a user might actually be missing.
BASE_SYSTEM_PACKAGES = frozenset({"libc6", "libgcc-s1", "libstdc++6", "libm6"})


class PackagingError(RuntimeError):
    """The archive cannot be produced honestly (missing license, no binaries, ...)."""


@dataclass
class LinkedLibrary:
    """One shared library a binary loads at run time."""

    soname: str
    path: str | None
    package: str | None = None
    package_version: str | None = None
    required_by: set[str] = field(default_factory=set)


# ── Reading the build ─────────────────────────────────────────────────────────


def read_cmake_cache(build_directory: Path) -> dict[str, str]:
    """Parse ``CMakeCache.txt`` into {variable: value}, dropping the type."""
    cache_path = build_directory / "CMakeCache.txt"
    if not cache_path.exists():
        raise PackagingError(f"{build_directory} has no CMakeCache.txt; is it a build directory?")
    entries: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        entries[key_and_type.split(":", 1)[0]] = value
    return entries


def dependency_source_directory(cache: dict[str, str], content_name: str) -> Path:
    """Where the build took a FetchContent dependency's sources from.

    An explicit ``FETCHCONTENT_SOURCE_DIR_<NAME>`` override wins, exactly as it
    does for FetchContent itself; otherwise the sources are under the base
    directory in ``<lowercase-name>-src``.
    """
    override = cache.get(f"FETCHCONTENT_SOURCE_DIR_{content_name.upper()}", "")
    if override:
        return Path(override)
    base = cache.get("FETCHCONTENT_BASE_DIR", "")
    return Path(base) / f"{content_name.lower()}-src"


def find_license_file(source_directory: Path) -> Path | None:
    """The first conventional license file in a dependency's source tree."""
    for name in LICENSE_FILE_NAMES:
        candidate = source_directory / name
        if candidate.is_file():
            return candidate
    return None


def pinned_versions(cmake_text: str) -> dict[str, str]:
    """{content name: version} for every ``FetchContent_Declare`` in the text.

    The version is the ``GIT_TAG`` when there is one, and otherwise the release
    tag embedded in the ``URL`` (``.../tags/v1.15.3.tar.gz`` or
    ``.../download/v3.11.3/json.tar.xz``). A leading ``v`` is dropped, so
    the SBOM carries the version number rather than the tag spelling.
    """
    versions: dict[str, str] = {}
    for match in re.finditer(r"FetchContent_Declare\(\s*([^\s)]+)(.*?)\)", cmake_text, re.DOTALL):
        name, body = match.group(1), match.group(2)
        git_tag = re.search(r"GIT_TAG\s+(\S+)", body)
        url = re.search(r"URL\s+(\S+)", body)
        version = None
        if git_tag:
            version = git_tag.group(1)
        elif url:
            tag = re.search(
                r"/(?:tags|download)/([^/]+?)(?:\.tar\.(?:gz|xz)|\.zip|/)", url.group(1)
            )
            if tag:
                version = tag.group(1)
        if version:
            versions[name] = version[1:] if re.match(r"v\d", version) else version
    return versions


# ── Reading the binaries ──────────────────────────────────────────────────────


def parse_ldd_output(text: str) -> list[tuple[str, str | None]]:
    """[(soname, resolved path or None)] from ``ldd`` output.

    The vDSO has no file and the dynamic loader is listed by path alone; both
    are skipped. A library ``ldd`` cannot find is returned with no path, so
    the caller can refuse to package a binary that would not start.
    """
    libraries: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("linux-vdso") or "=>" not in line:
            continue
        soname, target = (part.strip() for part in line.split("=>", 1))
        if target.startswith("not found"):
            libraries.append((soname, None))
            continue
        path = target.split(" (", 1)[0].strip()
        libraries.append((soname, path or None))
    return libraries


def parse_otool_output(text: str) -> list[tuple[str, str | None]]:
    """[(soname, path)] from ``otool -L`` output, skipping the binary's own line."""
    libraries: list[tuple[str, str | None]] = []
    for line in text.splitlines()[1:]:
        path = line.strip().split(" (", 1)[0].strip()
        if path:
            libraries.append((Path(path).name, path))
    return libraries


def linked_libraries(binaries: list[Path]) -> list[LinkedLibrary]:
    """Every shared library any of ``binaries`` loads, merged by soname."""
    is_macos = sys.platform == "darwin"
    merged: dict[str, LinkedLibrary] = {}
    for binary in binaries:
        command = ["otool", "-L", str(binary)] if is_macos else ["ldd", str(binary)]
        output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
        parsed = parse_otool_output(output) if is_macos else parse_ldd_output(output)
        for soname, path in parsed:
            if path is None:
                raise PackagingError(
                    f"{binary.name} needs {soname}, which this system cannot find; "
                    f"an archive built here would not start anywhere"
                )
            library = merged.setdefault(soname, LinkedLibrary(soname=soname, path=path))
            library.required_by.add(binary.name)
    return sorted(merged.values(), key=lambda library: library.soname)


def resolve_debian_packages(libraries: list[LinkedLibrary]) -> None:
    """Fill in the owning Debian package and its version, where dpkg knows.

    The path is resolved first: libblas.so.3 and liblapack.so.3 are
    update-alternatives symlinks, and the package that matters is the one the
    alternative points at (OpenBLAS), not the one that owns the link name.
    Usrmerge means a library may be registered under /usr/lib or /lib, so both
    spellings are tried.
    """
    if shutil.which("dpkg-query") is None:
        return
    for library in libraries:
        if library.path is None:
            continue
        real = os.path.realpath(library.path)
        candidates = [real]
        if real.startswith("/usr/lib/"):
            candidates.append(real[len("/usr") :])
        elif real.startswith("/lib/"):
            candidates.append("/usr" + real)
        for candidate in candidates:
            owner = subprocess.run(["dpkg", "-S", candidate], capture_output=True, text=True)
            if owner.returncode != 0:
                continue
            package = owner.stdout.split(":", 1)[0].strip()
            version = subprocess.run(
                ["dpkg-query", "-W", "-f", "${Version}", package],
                capture_output=True,
                text=True,
            )
            library.package = package
            library.package_version = version.stdout.strip() or None
            break


def install_line(libraries: list[LinkedLibrary]) -> str | None:
    """The ``apt-get install`` line for the runtime packages a user may lack."""
    packages = sorted(
        {
            library.package
            for library in libraries
            if library.package and library.package not in BASE_SYSTEM_PACKAGES
        }
    )
    if not packages:
        return None
    return "sudo apt-get install -y " + " ".join(packages)


# ── The SBOM ──────────────────────────────────────────────────────────────────


def purl(kind: str, name: str, version: str | None) -> str:
    return f"pkg:{kind}/{name}" + (f"@{version}" if version else "")


def build_sbom(
    version: str,
    commit: str,
    platform_name: str,
    static_components: list[dict],
    libraries: list[LinkedLibrary],
) -> dict:
    """A CycloneDX 1.5 document describing one release archive.

    Static dependencies are ``required`` components carried inside the
    binaries. Dynamic libraries are ``required`` too -- the binaries do not
    start without them -- but carry a ``q3:linkage`` property of ``dynamic``
    and ``q3:distributed`` of ``false``: they are the user's system's, not
    ours, which is the distinction a license review of this archive turns on.
    """
    components = list(static_components)
    # One component per package, not per library: libfftw3 and
    # libfftw3_threads are one Debian package, as are libc and libm, and a
    # component list that repeats a package reads as two dependencies.
    grouped: dict[str, list[LinkedLibrary]] = {}
    for library in libraries:
        grouped.setdefault(library.package or library.soname, []).append(library)
    for name, members in grouped.items():
        first = members[0]
        component = {
            "type": "library",
            "name": name,
            "scope": "required",
            "properties": [
                {"name": "q3:linkage", "value": "dynamic"},
                {"name": "q3:distributed", "value": "false"},
                {"name": "q3:soname", "value": ",".join(m.soname for m in members)},
                {
                    "name": "q3:required-by",
                    "value": ",".join(sorted(set().union(*(m.required_by for m in members)))),
                },
            ],
        }
        if first.package_version:
            component["version"] = first.package_version
        if first.package:
            component["purl"] = purl("deb", f"ubuntu/{first.package}", first.package_version)
        components.append(component)

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, f'q3/{version}/{platform_name}')}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "q3",
                "version": version,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
                "properties": [
                    {"name": "q3:commit", "value": commit},
                    {"name": "q3:platform", "value": platform_name},
                ],
            },
        },
        "components": components,
    }


def system_static_libxc_archive(cache: dict[str, str]) -> Path | None:
    """The system ``libxc.a`` the build linked, if it took LibXC statically.

    This is the case ``ldd`` cannot see. Ubuntu's ``libxc-dev`` ships both a
    shared and a static library, and its CMake package exports the *static*
    one as the ``Libxc::xc`` target -- so a build that finds the system LibXC
    compiles it into every executable, and no shared ``libxc`` ever appears in
    the link table. A packager that trusted ``ldd`` would ship MPL-2.0 code
    with no notice and an SBOM that omits it.

    The imported location is read from the package's own exported target
    files, with ``${_IMPORT_PREFIX}`` resolved the way CMake resolves it: three
    directories above ``<prefix>/share/cmake/Libxc``.
    """
    libxc_directory = cache.get("Libxc_DIR", "")
    if not libxc_directory or not Path(libxc_directory).is_dir():
        return None
    prefix = Path(libxc_directory).parents[2]
    for targets_file in sorted(Path(libxc_directory).glob("*Targets-*.cmake")):
        text = targets_file.read_text(encoding="utf-8", errors="replace")
        for location in re.findall(r'IMPORTED_LOCATION\w*\s+"([^"]+)"', text):
            resolved = location.replace("${_IMPORT_PREFIX}", str(prefix))
            if resolved.endswith(".a"):
                return Path(resolved)
    return None


def binary_defines_symbol(binary: Path, symbol: str) -> bool:
    """Whether ``binary`` itself defines ``symbol`` (rather than importing it)."""
    if shutil.which("nm") is None:
        return False
    listing = subprocess.run(["nm", str(binary)], capture_output=True, text=True).stdout
    return any(
        line.split()[-1] == symbol and line.split()[-2] in {"T", "t"}
        for line in listing.splitlines()
        if len(line.split()) >= 2
    )


def debian_package_of(path: Path) -> tuple[str | None, str | None]:
    """(package, version) owning a file, or (None, None) without dpkg."""
    if shutil.which("dpkg-query") is None:
        return None, None
    owner = subprocess.run(["dpkg", "-S", str(path)], capture_output=True, text=True)
    if owner.returncode != 0:
        return None, None
    package_name = owner.stdout.split(":", 1)[0].strip()
    version = subprocess.run(
        ["dpkg-query", "-W", "-f", "${Version}", package_name], capture_output=True, text=True
    ).stdout.strip()
    return package_name, version or None


# ── Assembling the archive ────────────────────────────────────────────────────


def static_components_and_licenses(
    cache: dict[str, str], cmake_text: str, licenses_directory: Path
) -> list[dict]:
    """Copy each static dependency's license, returning its SBOM component.

    LibXC is the one optional entry: when the build took it from the system it
    is a shared library and is reported with the dynamic list instead, so its
    absence from the FetchContent tree is expected rather than an error.
    """
    versions = pinned_versions(cmake_text)
    components = []
    licenses_directory.mkdir(parents=True, exist_ok=True)
    for dependency in STATIC_DEPENDENCIES:
        source = dependency_source_directory(cache, dependency.content_name)
        if not source.is_dir():
            if dependency.required:
                raise PackagingError(
                    f"{dependency.content_name}: no source tree at {source}, so its license "
                    f"cannot be shipped with the binaries that contain it"
                )
            continue
        license_file = find_license_file(source)
        if license_file is None:
            raise PackagingError(
                f"{dependency.content_name}: no license file in {source} "
                f"(looked for {', '.join(LICENSE_FILE_NAMES)})"
            )
        shutil.copyfile(license_file, licenses_directory / f"{dependency.content_name}.txt")
        version = versions.get(dependency.content_name)
        component = {
            "type": "library",
            "name": dependency.content_name,
            "scope": "required",
            "licenses": [{"license": {"id": dependency.license_id}}],
            "purl": purl(dependency.purl_type, dependency.purl_name, version),
            "properties": [
                {"name": "q3:linkage", "value": "static"},
                {"name": "q3:distributed", "value": "true"},
            ],
        }
        if version:
            component["version"] = version
        components.append(component)
    return components


def system_static_libxc_component(
    cache: dict[str, str], binaries: list[Path], licenses_directory: Path
) -> dict | None:
    """The SBOM component for a system LibXC compiled into the binaries.

    Returns None when LibXC came from the FetchContent tree (already handled
    with the other static dependencies) or is genuinely linked as a shared
    library (then it is in the dynamic list). Otherwise its Debian copyright
    file -- which carries the MPL-2.0 text and the upstream notices -- is
    copied beside the other licenses, and packaging stops if it cannot be.
    """
    archive = system_static_libxc_archive(cache)
    if archive is None:
        return None
    if not any(binary_defines_symbol(binary, "xc_func_init") for binary in binaries):
        return None
    package_name, version = debian_package_of(archive)
    copyright_file = Path("/usr/share/doc") / (package_name or "") / "copyright"
    if not package_name or not copyright_file.is_file():
        raise PackagingError(
            f"LibXC is compiled into the binaries from {archive}, but its license text "
            f"cannot be located; it is MPL-2.0 and must ship with them"
        )
    licenses_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(copyright_file, licenses_directory / "Libxc.txt")
    component = {
        "type": "library",
        "name": "Libxc",
        "scope": "required",
        "licenses": [{"license": {"id": "MPL-2.0"}}],
        "purl": purl("deb", f"ubuntu/{package_name}", version),
        "properties": [
            {"name": "q3:linkage", "value": "static"},
            {"name": "q3:distributed", "value": "true"},
            {"name": "q3:source", "value": str(archive)},
        ],
    }
    if version:
        component["version"] = version
    return component


def write_install_notes(root: Path, platform_name: str, libraries: list[LinkedLibrary]) -> None:
    """INSTALL.txt: what the archive is, and what the system must provide."""
    lines = [
        f"Q3 release archive for {platform_name}.",
        "",
        "The executables are in bin/. They link the libraries below dynamically;",
        "those are not in this archive and must come from your system.",
        "",
    ]
    command = install_line(libraries)
    if command:
        lines += ["On Ubuntu 24.04 (the system this archive was built and tested on):", ""]
        lines += [f"    {command}", ""]
    lines.append("Linked libraries:")
    for library in libraries:
        owner = f"  ({library.package} {library.package_version or ''})".rstrip()
        lines.append(f"    {library.soname}{owner if library.package else ''}")
    lines += [
        "",
        "Licenses: Q3 is Apache-2.0 (LICENSE.md). The libraries compiled into the",
        "executables carry their own licenses, in licenses/. sbom.cdx.json lists",
        "every component, statically and dynamically linked, with its version.",
    ]
    (root / "INSTALL.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalized_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip what differs between two builds of the same content.

    Owner, group and modification time carry the runner, not the release;
    normalizing them is what lets two archives of identical files hash alike.
    """
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    return info


def write_archive(root: Path, archive_path: Path) -> None:
    """A gzip tarball of ``root``, members in sorted order.

    The gzip stream is opened by hand because ``tarfile``'s own ``w:gz`` writes
    the current time and the output file's name into the gzip header, so two
    archives of byte-identical members would still hash differently.
    """
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    with (
        archive_path.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=epoch
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        archive.add(root, arcname=root.name, recursive=False, filter=normalized_tarinfo)
        for path in sorted(root.rglob("*")):
            archive.add(
                path,
                arcname=str(Path(root.name) / path.relative_to(root)),
                recursive=False,
                filter=normalized_tarinfo,
            )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


SYSTEM_NAMES = {"linux": "linux", "darwin": "macos"}
MACHINE_NAMES = {"amd64": "x86_64", "aarch64": "arm64"}


def default_platform_name() -> str:
    """``<system>-<machine>`` in the spelling the release assets use."""
    system = SYSTEM_NAMES.get(sys.platform, sys.platform)
    machine = platform_module.machine().lower()
    return f"{system}-{MACHINE_NAMES.get(machine, machine)}"


def package(
    build_directory: Path, version: str, commit: str, platform_name: str, out_directory: Path
) -> tuple[Path, Path]:
    """Install, collect, describe and archive one build. Returns (archive, sbom)."""
    cache = read_cmake_cache(build_directory)
    name = f"q3-{version}-{platform_name}"
    staging = out_directory / "staging"
    root = staging / name
    if root.exists():
        shutil.rmtree(root)

    subprocess.run(
        ["cmake", "--install", str(build_directory), "--prefix", str(root)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    binaries = sorted(path for path in (root / "bin").glob("*") if path.is_file())
    if not binaries:
        raise PackagingError(f"cmake --install put no executables in {root / 'bin'}")

    for document in TOP_LEVEL_DOCUMENTS:
        source = REPOSITORY_ROOT / document
        if source.exists():
            shutil.copyfile(source, root / document)

    static_components = static_components_and_licenses(
        cache, CMAKE_LISTS_PATH.read_text(encoding="utf-8"), root / "licenses"
    )
    libxc_component = system_static_libxc_component(cache, binaries, root / "licenses")
    if libxc_component is not None:
        static_components.append(libxc_component)
    libraries = linked_libraries(binaries)
    resolve_debian_packages(libraries)
    write_install_notes(root, platform_name, libraries)

    sbom = build_sbom(version, commit, platform_name, static_components, libraries)
    sbom_text = json.dumps(sbom, indent=2) + "\n"
    (root / "sbom.cdx.json").write_text(sbom_text, encoding="utf-8")
    sbom_path = out_directory / f"{name}.sbom.cdx.json"
    sbom_path.write_text(sbom_text, encoding="utf-8")

    archive_path = out_directory / f"{name}.tar.gz"
    write_archive(root, archive_path)
    return archive_path, sbom_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--version", required=True, help="release tag, e.g. 2026.10.4-0")
    parser.add_argument("--commit", default="", help="the commit the build was made from")
    parser.add_argument("--platform", default=None, help="e.g. linux-x86_64 (default: detected)")
    parser.add_argument("--out", type=Path, required=True, help="directory for the archive")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    try:
        archive_path, sbom_path = package(
            arguments.build_dir.resolve(),
            arguments.version,
            arguments.commit,
            arguments.platform or default_platform_name(),
            arguments.out.resolve(),
        )
    except PackagingError as error:
        print(f"Packaging refused: {error}", file=sys.stderr)
        return 1
    print(archive_path)
    print(sbom_path)
    print(f"sha256 {sha256_of(archive_path)}  {archive_path.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
