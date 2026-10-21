"""tests/python/test_package_release.py — Unit tests for the release packager.

Covers ``scripts/repo/package_release.py``: dependency pins read from the real
``CMakeLists.txt``, ``ldd`` and ``otool`` parsing, where a dependency's sources
and license are found, detection of a system LibXC compiled in from its static
archive (the case ``ldd`` cannot see), the install line and the SBOM shape, and
the guarantee that two archives of the same files are byte-identical.
"""

import gzip
import hashlib
import importlib.util
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "package_release.py"


def _load_packager():
    """Import the packager script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("package_release", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


packager = _load_packager()


# ── Dependency pins ───────────────────────────────────────────────────────────


def test_every_shipped_dependency_has_a_pin_in_the_real_cmake_lists():
    """The SBOM's versions come from here, so a pin the parser misses is a blank."""
    versions = packager.pinned_versions(packager.CMAKE_LISTS_PATH.read_text())
    for dependency in packager.STATIC_DEPENDENCIES:
        assert dependency.content_name in versions, dependency.content_name
        assert versions[dependency.content_name], dependency.content_name


def test_pins_are_read_from_git_tags_and_release_urls_alike():
    cmake_text = """
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_Declare(
        spdlog
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz
    )
    FetchContent_Declare(
        spglib
        GIT_REPOSITORY https://github.com/spglib/spglib.git
        GIT_TAG        v2.7.0
    )
    FetchContent_Declare(
        Libxc
        GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
        GIT_TAG        7.0.0
    )
    """
    assert packager.pinned_versions(cmake_text) == {
        "nlohmann_json": "3.11.3",
        "spdlog": "1.15.3",
        "spglib": "2.7.0",
        "Libxc": "7.0.0",
    }


# ── Link tables ───────────────────────────────────────────────────────────────

LDD_OUTPUT = """\
\tlinux-vdso.so.1 (0x00007ffd5b9f0000)
\tlibfftw3.so.3 => /lib/x86_64-linux-gnu/libfftw3.so.3 (0x00007f1e7c000000)
\tliblapack.so.3 => /lib/x86_64-linux-gnu/liblapack.so.3 (0x00007f1e7b800000)
\tlibmissing.so.1 => not found
\t/lib64/ld-linux-x86-64.so.2 (0x00007f1e7c600000)
"""


def test_ldd_output_yields_each_library_with_its_path():
    parsed = packager.parse_ldd_output(LDD_OUTPUT)
    assert ("libfftw3.so.3", "/lib/x86_64-linux-gnu/libfftw3.so.3") in parsed
    assert ("liblapack.so.3", "/lib/x86_64-linux-gnu/liblapack.so.3") in parsed


def test_ldd_skips_the_vdso_and_the_loader():
    """Neither is a library the user's system has to provide."""
    names = [soname for soname, _ in packager.parse_ldd_output(LDD_OUTPUT)]
    assert not any(name.startswith("linux-vdso") for name in names)
    assert not any("ld-linux" in name for name in names)


def test_ldd_reports_a_missing_library_with_no_path():
    assert ("libmissing.so.1", None) in packager.parse_ldd_output(LDD_OUTPUT)


def test_otool_output_skips_the_binarys_own_line():
    output = """build/q3:
\t/opt/homebrew/opt/fftw/lib/libfftw3.3.dylib (compatibility version 10.0.0)
\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1700.255.0)
"""
    assert packager.parse_otool_output(output) == [
        ("libfftw3.3.dylib", "/opt/homebrew/opt/fftw/lib/libfftw3.3.dylib"),
        ("libc++.1.dylib", "/usr/lib/libc++.1.dylib"),
    ]


def test_a_library_the_system_cannot_find_stops_the_packaging(monkeypatch, tmp_path):
    """An archive built against a missing library would not start anywhere."""
    monkeypatch.setattr(packager.sys, "platform", "linux")
    monkeypatch.setattr(
        packager.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, LDD_OUTPUT, ""),
    )
    with pytest.raises(packager.PackagingError, match="libmissing.so.1"):
        packager.linked_libraries([tmp_path / "q3"])


# ── Where sources and licenses are found ──────────────────────────────────────


def test_an_explicit_source_override_wins_over_the_base_directory():
    cache = {
        "FETCHCONTENT_BASE_DIR": "/build/_deps",
        "FETCHCONTENT_SOURCE_DIR_YAML-CPP": "/opt/deps/yaml-cpp",
        "FETCHCONTENT_SOURCE_DIR_SPGLIB": "",
    }
    assert packager.dependency_source_directory(cache, "yaml-cpp") == Path("/opt/deps/yaml-cpp")
    assert packager.dependency_source_directory(cache, "spglib") == Path("/build/_deps/spglib-src")


def test_the_cmake_cache_is_parsed_without_its_types(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text(
        "// a comment\n# another\nFETCHCONTENT_BASE_DIR:PATH=/b/_deps\nQ3_GIT_COMMIT:STRING=\n"
    )
    assert packager.read_cmake_cache(tmp_path) == {
        "FETCHCONTENT_BASE_DIR": "/b/_deps",
        "Q3_GIT_COMMIT": "",
    }


def _cache_with_one_dependency_tree(tmp_path, name, license_name="LICENSE.MIT"):
    base = tmp_path / "_deps"
    for dependency in packager.STATIC_DEPENDENCIES:
        if dependency.content_name == "Libxc":
            continue
        source = base / f"{dependency.content_name.lower()}-src"
        source.mkdir(parents=True)
        if dependency.content_name != name:
            (source / "LICENSE").write_text(f"{dependency.content_name} license\n")
        elif license_name:
            (source / license_name).write_text(f"{name} license\n")
    return {"FETCHCONTENT_BASE_DIR": str(base)}


def test_every_static_dependency_ships_its_license(tmp_path):
    cache = _cache_with_one_dependency_tree(tmp_path, "yaml-cpp")
    components = packager.static_components_and_licenses(
        cache, packager.CMAKE_LISTS_PATH.read_text(), tmp_path / "licenses"
    )
    shipped = {path.stem for path in (tmp_path / "licenses").glob("*.txt")}
    assert shipped == {"nlohmann_json", "yaml-cpp", "spdlog", "spglib", "pocketfft"}
    assert {component["name"] for component in components} == shipped
    assert (tmp_path / "licenses" / "yaml-cpp.txt").read_text() == "yaml-cpp license\n"


def test_a_static_dependency_without_a_license_file_stops_the_packaging(tmp_path):
    """MIT and BSD both require the notice to travel with a binary distribution."""
    cache = _cache_with_one_dependency_tree(tmp_path, "spdlog", license_name=None)
    with pytest.raises(packager.PackagingError, match="spdlog: no license file"):
        packager.static_components_and_licenses(
            cache, packager.CMAKE_LISTS_PATH.read_text(), tmp_path / "licenses"
        )


def test_a_missing_optional_tree_is_skipped_but_a_required_one_is_not(tmp_path):
    cache = _cache_with_one_dependency_tree(tmp_path, "spglib")
    # Libxc has no tree here, and it is optional: no error, no component.
    components = packager.static_components_and_licenses(
        cache, packager.CMAKE_LISTS_PATH.read_text(), tmp_path / "licenses"
    )
    assert "Libxc" not in {component["name"] for component in components}

    shutil.rmtree(tmp_path / "_deps" / "spdlog-src")
    with pytest.raises(packager.PackagingError, match="spdlog: no source tree"):
        packager.static_components_and_licenses(
            cache, packager.CMAKE_LISTS_PATH.read_text(), tmp_path / "licenses2"
        )


# ── The LibXC case ldd cannot see ─────────────────────────────────────────────


def _fake_libxc_package(tmp_path, library_name):
    libxc_directory = tmp_path / "usr" / "share" / "cmake" / "Libxc"
    libxc_directory.mkdir(parents=True)
    (libxc_directory / "LibxcTargets-C-release.cmake").write_text(
        "set_target_properties(Libxc::xc PROPERTIES\n"
        f'  IMPORTED_LOCATION_RELEASE "${{_IMPORT_PREFIX}}/lib/x86_64-linux-gnu/{library_name}"\n'
        ")\n"
    )
    return {"Libxc_DIR": str(libxc_directory)}


def test_a_system_libxc_exported_as_a_static_archive_is_found(tmp_path):
    """Ubuntu's libxc-dev exports libxc.a, so it is compiled into the binary."""
    cache = _fake_libxc_package(tmp_path, "libxc.a")
    assert packager.system_static_libxc_archive(cache) == (
        tmp_path / "usr" / "lib" / "x86_64-linux-gnu" / "libxc.a"
    )


def test_a_system_libxc_exported_as_a_shared_library_is_left_to_the_link_table(tmp_path):
    cache = _fake_libxc_package(tmp_path, "libxc.so.9")
    assert packager.system_static_libxc_archive(cache) is None


def test_no_system_libxc_means_nothing_to_detect():
    assert packager.system_static_libxc_archive({}) is None


# ── The install line and the SBOM ─────────────────────────────────────────────


def _library(soname, package, version="1.0", required_by=("q3",)):
    return packager.LinkedLibrary(
        soname=soname,
        path=f"/lib/{soname}",
        package=package,
        package_version=version,
        required_by=set(required_by),
    )


def test_the_install_line_names_only_what_a_user_might_lack():
    libraries = [
        _library("libc.so.6", "libc6"),
        _library("libfftw3.so.3", "libfftw3-double3"),
        _library("libblas.so.3", "libopenblas0-openmp"),
        _library("libstdc++.so.6", "libstdc++6"),
    ]
    assert packager.install_line(libraries) == (
        "sudo apt-get install -y libfftw3-double3 libopenblas0-openmp"
    )


def test_no_known_packages_means_no_install_line():
    assert packager.install_line([_library("libfoo.dylib", None)]) is None


def test_the_sbom_lists_one_component_per_package_and_marks_what_ships():
    libraries = [
        _library("libfftw3.so.3", "libfftw3-double3", "3.3.10"),
        _library("libfftw3_threads.so.3", "libfftw3-double3", "3.3.10", ("q3_utility",)),
        _library("libc.so.6", "libc6", "2.39"),
    ]
    static = [{"type": "library", "name": "spglib", "version": "2.7.0", "properties": []}]
    sbom = packager.build_sbom("2026.10.4-0", "abc123", "linux-x86_64", static, libraries)

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert sbom["metadata"]["component"]["version"] == "2026.10.4-0"
    names = [component["name"] for component in sbom["components"]]
    assert names == ["spglib", "libfftw3-double3", "libc6"]

    fftw = sbom["components"][1]
    properties = {prop["name"]: prop["value"] for prop in fftw["properties"]}
    assert properties["q3:linkage"] == "dynamic"
    assert properties["q3:distributed"] == "false"
    assert properties["q3:soname"] == "libfftw3.so.3,libfftw3_threads.so.3"
    assert properties["q3:required-by"] == "q3,q3_utility"
    assert fftw["purl"] == "pkg:deb/ubuntu/libfftw3-double3@3.3.10"


def test_the_sbom_serial_number_is_stable_for_one_version_and_platform():
    """A deterministic serial is what lets two packagings of one build agree."""
    first = packager.build_sbom("1.0", "a", "linux-x86_64", [], [])
    second = packager.build_sbom("1.0", "b", "linux-x86_64", [], [])
    other = packager.build_sbom("1.0", "a", "linux-arm64", [], [])
    assert first["serialNumber"] == second["serialNumber"]
    assert first["serialNumber"] != other["serialNumber"]


# ── The archive itself ────────────────────────────────────────────────────────


def _tree(tmp_path):
    root = tmp_path / "q3-1.0-linux-x86_64"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "q3").write_bytes(b"\x7fELF fake")
    (root / "licenses").mkdir()
    (root / "licenses" / "spglib.txt").write_text("BSD\n")
    (root / "INSTALL.txt").write_text("install\n")
    return root


def test_every_archive_carries_the_third_party_notices():
    """The notices name FFTW as the GPL runtime dependency; the archive must carry them."""
    assert "THIRD_PARTY_NOTICES.md" in packager.TOP_LEVEL_DOCUMENTS
    assert (packager.REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").is_file()


def test_install_notes_say_fftw_is_a_gpl_runtime_dependency(tmp_path):
    fftw = packager.LinkedLibrary(
        "libfftw3.so.3", "/lib/libfftw3.so.3", "libfftw3-double3", "3.3.10"
    )
    packager.write_install_notes(tmp_path, "linux-x86_64", [fftw])
    notes = (tmp_path / "INSTALL.txt").read_text()
    assert "FFTW" in notes and "GPL-2.0-or-later" in notes
    assert "THIRD_PARTY_NOTICES.md" in notes


def test_two_archives_of_the_same_files_are_byte_identical(tmp_path, monkeypatch):
    """Owner, group, mtime and the gzip header's own timestamp all normalized."""
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    root = _tree(tmp_path)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    packager.write_archive(root, first)
    (root / "INSTALL.txt").touch()  # a new mtime on disk must not reach the archive
    packager.write_archive(root, second)
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )


def test_archive_members_are_sorted_rooted_and_anonymous(tmp_path):
    root = _tree(tmp_path)
    archive_path = tmp_path / "out.tar.gz"
    packager.write_archive(root, archive_path)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
    names = [member.name for member in members]
    assert names[0] == root.name
    assert names[1:] == sorted(names[1:])
    assert all(name.startswith(root.name) for name in names)
    assert all(member.uid == 0 and member.uname == "" and member.mtime == 0 for member in members)


def test_the_gzip_header_carries_no_filename_or_time(tmp_path):
    root = _tree(tmp_path)
    archive_path = tmp_path / "out.tar.gz"
    packager.write_archive(root, archive_path)
    header = archive_path.read_bytes()[:10]
    assert header[4:8] == b"\x00\x00\x00\x00"  # MTIME
    assert not (header[3] & 0x08)  # FNAME flag clear
    with gzip.open(io.BytesIO(archive_path.read_bytes())) as stream:
        assert stream.read(1)
