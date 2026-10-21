"""tests/python/test_generate_statistics.py — the core's and each add-on's counts, apart.

Covers ``scripts/repo/generate_statistics.py``: the core's STATISTICS.md leaves every
add-on path out, each add-on's own file counts only its paths, ``--check`` sees every
file, and an export can recount the tree it writes.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATOR_PATH = REPOSITORY_ROOT / "scripts" / "repo" / "generate_statistics.py"


def _load_generator():
    """Import the generator script as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("generate_statistics", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


def write(root, relative_path, text):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_tree(root):
    """A core with two headers and one test of each kind, and an add-on `alpha`."""
    write(root, "src/layer_0_core/thing.hpp", "int thing;\n\nint other;\n")
    write(root, "src/io/reader.hpp", "int reader;\n")
    write(root, "tests/unit/layer_0_core/test_thing.cpp", "TEST(Thing, Works) {}\n")
    write(root, "tests/validation/test_core.py", "def test_core():\n    pass\n")
    write(root, "src/addons/alpha/reader.hpp", "int alpha;\nint beta;\n")
    write(
        root,
        "tests/unit/addons/alpha/test_reader.cpp",
        "TEST(Alpha, Reads) {}\nTEST_F(Alpha, Writes) {}\n",
    )
    write(root, "tests/validation/addons/alpha/test_deck.py", "def test_deck():\n    pass\n")
    write(root, "tests/regression/addons/alpha/test_export.py", "def test_export():\n    pass\n")


def test_the_core_counts_leave_every_addon_path_out(tmp_path):
    make_tree(tmp_path)
    metrics = generator.collect_metrics(tmp_path)
    assert metrics["Header files (`.hpp`)"] == 2
    assert metrics["Lines of C++ (non-empty)"] == 3
    assert metrics["C++ unit/integration tests"] == 1
    assert metrics["Python validation tests"] == 1
    assert metrics["Python regression tests"] == 0


def test_an_addon_counts_only_its_own_paths(tmp_path):
    make_tree(tmp_path)
    metrics = generator.collect_metrics(tmp_path, "alpha")
    assert metrics["Header files (`.hpp`)"] == 1
    assert metrics["Lines of C++ (non-empty)"] == 2
    assert metrics["C++ unit/integration tests"] == 2
    assert metrics["Python validation tests"] == 1
    assert metrics["Python regression tests"] == 1


def test_each_addon_gets_its_own_file_and_the_core_file_names_none(tmp_path):
    make_tree(tmp_path)
    files = generator.expected_files(tmp_path)
    core, addon = tmp_path / "STATISTICS.md", tmp_path / "src/addons/alpha/STATISTICS.md"
    assert set(files) == {core, addon}
    assert "alpha" not in files[core]
    assert "`alpha`" in files[addon]


def test_a_tree_without_addons_gets_the_core_file_alone(tmp_path):
    write(tmp_path, "src/thing.hpp", "int thing;\n")
    assert list(generator.expected_files(tmp_path)) == [tmp_path / "STATISTICS.md"]


def test_check_sees_a_stale_addon_file(tmp_path):
    """--check covers every file, so an add-on's cannot drift unnoticed."""
    make_tree(tmp_path)
    write(tmp_path, "scripts/repo/generate_statistics.py", GENERATOR_PATH.read_text())
    command = [sys.executable, "scripts/repo/generate_statistics.py"]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    current = subprocess.run([*command, "--check"], cwd=tmp_path, capture_output=True)
    assert current.returncode == 0

    write(tmp_path, "src/addons/alpha/more.hpp", "int more;\n")
    stale = subprocess.run([*command, "--check"], cwd=tmp_path, capture_output=True, text=True)
    assert stale.returncode == 1
    assert stale.stderr.startswith("Stale: src/addons/alpha/STATISTICS.md."), stale.stderr


def test_an_export_recounts_the_tree_it_writes(tmp_path):
    """A copied STATISTICS.md describes another tree; the recount describes this one."""
    write(tmp_path, "src/thing.hpp", "int thing;\n")
    write(tmp_path, "STATISTICS.md", "counts from another tree\n")
    generator.write_core_statistics(tmp_path)
    assert "| Header files (`.hpp`) | 1 |" in (tmp_path / "STATISTICS.md").read_text()
