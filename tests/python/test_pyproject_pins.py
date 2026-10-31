"""tests/python/test_pyproject_pins.py — pyproject.toml is the one pin of what builds Q3.

The Dockerfiles and the release workflow install ``".[build]"``, so they take
every version from it. pre-commit cannot read pyproject.toml, so its hooks pin
the generators' dependencies themselves, one exact version each; this keeps
those pins inside the ranges pyproject.toml allows.
"""

import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent


def _pyproject_requirements():
    """Every requirement pyproject.toml states, base and extras, by distribution name."""
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())["project"]
    requirements = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        requirements.extend(extra)
    return {Requirement(requirement).name: Requirement(requirement) for requirement in requirements}


def test_no_image_or_workflow_repeats_the_esse_version():
    for path in [
        REPOSITORY_ROOT / "Dockerfile",
        REPOSITORY_ROOT / "Dockerfile.mpi",
        REPOSITORY_ROOT / "Dockerfile.cuda",
        *(REPOSITORY_ROOT / ".github" / "workflows").glob("*.yml"),
    ]:
        assert "mat3ra-esse==" not in path.read_text(), f"{path.name} pins mat3ra-esse itself"


def test_the_pre_commit_hooks_pin_versions_pyproject_allows():
    """A hook pins one exact version, so its generated files are reproducible; pyproject.toml's range must allow it."""
    requirements = _pyproject_requirements()
    configuration = yaml.safe_load((REPOSITORY_ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [hook for repository in configuration["repos"] for hook in repository["hooks"]]
    checked = 0
    for hook in hooks:
        for pinned in hook.get("additional_dependencies", []):
            name, _, version = pinned.partition("==")
            if name in requirements and version:
                assert requirements[name].specifier.contains(version, prereleases=True), (
                    f"pre-commit hook {hook['id']} pins {pinned}; "
                    f"pyproject.toml requires {requirements[name]}"
                )
                checked += 1
    assert checked, "no pre-commit hook pins a package pyproject.toml requires"
