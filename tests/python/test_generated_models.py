"""tests/python/test_generated_models.py — the Python models are generated from the schemas, and current.

``q3.generated`` is written by ``scripts/repo/generate_python_from_schemas.py``
from the schemas ``schemas/manifest.yaml`` lists, as ``src/generated/`` is for
C++. A schema changed without regenerating fails here, at unit-test speed,
rather than as a reader that silently drops the new key.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATOR = REPOSITORY_ROOT / "scripts" / "repo" / "generate_python_from_schemas.py"


@pytest.mark.skipif(
    importlib.util.find_spec("datamodel_code_generator") is None,
    reason="datamodel-code-generator (the dev extra) is not installed",
)
def test_the_generated_models_match_the_schemas():
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "module",
    [
        "q3.generated.results",
        "q3.generated.results.properties",
        "q3.generated.interpolation.results",
    ],
)
def test_a_generated_model_imports_on_its_own(module):
    """Importing a generated model first, before q3.io, must not meet it half-built.

    Their bases are in q3.io.results.base and q3.io.interpolation.results.base,
    and q3.io's packages import nothing themselves; a re-export in any of their
    __init__.py files would import the very module being built.
    """
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
