"""tests/python/test_load_results_json.py — every loaded results.json is checked against its schema.

``helpers.load_results_json`` is how every suite reads a run's numbers, so it
is where a file that breaks ``schemas/q3/results.json`` is caught: whatever
system, flavor or option produced it. These tests pin that it refuses such a
file, and says where it broke, without running the engine.
"""

import json

import pytest

from helpers import load_results_json, repository_schemas, results_schema_errors

pytestmark = pytest.mark.skipif(
    repository_schemas() is None,
    reason="mat3ra-esse, which supplies the ESSE schemas the results schema references, is not installed",
)


def test_a_document_missing_a_group_is_reported_with_its_path():
    errors = results_schema_errors({"structure": {}, "provenance": {}, "properties": {}})
    assert "/: 'runtime_log' is a required property" in errors
    assert "/properties: 'records' is a required property" in errors


def test_loading_a_file_that_breaks_the_schema_fails_the_test(tmp_path):
    (tmp_path / "si_results.json").write_text(json.dumps({"energy_in_hartrees": 1.0}))
    with pytest.raises(pytest.fail.Exception, match="does not match schemas/q3/results.json"):
        load_results_json(tmp_path, "si")
