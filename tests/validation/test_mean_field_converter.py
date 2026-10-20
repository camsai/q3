"""
End-to-end validation for the mean-field converter seam
(`src/io/mean_field_record/`, `q3_utility convert`).

The claim this file gates is the one the whole seam exists to make:

    converting a saved run produces EXACTLY what the in-process export
    would have produced.

If that ever stops holding, the converter has become a second, divergent
implementation of the format -- which is the failure mode the seam was
introduced to prevent -- and every downstream BerkeleyGW consumer inherits
the divergence silently. So the comparison here is byte-for-byte, not
field-by-field: an approximate check would pass on exactly the kind of
low-level drift (record padding, a Fortran record marker, a truncated
mantissa) that a binary format consumer trips over.

Two bytes legitimately differ, and only two: records 1 of WFN and RHO carry
the file's creation date and time (pw2bgw.f90:730-737 / :1574-1581), which
are wall-clock stamps and therefore differ between any two runs -- including
two runs of the *same* code. Those records are excluded from the byte
comparison by skipping the first Fortran record (whose length is
self-describing, so no magic offset is hard-coded here) and then checked
separately for everything except the timestamps.

Skips cleanly when the converter binary was not built (the GPU CI workflows
build `--target q3 q3_gpu_*`, not the full target set) or when the silicon
run did not produce an export.
"""

import shutil
import struct
import subprocess
from pathlib import Path

import pytest
import yaml

from helpers import EXAMPLES_DIRECTORY, apply_config_overlay, run_q3
from helpers_berkeleygw import (
    read_berkeleygw_charge_density_file,
    read_berkeleygw_wavefunction_file,
)

SILICON_EXAMPLE_DIRECTORY = EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon"
NC_INPUT = SILICON_EXAMPLE_DIRECTORY / "nc.yml"

RECORD_FILE_NAME = "mean-field.q3mf"
# Records 1 of WFN and RHO hold stitle + sdate + stime. Only the first is
# reproducible; the other two are wall-clock stamps.
TIMESTAMP_FIELDS = ("creation_date", "creation_time")


def _first_fortran_record_end(path):
    """Byte offset just past record 1 of a Fortran unformatted sequential file.

    The layout is ``[int32 length][payload][int32 length]``, so the offset is
    derivable from the file itself -- deliberately, so that a change to record
    1's contents does not silently turn this into a comparison of the wrong
    bytes.
    """
    with open(path, "rb") as file_handle:
        leading_length_bytes = file_handle.read(4)
    if len(leading_length_bytes) != 4:
        raise AssertionError(f"{path} is too short to contain a Fortran record marker")
    (payload_length,) = struct.unpack("<i", leading_length_bytes)
    if payload_length < 0:
        raise AssertionError(f"{path}: record 1 declares a negative length ({payload_length})")
    return 4 + payload_length + 4


def _assert_identical_after_first_record(converted_path, in_process_path, label):
    """Both files must agree byte-for-byte from record 2 onwards."""
    converted_bytes = converted_path.read_bytes()
    in_process_bytes = in_process_path.read_bytes()

    converted_start = _first_fortran_record_end(converted_path)
    in_process_start = _first_fortran_record_end(in_process_path)
    assert converted_start == in_process_start, (
        f"{label}: record 1 has a different length in the converted file "
        f"({converted_start} bytes incl. markers) than in the in-process export "
        f"({in_process_start} bytes)"
    )

    converted_tail = converted_bytes[converted_start:]
    in_process_tail = in_process_bytes[in_process_start:]
    assert len(converted_tail) == len(in_process_tail), (
        f"{label}: converted file is {len(converted_bytes)} bytes, in-process export is "
        f"{len(in_process_bytes)} bytes (excluding record 1: "
        f"{len(converted_tail)} vs {len(in_process_tail)})"
    )
    if converted_tail != in_process_tail:
        first_difference = next(
            index
            for index, (left, right) in enumerate(zip(converted_tail, in_process_tail))
            if left != right
        )
        absolute_offset = converted_start + first_difference
        raise AssertionError(
            f"{label}: converted output differs from the in-process export at byte "
            f"{absolute_offset} (0x{absolute_offset:x}): converted has "
            f"0x{converted_tail[first_difference]:02x}, in-process has "
            f"0x{in_process_tail[first_difference]:02x}"
        )


def _normalize(value):
    """Plain-Python form of a reader's return value, comparable with ``==``.

    The readers return nested dicts/lists holding numpy arrays, which cannot be
    compared with ``==`` directly (elementwise result, ambiguous truth value).
    Timestamps and the reader's own bookkeeping (``source_path``, which differs
    by construction because the two files live in different directories) are
    dropped wherever they appear, at any depth.
    """
    if isinstance(value, dict):
        return {
            key: _normalize(nested)
            for key, nested in sorted(value.items())
            if key not in TIMESTAMP_FIELDS and key != "source_path"
        }
    if hasattr(value, "tolist"):  # numpy array or scalar
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _assert_headers_agree_except_timestamps(converted_header, in_process_header, label):
    """Record 1 minus the wall-clock stamps, plus every other parsed field."""
    converted = _normalize(converted_header)
    in_process = _normalize(in_process_header)
    assert set(converted) == set(in_process), (
        f"{label}: the two files parse to different key sets -- "
        f"only in converted: {sorted(set(converted) - set(in_process))}, "
        f"only in in-process: {sorted(set(in_process) - set(converted))}"
    )
    for key in sorted(converted):
        assert converted[key] == in_process[key], (
            f"{label}: field {key!r} differs -- converted {converted[key]!r}, "
            f"in-process {in_process[key]!r}"
        )


def _run_silicon_export(q3_command, run_directory):
    """Runs the silicon example with a BerkeleyGW export; returns its outdir.

    `use_deterministic_fft_planning` is on so that eigenvector gauge choices in
    degenerate subspaces are reproducible; without it a *re-run* would not be
    byte-identical, which would make this test's premise untestable. It is not
    needed for the converter comparison itself (that reads one run's record),
    but it keeps the exported baseline stable if this file is ever extended to
    compare across runs.
    """
    input_path = apply_config_overlay(
        NC_INPUT,
        {
            "control": {
                "mean_field_export": "berkeleygw",
                "use_deterministic_fft_planning": True,
            }
        },
        run_directory,
    )
    result = run_q3(q3_command, input_path, working_directory=str(run_directory))
    if result.returncode != 0:
        pytest.skip(
            "silicon run with control.mean_field_export: berkeleygw did not succeed "
            f"(exit {result.returncode}); stderr tail: {result.stderr[-400:]}"
        )

    with open(input_path) as file_handle:
        merged_config = yaml.safe_load(file_handle)
    outdir = Path(merged_config["control"]["outdir"])
    prefix = merged_config["control"].get("prefix", "q3")
    return outdir, prefix


@pytest.fixture(scope="module")
def silicon_export(q3_command, tmp_path_factory):
    """One silicon export shared by every test here (the SCF is the slow part)."""
    run_directory = tmp_path_factory.mktemp("mean_field_converter")
    outdir, prefix = _run_silicon_export(q3_command, run_directory)
    if not (outdir / "WFN").is_file() or not (outdir / "RHO").is_file():
        pytest.skip(f"silicon run produced no WFN/RHO in {outdir}")
    return {"outdir": outdir, "prefix": prefix, "run_directory": run_directory}


def test_converter_reproduces_the_in_process_export_byte_for_byte(
    q3_convert_command, silicon_export, tmp_path
):
    """`q3_utility convert` on a saved record == the export the run wrote itself."""
    outdir = silicon_export["outdir"]
    save_directory = outdir / f"{silicon_export['prefix']}.save"
    record_path = save_directory / RECORD_FILE_NAME

    # Not a skip: the run above DID export, and record_and_export.hpp writes the
    # record whenever `mean_field_export` is not "none". A missing record here is
    # the seam being broken, not the feature being absent.
    assert record_path.is_file(), (
        f"the run exported to BerkeleyGW but wrote no {RECORD_FILE_NAME} in {save_directory}; "
        "export_and_record_mean_field() should write one whenever mean_field_export != 'none'"
    )

    converted_directory = tmp_path / "converted"
    converted_directory.mkdir()
    result = subprocess.run(
        list(q3_convert_command)
        + [str(save_directory), "--format", "berkeleygw", "--outdir", str(converted_directory)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"q3_utility convert failed (exit {result.returncode})\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    )

    for file_name in ("WFN", "RHO"):
        converted_path = converted_directory / file_name
        in_process_path = outdir / file_name
        assert converted_path.is_file(), (
            f"q3_utility convert produced no {file_name} in {converted_directory}"
        )
        _assert_identical_after_first_record(converted_path, in_process_path, file_name)

    # vxc.dat is ASCII with no timestamp, so it must match in full.
    converted_vxc = converted_directory / "vxc.dat"
    in_process_vxc = outdir / "vxc.dat"
    if in_process_vxc.is_file():
        assert converted_vxc.is_file(), (
            f"q3_utility convert produced no vxc.dat in {converted_directory}"
        )
        assert converted_vxc.read_bytes() == in_process_vxc.read_bytes(), (
            "converted vxc.dat differs from the in-process export"
        )

    # Record 1 is excluded from the byte comparison above, so check it here:
    # everything in it except the two wall-clock stamps must still agree.
    _assert_headers_agree_except_timestamps(
        read_berkeleygw_wavefunction_file(str(converted_directory / "WFN")),
        read_berkeleygw_wavefunction_file(str(outdir / "WFN")),
        "WFN",
    )
    _assert_headers_agree_except_timestamps(
        read_berkeleygw_charge_density_file(str(converted_directory / "RHO")),
        read_berkeleygw_charge_density_file(str(outdir / "RHO")),
        "RHO",
    )


def test_converter_defaults_outdir_to_the_save_directorys_parent(
    q3_convert_command, silicon_export, tmp_path
):
    """With no --outdir, output lands where an in-process export would put it.

    Verified against a COPY of the save directory rather than the original, so
    the default cannot be satisfied by the in-process export's own leftovers.
    """
    staged_outdir = tmp_path / "staged"
    staged_save = staged_outdir / f"{silicon_export['prefix']}.save"
    staged_save.mkdir(parents=True)
    shutil.copy(
        silicon_export["outdir"] / f"{silicon_export['prefix']}.save" / RECORD_FILE_NAME,
        staged_save / RECORD_FILE_NAME,
    )

    result = subprocess.run(
        list(q3_convert_command) + [str(staged_save)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"q3_utility convert failed: {result.stderr[-2000:]}"
    for file_name in ("WFN", "RHO"):
        assert (staged_outdir / file_name).is_file(), (
            f"q3_utility convert did not default --outdir to the save directory's parent: "
            f"no {file_name} in {staged_outdir}"
        )
        _assert_identical_after_first_record(
            staged_outdir / file_name, silicon_export["outdir"] / file_name, file_name
        )


def test_converter_reports_a_missing_record_without_a_stack_trace(q3_convert_command, tmp_path):
    """A save directory with no record must fail with an actionable message."""
    empty_save = tmp_path / "q3.save"
    empty_save.mkdir()
    result = subprocess.run(
        list(q3_convert_command) + [str(empty_save)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0, "q3_utility convert succeeded on a save directory with no record"
    combined_output = result.stdout + result.stderr
    assert "mean_field_export" in combined_output, (
        "the missing-record message should name the control key that produces the record; "
        f"got: {combined_output[-2000:]}"
    )


def test_converter_rejects_a_file_that_is_not_a_record(q3_convert_command, tmp_path):
    """A wrong-magic file is diagnosed as such, not read as garbage."""
    save_directory = tmp_path / "q3.save"
    save_directory.mkdir()
    (save_directory / RECORD_FILE_NAME).write_bytes(b"not a q3 mean-field record at all\n" * 64)
    result = subprocess.run(
        list(q3_convert_command) + [str(save_directory)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0, (
        "q3_utility convert accepted a file that is not a mean-field record"
    )
    combined_output = result.stdout + result.stderr
    assert "magic" in combined_output.lower(), (
        f"expected a bad-magic diagnostic; got: {combined_output[-2000:]}"
    )


def test_no_record_is_written_when_no_export_was_requested(q3_command, tmp_path):
    """The record is opt-in: an ordinary run must not double its output.

    record_and_export.hpp deliberately ties the record to "the user asked for a
    mean-field export"; this pins that down, because writing it unconditionally
    would roughly double the on-disk size of every run.
    """
    input_path = apply_config_overlay(NC_INPUT, {}, tmp_path)
    result = run_q3(q3_command, input_path, working_directory=str(tmp_path))
    if result.returncode != 0:
        pytest.skip(f"baseline silicon run failed (exit {result.returncode})")

    with open(input_path) as file_handle:
        merged_config = yaml.safe_load(file_handle)
    outdir = Path(merged_config["control"]["outdir"])
    stray_records = list(outdir.rglob(RECORD_FILE_NAME))
    assert not stray_records, (
        f"a run with no mean_field_export wrote {RECORD_FILE_NAME}: {stray_records}"
    )
