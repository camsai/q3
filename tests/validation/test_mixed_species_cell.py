"""
A cell with one ultrasoft species and one norm-conserving species.

Combining pseudopotential classes in one calculation is ordinary practice —
one element needs a particular dataset (a core-hole capable ultrasoft, say)
and the others do not — and it used to segfault.

The augmentation code path is selected by a *whole-cell* flag
(``has_ultrasoft``, set once from whether *any* species in the set is
ultrasoft-or-PAW), not per species. Every method on
``UltrasoftPlanewaveBasis`` was therefore reachable with a norm-conserving
species index, and the ones that went on to call
``PseudopotentialSet::as_ultrasoft()`` (src/layer_3_representation/
pseudopotential/set.hpp) hit a ``dynamic_cast`` guarded only by
``assert(result != nullptr)`` — compiled out entirely in a Release build
(``NDEBUG``). Silicon carbide (Si ultrasoft + C norm-conserving,
examples/data/materials/silicon_carbide.json — a genuine zincblende
structure, not a synthetic probe) segfaulted a Release q3 build with exit
code 139 partway through the first SCF iteration, in
``NormConservingPlanewaveWavefunctionInitializer::apply_overlap_contiguous``.

That is fixed: the per-species gate now lives in ``UltrasoftPlanewaveBasis``
itself, so a species with no augmentation charges answers q_ij = 0 and
contributes no augmentation charge instead of reaching for a table it does
not have. The same gate closed a second, quieter defect on the density side
— ``add_augmentation_charges`` indexed the empty Q(G) table of a
norm-conserving species and read past the end of it on every SCF iteration.
See tests/unit/layer_3_representation/basis/planewave/pseudopotential/
ultrasoft/test_init.cpp for the unit-level gates; this test is the
end-to-end one, and it asserts the calculation actually converges rather
than merely surviving.

Run:
    .venv/bin/pytest tests/validation/test_mixed_species_cell.py -v --q3-bin ./build/q3
"""

import math

import pytest

from helpers import EXAMPLES_DIRECTORY, run_q3_and_load_results

SILICON_CARBIDE_MATERIAL = EXAMPLES_DIRECTORY / "data" / "materials" / "silicon_carbide.json"
SILICON_ULTRASOFT_DATASET = (
    EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "us" / "Si_PBE_USPP.UPF"
)

CARBON_NORM_CONSERVING_DATASET = EXAMPLES_DIRECTORY / "data" / "pseudopotentials" / "nc" / "C.upf"

INPUT_TEMPLATE = """material: {material}
model: {{type: dft, subtype: gga, functional: pbe}}
method:
  type: pseudopotential
  subtype: ultrasoft
  cutoffs: {{wavefunction: 40.0, density: 200.0}}
  pseudopotentials: {{Si: {silicon_dataset}, C: {carbon_dataset}}}
  scf:
    convergence_threshold: 1.0e-06
    max_iterations: 100
    mixing: {{mode: broyden, beta: 0.3}}
    kgrid: {{dimensions: [2, 2, 2], shifts: [1, 1, 1]}}
control: {{calculation: scf, outdir: {outdir}, prefix: sic_mixed}}
"""


def test_mixed_ultrasoft_and_norm_conserving_cell_converges(q3_command, tmp_path):
    """A cell mixing pseudopotential classes runs and reaches self-consistency."""
    input_path = tmp_path / "sic.yml"
    input_path.write_text(
        INPUT_TEMPLATE.format(
            material=SILICON_CARBIDE_MATERIAL,
            silicon_dataset=SILICON_ULTRASOFT_DATASET,
            carbon_dataset=CARBON_NORM_CONSERVING_DATASET,
            outdir=tmp_path / "output",
        )
    )

    # Asserts a clean exit itself, which is where the SIGSEGV used to land.
    results, _ = run_q3_and_load_results(
        q3_command, input_path, prefix="sic_mixed", working_directory=tmp_path, timeout=600
    )

    assert results.runtime_log.converged, "mixed ultrasoft + norm-conserving SCF did not converge"

    total_energy = results.properties.total_energy.value
    assert math.isfinite(total_energy), f"total energy is not finite: {total_energy}"

    # A Q3-against-Q3 anchor, not a comparison against another code: no external
    # total energy exists for this pair of datasets, so the number below is
    # simply what Q3 converges to, recorded to catch silent numerical drift in
    # the mixed-species path. The window is wide enough to absorb BLAS and
    # platform differences and still far tighter than the failure it guards --
    # the pre-fix behaviour was a segfault, and the density-side defect it also
    # covers read uninitialized memory into the augmentation charge.
    assert total_energy == pytest.approx(-19.890019987928248, abs=5.0e-3), (
        f"mixed-species total energy drifted: {total_energy!r} Ry"
    )
