"""
What the spin-polarized validation tests share: the collapse tolerance, the
reported moment, and the material and input files a case writes.
"""

import json
import re

COLLAPSE_TOLERANCE_RYDBERG = 1e-8
"""
A non-magnetic cell at nspin=2 against the same cell at nspin=1.

Both measured cases are bit-identical (Si PAW -15.932701594 Ry, Cu ultrasoft
-403.779228652 Ry), so this is far tighter than the LSDA norm-conserving bound
next door -- the two paths here share every routine except the spin index. Any
real accounting error in the spin plumbing has been at the mRy scale or larger.
"""


def parse_total_magnetization(standard_output):
    """The converged moment, which is reported to stdout and not to results.json."""
    matches = re.findall(r"total magnetization\s*=\s*(-?\d+\.\d+)\s*Bohr mag/cell", standard_output)
    if not matches:
        raise AssertionError("no 'total magnetization' line in Q3 output")
    return float(matches[-1])


def write_material(path, name, lattice_type, lattice_constant_angstrom, angle_degrees):
    path.write_text(
        json.dumps(
            {
                "name": name,
                "lattice": {
                    "type": lattice_type,
                    "a": lattice_constant_angstrom,
                    "b": lattice_constant_angstrom,
                    "c": lattice_constant_angstrom,
                    "alpha": angle_degrees,
                    "beta": angle_degrees,
                    "gamma": angle_degrees,
                    "units": {"length": "angstrom", "angle": "degree"},
                },
                "basis": {
                    "units": "crystal",
                    "elements": [{"id": 0, "value": name}],
                    "coordinates": [{"id": 0, "value": [0, 0, 0]}],
                },
                "isNonPeriodic": False,
            }
        )
    )


def write_input(path, material_path, subtype, dataset_path, element, output_directory, scf_extra):
    scf_lines = "\n".join(f"    {line}" for line in scf_extra)
    path.write_text(
        f"""material: {material_path}
model: {{type: dft, subtype: gga, functional: pbe}}
method:
  type: pseudopotential
  subtype: {subtype}
  cutoffs: {{wavefunction: 40.0, density: 320.0}}
  pseudopotentials: {{{element}: {dataset_path}}}
  scf:
    convergence_threshold: 1.0e-06
    max_iterations: 200
    mixing: {{mode: broyden, beta: 0.1, preconditioning: none}}
    smearing: {{type: gaussian, degauss: 0.02}}
{scf_lines}
control: {{calculation: scf, outdir: {output_directory}, prefix: spin_case}}
"""
    )
