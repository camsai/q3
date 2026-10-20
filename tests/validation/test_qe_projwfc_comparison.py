"""
Validation tests: compare Q3's PDOS/Löwdin-charge output against Quantum
ESPRESSO's projwfc.x, for a non-magnetic semiconductor (Si, norm-conserving
and ultrasoft) and a spin-polarized ferromagnetic metal (Ni).

Roadmap Window 2 gate: Löwdin
charges Δ < 0.01 e and per-orbital integrated PDOS at matched broadening/grid.

Both codes independently converge their own SCF, then run a matched-
resolution NSCF (a dense, explicit Monkhorst-Pack grid, same nominal
dimensions on both sides -- Q3's is unreduced, QE's is symmetry-reduced by
default, which is fine: a converged BZ integral should agree regardless of
how the grid was internally reduced) before projecting onto atomic orbitals.
This is an end-to-end agreement check, not a comparison against a frozen
reference, so it also (re-)exercises Q3's own SCF/NSCF/LSDA machinery.

Requires projwfc.x, which is not built by default alongside pw.x in the QE
LibXC reference build (see memory/qe-reference-build-location.md) -- build it
once with:
    cd reference/dft/q-e/build_libxc && \
        cmake --build . --target qe_pp_projwfc_exe -j8

Run:
    .venv/bin/pytest tests/validation/test_qe_projwfc_comparison.py -v \
        --q3-bin ./build/q3
"""

import glob
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from q3.workflows import run_pdos_workflow

from helpers import (
    EXAMPLES_DIRECTORY,
    QE_BIN_DIR,
    QE_PW,
    RYDBERG_TO_ELECTRONVOLT,
    generate_qe_input,
    requires_qe,
)

# ─── Constants ────────────────────────────────────────────────────────────────

QE_PROJWFC = str(QE_BIN_DIR / "projwfc.x")

RY_TO_EV = RYDBERG_TO_ELECTRONVOLT

# Roadmap Window 2 gate.
LOWDIN_CHARGE_TOLERANCE_E = 0.01

# Not explicitly gated by the roadmap text; generous margins above the
# empirically observed agreement (charges ~1e-4 e, PDOS integrals ~0.1-2%,
# correlation ~0.97-0.999) -- tight enough to catch a real regression,
# loose enough not to flake on ordinary cross-code numerical noise (different
# mixing/diagonalization paths reaching the "same" self-consistent density).
PDOS_INTEGRAL_TOLERANCE_E = 0.1
PDOS_MIN_CORRELATION = 0.9
SPILLING_PARAMETER_TOLERANCE = 0.005

L_LABEL_FROM_ANGULAR_MOMENTUM = {0: "s", 1: "p", 2: "d", 3: "f"}


# ─── QE projwfc.x drivers ───────────────────────────────────────────────────


def run_qe_pw(input_path, timeout=1200):
    """Run QE pw.x (scf or nscf) and return stdout. Raises if it didn't finish.

    Unlike helpers.run_qe_full, this doesn't require a "!    total energy"
    line (which NSCF runs never print -- there's no SCF cycle to converge).
    """
    result = subprocess.run(
        [QE_PW, "-in", input_path], capture_output=True, text=True, timeout=timeout
    )
    if "JOB DONE" not in result.stdout:
        raise RuntimeError(
            f"QE pw.x did not finish.\nstdout(tail): {result.stdout[-2000:]}\n"
            f"stderr(tail): {result.stderr[-2000:]}"
        )
    return result.stdout


def write_projwfc_input(
    input_path, *, outdir, prefix, filpdos, emin_ev, emax_ev, deltae_ev, degauss_ry
):
    """Write a QE &projwfc namelist input file.

    degauss is in Ry here (not eV like Emin/Emax/DeltaE): it feeds directly
    into the same Ry-unit module variable &SYSTEM's own degauss sets --
    confirmed from projwfc.x's documented namelist handling, not guessed.
    """
    lines = [
        "&projwfc",
        f"  outdir = '{outdir}/'",
        f"  prefix = '{prefix}'",
        f"  filpdos = '{filpdos}'",
        f"  Emin = {emin_ev}",
        f"  Emax = {emax_ev}",
        f"  DeltaE = {deltae_ev}",
        "  ngauss = 0",
        f"  degauss = {degauss_ry}",
        "/",
    ]
    with open(input_path, "w") as file_handle:
        file_handle.write("\n".join(lines) + "\n")
    return input_path


def run_qe_projwfc(input_path, timeout=1200):
    result = subprocess.run(
        [QE_PROJWFC, "-in", input_path], capture_output=True, text=True, timeout=timeout
    )
    if "JOB DONE" not in result.stdout:
        raise RuntimeError(
            f"QE projwfc.x did not finish.\nstdout(tail): {result.stdout[-3000:]}\n"
            f"stderr(tail): {result.stderr[-3000:]}"
        )
    return result.stdout


# ─── QE output parsing ──────────────────────────────────────────────────────


def parse_qe_lowdin_charges(stdout):
    """Parse the 'Lowdin Charges' stdout block projwfc.x writes.

    For nspin=1, projwfc.x writes ONE LINE PER ATOM PER l (not one combined
    line per atom): e.g. for Si (s,p), a "Atom # 1: total charge = X, s = Y"
    line, then a SEPARATE "Atom # 1: total charge = X, p = Z, px=.., ..." line.
    For nspin=2, the analogous top summary line instead covers ALL l in one
    shot (spin-summed) -- either way, accumulating matched groups across lines
    per atom (instead of overwriting) handles both correctly.

    Returns {atom_index (1-based int): {"total": float, "s": float, "p": float, ...}}
    -- values already summed over m and (for nspin=2) both spin channels.
    """
    charges = {}
    in_block = False
    for line in stdout.split("\n"):
        if "Lowdin Charges" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if "Spilling Parameter" in line:
            break
        match = re.match(r"\s*Atom #\s*(\d+):\s*total charge\s*=\s*([-\d.]+)(.*)", line)
        if match:
            atom_index = int(match.group(1))
            total = float(match.group(2))
            rest = match.group(3)
            entry = charges.setdefault(atom_index, {})
            entry["total"] = total
            for l_label, value in re.findall(r"\b([spdf])\s*=\s*(-?\d+\.\d+)", rest):
                entry[l_label] = float(value)
    if not charges:
        raise RuntimeError("No 'Lowdin Charges' block found in QE projwfc.x stdout")
    return charges


def parse_qe_spilling_parameter(stdout):
    for line in stdout.split("\n"):
        if "Spilling Parameter" in line:
            return float(line.split(":")[1].strip())
    raise RuntimeError("Spilling Parameter not found in QE projwfc.x output")


def find_pdos_atm_files(filpdos_path, atom_index, element):
    """Return [(n, l_label, path), ...] for one atom's .pdos_atm files, sorted by n.

    n is QE's PP_CHI-file-order wfc index (as it appears in the .pdos_atm
    filename), not a principal quantum number.
    """
    pattern = f"{filpdos_path}.pdos_atm#{atom_index}({element})_wfc#*"
    results = []
    for path in glob.glob(pattern):
        match = re.search(r"_wfc#(\d+)\(([spdf])\)", os.path.basename(path))
        if match:
            results.append((int(match.group(1)), match.group(2), path))
    results.sort(key=lambda entry: entry[0])
    return results


def load_pdos_atm_file(path, nspin):
    """Load a .pdos_atm#..(l) file; return (energies_ry, ldos_states_per_ry).

    QE's "ldos(E)" column is the m-summed PDOS for this specific (atom, n, l)
    shell -- already exactly the per-shell quantity Q3's atom_shell_pdos
    reports, no extra summation needed. QE prints in eV / states-per-eV;
    convert to Ry / states-per-Ry to match Q3's native units.
    """
    data = np.loadtxt(path, comments="#")
    energies_ev = data[:, 0]
    ldos_per_ev = data[:, 1] if nspin == 1 else data[:, 1] + data[:, 2]
    return energies_ev / RY_TO_EV, ldos_per_ev * RY_TO_EV


# ─── Q3 <-> QE comparison ────────────────────────────────────────────────────


def compare_lowdin_charges(q3_charges, qe_charges):
    """Compare Q3's atom_shell charges against QE's parsed Lowdin Charges block.

    Both codes report charges already summed over the magnetic quantum number m
    (Q3's atom_shell grouping; QE's per-l lines) -- this side-steps any
    real-spherical-harmonic m-ordering convention mismatch between the two
    codes, since a sum over a complete set of m for a given l is rotationally
    invariant regardless of basis choice.

    QE's stdout "Lowdin Charges" block is atom+l resolved only -- the charge
    table it prints has no shell/n axis, so when an atom has more than one
    shell of the same l (e.g. Ni's 3S and 4S), QE's printed "s" value is their
    SUM, not either individually. Q3's atom_shell keeps them separate (a
    strictly finer-grained result, independently cross-checked against QE's
    own per-shell .pdos_atm#..(s) FILES by compare_pdos_curves below, which --
    unlike the stdout summary -- IS shell-resolved). So: aggregate Q3's
    same-(atom,l) shells together before comparing to QE's stdout value.

    Returns a list of dicts: atom_index, l_label, shell_labels, q3, qe, delta.
    """
    aggregated = {}
    for entry in q3_charges["atom_shell"]:
        l_label = L_LABEL_FROM_ANGULAR_MOMENTUM[entry["angular_momentum"]]
        key = (entry["atom_index"], l_label)
        bucket = aggregated.setdefault(key, {"charge": 0.0, "shell_labels": []})
        bucket["charge"] += entry["charge"]
        bucket["shell_labels"].append(entry["shell_label"])

    rows = []
    for (atom_index, l_label), bucket in aggregated.items():
        atom_index_1based = atom_index + 1  # Q3 is 0-indexed, QE is 1-indexed
        qe_value = qe_charges.get(atom_index_1based, {}).get(l_label)
        q3_value = bucket["charge"]
        delta = abs(q3_value - qe_value) if qe_value is not None else None
        rows.append(
            {
                "atom_index": atom_index,
                "l_label": l_label,
                "shell_labels": bucket["shell_labels"],
                "q3": q3_value,
                "qe": qe_value,
                "delta": delta,
            }
        )
    return rows


def compare_pdos_curves(q3_pdos, filpdos_path, element_by_species_index, nspin):
    """Compare Q3's atom_shell_pdos curves against QE's .pdos_atm files.

    Q3 enumerates orbitals in UPF PP_CHI file order (see
    layer_3_representation/pseudopotential/planewave/internal/atomic_orbital_builder.hpp);
    QE's wfc index n in the .pdos_atm#N(elem)_wfc#n(l) filename follows the
    same PP_CHI order. When an atom has more than one shell of the same l
    (e.g. Ni's 3S and 4S), match Q3's k-th occurrence of that l to QE's k-th
    (by ascending n) file of that l -- not by l alone, and not by literal n
    value (a global PP_CHI index, not a per-l counter).

    Returns a list of dicts with integral/correlation/max_abs_diff per shell.
    """
    q3_energies_ry = np.array(q3_pdos["energies_ry"])
    de = q3_energies_ry[1] - q3_energies_ry[0]

    occurrence_counter = {}  # (atom_index, l_label) -> how many seen so far
    rows = []
    for entry in q3_pdos["atom_shell_pdos"]:
        atom_index_1based = entry["atom_index"] + 1
        element = element_by_species_index[entry["species_type_index"]]
        l_label = L_LABEL_FROM_ANGULAR_MOMENTUM[entry["angular_momentum"]]

        key = (entry["atom_index"], l_label)
        occurrence_index = occurrence_counter.get(key, 0)
        occurrence_counter[key] = occurrence_index + 1

        qe_files = find_pdos_atm_files(filpdos_path, atom_index_1based, element)
        matches = [f for f in qe_files if f[1] == l_label]
        if occurrence_index >= len(matches):
            rows.append(
                {
                    "atom_index": entry["atom_index"],
                    "shell_label": entry["shell_label"],
                    "l_label": l_label,
                    "qe_n": None,
                    "q3_integral": None,
                    "qe_integral": None,
                    "max_abs_diff": None,
                    "correlation": None,
                }
            )
            continue

        qe_n, _qe_l, qe_path = matches[occurrence_index]
        qe_energies_ry, qe_ldos_per_ry = load_pdos_atm_file(qe_path, nspin=nspin)

        q3_dos = np.array(entry["dos"])
        # Interpolate QE's curve onto Q3's energy grid for a direct comparison
        # (the two Emin/Emax/DeltaE -> ne bin-count conventions differ slightly).
        qe_dos_interp = np.interp(q3_energies_ry, qe_energies_ry, qe_ldos_per_ry)

        rows.append(
            {
                "atom_index": entry["atom_index"],
                "shell_label": entry["shell_label"],
                "l_label": l_label,
                "qe_n": qe_n,
                "q3_integral": float(q3_dos.sum() * de),
                "qe_integral": float(qe_dos_interp.sum() * de),
                "max_abs_diff": float(np.max(np.abs(q3_dos - qe_dos_interp))),
                "correlation": float(np.corrcoef(q3_dos, qe_dos_interp)[0, 1]),
            }
        )
    return rows


def run_pdos_workflow_and_clean_up(scf_yml_path, **kwargs):
    """run_pdos_workflow(), deleting the generated NSCF companion YAML after.

    write_nscf_input() copies material:/pseudopotentials: paths verbatim from
    the SCF template, which are relative to the SCF YAML's OWN directory --
    so (unlike the outdir it does rewrite to be absolute) they only resolve
    correctly if the NSCF companion is written alongside the SCF input
    (run_pdos_workflow's default nscf_input_path). Unlike
    band_structure--*/density_of_states--* (which ship a committed *_nscf.yml
    companion), total_energy--* example directories don't, so left at its
    default this would leave a stray untracked file after every test run --
    clean it up explicitly instead of fighting the path-resolution default.
    """
    scf_yml_path = Path(scf_yml_path)
    generated_nscf_path = scf_yml_path.parent / (scf_yml_path.stem + "_pdos_nscf.yml")
    try:
        return run_pdos_workflow(scf_yml_path, **kwargs)
    finally:
        generated_nscf_path.unlink(missing_ok=True)


def assert_pdos_gates(charge_rows, pdos_rows, q3_spilling, qe_spilling):
    for row in charge_rows:
        assert row["qe"] is not None, (
            f"No matching QE Lowdin charge for atom {row['atom_index']} "
            f"l={row['l_label']} (Q3 shells {row['shell_labels']})"
        )
        assert row["delta"] < LOWDIN_CHARGE_TOLERANCE_E, (
            f"Lowdin charge mismatch: atom {row['atom_index']} shells "
            f"{row['shell_labels']} (l={row['l_label']}): "
            f"Q3={row['q3']:.4f} e, QE={row['qe']:.4f} e, "
            f"|delta|={row['delta']:.4f} e (tolerance {LOWDIN_CHARGE_TOLERANCE_E} e)"
        )

    for row in pdos_rows:
        assert row["q3_integral"] is not None, (
            f"No matching QE .pdos_atm file for atom {row['atom_index']} shell {row['shell_label']}"
        )
        integral_delta = abs(row["q3_integral"] - row["qe_integral"])
        assert integral_delta < PDOS_INTEGRAL_TOLERANCE_E, (
            f"PDOS integral mismatch: atom {row['atom_index']} shell "
            f"{row['shell_label']}: Q3={row['q3_integral']:.4f}, "
            f"QE={row['qe_integral']:.4f}, |delta|={integral_delta:.4f} "
            f"(tolerance {PDOS_INTEGRAL_TOLERANCE_E})"
        )
        assert row["correlation"] > PDOS_MIN_CORRELATION, (
            f"PDOS shape mismatch: atom {row['atom_index']} shell "
            f"{row['shell_label']}: correlation={row['correlation']:.4f} "
            f"(minimum {PDOS_MIN_CORRELATION})"
        )

    spilling_delta = abs(q3_spilling - qe_spilling)
    assert spilling_delta < SPILLING_PARAMETER_TOLERANCE, (
        f"Spilling parameter mismatch: Q3={q3_spilling:.4f}, QE={qe_spilling:.4f}, "
        f"|delta|={spilling_delta:.4f} (tolerance {SPILLING_PARAMETER_TOLERANCE})"
    )


# ─── Tests ───────────────────────────────────────────────────────────────────


@requires_qe
@pytest.mark.skipif(not os.path.exists(QE_PROJWFC), reason="QE projwfc.x not built")
def test_si_pdos_matches_qe_projwfc(q3_command, tmp_path):
    """Si (insulator, non-magnetic): Löwdin charges and per-shell PDOS vs QE."""
    broadening_ry = 0.01
    pdos_kgrid = (6, 6, 6)

    q3_result = run_pdos_workflow_and_clean_up(
        EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "nc.yml",
        kgrid_dimensions=pdos_kgrid,
        kgrid_shifts=(0, 0, 0),
        broadening_ry=broadening_ry,
        energy_steps=1500,
        q3_binary=q3_command[0],
    )
    q3_charges = q3_result["lowdin_charges"]
    q3_pdos = q3_result["pdos"]

    energies_ry = np.array(q3_pdos["energies_ry"])
    emin_ry, emax_ry = float(energies_ry.min()), float(energies_ry.max())
    deltae_ry = float(energies_ry[1] - energies_ry[0])

    qe_scf_input = generate_qe_input(
        case_name="si_pdos",
        material_name="silicon",
        pp_type="nc",
        pp_filename="Si.upf",
        ecutwfc=48.0,
        ecutrho=192.0,
        kgrid=[2, 2, 2],
        kshift=[1, 1, 1],
        number_of_bands=8,
        mixing_beta=0.3,
        calculation="scf",
        output_directory=str(tmp_path),
    )
    run_qe_pw(qe_scf_input)

    qe_nscf_input = generate_qe_input(
        case_name="si_pdos",
        material_name="silicon",
        pp_type="nc",
        pp_filename="Si.upf",
        ecutwfc=48.0,
        ecutrho=192.0,
        kgrid=list(pdos_kgrid),
        kshift=[0, 0, 0],
        number_of_bands=8,
        calculation="nscf",
        startingpot="file",
        output_directory=str(tmp_path),
    )
    run_qe_pw(qe_nscf_input)

    filpdos = str(tmp_path / "si_pdos_out")
    projwfc_input = write_projwfc_input(
        str(tmp_path / "si_pdos_projwfc.in"),
        outdir=str(tmp_path / "si_pdos"),
        prefix="si_pdos",
        filpdos=filpdos,
        emin_ev=emin_ry * RY_TO_EV,
        emax_ev=emax_ry * RY_TO_EV,
        deltae_ev=deltae_ry * RY_TO_EV,
        degauss_ry=broadening_ry,
    )
    projwfc_stdout = run_qe_projwfc(projwfc_input)

    qe_charges = parse_qe_lowdin_charges(projwfc_stdout)
    qe_spilling = parse_qe_spilling_parameter(projwfc_stdout)

    charge_rows = compare_lowdin_charges(q3_charges, qe_charges)
    pdos_rows = compare_pdos_curves(q3_pdos, filpdos, {0: "Si"}, nspin=1)

    print(f"\n  Si spilling: Q3={q3_charges['spilling_parameter']:.4f} QE={qe_spilling:.4f}")
    for row in charge_rows:
        print(f"  {row['shell_labels']} (l={row['l_label']}): Q3={row['q3']:.4f} QE={row['qe']}")
    for row in pdos_rows:
        print(
            f"  shell {row['shell_label']}: integral Q3={row['q3_integral']:.4f} "
            f"QE={row['qe_integral']:.4f} corr={row['correlation']:.4f}"
        )

    assert_pdos_gates(charge_rows, pdos_rows, q3_charges["spilling_parameter"], qe_spilling)


@requires_qe
@pytest.mark.skipif(not os.path.exists(QE_PROJWFC), reason="QE projwfc.x not built")
def test_si_ultrasoft_pdos_matches_qe_projwfc(q3_command, tmp_path):
    """Si with an ultrasoft dataset: the S-metric projections against QE.

    The case the norm-conserving ones above cannot cover. Ultrasoft
    pseudopotentials give up norm conservation, so the Kohn-Sham states are
    orthonormal in the metric of S = 1 + sum_ij q_ij |beta_i><beta_j| rather
    than of the identity, and both codes project as <phi'_i|S|psi_nk> with the
    atomic orbitals Löwdin-orthogonalized in that same metric (Vanderbilt,
    PRB 41, 7892 (1990)). Q3's own sum rule for those projections -- the
    Löwdin charges accounting for the valence electron count -- is checked
    without a reference code in test_pdos_s_metric.py; this is the cross-code
    half.
    """
    broadening_ry = 0.01
    pdos_kgrid = (6, 6, 6)
    number_of_bands = 8

    q3_result = run_pdos_workflow_and_clean_up(
        EXAMPLES_DIRECTORY / "total_energy--semiconductor--silicon" / "us.yml",
        kgrid_dimensions=pdos_kgrid,
        kgrid_shifts=(0, 0, 0),
        number_of_bands=number_of_bands,
        broadening_ry=broadening_ry,
        energy_steps=1500,
        q3_binary=q3_command[0],
    )
    q3_charges = q3_result["lowdin_charges"]
    q3_pdos = q3_result["pdos"]

    energies_ry = np.array(q3_pdos["energies_ry"])
    emin_ry, emax_ry = float(energies_ry.min()), float(energies_ry.max())
    deltae_ry = float(energies_ry[1] - energies_ry[0])

    common_kwargs = dict(
        material_name="silicon",
        pp_type="us",
        pp_filename="Si_PBE_USPP.UPF",
        ecutwfc=40.0,
        ecutrho=200.0,
        number_of_bands=number_of_bands,
        output_directory=str(tmp_path),
    )

    qe_scf_input = generate_qe_input(
        case_name="si_us_pdos",
        kgrid=[2, 2, 2],
        kshift=[1, 1, 1],
        mixing_beta=0.3,
        calculation="scf",
        **common_kwargs,
    )
    run_qe_pw(qe_scf_input)

    qe_nscf_input = generate_qe_input(
        case_name="si_us_pdos",
        kgrid=list(pdos_kgrid),
        kshift=[0, 0, 0],
        calculation="nscf",
        startingpot="file",
        **common_kwargs,
    )
    run_qe_pw(qe_nscf_input)

    filpdos = str(tmp_path / "si_us_pdos_out")
    projwfc_input = write_projwfc_input(
        str(tmp_path / "si_us_pdos_projwfc.in"),
        outdir=str(tmp_path / "si_us_pdos"),
        prefix="si_us_pdos",
        filpdos=filpdos,
        emin_ev=emin_ry * RY_TO_EV,
        emax_ev=emax_ry * RY_TO_EV,
        deltae_ev=deltae_ry * RY_TO_EV,
        degauss_ry=broadening_ry,
    )
    projwfc_stdout = run_qe_projwfc(projwfc_input)

    qe_charges = parse_qe_lowdin_charges(projwfc_stdout)
    qe_spilling = parse_qe_spilling_parameter(projwfc_stdout)

    charge_rows = compare_lowdin_charges(q3_charges, qe_charges)
    pdos_rows = compare_pdos_curves(q3_pdos, filpdos, {0: "Si"}, nspin=1)

    print(
        f"\n  Si ultrasoft spilling: Q3={q3_charges['spilling_parameter']:.4f} QE={qe_spilling:.4f}"
    )
    for row in charge_rows:
        print(f"  {row['shell_labels']} (l={row['l_label']}): Q3={row['q3']:.4f} QE={row['qe']}")
    for row in pdos_rows:
        print(
            f"  shell {row['shell_label']}: integral Q3={row['q3_integral']:.4f} "
            f"QE={row['qe_integral']:.4f} corr={row['correlation']:.4f}"
        )

    assert_pdos_gates(charge_rows, pdos_rows, q3_charges["spilling_parameter"], qe_spilling)


@requires_qe
@pytest.mark.skipif(not os.path.exists(QE_PROJWFC), reason="QE projwfc.x not built")
def test_ni_lsda_pdos_matches_qe_projwfc(q3_command, tmp_path):
    """Ni (FCC, ferromagnetic, LSDA): Löwdin charges and per-shell PDOS vs QE.

    Also exercises the multiple-shells-of-the-same-l case (3S and 4S), unlike
    the Si case above.
    """
    broadening_ry = 0.02
    pdos_kgrid = (6, 6, 6)
    pdos_kshift = (1, 1, 1)
    number_of_bands = 16

    q3_result = run_pdos_workflow_and_clean_up(
        EXAMPLES_DIRECTORY / "total_energy--metal--magnetic--nickel" / "nc.yml",
        kgrid_dimensions=pdos_kgrid,
        kgrid_shifts=pdos_kshift,
        number_of_bands=number_of_bands,
        broadening_ry=broadening_ry,
        energy_steps=1500,
        q3_binary=q3_command[0],
    )
    q3_charges = q3_result["lowdin_charges"]
    q3_pdos = q3_result["pdos"]

    energies_ry = np.array(q3_pdos["energies_ry"])
    emin_ry, emax_ry = float(energies_ry.min()), float(energies_ry.max())
    deltae_ry = float(energies_ry[1] - energies_ry[0])

    common_kwargs = dict(
        material_name="nickel",
        pp_type="nc",
        pp_filename="Ni.upf",
        ecutwfc=60.0,
        ecutrho=240.0,
        number_of_bands=number_of_bands,
        smearing="gaussian",
        degauss=0.02,
        number_of_spin_components=2,
        starting_magnetization={"Ni": 0.3},
        output_directory=str(tmp_path),
    )

    qe_scf_input = generate_qe_input(
        case_name="ni_pdos",
        kgrid=[12, 12, 12],
        kshift=[1, 1, 1],
        mixing_beta=0.1,
        calculation="scf",
        **common_kwargs,
    )
    run_qe_pw(qe_scf_input)

    qe_nscf_input = generate_qe_input(
        case_name="ni_pdos",
        kgrid=list(pdos_kgrid),
        kshift=list(pdos_kshift),
        calculation="nscf",
        startingpot="file",
        **common_kwargs,
    )
    run_qe_pw(qe_nscf_input)

    filpdos = str(tmp_path / "ni_pdos_out")
    projwfc_input = write_projwfc_input(
        str(tmp_path / "ni_pdos_projwfc.in"),
        outdir=str(tmp_path / "ni_pdos"),
        prefix="ni_pdos",
        filpdos=filpdos,
        emin_ev=emin_ry * RY_TO_EV,
        emax_ev=emax_ry * RY_TO_EV,
        deltae_ev=deltae_ry * RY_TO_EV,
        degauss_ry=broadening_ry,
    )
    projwfc_stdout = run_qe_projwfc(projwfc_input)

    qe_charges = parse_qe_lowdin_charges(projwfc_stdout)
    qe_spilling = parse_qe_spilling_parameter(projwfc_stdout)

    charge_rows = compare_lowdin_charges(q3_charges, qe_charges)
    pdos_rows = compare_pdos_curves(q3_pdos, filpdos, {0: "Ni"}, nspin=2)

    print(f"\n  Ni spilling: Q3={q3_charges['spilling_parameter']:.4f} QE={qe_spilling:.4f}")
    for row in charge_rows:
        print(f"  {row['shell_labels']} (l={row['l_label']}): Q3={row['q3']:.4f} QE={row['qe']}")
    for row in pdos_rows:
        print(
            f"  shell {row['shell_label']}: integral Q3={row['q3_integral']:.4f} "
            f"QE={row['qe_integral']:.4f} corr={row['correlation']:.4f}"
        )

    assert_pdos_gates(charge_rows, pdos_rows, q3_charges["spilling_parameter"], qe_spilling)
