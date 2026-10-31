#!/usr/bin/env python3
"""Generate band structure + DOS comparison plots (Q3 vs Quantum ESPRESSO).

Produces individual plots (band structure + DOS panel) for each of the 4 cases
(Si NC, Si US, Cu NC, Cu US) and combined NC+US plots per material.

Usage:
    python scripts/plot/plot_bands_with_dos.py [--q3-bin BUILD/q3] [--qe-bin PATH/pw.x]

Paths are derived relative to this script's location (scripts/ inside Q3_ROOT).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import matplotlib
import numpy as np
import yaml as _yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# The q3 package lives in src/py/, which is not on sys.path when this script is
# run straight from a checkout (rather than from an installed q3).  Prepend it
# before importing q3 so both invocations work.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "py")
)

from q3.binary import run_q3  # noqa: E402 — needs the sys.path entry above
from q3.io.output_writer import extract_dos  # noqa: E402 — needs the sys.path entry above
from q3.io.results.classes import (  # noqa: E402 — needs the sys.path entry above
    Results,
    eigenvalue_arrays,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RY_TO_EV = 13.6058
BOHR_TO_ANGSTROM = 0.529177210903
LIBXC_PBE = "XC-000I-000I-101L-130L"

# FCC high-symmetry points (crystal fractional coordinates)
FCC_SPECIAL_POINTS = {
    "G": [0, 0, 0],
    "X": [0.5, 0, 0.5],
    "W": [0.5, 0.25, 0.75],
    "K": [0.375, 0.375, 0.75],
    "L": [0.5, 0.5, 0.5],
}
PATH_LABELS = ["G", "X", "W", "K", "G", "L"]
DISPLAY_LABELS = ["\u0393", "X", "W", "K", "\u0393", "L"]
POINTS_PER_SEGMENT = 30


def resolve_paths(arguments):
    """Resolve all project paths relative to this script's location."""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    q3_root = os.path.normpath(os.path.join(script_directory, "..", ".."))

    q3_binary = arguments.q3_bin or os.path.join(q3_root, "build", "q3")
    qe_binary = arguments.qe_bin

    # If no QE binary specified, try common relative location
    if qe_binary is None:
        candidate = os.path.normpath(
            os.path.join(q3_root, "..", "reference", "dft", "q-e", "build_libxc", "bin", "pw.x")
        )
        if os.path.isfile(candidate):
            qe_binary = candidate
        else:
            print(f"ERROR: QE binary not found at {candidate}")
            print("       Specify with --qe-bin /path/to/pw.x")
            sys.exit(1)

    pseudopotential_base = os.path.join(q3_root, "examples", "data", "pseudopotentials")
    figures_directory = os.path.join(q3_root, "docs", "figures")

    return q3_root, q3_binary, qe_binary, pseudopotential_base, figures_directory


# ---------------------------------------------------------------------------
# K-path construction
# ---------------------------------------------------------------------------
def build_kpath():
    """Build the FCC k-path and distance array."""
    kpoints = []
    for i in range(len(PATH_LABELS) - 1):
        point_start = np.array(FCC_SPECIAL_POINTS[PATH_LABELS[i]])
        point_end = np.array(FCC_SPECIAL_POINTS[PATH_LABELS[i + 1]])
        for j in range(POINTS_PER_SEGMENT):
            kpoints.append(point_start + (j / POINTS_PER_SEGMENT) * (point_end - point_start))
    kpoints.append(np.array(FCC_SPECIAL_POINTS[PATH_LABELS[-1]]))
    kpoints = np.array(kpoints)

    distances = [0.0]
    for i in range(1, len(kpoints)):
        distances.append(distances[-1] + np.linalg.norm(kpoints[i] - kpoints[i - 1]))
    distances = np.array(distances)

    high_symmetry_positions = [
        distances[i * POINTS_PER_SEGMENT] if i < len(PATH_LABELS) - 1 else distances[-1]
        for i in range(len(PATH_LABELS))
    ]
    return kpoints, distances, high_symmetry_positions


# ---------------------------------------------------------------------------
# QE output parsers
# ---------------------------------------------------------------------------
def parse_qe_fermi(stdout):
    """Extract Fermi energy from QE output."""
    for line in stdout.split("\n"):
        if "the Fermi energy is" in line:
            return float(line.split()[-2])
        if "highest occupied" in line and "lowest unoccupied" in line:
            numbers = line.split(":")[-1].split()
            if len(numbers) >= 2:
                return (float(numbers[0]) + float(numbers[1])) / 2
    return None


def parse_qe_bands(stdout):
    """Extract eigenvalues from QE output."""
    index = stdout.find("End of band")
    if index < 0:
        index = stdout.find("End of self")
    if index < 0:
        return None
    text = stdout[index:]
    eigenvalues = []
    current_kpoint = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("k ="):
            if current_kpoint:
                eigenvalues.append(np.array(current_kpoint))
            current_kpoint = []
        else:
            try:
                current_kpoint.extend(float(x) for x in line.split())
            except ValueError:
                pass
    if current_kpoint:
        eigenvalues.append(np.array(current_kpoint))
    return np.array(eigenvalues) if eigenvalues else None


def parse_qe_kweights(stdout):
    """Extract k-point weights from QE SCF output."""
    weights = []
    for line in stdout.split("\n"):
        if "wk =" in line:
            try:
                weights.append(float(line.split("wk =")[-1].strip()))
            except ValueError:
                pass
    return np.array(weights) if weights else None


# ---------------------------------------------------------------------------
# QE runner
# ---------------------------------------------------------------------------
def run_qe(
    qe_binary,
    kpoints,
    celldm1,
    species,
    positions,
    pseudopotential_directory,
    ecutwfc,
    ecutrho,
    number_of_bands,
    smearing="",
    kshifts="1 1 1",
):
    """Run QE SCF + bands calculation, return (band_eigs, scf_eigs, fermi, weights)."""
    number_of_bands_padded = number_of_bands + 4
    with tempfile.TemporaryDirectory() as temporary_directory:
        os.symlink(pseudopotential_directory, os.path.join(temporary_directory, "pseudo"))
        number_of_atoms = positions.strip().count("\n") + 1
        number_of_types = species.strip().count("\n") + 1

        # SCF input
        scf_input = (
            f"&CONTROL\n"
            f" calculation='scf', pseudo_dir='./pseudo/', outdir='./tmp/'\n"
            f" verbosity='high'\n"
            f"/\n"
            f"&SYSTEM\n"
            f" ibrav=2, celldm(1)={celldm1:.10f}\n"
            f" nat={number_of_atoms}, ntyp={number_of_types},"
            f" ecutwfc={ecutwfc}, ecutrho={ecutrho},"
            f" nbnd={number_of_bands_padded}\n"
            f" input_dft='{LIBXC_PBE}'\n"
        )
        if smearing:
            scf_input += f" {smearing}\n"
        scf_input += (
            f"/\n&ELECTRONS\n conv_thr=1e-8\n/\n"
            f"ATOMIC_SPECIES\n{species}\n"
            f"ATOMIC_POSITIONS crystal\n{positions}\n"
            f"K_POINTS automatic\n 8 8 8 {kshifts}\n"
        )
        with open(os.path.join(temporary_directory, "scf.in"), "w") as f:
            f.write(scf_input)
        os.makedirs(os.path.join(temporary_directory, "tmp"))

        result_scf = subprocess.run(
            [qe_binary, "-i", "scf.in"], cwd=temporary_directory, capture_output=True, text=True
        )
        if result_scf.returncode != 0:
            return None, None, None, None

        fermi_energy = parse_qe_fermi(result_scf.stdout)
        scf_eigenvalues = parse_qe_bands(result_scf.stdout)
        scf_weights = parse_qe_kweights(result_scf.stdout)

        # Bands input
        kpoints_block = f"{len(kpoints)}\n"
        for k in kpoints:
            kpoints_block += f" {k[0]:.10f} {k[1]:.10f} {k[2]:.10f} 1.0\n"
        bands_input = (
            f"&CONTROL\n"
            f" calculation='bands', pseudo_dir='./pseudo/', outdir='./tmp/'\n"
            f" verbosity='high'\n"
            f"/\n"
            f"&SYSTEM\n"
            f" ibrav=2, celldm(1)={celldm1:.10f}\n"
            f" nat={number_of_atoms}, ntyp={number_of_types},"
            f" ecutwfc={ecutwfc}, ecutrho={ecutrho},"
            f" nbnd={number_of_bands_padded}\n"
            f" input_dft='{LIBXC_PBE}'\n"
        )
        if smearing:
            bands_input += f" {smearing}\n"
        bands_input += (
            f"/\n&ELECTRONS\n/\n"
            f"ATOMIC_SPECIES\n{species}\n"
            f"ATOMIC_POSITIONS crystal\n{positions}\n"
            f"K_POINTS crystal\n{kpoints_block}"
        )
        with open(os.path.join(temporary_directory, "bands.in"), "w") as f:
            f.write(bands_input)

        result_bands = subprocess.run(
            [qe_binary, "-i", "bands.in"], cwd=temporary_directory, capture_output=True, text=True
        )
        if result_bands.returncode != 0:
            return None, None, None, None

        band_eigenvalues = parse_qe_bands(result_bands.stdout)
        if band_eigenvalues is not None and band_eigenvalues.shape[1] > number_of_bands:
            band_eigenvalues = band_eigenvalues[:, :number_of_bands]
        if scf_eigenvalues is not None and scf_eigenvalues.shape[1] > number_of_bands:
            scf_eigenvalues = scf_eigenvalues[:, :number_of_bands]

        return band_eigenvalues, scf_eigenvalues, fermi_energy, scf_weights


# ---------------------------------------------------------------------------
# DOS computation
# ---------------------------------------------------------------------------
def compute_dos(
    eigenvalues_ev, weights=None, sigma_ev=0.3, energy_min=-20, energy_max=20, number_of_points=2000
):
    """Compute weighted Gaussian-broadened DOS from eigenvalues.

    Args:
        eigenvalues_ev: (number_of_kpoints, number_of_bands) array in eV
        weights: (number_of_kpoints,) k-point integration weights (sum to 2.0)
        sigma_ev: Gaussian broadening width in eV
    """
    energies = np.linspace(energy_min, energy_max, number_of_points)
    dos = np.zeros_like(energies)
    number_of_kpoints, number_of_bands = eigenvalues_ev.shape
    if weights is None:
        weights = np.ones(number_of_kpoints) * 2.0 / number_of_kpoints
    for kpoint_index in range(number_of_kpoints):
        for band_index in range(number_of_bands):
            eigenvalue = eigenvalues_ev[kpoint_index, band_index]
            dos += weights[kpoint_index] * np.exp(-0.5 * ((energies - eigenvalue) / sigma_ev) ** 2)
    dos /= sigma_ev * math.sqrt(2 * math.pi)
    return energies, dos


def compute_fermi_silicon(eigenvalues_ev):
    """Estimate Fermi level for Si as midgap between bands 4 and 5."""
    return (eigenvalues_ev[:, 3].max() + eigenvalues_ev[:, 4].min()) / 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Q3 vs QE band structure + DOS comparison plots."
    )
    parser.add_argument("--q3-bin", default=None, help="Path to Q3 binary (default: BUILD_DIR/q3)")
    parser.add_argument(
        "--qe-bin",
        default=None,
        help="Path to QE pw.x binary (default: auto-detect from ../reference/)",
    )
    arguments = parser.parse_args()

    (q3_root, q3_binary, qe_binary, pseudopotential_base, figures_directory) = resolve_paths(
        arguments
    )

    print(f"Q3 root:  {q3_root}")
    print(f"Q3 bin:   {q3_binary}")
    print(f"QE bin:   {qe_binary}")
    print(f"Figures:  {figures_directory}")

    kpoints, distances, high_symmetry_positions = build_kpath()

    # --- Material parameters ---
    with open(os.path.join(q3_root, "examples", "data", "materials", "silicon.json")) as f:
        silicon = json.load(f)
    with open(os.path.join(q3_root, "examples", "data", "materials", "copper.json")) as f:
        copper = json.load(f)

    silicon_celldm = silicon["lattice"]["a"] * math.sqrt(2) / BOHR_TO_ANGSTROM
    copper_celldm = copper["lattice"]["a"] * math.sqrt(2) / BOHR_TO_ANGSTROM

    # (label, celldm1, species, positions, pp_dir, ecutwfc, ecutrho,
    #  nbnd, smearing, material, kshifts)
    cases = [
        (
            "Si NC",
            silicon_celldm,
            " Si 28.085 Si.upf",
            " Si 0.0 0.0 0.0\n Si 0.25 0.25 0.25",
            os.path.join(pseudopotential_base, "nc"),
            48,
            192,
            12,
            "",
            "Si",
            "0 0 0",
        ),
        (
            "Si US",
            silicon_celldm,
            " Si 28.085 Si_PBE_USPP.UPF",
            " Si 0.0 0.0 0.0\n Si 0.25 0.25 0.25",
            os.path.join(pseudopotential_base, "us"),
            40,
            200,
            12,
            "",
            "Si",
            "1 1 1",
        ),
        (
            "Cu NC",
            copper_celldm,
            " Cu 63.546 Cu.upf",
            " Cu 0.0 0.0 0.0",
            os.path.join(pseudopotential_base, "nc"),
            120,
            480,
            20,
            "occupations='smearing', smearing='gaussian', degauss=0.02",
            "Cu",
            "1 1 1",
        ),
        (
            "Cu US",
            copper_celldm,
            " Cu 63.546 cu_pbe_v1.2.uspp.F.UPF",
            " Cu 0.0 0.0 0.0",
            os.path.join(pseudopotential_base, "us"),
            40,
            200,
            20,
            "occupations='smearing', smearing='gaussian', degauss=0.02",
            "Cu",
            "1 1 1",
        ),
    ]

    q3_band_results = {
        "Si NC": os.path.join(
            q3_root,
            "examples",
            "band_structure--semiconductor--silicon",
            "output",
            "si_bands_nc_results.json",
        ),
        "Si US": os.path.join(
            q3_root,
            "examples",
            "band_structure--semiconductor--silicon",
            "output",
            "si_bands_us_results.json",
        ),
        "Cu NC": os.path.join(
            q3_root,
            "examples",
            "band_structure--metal--copper",
            "output",
            "cu_bands_nc_results.json",
        ),
        "Cu US": os.path.join(
            q3_root,
            "examples",
            "band_structure--metal--copper",
            "output",
            "cu_bands_us_results.json",
        ),
    }

    q3_scf_configs = {
        "Si NC": os.path.join(
            q3_root, "examples", "density_of_states--semiconductor--silicon", "nc.yml"
        ),
        "Si US": os.path.join(
            q3_root, "examples", "total_energy--semiconductor--silicon", "us.yml"
        ),
        "Cu NC": os.path.join(q3_root, "examples", "density_of_states--metal--copper", "nc.yml"),
        "Cu US": os.path.join(q3_root, "examples", "total_energy--metal--copper", "us.yml"),
    }

    # --- Step 1: Q3 SCF for DOS (8x8x8 k-grid) ---
    print("\nRunning Q3 SCF for DOS (8x8x8 k-grid)...")

    q3_dos_data = {}
    for label in q3_scf_configs:
        print(f"  {label}...", flush=True)
        try:
            source_config = q3_scf_configs[label]
            with open(source_config) as f:
                config = _yaml.safe_load(f)

            config.setdefault("control", {})["calculation"] = "self_consistent_field"
            config.setdefault("method", {}).setdefault("scf", {}).setdefault("kgrid", {})[
                "dimensions"
            ] = [8, 8, 8]

            number_of_bands = 12 if "Si" in label else 20
            config["method"]["scf"]["number_of_bands"] = number_of_bands

            output_directory = config.get("control", {}).get("outdir", "./output")
            if not os.path.isabs(output_directory):
                config["control"]["outdir"] = os.path.join(
                    os.path.dirname(os.path.abspath(source_config)), output_directory + "_dos"
                )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yml", dir=os.path.dirname(source_config), delete=False
            ) as temporary_file:
                _yaml.dump(config, temporary_file, default_flow_style=False)
                temporary_path = temporary_file.name

            result = run_q3(temporary_path, q3_binary=q3_binary, capture_output=True)
            os.unlink(temporary_path)

            q3_dos_data[label] = extract_dos(result, broadening_ry=0.022, energy_steps=2000)
            fermi_ev = q3_dos_data[label]["fermi_energy_ry"] * RY_TO_EV
            kpoint_count = len(result.properties.eigenvalues_at_kpoints.values)
            print(f"    OK, {kpoint_count} kpts, Ef={fermi_ev:.4f} eV")
        except Exception as exception:
            print(f"    FAILED: {exception}")

    # --- Step 2: QE bands + SCF ---
    print("\nRunning QE...")
    qe_data = {}
    for (
        label,
        celldm1,
        species,
        positions,
        pseudopotential_directory,
        ecutwfc,
        ecutrho,
        number_of_bands,
        smearing,
        material,
        kshifts,
    ) in cases:
        print(f"  {label}...", flush=True)
        band_eigenvalues, scf_eigenvalues, fermi_energy, scf_weights = run_qe(
            qe_binary,
            kpoints,
            celldm1,
            species,
            positions,
            pseudopotential_directory,
            ecutwfc,
            ecutrho,
            number_of_bands,
            smearing,
            kshifts=kshifts,
        )
        if band_eigenvalues is not None:
            print(f"    OK: {band_eigenvalues.shape}, Ef={fermi_energy} eV")
            qe_data[label] = {
                "bands": band_eigenvalues,
                "scf_eigs": scf_eigenvalues,
                "fermi": fermi_energy,
                "scf_weights": scf_weights,
            }
        else:
            print("    FAILED")

    # --- Step 3: Individual plots (band structure + DOS) ---
    os.makedirs(figures_directory, exist_ok=True)
    print("\nPlotting...")

    broadening_ev = 0.022 * RY_TO_EV

    for label, *rest in cases:
        material = rest[-2]
        if not os.path.exists(q3_band_results[label]):
            print(f"  Skip {label}: no NSCF")
            continue
        eigenvalues = eigenvalue_arrays(
            Results.read(q3_band_results[label]).properties.eigenvalues_at_kpoints
        )[0]
        q3_bands = np.array([kpoint[0] for kpoint in eigenvalues]) * RY_TO_EV

        if label not in qe_data:
            print(f"  Skip {label}: no QE")
            continue
        qe_bands = qe_data[label]["bands"]
        qe_fermi = qe_data[label]["fermi"]
        qe_scf_eigenvalues = qe_data[label]["scf_eigs"]
        number_of_common_bands = min(q3_bands.shape[1], qe_bands.shape[1])

        if material == "Si":
            q3_fermi = compute_fermi_silicon(q3_bands)
            qe_fermi_shifted = qe_fermi if qe_fermi else compute_fermi_silicon(qe_bands)
        else:
            qe_fermi_shifted = qe_fermi if qe_fermi else 0
            q3_fermi = qe_fermi_shifted

        q3_shifted = q3_bands - q3_fermi
        qe_shifted = qe_bands - qe_fermi_shifted

        # Create figure with band structure + DOS panel
        fig = plt.figure(figsize=(10, 7))
        grid_spec = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.0)
        axis_bands = fig.add_subplot(grid_spec[0])
        axis_dos = fig.add_subplot(grid_spec[1], sharey=axis_bands)

        for band_index in range(number_of_common_bands):
            axis_bands.plot(
                distances,
                q3_shifted[:, band_index],
                color="#2563eb",
                lw=1.5,
                alpha=0.85,
                label="Q3" if band_index == 0 else "",
            )
            axis_bands.plot(
                distances,
                qe_shifted[:, band_index],
                color="#dc2626",
                lw=1.0,
                alpha=0.55,
                ls="--",
                label="QE (LibXC)" if band_index == 0 else "",
            )

        axis_bands.axhline(0, color="#16a34a", lw=1.2, ls=":", alpha=0.8, label="$E_F$")
        for position in high_symmetry_positions:
            axis_bands.axvline(position, color="#94a3b8", lw=0.5)
        axis_bands.set_xticks(high_symmetry_positions)
        axis_bands.set_xticklabels(DISPLAY_LABELS, fontsize=12)
        axis_bands.set_ylabel("$E - E_F$ (eV)", fontsize=13)
        axis_bands.set_title(f"{label}: Q3 vs QE Band Structure", fontsize=14, fontweight="bold")
        axis_bands.legend(fontsize=9, loc="upper right")
        axis_bands.set_ylim(-20, 20)
        axis_bands.set_xlim(distances[0], distances[-1])

        delta_mev = (
            q3_bands[:, :number_of_common_bands] - qe_bands[:, :number_of_common_bands]
        ) * 1000
        max_error = np.max(np.abs(delta_mev))
        mean_error = np.mean(np.abs(delta_mev))
        axis_bands.text(
            0.03,
            0.03,
            f"max|\u0394E| = {max_error:.0f} meV\nmean|\u0394E| = {mean_error:.1f} meV",
            transform=axis_bands.transAxes,
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="#fef3c7", ec="#d97706", alpha=0.9),
        )

        # DOS panel
        if label in q3_dos_data:
            dos_energies = np.array(q3_dos_data[label]["energies_ry"]) * RY_TO_EV - q3_fermi
            dos_values = np.array(q3_dos_data[label]["dos"]) / RY_TO_EV
            axis_dos.plot(dos_values, dos_energies, color="#2563eb", lw=1.2, alpha=0.85, label="Q3")

        if qe_scf_eigenvalues is not None:
            qe_weights = qe_data[label].get("scf_weights", None)
            qe_dos_energies, qe_dos_values = compute_dos(
                qe_scf_eigenvalues - qe_fermi_shifted, weights=qe_weights, sigma_ev=broadening_ev
            )
            axis_dos.plot(
                qe_dos_values,
                qe_dos_energies,
                color="#dc2626",
                lw=1.0,
                alpha=0.55,
                ls="--",
                label="QE",
            )

        axis_dos.axhline(0, color="#16a34a", lw=1.2, ls=":", alpha=0.8)
        axis_dos.set_xlabel("DOS (states/eV)", fontsize=11)
        axis_dos.set_title("DOS", fontsize=14, fontweight="bold")
        axis_dos.legend(fontsize=8, loc="upper right")
        plt.setp(axis_dos.get_yticklabels(), visible=False)
        axis_dos.set_xlim(left=0)

        plt.tight_layout()
        filename = os.path.join(
            figures_directory, f"{label.lower().replace(' ', '_')}_band_structure.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}  (max={max_error:.0f} meV, mean={mean_error:.1f} meV)")

    # --- Step 4: Combined plots (NC+US side by side) ---
    print("\nGenerating combined plots...")
    for material_tag, material_cases in [("si", ["Si NC", "Si US"]), ("cu", ["Cu NC", "Cu US"])]:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        for index, label in enumerate(material_cases):
            axis = axes[index]
            if not os.path.exists(q3_band_results[label]) or label not in qe_data:
                axis.set_title(f"{label}: data missing")
                continue

            eigenvalues = eigenvalue_arrays(
                Results.read(q3_band_results[label]).properties.eigenvalues_at_kpoints
            )[0]
            q3_bands = np.array([kpoint[0] for kpoint in eigenvalues]) * RY_TO_EV
            qe_bands = qe_data[label]["bands"]
            qe_fermi = qe_data[label]["fermi"]
            number_of_common_bands = min(q3_bands.shape[1], qe_bands.shape[1])
            material = "Si" if "Si" in label else "Cu"

            if material == "Si":
                q3_fermi = compute_fermi_silicon(q3_bands)
                qe_fermi_shifted = qe_fermi if qe_fermi else compute_fermi_silicon(qe_bands)
            else:
                qe_fermi_shifted = qe_fermi if qe_fermi else 0
                q3_fermi = qe_fermi_shifted

            q3_shifted = q3_bands - q3_fermi
            qe_shifted = qe_bands - qe_fermi_shifted

            for band_index in range(number_of_common_bands):
                axis.plot(
                    distances,
                    q3_shifted[:, band_index],
                    color="#2563eb",
                    lw=1.5,
                    alpha=0.85,
                    label="Q3" if band_index == 0 else "",
                )
                axis.plot(
                    distances,
                    qe_shifted[:, band_index],
                    color="#dc2626",
                    lw=1.0,
                    alpha=0.55,
                    ls="--",
                    label="QE (LibXC)" if band_index == 0 else "",
                )

            axis.axhline(0, color="#16a34a", lw=1.2, ls=":", alpha=0.8, label="$E_F$")
            for position in high_symmetry_positions:
                axis.axvline(position, color="#94a3b8", lw=0.5)
            axis.set_xticks(high_symmetry_positions)
            axis.set_xticklabels(DISPLAY_LABELS, fontsize=12)
            axis.set_title(f"{label}: Q3 vs QE Band Structure", fontsize=14, fontweight="bold")
            axis.legend(fontsize=9, loc="upper right")
            axis.set_ylim(-20, 20)
            axis.set_xlim(distances[0], distances[-1])

            delta_mev = (
                q3_bands[:, :number_of_common_bands] - qe_bands[:, :number_of_common_bands]
            ) * 1000
            max_error = np.max(np.abs(delta_mev))
            mean_error = np.mean(np.abs(delta_mev))
            axis.text(
                0.03,
                0.03,
                f"max|\u0394E| = {max_error:.0f} meV\nmean|\u0394E| = {mean_error:.1f} meV",
                transform=axis.transAxes,
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="#fef3c7", ec="#d97706", alpha=0.9),
            )

        axes[0].set_ylabel("$E - E_F$ (eV)", fontsize=13)
        plt.tight_layout()
        filename = os.path.join(figures_directory, f"{material_tag}_band_structure.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")

    print("\nDone!")


if __name__ == "__main__":
    main()
