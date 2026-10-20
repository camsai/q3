#!/usr/bin/env python3
"""Generate a Quantum ESPRESSO pw.x SCF input from a Q3 material JSON.

Most decks under `examples/` already ship a hand-written QE counterpart beside
them, and those stay authoritative. This exists for the cases that do not: the
16-atom silicon supercell had none, and transcribing sixteen fractional
coordinates and a lattice constant by hand is a good way to introduce a
difference nobody notices until the energies disagree for the wrong reason.

The trust basis is `--selftest`, which regenerates a committed input from the
material JSON it was written from and diffs the two. It reproduces
`examples/total_energy--semiconductor--silicon/reference/nc/scf_qe_3x3x3.in`
byte for byte, all 54 positions and the lattice constant included, which is what
makes the generated inputs worth committing beside the hand-written ones.

    scripts/validation/generate_qe_input_from_material.py --selftest

    scripts/validation/generate_qe_input_from_material.py \
        --material examples/data/materials/silicon_2x2x2.json \
        --prefix qe_nc_2x2x2_k444 --kgrid 4 4 4 --kshift 1 1 1 --nbnd 40 \
        -o examples/total_energy--semiconductor--silicon/reference/nc/scf_qe_2x2x2_k444.in

It handles only what these decks need -- ibrav=2 (FCC) stored in the primitive
rhombohedral setting, crystal-fractional coordinates, one species -- and exits
on anything else rather than emitting a plausible-looking wrong answer.
"""

import argparse
import difflib
import json
import math
import pathlib
import sys

# CODATA bohr radius in angstrom, matching what the committed inputs were made with.
BOHR_IN_ANGSTROM = 0.529177210903


def celldm1_bohr(lattice):
    """Cubic FCC lattice constant in bohr, from the primitive vector length.

    The q3 material JSON stores FCC as the primitive rhombohedral setting
    (alpha=beta=gamma=60 deg), so `a` is the primitive vector length. QE's
    ibrav=2 wants the CUBIC constant, which is a_primitive * sqrt(2).
    """
    if lattice.get("type") != "FCC":
        raise SystemExit(f"unsupported lattice type {lattice.get('type')!r}; only FCC/ibrav=2")
    for ang in ("alpha", "beta", "gamma"):
        if abs(float(lattice[ang]) - 60.0) > 1e-6:
            raise SystemExit(f"expected primitive FCC with {ang}=60, got {lattice[ang]}")
    units = lattice.get("units", {}).get("length", "angstrom")
    if units != "angstrom":
        raise SystemExit(f"unsupported lattice length unit {units!r}")
    return float(lattice["a"]) * math.sqrt(2.0) / BOHR_IN_ANGSTROM


def species_and_positions(basis):
    if basis.get("units") != "crystal":
        raise SystemExit(f"expected crystal (fractional) coordinates, got {basis.get('units')!r}")
    elements = {e["id"]: e["value"] for e in basis["elements"]}
    coords = {c["id"]: c["value"] for c in basis["coordinates"]}
    if set(elements) != set(coords):
        raise SystemExit("elements and coordinates have mismatched ids")
    rows = [(elements[i], coords[i]) for i in sorted(elements)]
    unique = sorted({sym for sym, _ in rows})
    if len(unique) != 1:
        raise SystemExit(f"only single-species cells supported, found {unique}")
    return unique, rows


# Atomic masses for species these decks use.
MASSES = {"Si": 28.0855, "C": 12.0107, "Cu": 63.546}


def build(
    material,
    prefix,
    kgrid,
    kshift,
    nbnd,
    pseudo_dir,
    pseudo,
    ecutwfc,
    ecutrho,
    conv_thr,
    mixing_beta,
    mixing_mode,
    input_dft,
    outdir,
):
    lattice = material["lattice"]
    unique, rows = species_and_positions(material["basis"])
    sym = unique[0]
    mass = MASSES.get(sym)
    if mass is None:
        raise SystemExit(f"no mass on record for {sym}; add it to MASSES")

    L = []
    L.append("&CONTROL")
    L.append("  calculation = 'scf'")
    L.append(f"  pseudo_dir = '{pseudo_dir}'")
    L.append(f"  outdir = '{outdir}'")
    L.append(f"  prefix = '{prefix}'")
    L.append("  tstress = .true.")
    L.append("  tprnfor = .true.")
    L.append("/")
    L.append("&SYSTEM")
    L.append("  ibrav = 2")
    L.append(f"  celldm(1) = {celldm1_bohr(lattice):.14f}".rstrip("0").rstrip("."))
    L.append(f"  nat = {len(rows)}")
    L.append(f"  ntyp = {len(unique)}")
    L.append(f"  ecutwfc = {ecutwfc}")
    L.append(f"  ecutrho = {ecutrho}")
    L.append(f"  input_dft = '{input_dft}'")
    if nbnd is not None:
        L.append(f"  nbnd = {nbnd}")
    L.append("/")
    L.append("&ELECTRONS")
    L.append(f"  conv_thr = {conv_thr}")
    L.append(f"  mixing_beta = {mixing_beta}")
    L.append(f"  mixing_mode = '{mixing_mode}'")
    L.append("  diagonalization = 'david'")
    L.append("/")
    L.append("ATOMIC_SPECIES")
    L.append(f"  {sym} {mass} {pseudo}")
    L.append("ATOMIC_POSITIONS crystal")
    for s, (x, y, z) in rows:
        L.append(f"  {s}  {x:.12f}  {y:.12f}  {z:.12f}")
    L.append("K_POINTS automatic")
    L.append("  " + " ".join(str(v) for v in (*kgrid, *kshift)))
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--material", help="Q3 material JSON (defaulted under --selftest)")
    p.add_argument("--prefix", default="qe_gen")
    p.add_argument("--kgrid", nargs=3, type=int, default=[1, 1, 1])
    p.add_argument("--kshift", nargs=3, type=int, default=[0, 0, 0])
    p.add_argument("--nbnd", type=int)
    p.add_argument("--pseudo-dir", default="../../../data/pseudopotentials/nc/")
    p.add_argument("--pseudo", default="Si.upf")
    p.add_argument("--ecutwfc", default="48.0")
    p.add_argument("--ecutrho", default="192.0")
    p.add_argument("--conv-thr", default="1.0e-8")
    p.add_argument("--mixing-beta", default="0.3")
    p.add_argument("--mixing-mode", default="plain")
    p.add_argument("--input-dft", default="XC-000I-000I-101L-130L")
    p.add_argument("--outdir", default="./tmp")
    p.add_argument("-o", "--output")
    p.add_argument(
        "--selftest", action="store_true", help="regenerate the committed 54-atom input and diff it"
    )
    p.add_argument("--reference", help="committed input to compare against under --selftest")
    a = p.parse_args()

    if a.selftest:
        root = pathlib.Path(__file__).resolve().parents[2]
        if not a.material:
            a.material = root / "examples/data/materials/silicon_3x3x3.json"
        if not a.reference:
            a.reference = (
                root / "examples/total_energy--semiconductor--silicon/reference/nc/scf_qe_3x3x3.in"
            )

    material = json.load(open(a.material))

    if a.selftest:
        # Match the committed 54-atom file's settings exactly.
        got = build(
            material,
            "qe_nc_3x3x3",
            [1, 1, 1],
            [0, 0, 0],
            130,
            a.pseudo_dir,
            a.pseudo,
            a.ecutwfc,
            a.ecutrho,
            a.conv_thr,
            a.mixing_beta,
            a.mixing_mode,
            a.input_dft,
            a.outdir,
        )
        want = open(a.reference).read()
        if got == want:
            print("SELFTEST PASS: regenerated the committed input byte-for-byte")
            return 0
        print("SELFTEST FAIL: differences vs committed input")
        for line in difflib.unified_diff(
            want.splitlines(), got.splitlines(), "committed", "generated", lineterm="", n=1
        ):
            print("  " + line)
        return 1

    out = build(
        material,
        a.prefix,
        a.kgrid,
        a.kshift,
        a.nbnd,
        a.pseudo_dir,
        a.pseudo,
        a.ecutwfc,
        a.ecutrho,
        a.conv_thr,
        a.mixing_beta,
        a.mixing_mode,
        a.input_dft,
        a.outdir,
    )
    if a.output:
        open(a.output, "w").write(out)
        print(f"wrote {a.output}")
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
