"""q3.workflows.interpolation — band structures and densities of states from an
interpolation model.

The chain is three engine steps and one utility step:

    q3 scf  ->  q3 nscf (full grid; builds and saves the model)  ->  q3_utility interpolate
                                                                            |
                                                          bands along a path, DOS on a mesh

The NSCF run carries a ``control.interpolation`` block and leaves
``<outdir>/<prefix>.save/interpolation.q3ip`` behind -- the model. ``q3_utility
interpolate`` reads it and evaluates it at whatever k-points an evaluation
manifest asks for, which is cheap: evaluating a saved model at 10^5 k-points
costs seconds next to the NSCF that built it.

``method`` names how the model is built. ``wannier`` (the only method today)
uses Wannier functions in the one-shot projection gauge, so it needs the
Wannier-specific keys ``projection_centers`` and ``projection_decay_bohr``; a
future method may need none of them and the block is validated by the method.

The k-mesh the model is built on is the one number that decides its accuracy
between mesh points -- on silicon the automatic recipe moves from 826 to 12.6
meV rms between 4x4x4 and 8x8x8 while the gauge moves it 1.4x at most -- so
the workflow chooses it when no ``grid`` is given: :func:`mesh_for_spacing`
takes n_i = ceil(|b_i| / spacing) per reciprocal vector, rounded up to even and
never below four, at the published default spacing of 0.2 per angstrom
(Vitale et al., npj Comput. Mater. 6, 66 (2020)).

Lower-level helpers are exported beside the workflow, because the validation
suite and the example need the steps separately:

- :func:`write_interpolation_nscf_input` -- the NSCF manifest that builds the
  model (full grid + the block) from an SCF manifest;
- :func:`write_interpolation_evaluation_input` and :func:`evaluate_interpolation`
  -- the evaluation manifest and the ``q3_utility interpolate`` run;
- :func:`import_wannier90_record` -- another code's Wannier90 files into a
  record, through ``q3_utility convert --from wannier90``;
- :func:`mesh_for_spacing` -- the mesh rule, over the lattice of the manifest's
  material (:func:`q3.io.input_reader.load_material`).

Example::

    from q3.workflows import run_interpolation_workflow

    result = run_interpolation_workflow(
        "examples/band_structure--semiconductor--silicon--wannier/nc_scf.yml",
        grid=(4, 4, 4),  # or omit it: 0.2 per angstrom gives silicon 10x10x10
        projection_centers=[(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)],
        band_path=["G", "X", "L", "G"],
        dos_mesh=(24, 24, 24),
    )
    result["band_structure"]["eigenvalues_ry"]  # [k-point][spin][band]
    result["density_of_states"]["dos"]  # states / Ry / cell, spin included
"""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path
from typing import Self

import yaml

from q3.binary import find_binary, run_q3, run_q3_utility
from q3.io.input_reader import (
    load_material,
    make_file_references_absolute,
    pseudopotential_paths,
)
from q3.io.input_writer import write_nscf_input
from q3.io.interpolation_results import InterpolationResults
from q3.io.interpolation_settings import (
    CaptureSettings,
    EvaluationSettings,
    Grid,
    validated_grid,
)
from q3.io.output_writer import extract_band_structure
from q3.structure.lattice import reciprocal_lattice
from q3.workflows.kpath import (
    DEFAULT_POINTS_PER_SEGMENT,
    interpolate_kpath,
    kpoints_list_with_uniform_weight,
)

KPoint = tuple[float, float, float]

#: The published k-point spacing of the automated protocol, in inverse angstrom
#: with the 2 pi included, as reciprocal vectors are measured here.
DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM = 0.2

#: The Wannier centres are finite-difference Berry phases, which need k+b and
#: k-b to be distinct points: two points per axis is refused by the localiser,
#: three is odd, so four is the least the rule ever produces.
MINIMUM_MESH_POINTS_PER_AXIS = 4


def mesh_for_spacing(
    lattice: dict,
    spacing_inverse_angstrom: float = DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM,
) -> Grid:
    """The k-mesh a reciprocal-space spacing implies, for an ESSE ``lattice`` node.

    ``n_i = ceil(|b_i| / spacing)`` is made's rule
    (``ReciprocalLattice.get_dimensions_from_spacing``), applied to the cell Q3
    builds. Two constraints of this protocol are applied on top of it: each
    count is rounded UP to even, so a coarser retry can halve it and stay a
    proper grid, and is never below four -- the Wannier centres are
    finite-difference Berry phases that need k+b and k-b to be distinct points,
    which two points per axis cannot give, and three is odd.

    Silicon at the default 0.2 per angstrom gets 10x10x10; its 4x4x4, 6x6x6 and
    8x8x8 examples are the spacings 0.5, 0.34 and 0.25.
    """
    if not spacing_inverse_angstrom > 0.0:
        raise ValueError(f"the k-point spacing must be positive, got {spacing_inverse_angstrom!r}")
    # made measures reciprocal vectors without the 2 pi; the spacing this
    # protocol publishes carries it, as does `reciprocal_lattice_vectors_...`.
    counts = reciprocal_lattice(lattice).get_dimensions_from_spacing(
        spacing_inverse_angstrom / (2.0 * math.pi)
    )
    grid = []
    for value in counts:
        count = int(value)
        if count % 2:
            count += 1
        grid.append(max(count, MINIMUM_MESH_POINTS_PER_AXIS))
    return (grid[0], grid[1], grid[2])


def full_grid_kpoints(grid: Grid) -> list[tuple[float, float, float, float]]:
    """Every point of a Gamma-centred grid with uniform weight, as ``kpoints_list`` rows.

    The full, unreduced grid on purpose: the Wannier construction needs each
    k-point's neighbours to exist, which a symmetry-reduced set cannot offer.
    Enumeration order is the first index slowest, matching the engine's own.

    The weights sum to **2**, not 1, because that is the convention
    ``KPointsGrid::generate`` uses: it embeds the spin degeneracy in the weight
    rather than applying it later. An explicit ``kpoints_list`` is taken as
    given, so weights summing to 1 tell the engine it has half the electrons it
    has, and the Fermi bisection then lands far too high. Measured on copper
    before this was fixed: 58.7 eV against a true 16.9 eV, above the top of a
    fourteen-band window, which froze every band and made disentanglement
    infeasible. The eigenvalues never depended on this; the Fermi level and
    everything downstream of it did.

    Summing to 2 is also right for a spin-polarized run, which is the next thing
    a reader wonders. The engine halves every weight itself when there are two
    spin components -- an explicit list included, not only a generated grid --
    because LSDA enumerates the two channels as separate k-point states. So the
    spin factor is applied here exactly once either way, and this function does
    not need to know which case it is in.
    """
    n1, n2, n3 = validated_grid(grid)
    weight = 2.0 / (n1 * n2 * n3)
    return [
        (x / n1, y / n2, z / n3, weight) for x in range(n1) for y in range(n2) for z in range(n3)
    ]


def write_interpolation_nscf_input(
    output_path: str | Path,
    *,
    scf_input_path: str | Path,
    settings: CaptureSettings,
    number_of_bands: int | None = None,
) -> Path:
    """Write the NSCF manifest that builds the model: the full grid plus the block.

    Everything else is copied from *scf_input_path* by ``write_nscf_input``
    (same prefix and outdir, so the run finds the SCF checkpoint and the
    record lands beside it). *settings* is the block as the engine reads it
    (:class:`~q3.io.interpolation_settings.CaptureSettings`): only the keys it
    sets are written, and the engine's defaults apply to the rest.
    """
    path = write_nscf_input(
        output_path,
        scf_input_path=scf_input_path,
        kpoints_list=full_grid_kpoints(settings.grid),
        number_of_bands=number_of_bands,
        uniform_grid=True,
    )
    config = yaml.safe_load(path.read_text())
    config["control"]["interpolation"] = settings.to_block()
    with open(path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def write_interpolation_evaluation_input(
    output_path: str | Path,
    *,
    template_input_path: str | Path,
    outdir: str | Path,
    prefix: str,
    settings: EvaluationSettings | None = None,
    kpoints: list[KPoint] | None = None,
    dos_broadening_ry: float | None = None,
) -> Path:
    """Write the manifest ``q3_utility interpolate`` evaluates.

    ``material``, ``model`` and ``method`` come from *template_input_path* (an
    SCF or NSCF manifest of the same calculation) with their file references
    made absolute; ``control`` is replaced by ``calculation: interpolation``,
    the *outdir*/*prefix* that name the run holding the record, the *kpoints*
    to evaluate, and *settings* -- the block as the engine reads it
    (:class:`~q3.io.interpolation_settings.EvaluationSettings`), of which only
    the keys it sets are written. Disentanglement and its thresholds belong
    there rather than on the capture step: the record stores the raw window,
    so the same capture serves a run with disentanglement and a run without.
    """
    output_path = Path(output_path).expanduser().resolve()
    template_input_path = Path(template_input_path).expanduser().resolve()
    with open(template_input_path) as handle:
        config = copy.deepcopy(yaml.safe_load(handle))
    make_file_references_absolute(config, template_input_path.parent)
    config.get("method", {}).get("scf", {}).pop("kgrid", None)

    settings = settings if settings is not None else EvaluationSettings()
    control: dict = {
        "calculation": "interpolation",
        "outdir": str(Path(outdir).expanduser().resolve()),
        "prefix": prefix,
        "interpolation": settings.to_block(),
    }
    if kpoints:
        control["kpoints_list"] = kpoints_list_with_uniform_weight(kpoints)
    if dos_broadening_ry is not None:
        control["dos_broadening"] = float(dos_broadening_ry)
    config["control"] = control

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output_path


def evaluate_interpolation(
    evaluation_input_path: str | Path,
    *,
    q3_utility_binary: str | Path | None = None,
    capture_output: bool = True,
) -> dict:
    """Run ``q3_utility interpolate`` on an evaluation manifest and return its results file.

    The file is ``<outdir>/<prefix>_interpolation_results.json`` next to the
    record, with ``eigenvalues_ry`` shaped ``[k-point][spin][band]`` like the
    engine's own results file. :func:`evaluated_interpolation_results` is the
    same run with the file typed.
    """
    return evaluated_interpolation_results(
        evaluation_input_path, q3_utility_binary=q3_utility_binary, capture_output=capture_output
    ).to_document()


def evaluated_interpolation_results(
    evaluation_input_path: str | Path,
    *,
    q3_utility_binary: str | Path | None = None,
    capture_output: bool = True,
) -> InterpolationResults:
    """Run ``q3_utility interpolate`` on an evaluation manifest and read what it wrote."""
    evaluation_input_path = Path(evaluation_input_path).expanduser().resolve()
    # q3_utility, like q3, takes a relative control.outdir against its working
    # directory. Running it from the manifest's directory makes that the
    # manifest's own directory, which is where this function then looks too.
    run_q3_utility(
        ["interpolate", str(evaluation_input_path)],
        q3_utility_binary=q3_utility_binary,
        capture_output=capture_output,
        cwd=evaluation_input_path.parent,
    )
    with open(evaluation_input_path) as handle:
        control = yaml.safe_load(handle).get("control", {})
    outdir = Path(control.get("outdir", "."))
    if not outdir.is_absolute():
        outdir = evaluation_input_path.parent / outdir
    results_path = outdir / interpolation_results_file_name(control.get("prefix", "q3"))
    if not results_path.exists():
        raise FileNotFoundError(f"q3_utility interpolate wrote no results at {results_path}")
    return InterpolationResults.read(results_path)


def interpolation_results_file_name(prefix: str) -> str:
    """What ``q3_utility interpolate`` names its results file, under the run's outdir."""
    return f"{prefix}_interpolation_results.json"


def fermi_energy_ry(*results_by_precedence: dict | None) -> float | None:
    """The Fermi level of the first of *results_by_precedence* that was computed; None with none.

    Which run's Fermi level is the right one depends on what it is for, so the
    caller states the order: the SCF's is the authoritative one to report
    bands against, and the capture's is the one the classifier measured the
    frozen window from.
    """
    for results in results_by_precedence:
        if results is not None:
            return float(results.get("fermi_energy_ry", 0.0))
    return None


class EngineChain:
    """One SCF manifest, and the steps that run the engine against it.

    Rewrites the manifest once with absolute paths, so every later step can run
    from a temporary manifest beside the original -- the engine resolves a
    manifest's relative paths against the manifest, so a copy anywhere else
    would lose the material and the pseudopotentials. Each step writes its own
    manifest rather than editing one in place, because the engine refuses keys
    that belong to another step by name.
    """

    def __init__(self, scf_input_path: Path, *, q3_binary, q3_utility_binary, capture_output: bool):
        self.scf_input_path = scf_input_path
        with open(scf_input_path) as handle:
            self.config = yaml.safe_load(handle)
        self.material = load_material(self.config, scf_input_path.parent)
        self.binary = find_binary(q3_binary)
        self.q3_utility_binary = q3_utility_binary
        self.capture_output = capture_output

        control = self.config.setdefault("control", {})
        control["calculation"] = "self_consistent_field"
        outdir = control.get("outdir", "./output")
        if not Path(outdir).is_absolute():
            outdir = str((scf_input_path.parent / outdir).resolve())
        control["outdir"] = outdir
        self.outdir = Path(outdir)
        self.prefix = control.get("prefix", "q3")
        make_file_references_absolute(self.config, scf_input_path.parent)
        self.effective_scf_path = self._temporary("_scf.yml")
        with open(self.effective_scf_path, "w") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)

    def close(self) -> None:
        """Remove the rewritten SCF manifest; the run directory itself is the caller's."""
        self.effective_scf_path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def _temporary(self, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, dir=self.scf_input_path.parent, delete=False
        ) as handle:
            return Path(handle.name)

    def _run(self, suffix: str, write, run) -> dict:
        """Write one step's manifest, run it, and always remove it."""
        path = self._temporary(suffix)
        try:
            write(path)
            return run(path)
        finally:
            path.unlink(missing_ok=True)

    def run_scf(self) -> dict:
        return run_q3(
            self.effective_scf_path, q3_binary=self.binary, capture_output=self.capture_output
        )

    def capture(self, settings: CaptureSettings, number_of_bands: int | None = None) -> dict:
        """The full-grid NSCF that builds the model and saves the record.

        The expensive step, and the only one that touches plane waves: every
        evaluation afterwards reads the record back, in seconds.
        """
        return self._run(
            "_capture.yml",
            lambda path: write_interpolation_nscf_input(
                path,
                scf_input_path=self.effective_scf_path,
                settings=settings,
                number_of_bands=number_of_bands,
            ),
            lambda path: run_q3(path, q3_binary=self.binary, capture_output=self.capture_output),
        )

    def reference(self, kpoints: list[KPoint], number_of_bands: int) -> dict:
        """A direct NSCF at *kpoints* -- the bands a model is judged against."""
        return self._run(
            "_reference.yml",
            lambda path: write_nscf_input(
                path,
                scf_input_path=self.effective_scf_path,
                kpoints_list=kpoints_list_with_uniform_weight(kpoints),
                number_of_bands=number_of_bands,
            ),
            lambda path: run_q3(path, q3_binary=self.binary, capture_output=self.capture_output),
        )

    def evaluate(
        self,
        kpoints: list[KPoint] | None,
        settings: EvaluationSettings | None = None,
        dos_broadening_ry: float | None = None,
    ) -> InterpolationResults:
        """Read the saved record back at *kpoints* under *settings*, with no DFT."""
        return self._run(
            "_evaluate.yml",
            lambda path: write_interpolation_evaluation_input(
                path,
                template_input_path=self.effective_scf_path,
                outdir=self.outdir,
                prefix=self.prefix,
                settings=settings,
                kpoints=kpoints,
                dos_broadening_ry=dos_broadening_ry,
            ),
            lambda path: evaluated_interpolation_results(
                path, q3_utility_binary=self.q3_utility_binary, capture_output=self.capture_output
            ),
        )

    def pseudopotential_paths(self) -> dict[str, Path]:
        """Each species of the material and the dataset the manifest names for it."""
        return pseudopotential_paths(self.config, self.material, self.scf_input_path.parent)

    def results_path(self) -> Path:
        """Where `evaluate` leaves its results."""
        return self.outdir / interpolation_results_file_name(self.prefix)

    def mesh(self, grid: Grid | None, spacing: float) -> Grid:
        """The full grid a capture runs on: *grid* as given, validated, or the rule's for *spacing*."""
        if grid is not None:
            return validated_grid(grid)
        return mesh_for_spacing(self.material["lattice"], spacing)

    def lattice_type(self, lattice_type: str | None = None) -> str:
        """The lattice type a path of labels is read with: as given, or the material's own.

        A material that gives its lattice as vectors without a type names no
        high-symmetry table, and the k-points must then be given explicitly.
        """
        resolved = (
            lattice_type if lattice_type is not None else self.material["lattice"].get("type")
        )
        if resolved is None:
            raise ValueError(
                "lattice_type was not passed and could not be inferred: the manifest's "
                "material gives its lattice as vectors without a type -- pass lattice_type, "
                "or pass the k-points explicitly."
            )
        return str(resolved)

    def kpath(
        self, band_path: list[str], lattice_type: str | None, points_per_segment: int
    ) -> tuple[list[KPoint], list[float]]:
        """The k-points of a path of high-symmetry labels, and the distances along it."""
        return interpolate_kpath(
            band_path,
            self.lattice_type(lattice_type),
            points_per_segment=points_per_segment,
            lattice=self.material["lattice"],
        )


def import_wannier90_record(
    seedname: str | Path,
    grid: Grid,
    *,
    outdir: str | Path = ".",
    prefix: str | None = None,
    q3_utility_binary: str | Path | None = None,
    capture_output: bool = True,
) -> Path:
    """Import ``<seedname>.amn/.eig/.nnkp`` into ``<outdir>/<prefix>.save/interpolation.q3ip``.

    Another code's matrices then go through exactly the evaluation path Q3's
    own do, which is how the engine is checked against Quantum ESPRESSO.
    ``grid`` is required because a ``.nnkp`` lists k-points without saying what
    grid they came from. Returns the record's path.
    """
    grid = validated_grid(grid)
    seedname = Path(seedname)
    prefix_name = prefix or seedname.name
    outdir = Path(outdir).expanduser().resolve()
    arguments = [
        "convert",
        "--from",
        "wannier90",
        "--seedname",
        str(seedname),
        "--grid",
        *(str(value) for value in grid),
        "--outdir",
        str(outdir),
        "--prefix",
        prefix_name,
    ]
    run_q3_utility(arguments, q3_utility_binary=q3_utility_binary, capture_output=capture_output)
    return outdir / f"{prefix_name}.save" / "interpolation.q3ip"


def run_interpolation_workflow(
    scf_input_path: str | Path,
    *,
    grid: Grid | None = None,
    kpoint_spacing_inverse_angstrom: float = DEFAULT_KPOINT_SPACING_INVERSE_ANGSTROM,
    method: str = "wannier",
    projection_centers: list[KPoint] | None = None,
    projection_decay_bohr: float | None = None,
    export_formats: list[str] | tuple[str, ...] = (),
    band_path: list[str] | None = None,
    lattice_type: str | None = None,
    points_per_segment: int | None = None,
    kpoints: list[KPoint] | None = None,
    dos_mesh: Grid | None = None,
    dos_broadening_ry: float | None = None,
    disentanglement: str | None = None,
    projectability_minimum: float | None = None,
    projectability_maximum: float | None = None,
    frozen_window_above_fermi_ev: float | None = None,
    number_of_bands: int | None = None,
    q3_binary: str | Path | None = None,
    q3_utility_binary: str | Path | None = None,
    skip_scf: bool = False,
    capture_output: bool = True,
) -> dict:
    """SCF, full-grid NSCF that builds the model, then ``q3_utility interpolate``.

    Parameters
    ----------
    scf_input_path:
        A Q3 SCF manifest. Its prefix and outdir are reused for every step so
        the checkpoint and the record are found. It may carry
        ``control.band_path`` and ``control.band_path_points_per_segment``,
        which are the defaults for *band_path* and *points_per_segment*.
    grid, kpoint_spacing_inverse_angstrom:
        The full k-point grid the model is built on, e.g. ``(4, 4, 4)``. Every
        point of it is diagonalized in the NSCF; the interpolation error falls
        quickly with its density (silicon valence: ~68 meV rms at 4x4x4,
        ~10 meV at 8x8x8). ``None`` chooses it from the material by
        :func:`mesh_for_spacing` at *kpoint_spacing_inverse_angstrom*, whose
        default of 0.2 gives silicon 10x10x10; the choice is returned as
        ``grid``.
    method, projection_centers, projection_decay_bohr, export_formats:
        The ``control.interpolation`` block; see ``docs/io_reference.md``.
    band_path, lattice_type, points_per_segment:
        High-symmetry labels resolved through ``q3.workflows.kpath`` (the
        lattice type defaults to the material's own), or pass *kpoints*
        directly. Either, both or neither: with neither and no *dos_mesh*
        there is nothing to evaluate.
    dos_mesh, dos_broadening_ry:
        Dense mesh and Gaussian width for the interpolated density of states.
    disentanglement, projectability_minimum, projectability_maximum,
    frozen_window_above_fermi_ev:
        Pass ``disentanglement="projectability"`` when the window holds more
        bands than trial orbitals, as an entangled metal does. Read at the
        evaluation step, so the same NSCF capture serves both answers.
    number_of_bands:
        NSCF band count; ``None`` keeps the SCF value. With the ``wannier``
        method it must equal the number of trial orbitals, unless
        *disentanglement* is asked for -- which is what makes a wider window
        usable at all.
    skip_scf:
        Reuse an existing SCF checkpoint in the outdir.

    Returns
    -------
    dict
        ``grid`` (the mesh actually captured, chosen or given),
        ``scf_results`` (``None`` if skipped), ``nscf_results``,
        ``interpolation_results`` (the raw ``q3_utility`` file),
        ``band_structure`` (``extract_band_structure`` shape, when k-points
        were evaluated) and ``density_of_states`` (``energies_ry``, ``dos`` in
        states per Ry per cell **with** the spin factor of two, matching
        ``run_dos_workflow``, and ``fermi_energy_ry``; when a mesh was given).
    """
    scf_input_path = Path(scf_input_path).expanduser().resolve()
    with EngineChain(
        scf_input_path,
        q3_binary=q3_binary,
        q3_utility_binary=q3_utility_binary,
        capture_output=capture_output,
    ) as chain:
        grid = chain.mesh(grid, kpoint_spacing_inverse_angstrom)

        input_control = chain.config.get("control", {})
        if band_path is None and kpoints is None:
            band_path = input_control.get("band_path")
        if points_per_segment is None:
            points_per_segment = input_control.get(
                "band_path_points_per_segment", DEFAULT_POINTS_PER_SEGMENT
            )
        if band_path is not None and kpoints is None:
            # Refused here rather than after the SCF: a path needs a table to read its labels with.
            lattice_type = chain.lattice_type(lattice_type)

        scf_results = None if skip_scf else chain.run_scf()
        nscf_results = chain.capture(
            CaptureSettings(
                grid=list(grid),
                method=method,
                projection_centers=projection_centers,
                projection_decay_bohr=projection_decay_bohr,
                export_formats=list(export_formats),
            ),
            number_of_bands=number_of_bands,
        )

        kpath_distances: list[float] | None = None
        if kpoints is None and band_path is not None:
            kpoints, kpath_distances = chain.kpath(band_path, lattice_type, points_per_segment)

        interpolation_results = chain.evaluate(
            kpoints,
            EvaluationSettings(
                method=method,
                dos_mesh=dos_mesh,
                disentanglement=disentanglement,
                projectability_minimum=projectability_minimum,
                projectability_maximum=projectability_maximum,
                frozen_window_above_fermi_ev=frozen_window_above_fermi_ev,
            ),
            dos_broadening_ry=dos_broadening_ry,
        ).to_document()

    # The SCF Fermi energy is the authoritative one: an NSCF on a k-set that is
    # not a proper Monkhorst-Pack grid, or with every requested band occupied,
    # reports a Fermi level that is not physically meaningful.
    fermi_level_ry = fermi_energy_ry(scf_results, nscf_results)

    result: dict = {
        "grid": list(grid),
        "scf_results": scf_results,
        "nscf_results": nscf_results,
        "interpolation_results": interpolation_results,
    }
    if kpoints:
        if kpath_distances is None:
            kpath_distances = list(range(len(kpoints)))
        result["band_structure"] = extract_band_structure(
            interpolation_results,
            kpoints=list(kpoints),
            kpath_distances=kpath_distances,
            fermi_energy_ry=fermi_level_ry,
        )
    density = interpolation_results.get("density_of_states")
    if density is not None:
        result["density_of_states"] = {
            "energies_ry": density["energies_ry"],
            # The utility reports states per band per k-point; the spin factor of
            # two makes it states per cell for an unpolarized system, the
            # convention run_dos_workflow uses.
            "dos": [2.0 * value for value in density["dos"]],
            "fermi_energy_ry": fermi_level_ry,
        }
    return result
