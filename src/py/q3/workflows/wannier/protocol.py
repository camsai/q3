"""q3.workflows.wannier.protocol — :class:`WannierProtocol`, the automated protocol end to end.

SCF, capture on the chosen mesh, a reference calculation, and the retry ladder
(:mod:`q3.workflows.wannier.rungs`) walked once per mesh; then the mesh rung,
the one step that recaptures; and the verdict written into the results file
beside what the model says of itself.
"""

from __future__ import annotations

from typing import Literal

from q3.constants import RYDBERG_TO_ELECTRONVOLT
from q3.io.interpolation.results.classes import InterpolationResults, LadderAttempt
from q3.io.interpolation_settings import CaptureSettings
from q3.io.results.classes import Results
from q3.workflows.band_distance import eigenvalues_ev
from q3.workflows.interpolation import EngineChain, Grid, KPoint, fermi_energy_ry
from q3.workflows.wannier.attempt_status import AttemptStatus
from q3.workflows.wannier.evaluation import Evaluation
from q3.workflows.wannier.ladder_state import LadderState
from q3.workflows.wannier.protocol_options import ProtocolOptions
from q3.workflows.wannier.rung_name import RungName
from q3.workflows.wannier.rungs import LADDER
from q3.workflows.wannier.trial_orbitals import TrialOrbitals
from q3.workflows.wannier.utils import finer_mesh, off_mesh_reference_kpoints
from q3.workflows.wannier.workflow_result import WorkflowResult

#: What a run that stopped before any evaluation says of itself.
NOTHING_WAS_EVALUATED = "nothing was evaluated"


class WannierProtocol:
    """One run of the protocol over one SCF manifest.

    What the datasets decide is read when the protocol is built: the trial
    orbitals and their count, and the band window sized from it. :meth:`run`
    is the mesh loop -- bounded by how many recaptures were allowed, since the
    mesh rung is the only one that touches DFT -- and each pass is a capture, a
    reference and one walk of the ladder.
    """

    def __init__(self, chain: EngineChain, options: ProtocolOptions):
        self.chain = chain
        self.options = options
        trial_orbitals = TrialOrbitals.of(chain.material, chain.pseudopotential_paths())
        self.wannier_functions = trial_orbitals.number_of_wannier_functions
        self.trial_shells = trial_orbitals.shells_by_species
        self.window = (
            int(options.number_of_bands) if options.number_of_bands else trial_orbitals.band_window
        )
        if self.window < self.wannier_functions:
            raise ValueError(
                f"number_of_bands = {self.window} is below the {self.wannier_functions} trial "
                "orbitals"
            )
        self.fixed_reference: list[KPoint] | None = (
            list(options.reference_kpoints) if options.reference_kpoints is not None else None
        )
        if self.fixed_reference is None and options.band_path is not None:
            self.fixed_reference, _ = chain.kpath(
                options.band_path, options.lattice_type, options.points_per_segment
            )

    def run(self) -> WorkflowResult:
        """SCF, then capture, judge and retry until a model is accepted or nothing is left to try."""
        options = self.options
        spacing = float(options.kpoint_spacing_inverse_angstrom)
        mesh = self.chain.mesh(options.grid, spacing)
        scf_results = None if options.skip_scf else self.chain.run_scf()

        attempts: tuple[LadderAttempt, ...] = ()
        status: Literal["accept", "reject"] = "reject"
        reason = NOTHING_WAS_EVALUATED
        judged: LadderState | None = None
        nscf_results = None
        reference_results = None
        reference_set: list[KPoint] = []

        for mesh_pass in range(options.maximum_mesh_refinements + 1):
            nscf_results = self._captured_on(mesh)
            reference_set = (
                list(self.fixed_reference)
                if self.fixed_reference is not None
                else off_mesh_reference_kpoints(mesh)
            )
            reference_results = self.chain.reference(reference_set, self.window)
            state = self._climbed(
                self._evaluation(mesh, reference_set, reference_results, nscf_results), attempts
            )
            attempts = state.attempts
            if state.stop_reason is not None:
                reason = state.stop_reason
                break
            judged = state
            reason = state.record.reason
            if state.accepted:
                status = "accept"
                break

            finer = finer_mesh(mesh, options.grid, self.chain.material["lattice"], spacing)
            if mesh_pass == options.maximum_mesh_refinements or finer is None:
                break
            spacing /= 2.0
            attempts = attempts + (self._recapture(finer, spacing, reason),)
            mesh = finer

        return WorkflowResult(
            status=status,
            reason=reason,
            grid=mesh,
            number_of_wannier_functions=self.wannier_functions,
            trial_shells=self.trial_shells,
            number_of_bands=self.window,
            attempts=attempts,
            band_distance=judged.distance if judged is not None else None,
            interpolation_results=self._published(judged, status, reason, attempts),
            reference_results=reference_results,
            reference_kpoints=reference_set,
            scf_results=scf_results,
            nscf_results=nscf_results,
            fermi_energy_ry=fermi_energy_ry(nscf_results, scf_results),
        )

    def _captured_on(self, mesh: Grid) -> Results:
        """The full-grid NSCF on *mesh* that saves the record every rung re-evaluates."""
        return self.chain.capture(
            CaptureSettings(
                grid=list(mesh),
                export_formats=list(self.options.export_formats),
                trial_shells=self.trial_shells,
            ),
            number_of_bands=self.window,
        )

    def _evaluation(
        self,
        mesh: Grid,
        reference_set: list[KPoint],
        reference_results: Results,
        nscf_results: Results,
    ) -> Evaluation:
        """What one mesh's record is evaluated and judged against.

        The Fermi level is the capture's, not the SCF's: it is the one the
        classifier measures the frozen window from, so the window the band
        distance weighs and the window that is frozen coincide. For an
        insulator the engine puts it in the gap.
        """
        return Evaluation(
            chain=self.chain,
            mesh=mesh,
            wannier_functions=self.wannier_functions,
            reference_set=reference_set,
            reference_ev=eigenvalues_ev(reference_results),
            fermi_energy_ev=fermi_energy_ry(nscf_results) * RYDBERG_TO_ELECTRONVOLT,
            gate=self.options.gate,
            fixed_reference=self.fixed_reference is not None,
            dos_mesh=self.options.dos_mesh,
            dos_broadening_ry=self.options.dos_broadening_ry,
            maximum_threshold_attempts=self.options.maximum_threshold_attempts,
        )

    @staticmethod
    def _climbed(evaluation: Evaluation, attempts: tuple[LadderAttempt, ...]) -> LadderState:
        """The default evaluation, then the rungs in order; each judges what the last one left."""
        state = evaluation.initial_state(attempts)
        state = evaluation.attempt(state, RungName.DEFAULT, state.settings)
        for rung_class in LADDER:
            rung = rung_class(evaluation)
            if rung.applies(state):
                state = rung.attempt(state)
            if state.stop_reason is not None:
                break
        return state

    @staticmethod
    def _recapture(finer: Grid, spacing: float, reason: str) -> LadderAttempt:
        """The mesh rung's entry in the history: it evaluates nothing, it sends the run back."""
        return LadderAttempt(
            rung=RungName.MESH,
            grid=list(finer),
            settings={"kpoint_spacing_inverse_angstrom": spacing},
            status=AttemptStatus.RECAPTURE,
            reason=f"{reason}; recapturing on {finer[0]}x{finer[1]}x{finer[2]}",
        )

    def _published(
        self,
        judged: LadderState | None,
        status: str,
        reason: str,
        attempts: tuple[LadderAttempt, ...],
    ) -> InterpolationResults | None:
        """The judged model's results file with the verdict and the ladder written into it."""
        if judged is None or judged.results is None:
            return None
        document = judged.results
        quality = document.quality_block()
        quality.status = status
        quality.reason = reason
        quality.ladder = list(attempts)
        results_path = self.chain.results_path()
        if results_path.exists():
            document.write(results_path)
        return document
