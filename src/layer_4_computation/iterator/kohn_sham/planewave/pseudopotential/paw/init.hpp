#pragma once

/**
 * PawPlanewavePseudopotentialKohnShamSCFIterator — SCF with PAW datasets.
 *
 * PAW is a layered extension of ultrasoft here, so this extends the ultrasoft
 * iterator rather than standing beside it. Everything the ultrasoft pathway
 * does is what PAW needs too — the compensation charge on the grid, the
 * S-operator from the dataset's q_ij, the projector overlap coefficients — and
 * three things are added on top:
 *
 *   1. **D gets a one-center term.** After the ultrasoft D is built from
 *      ∫v_eff·Q, (D¹ − D̃¹)_Hxc is added per atom, refreshed each step from
 *      the current occupancies.
 *   2. **The total energy gets Σ_a (E¹_a − Ẽ¹_a),** and the double-counting
 *      correction gets −Σ_a Σ_ij ρ_ij D^oc_ij.
 *   3. **ρ_ij is symmetrized over the point group** before either of those
 *      reads it. Ultrasoft needs no such step: its only consumer of ρ_ij is
 *      the augmentation charge, which reaches the grid and is repaired there
 *      by the density symmetrizer. The one-centre terms never touch a grid.
 *
 * That second subtraction is not optional bookkeeping. The one-center
 * potential entered the band energy through D, so without removing it the
 * on-site terms are counted twice — the same correction the grid potential
 * needs, applied to the on-site part of the same sum. Quantum ESPRESSO
 * subtracts the identical Σ_ij ρ_ij D^oc_ij inside its own double-counting
 * function, alongside its −∫v·n. Getting this wrong
 * produces a converged SCF with a wrong total, which is the failure mode the
 * A2.6 gate is designed to catch and the reason the two pieces live in
 * separate energy fields.
 *
 * **A non-self-consistent run takes ρ_ij from the checkpoint.** It diagonalizes
 * once, before it constructs any density, so the one-centre term it would
 * otherwise use is the empty one the engine starts with: D without it put
 * copper's 3s at −196.3 eV where the SCF had −92.8. The converged occupations
 * are restored as soon as the engine is built, and the term stays built from
 * them for the rest of the run.
 *
 * **The projector overlap coefficients are ρ_ij** (Quantum ESPRESSO's
 * `becsum`). No conversion happens between the array the ultrasoft
 * iterator accumulates and the occupancy matrix the one-center engine
 * consumes; they are the same object in the same channel order. That
 * agreement is a property of the dataset, and `PawOneCenterTerms::build`
 * refuses datasets where it does not hold rather than computing something
 * plausible and wrong.
 */

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_4_computation/paw/one_center/occupation_symmetrization.hpp"
#include "layer_4_computation/paw/one_center/scf_terms.hpp"

namespace q3 {

class PawPlanewavePseudopotentialKohnShamSCFIterator
    : public UltrasoftPlanewavePseudopotentialKohnShamSCFIterator {
  public:
    explicit PawPlanewavePseudopotentialKohnShamSCFIterator(KohnShamSCFIteratorConfig config)
        : UltrasoftPlanewavePseudopotentialKohnShamSCFIterator(std::move(config)) {}
    ~PawPlanewavePseudopotentialKohnShamSCFIterator() override = default;

    [[nodiscard]] std::vector<Matrix<Real>> one_center_occupations() const override {
        return projector_overlap_coefficients_by_spin_;
    }

    [[nodiscard]] bool requires_one_center_occupations() const override {
        return true;
    }

    void
    load_one_center_occupations(const std::vector<Matrix<Real>>& occupations_by_spin) override {
        preloaded_one_center_occupations_ = occupations_by_spin;
    }

  protected:
    paw_one_center::PawOneCenterTerms one_center_terms_;

    /**
     * The point-group average of ρ_ij, built once from this cell's operations.
     *
     * Inactive — and `symmetrize()` a no-op — whenever the density symmetrizer
     * is, which is the point: `control.disable_symmetry`, a Γ-only basis and a
     * cell with no operation beyond the identity all stand this down as well,
     * so the on-site channel and the grid never disagree about whether the run
     * is imposing symmetry.
     */
    paw_one_center::OnSiteOccupationSymmetrizer occupation_symmetrizer_;

    /**
     * ρ_ij handed over from a checkpoint, held until the one-centre engine is
     * built and then kept as what D's one-centre term is built from. Empty for
     * a self-consistent run, which builds the term from its own occupations.
     */
    std::vector<Matrix<Real>> preloaded_one_center_occupations_;
    Matrix<Real> previous_projector_overlaps_;
    std::vector<Matrix<Real>> previous_projector_overlaps_by_spin_;

    /**
     * ρ_ij^out per spin: the occupancies this iteration's ψ produced,
     * symmetrized, *before* damping replaces them with the next input. The
     * energy needs all three arrays -- in, out and mixed -- and damping
     * overwrites the out array in place, so this is the copy that survives.
     */
    std::vector<Matrix<Real>> output_projector_overlaps_by_spin_;

    /**
     * Σ ρ_ij^out · D^oc(ρ^in): the part of the band energy that D's one-centre
     * term put there. Band energy is a sum over ψ_out, whose projections are
     * ρ^out, against the Hamiltonian this iteration was diagonalized with,
     * whose one-centre D was built from ρ^in -- so this is what the
     * double-counting term has to remove.
     */
    Real one_center_band_energy_term_{0.0};

    /**
     * −Σ (ρ_ij^mix − ρ_ij^out) · D^oc(ρ^mix): the one-centre half of the SCF
     * correction, the term that makes the reported energy second order in the
     * on-site residual. The grid density has always had its half
     * (`compute_delta_escf_correction`); see `compute_double_counting_correction`
     * for why the one-centre channel needs one too.
     */
    Real one_center_scf_correction_{0.0};

    /**
     * ‖ρ_ij^fresh − ρ_ij^damped‖² — how far the occupancies just accumulated
     * from ψ are from the ones that built D for this iteration, *before* the
     * damping factor scales the step actually taken. Same squared-magnitude
     * convention `density_residual_` uses, and the same role
     * `hubbard_occupation_residual_` plays for DFT+U: it is what tells the SCF
     * that the on-site channel is still moving.
     *
     * Large sentinel until the first real damping step, so it can never make a
     * run look converged before ρ_ij has been compared against anything.
     */
    Real occupancy_residual_{std::numeric_limits<Real>::max()};

    /**
     * How far ρ_ij is allowed to move per iteration. Chosen to be loose enough
     * that Si and C converge in the same iteration count they did undamped,
     * and tight enough to hold the stiff Cu channel.
     */
    static constexpr Real occupancy_damping_ = 0.4;

    void on_charge_density_initialized() override {
        UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::on_charge_density_initialized();
        if (!one_center_terms_.is_active()) {
            one_center_terms_.build(
                crystal_structure(),
                quantum_system_snapshot_->pseudopotential_basis().pseudopotential_set(),
                exchange_correlation());
            log::info("     PAW: one-center engine active for {} atoms",
                      crystal_structure().number_of_atoms());
            build_occupation_symmetrizer();
        }
        if (!preloaded_one_center_occupations_.empty())
            restore_one_center_occupations();
    }

    void construct_density_from_states() override {
        UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::construct_density_from_states();
        /**
         * A run given the converged occupations keeps the one-centre term built
         * from them. The ones just accumulated come from this run's own k-points
         * and occupations -- a band path's, say -- and rebuilding the term from
         * them would hand the first-iteration re-diagonalization a different D
         * than the pass it repeats.
         */
        /**
         * D still holds the one-centre term built from ρ^in -- the one the
         * Hamiltonian just diagonalized used -- so this is the moment to
         * contract it with ρ^out.
         */
        one_center_band_energy_term_ =
            contract_with_one_center_potential(output_projector_overlaps_by_spin_);
        if (!preloaded_one_center_occupations_.empty()) {
            update_one_center_scf_correction();
            return;
        }
        auto timer_one_center_start = std::chrono::high_resolution_clock::now();
        refresh_one_center_terms();
        update_one_center_scf_correction();
        auto timer_one_center_end = std::chrono::high_resolution_clock::now();
        log::info(
            "    PAW one-center refresh: {:.1f} ms",
            std::chrono::duration<double, std::milli>(timer_one_center_end - timer_one_center_start)
                .count());
    }

    /**
     * Damp ρ_ij between iterations, the way QE mixes `rho%bec` alongside the
     * density.
     *
     * Without this the on-site occupancies are recomputed from fresh ψ every
     * step and fed straight back through D, a loop with gain 1 that the
     * density mixer cannot damp — lowering the mixing beta does nothing to it,
     * which is how the problem identifies itself. Whether that matters is a
     * property of the dataset: Q3 converges Si and C undamped, but the
     * PseudoDojo Cu PAW dataset has a second p channel carrying an
     * augmentation charge of 25.2 electrons and D_ion of −71.8 Ry, giving a
     * one-center D of ~217 Ry, and the loop oscillates between two states
     * ±212 Ry apart until it is damped.
     *
     * Linear damping rather than the full mixer: the fixed point is unchanged
     * either way, since at self-consistency the new and old occupancies are
     * equal, so this only sets how fast the loop is allowed to move. Feeding
     * ρ_ij through the density mixer as QE does would converge stiff cases in
     * fewer iterations, and is the upgrade if one proves too slow.
     *
     * The damped array is what the augmentation charge on the grid sees too,
     * which is deliberate: the density and the on-site terms have to be built
     * from the same occupancies or the energy is not a functional of one state.
     */
    void accumulate_projector_overlaps() override {
        UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::accumulate_projector_overlaps();
        symmetrize_projector_overlaps();
        output_projector_overlaps_by_spin_ = projector_overlap_coefficients_by_spin_;
        /**
         * The per-spin arrays are damped by the same factor and in the same
         * step. Damping them independently, or damping only the total, would
         * let ρ↑ + ρ↓ drift away from the total the augmentation charge on the
         * grid was built from, and the energy would stop being a functional of
         * one state.
         */
        occupancy_residual_ =
            damp_towards_previous(projector_overlap_coefficients_, previous_projector_overlaps_);
        previous_projector_overlaps_by_spin_.resize(projector_overlap_coefficients_by_spin_.size());
        for (size_t spin = 0; spin < projector_overlap_coefficients_by_spin_.size(); spin++)
            occupancy_residual_ =
                std::max(occupancy_residual_,
                         damp_towards_previous(projector_overlap_coefficients_by_spin_[spin],
                                               previous_projector_overlaps_by_spin_[spin]));
    }

    /**
     * The density residual alone cannot see the on-site channel.
     *
     * ρ_ij is a second self-consistent quantity with its own feedback loop,
     * damped here at a fixed rate rather than driven by the density mixer, and
     * it converges more slowly than the density and with a period-2
     * oscillation. Measured on Cu PAW at 80 Ry, the density residual runs two
     * to four orders of magnitude below ‖δρ_ij‖² near convergence, and the gap
     * widens as the run proceeds: at dr2 = 1.5e-13 the occupancy residual is
     * still 6.7e-10. ρ_ij reaches the grid only through the compensation
     * charge, which is confined to the augmentation sphere and so sits at the
     * high-|G| end the Hartree metric's 1/|G|² weight suppresses, while the
     * one-center energy integrates it unweighted. Exiting on dr2
     * alone therefore stops while E¹ − Ẽ¹ is still moving, and because the
     * one-center channel has no first-order SCF correction of its own (the
     * `delta_escf` analogue that makes the grid part second-order in its
     * error), that residual motion lands in the total energy at first order.
     * On Cu at 80 Ry that left ~0.6 mRy of scatter in the total energy while
     * the stress on the same runs moved only ~0.4 kbar, which is why the
     * defect reads as an energy-only one: the stress is evaluated once from
     * the final state rather than assembled across iterations, so it carries
     * no analogous first-order term.
     *
     * Same reasoning, same shape as the DFT+U occupation-matrix gate in
     * `PlanewavePseudopotentialKohnShamSCFIterator::is_iteration_converged()`:
     * the density can be self-consistent while a potential built from a
     * still-relaxing occupancy is not.
     *
     * `occupancy_residual_` is refreshed in `accumulate_projector_overlaps()`,
     * which `perform_iteration()` always runs before this is checked, so it is
     * never stale by an iteration. It is computed from arrays the ultrasoft
     * accumulation has already pool- and band-reduced, by pure arithmetic with
     * a rank-invariant damping scalar, so it needs no reduction of its own --
     * unlike `density_residual_`, whose per-rank FFTW plan noise does.
     */
    bool is_iteration_converged() const override {
        return UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::is_iteration_converged() &&
               occupancy_residual_ < convergence_threshold_;
    }

    /**
     * One damping step, in place; seeds `previous` on the first call.
     *
     * Returns ‖fresh − damped‖² for that array, or the unconverged sentinel on
     * the seeding call, where there is no previous array to have moved away
     * from and zero would be a false "already converged".
     */
    [[nodiscard]] Real damp_towards_previous(Matrix<Real>& current_matrix,
                                             Matrix<Real>& previous_matrix) const {
        auto& current = current_matrix.raw();
        auto& previous = previous_matrix.raw();
        if (previous.size() != current.size()) {
            previous_matrix = current_matrix;
            return std::numeric_limits<Real>::max();
        }
        Real residual = 0.0;
        for (size_t index = 0; index < current.size(); index++) {
            Real step = current[index] - previous[index];
            residual += step * step;
            current[index] = previous[index] + occupancy_damping_ * step;
        }
        previous_matrix = current_matrix;
        return residual;
    }

    void prepare_hamiltonian() override {
        UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::prepare_hamiltonian();
        /**
         * The base wired its own D into the quantum system; the one-center
         * term is added to that same matrix in place, so nothing downstream
         * needs to know a PAW branch exists.
         */
        add_one_center_to_d_matrix();
    }

    /**
     * −Σ_a Σ_ij ρ_ij D^oc_ij, on top of the base's −∫ρ·V_Hxc.
     *
     * The base's term corrects for the grid potential inside the band energy;
     * this corrects for the on-site potential that entered it through D. They
     * are the same correction applied to two different parts of the same sum,
     * which is why this adds rather than replaces.
     */
    Real compute_double_counting_correction() const override {
        Real correction = UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::
            compute_double_counting_correction();
        /**
         * Three occupancy arrays are in play by the time the energy is
         * assembled, and each term has to use the right one:
         *
         *   ρ^in   built the D this iteration's Hamiltonian was diagonalized with
         *   ρ^out  the projections of the ψ that came out of it
         *   ρ^mix  the damped array the next iteration will start from, which
         *          is also what E¹ and the refreshed D^oc are now built from
         *
         * The band energy contains Σ ρ^out · D^oc(ρ^in), so that is what is
         * removed, and E¹ is evaluated at ρ^mix with the first-order term that
         * carries it back to ρ^out:
         *
         *   E¹(ρ^out) ≈ E¹(ρ^mix) − Σ (ρ^mix − ρ^out) · D^oc(ρ^mix)
         *
         * which is the one-centre half of the SCF correction and the same
         * expansion `compute_delta_escf_correction` makes for the grid. With
         * both, the reported energy is second order in the residual of either
         * channel. Quantum ESPRESSO's `delta_escf` carries the same PAW term.
         *
         * What this used to subtract was Σ ρ^mix · D^oc(ρ^mix), mixed against
         * refreshed, while the band energy holds out against input. The
         * difference, Σ ρ^out · (D^oc(ρ^mix) − D^oc(ρ^in)), is first order in
         * the step the damping takes, and the damping takes a fixed fraction
         * of the residual -- so the energy carried an error first order in the
         * on-site residual. On silicon PAW at Γ that kept the energy drifting
         * by ~3e-9 Ry per iteration after the density residual had reached
         * 1e-14, and left a run stopped at a 1e-12 threshold 6e-9 Ry from
         * where it settles.
         *
         * At self-consistency the three arrays coincide and this reduces to
         * the old expression, so a converged energy does not move; what moves
         * is how close an unconverged one already is.
         */
        return correction - one_center_band_energy_term_ + one_center_scf_correction_;
    }

    void compute_energy_components() override {
        UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::compute_energy_components();
        energies_.one_center_energy = one_center_terms_.energy();
    }

    void report_convergence_details() override {
        UltrasoftPlanewavePseudopotentialKohnShamSCFIterator::report_convergence_details();
        /**
         * Second rung of the failure ladder. Printed in QE's labels and order
         * for readability, but see `BranchTotals`: QE's four values are on a
         * different footing and are not directly differenceable against these.
         */
        const auto& branches = one_center_terms_.branch_totals();
        log::info("      -> PAW hartree energy AE = {:14.9f}", branches.all_electron_hartree);
        log::info("      -> PAW hartree energy PS = {:14.9f}", branches.pseudo_hartree);
        log::info("      -> PAW xc energy AE      = {:14.9f}", branches.all_electron_xc);
        log::info("      -> PAW xc energy PS      = {:14.9f}", branches.pseudo_xc);
    }

  private:
    /**
     * Put the checkpoint's ρ_ij where the one-centre engine reads it, and build
     * the term from it now, before the first Hamiltonian is prepared.
     *
     * Refused when the arrays do not have this run's shape -- another cell, a
     * different spin treatment or other datasets -- since reading them anyway
     * would build D from some other calculation's atoms.
     */
    void restore_one_center_occupations() {
        const auto& preloaded = preloaded_one_center_occupations_;
        if (preloaded.size() != projector_overlap_coefficients_by_spin_.size())
            throw std::runtime_error(
                "PAW: the checkpoint's on-site occupations have " +
                std::to_string(preloaded.size()) + " spin channel(s), but this run has " +
                std::to_string(projector_overlap_coefficients_by_spin_.size()) +
                ". A non-self-consistent run must use the spin treatment of the SCF that wrote "
                "the checkpoint.");
        for (size_t spin = 0; spin < preloaded.size(); spin++) {
            const auto& expected = projector_overlap_coefficients_by_spin_[spin];
            if (preloaded[spin].rows() != expected.rows() ||
                preloaded[spin].cols() != expected.cols())
                throw std::runtime_error(
                    "PAW: the checkpoint's on-site occupations are " +
                    std::to_string(preloaded[spin].rows()) + " by " +
                    std::to_string(preloaded[spin].cols()) + ", but this run's are " +
                    std::to_string(expected.rows()) + " by " + std::to_string(expected.cols()) +
                    ". A non-self-consistent run must use the structure and datasets of the "
                    "SCF that wrote the checkpoint.");
        }
        projector_overlap_coefficients_by_spin_ = preloaded;
        auto& total = projector_overlap_coefficients_.raw();
        std::fill(total.begin(), total.end(), 0.0);
        for (const auto& per_spin : preloaded) {
            const auto& values = per_spin.raw();
            for (size_t index = 0; index < total.size(); index++)
                total[index] += values[index];
        }
        refresh_one_center_terms();
        log::info("     PAW: one-center terms built from the checkpoint's on-site occupations");
    }

    /**
     * Build the ρ_ij symmetrizer from this cell's operations and datasets.
     *
     * Skipped, leaving the symmetrizer inactive, exactly when the density
     * symmetrizer stood down: `control.disable_symmetry`, a Γ-only basis, or a
     * cell whose only operation is the identity. Two of those are correctness
     * conditions rather than optimizations — a Γ-only run's half-sphere basis
     * breaks the point group by construction, and `disable_symmetry` is the
     * escape hatch for a cell whose spglib operations are not the ones the user
     * means — and the third is a no-op either way. Asking the density
     * symmetrizer rather than re-deriving the conditions keeps the two from
     * drifting apart, which is what matters: ρ_ij and the grid density are two
     * halves of one charge density, and symmetrizing one without the other
     * would make the energy a functional of neither.
     *
     * The channel layout comes from the projector mapping table's own
     * per-channel radial index and l, so the expansion rule lives in one place.
     */
    void build_occupation_symmetrizer() {
        if (!density_symmetrizer_.is_initialized())
            return;
        const auto& basis = quantum_system_snapshot_->pseudopotential_basis();
        std::vector<paw_one_center::ChannelAngularLayout> layout_by_species;
        layout_by_species.reserve(static_cast<size_t>(crystal_structure().number_of_species()));
        for (int species_index = 0; species_index < crystal_structure().number_of_species();
             species_index++) {
            int channels = basis.number_of_projector_channels(species_index);
            std::vector<int> beta_index(static_cast<size_t>(channels));
            std::vector<int> angular_momentum(static_cast<size_t>(channels));
            for (int channel = 0; channel < channels; channel++) {
                beta_index[static_cast<size_t>(channel)] =
                    basis.projector_to_beta_index(species_index, channel);
                angular_momentum[static_cast<size_t>(channel)] =
                    basis.projector_angular_momentum(species_index, channel);
            }
            layout_by_species.push_back(
                paw_one_center::build_channel_angular_layout(beta_index, angular_momentum));
        }
        occupation_symmetrizer_.build(
            crystal_structure(), cached_symmetry_operations(), std::move(layout_by_species));
        if (occupation_symmetrizer_.is_active())
            log::info("     PAW: on-site occupations symmetrized over {} symmetry operations",
                      occupation_symmetrizer_.number_of_symmetry_operations());
    }

    /**
     * Restore to ρ_ij the symmetry the irreducible-wedge sum does not carry.
     *
     * Before the damping step, so that what is compared against the previous
     * iteration — and what `occupancy_residual_` measures — is the quantity the
     * one-centre term is actually built from. Damping a symmetrized array
     * towards a symmetrized one keeps it symmetric, a convex combination being
     * an average of averages.
     *
     * Every array, per-spin and total together: they are the same numbers
     * summed differently, and symmetrization is linear, so averaging them
     * apart cannot make them disagree.
     *
     * Needs no reduction of its own, for the reason the damping step below
     * does not: the base has already pool- and band-reduced these arrays, so
     * every rank runs the same arithmetic on the same values and lands on the
     * same answer.
     *
     * Not on a run reading ρ_ij from a checkpoint. There the occupations that
     * matter came from an SCF that already symmetrized them, and this pass's
     * own k-points are a band path — an arbitrary set, not a mesh closed under
     * the point group — so averaging what it accumulates over the group would
     * assert a symmetry the sampling does not have.
     */
    void symmetrize_projector_overlaps() {
        if (!occupation_symmetrizer_.is_active() || !preloaded_one_center_occupations_.empty())
            return;
        occupation_symmetrizer_.symmetrize(crystal_structure(), projector_overlap_coefficients_);
        for (auto& per_spin : projector_overlap_coefficients_by_spin_)
            occupation_symmetrizer_.symmetrize(crystal_structure(), per_spin);
    }

    /// Recompute the one-center energy and D from the occupancies just accumulated.
    void refresh_one_center_terms() {
        if (!one_center_terms_.is_active())
            return;
        std::vector<const Matrix<Real>*> coefficient_pointers_by_spin;
        coefficient_pointers_by_spin.reserve(projector_overlap_coefficients_by_spin_.size());
        for (const auto& per_spin : projector_overlap_coefficients_by_spin_)
            coefficient_pointers_by_spin.push_back(&per_spin);
        one_center_terms_.update_by_spin(
            crystal_structure(), coefficient_pointers_by_spin, projector_overlap_row_capacity());
    }

    void add_one_center_to_d_matrix() {
        if (!one_center_terms_.is_active())
            return;
        int number_of_atoms = crystal_structure().number_of_atoms();
        /**
         * Added to every spin channel's D, not the spin-free matrix: the base
         * built one D per channel, and `element()` reads those, so a correction
         * written only to the shared matrix would be dropped.
         */
        for (int spin = 0; spin < number_of_spin_components(); spin++) {
            std::vector<std::vector<Real>> corrections(number_of_atoms);
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++)
                corrections[atom_index] = one_center_terms_.d_correction(atom_index, spin);
            nonlocal_effective_potential_.add_per_atom_d_terms_for_spin(spin, corrections);
        }
    }

    /// Σ_a Σ_ij ρ_ij D^oc_ij, over the same channel indexing ρ_ij uses.
    /**
     * −Σ (ρ^mix − ρ^out) · D^oc(ρ^mix), with D^oc as it now stands.
     *
     * Called after the refresh (or, for a run reading ρ_ij from a checkpoint,
     * in place of it, where D^oc never moved and ρ^mix is what was preloaded).
     */
    void update_one_center_scf_correction() {
        one_center_scf_correction_ =
            contract_with_one_center_potential(output_projector_overlaps_by_spin_) -
            contract_with_one_center_potential(projector_overlap_coefficients_by_spin_);
    }

    [[nodiscard]] Real contract_with_one_center_potential(
        const std::vector<Matrix<Real>>& coefficients_by_spin) const {
        if (!one_center_terms_.is_active())
            return 0.0;
        Real accumulated = 0.0;
        /**
         * Σ_σ Σ_a Σ_ij ρ^σ_ij D^oc,σ_ij — each channel's occupancies against its
         * own D. Pairing the spin-summed ρ_ij with one channel's D would
         * double count one channel and drop the other.
         *
         * Indexed by the atom's own channel count, which is the stride ρ_ij
         * is written with; see the note in `PawOneCenterTerms::update_by_spin`.
         */
        for (size_t spin = 0; spin < coefficients_by_spin.size(); spin++) {
            const auto& coefficients_for_spin = coefficients_by_spin[spin];
            for (int atom_index = 0; atom_index < crystal_structure().number_of_atoms();
                 atom_index++) {
                const auto& correction =
                    one_center_terms_.d_correction(atom_index, static_cast<int>(spin));
                if (correction.empty())
                    continue;
                int species_index = crystal_structure().species_type_index(atom_index);
                int number_of_channels = one_center_terms_.number_of_channels(species_index);
                for (int first = 0; first < number_of_channels; first++) {
                    for (int second = 0; second < number_of_channels; second++) {
                        accumulated +=
                            coefficients_for_spin[atom_index][first * number_of_channels + second] *
                            correction[first * number_of_channels + second];
                    }
                }
            }
        }
        return accumulated;
    }

    /**
     * Allocated row length of the projector overlap array.
     *
     * The ultrasoft iterator sizes ρ_ij rows by the *maximum* channel count
     * over species so a mixed cell's array is rectangular. It is **not** the
     * indexing stride: the accumulation writes at `i * nh_atom + j`. Passed
     * down only so the reader can bound-check itself.
     */
    [[nodiscard]] int projector_overlap_row_capacity() const {
        int maximum_channels = 0;
        const auto& basis = quantum_system_snapshot_->pseudopotential_basis();
        for (int species_index = 0; species_index < crystal_structure().number_of_species();
             species_index++)
            maximum_channels =
                std::max(maximum_channels, basis.number_of_projector_channels(species_index));
        return maximum_channels;
    }
};

} // namespace q3
