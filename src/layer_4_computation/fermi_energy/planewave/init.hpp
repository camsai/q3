#pragma once

/**
 * PlanewaveFermiEnergyCalculator — planewave Fermi energy and occupation calculator.
 *
 * Computes the Fermi energy from eigenvalues on a PlanewaveKohnShamQuantumSystem,
 * dispatching to fixed (insulator) or smeared (metal) occupations.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/fermi_energy/init.hpp"
#include "layer_4_computation/iterator/internal/smearing/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"

namespace q3 {

/// Which end of the electron-count plateau a Fermi-energy bisection converges to.
enum class PlateauEdge { Lower, Upper };

/// Planewave Fermi energy calculator — operates on PlanewaveKohnShamQuantumSystem.
class PlanewaveFermiEnergyCalculator : public FermiEnergyCalculator {
  public:
    /**
     * Default constructor: leaves quantum_system_ unset.
     * calculate_occupations() asserts that it has been set before use.
     */
    PlanewaveFermiEnergyCalculator() = default;

    PlanewaveFermiEnergyCalculator(
        PlanewaveKohnShamQuantumSystem& quantum_system,
        const SmearingScheme* smearing,
        const execution::PlanewaveExecutionPolicy* execution_policy = nullptr)
        : quantum_system_(&quantum_system)
        , smearing_(smearing)
        , execution_policy_(execution_policy) {}

    Real calculate_occupations() override {
        if (quantum_system_ == nullptr) {
            throw std::logic_error(
                "PlanewaveFermiEnergyCalculator::calculate_occupations() called before the "
                "quantum system was assigned. The calculator is wired after construction, so "
                "there is a window in which it is not yet usable; this reports that window "
                "instead of dereferencing null, which is what the assert it replaces did in a "
                "release build.");
        }
        if (uses_smearing()) {
            if (!kpoint_weights_sample_the_brillouin_zone()) {
                return quantum_system_->fermi_energy();
            }
            calculate_smeared_occupations();
        } else {
            calculate_fixed_occupations();
        }
        return quantum_system_->fermi_energy();
    }

    /// Check if this calculator uses smearing.
    bool uses_smearing() const {
        return smearing_ != nullptr && smearing_->uses_smearing();
    }

    const SmearingScheme* smearing() const {
        return smearing_;
    }

  private:
    /**
     * Whether this k-point set is a Brillouin-zone quadrature at all.
     *
     * A smeared Fermi energy is found by bisecting sum_k w_k sum_n f(e_nk)
     * against the electron count, which is a BZ integral and means nothing
     * unless the weights are the quadrature's. KPointsGrid produces weights
     * summing to the spin degeneracy -- 2 unpolarized, 1 per channel when the
     * two spins are enumerated as separate states -- and a symmetry reduction
     * preserves that sum. A BAND PATH does not: q3.workflows.band_structure
     * gives every point weight 1, so the sum is the number of points on the
     * path and the count comes out tens of times too large at every trial
     * energy. The bisection then never reaches the target and converges to its
     * own lower bracket, which is the smallest eigenvalue in the run.
     *
     * That is not a subtle error. Measured on copper: a band-path NSCF reported
     * a Fermi energy of -7.046 Ry, BELOW the lowest band it had just computed,
     * where the SCF on the same density reports 1.243 Ry. It is reported rather
     * than computed here because a path samples lines through the zone, not the
     * zone -- no weighting of it can be made into the integral, so the honest
     * answer is that this run cannot determine one.
     *
     * The Fermi energy is left as it stands, which is the same choice the
     * missing-band-edge path below makes and for the same reason: a zero a
     * reader can recognize as "not determined" beats a plausible-looking number
     * that is wrong. Use the SCF's value, which the workflows carry.
     */
    [[nodiscard]] bool kpoint_weights_sample_the_brillouin_zone() const {
        const KPointsGrid& grid = quantum_system_->kpoint_grid();
        Real total_weight = 0.0;
        for (int kpoint_index = 0; kpoint_index < grid.number_of_kpoints(); kpoint_index++) {
            total_weight += grid[kpoint_index].weight;
        }
        /**
         * With pool parallelism this grid holds only the pool's own k-points,
         * so the local sum is a fraction of the whole and every multi-rank run
         * would fail this test. Summing across pools first is what makes the
         * answer -- and therefore the decision every rank takes below -- the
         * same on one rank and on four, which is exactly the invariance the MPI
         * topology tests check and which the first version of this guard broke.
         */
        if (execution_policy_ != nullptr && execution_policy_->number_of_pools() > 1) {
            execution_policy_->reduce_scalar(total_weight);
        }
        const Real expected = quantum_system_->spin_degeneracy();
        if (std::abs(total_weight - expected) <= math::tolerances::kpoint_weight_sum) {
            return true;
        }
        warn_once("Fermi energy: the {} k-point weights sum to {:.6f}, not the {:.1f} a "
                  "Brillouin-zone sampling carries, so this k-point set is not one this "
                  "calculation can integrate over -- a band path is the usual reason, since it "
                  "traces lines through the zone rather than filling it. No Fermi energy is "
                  "determined from it; the value stays at {:.8f} Ry. Take the Fermi energy from "
                  "the self-consistent run that produced the density.",
                  grid.number_of_kpoints(),
                  total_weight,
                  expected,
                  quantum_system_->fermi_energy());
        return false;
    }

    /**
     * For fixed occupations (insulators): set HOMO-LUMO midpoint as Fermi energy.
     *
     * The midpoint needs both edges to exist, and one of them can be missing. A
     * run whose band count equals its occupied-band count -- an NSCF asked for
     * exactly the occupied manifold, as a Wannier workflow does -- has no LUMO
     * to pair the HOMO with. Reporting the midpoint anyway is how
     * `fermi_energy_ry: 5e+29` reached a results file: the LUMO search starts
     * at a +1e30 sentinel that nothing overwrites, and half of it survives into
     * the answer. Both edges are handled here rather than only that one,
     * because the HOMO search has the same shape at the other end.
     *
     * With only one edge, that edge is the answer: Quantum ESPRESSO reports
     * "highest occupied level" alone where it would otherwise print "highest
     * occupied, lowest unoccupied level", and this is the same statement --
     * everything known about where E_F lies, without inventing the half that
     * was never computed. A warning says so, since a run asked for a Fermi
     * energy and got a band edge -- once per calculator, not once per SCF
     * iteration, which would bury the rest of the log in a repetition of a
     * fact that cannot change between iterations.
     */
    void calculate_fixed_occupations() const {
        int number_of_kpoint_states = quantum_system_->number_of_kpoint_states();
        int number_of_bands = quantum_system_->number_of_bands();
        int occupied_bands = quantum_system_->number_of_occupied_bands_per_spin();

        /**
         * Both predicates are properties of the band count and the electron
         * count, which every pool shares, so they cannot make the reductions
         * below disagree between ranks.
         */
        const bool has_occupied_band = occupied_bands > 0 && occupied_bands <= number_of_bands;
        const bool has_unoccupied_band = occupied_bands < number_of_bands;

        // Fermi energy is midpoint of HOMO-LUMO gap
        Real highest_occupied_molecular_orbital = -1e30;
        Real lowest_unoccupied_molecular_orbital = 1e30;
        for (int kpoint_state_index = 0; kpoint_state_index < number_of_kpoint_states;
             kpoint_state_index++) {
            if (has_occupied_band) {
                highest_occupied_molecular_orbital =
                    std::max(highest_occupied_molecular_orbital,
                             quantum_system_->eigenvalue(kpoint_state_index, occupied_bands - 1));
            }
            if (has_unoccupied_band) {
                lowest_unoccupied_molecular_orbital =
                    std::min(lowest_unoccupied_molecular_orbital,
                             quantum_system_->eigenvalue(kpoint_state_index, occupied_bands));
            }
        }

        /**
         * With pool parallelism, each pool has only local eigenvalues.
         * Reduce HOMO (max) and LUMO (min) across all pools to find global gap.
         */
        if (execution_policy_ != nullptr && execution_policy_->number_of_pools() > 1) {
            execution_policy_->reduce_scalar_max(highest_occupied_molecular_orbital);
            execution_policy_->reduce_scalar_min(lowest_unoccupied_molecular_orbital);
        }

        if (has_occupied_band && has_unoccupied_band) {
            quantum_system_->set_fermi_energy(
                0.5 * (highest_occupied_molecular_orbital + lowest_unoccupied_molecular_orbital));
            return;
        }

        if (has_occupied_band) {
            warn_once("Fermi energy: {} band(s) requested for {} occupied band(s), so there is no "
                      "unoccupied level to bound the gap from above; reporting the highest "
                      "occupied level ({:.8f} Ry) instead of a mid-gap Fermi energy.",
                      number_of_bands,
                      occupied_bands,
                      highest_occupied_molecular_orbital);
            quantum_system_->set_fermi_energy(highest_occupied_molecular_orbital);
            return;
        }

        if (has_unoccupied_band) {
            warn_once("Fermi energy: no occupied bands ({} electron(s)), so there is no occupied "
                      "level to bound the gap from below; reporting the lowest unoccupied level "
                      "({:.8f} Ry).",
                      quantum_system_->number_of_electrons(),
                      lowest_unoccupied_molecular_orbital);
            quantum_system_->set_fermi_energy(lowest_unoccupied_molecular_orbital);
            return;
        }

        /**
         * Neither edge exists, so there is no eigenvalue to report at all. The
         * system's own initial value stands rather than a sentinel: a zero that
         * a reader can recognize as "not determined" beats 5e+29, and a run
         * with no bands has nothing else to say.
         */
        warn_once("Fermi energy: no bands to report one from; leaving it at {:.8f} Ry.",
                  quantum_system_->fermi_energy());
    }

    /**
     * Report a missing band edge once for this calculator's lifetime.
     *
     * Which edge is missing follows from the band and electron counts, so it
     * holds for every iteration of a run; saying it each time would add a line
     * per SCF step and tell the reader nothing new after the first.
     */
    template <typename... Arguments>
    void warn_once(const char* message, Arguments&&... arguments) const {
        if (has_warned_about_missing_band_edge_) {
            return;
        }
        has_warned_about_missing_band_edge_ = true;
        log::warn(message, std::forward<Arguments>(arguments)...);
    }

    /**
     * For smeared occupations (metals): bisection to find Fermi energy,
     * then set fractional occupations.
     */
    void calculate_smeared_occupations() const {
        int number_of_kpoint_states = quantum_system_->number_of_kpoint_states();
        int number_of_bands = quantum_system_->number_of_bands();
        Real number_of_electrons = quantum_system_->number_of_electrons();
        Real smearing_width = smearing_->smearing_width();

        /**
         * Every k-point state's weight and eigenvalues, in global order.
         *
         * The bisection below evaluates the electron count ~100 times per
         * plateau edge. It used to sum each pool's states locally and
         * Allreduce the partial sum on every trial energy -- two hundred
         * scalar collectives per SCF iteration, and a count whose rounding
         * depended on how the k-points were split across pools, so the Fermi
         * energy itself moved in its last bits with the topology. Gathering
         * the (tiny) eigenvalue table once and bisecting on it locally removes
         * both: no collective inside the loop, and every rank sums the same
         * numbers in the same global order a serial run does, so the count --
         * and with it E_F -- is bit-identical on 1 rank and on N pools for
         * identical eigenvalues. Serial runs take the local table unchanged,
         * which is the order and arithmetic they always had.
         */
        const std::vector<Real> state_table = global_kpoint_state_table();
        const int values_per_state = number_of_bands + 1;
        const int number_of_global_states = static_cast<int>(state_table.size()) / values_per_state;
        const auto weight_of = [&](int state_index) {
            return state_table[static_cast<std::size_t>(state_index) * values_per_state];
        };
        const auto eigenvalue_of = [&](int state_index, int band_index) {
            return state_table[static_cast<std::size_t>(state_index) * values_per_state + 1 +
                               band_index];
        };

        // Find eigenvalue range
        Real eigenvalue_minimum = 1e30, eigenvalue_maximum = -1e30;
        for (int state_index = 0; state_index < number_of_global_states; state_index++) {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                Real energy = eigenvalue_of(state_index, band_index);
                eigenvalue_minimum = std::min(eigenvalue_minimum, energy);
                eigenvalue_maximum = std::max(eigenvalue_maximum, energy);
            }
        }

        eigenvalue_minimum -= 1.0;
        eigenvalue_maximum += 1.0;

        /**
         * Σ_{k,σ,n} w_k f((ε − E_F)/σ) over the global table. K-point weights
         * already carry the spin factor: they sum to 2 (spin degeneracy) for
         * unpolarized runs and are halved per channel for LSDA, so there is
         * no extra factor. Summed in one fixed order on every rank, so the two
         * bisections agree across ranks by construction, with no reduction.
         */
        auto count_electrons = [&](Real trial_fermi_energy) {
            Real total_electrons = 0.0;
            for (int state_index = 0; state_index < number_of_global_states; state_index++) {
                Real weight = weight_of(state_index);
                for (int band_index = 0; band_index < number_of_bands; band_index++) {
                    Real energy_difference_ratio =
                        (eigenvalue_of(state_index, band_index) - trial_fermi_energy) /
                        smearing_width;
                    total_electrons += weight * smearing_->occupation(energy_difference_ratio);
                }
            }
            return total_electrons;
        };

        /**
         * Bisect for one end of the electron-count plateau and return it.
         *
         * `PlateauEdge::Lower` breaks ties toward the bottom (the first E_F
         * reaching the target count), `Upper` toward the top, within the
         * count tolerance. On a metal the two differ only by that tolerance
         * divided by the density of states; across a gap they bracket it.
         */
        auto bisect_for_fermi_energy = [&](PlateauEdge edge) {
            bool take_upper_edge = edge == PlateauEdge::Upper;
            Real lower_bound = eigenvalue_minimum;
            Real upper_bound = eigenvalue_maximum;
            Real trial_fermi_energy = 0.0;
            for (int iteration = 0; iteration < 100; iteration++) {
                trial_fermi_energy = 0.5 * (lower_bound + upper_bound);
                Real total_electrons = count_electrons(trial_fermi_energy);
                /**
                 * On the plateau the electron count equals the target, and
                 * which way the tie breaks decides which end of the plateau
                 * this converges to: `<` walks down to its bottom, `<=` (with
                 * a tolerance, since the count only equals the target to
                 * within the smearing tails) walks up to its top.
                 */
                bool below_target =
                    take_upper_edge
                        ? total_electrons <=
                              number_of_electrons + math::tolerances::fermi_electron_count_plateau
                        : total_electrons < number_of_electrons;
                if (below_target)
                    lower_bound = trial_fermi_energy;
                else
                    upper_bound = trial_fermi_energy;
                if (std::abs(upper_bound - lower_bound) <
                    math::tolerances::fermi_energy_bisection_convergence)
                    break;
            }
            return trial_fermi_energy;
        };

        /**
         * Where E_F sits *inside a gap* is not arbitrary once the occupations
         * are smeared.
         *
         * The electron count is flat across a gap -- every E_F in it returns
         * the same integer count -- so a single bisection converges to
         * whichever end its tie-breaking favours, which is the bottom: the
         * valence band edge. That places occupied states directly at E_F, and
         * the smearing then spills them. Measured on FeO with U on Fe 3d and
         * degauss = 0.02 Ry: E_F landed 1.06 eV (3.9 sigma) below the value an
         * independent implementation reports, carrying 4.2 mRy of spurious
         * entropy where that implementation has exactly zero, and moving the
         * total energy by 2.4 mRy. The error vanishes below degauss ~ 0.01 Ry,
         * where the tails no longer reach -- which is why it hid: it is
         * invisible in the sigma -> 0 limit the tests mostly live in, and grows
         * with the smearing width users actually pick for hard magnetic cases.
         *
         * Taking the midpoint of the bracket puts E_F as far from both band
         * edges as the gap allows. On a metal there is no plateau, so the two
         * ends differ only by the count tolerance divided by the density of
         * states, and the midpoint would leave the cell holding half that
         * tolerance in extra electrons. The width test below keeps a metal on
         * the single-bisection answer, where the count is exact. It compares
         * against the smearing width: a gap the smearing straddles spans
         * several of them (FeO's lower end alone sat 3.9 sigma below the
         * independent value above), while a metal's bracket is far narrower --
         * 7.6e-8 Ry on copper, whose density of states is 13 states/Ry. The
         * bisection tolerance it once compared against, 1e-12 Ry, took the
         * midpoint on every metal: copper then carried 19 + 5e-7 electrons, and
         * its total energy rose by E_F x 5e-7 = 0.62 μRy.
         *
         * Refining further to the true mid-gap -- the highest eigenvalue below
         * the plateau and the lowest above it -- is the sigma-independent
         * choice and was tried: it improves degauss = 0.02 to within 0.34 mRy,
         * but destabilises small smearings, where the count stays flat far past
         * the gap, the bracket spans other bands, and the spectrum gets split
         * in the wrong place. Identifying the gap robustly needs the
         * occupied-state count rather than the current E_F, which is a larger
         * change than this fix.
         */
        Real fermi_energy = bisect_for_fermi_energy(PlateauEdge::Lower);
        Real fermi_energy_plateau_top = bisect_for_fermi_energy(PlateauEdge::Upper);
        if (fermi_energy_plateau_top - fermi_energy > smearing_width)
            fermi_energy = 0.5 * (fermi_energy + fermi_energy_plateau_top);

        quantum_system_->set_fermi_energy(fermi_energy);

        // Compute fractional occupations
        for (int kpoint_state_index = 0; kpoint_state_index < number_of_kpoint_states;
             kpoint_state_index++) {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                Real energy_difference_ratio =
                    (quantum_system_->eigenvalue(kpoint_state_index, band_index) - fermi_energy) /
                    smearing_width;
                quantum_system_->set_occupation(
                    kpoint_state_index, band_index, smearing_->occupation(energy_difference_ratio));
            }
        }
    }

    /**
     * This pool's k-point states as rows of [w_k, e_1k, ..., e_Nk], assembled
     * across pools into the global spin-major order (one gather, no per-step
     * reduction). Without pool parallelism the local rows are the global ones.
     */
    [[nodiscard]] std::vector<Real> global_kpoint_state_table() const {
        const int number_of_kpoint_states = quantum_system_->number_of_kpoint_states();
        const int number_of_bands = quantum_system_->number_of_bands();
        const int values_per_state = number_of_bands + 1;
        const KPointsGrid& grid = quantum_system_->kpoint_grid();
        const int local_number_of_kpoints = grid.number_of_kpoints();
        std::vector<Real> table(static_cast<std::size_t>(number_of_kpoint_states) *
                                values_per_state);
        if (local_number_of_kpoints == 0) {
            table.clear();
        }
        for (int kpoint_state_index = 0;
             local_number_of_kpoints > 0 && kpoint_state_index < number_of_kpoint_states;
             kpoint_state_index++) {
            const std::size_t row = static_cast<std::size_t>(kpoint_state_index) * values_per_state;
            table[row] = grid[kpoint_state_index % local_number_of_kpoints].weight;
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                table[row + 1 + band_index] =
                    quantum_system_->eigenvalue(kpoint_state_index, band_index);
            }
        }
        if (execution_policy_ == nullptr || execution_policy_->number_of_pools() <= 1) {
            return table;
        }
        /**
         * The spin-block count comes from the spin setting, not from this
         * pool's state count, so a pool that owns no k-point still passes the
         * same dimensions as every other pool into the collective.
         */
        const int number_of_spin_blocks = quantum_system_->number_of_spin_components == 2 ? 2 : 1;
        return execution_policy_->assemble_global_kpoint_state_data(
            table, local_number_of_kpoints, number_of_spin_blocks, values_per_state);
    }

    mutable bool has_warned_about_missing_band_edge_ = false;
    PlanewaveKohnShamQuantumSystem* quantum_system_ = nullptr;
    const SmearingScheme* smearing_ = nullptr;
    const execution::PlanewaveExecutionPolicy* execution_policy_ = nullptr;
};

} // namespace q3
