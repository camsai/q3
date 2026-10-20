#pragma once

#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"
#include "layer_4_computation/state/init.hpp"
#include "layer_4_computation/state/internal/energies/kohn_sham/planewave/init.hpp"

namespace q3 {

class PlanewaveBasis;
class XCFunctional;

/**
 * Converged SCF state for a plane-wave Kohn-Sham calculation.
 *
 * Extends SCFState with the fields that are specific to a plane-wave
 * representation: charge density, effective potential, and non-owning
 * pointers to the basis and XC functional (which are owned by the iterator).
 *
 * The quantum system snapshot is declared as a pure virtual accessor here
 * (covariant return), so that concrete subtypes can own the correctly typed
 * quantum system by value.
 */
struct PlanewaveKohnShamSCFState : public SCFState {
    PlanewaveKohnShamSCFState() = default;
    PlanewaveKohnShamSCFState(PlanewaveKohnShamSCFState&&) = default;
    PlanewaveKohnShamSCFState& operator=(PlanewaveKohnShamSCFState&&) = default;
    PlanewaveKohnShamSCFState(const PlanewaveKohnShamSCFState&) = delete;
    PlanewaveKohnShamSCFState& operator=(const PlanewaveKohnShamSCFState&) = delete;

    std::vector<Complex> charge_density_g;              // total density in G-space
    std::vector<std::vector<Real>> charge_density_real; // [ispin][ir] in real space
    std::vector<Complex> total_potential_g;             // total potential in G-space
    /**
     * The same potential per spin channel, un-averaged. Empty for an
     * unpolarized run, where `total_potential_g` already is the whole story.
     *
     * Layer 5 needs both: the augmentation force and stress contract the
     * potential against the projector occupancies, and doing that with the
     * spin *average* against the spin *sum* drops
     * ∫(V↑−V↓)(ρ↑−ρ↓) dQ/dR / 2 — identically zero without a moment, and not
     * zero with one.
     */
    std::vector<std::vector<Complex>> total_potential_g_by_spin;
    std::vector<Complex> exchange_correlation_potential_g; // XC potential in G-space
    std::vector<std::vector<Real>> real_space_potential;   // [ispin][ir] effective potential

    const PlanewaveBasis* planewave_basis{nullptr};
    const XCFunctional* exchange_correlation{nullptr};
    const Crystal* crystal_structure{nullptr};
    const KPointsGrid* kpoint_grid{nullptr};
    /**
     * Non-owning pointer to the execution policy, set by the iterator.
     *
     * Forces and stress calculators use this to perform pool-level reductions
     * (MPI_Allreduce across k-point pools) after accumulating k-point-local
     * contributions. Null in serial mode; non-null in MPI mode.
     */
    const execution::PlanewaveExecutionPolicy* execution_policy{nullptr};

    /**
     * Return the correctly typed quantum system snapshot.
     * Overridden with covariant return in each concrete subtype.
     */
    [[nodiscard]] virtual const PlanewaveKohnShamQuantumSystem& quantum_system() const = 0;

    /// Narrow the energies return type to the planewave level.
    const PlanewaveKohnShamEnergies& energies() const override = 0;

    // ----- Convenience accessors (delegate to quantum_system() and planewave_basis) -----

    [[nodiscard]] int number_of_plane_waves(int kpoint_index) const {
        return planewave_basis->number_of_plane_waves_by_kpoint_index(kpoint_index);
    }
    [[nodiscard]] const std::vector<int>& g_vector_indices_at_kpoint(int kpoint_index) const {
        return planewave_basis->g_vector_indices_by_kpoint_index(kpoint_index);
    }
    [[nodiscard]] Complex
    wavefunction_coefficient(int kpoint_index, int band_index, int g_vector_index) const {
        return quantum_system().wavefunction(kpoint_index, band_index, g_vector_index);
    }
    [[nodiscard]] Real eigenvalue(int kpoint_index, int band_index) const {
        return quantum_system().eigenvalue(kpoint_index, band_index);
    }
    [[nodiscard]] Real occupation(int kpoint_index, int band_index) const {
        return quantum_system().occupation_by_kpoint_state_index_and_band_index(kpoint_index,
                                                                                band_index);
    }
    [[nodiscard]] const std::vector<Real>& charge_density_for_spin(int spin_index) const {
        return charge_density_real[spin_index];
    }
    [[nodiscard]] int number_of_occupied_bands_per_spin() const {
        return quantum_system().number_of_occupied_bands_per_spin();
    }
    [[nodiscard]] Real spin_degeneracy() const {
        return quantum_system().spin_degeneracy();
    }
    [[nodiscard]] int number_of_bands() const {
        return quantum_system().number_of_bands();
    }
    [[nodiscard]] int number_of_spin_components() const {
        return quantum_system().number_of_spin_components;
    }
};

} // namespace q3
