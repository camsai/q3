#pragma once

/**
 * NormConservingPlanewaveBasis — static pseudopotential data in the plane-wave basis.
 *
 * Owns all norm-conserving pseudopotential data that is computed once from
 * (Crystal, PlanewaveBasis, PseudopotentialSet) and never changes during SCF:
 * - PseudopotentialSet (owned copy)
 * - BetaFunctionTable (tabulated β(q) projectors)
 * - ProjectorMappingTable (KB projector → (l,m) channel expansion)
 * - CoreChargeData (NLCC core charge density)
 * - Local potential V_loc(G) per species and total
 * - Atomic charge density ρ_at(G) per species
 * - Mesh truncation sizes for radial integration
 *
 * The ultrasoft subclass UltrasoftPlanewaveBasis adds
 * augmentation Q-tables, spherical harmonics, and Gaunt coefficients.
 */

#include <cmath>
#include <memory>
#include <vector>

#include "layer_0_core/math/interpolation/lagrange.hpp"
#include "layer_0_core/math/spherical/harmonics.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/math/transform/spherical_bessel.hpp"
#include "layer_0_core/types/matrix.hpp"

#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/charge_atomic.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/charge_core.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/potential_local.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/factory.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/beta_function_table/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/core_charge_data.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/core_charge_density_table.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/local_potential_table.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/internal/projector_mapping_table.hpp"

namespace q3 {

// Forward declarations
class Crystal;

class NormConservingPlanewaveBasis {
  public:
    NormConservingPlanewaveBasis() = default;
    virtual ~NormConservingPlanewaveBasis() = default;

    /**
     * Initialize all pseudopotential data from crystal, basis, and PP files.
     * Must be called after PlanewaveBasis::generate().
     */
    virtual void init(const Crystal& crystal_structure,
                      const PlanewaveBasis& planewave_basis,
                      const PseudopotentialSet& pseudopotential_set) {
        planewave_basis_ptr_ = &planewave_basis;
        pseudopotential_set_ = pseudopotential_set.clone();
        int number_of_species_types = static_cast<int>(pseudopotential_set_.size());

        Real cell_volume = crystal_structure.cell_volume();
        const auto& g_vector_magnitudes = planewave_basis.g_vector_magnitudes();
        // Step 1: Compute mesh truncation sizes for radial integration
        compute_mesh_truncation_sizes(number_of_species_types);

        /**
         * Step 2: Tabulate beta functions on a uniform q-grid, with the table
         * implementation chosen by what the datasets carry (see the factory).
         */
        beta_function_table_ = create_beta_function_table(pseudopotential_set_);
        beta_function_table_->initialize(
            planewave_basis, pseudopotential_set_, number_of_species_types);

        // Step 3: Tabulate local potential V_loc(q) on uniform q-grid
        local_potential_table_.initialize(beta_function_table_->q_grid(),
                                          pseudopotential_set_,
                                          mesh_sizes_,
                                          cell_volume,
                                          number_of_species_types);

        // Step 4: Tabulate core charge density ρ_core(q) on uniform q-grid
        core_charge_density_table_.initialize(beta_function_table_->q_grid(),
                                              pseudopotential_set_,
                                              mesh_sizes_,
                                              number_of_species_types);

        // Step 5: Compute local potential in G-space from pre-tabulated values
        int number_of_g_vectors = planewave_basis.number_of_g_vectors();
        local_potential_g_per_species_.resize(number_of_species_types);
        for (int type_index = 0; type_index < number_of_species_types; type_index++) {
            local_potential_g_per_species_[type_index].resize(number_of_g_vectors);
            for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
                local_potential_g_per_species_[type_index][g_vector_index] =
                    local_potential_table_.interpolate(type_index,
                                                       g_vector_magnitudes[g_vector_index]);
            }
        }
        // Compute structure factors and total Vloc(G)
        total_local_potential_g_.resize(number_of_g_vectors, Complex(0.0, 0.0));
        for (int g_vector_index = 0; g_vector_index < number_of_g_vectors; g_vector_index++) {
            for (int type_index = 0; type_index < number_of_species_types; type_index++) {
                Complex structure_factor(0.0, 0.0);
                for (int atom_index = 0; atom_index < crystal_structure.number_of_atoms();
                     atom_index++) {
                    if (crystal_structure.species_type_index(atom_index) == type_index) {
                        Vector3D atomic_position =
                            crystal_structure.convert_from_crystal_to_cartesian(atom_index);
                        const auto& g_vector = planewave_basis.g_vectors()[g_vector_index];
                        Real phase = constants::two_pi * g_vector.g_cart.dot(atomic_position);
                        structure_factor += compute_phase_factor(phase);
                    }
                }
                total_local_potential_g_[g_vector_index] +=
                    local_potential_g_per_species_[type_index][g_vector_index] * structure_factor;
            }
        }

        // Step 6: Compute atomic charge density in G-space
        atomic_charge_density_g_ =
            tabulate_atomic_charge_density_in_g_space(pseudopotential_set_,
                                                      mesh_sizes_,
                                                      g_vector_magnitudes,
                                                      beta_function_table_->q_spacing(),
                                                      beta_function_table_->number_of_q_points());

        // Step 7: Build projector mapping tables
        projector_mapping_table_.build(
            crystal_structure, pseudopotential_set_, number_of_species_types);

        // Step 8: Compute NLCC core charge density in G-space from pre-tabulated values
        core_charge_data_.initialize(crystal_structure,
                                     planewave_basis,
                                     pseudopotential_set_,
                                     g_vector_magnitudes,
                                     cell_volume,
                                     core_charge_density_table_);
    }

    // ─── PlanewaveBasis access ──────────────────────────────────────

    /**
     * Access the plane-wave basis set this PP basis was built from.
     */
    [[nodiscard]] const PlanewaveBasis& planewave_basis() const {
        return *planewave_basis_ptr_;
    }

    // ─── Pseudopotential accessors ──────────────────────────────────

    /**
     * Access the collection of pseudopotentials for all species.
     */
    [[nodiscard]] const PseudopotentialSet& pseudopotential_set() const {
        return pseudopotential_set_;
    }

    /**
     * Return the local pseudopotential in G-space for a single species.
     * @param species_type_index  Index into the pseudopotential set (0-based)
     * @return V_loc(|G|) values indexed by G-vector index
     */
    [[nodiscard]] const std::vector<Real>& local_potential_g(int species_type_index) const {
        return local_potential_g_per_species_[species_type_index];
    }

    /**
     * Return the total local pseudopotential in G-space (structure-factor-weighted sum
     * over all atoms and species).
     */
    [[nodiscard]] const std::vector<Complex>& total_local_potential_g() const {
        return total_local_potential_g_;
    }

    /**
     * Total number of Kleinman-Bylander projectors across all atoms
     * (sum of projector channels per atom, accounting for angular momentum degeneracy).
     */
    [[nodiscard]] int total_number_of_projectors() const {
        return projector_mapping_table_.total_number_of_projectors();
    }

    /**
     * Number of radial beta projectors for a given species (before angular expansion).
     */
    [[nodiscard]] int number_of_projectors(int species_type) const {
        return static_cast<int>(pseudopotential_set_[species_type].projectors.size());
    }

    /**
     * Angular momentum quantum number l for a specific beta projector of a species.
     */
    [[nodiscard]] int beta_angular_momentum(int species_type, int projector_index) const {
        return pseudopotential_set_[species_type].projectors[projector_index].angular_momentum;
    }

    /**
     * Number of expanded projector channels for a species
     * (sum of 2l+1 over all beta projectors, i.e. including magnetic quantum numbers).
     */
    [[nodiscard]] int number_of_projector_channels(int species_type) const {
        return projector_mapping_table_.number_of_channels(species_type);
    }

    /**
     * Bare D-matrix element D^0_{ij} from the pseudopotential file (species-level).
     * For norm-conserving PPs, this is the sole nonlocal coupling.
     * For ultrasoft PPs, this is the ion-bare part before screening.
     * @param species_type  Species index
     * @param projector_a   Beta projector index i (radial)
     * @param projector_b   Beta projector index j (radial)
     */
    [[nodiscard]] Real
    bare_d_matrix_element(int species_type, int projector_a, int projector_b) const {
        return pseudopotential_set_[species_type].d_ion_matrix[projector_a][projector_b];
    }

    /**
     * Map an expanded projector channel index back to its radial beta projector index.
     */
    [[nodiscard]] int projector_to_beta_index(int species_type, int projector_index) const {
        return projector_mapping_table_.to_beta_index(species_type, projector_index);
    }

    /**
     * Combined (l,m) index for an expanded projector channel (QE convention: l² + offset).
     */
    [[nodiscard]] int projector_combined_lm_index(int species_type, int projector_index) const {
        return projector_mapping_table_.combined_lm_index(species_type, projector_index);
    }

    /**
     * Angular momentum l of an expanded projector channel.
     */
    [[nodiscard]] int projector_angular_momentum(int species_type, int projector_index) const {
        return projector_mapping_table_.angular_momentum(species_type, projector_index);
    }

    /**
     * Magnetic quantum number m of an expanded projector channel.
     */
    [[nodiscard]] int projector_magnetic_quantum_number(int species_type,
                                                        int projector_index) const {
        return projector_mapping_table_.magnetic_quantum_number(species_type, projector_index);
    }

    /**
     * Atomic charge density in G-space for a species (used as initial guess).
     */
    [[nodiscard]] const std::vector<Real>& atomic_charge_density_g(int species_type) const {
        return atomic_charge_density_g_[species_type];
    }

    /**
     * Whether any species has nonlinear core correction (NLCC) for XC.
     */
    [[nodiscard]] bool has_nonlinear_core_correction() const {
        return core_charge_data_.has_nonlinear_core_correction();
    }

    /**
     * NLCC core charge density in G-space (structure-factor-weighted sum over atoms).
     */
    [[nodiscard]] const std::vector<Complex>& core_charge_density_reciprocal() const {
        return core_charge_data_.core_charge_density_reciprocal();
    }

    /**
     * Interpolate NLCC core charge density from pre-tabulated values.
     * Uses 4-point Lagrange interpolation on the uniform q-grid.
     * @param element_type  Species index
     * @param q_value       Wavevector magnitude |q| in atomic units (1/Bohr)
     * @return Interpolated core charge density at |q|
     */
    [[nodiscard]] Real interpolate_rho_core(int element_type, Real q_value) const {
        return core_charge_density_table_.interpolate(element_type, q_value);
    }

    /**
     * Access the pre-tabulated beta function table for direct interpolation.
     */
    [[nodiscard]] const BetaFunctionTable& beta_function_table() const {
        return *beta_function_table_;
    }

    /**
     * Interpolate beta projector function β(|q|) from the pre-tabulated q-grid.
     * @param species_type_index  Species index
     * @param beta_index          Radial beta projector index
     * @param q_value             Wavevector magnitude |q| in atomic units (1/Bohr)
     */
    [[nodiscard]] Real
    interpolate_beta(int species_type_index, int beta_index, Real q_value) const {
        return beta_function_table_->interpolate(species_type_index, beta_index, q_value);
    }

    /**
     * Interpolate the derivative dβ/dq of a beta projector function.
     * Uses 4-point Lagrange derivative interpolation.
     * @param species_type_index  Species index
     * @param beta_index          Radial beta projector index
     * @param q_value             Wavevector magnitude |q| in atomic units (1/Bohr)
     */
    [[nodiscard]] Real
    interpolate_dbeta(int species_type_index, int beta_index, Real q_value) const {
        return beta_function_table_->interpolate_derivative(
            species_type_index, beta_index, q_value);
    }

    /**
     * Interpolate local potential V_loc(|q|) from pre-tabulated uniform q-grid.
     * Replaces the expensive per-call radial integration with O(1) Lagrange interpolation.
     */
    [[nodiscard]] Real interpolate_local_potential(int element_type, Real q_value) const {
        return local_potential_table_.interpolate(element_type, q_value);
    }

    /**
     * Interpolate derivative dV_loc/dq from pre-tabulated uniform q-grid.
     */
    [[nodiscard]] Real interpolate_local_potential_derivative(int element_type,
                                                              Real q_value) const {
        return local_potential_table_.interpolate_derivative(element_type, q_value);
    }

    /**
     * Interpolate core charge density ρ_core(|q|) from pre-tabulated uniform q-grid.
     * Replaces the expensive per-call radial integration with O(1) Lagrange interpolation.
     */
    [[nodiscard]] Real interpolate_core_charge_density(int element_type, Real q_value) const {
        return core_charge_density_table_.interpolate(element_type, q_value);
    }

    /**
     * Interpolate derivative dρ_core/dq from pre-tabulated uniform q-grid.
     */
    [[nodiscard]] Real interpolate_core_charge_density_derivative(int element_type,
                                                                  Real q_value) const {
        return core_charge_density_table_.interpolate_derivative(element_type, q_value);
    }

    /**
     * Access the local potential interpolation table.
     */
    [[nodiscard]] const LocalPotentialTable& local_potential_table() const {
        return local_potential_table_;
    }

    /**
     * Access the core charge density interpolation table.
     */
    [[nodiscard]] const CoreChargeDensityTable& core_charge_density_table() const {
        return core_charge_density_table_;
    }

    // ─── Projector mapping table access ─────────────────────────────

    /**
     * Access the projector mapping table for expanded projector channel queries.
     */
    [[nodiscard]] const ProjectorMappingTable& projector_mapping_table() const {
        return projector_mapping_table_;
    }

    // ─── Ultrasoft detection ────────────────────────────────────────

    /**
     * Whether this basis uses ultrasoft pseudopotentials.
     * Base class returns false; UltrasoftPlanewaveBasis returns true.
     */
    [[nodiscard]] virtual bool is_ultrasoft() const {
        return false;
    }

  protected:
    const PlanewaveBasis* planewave_basis_ptr_ = nullptr;

    PseudopotentialSet pseudopotential_set_;
    std::vector<std::vector<Real>> local_potential_g_per_species_;
    std::vector<std::vector<Real>> atomic_charge_density_g_;
    std::vector<Complex> total_local_potential_g_;

    std::unique_ptr<BetaFunctionTable> beta_function_table_;
    LocalPotentialTable local_potential_table_;
    CoreChargeDensityTable core_charge_density_table_;
    ProjectorMappingTable projector_mapping_table_;
    CoreChargeData core_charge_data_;

    Real cell_volume_{1.0};
    std::vector<int> mesh_sizes_;

    // ─── Init helper (mesh truncation) ──────────────────────────────

    /**
     * Compute mesh truncation sizes for radial integration of each species.
     */
    void compute_mesh_truncation_sizes(int number_of_species_types) {
        std::vector<int> mesh_truncation_sizes(number_of_species_types);
        for (int type_index = 0; type_index < number_of_species_types; type_index++) {
            const auto& pseudopotential = pseudopotential_set_[type_index];
            if (math::tolerances::use_radial_integration_truncation) {
                mesh_truncation_sizes[type_index] = pseudopotential.mesh_size;
                for (int radial_index = 0; radial_index < pseudopotential.mesh_size;
                     radial_index++) {
                    if (pseudopotential.radial_grid[radial_index] >
                        math::tolerances::radial_integration_truncation_radius) {
                        mesh_truncation_sizes[type_index] = radial_index + 1;
                        break;
                    }
                }
                mesh_truncation_sizes[type_index] =
                    2 * ((mesh_truncation_sizes[type_index] + 1) / 2) - 1;
                if (mesh_truncation_sizes[type_index] > pseudopotential.mesh_size) {
                    mesh_truncation_sizes[type_index] = pseudopotential.mesh_size;
                    if (mesh_truncation_sizes[type_index] % 2 == 0) {
                        mesh_truncation_sizes[type_index]--;
                    }
                }
            } else {
                mesh_truncation_sizes[type_index] = pseudopotential.mesh_size;
                if (mesh_truncation_sizes[type_index] % 2 == 0) {
                    mesh_truncation_sizes[type_index]--;
                }
            }
        }
        mesh_sizes_ = mesh_truncation_sizes;
    }
};

} // namespace q3
