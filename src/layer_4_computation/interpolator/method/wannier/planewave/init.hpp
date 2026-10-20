#pragma once

/**
 * PlanewaveWannierCapture — assembles a WannierInputs from
 * a converged plane-wave calculation's k-point grid and band data.
 *
 * Ties together three kernels: KPointNeighborShells (b-vectors and weights
 * for the grid), PlanewaveBlochOverlapCalculator (M_mn(k,b)), and
 * PlanewaveTrialOrbitalProjectionCalculator (A_mn(k) and, from it, the
 * per-band projectabilities p_nk = sum_m |A_mn(k)|^2 that projectability
 * disentanglement needs). None of those kernels know about each other or
 * about WannierInputs's layout; this class is the seam that does.
 *
 * The trial orbitals arrive as a PlanewaveLocalOrbitalSet -- the
 * pseudopotentials' atomic wavefunctions, hydrogenic orbitals at explicit
 * centres, or anything else -- so there is ONE assembly path for every
 * source of orbitals. build() covers norm-conserving pseudopotentials
 * (S = I); build_ultrasoft() adds the ultrasoft/PAW augmentation term,
 * threading one set of per-k-point Kleinman-Bylander projectors and
 * OverlapOperators through both the S-metric projections and the Bloch
 * overlap's augmentation contribution. Both entry points hand the same
 * private assembly the same arguments plus, for ultrasoft, an augmentation
 * context; nothing is written twice.
 *
 * The builder refuses, rather than approximates, in the cases where it
 * would otherwise produce plausible-looking but wrong matrices: a
 * spin-polarized quantum system (the record has no per-spin layout and
 * would label one channel's matrices as the whole system), Gamma-only
 * storage (half the G-sphere; the overlap kernel would silently see half of
 * every state), a k-point set that is not the full grid (k+b is resolved
 * onto another grid point by exact integer indexing), and a trial set that
 * is empty or larger than the band count (the gauge is an SVD of the
 * bands-by-orbitals block and cannot have more Wannier functions than
 * bands). Putting these here rather than at the call site means every
 * caller -- the NSCF hook, a test, a future SCF-side hook -- gets them.
 *
 * Takes the plane-wave state as explicit arguments (crystal, basis,
 * pseudopotential set, k-point grid, per-k-point G-vector lists, and a
 * PlanewaveKohnShamQuantumSystem for wavefunctions/eigenvalues/Fermi energy)
 * rather than a live SCF/NSCF iterator reference, so it is usable both from
 * the NSCF hook and from a unit test that builds a standalone quantum
 * system directly.
 *
 * Spin-unpolarized only for now: kpoint_state_index is taken equal to the
 * physical kpoint_index throughout, which the spin refusal above guarantees.
 */

#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoint_neighbor_shells.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/internal/miller_index_lookup.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/init.hpp"
#include "layer_3_representation/basis/planewave/pseudopotential/ultrasoft/init.hpp"
#include "layer_3_representation/pseudopotential/planewave/internal/projector.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/pseudopotential/factory.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/internal/overlaps.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/internal/projections.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"

namespace q3 {

class PlanewaveWannierCapture {
  public:
    PlanewaveWannierCapture(const Crystal& crystal_structure,
                            const PlanewaveBasis& planewave_basis,
                            const PseudopotentialSet& pseudopotential_set)
        : crystal_structure_(crystal_structure)
        , planewave_basis_(planewave_basis)
        , pseudopotential_set_(pseudopotential_set) {
        global_miller_lookup_.build(planewave_basis.g_vectors());
    }

    /**
     * Norm-conserving pseudopotentials, S = I.
     *
     * `g_vector_indices_by_kpoint[kpoint_index]` and `quantum_system` must
     * come from the same calculation -- the G-vector list a k-point's
     * wavefunction coefficients are stored against, not a fresh
     * recomputation, exactly as the kernels already require of their own
     * callers. `quantum_system` must hold `number_of_bands` bands at every
     * one of `kpoint_grid`'s physical k-points.
     */
    [[nodiscard]] WannierInputs
    build(const KPointsGrid& kpoint_grid,
          int grid_dimension_x,
          int grid_dimension_y,
          int grid_dimension_z,
          int number_of_bands,
          const std::vector<std::vector<int>>& g_vector_indices_by_kpoint,
          const PlanewaveKohnShamQuantumSystem& quantum_system,
          const PlanewaveLocalOrbitalSet& trial_orbitals,
          int first_band_index = 0) const {
        return assemble(kpoint_grid,
                        {grid_dimension_x, grid_dimension_y, grid_dimension_z},
                        number_of_bands,
                        g_vector_indices_by_kpoint,
                        quantum_system,
                        trial_orbitals,
                        nullptr,
                        first_band_index);
    }

    /**
     * Ultrasoft/PAW: the same record with the augmentation term in M_mn(k,b)
     * and the S metric in A_mn(k). `quantum_system` must have been through
     * UltrasoftPlanewaveKohnShamQuantumSystem::init() so ultrasoft_basis()
     * is populated.
     */
    [[nodiscard]] WannierInputs
    build_ultrasoft(const KPointsGrid& kpoint_grid,
                    int grid_dimension_x,
                    int grid_dimension_y,
                    int grid_dimension_z,
                    int number_of_bands,
                    const std::vector<std::vector<int>>& g_vector_indices_by_kpoint,
                    const UltrasoftPlanewaveKohnShamQuantumSystem& quantum_system,
                    const PlanewaveLocalOrbitalSet& trial_orbitals,
                    const execution::PlanewaveExecutionPolicy& execution_policy,
                    int first_band_index = 0) const {
        const int number_of_kpoints = kpoint_grid.number_of_kpoints();
        if (static_cast<int>(g_vector_indices_by_kpoint.size()) != number_of_kpoints) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture::build_ultrasoft: "
                "g_vector_indices_by_kpoint must have one entry per physical k-point.");
        }

        /**
         * Both consumers (the projections' overlap operator and the Bloch
         * overlap's augmentation term) need every k-point's Kleinman-Bylander
         * projector matrix -- a (k, neighbour) pair needs the neighbour's
         * too -- so compute the whole set once up front rather than once
         * for a k-point's own turn and again whenever another k-point's
         * iteration needs it as a neighbour.
         */
        AugmentationContext augmentation{quantum_system.ultrasoft_basis(),
                                         quantum_system,
                                         execution_policy,
                                         std::vector<Matrix<Complex>>(number_of_kpoints)};
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            compute_kleinman_bylander_projectors(
                kpoint_grid[kpoint_index].coordinates,
                crystal_structure_,
                planewave_basis_,
                pseudopotential_set_,
                augmentation.ultrasoft_basis.beta_function_table(),
                g_vector_indices_by_kpoint[kpoint_index],
                static_cast<int>(g_vector_indices_by_kpoint[kpoint_index].size()),
                augmentation.ultrasoft_basis.total_number_of_projectors(),
                augmentation.kb_projectors_by_kpoint[kpoint_index]);
        }

        return assemble(kpoint_grid,
                        {grid_dimension_x, grid_dimension_y, grid_dimension_z},
                        number_of_bands,
                        g_vector_indices_by_kpoint,
                        quantum_system,
                        trial_orbitals,
                        &augmentation,
                        first_band_index);
    }

  private:
    /**
     * Where each trial orbital was placed, Cartesian in bohr: the frame and
     * unit of the b-vectors, so the localiser can fix every diagonal phase's
     * branch from a position that is known rather than from a gauge that is
     * not yet trusted. The orbital set is the only thing that knows this.
     */
    [[nodiscard]] std::vector<Vector3D>
    trial_orbital_centers_bohr(const PlanewaveLocalOrbitalSet& trial_orbitals) const {
        /// r = f1 a1 + f2 a2 + f3 a3 is r = A^T f for a lattice whose rows are a1, a2, a3.
        const Matrix3x3 cartesian_from_crystal =
            crystal_structure_.lattice_vectors_bohr().transpose();
        std::vector<Vector3D> centers_bohr;
        for (const Vector3D& center_crystal : trial_orbitals.centers_crystal()) {
            centers_bohr.push_back(cartesian_from_crystal * center_crystal);
        }
        return centers_bohr;
    }

    /// What the ultrasoft/PAW path carries beyond the norm-conserving one.
    struct AugmentationContext {
        const UltrasoftPlanewaveBasis& ultrasoft_basis;
        const UltrasoftPlanewaveKohnShamQuantumSystem& quantum_system;
        const execution::PlanewaveExecutionPolicy& execution_policy;
        std::vector<Matrix<Complex>> kb_projectors_by_kpoint;
    };

    [[nodiscard]] WannierInputs
    assemble(const KPointsGrid& kpoint_grid,
             std::array<int, 3> grid_dimensions,
             int number_of_bands,
             const std::vector<std::vector<int>>& g_vector_indices_by_kpoint,
             const PlanewaveKohnShamQuantumSystem& quantum_system,
             const PlanewaveLocalOrbitalSet& trial_orbitals,
             const AugmentationContext* augmentation,
             int first_band_index) const {
        const int number_of_kpoints = kpoint_grid.number_of_kpoints();
        if (static_cast<int>(g_vector_indices_by_kpoint.size()) != number_of_kpoints) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture: g_vector_indices_by_kpoint must have one entry "
                "per physical k-point.");
        }
        if (quantum_system.number_of_spin_components != 1) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture does not support spin-polarized systems yet: the "
                "interface record has no per-spin layout, so the matrices would describe one spin "
                "channel while claiming to describe the system.");
        }
        if (planewave_basis_.is_gamma_only()) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture does not support Gamma-only storage: it keeps "
                "half the G-sphere and reconstructs the rest by conjugation, which the overlap "
                "kernel does not undo, so every overlap would be taken over half a state.");
        }
        const int expected_kpoints = grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2];
        if (number_of_kpoints != expected_kpoints) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture needs the FULL, unreduced " +
                std::to_string(grid_dimensions[0]) + "x" + std::to_string(grid_dimensions[1]) +
                "x" + std::to_string(grid_dimensions[2]) + " grid (" +
                std::to_string(expected_kpoints) + " k-points) but was given " +
                std::to_string(number_of_kpoints) +
                ": k+b is resolved onto another grid point by exact integer indexing, which a "
                "symmetry-reduced set cannot satisfy.");
        }
        const int number_of_trial_orbitals = trial_orbitals.number_of_orbitals();
        if (number_of_trial_orbitals == 0) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture: the trial orbital set is empty (" +
                trial_orbitals.description() + ").");
        }
        if (first_band_index < 0) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture: the first band index must not be negative.");
        }
        if (first_band_index + number_of_bands > quantum_system.number_of_bands()) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture: bands " + std::to_string(first_band_index + 1) + " to " +
                std::to_string(first_band_index + number_of_bands) +
                " were asked for but the calculation computed only " +
                std::to_string(quantum_system.number_of_bands()) +
                ". Raise the run's number_of_bands to at least the range's last band.");
        }
        if (number_of_trial_orbitals > number_of_bands) {
            throw std::invalid_argument(
                "PlanewaveWannierCapture: " + trial_orbitals.description() +
                " were given but only " + std::to_string(number_of_bands) +
                " bands are available; the gauge cannot have more Wannier functions than bands. "
                "Either request at least " +
                std::to_string(number_of_trial_orbitals) +
                " bands or choose fewer trial orbitals (control.interpolation.projection_centers "
                "selects explicit centres in place of the pseudo-atomic set).");
        }

        WannierInputs record;
        record.kpoint_grid_dimensions = grid_dimensions;
        record.kpoint_coordinates.reserve(number_of_kpoints);
        for (const KPoint& kpoint : kpoint_grid.kpoints()) {
            record.kpoint_coordinates.push_back(kpoint.coordinates);
        }

        KPointNeighborShells shells;
        shells.compute(crystal_structure_.reciprocal_vectors_bohr(),
                       grid_dimensions[0],
                       grid_dimensions[1],
                       grid_dimensions[2]);
        record.neighbor_directions = shells.neighbor_directions();
        const int number_of_directions = static_cast<int>(record.neighbor_directions.size());

        record.number_of_bands = number_of_bands;
        record.number_of_trial_orbitals = number_of_trial_orbitals;
        record.trial_orbital_centers_bohr = trial_orbital_centers_bohr(trial_orbitals);
        record.fermi_energy_rydberg = quantum_system.fermi_energy();
        record.number_of_spin_components = 1;
        record.eigenvalues_rydberg.assign(number_of_kpoints,
                                          std::vector<Real>(number_of_bands, 0.0));
        record.projectabilities.assign(number_of_kpoints, std::vector<Real>(number_of_bands, 0.0));
        record.projection_matrices.assign(number_of_kpoints, {});
        record.overlap_matrices.assign(number_of_kpoints, {});

        PlanewaveTrialOrbitalProjectionCalculator projection_calculator(trial_orbitals);
        PlanewaveBlochOverlapCalculator bloch_overlap_calculator(planewave_basis_,
                                                                 global_miller_lookup_);

        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                record.eigenvalues_rydberg[kpoint_index][band_index] =
                    quantum_system.eigenvalue(kpoint_index, first_band_index + band_index);
            }

            // ── A_mn(k) and the projectabilities it defines ─────────────
            std::unique_ptr<OverlapOperator> overlap_operator;
            if (augmentation != nullptr) {
                overlap_operator =
                    create_overlap_operator(crystal_structure_,
                                            augmentation->quantum_system,
                                            augmentation->kb_projectors_by_kpoint[kpoint_index],
                                            augmentation->execution_policy);
            }
            std::vector<std::vector<Complex>> projections =
                projection_calculator.compute_projections_at_kpoint(
                    kpoint_index,
                    kpoint_index,
                    static_cast<int>(g_vector_indices_by_kpoint[kpoint_index].size()),
                    number_of_bands,
                    g_vector_indices_by_kpoint[kpoint_index],
                    crystal_structure_,
                    kpoint_grid,
                    planewave_basis_,
                    quantum_system,
                    overlap_operator.get(),
                    first_band_index);

            std::vector<Complex>& packed_projections = record.projection_matrices[kpoint_index];
            packed_projections.assign(static_cast<std::size_t>(number_of_bands) *
                                          number_of_trial_orbitals,
                                      Complex(0.0, 0.0));
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                Real projectability = 0.0;
                for (int orbital_index = 0; orbital_index < number_of_trial_orbitals;
                     orbital_index++) {
                    const Complex amplitude = projections[band_index][orbital_index];
                    packed_projections[static_cast<std::size_t>(band_index) +
                                       static_cast<std::size_t>(orbital_index) * number_of_bands] =
                        amplitude;
                    projectability += std::norm(amplitude);
                }
                record.projectabilities[kpoint_index][band_index] = projectability;
            }

            // ── M_mn(k,b) over every neighbour direction ─────────────────
            std::vector<std::vector<Complex>>& overlaps_at_kpoint =
                record.overlap_matrices[kpoint_index];
            overlaps_at_kpoint.resize(number_of_directions);
            for (int direction_index = 0; direction_index < number_of_directions;
                 direction_index++) {
                const KPointNeighborDirection& direction =
                    record.neighbor_directions[direction_index];
                const ResolvedKPointNeighbor resolved =
                    KPointNeighborShells::resolve_neighbor(kpoint_index,
                                                           direction.grid_step,
                                                           record.kpoint_grid_dimensions,
                                                           kpoint_grid.kpoints());
                const int neighbor_kpoint_index = resolved.neighbor_kpoint_index;
                auto band_at_kpoint = [&](int band_index) {
                    return quantum_system.wavefunction_band_data(kpoint_index,
                                                                 first_band_index + band_index);
                };
                auto band_at_neighbor = [&](int band_index) {
                    return quantum_system.wavefunction_band_data(neighbor_kpoint_index,
                                                                 first_band_index + band_index);
                };

                BlochOverlapResult result = bloch_overlap_calculator.compute(
                    g_vector_indices_by_kpoint[kpoint_index],
                    number_of_bands,
                    band_at_kpoint,
                    g_vector_indices_by_kpoint[neighbor_kpoint_index],
                    number_of_bands,
                    band_at_neighbor,
                    resolved.periodic_image_shift);

                if (augmentation != nullptr) {
                    bloch_overlap_calculator.add_augmentation_contribution(
                        result,
                        crystal_structure_,
                        augmentation->ultrasoft_basis,
                        augmentation->kb_projectors_by_kpoint[kpoint_index],
                        static_cast<int>(g_vector_indices_by_kpoint[kpoint_index].size()),
                        number_of_bands,
                        band_at_kpoint,
                        augmentation->kb_projectors_by_kpoint[neighbor_kpoint_index],
                        static_cast<int>(g_vector_indices_by_kpoint[neighbor_kpoint_index].size()),
                        number_of_bands,
                        band_at_neighbor,
                        direction.reciprocal_lattice_translation);
                }
                overlaps_at_kpoint[direction_index] = std::move(result.values);
            }
        }

        return record;
    }

    const Crystal& crystal_structure_;
    const PlanewaveBasis& planewave_basis_;
    const PseudopotentialSet& pseudopotential_set_;
    MillerIndexLookup global_miller_lookup_;
};

} // namespace q3
