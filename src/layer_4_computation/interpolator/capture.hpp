#pragma once

/**
 * capture_interpolation_record — the NSCF-side half of the interpolation
 * umbrella for the plane-wave representation: from a converged plane-wave
 * state to an InterpolationRecord, dispatching on the method the input asked
 * for.
 *
 * This is the only place that knows both the interpolation specification and
 * the plane-wave iterator, so the flavor that calls it stays free of plane
 * waves, k-point grids and pseudopotential types -- it hands over its
 * iterator and gets a record back, the way it hands its state to
 * build_mean_field_record(). Everything representation-specific about a
 * method's construction -- which trial orbitals, whether the pseudopotentials
 * carry augmentation charges, the topology it can run under -- is decided
 * here or in the method's own plane-wave builder.
 *
 * Refusals, each with its reason:
 *
 * - **Topology.** M_mn(k,b) couples each k-point to its neighbours, and
 *   k-point pools hold only their own k-points, so a neighbour generally
 *   lives on another rank and nothing exchanges it yet. This cannot fire
 *   today -- main_mpi.cpp constructs only SelfConsistentFieldFlavor, and the
 *   serial and GPU policies report that one process owns everything -- and is
 *   insurance on the reasoning the mean-field export states: one code path,
 *   guarded now. The condition reads world size, not this rank's share, so
 *   every rank decides alike.
 * - **Method.** Only "wannier" has a plane-wave builder today; a method the
 *   parser accepted but nothing here can build is refused by name rather than
 *   silently producing a Wannier record.
 * - **Augmentation.** Ultrasoft and PAW need the S metric in A_mn(k) and the
 *   augmentation term in M_mn(k,b); the norm-conserving path would silently
 *   omit both and write complete-looking, wrong matrices. The dispatch is on
 *   the pseudopotential set, and the ultrasoft quantum system it requires is
 *   reached by a checked downcast -- the same object the iterator factory
 *   built for this run, which the ultrasoft iterator itself downcasts to.
 */

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/factory.hpp"
#include "layer_3_representation/basis/planewave/local_orbital/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/ultrasoft/init.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"
#include "layer_4_computation/interpolator/input_data/specification.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"

namespace q3 {

namespace internal {

/// The crystal's lattice in the record's units: Cartesian bohr, and inverse bohr with the 2 pi.
inline void record_lattice(const Crystal& crystal_structure, InterpolationRecord& record) {
    const Real lattice_parameter_bohr = crystal_structure.lattice_parameter();
    for (int row = 0; row < 3; row++) {
        record.real_lattice_bohr[row] =
            crystal_structure.lattice_vectors()[row] * lattice_parameter_bohr;
        record.reciprocal_lattice_inverse_bohr[row] =
            crystal_structure.reciprocal_lattice_vectors()[row] *
            (constants::two_pi / lattice_parameter_bohr);
    }
}

/// The Wannier method's record: interface matrices on the full k-grid.
[[nodiscard]] inline WannierInputs
build_wannier_interface_record(const InterpolationSpecification& specification,
                               PlanewavePseudopotentialKohnShamSCFIterator& iterator,
                               int number_of_bands) {
    const std::array<int, 3>& grid_dimensions = specification.grid;
    /**
     * Without a range the model is built from the lowest bands, which is what
     * an isolated valence manifold needs. With one, the range says where the
     * manifold starts -- semicore states from the pseudopotential sit below
     * it -- and how many bands it holds.
     */
    const int first_band_index = specification.first_band_index();
    const int selected_bands = specification.has_band_selection()
                                   ? specification.number_of_selected_bands()
                                   : number_of_bands;
    if (specification.has_band_selection()) {
        log::info("NSCF: building the model from bands {} to {} of {}",
                  specification.bands[0],
                  specification.bands[1],
                  number_of_bands);
    }
    const Crystal& crystal_structure = iterator.crystal_structure();
    const PseudopotentialSet& pseudopotential_set = iterator.pseudopotential_set();
    std::unique_ptr<PlanewaveLocalOrbitalSet> trial_orbitals =
        create_planewave_local_orbital_set(crystal_structure,
                                           pseudopotential_set,
                                           specification.projection_centers,
                                           specification.projection_decay_bohr,
                                           specification.trial_shells);
    log::info("NSCF: building Wannier interface matrices on the {}x{}x{} grid ({} k-points) "
              "from {}",
              grid_dimensions[0],
              grid_dimensions[1],
              grid_dimensions[2],
              iterator.kpoint_grid().number_of_kpoints(),
              trial_orbitals->description());

    PlanewaveWannierCapture builder(
        crystal_structure, iterator.planewave_basis(), pseudopotential_set);
    if (pseudopotential_set.has_augmentation_charges()) {
        /**
         * The augmentation tables and the electronic state are in two
         * different objects, and the snapshot is the one to ask.
         * `iterator.quantum_system()` returns a by-value member declared as
         * the base PlanewaveKohnShamQuantumSystem, so its dynamic type is
         * always exactly that and a cast of it can never succeed, for any
         * dataset. The ultrasoft ITERATOR's snapshot is the
         * augmentation-aware quantum system.
         *
         * The snapshot carries the pseudopotential data but only the
         * electronic state of its last refresh, and the builder reads
         * wavefunctions, eigenvalues and the Fermi energy out of the same
         * object it reads ultrasoft_basis() from, so refresh it first.
         */
        iterator.refresh_quantum_system_snapshot();
        const auto* ultrasoft_system = dynamic_cast<const UltrasoftPlanewaveKohnShamQuantumSystem*>(
            &iterator.norm_conserving_quantum_system());
        if (ultrasoft_system == nullptr) {
            throw std::logic_error(
                "control.interpolation: the pseudopotentials carry augmentation charges but the "
                "quantum-system snapshot is not the ultrasoft one; the iterator factory and this "
                "builder disagree about the run's pseudopotential type.");
        }
        log::info("NSCF: ultrasoft/PAW pseudopotentials -- including the augmentation term in "
                  "M_mn(k,b) and the S metric in A_mn(k)");
        return builder.build_ultrasoft(iterator.kpoint_grid(),
                                       grid_dimensions[0],
                                       grid_dimensions[1],
                                       grid_dimensions[2],
                                       selected_bands,
                                       iterator.g_vector_indices(),
                                       *ultrasoft_system,
                                       *trial_orbitals,
                                       iterator.execution_policy(),
                                       first_band_index);
    }
    return builder.build(iterator.kpoint_grid(),
                         grid_dimensions[0],
                         grid_dimensions[1],
                         grid_dimensions[2],
                         selected_bands,
                         iterator.g_vector_indices(),
                         iterator.quantum_system(),
                         *trial_orbitals,
                         first_band_index);
}

} // namespace internal

[[nodiscard]] inline InterpolationRecord
capture_interpolation_record(const InterpolationSpecification& specification,
                             PlanewavePseudopotentialKohnShamSCFIterator& iterator,
                             int number_of_bands) {
    if (!iterator.execution_policy().one_process_owns_complete_state()) {
        throw std::runtime_error(
            "control.interpolation is not supported under MPI: building the model needs each "
            "k-point's neighbours, and k-point pools hold only their own k-points, so no rank "
            "can reach the states it would have to overlap. Nothing exchanges them across pools "
            "yet. Run the NSCF that builds the model with a serial binary; evaluating the model "
            "afterwards is cheap and serial anyway.");
    }

    InterpolationRecord record;
    record.method = specification.method;
    internal::record_lattice(iterator.crystal_structure(), record);

    if (specification.method == "wannier") {
        record.wannier =
            internal::build_wannier_interface_record(specification, iterator, number_of_bands);
        record.fermi_energy_rydberg = record.wannier.fermi_energy_rydberg;
        if (record.wannier.number_of_bands != record.wannier.number_of_trial_orbitals) {
            /**
             * Not a defect: a window wider than the orbital count is what
             * disentanglement is for, and the record deliberately keeps the
             * raw window so the choice belongs to the evaluation step. It is
             * still worth saying, because the one thing that will NOT work is
             * evaluating this record with the projection gauge alone.
             */
            log::info("NSCF: the record holds {} bands and {} trial orbitals, so it needs a "
                      "subspace choice: evaluate it with "
                      "control.interpolation.disentanglement: projectability, or export it "
                      "(export_formats: [wannier90]) for a code that disentangles. The one-shot "
                      "projection gauge on its own needs an isolated manifold with as many bands "
                      "as Wannier functions.",
                      record.wannier.number_of_bands,
                      record.wannier.number_of_trial_orbitals);
        }
        return record;
    }
    throw std::runtime_error("control.interpolation.method '" + specification.method +
                             "' has no plane-wave builder; supported: wannier.");
}

} // namespace q3
