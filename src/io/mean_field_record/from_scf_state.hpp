#pragma once

/**
 * build_mean_field_record() — populates a MeanFieldRecord from a live SCF state.
 *
 * This is the in-process half of the converter seam (see
 * mean_field_record/init.hpp): it flattens the engine's object graph -- crystal
 * structure, plane-wave basis, k-point grid, quantum system -- into the plain
 * record the format writers consume. The standalone converter builds the same
 * record from `<prefix>.save/` instead, so both paths drive identical writer
 * code.
 *
 * Copies are deliberate and cheap relative to what is about to be written: the
 * heavy data (wavefunction coefficients) is NOT copied -- it stays behind
 * ScfStateMeanFieldBandSource, which hands out the quantum system's own
 * pointers exactly as the exporters used to fetch them.
 */

#include <memory>
#include <stdexcept>
#include <vector>

#include "io/mean_field_record/init.hpp"
#include "io/mean_field_record/scf_state_projector_source.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"

namespace q3 {

/**
 * Band source backed by a live quantum system.
 *
 * Holds a reference, not a copy: the record is built and consumed inside the
 * flavor's own scope, where the SCF state outlives it. A file-backed source is
 * used when that is not true.
 */
class ScfStateMeanFieldBandSource : public MeanFieldBandSource {
  public:
    explicit ScfStateMeanFieldBandSource(const NormConservingPlanewaveKohnShamSCFState& scf_state)
        : scf_state_(scf_state) {}

    [[nodiscard]] const Complex* coefficients(int kpoint_state_index,
                                              int band_index) const override {
        return scf_state_.quantum_system().wavefunction_band_data(kpoint_state_index, band_index);
    }

  private:
    const NormConservingPlanewaveKohnShamSCFState& scf_state_;
};

/**
 * Flattens `scf_state` into a MeanFieldRecord.
 *
 * Throws if the state is missing the structural pieces every format needs,
 * rather than letting each writer discover the null pointer separately -- the
 * writers used to each repeat that check.
 */
[[nodiscard]] inline MeanFieldRecord
build_mean_field_record(const NormConservingPlanewaveKohnShamSCFState& scf_state) {
    if (!scf_state.planewave_basis || !scf_state.kpoint_grid || !scf_state.crystal_structure) {
        throw std::runtime_error(
            "build_mean_field_record: the SCF state is missing planewave_basis, kpoint_grid or "
            "crystal_structure, so no mean-field record can be built from it.");
    }

    const Crystal& crystal_structure = *scf_state.crystal_structure;
    const PlanewaveBasis& planewave_basis = *scf_state.planewave_basis;
    const KPointsGrid& kpoint_grid = *scf_state.kpoint_grid;
    const auto& quantum_system = scf_state.quantum_system();

    MeanFieldRecord record;

    record.bravais_lattice_index = crystal_structure.bravais_lattice_index();
    record.lattice_parameter = crystal_structure.lattice_parameter();
    record.two_pi_over_lattice_parameter = crystal_structure.two_pi_over_lattice_parameter();
    record.cell_volume = crystal_structure.cell_volume();
    record.lattice_vectors = crystal_structure.lattice_vectors();
    record.reciprocal_lattice_vectors = crystal_structure.reciprocal_lattice_vectors();

    int number_of_atoms = crystal_structure.number_of_atoms();
    record.atom_positions_cartesian.reserve(number_of_atoms);
    record.atomic_numbers.reserve(number_of_atoms);
    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        record.atom_positions_cartesian.push_back(
            crystal_structure.convert_from_crystal_to_cartesian(atom_index));
        record.atomic_numbers.push_back(crystal_structure.atomic_number(atom_index));
    }

    record.symmetry_operations = SymmetryAnalyzer::get_symmetry_operations(crystal_structure);

    const auto& fft_rho = planewave_basis.fft_rho();
    record.fft_grid_dimensions = {
        fft_rho.grid_dimension_1(), fft_rho.grid_dimension_2(), fft_rho.grid_dimension_3()};
    const auto& g_vectors = planewave_basis.g_vectors();
    record.charge_density_miller_indices.reserve(g_vectors.size());
    for (const auto& g_vector : g_vectors) {
        record.charge_density_miller_indices.push_back(g_vector.miller);
    }
    record.is_gamma_only = planewave_basis.is_gamma_only();
    record.charge_density_energy_cutoff = planewave_basis.charge_density_energy_cutoff();
    record.wavefunction_energy_cutoff = planewave_basis.wavefunction_energy_cutoff();

    record.kpoint_grid_dimensions = kpoint_grid.grid_dimensions();
    record.kpoint_grid_shifts = kpoint_grid.grid_shifts();
    int number_of_kpoints = kpoint_grid.number_of_kpoints();
    record.kpoint_coordinates.reserve(number_of_kpoints);
    record.kpoint_weights.reserve(number_of_kpoints);
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
        record.kpoint_coordinates.push_back(kpoint_grid[kpoint_index].coordinates);
        record.kpoint_weights.push_back(kpoint_grid[kpoint_index].weight);
    }
    const auto& plane_waves_per_kpoint = planewave_basis.kpoint_number_of_plane_waves();
    record.plane_waves_per_kpoint.assign(plane_waves_per_kpoint.begin(),
                                         plane_waves_per_kpoint.end());
    record.kpoint_g_vector_indices = planewave_basis.kpoint_g_vector_indices();

    record.number_of_bands = quantum_system.number_of_bands();
    record.number_of_spin_components = scf_state.number_of_spin_components();

    int number_of_kpoint_states = record.number_of_kpoint_states();
    record.eigenvalues_rydberg.resize(number_of_kpoint_states);
    record.occupations.resize(number_of_kpoint_states);
    for (int state_index = 0; state_index < number_of_kpoint_states; state_index++) {
        record.eigenvalues_rydberg[state_index].resize(record.number_of_bands);
        record.occupations[state_index].resize(record.number_of_bands);
        for (int band_index = 0; band_index < record.number_of_bands; band_index++) {
            record.eigenvalues_rydberg[state_index][band_index] =
                quantum_system.eigenvalue(state_index, band_index);
            record.occupations[state_index][band_index] =
                quantum_system.occupation_by_kpoint_state_index_and_band_index(state_index,
                                                                               band_index);
        }
    }

    record.band_source = std::make_shared<ScfStateMeanFieldBandSource>(scf_state);

    /**
     * The engine stores rho(r); the record carries rho(G), so the transform
     * happens here -- once, while a PlanewaveBasis is still in hand -- rather
     * than inside each format writer, where a file-backed converter could not
     * perform it. See MeanFieldRecord::charge_density_g.
     */
    record.charge_density_g.resize(scf_state.charge_density_real.size());
    for (size_t spin_index = 0; spin_index < scf_state.charge_density_real.size(); spin_index++) {
        planewave_basis.rho_r_to_g(scf_state.charge_density_real[spin_index],
                                   record.charge_density_g[spin_index]);
    }

    record.exchange_correlation_potential_g = scf_state.exchange_correlation_potential_g;

    /**
     * The per-spin copy when the state has one -- an NSCF run fills it from the
     * checkpoint's V_eff, and a spin-polarized SCF from its own -- and otherwise
     * the single potential, which is the whole story for an unpolarized run
     * (PlanewaveKohnShamSCFState::total_potential_g_by_spin).
     */
    if (!scf_state.total_potential_g_by_spin.empty()) {
        record.self_consistent_potential_g = scf_state.total_potential_g_by_spin;
    } else if (!scf_state.total_potential_g.empty()) {
        record.self_consistent_potential_g = {scf_state.total_potential_g};
    }

    return record;
}

/**
 * Adds the nonlocal pseudopotential -- species, D blocks, projector source -- to
 * a record built by build_mean_field_record(), for the BerkeleyGW `VKB` file.
 *
 * Refuses rather than writing a file that describes a different Hamiltonian
 * from the one the bands solve: an ultrasoft or PAW dataset turns H ψ = E S ψ
 * into a generalized problem whose S no `VKB` reader applies, and DFT+U adds a
 * projector term that D does not carry.
 */
inline void add_kleinman_bylander_projectors_to_mean_field_record(
    MeanFieldRecord& record, const NormConservingPlanewaveKohnShamSCFState& scf_state) {
    const auto& quantum_system = scf_state.quantum_system();
    const auto& pseudopotential_basis = quantum_system.pseudopotential_basis();
    quantum_system.require_identity_overlap(
        "mean_field_export_kleinman_bylander_projectors (BerkeleyGW VKB)");
    if (!scf_state.hubbard_manifolds.empty()) {
        throw std::runtime_error(
            "mean_field_export_kleinman_bylander_projectors is not available with DFT+U: the "
            "Hubbard term is a nonlocal potential of its own, which a BerkeleyGW VKB file "
            "cannot carry.");
    }

    const Crystal& crystal_structure = *scf_state.crystal_structure;
    int number_of_atoms = crystal_structure.number_of_atoms();
    int number_of_species = static_cast<int>(pseudopotential_basis.pseudopotential_set().size());

    record.projectors_per_species.resize(number_of_species);
    for (int species_index = 0; species_index < number_of_species; species_index++) {
        record.projectors_per_species[species_index] =
            pseudopotential_basis.number_of_projector_channels(species_index);
    }
    record.atom_species_indices.resize(number_of_atoms);
    for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
        record.atom_species_indices[atom_index] = crystal_structure.species_type_index(atom_index);
    }

    record.nonlocal_coefficients.assign(record.number_of_spin_components, {});
    for (int spin_index = 0; spin_index < record.number_of_spin_components; spin_index++) {
        auto& coefficients_by_atom = record.nonlocal_coefficients[spin_index];
        coefficients_by_atom.resize(number_of_atoms);
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            int species_index = record.atom_species_indices[atom_index];
            int number_of_channels = record.projectors_per_species[species_index];
            auto& block = coefficients_by_atom[atom_index];
            block.resize(static_cast<size_t>(number_of_channels) * number_of_channels);
            for (int channel_j = 0; channel_j < number_of_channels; channel_j++) {
                for (int channel_i = 0; channel_i < number_of_channels; channel_i++) {
                    block[static_cast<size_t>(channel_i) +
                          static_cast<size_t>(channel_j) * number_of_channels] =
                        quantum_system.d_matrix_element(
                            atom_index, species_index, channel_i, channel_j, spin_index);
                }
            }
        }
    }

    record.projector_source = std::make_shared<ScfStateMeanFieldProjectorSource>(scf_state);
}

} // namespace q3
