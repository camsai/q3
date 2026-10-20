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

    return record;
}

} // namespace q3
