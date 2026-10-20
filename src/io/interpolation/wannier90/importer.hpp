#pragma once

/**
 * import_wannier90_files — another code's (or Q3's own) Wannier90-format
 * files into an InterpolationRecord, so the same interpolator that evaluates
 * a native record evaluates them.
 *
 * This is how the interpolation engine is validated independently of Q3's
 * plane-wave kernels: pw2wannier90's matrices go through exactly the code
 * path Q3's do, and a disagreement points at one side or the other rather
 * than at the pair. It is also how a user brings matrices from Quantum
 * ESPRESSO into Q3.
 *
 * What is imported is what the interpolation uses: the k-points and lattice
 * from `.nnkp`, the projections from `.amn`, the eigenvalues from `.eig`
 * (eV, converted to Rydberg). The overlaps are NOT imported: the projection
 * gauge never reads them, the localizer that would is not built yet, and
 * carrying them faithfully means re-deriving pw2wannier90's neighbour order
 * against Q3's own shell enumeration -- work with no consumer today. The
 * record says so by having no overlaps, and the Wannier90 exporter refuses
 * to write an empty `.mmn` for it. The Fermi energy is likewise unknown to
 * these files and left at zero.
 *
 * `grid` must be given: a `.nnkp` lists k-points without saying what grid
 * they came from, and the Wigner-Seitz construction needs the shape.
 */

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/interpolation/wannier90/reader.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"

namespace q3 {

[[nodiscard]] inline InterpolationRecord import_wannier90_files(const std::string& seedname,
                                                                const std::array<int, 3>& grid) {
    for (int axis = 0; axis < 3; axis++) {
        if (grid[axis] <= 0) {
            throw std::invalid_argument(
                "import_wannier90_files: the k-point grid must be three positive integers");
        }
    }
    const auto neighbor_list = Wannier90FormatReader::read_neighbor_list(seedname + ".nnkp");
    const auto projections = Wannier90FormatReader::read_projections(seedname + ".amn");
    const auto eigenvalues = Wannier90FormatReader::read_eigenvalues(seedname + ".eig");

    const int number_of_kpoints = projections.number_of_kpoints;
    if (eigenvalues.number_of_kpoints != number_of_kpoints) {
        throw std::runtime_error("import_wannier90_files: the .eig and .amn disagree on the "
                                 "k-point count (" +
                                 std::to_string(eigenvalues.number_of_kpoints) + " vs " +
                                 std::to_string(number_of_kpoints) + ")");
    }
    if (eigenvalues.number_of_bands != projections.number_of_bands) {
        throw std::runtime_error(
            "import_wannier90_files: the .eig and .amn disagree on the band count (" +
            std::to_string(eigenvalues.number_of_bands) + " vs " +
            std::to_string(projections.number_of_bands) +
            "); the gauge would be applied to the wrong eigenvalues");
    }
    if (static_cast<int>(neighbor_list.kpoints.size()) != number_of_kpoints) {
        throw std::runtime_error("import_wannier90_files: the .nnkp and .amn disagree on the "
                                 "k-point count (" +
                                 std::to_string(neighbor_list.kpoints.size()) + " vs " +
                                 std::to_string(number_of_kpoints) + ")");
    }
    if (grid[0] * grid[1] * grid[2] != number_of_kpoints) {
        throw std::runtime_error("import_wannier90_files: a " + std::to_string(grid[0]) + "x" +
                                 std::to_string(grid[1]) + "x" + std::to_string(grid[2]) +
                                 " grid implies " + std::to_string(grid[0] * grid[1] * grid[2]) +
                                 " k-points, but the files hold " +
                                 std::to_string(number_of_kpoints));
    }

    InterpolationRecord record;
    record.method = "wannier";
    for (int row = 0; row < 3; row++) {
        record.real_lattice_bohr[row] =
            neighbor_list.real_lattice[row] * constants::ANGSTROM_TO_BOHR;
        record.reciprocal_lattice_inverse_bohr[row] =
            neighbor_list.reciprocal_lattice[row] * constants::BOHR_RADIUS_ANGSTROM;
    }

    WannierInputs& wannier = record.wannier;
    wannier.kpoint_grid_dimensions = grid;
    wannier.kpoint_coordinates = neighbor_list.kpoints;
    wannier.number_of_bands = projections.number_of_bands;
    wannier.number_of_trial_orbitals = projections.number_of_wannier_functions;
    wannier.projection_matrices = projections.projections;
    wannier.eigenvalues_rydberg = eigenvalues.eigenvalues_rydberg;

    /// p_nk = sum_m |A_mn|^2 over the trial orbitals, same layout the builder writes.
    wannier.projectabilities.resize(static_cast<std::size_t>(number_of_kpoints));
    for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
        std::vector<Real>& row = wannier.projectabilities[kpoint_index];
        row.assign(static_cast<std::size_t>(wannier.number_of_bands), 0.0);
        const std::vector<Complex>& block = wannier.projection_matrices[kpoint_index];
        for (int band = 0; band < wannier.number_of_bands; band++) {
            for (int orbital = 0; orbital < wannier.number_of_trial_orbitals; orbital++) {
                row[band] += std::norm(block[orbital * wannier.number_of_bands + band]);
            }
        }
    }
    return record;
}

} // namespace q3
