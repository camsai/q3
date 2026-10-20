#pragma once

/**
 * VolumetricCommand — `q3_utility volumetric <input.yml>`: a finished run's saved
 * fields as a file a visualizer reads.
 *
 * The density, the effective potential and a single band all live on the same
 * real-space grid in a run's `<prefix>.save/` directory, and until now nothing
 * could get them out of it. This turns them into a Gaussian cube, an XCrySDen
 * `.xsf` or a format an add-on registers, without rerunning anything — Q3's analogue of Quantum
 * ESPRESSO's `pp.x`, and the same shape as the other `q3_utility` subcommands: a
 * converter over saved records, never part of the SCF path.
 *
 *   q3_utility volumetric <input.yml> [--field density|potential|wavefunction]
 *                                  [--format cube|xsf|<add-on format>]
 *                                  [--spin 1|2] [--kpoint K] [--band N]
 *                                  [--outdir <dir>] [--output <file>]
 *
 * The input is the manifest the run was launched with: its `control.outdir`
 * and `control.prefix` say which save directory to read, and its cell and
 * cutoffs say what grid the saved arrays are on. That is the whole reason the
 * command takes a manifest rather than a directory — the save directory holds
 * the numbers but not the cell they belong to, and a volumetric file is
 * meaningless without one. `q3_utility interpolate` reads its input the same way
 * and for the same reason.
 *
 * Where each field comes from:
 *
 *   - `density` reads `charge-density.dat` — rho(G) — and transforms it onto
 *     the grid. That file is the one a Quantum ESPRESSO run also writes, so
 *     the same command converts a QE density (see the reader's header for the
 *     two dialects). A spin-polarized Q3 run writes the two channels summed
 *     into that one file, so `--spin 2` has nothing to select there and says
 *     so; the per-channel arrays live in `effective-potential.q3cp`, which is
 *     what `potential` reads.
 *   - `potential` reads `effective-potential.q3cp`, which stores V_eff(r)
 *     directly; no transform is involved.
 *   - `wavefunction` reads `wfc<k>.dat` and writes |psi_nk(r)|^2, normalized so
 *     that its integral over the cell is 1. The band amplitude itself is
 *     complex at a general k-point and has no single real value to plot, so
 *     the probability density is what is written — and its integral is a
 *     number worth checking, which a phase-dependent quantity would not be.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/charge_density/kohn_sham/planewave/checkpoint/qe_reader.hpp"
#include "io/input/data.hpp"
#include "io/input/parser/factory.hpp"
#include "io/input/system_builder.hpp"
#include "io/logger.hpp"
#include "io/pseudopotential/loader.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/q3.hpp"
#include "io/volumetric/factory.hpp"
#include "io/volumetric/field.hpp"
#include "io/volumetric/inverse_transform.hpp"
#include "io/wavefunction/kohn_sham/planewave/checkpoint/qe_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/grid_sizer.hpp"

namespace q3 {

class VolumetricCommand {
  public:
    static void print_usage() {
        log::error("Usage: q3_utility volumetric <input.yml> [--field <name>] [--format <name>] "
                   "[--spin <n>] [--kpoint <k>] [--band <n>] [--outdir <dir>] [--output <file>]");
        log::error("  <input.yml>   the manifest the run used; its control.outdir/prefix name the "
                   "save directory, and its cell and cutoffs the grid");
        log::error("  --field       density (default), potential, or wavefunction");
        log::error("  --format      {} (cube is the default)",
                   volumetric_field_writer_registry().canonical_names(", "));
        log::error("  --spin        spin channel, 1-based (default 1)");
        log::error("  --kpoint      which wfc<k>.dat to read, 1-based (default 1)");
        log::error("  --band        which band of it, 1-based (default 1)");
        log::error("  --outdir      where to write (default: the run's own outdir)");
        log::error("  --output      the full output path, overriding --outdir and the default "
                   "file name");
    }

    /// Returns the process exit code; argument errors print usage, everything else throws.
    static int run(const std::vector<std::string>& arguments) {
        std::string input_file;
        std::string field_name = "density";
        std::string format = "cube";
        std::string output_directory;
        std::string output_file;
        int spin_channel = 1;
        int kpoint_number = 1;
        int band_number = 1;

        for (std::size_t index = 0; index < arguments.size(); index++) {
            const std::string& argument = arguments[index];
            auto take_value = [&](const char* flag) -> std::string {
                if (index + 1 >= arguments.size())
                    throw std::runtime_error(std::string(flag) + " needs a value");
                return arguments[++index];
            };
            if (argument == "--field") {
                field_name = take_value("--field");
            } else if (argument == "--format") {
                format = take_value("--format");
            } else if (argument == "--spin") {
                spin_channel = parse_positive_integer("--spin", take_value("--spin"));
            } else if (argument == "--kpoint") {
                kpoint_number = parse_positive_integer("--kpoint", take_value("--kpoint"));
            } else if (argument == "--band") {
                band_number = parse_positive_integer("--band", take_value("--band"));
            } else if (argument == "--outdir") {
                output_directory = take_value("--outdir");
            } else if (argument == "--output") {
                output_file = take_value("--output");
            } else if (argument == "-h" || argument == "--help") {
                print_usage();
                return 0;
            } else if (argument.rfind("-", 0) == 0) {
                log::error("Unknown option '{}'", argument);
                print_usage();
                return 1;
            } else if (input_file.empty()) {
                input_file = argument;
            } else {
                log::error("Unexpected extra argument '{}'", argument);
                print_usage();
                return 1;
            }
        }
        if (input_file.empty()) {
            print_usage();
            return 1;
        }

        const InputData input = parse_input(input_file);
        const std::string save_directory = input.output_directory + "/" + input.prefix + ".save";
        if (!std::filesystem::is_directory(save_directory)) {
            throw std::runtime_error(
                "no save directory at '" + save_directory +
                "'. It is written by the run this input describes; point control.outdir and "
                "control.prefix at a run that has finished.");
        }

        const Crystal crystal = build_crystal(input);
        const std::array<int, 3> grid_dimensions = charge_density_grid_dimensions(input, crystal);
        log::info("Reading {} on a {}x{}x{} grid from {}",
                  field_name,
                  grid_dimensions[0],
                  grid_dimensions[1],
                  grid_dimensions[2],
                  save_directory);

        VolumetricField field = build_field(field_name,
                                            save_directory,
                                            crystal,
                                            grid_dimensions,
                                            spin_channel,
                                            kpoint_number,
                                            band_number);

        auto writer = create_volumetric_field_writer(format);
        std::string path = output_file;
        if (path.empty()) {
            const std::string directory =
                output_directory.empty() ? input.output_directory : output_directory;
            std::filesystem::create_directories(directory);
            path = directory + "/" + writer->default_file_name(input.prefix + "_" + field_name);
        }
        writer->write(path, field);

        log::info("Wrote {} ({}) to {}", field.description, format, path);
        /**
         * The integral is the cheapest statement that the file is what it
         * says: for a density it is the valence electron count, and for a
         * band's probability density it is 1.
         */
        log::info("  integral over the cell: {:.8f} {} (cell volume {:.6f} bohr^3)",
                  field.integral(),
                  field.is_a_density ? std::string("electrons") : field.unit + "*bohr^3",
                  field.cell_volume_bohr3());
        return 0;
    }

  private:
    static int parse_positive_integer(const char* flag, const std::string& text) {
        std::size_t consumed = 0;
        int value = 0;
        try {
            value = std::stoi(text, &consumed);
        } catch (const std::exception&) {
            consumed = 0;
        }
        if (consumed != text.size() || value <= 0)
            throw std::runtime_error(std::string(flag) + " expects a positive integer, got '" +
                                     text + "'");
        return value;
    }

    /**
     * The crystal the run used, with each species' valence charge filled in
     * from its pseudopotential.
     *
     * The valence charges only decorate the output — a cube's per-atom charge
     * column, and the electron count the log reports — so a dataset that
     * cannot be reached leaves them zero and the field is written anyway.
     * Refusing to convert a density because a pseudopotential moved would be a
     * poor trade.
     */
    static Crystal build_crystal(const InputData& input) {
        std::vector<AtomSpecies> species = input.species;
        try {
            load_pseudopotentials(species, input.pseudopotential_directory);
        } catch (const std::exception& error) {
            log::warn("Could not read this run's pseudopotentials ({}); the atoms' valence "
                      "charges will be written as zero.",
                      error.what());
            species = input.species;
        }
        return build_crystal_from_input(input, species);
    }

    /// The dense (charge-density) FFT grid, sized exactly as the run sized it.
    static std::array<int, 3> charge_density_grid_dimensions(const InputData& input,
                                                             const Crystal& crystal) {
        const Real cutoff = input.charge_density_energy_cutoff > 0.0
                                ? input.charge_density_energy_cutoff
                                : 4.0 * input.wavefunction_energy_cutoff;
        int fft_factors[3];
        FFTGridSizer::compute_fft_factors(crystal, fft_factors);
        std::array<int, 3> dimensions = {0, 0, 0};
        FFTGridSizer::calculate_grid_dimensions(cutoff,
                                                crystal.two_pi_over_lattice_parameter_squared(),
                                                crystal.lattice_vectors(),
                                                crystal.reciprocal_lattice_vectors(),
                                                dimensions[0],
                                                dimensions[1],
                                                dimensions[2],
                                                fft_factors);
        return dimensions;
    }

    /// Everything the formats share: the cell, the atoms and the grid shape.
    static VolumetricField empty_field(const Crystal& crystal,
                                       const std::array<int, 3>& grid_dimensions) {
        VolumetricField field;
        field.grid_dimensions = grid_dimensions;
        field.lattice_vectors_bohr = crystal.lattice_vectors_bohr();
        for (const auto& species : crystal.species()) {
            field.species_symbols.push_back(species.symbol);
            field.species_valence_charges.push_back(species.valence_charge);
        }
        for (int atom_index = 0; atom_index < crystal.number_of_atoms(); atom_index++) {
            field.atom_species_indices.push_back(crystal.species_type_index(atom_index));
            field.atom_positions_crystal.push_back(crystal.fractional_position(atom_index));
        }
        return field;
    }

    static VolumetricField build_field(const std::string& field_name,
                                       const std::string& save_directory,
                                       const Crystal& crystal,
                                       const std::array<int, 3>& grid_dimensions,
                                       int spin_channel,
                                       int kpoint_number,
                                       int band_number) {
        if (field_name == "density")
            return build_density(save_directory, crystal, grid_dimensions, spin_channel);
        if (field_name == "potential")
            return build_potential(save_directory, crystal, grid_dimensions, spin_channel);
        if (field_name == "wavefunction") {
            return build_band_probability_density(
                save_directory, crystal, grid_dimensions, kpoint_number, band_number);
        }
        throw std::runtime_error("Unknown field '" + field_name +
                                 "'. Supported: density, potential, wavefunction");
    }

    static VolumetricField build_density(const std::string& save_directory,
                                         const Crystal& crystal,
                                         const std::array<int, 3>& grid_dimensions,
                                         int spin_channel) {
        const std::string path = save_directory + "/charge-density.dat";
        if (!std::filesystem::exists(path))
            throw std::runtime_error("no charge density at '" + path + "'.");
        QEPlanewaveKohnShamChargeDensityReader reader;
        reader.read(path);
        if (spin_channel > reader.number_of_spin_components()) {
            throw std::runtime_error("'" + path + "' holds " +
                                     std::to_string(reader.number_of_spin_components()) +
                                     " spin channel(s); --spin " + std::to_string(spin_channel) +
                                     " asks for one that is not there.");
        }

        require_grid_holds(reader.miller_indices(), grid_dimensions, path);

        VolumetricField field = empty_field(crystal, grid_dimensions);
        field.values = real_field_from_reciprocal_space(grid_dimensions,
                                                        reader.miller_indices(),
                                                        reader.charge_density_g()[spin_channel - 1],
                                                        reader.is_gamma_only());
        field.description = "charge density";
        field.unit = "e/bohr^3";
        field.is_a_density = true;
        return field;
    }

    static VolumetricField build_potential(const std::string& save_directory,
                                           const Crystal& crystal,
                                           const std::array<int, 3>& grid_dimensions,
                                           int spin_channel) {
        const std::string path = save_directory + "/effective-potential.q3cp";
        if (!std::filesystem::exists(path))
            throw std::runtime_error("no effective potential at '" + path + "'.");
        KohnShamQuantumSystemSnapshot snapshot;
        Q3PlanewaveKohnShamQuantumSystemCheckpoint checkpoint;
        if (!checkpoint.load(path, snapshot))
            throw std::runtime_error("'" + path + "' could not be read as a Q3 checkpoint.");
        if (spin_channel > static_cast<int>(snapshot.effective_potential_real.size())) {
            throw std::runtime_error(
                "'" + path + "' holds " + std::to_string(snapshot.effective_potential_real.size()) +
                " potential channel(s); --spin " + std::to_string(spin_channel) +
                " asks for one that is not there.");
        }

        VolumetricField field = empty_field(crystal, grid_dimensions);
        field.values = snapshot.effective_potential_real[spin_channel - 1];
        require_grid_size(field, path, "effective potential");
        field.description = "effective potential";
        field.unit = "Ry";
        return field;
    }

    /**
     * |psi_nk(r)|^2, with `sum_G |c_G|^2 = 1` taken to mean the band is
     * normalized over the cell.
     *
     * `inverse_raw` gives `w(r) = sum_G c_G exp(i G·r)` directly, so the
     * amplitude is `psi(r) = w(r) / sqrt(Omega)` and the integral of |psi|^2
     * over the cell comes back as `sum_G |c_G|^2` by Parseval — 1 for a
     * normalized band, which is what `tests/python/test_volumetric_export.py`
     * checks out of the written file. The `exp(i k·r)` factor is dropped: it
     * has modulus one and cancels out of |psi|^2 exactly.
     */
    static VolumetricField build_band_probability_density(const std::string& save_directory,
                                                          const Crystal& crystal,
                                                          const std::array<int, 3>& grid_dimensions,
                                                          int kpoint_number,
                                                          int band_number) {
        const std::string path = save_directory + "/wfc" + std::to_string(kpoint_number) + ".dat";
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("no wavefunction at '" + path +
                                     "'. The save directory holds one wfc<k>.dat per k-point of "
                                     "the run's irreducible wedge.");
        }
        QEPlanewaveKohnShamWavefunctionReader reader;
        reader.read(path);
        if (band_number > reader.number_of_bands()) {
            throw std::runtime_error("'" + path + "' holds " +
                                     std::to_string(reader.number_of_bands()) +
                                     " band(s); --band " + std::to_string(band_number) +
                                     " asks for one that is not there.");
        }

        require_grid_holds(reader.miller_indices(), grid_dimensions, path);

        const std::vector<Complex> amplitude =
            complex_field_from_reciprocal_space(grid_dimensions,
                                                reader.miller_indices(),
                                                reader.band_coefficients(band_number - 1),
                                                reader.is_gamma_only());
        VolumetricField field = empty_field(crystal, grid_dimensions);
        const Real inverse_volume = 1.0 / crystal.cell_volume();
        field.values.resize(amplitude.size());
        for (std::size_t index = 0; index < amplitude.size(); index++)
            field.values[index] = std::norm(amplitude[index]) * inverse_volume;
        field.description = "band " + std::to_string(band_number) +
                            " probability density at "
                            "k-point " +
                            std::to_string(kpoint_number);
        field.unit = "e/bohr^3";
        field.is_a_density = true;
        return field;
    }

    /**
     * A saved G-vector set that does not fit the grid the input describes means
     * the two do not belong together. The transform drops what does not fit
     * rather than aliasing it, so the result would be a smooth, plausible field
     * missing its highest components -- which is why this is checked here
     * instead of left to look like a converged, slightly soft density.
     */
    static void require_grid_holds(const std::vector<IntegerVector3D>& miller_indices,
                                   const std::array<int, 3>& grid_dimensions,
                                   const std::string& path) {
        std::array<int, 3> extent = {0, 0, 0};
        for (const IntegerVector3D& miller : miller_indices) {
            extent[0] = std::max(extent[0], std::abs(miller.x));
            extent[1] = std::max(extent[1], std::abs(miller.y));
            extent[2] = std::max(extent[2], std::abs(miller.z));
        }
        for (int axis = 0; axis < 3; axis++) {
            if (extent[axis] <= grid_dimensions[axis] / 2)
                continue;
            throw std::runtime_error(
                "'" + path + "' reaches Miller index " + std::to_string(extent[axis]) +
                " along axis " + std::to_string(axis + 1) + ", which a " +
                std::to_string(grid_dimensions[0]) + "x" + std::to_string(grid_dimensions[1]) +
                "x" + std::to_string(grid_dimensions[2]) +
                " grid cannot hold. This input's cell and cutoffs do not describe the run that "
                "wrote that file.");
        }
    }

    /**
     * A saved real-space array whose length disagrees with the grid the input
     * describes means the two do not belong together, and every number written
     * from them would be silently misplaced.
     */
    static void require_grid_size(const VolumetricField& field,
                                  const std::string& path,
                                  const std::string& what) {
        if (static_cast<int>(field.values.size()) == field.number_of_grid_points())
            return;
        throw std::runtime_error(
            "the " + what + " in '" + path + "' has " + std::to_string(field.values.size()) +
            " grid points, but this input describes a " + std::to_string(field.grid_dimensions[0]) +
            "x" + std::to_string(field.grid_dimensions[1]) + "x" +
            std::to_string(field.grid_dimensions[2]) + " grid (" +
            std::to_string(field.number_of_grid_points()) +
            " points). The input and the save directory are from different runs.");
    }
};

} // namespace q3
