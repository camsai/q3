#pragma once

/**
 * Q3PlanewaveKohnShamQuantumSystemCheckpoint — Q3 native format.
 *
 * Serializes and deserializes the full SCF state (header, charge density,
 * eigenvalues, and optionally wavefunctions) to/from a binary file.
 * File format: magic number + version, then raw arrays. Version 2 added the
 * effective potential; version 3 adds the PAW on-site occupations after it,
 * which a version-2 file lacks and reads back as empty.
 *
 * Implements save/load from KohnShamQuantumSystemCheckpoint. When
 * save_wavefunctions is true, the snapshot must be a
 * PlanewaveKohnShamQuantumSystemSnapshot (dynamic_cast).
 */

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/quantum_system/kohn_sham/planewave/checkpoint/init.hpp"

namespace q3 {

class Q3PlanewaveKohnShamQuantumSystemCheckpoint : public PlanewaveKohnShamQuantumSystemCheckpoint {
  public:
    static constexpr uint32_t MAGIC = 0x51334350; // "Q3CP"
    static constexpr uint32_t VERSION = 3;

    bool save(const std::string& filename,
              const KohnShamQuantumSystemSnapshot& snapshot,
              bool save_wavefunctions = false) const override {
        std::ofstream file_stream(filename, std::ios::binary);
        if (!file_stream.is_open())
            return false;

        write_header(file_stream, snapshot, save_wavefunctions);
        write_charge_density(file_stream, snapshot);
        write_eigenvalues(file_stream, snapshot);
        write_effective_potential(file_stream, snapshot);
        write_one_center_occupations(file_stream, snapshot);

        if (save_wavefunctions) {
            const auto* planewave_snapshot =
                dynamic_cast<const PlanewaveKohnShamQuantumSystemSnapshot*>(&snapshot);
            if (planewave_snapshot) {
                write_wavefunctions(file_stream, *planewave_snapshot);
            }
        }

        return file_stream.good();
    }

    bool load(const std::string& filename, KohnShamQuantumSystemSnapshot& snapshot) const override {
        std::ifstream file_stream(filename, std::ios::binary);
        if (!file_stream.is_open())
            return false;

        uint8_t has_wavefunctions = 0;
        uint32_t file_version = 0;
        if (!read_header(file_stream, snapshot, has_wavefunctions, file_version))
            return false;

        read_charge_density(file_stream, snapshot);
        read_eigenvalues(file_stream, snapshot);

        // V_eff is present only in v2+. Leave empty for v1 — NSCF will reconstruct.
        if (file_version >= 2) {
            read_effective_potential(file_stream, snapshot);
        }
        // The PAW on-site occupations are present only in v3+; an older file leaves them empty.
        if (file_version >= 3) {
            read_one_center_occupations(file_stream, snapshot);
        }

        if (has_wavefunctions) {
            auto* planewave_snapshot =
                dynamic_cast<PlanewaveKohnShamQuantumSystemSnapshot*>(&snapshot);
            if (planewave_snapshot) {
                read_wavefunctions(file_stream, *planewave_snapshot);
            }
        }

        return file_stream.good();
    }

  private:
    template <typename T> static void write_value(std::ofstream& file_stream, const T& value) {
        file_stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T> static void read_value(std::ifstream& file_stream, T& value) {
        file_stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    static void write_header(std::ofstream& file_stream,
                             const KohnShamQuantumSystemSnapshot& snapshot,
                             bool save_wavefunctions) {
        write_value(file_stream, MAGIC);
        write_value(file_stream, VERSION);
        write_value(file_stream, snapshot.iteration);
        write_value(file_stream, snapshot.number_of_spin_components);
        write_value(file_stream, snapshot.number_of_bands);
        write_value(file_stream, snapshot.number_of_kpoint_states);
        write_value(file_stream, snapshot.total_energy);
        write_value(file_stream, snapshot.converged);
        uint8_t has_wavefunctions = save_wavefunctions ? 1 : 0;
        write_value(file_stream, has_wavefunctions);
    }

    static void write_charge_density(std::ofstream& file_stream,
                                     const KohnShamQuantumSystemSnapshot& snapshot) {
        int number_of_spin_density_components =
            static_cast<int>(snapshot.charge_density_real.size());
        write_value(file_stream, number_of_spin_density_components);
        for (const auto& density_component : snapshot.charge_density_real) {
            int number_of_grid_points = static_cast<int>(density_component.size());
            write_value(file_stream, number_of_grid_points);
            file_stream.write(reinterpret_cast<const char*>(density_component.data()),
                              number_of_grid_points * sizeof(Real));
        }
    }

    static void write_eigenvalues(std::ofstream& file_stream,
                                  const KohnShamQuantumSystemSnapshot& snapshot) {
        for (size_t kpoint_state_index = 0; kpoint_state_index < snapshot.eigenvalues.rows();
             kpoint_state_index++) {
            int number_of_bands_column = static_cast<int>(snapshot.eigenvalues.cols());
            write_value(file_stream, number_of_bands_column);
            file_stream.write(
                reinterpret_cast<const char*>(snapshot.eigenvalues[kpoint_state_index]),
                number_of_bands_column * sizeof(Real));
        }
    }

    static void write_wavefunctions(std::ofstream& file_stream,
                                    const PlanewaveKohnShamQuantumSystemSnapshot& snapshot) {
        for (const auto& kpoint_bands : snapshot.wavefunction_coefficients) {
            for (const auto& band : kpoint_bands) {
                int number_of_plane_waves = static_cast<int>(band.size());
                write_value(file_stream, number_of_plane_waves);
                file_stream.write(reinterpret_cast<const char*>(band.data()),
                                  number_of_plane_waves * sizeof(Complex));
            }
        }
    }

    static bool read_header(std::ifstream& file_stream,
                            KohnShamQuantumSystemSnapshot& snapshot,
                            uint8_t& has_wavefunctions,
                            uint32_t& file_version) {
        uint32_t magic, version;
        read_value(file_stream, magic);
        read_value(file_stream, version);
        if (magic != MAGIC || version < 1 || version > VERSION)
            return false;

        file_version = version;
        read_value(file_stream, snapshot.iteration);
        read_value(file_stream, snapshot.number_of_spin_components);
        read_value(file_stream, snapshot.number_of_bands);
        read_value(file_stream, snapshot.number_of_kpoint_states);
        read_value(file_stream, snapshot.total_energy);
        read_value(file_stream, snapshot.converged);
        read_value(file_stream, has_wavefunctions);
        return true;
    }

    static void read_charge_density(std::ifstream& file_stream,
                                    KohnShamQuantumSystemSnapshot& snapshot) {
        int number_of_spin_density_components;
        read_value(file_stream, number_of_spin_density_components);
        if (number_of_spin_density_components != snapshot.number_of_spin_components) {
            log::error(
                "Checkpoint corrupt: header declares {} spin component(s) but charge density "
                "section has {}.",
                snapshot.number_of_spin_components,
                number_of_spin_density_components);
            file_stream.setstate(std::ios::failbit);
            return;
        }
        snapshot.charge_density_real.resize(number_of_spin_density_components);
        for (auto& density_component : snapshot.charge_density_real) {
            int number_of_grid_points;
            read_value(file_stream, number_of_grid_points);
            density_component.resize(number_of_grid_points);
            file_stream.read(reinterpret_cast<char*>(density_component.data()),
                             number_of_grid_points * sizeof(Real));
        }
    }

    static void read_eigenvalues(std::ifstream& file_stream,
                                 KohnShamQuantumSystemSnapshot& snapshot) {
        snapshot.eigenvalues.resize(
            snapshot.number_of_kpoint_states, snapshot.number_of_bands, 0.0);
        for (int kpoint_state_index = 0; kpoint_state_index < snapshot.number_of_kpoint_states;
             kpoint_state_index++) {
            int number_of_bands_column;
            read_value(file_stream, number_of_bands_column);
            if (number_of_bands_column != snapshot.number_of_bands) {
                log::error("Checkpoint corrupt: header declares {} band(s) per k-point but "
                           "eigenvalue row {} has {}.",
                           snapshot.number_of_bands,
                           kpoint_state_index,
                           number_of_bands_column);
                file_stream.setstate(std::ios::failbit);
                return;
            }
            file_stream.read(reinterpret_cast<char*>(snapshot.eigenvalues[kpoint_state_index]),
                             number_of_bands_column * sizeof(Real));
        }
    }

    static void write_effective_potential(std::ofstream& file_stream,
                                          const KohnShamQuantumSystemSnapshot& snapshot) {
        int number_of_spin_components = static_cast<int>(snapshot.effective_potential_real.size());
        write_value(file_stream, number_of_spin_components);
        for (const auto& potential_component : snapshot.effective_potential_real) {
            int number_of_grid_points = static_cast<int>(potential_component.size());
            write_value(file_stream, number_of_grid_points);
            file_stream.write(reinterpret_cast<const char*>(potential_component.data()),
                              number_of_grid_points * sizeof(Real));
        }
    }

    static void read_effective_potential(std::ifstream& file_stream,
                                         KohnShamQuantumSystemSnapshot& snapshot) {
        int number_of_spin_components;
        read_value(file_stream, number_of_spin_components);
        snapshot.effective_potential_real.resize(number_of_spin_components);
        for (auto& potential_component : snapshot.effective_potential_real) {
            int number_of_grid_points;
            read_value(file_stream, number_of_grid_points);
            potential_component.resize(number_of_grid_points);
            file_stream.read(reinterpret_cast<char*>(potential_component.data()),
                             number_of_grid_points * sizeof(Real));
        }
    }

    static void write_one_center_occupations(std::ofstream& file_stream,
                                             const KohnShamQuantumSystemSnapshot& snapshot) {
        int number_of_spin_components = static_cast<int>(snapshot.one_center_occupations.size());
        write_value(file_stream, number_of_spin_components);
        for (const auto& occupations : snapshot.one_center_occupations) {
            int number_of_atoms = static_cast<int>(occupations.rows());
            int row_length = static_cast<int>(occupations.cols());
            write_value(file_stream, number_of_atoms);
            write_value(file_stream, row_length);
            file_stream.write(reinterpret_cast<const char*>(occupations.raw().data()),
                              occupations.raw().size() * sizeof(Real));
        }
    }

    static void read_one_center_occupations(std::ifstream& file_stream,
                                            KohnShamQuantumSystemSnapshot& snapshot) {
        int number_of_spin_components;
        read_value(file_stream, number_of_spin_components);
        if (!file_stream.good() || number_of_spin_components < 0 || number_of_spin_components > 2) {
            log::error("Checkpoint corrupt: the on-site occupation section declares {} spin "
                       "component(s).",
                       number_of_spin_components);
            file_stream.setstate(std::ios::failbit);
            return;
        }
        snapshot.one_center_occupations.assign(number_of_spin_components, {});
        for (auto& occupations : snapshot.one_center_occupations) {
            int number_of_atoms;
            int row_length;
            read_value(file_stream, number_of_atoms);
            read_value(file_stream, row_length);
            if (!file_stream.good() || number_of_atoms < 0 || row_length < 0) {
                log::error("Checkpoint corrupt: on-site occupations declared as {} atom(s) by "
                           "{} entries.",
                           number_of_atoms,
                           row_length);
                file_stream.setstate(std::ios::failbit);
                return;
            }
            occupations.assign(number_of_atoms, row_length, 0.0);
            file_stream.read(reinterpret_cast<char*>(occupations.raw().data()),
                             occupations.raw().size() * sizeof(Real));
        }
    }

    static void read_wavefunctions(std::ifstream& file_stream,
                                   PlanewaveKohnShamQuantumSystemSnapshot& snapshot) {
        snapshot.wavefunction_coefficients.resize(snapshot.number_of_kpoint_states);
        for (auto& kpoint_bands : snapshot.wavefunction_coefficients) {
            kpoint_bands.resize(snapshot.number_of_bands);
            for (auto& band : kpoint_bands) {
                int number_of_plane_waves;
                read_value(file_stream, number_of_plane_waves);
                band.resize(number_of_plane_waves);
                file_stream.read(reinterpret_cast<char*>(band.data()),
                                 number_of_plane_waves * sizeof(Complex));
            }
        }
    }
};

} // namespace q3
