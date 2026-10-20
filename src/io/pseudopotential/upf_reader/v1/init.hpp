#pragma once

/**
 * UPF v1 core reader.
 *
 * Reads the common (norm-conserving) sections of a UPF v1 format file:
 * PP_HEADER, PP_R, PP_RAB, PP_LOCAL, PP_NONLOCAL (beta projectors + D matrix),
 * PP_PSWFC (atomic wavefunctions), PP_RHOATOM, and PP_NLCC (core charge).
 *
 * After this reader runs, the Pseudopotential struct has all fields populated
 * that are common to both norm-conserving and ultrasoft pseudopotential_set.
 * Ultrasoft-specific sections (PP_QIJ) are handled by UltrasoftUPFv1Reader.
 */

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "io/logger.hpp"
#include "io/pseudopotential/upf_reader/utilities.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"

namespace q3 {

class UPFv1Reader {
  public:
    UPFv1Reader(Pseudopotential& pseudopotential, std::ifstream& file_stream)
        : pseudopotential_(pseudopotential), file_stream_(file_stream) {}

    void read() {
        read_header();
        read_mesh();
        read_local_potential();
        read_beta_projectors();
        read_d_matrix();
        read_pseudo_wavefunctions();
        read_atomic_charge_density();
        read_core_charge_density();
    }

  private:
    void read_header() {
        using namespace upf_utilities;
        find_tag(file_stream_, "<PP_HEADER>");
        std::string line;
        std::getline(file_stream_, line); // version
        std::getline(file_stream_, line);
        {
            std::istringstream stream(line);
            stream >> pseudopotential_.element;
        }
        std::getline(file_stream_, line);
        {
            std::istringstream stream(line);
            std::string token;
            stream >> token;
            std::string canonical_type = canonical_pseudopotential_type(token);
            if (canonical_type == "PAW")
                /**
                 * Reject here, before the later sections are parsed: a PAW v1 file
                 * is missing the sections this reader requires, and letting it fail
                 * there would misreport a recognized-but-unsupported type as a
                 * malformed file (or worse, run it down the NC path).
                 */
                throw UnsupportedPseudopotentialTypeError(
                    "PAW pseudopotential not supported: UPF v1 header declares type PAW. "
                    "Q3 reads PAW datasets in UPF v2 format only; convert the file to v2, "
                    "or use an NC or US file instead.");
            pseudopotential_.pseudopotential_type = canonical_type;
        }
        std::getline(file_stream_, line);
        {
            std::istringstream stream(line);
            std::string token;
            stream >> token;
            pseudopotential_.has_nonlinear_core_correction =
                (token == "T" || token == "t" || token == ".TRUE." || token == ".true.");
        }
        std::getline(file_stream_, line);
        pseudopotential_.xc_functional = trim(line.substr(0, line.find("Exchange")));
        std::getline(file_stream_, line);
        {
            std::istringstream stream(line);
            stream >> pseudopotential_.valence_charge;
        }
        std::getline(file_stream_, line); // total energy
        std::getline(file_stream_, line); // suggested cutoff
        std::getline(file_stream_, line);
        {
            std::istringstream stream(line);
            stream >> pseudopotential_.maximum_angular_momentum;
        }
        std::getline(file_stream_, line);
        {
            std::istringstream stream(line);
            stream >> pseudopotential_.mesh_size;
        }
        std::getline(file_stream_, line);
        {
            std::istringstream stream(line);
            stream >> pseudopotential_.number_of_wavefunctions >>
                pseudopotential_.number_of_projectors;
        }
        std::getline(file_stream_, line);
        for (int wavefunction_index = 0;
             wavefunction_index < pseudopotential_.number_of_wavefunctions;
             wavefunction_index++)
            std::getline(file_stream_, line);

        check_header_is_usable();
    }

    /**
     * Reject a header whose required quantities did not parse.
     *
     * v1 headers are positional, read with stream extraction rather than named
     * attributes, so a truncated or misaligned file does not fail here — the
     * extraction just leaves the field at zero and the reader carries on to
     * request zero-length grids, failing much later with nothing pointing back
     * at the header. These bounds turn that into an immediate, nameable error.
     * The v2 reader gets the same protection per-attribute through
     * required_integer_attribute / required_real_attribute.
     */
    void check_header_is_usable() {
        auto reject = [this](const std::string& detail) {
            throw std::runtime_error("Malformed UPF v1 file for element '" +
                                     pseudopotential_.element + "': " + detail +
                                     ". The PP_HEADER block is truncated or misaligned.");
        };
        if (pseudopotential_.mesh_size <= 0)
            reject("mesh size read as " + std::to_string(pseudopotential_.mesh_size));
        if (pseudopotential_.valence_charge <= 0.0)
            reject("valence charge read as " + std::to_string(pseudopotential_.valence_charge));
        if (pseudopotential_.number_of_projectors < 0)
            reject("projector count read as " +
                   std::to_string(pseudopotential_.number_of_projectors));
        if (pseudopotential_.maximum_angular_momentum < 0)
            reject("maximum angular momentum read as " +
                   std::to_string(pseudopotential_.maximum_angular_momentum));
    }

    void read_mesh() {
        using namespace upf_utilities;
        find_tag(file_stream_, "<PP_R>");
        pseudopotential_.radial_grid = read_reals(file_stream_, pseudopotential_.mesh_size);
        find_tag(file_stream_, "<PP_RAB>");
        pseudopotential_.radial_grid_derivative =
            read_reals(file_stream_, pseudopotential_.mesh_size);
    }

    void read_local_potential() {
        using namespace upf_utilities;
        find_tag(file_stream_, "<PP_LOCAL>");
        pseudopotential_.potential_local.radial_values =
            read_reals(file_stream_, pseudopotential_.mesh_size);
    }

    void read_beta_projectors() {
        using namespace upf_utilities;
        find_tag(file_stream_, "<PP_NONLOCAL>");
        pseudopotential_.projectors.resize(pseudopotential_.number_of_projectors);
        for (int beta_index = 0; beta_index < pseudopotential_.number_of_projectors; beta_index++) {
            find_tag(file_stream_, "<PP_BETA>");
            // Next line: "idx  l   Beta  L" -- parse with istringstream
            std::string beta_line;
            std::getline(file_stream_, beta_line);
            {
                std::istringstream stream(beta_line);
                int projector_index, angular_momentum;
                stream >> projector_index >> angular_momentum;
                pseudopotential_.projectors[beta_index].angular_momentum = angular_momentum;
            }
            // Next line: nr_cutoff
            std::getline(file_stream_, beta_line);
            {
                std::istringstream stream(beta_line);
                int cutoff_index;
                stream >> cutoff_index;
                pseudopotential_.projectors[beta_index].number_of_radial_cutoff_points =
                    cutoff_index;
            }
            pseudopotential_.projectors[beta_index].radial_values =
                read_reals(file_stream_, pseudopotential_.mesh_size);
            pseudopotential_.projectors[beta_index].radial_values.resize(pseudopotential_.mesh_size,
                                                                         0.0);
        }
    }

    void read_d_matrix() {
        using namespace upf_utilities;
        find_tag(file_stream_, "<PP_DIJ>");
        int number_of_d_entries;
        {
            std::string dij_line;
            std::getline(file_stream_, dij_line);
            std::istringstream stream(dij_line);
            stream >> number_of_d_entries;
        }
        pseudopotential_.d_ion_matrix.resize(
            pseudopotential_.number_of_projectors,
            std::vector<Real>(pseudopotential_.number_of_projectors, 0.0));
        for (int entry_index = 0; entry_index < number_of_d_entries; entry_index++) {
            int row_index, column_index;
            Real value;
            file_stream_ >> row_index >> column_index >> value;
            file_stream_.clear(); // In case there's trailing text
            pseudopotential_.d_ion_matrix[row_index - 1][column_index - 1] = value;
            pseudopotential_.d_ion_matrix[column_index - 1][row_index - 1] = value;
        }
    }

    void read_pseudo_wavefunctions() {
        using namespace upf_utilities;
        find_tag(file_stream_, "<PP_PSWFC>");
        pseudopotential_.wavefunctions_atomic.resize(pseudopotential_.number_of_wavefunctions);
        for (int wavefunction_index = 0;
             wavefunction_index < pseudopotential_.number_of_wavefunctions;
             wavefunction_index++) {
            std::string line;
            do {
                std::getline(file_stream_, line);
            } while (trim(line).empty());
            std::istringstream stream(line);
            std::string label;
            int angular_momentum;
            Real occupations;
            stream >> label >> angular_momentum >> occupations;
            pseudopotential_.wavefunctions_atomic[wavefunction_index].label = label;
            pseudopotential_.wavefunctions_atomic[wavefunction_index].angular_momentum =
                angular_momentum;
            pseudopotential_.wavefunctions_atomic[wavefunction_index].occupations = occupations;
            pseudopotential_.wavefunctions_atomic[wavefunction_index].radial_values =
                read_reals(file_stream_, pseudopotential_.mesh_size);
        }
    }

    void read_atomic_charge_density() {
        using namespace upf_utilities;
        find_tag(file_stream_, "<PP_RHOATOM>");
        pseudopotential_.charge_atomic.radial_values =
            read_reals(file_stream_, pseudopotential_.mesh_size);
    }

    void read_core_charge_density() {
        using namespace upf_utilities;
        if (!pseudopotential_.has_nonlinear_core_correction)
            return;

        file_stream_.clear();
        file_stream_.seekg(0);
        if (find_tag(file_stream_, "<PP_NLCC>")) {
            pseudopotential_.charge_core.radial_values =
                read_reals(file_stream_, pseudopotential_.mesh_size);
            log::info("     PP {}: NLCC core charge read ({} points)",
                      pseudopotential_.element,
                      static_cast<int>(pseudopotential_.charge_core.radial_values.size()));
        } else {
            log::warn("     PP {}: has_nlcc=true but PP_NLCC section not found",
                      pseudopotential_.element);
            pseudopotential_.has_nonlinear_core_correction = false;
        }
    }

    Pseudopotential& pseudopotential_;
    std::ifstream& file_stream_;
};

} // namespace q3
