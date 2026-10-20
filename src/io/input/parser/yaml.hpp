#pragma once

/**
 * YAMLInputParser — YAML/JSON manifest file parser.
 *
 * Reads ESSE-aligned YAML input files with optional include_yaml() directives.
 * Supports both inline material definitions and references to external JSON files.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "generated/q3/input.hpp"
#include "io/input/parser/init.hpp"
#include "io/input/utils.hpp"
#include "io/logger.hpp"
#include "layer_0_core/types/constants.hpp"

namespace q3 {

class YAMLInputParser : public InputParser {
  public:
    InputData parse(const std::string& filename) const override {
        InputData data;

        std::string manifest_directory = std::filesystem::path(filename).parent_path().string();
        if (manifest_directory.empty())
            manifest_directory = ".";

        YAML::Node root = YAML::LoadFile(filename);

        resolve_includes(root, manifest_directory);

        reject_unknown_keys(root, "");

        if (!root["material"])
            throw std::runtime_error("input.yml must contain 'material' key");

        const auto& material_node = root["material"];
        if (material_node.IsScalar()) {
            std::string material_path =
                detail::resolve_path(manifest_directory, material_node.as<std::string>());
            nlohmann::json material_json = detail::read_json_file(material_path);
            parse_material(material_json, data);
        } else if (material_node.IsMap()) {
            nlohmann::json material_json;

            if (material_node["lattice"]) {
                nlohmann::json lattice;
                const auto& lattice_node = material_node["lattice"];
                if (lattice_node["type"])
                    lattice["type"] = lattice_node["type"].as<std::string>();
                for (const char* length_key : {"a", "b", "c", "alpha", "beta", "gamma"}) {
                    if (lattice_node[length_key])
                        lattice[length_key] = lattice_node[length_key].as<Real>();
                }
                if (lattice_node["units"]) {
                    // units may be a plain string or {length: ..., angle: ...}.
                    if (lattice_node["units"].IsMap()) {
                        nlohmann::json units_json;
                        if (lattice_node["units"]["length"])
                            units_json["length"] =
                                lattice_node["units"]["length"].as<std::string>();
                        if (lattice_node["units"]["angle"])
                            units_json["angle"] = lattice_node["units"]["angle"].as<std::string>();
                        lattice["units"] = units_json;
                    } else {
                        lattice["units"] = lattice_node["units"].as<std::string>();
                    }
                }
                if (lattice_node["vectors"]) {
                    nlohmann::json vectors;
                    const auto& vectors_node = lattice_node["vectors"];
                    for (const char* row_key : {"a", "b", "c"}) {
                        if (vectors_node[row_key]) {
                            nlohmann::json row = nlohmann::json::array();
                            for (const auto& value : vectors_node[row_key])
                                row.push_back(value.as<Real>());
                            vectors[row_key] = row;
                        }
                    }
                    if (vectors_node["alat"])
                        vectors["alat"] = vectors_node["alat"].as<Real>();
                    if (vectors_node["units"]) {
                        /**
                         * Same string-or-{length: ...} forms as the lattice-level
                         * units key, so inline materials parse identically to
                         * file-referenced material JSON.
                         */
                        if (vectors_node["units"].IsMap()) {
                            nlohmann::json units_json;
                            if (vectors_node["units"]["length"])
                                units_json["length"] =
                                    vectors_node["units"]["length"].as<std::string>();
                            vectors["units"] = units_json;
                        } else {
                            vectors["units"] = vectors_node["units"].as<std::string>();
                        }
                    }
                    lattice["vectors"] = vectors;
                }
                material_json["lattice"] = lattice;
            }

            if (material_node["basis"]) {
                nlohmann::json basis;
                const auto& basis_node = material_node["basis"];
                if (basis_node["elements"]) {
                    nlohmann::json elements = nlohmann::json::array();
                    for (const auto& element_node : basis_node["elements"]) {
                        elements.push_back({{"id", element_node["id"].as<int>()},
                                            {"value", element_node["value"].as<std::string>()}});
                    }
                    basis["elements"] = elements;
                }
                if (basis_node["coordinates"]) {
                    nlohmann::json coordinates = nlohmann::json::array();
                    for (const auto& coordinate_node : basis_node["coordinates"]) {
                        nlohmann::json coordinate_value = nlohmann::json::array();
                        for (const auto& value : coordinate_node["value"])
                            coordinate_value.push_back(value.as<Real>());
                        coordinates.push_back(
                            {{"id", coordinate_node["id"].as<int>()}, {"value", coordinate_value}});
                    }
                    basis["coordinates"] = coordinates;
                }
                if (basis_node["units"])
                    basis["units"] = basis_node["units"].as<std::string>();
                material_json["basis"] = basis;
            }

            parse_material(material_json, data);
        } else {
            throw std::runtime_error("'material' must be a file path (string) or inline object");
        }

        if (root["model"]) {
            parse_model(root["model"], data);
        }

        if (root["method"]) {
            parse_method(root["method"], manifest_directory, data);
        }

        if (root["control"]) {
            parse_control(root["control"], data);
        }

        return data;
    }

  private:
    /**
     * Reject any key in a YAML map that the schema doesn't list for this
     * manifest level, so a misspelled or unsupported field fails fast instead
     * of silently falling back to a default (the field it was meant to
     * override never applies).
     *
     * The allow-lists are not written here: `schema_path` indexes the ones
     * generated from schemas/q3/input.json into
     * `q3::generated::input_schema`, so the schema is the single place a key
     * is added and the parser cannot drift away from it. Asking for a path
     * the schema doesn't close throws -- that is a bug in this file, not bad
     * input.
     *
     * Does nothing if node isn't a map (absent/scalar/sequence nodes have no
     * keys to check). Only the levels Q3 owns end to end are closed in the
     * schema: material is deliberately left open there, since the broader
     * ESSE material schema legitimately carries fields Q3 doesn't consume.
     */
    static void reject_unknown_keys(const YAML::Node& node, const std::string& schema_path) {
        if (!node.IsMap())
            return;
        const std::vector<std::string>& known_keys =
            generated::input_schema::allowed_keys(schema_path);
        const std::string context = describe_schema_path(schema_path);
        for (auto iterator = node.begin(); iterator != node.end(); ++iterator) {
            std::string key = iterator->first.as<std::string>();
            if (std::find(known_keys.begin(), known_keys.end(), key) == known_keys.end()) {
                throw std::runtime_error("Unknown key '" + key + "' in " + context +
                                         " -- check for a typo");
            }
        }
    }

    /**
     * How a schema path is named in an error message: the document root is
     * "manifest root", an array's items are an "... entry", and every other
     * path already reads as the dotted manifest key it is.
     */
    static std::string describe_schema_path(const std::string& schema_path) {
        if (schema_path.empty())
            return "manifest root";
        const std::string items_suffix = "[]";
        if (schema_path.size() > items_suffix.size() &&
            schema_path.compare(
                schema_path.size() - items_suffix.size(), items_suffix.size(), items_suffix) == 0)
            return schema_path.substr(0, schema_path.size() - items_suffix.size()) + " entry";
        return schema_path;
    }

    /**
     * Try to parse an include_yaml("path") directive from a scalar string.
     * Returns the path if matched, empty string otherwise.
     */
    static std::string parse_include_directive(const std::string& value) {
        const std::string prefix = "include_yaml(";
        if (value.size() < prefix.size() + 3)
            return "";
        if (value.substr(0, prefix.size()) != prefix)
            return "";
        if (value.back() != ')')
            return "";

        std::string inner = value.substr(prefix.size(), value.size() - prefix.size() - 1);
        if (inner.size() >= 2 && ((inner.front() == '"' && inner.back() == '"') ||
                                  (inner.front() == '\'' && inner.back() == '\''))) {
            inner = inner.substr(1, inner.size() - 2);
        }
        return inner;
    }

    /**
     * Recursively resolve include_yaml() directives in a YAML tree.
     */
    static void resolve_includes(YAML::Node node, const std::string& base_directory) {
        if (node.IsScalar()) {
            std::string include_path = parse_include_directive(node.as<std::string>());
            if (!include_path.empty()) {
                std::string full_path = detail::resolve_path(base_directory, include_path);
                YAML::Node included = YAML::LoadFile(full_path);
                std::string include_directory =
                    std::filesystem::path(full_path).parent_path().string();
                if (include_directory.empty())
                    include_directory = ".";
                resolve_includes(included, include_directory);
                node = included;
            }
        } else if (node.IsMap()) {
            for (auto iterator = node.begin(); iterator != node.end(); ++iterator) {
                resolve_includes(iterator->second, base_directory);
            }
        } else if (node.IsSequence()) {
            for (size_t index = 0; index < node.size(); index++) {
                resolve_includes(node[index], base_directory);
            }
        }
    }

    /**
     * Length unit ("bohr" default) from an ESSE lattice `units` entry, which
     * may be a plain string or an object like {"length": "angstrom", "angle": "degree"}.
     */
    static std::string read_lattice_length_units(const nlohmann::json& lattice_node) {
        if (lattice_node.contains("units")) {
            if (lattice_node["units"].is_string()) {
                return lattice_node["units"].get<std::string>();
            }
            if (lattice_node["units"].is_object() && lattice_node["units"].contains("length")) {
                return lattice_node["units"]["length"].get<std::string>();
            }
        }
        return "bohr";
    }

  public:
    /// Parse ESSE material JSON into InputData fields; parse_input() also reads `--material` here.
    static void parse_material(const nlohmann::json& material_json, InputData& data) {
        if (!material_json.contains("lattice"))
            throw std::runtime_error("Material JSON must contain 'lattice'");

        const auto& lattice_node = material_json["lattice"];
        std::string lattice_units = read_lattice_length_units(lattice_node);
        Real length_to_bohr =
            (lattice_units == "angstrom") ? 1.0 / constants::BOHR_RADIUS_ANGSTROM : 1.0;

        if (lattice_node.contains("vectors")) {
            /**
             * Explicit vectors: {"a": [ax,ay,az], "b": [...], "c": [...],
             *                    "alat": 1.0, "units": "angstrom"} (ESSE shape).
             */
            const auto& vectors_node = lattice_node["vectors"];
            if (!vectors_node.contains("a") || !vectors_node.contains("b") ||
                !vectors_node.contains("c"))
                throw std::runtime_error("lattice.vectors must contain 'a', 'b' and 'c' rows");
            /**
             * The annotation on the vectors themselves is more specific than the
             * enclosing lattice-level default, so it takes precedence.
             */
            std::string vector_units = vectors_node.contains("units")
                                           ? read_lattice_length_units(vectors_node)
                                           : lattice_units;
            Real vector_to_bohr =
                (vector_units == "angstrom") ? 1.0 / constants::BOHR_RADIUS_ANGSTROM : 1.0;
            Real alat_multiplier =
                vectors_node.contains("alat") ? vectors_node["alat"].get<Real>() : 1.0;

            Matrix3x3 lattice_vectors_bohr;
            const char* row_keys[3] = {"a", "b", "c"};
            for (int row = 0; row < 3; ++row) {
                const auto& row_node = vectors_node[row_keys[row]];
                if (!row_node.is_array() || row_node.size() != 3)
                    throw std::runtime_error(std::string("lattice.vectors.") + row_keys[row] +
                                             " must be an array of 3 numbers");
                for (int column = 0; column < 3; ++column) {
                    lattice_vectors_bohr[row][column] =
                        row_node[column].get<Real>() * alat_multiplier * vector_to_bohr;
                }
            }
            data.set_free_lattice_vectors(lattice_vectors_bohr);
        } else if (lattice_node.contains("type") && lattice_node.contains("a")) {
            std::string lattice_type = lattice_node["type"].get<std::string>();
            for (auto& character : lattice_type)
                character = static_cast<char>(toupper(character));
            Real lattice_parameter_a = lattice_node["a"].get<Real>();

            if (lattice_type == "FCC" || lattice_type == "BCC" || lattice_type == "SC" ||
                lattice_type == "CUB") {
                /**
                 * Cubic lattices keep the Bravais-index construction (the ESSE
                 * "a" is the primitive vector length; convert to the conventional
                 * lattice constant alat).
                 */
                if (lattice_type == "FCC") {
                    data.bravais_lattice_index = 2;
                } else if (lattice_type == "BCC") {
                    data.bravais_lattice_index = 3;
                } else {
                    data.bravais_lattice_index = 1;
                }

                data.cell_dimension_1 = lattice_parameter_a * length_to_bohr;
                if (data.bravais_lattice_index == 2) {
                    data.cell_dimension_1 *= std::sqrt(2.0);
                } else if (data.bravais_lattice_index == 3 || data.bravais_lattice_index == -3) {
                    data.cell_dimension_1 *= 2.0 / std::sqrt(3.0);
                }
            } else {
                /**
                 * General lattice from cell parameters a, b, c, alpha, beta, gamma
                 * (degrees). Missing lengths default to a; missing angles default
                 * to 90°, except gamma = 120° for HEX.
                 */
                Real lattice_parameter_b = lattice_node.contains("b")
                                               ? lattice_node["b"].get<Real>()
                                               : lattice_parameter_a;
                Real lattice_parameter_c = lattice_node.contains("c")
                                               ? lattice_node["c"].get<Real>()
                                               : lattice_parameter_a;
                Real alpha_degrees =
                    lattice_node.contains("alpha") ? lattice_node["alpha"].get<Real>() : 90.0;
                Real beta_degrees =
                    lattice_node.contains("beta") ? lattice_node["beta"].get<Real>() : 90.0;
                Real default_gamma = (lattice_type == "HEX") ? 120.0 : 90.0;
                Real gamma_degrees = lattice_node.contains("gamma")
                                         ? lattice_node["gamma"].get<Real>()
                                         : default_gamma;

                Matrix3x3 lattice_vectors_bohr =
                    get_lattice_vectors_from_cell_parameters(lattice_parameter_a * length_to_bohr,
                                                             lattice_parameter_b * length_to_bohr,
                                                             lattice_parameter_c * length_to_bohr,
                                                             alpha_degrees,
                                                             beta_degrees,
                                                             gamma_degrees);
                data.set_free_lattice_vectors(lattice_vectors_bohr);
            }
        } else {
            throw std::runtime_error("Lattice must specify either 'type' + 'a' or 'vectors'");
        }

        if (!material_json.contains("basis"))
            throw std::runtime_error("Material JSON must contain 'basis'");

        const auto& basis_node = material_json["basis"];
        if (!basis_node.contains("elements") || !basis_node.contains("coordinates"))
            throw std::runtime_error("Basis must contain 'elements' and 'coordinates'");

        const auto& elements_node = basis_node["elements"];
        const auto& coordinates_node = basis_node["coordinates"];

        if (elements_node.size() != coordinates_node.size())
            throw std::runtime_error("elements and coordinates arrays must have same length");

        std::map<std::string, int> species_map;
        for (const auto& element_node : elements_node) {
            std::string element_symbol = element_node["value"].get<std::string>();
            if (species_map.find(element_symbol) == species_map.end()) {
                int species_index = static_cast<int>(species_map.size());
                species_map[element_symbol] = species_index;
                AtomSpecies species_data;
                species_data.symbol = element_symbol;
                species_data.mass = 0.0;
                species_data.valence_charge = 0.0;
                data.species.push_back(species_data);
            }
        }
        data.number_of_species = static_cast<int>(species_map.size());

        for (size_t element_index = 0; element_index < elements_node.size(); element_index++) {
            std::string element_symbol = elements_node[element_index]["value"].get<std::string>();
            const auto& coordinate_value = coordinates_node[element_index]["value"];
            Atom atom_data;
            atom_data.species_index = species_map[element_symbol];
            atom_data.position = {coordinate_value[0].get<Real>(),
                                  coordinate_value[1].get<Real>(),
                                  coordinate_value[2].get<Real>()};
            data.atoms.push_back(atom_data);
        }
        data.number_of_atoms = static_cast<int>(data.atoms.size());

        /**
         * ESSE basis.constraints, as mat3ra-made writes them: per site id, `true`
         * where a Cartesian component may move (pw.x if_pos 1).
         */
        if (basis_node.contains("constraints")) {
            for (const auto& constraint_node : basis_node["constraints"]) {
                for (size_t atom_index = 0; atom_index < coordinates_node.size(); ++atom_index) {
                    if (coordinates_node[atom_index]["id"] != constraint_node["id"])
                        continue;
                    for (int axis = 0; axis < 3; ++axis)
                        data.atoms[atom_index].movable_components[axis] =
                            constraint_node["value"][axis].get<bool>();
                }
            }
        }
    }

  private:
    static void parse_model(const YAML::Node& model_node, InputData& data) {
        /**
         * "type"/"subtype" (e.g. "dft"/"gga") are accepted but not yet consumed:
         * DFT/Kohn-Sham is Q3's only supported model, so there is nothing to
         * branch on yet. Every example in this repo sets them, self-documenting
         * the calculation even though Q3 doesn't act on them today.
         */
        reject_unknown_keys(model_node, "model");

        // Kept as written; parse_input() lower-cases every reader's functional.
        if (model_node["functional"])
            data.exchange_correlation_functional = model_node["functional"].as<std::string>();
        if (model_node["dispersion"]) {
            std::string dispersion_name = model_node["dispersion"].as<std::string>();
            for (auto& character : dispersion_name)
                character = static_cast<char>(tolower(character));
            /**
             * Validate here (in addition to create_dispersion_calculator's own
             * run-time throw) so a typo fails at parse time, before any SCF
             * work starts -- previously an unrecognized name was silently
             * inert, running without the correction the user asked for.
             */
            if (dispersion_name != "none" && dispersion_name != "d3bj")
                throw std::runtime_error("model.dispersion must be 'none' or 'd3bj', got '" +
                                         dispersion_name + "'");
            data.dispersion_correction = dispersion_name;
        }
    }

    static void parse_method(const YAML::Node& method_node,
                             const std::string& manifest_directory,
                             InputData& data) {
        /**
         * "type" (e.g. "pseudopotential") is accepted but not yet consumed:
         * planewave-pseudopotential is Q3's only supported method.
         */
        reject_unknown_keys(method_node, "method");

        if (method_node["pseudopotentials"]) {
            const auto& pseudopotentials_node = method_node["pseudopotentials"];
            for (auto iterator = pseudopotentials_node.begin();
                 iterator != pseudopotentials_node.end();
                 ++iterator) {
                std::string element = iterator->first.as<std::string>();
                std::string pseudopotential_filename = iterator->second.as<std::string>();
                std::string pseudopotential_path =
                    detail::resolve_path(manifest_directory, pseudopotential_filename);

                bool found = false;
                for (auto& species_data : data.species) {
                    if (species_data.symbol == element) {
                        species_data.pseudopotential_file = pseudopotential_path;
                        found = true;
                    }
                }
                if (!found) {
                    log::warn(
                        "Pseudopotential specified for element '{}' but not found in material "
                        "basis",
                        element);
                }
            }

            if (!data.species.empty() && !data.species[0].pseudopotential_file.empty()) {
                data.pseudopotential_directory =
                    std::filesystem::path(data.species[0].pseudopotential_file)
                        .parent_path()
                        .string();
                if (data.pseudopotential_directory.empty())
                    data.pseudopotential_directory = ".";
            }
        }

        if (method_node["cutoffs"]) {
            const auto& cutoffs_node = method_node["cutoffs"];
            reject_unknown_keys(cutoffs_node, "method.cutoffs");
            if (cutoffs_node["wavefunction"])
                data.wavefunction_energy_cutoff = cutoffs_node["wavefunction"].as<Real>();
            if (cutoffs_node["density"])
                data.charge_density_energy_cutoff = cutoffs_node["density"].as<Real>();
        }

        YAML::Node self_consistent_field_node;
        if (method_node["self_consistent_field"]) {
            self_consistent_field_node = method_node["self_consistent_field"];
        } else if (method_node["scf"]) {
            self_consistent_field_node = method_node["scf"];
        }

        if (self_consistent_field_node) {
            /**
             * The QE abbreviation `nspin` belongs to the QE-namelist parser
             * only, so the manifest schema does not list it. Catch it before
             * the generic rejection below, to answer a mistyped manifest with
             * the key it should have used rather than a bare "unknown key" --
             * and so it can never silently run unpolarized.
             */
            if (self_consistent_field_node["nspin"])
                throw std::runtime_error(
                    "'nspin' is not a manifest key; use method.scf.number_of_spin_components");

            /**
             * Both spellings of this block share one schema definition, so
             * they share one allow-list; "method.scf" names it in errors
             * either way.
             */
            reject_unknown_keys(self_consistent_field_node, "method.scf");

            if (self_consistent_field_node["convergence_threshold"])
                data.convergence_threshold =
                    self_consistent_field_node["convergence_threshold"].as<Real>();
            if (self_consistent_field_node["max_iterations"])
                data.maximum_iterations = self_consistent_field_node["max_iterations"].as<int>();
            if (self_consistent_field_node["diagonalization"])
                data.diagonalization_method =
                    self_consistent_field_node["diagonalization"].as<std::string>();
            if (self_consistent_field_node["number_of_bands"])
                data.number_of_bands = self_consistent_field_node["number_of_bands"].as<int>();
            if (self_consistent_field_node["total_charge"])
                data.total_charge = self_consistent_field_node["total_charge"].as<Real>();

            if (self_consistent_field_node["hubbard"]) {
                const auto& hubbard_node = self_consistent_field_node["hubbard"];
                if (!hubbard_node.IsSequence())
                    throw std::runtime_error(
                        "method.scf.hubbard must be a list of {species, orbital_label, u_ev}");
                for (const auto& entry : hubbard_node) {
                    reject_unknown_keys(entry, "method.scf.hubbard[]");
                    if (!entry["species"] || !entry["orbital_label"] || !entry["u_ev"])
                        throw std::runtime_error(
                            "each method.scf.hubbard entry needs species, orbital_label, and "
                            "u_ev");
                    Real u_ev = entry["u_ev"].as<Real>();
                    data.hubbard.push_back({entry["species"].as<std::string>(),
                                            entry["orbital_label"].as<std::string>(),
                                            u_ev * 2.0 / constants::HARTREE_TO_EV});
                }
            }

            if (self_consistent_field_node["hubbard_projectors"]) {
                std::string hubbard_projector_type =
                    self_consistent_field_node["hubbard_projectors"].as<std::string>();
                if (hubbard_projector_type != "atomic" && hubbard_projector_type != "ortho-atomic")
                    throw std::runtime_error(
                        "method.scf.hubbard_projectors must be 'atomic' or 'ortho-atomic', got '" +
                        hubbard_projector_type + "'");
                data.hubbard_projector_type = hubbard_projector_type;
            }

            if (self_consistent_field_node["hubbard_fixed_occupation_iterations"]) {
                int fixed_iterations =
                    self_consistent_field_node["hubbard_fixed_occupation_iterations"].as<int>();
                if (fixed_iterations < 0)
                    throw std::runtime_error(
                        "method.scf.hubbard_fixed_occupation_iterations must be >= 0, got " +
                        std::to_string(fixed_iterations));
                data.hubbard_fixed_occupation_iterations = fixed_iterations;
            }

            if (self_consistent_field_node["hubbard_starting_eigenvalues"]) {
                const auto& seeding_node =
                    self_consistent_field_node["hubbard_starting_eigenvalues"];
                if (!seeding_node.IsSequence())
                    throw std::runtime_error("method.scf.hubbard_starting_eigenvalues must be a "
                                             "list of {species, eigenvalue, spin, value}");
                for (const auto& entry : seeding_node) {
                    reject_unknown_keys(entry, "method.scf.hubbard_starting_eigenvalues[]");
                    if (!entry["species"] || !entry["eigenvalue"] || !entry["spin"] ||
                        !entry["value"])
                        throw std::runtime_error(
                            "each method.scf.hubbard_starting_eigenvalues entry needs species, "
                            "eigenvalue, spin, and value");
                    /**
                     * 1-based on the user side, matching the drop-in
                     * `starting_ns_eigenvalue(m,ispin,ityp)` numbering so a
                     * deck translated between the two formats keeps its
                     * indices verbatim. Eigenvalues are ascending: the largest
                     * m is the most occupied orbital. Range rules live in the
                     * shared factory so both readers enforce one copy of them.
                     */
                    data.hubbard_starting_eigenvalues.push_back(make_hubbard_starting_eigenvalue(
                        entry["species"].as<std::string>(),
                        entry["eigenvalue"].as<int>(),
                        entry["spin"].as<int>(),
                        entry["value"].as<Real>(),
                        "method.scf.hubbard_starting_eigenvalues"));
                }
            }

            if (self_consistent_field_node["davidson"]) {
                const auto& davidson_node = self_consistent_field_node["davidson"];
                reject_unknown_keys(davidson_node, "method.scf.davidson");
                if (davidson_node["basis_expansion_factor"]) {
                    int factor = davidson_node["basis_expansion_factor"].as<int>();
                    if (factor < 2) {
                        throw std::runtime_error(
                            "method.scf.davidson.basis_expansion_factor must be at least 2: the "
                            "working basis has to be able to grow past the bands it is solving "
                            "for, got " +
                            std::to_string(factor));
                    }
                    data.davidson_basis_expansion_factor = factor;
                    data.davidson_basis_expansion_factor_set_by_deck = true;
                }
            }

            if (self_consistent_field_node["mixing"]) {
                const auto& mixing_node = self_consistent_field_node["mixing"];
                reject_unknown_keys(mixing_node, "method.scf.mixing");
                if (mixing_node["mode"])
                    data.mixing_mode = mixing_node["mode"].as<std::string>();
                if (mixing_node["beta"])
                    data.mixing_beta = mixing_node["beta"].as<Real>();
                if (mixing_node["ndim"])
                    data.mixing_history_size = mixing_node["ndim"].as<int>();
                if (mixing_node["preconditioning"])
                    data.mixing_preconditioning = mixing_node["preconditioning"].as<std::string>();
                if (mixing_node["q0"])
                    data.mixing_q0 = mixing_node["q0"].as<Real>();
            }

            /**
             * Collinear spin polarization:
             *   method.scf.number_of_spin_components: 2
             *   method.scf.starting_magnetization:        # per species, in [-1, 1]
             *     Fe: 0.4
             *     Cr2: -0.5                               # AFM sublattice label
             * (The QE abbreviation `nspin` is rejected above.)
             */
            if (self_consistent_field_node["number_of_spin_components"]) {
                data.number_of_spin_components =
                    self_consistent_field_node["number_of_spin_components"].as<int>();
                if (data.number_of_spin_components != 1 && data.number_of_spin_components != 2)
                    throw std::runtime_error(
                        "number_of_spin_components must be 1 (unpolarized) or 2 (collinear "
                        "LSDA), got " +
                        std::to_string(data.number_of_spin_components));
            }

            if (self_consistent_field_node["starting_magnetization"]) {
                const auto& magnetization_node =
                    self_consistent_field_node["starting_magnetization"];
                if (!magnetization_node.IsMap())
                    throw std::runtime_error(
                        "starting_magnetization must be a map of species symbol to value, "
                        "e.g. {Fe: 0.4}");
                data.starting_magnetization.assign(data.species.size(), 0.0);
                for (auto iterator = magnetization_node.begin();
                     iterator != magnetization_node.end();
                     ++iterator) {
                    std::string species_symbol = iterator->first.as<std::string>();
                    Real magnetization_value = iterator->second.as<Real>();
                    bool found = false;
                    for (size_t species_index = 0; species_index < data.species.size();
                         ++species_index) {
                        if (data.species[species_index].symbol == species_symbol) {
                            data.starting_magnetization[species_index] = magnetization_value;
                            found = true;
                        }
                    }
                    if (!found)
                        throw std::runtime_error("starting_magnetization specified for species '" +
                                                 species_symbol +
                                                 "' which is not in the material basis");
                }
            }

            /**
             * Per-atom starting moments, one per atom of the material basis in
             * order (same [-1, 1] convention):
             *   method.scf.atomic_starting_magnetization: [0.5, -0.5]
             * Lets atoms of one species start antiparallel without relabeling
             * them as separate species. Exclusive with the species-keyed map,
             * since giving both would leave the precedence implicit.
             */
            if (self_consistent_field_node["atomic_starting_magnetization"]) {
                const auto& atomic_node =
                    self_consistent_field_node["atomic_starting_magnetization"];
                if (!atomic_node.IsSequence())
                    throw std::runtime_error(
                        "atomic_starting_magnetization must be a list with one value per "
                        "atom of the material basis, e.g. [0.5, -0.5]");
                if (self_consistent_field_node["starting_magnetization"])
                    throw std::runtime_error("give either starting_magnetization (per species) or "
                                             "atomic_starting_magnetization (per atom), not both");
                if (atomic_node.size() != data.atoms.size())
                    throw std::runtime_error("atomic_starting_magnetization lists " +
                                             std::to_string(atomic_node.size()) +
                                             " values but the material basis has " +
                                             std::to_string(data.atoms.size()) + " atoms");
                data.atomic_starting_magnetization.clear();
                for (const auto& value_node : atomic_node) {
                    Real magnetization_value = value_node.as<Real>();
                    if (magnetization_value < -1.0 || magnetization_value > 1.0)
                        throw std::runtime_error(
                            "atomic_starting_magnetization values must be in [-1, 1], got " +
                            std::to_string(magnetization_value));
                    data.atomic_starting_magnetization.push_back(magnetization_value);
                }
            }

            if (self_consistent_field_node["smearing"]) {
                const auto& smearing_node = self_consistent_field_node["smearing"];
                reject_unknown_keys(smearing_node, "method.scf.smearing");
                if (smearing_node["type"])
                    data.smearing = smearing_node["type"].as<std::string>();
                if (smearing_node["degauss"])
                    data.smearing_width = smearing_node["degauss"].as<Real>();
                if (smearing_node["order"]) {
                    data.methfessel_paxton_order = smearing_node["order"].as<int>();
                    const std::string& kind = data.smearing;
                    if (kind != "methfessel-paxton" && kind != "m-p" && kind != "mp")
                        throw std::runtime_error(
                            "method.scf.smearing.order applies to Methfessel-Paxton smearing "
                            "only; smearing type is '" +
                            kind + "'");
                    if (data.methfessel_paxton_order < 1)
                        throw std::runtime_error("method.scf.smearing.order must be >= 1, got " +
                                                 std::to_string(data.methfessel_paxton_order));
                }
            }

            if (self_consistent_field_node["kgrid"]) {
                const auto& kpoint_grid_node = self_consistent_field_node["kgrid"];
                reject_unknown_keys(kpoint_grid_node, "method.scf.kgrid");
                if (kpoint_grid_node["dimensions"]) {
                    auto dimensions_node = kpoint_grid_node["dimensions"];
                    data.kpoint_grid_dimension_1 = dimensions_node[0].as<int>();
                    data.kpoint_grid_dimension_2 = dimensions_node[1].as<int>();
                    data.kpoint_grid_dimension_3 = dimensions_node[2].as<int>();
                }
                if (kpoint_grid_node["shifts"]) {
                    auto shifts_node = kpoint_grid_node["shifts"];
                    data.kpoint_grid_shift_1 = shifts_node[0].as<int>();
                    data.kpoint_grid_shift_2 = shifts_node[1].as<int>();
                    data.kpoint_grid_shift_3 = shifts_node[2].as<int>();
                }
                if (kpoint_grid_node["ibz_reduction"]) {
                    data.ibz_reduction_method = kpoint_grid_node["ibz_reduction"].as<std::string>();
                }
            }
        }

        if (method_node["band_structure"]) {
            const auto& bs_node = method_node["band_structure"];
            reject_unknown_keys(bs_node, "method.band_structure");
            if (bs_node["number_of_bands"])
                data.number_of_bands = bs_node["number_of_bands"].as<int>();
        }
    }

    /**
     * `control.interpolation` (see InterpolationSpecification for what the
     * block means on each calculation type).
     *
     * Two refusals live here rather than downstream. A block on a calculation
     * that never reads it (an SCF, or a Python-orchestrated workflow) would
     * let the run exit cleanly having built nothing -- the silently-ignored
     * control field q3.workflows.* once suffered from. And a build step
     * without `grid` cannot resolve k+b onto the mesh, so it is refused with
     * the reason instead of failing inside the neighbour arithmetic.
     */
    /// control.interpolation.<key> as a positive number; an absent key leaves `value` as it was.
    template <typename T>
    static void
    read_positive_interpolation_value(const YAML::Node& block, const char* key, T& value) {
        if (!block[key]) {
            return;
        }
        value = block[key].as<T>();
        if (!(value > T{0})) {
            throw std::runtime_error(std::string("control.interpolation.") + key +
                                     " must be positive.");
        }
    }

    /// control.interpolation.<key> as one of `choices`; an absent key leaves `value` as it was.
    static void read_interpolation_choice(const YAML::Node& block,
                                          const char* key,
                                          const std::vector<std::string>& choices,
                                          std::string& value) {
        if (!block[key]) {
            return;
        }
        value = block[key].as<std::string>();
        if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
            return;
        }
        /// "'a', 'b' or 'c'": the choices as a sentence names them.
        std::string named;
        for (std::size_t index = 0; index < choices.size(); index++) {
            named += (index == 0 ? "" : (index + 1 == choices.size() ? " or " : ", "));
            named += "'" + choices[index] + "'";
        }
        throw std::runtime_error(std::string("control.interpolation.") + key + " must be " + named +
                                 ", got '" + value + "'.");
    }

    /**
     * control.interpolation.<key> as three positive integers; an absent key leaves `mesh` as it
     * was.
     */
    static void
    read_interpolation_mesh(const YAML::Node& block, const char* key, std::array<int, 3>& mesh) {
        const YAML::Node node = block[key];
        if (!node) {
            return;
        }
        const std::string refusal = std::string("control.interpolation.") + key +
                                    " must be three positive integers [n1, n2, n3].";
        if (!node.IsSequence() || node.size() != 3) {
            throw std::runtime_error(refusal);
        }
        for (std::size_t axis = 0; axis < 3; axis++) {
            mesh[axis] = node[axis].as<int>();
            if (mesh[axis] <= 0) {
                throw std::runtime_error(refusal);
            }
        }
    }

    /// The keys an NSCF build request reads, and the keys `q3_utility interpolate` reads.
    static constexpr std::array<const char*, 6> interpolation_build_step_keys = {
        "grid",
        "bands",
        "projection_centers",
        "projection_decay_bohr",
        "trial_shells",
        "export_formats"};
    static constexpr std::array<const char*, 9> interpolation_evaluation_step_keys = {
        "dos_mesh",
        "disentanglement",
        "disentanglement_maximum_iterations",
        "disentanglement_convergence_tolerance",
        "localization",
        "localization_maximum_iterations",
        "projectability_minimum",
        "projectability_maximum",
        "frozen_window_above_fermi_ev"};

    /// What the capture is built from: the grid, the band range and the trial orbitals.
    static void parse_interpolation_build_step(const YAML::Node& interpolation_node,
                                               InterpolationSpecification& specification) {
        read_interpolation_mesh(interpolation_node, "grid", specification.grid);
        if (interpolation_node["bands"]) {
            const YAML::Node node = interpolation_node["bands"];
            if (!node.IsSequence() || node.size() != 2) {
                throw std::runtime_error(
                    "control.interpolation.bands must be [first, last], two 1-based band indices.");
            }
            specification.bands[0] = node[0].as<int>();
            specification.bands[1] = node[1].as<int>();
            if (specification.bands[0] < 1 || specification.bands[1] < specification.bands[0]) {
                throw std::runtime_error(
                    "control.interpolation.bands must be [first, last] with 1 <= first <= last, "
                    "got [" +
                    std::to_string(specification.bands[0]) + ", " +
                    std::to_string(specification.bands[1]) + "].");
            }
        }
        read_positive_interpolation_value(
            interpolation_node, "projection_decay_bohr", specification.projection_decay_bohr);
        if (interpolation_node["projection_centers"]) {
            for (const auto& center_node : interpolation_node["projection_centers"]) {
                if (!center_node.IsSequence() || center_node.size() != 3) {
                    throw std::runtime_error(
                        "control.interpolation.projection_centers: each entry must be three "
                        "crystal (fractional) coordinates.");
                }
                specification.projection_centers.emplace_back(center_node[0].as<Real>(),
                                                              center_node[1].as<Real>(),
                                                              center_node[2].as<Real>());
            }
        }
        if (interpolation_node["trial_shells"]) {
            const YAML::Node node = interpolation_node["trial_shells"];
            if (!node.IsMap()) {
                throw std::runtime_error(
                    "control.interpolation.trial_shells must map each species to a list of "
                    "0-based shell indices, e.g. {Si: [0, 2]}.");
            }
            for (const auto& entry : node) {
                const std::string species = entry.first.as<std::string>();
                if (!entry.second.IsSequence() || entry.second.size() == 0) {
                    throw std::runtime_error("control.interpolation.trial_shells." + species +
                                             " must be a non-empty list of 0-based shell indices.");
                }
                std::vector<int> shells;
                for (const auto& index_node : entry.second) {
                    const int index = index_node.as<int>();
                    if (index < 0 ||
                        std::find(shells.begin(), shells.end(), index) != shells.end()) {
                        throw std::runtime_error(
                            "control.interpolation.trial_shells." + species +
                            " must list distinct, non-negative shell indices.");
                    }
                    shells.push_back(index);
                }
                specification.trial_shells[species] = shells;
            }
        }
        if (!specification.trial_shells.empty() && !specification.projection_centers.empty()) {
            throw std::runtime_error(
                "control.interpolation.trial_shells chooses among the pseudo-atomic orbitals and "
                "projection_centers replaces them with hydrogenic ones; give one or the other.");
        }
        if (interpolation_node["export_formats"]) {
            for (const auto& format_node : interpolation_node["export_formats"]) {
                const std::string format = format_node.as<std::string>();
                if (format != "wannier90") {
                    throw std::runtime_error(
                        "control.interpolation.export_formats entries must be 'wannier90', got "
                        "'" +
                        format + "' (refused here rather than after the NSCF has run).");
                }
                specification.export_formats.push_back(format);
            }
        }
    }

    /// How the saved record is evaluated: the subspace, the gauge, and the density of states.
    static void parse_interpolation_evaluation_step(const YAML::Node& interpolation_node,
                                                    InterpolationSpecification& specification) {
        read_interpolation_mesh(interpolation_node, "dos_mesh", specification.dos_mesh);
        read_interpolation_choice(interpolation_node,
                                  "disentanglement",
                                  {"none", "projectability"},
                                  specification.disentanglement);
        if (interpolation_node["projectability_minimum"]) {
            specification.projectability_minimum =
                interpolation_node["projectability_minimum"].as<Real>();
        }
        if (interpolation_node["projectability_maximum"]) {
            specification.projectability_maximum =
                interpolation_node["projectability_maximum"].as<Real>();
        }
        if (interpolation_node["frozen_window_above_fermi_ev"]) {
            /**
             * A number in eV, or the word "auto" (the default): the classifier
             * then applies the published window, E_F + 2 eV, to metal and
             * insulator alike.
             */
            const YAML::Node node = interpolation_node["frozen_window_above_fermi_ev"];
            const std::string text = node.IsScalar() ? node.as<std::string>() : "";
            if (text == "auto") {
                specification.frozen_window_is_automatic = true;
            } else {
                try {
                    specification.frozen_window_above_fermi_ev = node.as<Real>();
                } catch (const YAML::Exception&) {
                    throw std::runtime_error(
                        "control.interpolation.frozen_window_above_fermi_ev must be a number in "
                        "electronvolts or 'auto', got '" +
                        text + "'.");
                }
                specification.frozen_window_is_automatic = false;
            }
        }
        read_positive_interpolation_value(interpolation_node,
                                          "localization_maximum_iterations",
                                          specification.localization_maximum_iterations);
        read_positive_interpolation_value(interpolation_node,
                                          "disentanglement_maximum_iterations",
                                          specification.disentanglement_maximum_iterations);
        read_positive_interpolation_value(interpolation_node,
                                          "disentanglement_convergence_tolerance",
                                          specification.disentanglement_convergence_tolerance);
        if (!(specification.projectability_minimum < specification.projectability_maximum)) {
            throw std::runtime_error("control.interpolation.projectability_minimum must be below "
                                     "projectability_maximum, got " +
                                     std::to_string(specification.projectability_minimum) +
                                     " and " +
                                     std::to_string(specification.projectability_maximum) + ".");
        }
        read_interpolation_choice(interpolation_node,
                                  "localization",
                                  {"auto", "maximal", "none"},
                                  specification.localization);
    }

    /**
     * One block, two steps, and a key written on the wrong one used to be
     * ignored in silence. `bands` cannot be honoured at evaluation because the
     * record holds only what was captured, and `disentanglement` cannot be
     * honoured at capture because the record deliberately stores the
     * untouched window. So neither can be quietly accepted where it does
     * nothing.
     */
    template <std::size_t Count>
    static void refuse_misplaced_interpolation_keys(const YAML::Node& interpolation_node,
                                                    const std::array<const char*, Count>& keys,
                                                    const std::string& calculation,
                                                    const char* where_it_belongs) {
        for (const char* key : keys) {
            if (interpolation_node[key]) {
                throw std::runtime_error(std::string("control.interpolation.") + key +
                                         " has no effect on calculation: " + calculation +
                                         ", so it is refused rather than ignored. " +
                                         where_it_belongs);
            }
        }
    }

    static void parse_interpolation(const YAML::Node& interpolation_node, InputData& data) {
        reject_unknown_keys(interpolation_node, "control.interpolation");
        InterpolationSpecification& specification = data.interpolation;

        if (!interpolation_node["method"]) {
            throw std::runtime_error(
                "control.interpolation.method is required (supported: wannier).");
        }
        specification.method = interpolation_node["method"].as<std::string>();
        if (specification.method != "wannier") {
            throw std::runtime_error("control.interpolation.method must be 'wannier', got '" +
                                     specification.method + "'.");
        }
        parse_interpolation_build_step(interpolation_node, specification);
        parse_interpolation_evaluation_step(interpolation_node, specification);

        const bool builds = data.calculation == "non_self_consistent_field";
        const bool evaluates = data.calculation == "interpolation";
        if (!builds && !evaluates) {
            throw std::runtime_error(
                "control.interpolation is set on calculation: " + data.calculation +
                ", which would ignore it. It is built by calculation: nscf "
                "(non_self_consistent_field) and evaluated by calculation: interpolation "
                "(run with q3_utility interpolate).");
        }
        if (builds) {
            refuse_misplaced_interpolation_keys(
                interpolation_node,
                interpolation_evaluation_step_keys,
                data.calculation,
                "It is read when the model is evaluated -- put it in the manifest you pass to "
                "q3_utility interpolate.");
        } else {
            refuse_misplaced_interpolation_keys(
                interpolation_node,
                interpolation_build_step_keys,
                data.calculation,
                "It is read when the record is captured -- put it in the NSCF manifest that "
                "builds the record.");
        }

        if (builds && !specification.has_grid()) {
            throw std::runtime_error(
                "control.interpolation.grid is required on the NSCF build step: the k-points "
                "arrive as an explicit control.kpoints_list, which carries no grid shape of its "
                "own, and k+b is resolved onto the grid by exact integer indexing.");
        }
    }

    static void parse_control(const YAML::Node& control_node, InputData& data) {
        reject_unknown_keys(control_node, "control");

        if (control_node["calculation"]) {
            std::string calculation_type = control_node["calculation"].as<std::string>();
            if (calculation_type == "scf")
                calculation_type = "self_consistent_field";
            if (calculation_type == "nscf")
                calculation_type = "non_self_consistent_field";
            if (calculation_type == "bands")
                calculation_type = "band_structure";
            if (calculation_type == "relax")
                calculation_type = "relaxation";
            if (calculation_type == "dos")
                calculation_type = "density_of_states";
            data.calculation = calculation_type;
        }
        if (control_node["outdir"])
            data.output_directory = control_node["outdir"].as<std::string>();
        if (control_node["prefix"])
            data.prefix = control_node["prefix"].as<std::string>();
        if (control_node["verbosity"]) {
            /**
             * Validated here rather than handed to spdlog's level parser,
             * which maps anything it does not recognize to "off" -- so a
             * misspelled verbosity would have silenced the run instead of
             * making it louder, and said nothing about why.
             */
            std::string verbosity = control_node["verbosity"].as<std::string>();
            if (verbosity != "default" && verbosity != "high")
                throw std::runtime_error("control.verbosity must be 'default' or 'high', got '" +
                                         verbosity + "'");
            data.verbosity = verbosity;
        }
        if (control_node["disable_symmetry"])
            data.disable_symmetry = control_node["disable_symmetry"].as<bool>();
        if (control_node["nonlocal_stress_derivative_method"])
            data.nonlocal_stress_derivative_method =
                control_node["nonlocal_stress_derivative_method"].as<std::string>();
        if (control_node["number_of_kpoint_pools"])
            data.number_of_kpoint_pools = control_node["number_of_kpoint_pools"].as<int>();
        if (control_node["use_deterministic_fft_planning"])
            data.use_deterministic_fft_planning =
                control_node["use_deterministic_fft_planning"].as<bool>();
        if (control_node["use_exact_wavefunction_grid"])
            data.use_exact_wavefunction_grid =
                control_node["use_exact_wavefunction_grid"].as<bool>();
        if (control_node["isolated_system_correction"]) {
            std::string isolated_system_correction =
                control_node["isolated_system_correction"].as<std::string>();
            if (isolated_system_correction != "none" &&
                isolated_system_correction != "makov-payne" &&
                isolated_system_correction != "martyna-tuckerman")
                throw std::runtime_error(
                    "control.isolated_system_correction must be 'none', 'makov-payne', or "
                    "'martyna-tuckerman', got '" +
                    isolated_system_correction + "'");
            data.isolated_system_correction = isolated_system_correction;
        }
        if (control_node["mean_field_export"]) {
            std::string mean_field_export = control_node["mean_field_export"].as<std::string>();
            if (mean_field_export != "none" && mean_field_export != "berkeleygw")
                throw std::runtime_error(
                    "control.mean_field_export must be 'none' or 'berkeleygw', got '" +
                    mean_field_export + "'");
            data.mean_field_export = mean_field_export;
        }

        /**
         * Band path specification: list of high-symmetry point labels.
         * Accepted formats:
         *   band_path: ["Γ", "X", "W", "L", "Γ", "K"]          # labels only
         *   band_path:
         *     - label: "Γ"
         *       coordinates: [0.0, 0.0, 0.0]                   # explicit fractional coords
         *     - label: "X"
         *   band_path_points_per_segment: 50
         */
        if (control_node["band_path"]) {
            const auto& band_path_node = control_node["band_path"];
            data.band_path.clear();

            if (band_path_node.IsSequence()) {
                for (const auto& vertex_node : band_path_node) {
                    std::string label;
                    Vector3D coordinates{0.0, 0.0, 0.0};
                    bool has_coordinates = false;

                    if (vertex_node.IsScalar()) {
                        // Simple label string: "Γ"
                        label = vertex_node.as<std::string>();
                    } else if (vertex_node.IsMap()) {
                        // Map with optional explicit coordinates.
                        reject_unknown_keys(vertex_node, "control.band_path[]");
                        if (vertex_node["label"])
                            label = vertex_node["label"].as<std::string>();
                        if (vertex_node["coordinates"]) {
                            has_coordinates = true;
                            const auto& coords_node = vertex_node["coordinates"];
                            coordinates.x = coords_node[0].as<Real>();
                            coordinates.y = coords_node[1].as<Real>();
                            coordinates.z = coords_node[2].as<Real>();
                        }
                    }

                    if (!label.empty()) {
                        /**
                         * The engine's band-structure flavor only needs labels
                         * (Python resolved the path into kpoints_list already);
                         * q3_utility interpolate walks the vertices itself and has
                         * no high-symmetry table, and a vertex defaulted to the
                         * origin is indistinguishable from a genuine Gamma
                         * downstream -- so the refusal has to happen here.
                         */
                        if (!has_coordinates && data.calculation == "interpolation") {
                            throw std::runtime_error(
                                "control.band_path vertex '" + label +
                                "' has no coordinates. On calculation: interpolation every "
                                "vertex needs `coordinates: [k1, k2, k3]` (crystal), or let "
                                "q3.workflows.interpolation resolve the labels into an explicit "
                                "control.kpoints_list.");
                        }
                        data.band_path.emplace_back(label, coordinates);
                    }
                }
            }
        }

        if (control_node["band_path_points_per_segment"])
            data.band_path_points_per_segment =
                control_node["band_path_points_per_segment"].as<int>();

        /**
         * Explicit k-point list for NSCF calculations.
         * Format (crystal coordinates + weight):
         *   kpoints_list:
         *     - [0.0, 0.0, 0.0, 1.0]
         *     - [0.5, 0.0, 0.0, 1.0]
         */
        if (control_node["kpoints_list"]) {
            const auto& kpoints_list_node = control_node["kpoints_list"];
            data.kpoints_list.clear();

            if (kpoints_list_node.IsSequence()) {
                for (const auto& kpoint_node : kpoints_list_node) {
                    if (kpoint_node.IsSequence() && kpoint_node.size() == 4) {
                        Vector3D coordinates{kpoint_node[0].as<Real>(),
                                             kpoint_node[1].as<Real>(),
                                             kpoint_node[2].as<Real>()};
                        Real weight = kpoint_node[3].as<Real>();
                        data.kpoints_list.emplace_back(coordinates, weight);
                    }
                }
            }
        }

        // DOS parameters
        if (control_node["dos_broadening"])
            data.dos_broadening = control_node["dos_broadening"].as<Real>();
        if (control_node["dos_energy_min"])
            data.dos_energy_min = control_node["dos_energy_min"].as<Real>();
        if (control_node["dos_energy_max"])
            data.dos_energy_max = control_node["dos_energy_max"].as<Real>();
        if (control_node["dos_energy_steps"])
            data.dos_energy_steps = control_node["dos_energy_steps"].as<int>();

        // Atomic-orbital (ortho-atomic/Löwdin) projections for NSCF runs.
        if (control_node["compute_projections"])
            data.compute_projections = control_node["compute_projections"].as<bool>();

        if (control_node["interpolation"])
            parse_interpolation(control_node["interpolation"], data);

        // Relaxation parameters
        if (control_node["relax_max_steps"])
            data.relax_maximum_steps = control_node["relax_max_steps"].as<int>();
        if (control_node["relax_force_threshold"])
            data.relax_force_threshold = control_node["relax_force_threshold"].as<Real>();
        if (control_node["relax_energy_threshold"])
            data.relax_energy_threshold = control_node["relax_energy_threshold"].as<Real>();
    }
};

} // namespace q3
