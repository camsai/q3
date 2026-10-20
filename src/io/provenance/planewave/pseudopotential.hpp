#pragma once

/**
 * PseudopotentialProvenanceRecord — a plane-wave provenance record whose
 * potentials are pseudopotentials, and which therefore has datasets to identify.
 *
 * Everything a reader needs in order to reproduce a run reduces, on the
 * potential side, to *which files were read*. That is a fact about this
 * representation and nothing above it, so it is contributed here: the record
 * for the quantum system is built first, and this subclass makes it a record
 * for a pseudopotential calculation.
 *
 * Reading the datasets a second time here is deliberate. The calculation is
 * over by the time this runs, and reporting on it must not be able to end it:
 * every failure below degrades one species' entry to what is still known rather
 * than propagating. An entry that admits it does not know the hash is honest;
 * an aborted run that had already computed its answer is not.
 */

#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/logger.hpp"
#include "io/provenance/planewave/init.hpp"
#include "io/pseudopotential/loader.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"
#include "utils/sha256.hpp"

namespace q3 {

/**
 * Map Q3's pseudopotential type tag to the consumer schema's spelling.
 *
 * Q3 carries the UPF-style short form ("NC", "US", "PAW"); the registry's
 * identity block uses spelled-out names. Anything unrecognized passes through
 * verbatim rather than being coerced into one of the three, so a new type shows
 * up as itself instead of silently registering as norm-conserving.
 */
[[nodiscard]] inline std::string
pseudopotential_kind_name(const std::string& pseudopotential_type) {
    if (pseudopotential_type == "NC") {
        return "norm_conserving";
    }
    if (pseudopotential_type == "US") {
        return "ultrasoft";
    }
    if (pseudopotential_type == "PAW") {
        return "projector_augmented_wave";
    }
    return pseudopotential_type;
}

class PseudopotentialProvenanceRecord : public PlanewaveProvenanceRecord {
  public:
    using PlanewaveProvenanceRecord::PlanewaveProvenanceRecord;

  protected:
    void add_representation_method_details(nlohmann::json& method) const override {
        PlanewaveProvenanceRecord::add_representation_method_details(method);
        method["pseudopotentials"] = pseudopotentials_json();
    }

  private:
    /**
     * Per-species pseudopotential identity: which file, and what is in it.
     *
     * The SHA-256 of the file contents is the identity the registry already
     * uses, and the valence electron count is what a verifier integrates a
     * submitted density against as a pre-flight check, so both are worth the
     * extra read.
     */
    [[nodiscard]] nlohmann::json pseudopotentials_json() const {
        nlohmann::json pseudopotentials = nlohmann::json::array();

        std::vector<AtomSpecies> species = input().species;
        PseudopotentialSet loaded_set;
        bool set_is_loaded = false;
        try {
            loaded_set = load_pseudopotentials(species, input().pseudopotential_directory);
            set_is_loaded = true;
        } catch (const std::exception& error) {
            log::warn(
                "Provenance: cannot re-read pseudopotentials ({}); recording file identity only.",
                error.what());
        }

        for (std::size_t species_index = 0; species_index < input().species.size();
             species_index++) {
            const AtomSpecies& one_species = input().species[species_index];
            nlohmann::json identity = species_identity_json(one_species);

            if (set_is_loaded && species_index < static_cast<std::size_t>(loaded_set.size())) {
                const Pseudopotential& pseudopotential =
                    loaded_set[static_cast<int>(species_index)];
                identity["kind"] = pseudopotential_kind_name(pseudopotential.pseudopotential_type);
                identity["valence_electron_count"] = pseudopotential.valence_charge;
            }

            pseudopotentials.push_back(identity);
        }
        return pseudopotentials;
    }

    /// One species' file identity: its dataset's name, and the hash of its contents.
    [[nodiscard]] nlohmann::json species_identity_json(const AtomSpecies& one_species) const {
        nlohmann::json identity;
        identity["element_symbol"] = one_species.symbol;
        identity["content_sha256"] = "";

        /**
         * Everything below is inside the guard, path resolution included.
         * resolve_pseudopotential_path() calls std::filesystem::exists(), which
         * throws for anything other than a plain not-found -- a stale mount or
         * a permissions change during an hours-long run reaches it as
         * "Permission denied", not as false. Outside the guard that exception
         * would escape Flavor::run() before the results file is written, and a
         * converged answer would be lost to a failure in the code that only
         * reports on it.
         */
        try {
            /**
             * The file's name, deliberately not the path it was read from.
             * Input parsers resolve pseudopotential paths against the manifest,
             * so the stored value is typically absolute and machine-specific --
             * and this block is retained, copied between machines, and
             * registered. A record that carries someone's home directory is not
             * portable, and the path is not what identifies the dataset anyway:
             * content_sha256 is, and the name is the human label beside it.
             */
            identity["source_filename"] =
                std::filesystem::path(one_species.pseudopotential_file).filename().string();

            const std::string resolved_path = resolve_pseudopotential_path(
                one_species.pseudopotential_file, input().pseudopotential_directory);
            identity["content_sha256"] = SHA256::hexadecimal_digest_of_file(resolved_path);
        } catch (const std::exception& error) {
            log::warn("Provenance: cannot identify the dataset for species '{}' ({}); "
                      "recording an empty digest.",
                      one_species.symbol,
                      error.what());
        }
        return identity;
    }
};

} // namespace q3
