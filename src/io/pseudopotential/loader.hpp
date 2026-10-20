#pragma once

/**
 * Pseudopotential loader — resolves paths and reads dataset files.
 *
 * Provides a single entry point that, given a list of species and a
 * pseudopotential directory, returns a fully populated PseudopotentialSet.
 * Also patches each species' valence_charge from the loaded PP data, and
 * reports the functional the datasets were generated with for an input that
 * names none.
 */

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/pseudopotential/factory.hpp"
#include "io/pseudopotential/functional_label.hpp"
#include "layer_1_structure/crystal/atom_species.hpp"
#include "layer_3_representation/pseudopotential/set.hpp"

namespace q3 {

/**
 * Resolve the full filesystem path for a pseudopotential file.
 *
 * Handles three cases:
 *   1. Absolute path — use as-is
 *   2. Already prepended with the PP directory — use as-is
 *   3. Relative path — prepend the PP directory
 */
inline std::string resolve_pseudopotential_path(const std::string& pseudopotential_file,
                                                const std::string& pseudopotential_directory) {

    if (std::filesystem::path(pseudopotential_file).is_absolute()) {
        return pseudopotential_file;
    }

    bool already_prepended = false;
    if (!pseudopotential_directory.empty() && pseudopotential_directory != "." &&
        pseudopotential_directory != "./") {
        std::string normalized_directory = pseudopotential_directory;
        if (normalized_directory.back() == '/') {
            normalized_directory.pop_back();
        }
        if (pseudopotential_file.rfind(normalized_directory, 0) == 0) {
            already_prepended = true;
        }
    }

    if (already_prepended || std::filesystem::exists(pseudopotential_file)) {
        return pseudopotential_file;
    }

    std::string full_path = pseudopotential_directory;
    if (!full_path.empty() && full_path.back() != '/') {
        full_path += '/';
    }
    full_path += pseudopotential_file;
    return full_path;
}

/**
 * Which dataset inside its file a species is.
 *
 * An input format may point every species at one file of concatenated
 * datasets, joined in species order. So the Nth species naming a given path
 * takes the Nth dataset in it, which needs no extra field on AtomSpecies.
 * Every other format names one
 * file per species, so this stays 0 and nothing changes.
 */
inline int dataset_index_within_file(const std::vector<AtomSpecies>& species,
                                     int species_index,
                                     const std::string& pseudopotential_directory) {
    const std::string pseudopotential_path = resolve_pseudopotential_path(
        species[species_index].pseudopotential_file, pseudopotential_directory);
    int dataset_index = 0;
    for (int earlier = 0; earlier < species_index; ++earlier) {
        if (resolve_pseudopotential_path(species[earlier].pseudopotential_file,
                                         pseudopotential_directory) == pseudopotential_path)
            dataset_index++;
    }
    return dataset_index;
}

/**
 * The functional the pseudopotentials were generated with: what a run whose
 * input names no functional uses.
 *
 * This is pw.x's documented rule, where `input_dft` overrides the functional
 * read from the pseudopotential files and without it the files decide. Q3 used
 * to fall back on a default of its own, Perdew-Zunger LDA, which ran PBE
 * datasets as LDA: 127 meV/atom away on the silicon example, converged, and
 * silent.
 *
 * Every dataset must state a functional that functional_from_pseudopotential_label
 * maps, and all of them must agree. Otherwise this throws and names the
 * dataset, because a total energy from datasets of different functionals
 * belongs to none of them. An input can always settle the question by naming
 * the functional itself.
 *
 * Returns empty when some species names no dataset at all, such as a deck
 * whose dataset file is missing: nothing states a functional, and the
 * missing dataset is reported where pseudopotentials are loaded. Nothing runs
 * on an empty name, since the exchange-correlation factory refuses it.
 */
inline std::string functional_from_pseudopotentials(const std::vector<AtomSpecies>& species,
                                                    const std::string& pseudopotential_directory) {
    const std::string remedy = " Name the functional in the input: model.functional in a YAML "
                               "deck, input_dft in a QE-format deck, or the functional tag "
                               "of the format in use.";
    if (species.empty())
        return {};
    for (const auto& one_species : species) {
        if (one_species.pseudopotential_file.empty())
            return {};
    }

    std::string resolved_functional;
    std::string resolved_from;
    for (int species_index = 0; species_index < static_cast<int>(species.size()); ++species_index) {
        const std::string pseudopotential_path = resolve_pseudopotential_path(
            species[species_index].pseudopotential_file, pseudopotential_directory);
        const std::string source =
            pseudopotential_path + " (" + species[species_index].symbol + ")";

        std::string label;
        try {
            label =
                PseudopotentialFactory::read(
                    pseudopotential_path,
                    dataset_index_within_file(species, species_index, pseudopotential_directory))
                    ->xc_functional;
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "The input names no exchange-correlation functional, so Q3 takes it from the "
                "pseudopotentials, and " +
                source + " cannot be read: " + error.what());
        }

        const std::string functional = functional_from_pseudopotential_label(label);
        if (functional.empty())
            throw std::runtime_error(
                "The input names no exchange-correlation functional, and the pseudopotential " +
                source +
                (label.empty()
                     ? std::string(" states none")
                     : " states '" + label + "', which Q3 does not map to a functional it runs") +
                "." + remedy);

        if (resolved_functional.empty()) {
            resolved_functional = functional;
            resolved_from = source;
        } else if (functional != resolved_functional) {
            throw std::runtime_error(
                "The input names no exchange-correlation functional, and its pseudopotentials "
                "were generated with different ones: " +
                resolved_from + " states " + resolved_functional + ", " + source + " states " +
                functional + ". Q3 will not pick one, because the result would belong to neither." +
                remedy);
        }
    }

    log::info("     Exchange-correlation functional: {}, as the pseudopotentials state (the "
              "input names none)",
              resolved_functional);
    return resolved_functional;
}

/**
 * Load pseudopotentials for all species and patch their valence_charge.
 *
 * Resolves pseudopotential file paths, reads each dataset, and updates
 * each species' valence_charge from the loaded pseudopotential data.
 *
 * Reads every registered dataset format alike — the format is chosen from the
 * file's own first line (PseudopotentialFactory::read), so a dataset named in
 * an input works without the user saying which format it is.
 *
 * This is the one place pseudopotentials enter a calculation. It used to
 * refuse PAW datasets, because the downstream factories dispatched on
 * `pseudopotential_type_enum() == ultrasoft` and a PAW file that got this far
 * would have been computed as if it were norm-conserving. That guard came out
 * when the PAW SCF iterator landed: the factories now dispatch on capability,
 * and PAW has its own iterator branch.
 *
 * Returns the loaded PseudopotentialSet.
 */
inline PseudopotentialSet load_pseudopotentials(std::vector<AtomSpecies>& species,
                                                const std::string& pseudopotential_directory) {

    PseudopotentialSet pseudopotential_set;

    for (int species_index = 0; species_index < static_cast<int>(species.size()); ++species_index) {
        std::string pseudopotential_path = resolve_pseudopotential_path(
            species[species_index].pseudopotential_file, pseudopotential_directory);
        int dataset_index =
            dataset_index_within_file(species, species_index, pseudopotential_directory);

        log::info("     Reading PP: {}{}",
                  pseudopotential_path,
                  dataset_index > 0 ? " (dataset " + std::to_string(dataset_index + 1) + ")" : "");
        auto pseudopotential = PseudopotentialFactory::read(pseudopotential_path, dataset_index);

        const auto* paw_pseudopotential =
            dynamic_cast<const PawPseudopotential*>(pseudopotential.get());
        if (paw_pseudopotential != nullptr && paw_pseudopotential->has_core_energy) {
            throw std::runtime_error(
                "PAW UPF dataset " + pseudopotential_path +
                " declares a core_energy reference constant, which Q3 does not yet apply. "
                "Use a supported PAW dataset without core_energy; refusing "
                "this dataset prevents a silently wrong absolute energy and unstable SCF.");
        }

        // Patch valence_charge from loaded pseudopotential data
        species[species_index].valence_charge = pseudopotential->valence_charge;

        pseudopotential_set.push_back(std::move(pseudopotential));
    }

    return pseudopotential_set;
}

} // namespace q3
