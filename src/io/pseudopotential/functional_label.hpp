#pragma once

/**
 * The functional a pseudopotential's header names, as a name Q3's
 * exchange-correlation factory runs.
 *
 * Datasets record the functional they were generated with in Quantum
 * ESPRESSO's vocabulary: UPF v2's `functional` header attribute, or the UPF v1
 * header line that ends "Exchange-Correlation functional". The label takes one
 * of two shapes:
 *
 *   - a short name, such as "PBE" or "PZ";
 *   - the four components the functional is assembled from, in the order local
 *     exchange, local correlation, gradient correction to exchange, gradient
 *     correction to correlation, so "SLA PW PBX PBC" is PBE. Generators
 *     separate them with spaces or dashes, older files spell PBE's two gradient
 *     terms "PBE PBE", and v1 headers append the short name as a fifth word.
 *
 * Case never matters. Every name returned is lower-cased, the spelling Q3
 * records a functional in, so labels that differ only in case compare equal
 * when the datasets of one input are checked for agreement. QE's LibXC
 * spelling ("XC-000I-000I-101L-130L") is already a name the factory reads, so
 * it passes through, lower-cased like the rest.
 *
 * Only labels whose meaning is unambiguous are mapped. Anything else returns
 * empty: Perdew-Wang LDA ("PW", which is not Q3's PZ), a nonlocal or meta-GGA
 * term, or a dataset title that names no functional. The caller then asks the
 * input to name the functional rather than choosing a nearby one.
 */

#include <cctype>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace q3 {

/// The functional a pseudopotential label names, or empty when it names none Q3 maps.
inline std::string functional_from_pseudopotential_label(const std::string& label) {
    const std::size_t first = label.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    std::string trimmed = label.substr(first, label.find_last_not_of(" \t\r\n") - first + 1);
    for (auto& character : trimmed)
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));

    std::vector<std::string> words;
    std::string word;
    for (char character : trimmed + " ") {
        if (std::isspace(static_cast<unsigned char>(character)) || character == '-' ||
            character == '+') {
            if (!word.empty())
                words.push_back(word);
            word.clear();
            continue;
        }
        word += character;
    }

    if (words.empty())
        return {};
    if (words.front() == "xc" && trimmed[2] == '-')
        return trimmed;

    static const std::map<std::string, std::string> short_names = {
        {"pz", "pz"},
        {"lda", "pz"},
        {"pbe", "pbe"},
        {"pbesol", "pbesol"},
        {"blyp", "blyp"},
        {"revpbe", "revpbe"},
        {"rpbe", "rpbe"},
    };
    if (words.size() == 1) {
        auto found = short_names.find(words.front());
        return found == short_names.end() ? std::string{} : found->second;
    }

    static const std::map<std::vector<std::string>, std::string> component_sets = {
        {{"sla", "pz", "nogx", "nogc"}, "pz"},
        {{"sla", "pw", "pbx", "pbc"}, "pbe"},
        {{"sla", "pw", "pbe", "pbe"}, "pbe"},
        {{"sla", "pw", "psx", "psc"}, "pbesol"},
        {{"sla", "lyp", "b88", "blyp"}, "blyp"},
        {{"sla", "pw", "revx", "pbc"}, "revpbe"},
        {{"sla", "pw", "rpb", "pbc"}, "revpbe"},
    };
    if (words.size() < 4)
        return {};
    auto found = component_sets.find(std::vector<std::string>(words.begin(), words.begin() + 4));
    if (found == component_sets.end())
        return {};

    /**
     * A word past the fourth is either the short name a v1 header repeats,
     * which must agree, or a further term, such as a nonlocal correlation
     * kernel, that names a functional Q3 does not run.
     */
    for (auto extra = words.begin() + 4; extra != words.end(); ++extra) {
        auto short_name = short_names.find(*extra);
        if (short_name == short_names.end() || short_name->second != found->second)
            return {};
    }
    return found->second;
}

} // namespace q3
