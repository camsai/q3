#pragma once

// @generated from properties_directory/non_scalar/total_energy_contributions.json — DO NOT EDIT

#include <optional>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class TotalEnergyContributionsTemperatureEntropyNameEnum {
    TEMPERATURE_ENTROPY
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsTemperatureEntropyNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsTemperatureEntropyNameEnum::TEMPERATURE_ENTROPY: json_output = "temperature_entropy"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsTemperatureEntropyNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "temperature_entropy") {
        enum_value = TotalEnergyContributionsTemperatureEntropyNameEnum::TEMPERATURE_ENTROPY;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsTemperatureEntropyNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsHarrisFoulkesNameEnum {
    HARRIS_FOULKES
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsHarrisFoulkesNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsHarrisFoulkesNameEnum::HARRIS_FOULKES: json_output = "harris_foulkes"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsHarrisFoulkesNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "harris_foulkes") {
        enum_value = TotalEnergyContributionsHarrisFoulkesNameEnum::HARRIS_FOULKES;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsHarrisFoulkesNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsSmearingNameEnum {
    SMEARING
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsSmearingNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsSmearingNameEnum::SMEARING: json_output = "smearing"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsSmearingNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "smearing") {
        enum_value = TotalEnergyContributionsSmearingNameEnum::SMEARING;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsSmearingNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsOneElectronNameEnum {
    ONE_ELECTRON
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsOneElectronNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsOneElectronNameEnum::ONE_ELECTRON: json_output = "one_electron"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsOneElectronNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "one_electron") {
        enum_value = TotalEnergyContributionsOneElectronNameEnum::ONE_ELECTRON;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsOneElectronNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsHartreeNameEnum {
    HARTREE
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsHartreeNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsHartreeNameEnum::HARTREE: json_output = "hartree"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsHartreeNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "hartree") {
        enum_value = TotalEnergyContributionsHartreeNameEnum::HARTREE;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsHartreeNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsExchangeNameEnum {
    EXCHANGE
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsExchangeNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsExchangeNameEnum::EXCHANGE: json_output = "exchange"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsExchangeNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "exchange") {
        enum_value = TotalEnergyContributionsExchangeNameEnum::EXCHANGE;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsExchangeNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsExchangeCorrelationNameEnum {
    EXCHANGE_CORRELATION
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsExchangeCorrelationNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsExchangeCorrelationNameEnum::EXCHANGE_CORRELATION: json_output = "exchange_correlation"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsExchangeCorrelationNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "exchange_correlation") {
        enum_value = TotalEnergyContributionsExchangeCorrelationNameEnum::EXCHANGE_CORRELATION;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsExchangeCorrelationNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsEwaldNameEnum {
    EWALD
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsEwaldNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsEwaldNameEnum::EWALD: json_output = "ewald"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsEwaldNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "ewald") {
        enum_value = TotalEnergyContributionsEwaldNameEnum::EWALD;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsEwaldNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsAlphaZNameEnum {
    ALPHAZ
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsAlphaZNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsAlphaZNameEnum::ALPHAZ: json_output = "alphaZ"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsAlphaZNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "alphaZ") {
        enum_value = TotalEnergyContributionsAlphaZNameEnum::ALPHAZ;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsAlphaZNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsAtomicEnergyNameEnum {
    ATOMIC_ENERGY
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsAtomicEnergyNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsAtomicEnergyNameEnum::ATOMIC_ENERGY: json_output = "atomic_energy"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsAtomicEnergyNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "atomic_energy") {
        enum_value = TotalEnergyContributionsAtomicEnergyNameEnum::ATOMIC_ENERGY;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsAtomicEnergyNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsEigenvaluesNameEnum {
    EIGENVALUES
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsEigenvaluesNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsEigenvaluesNameEnum::EIGENVALUES: json_output = "eigenvalues"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsEigenvaluesNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "eigenvalues") {
        enum_value = TotalEnergyContributionsEigenvaluesNameEnum::EIGENVALUES;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsEigenvaluesNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsPAWDoubleCounting2NameEnum {
    PAW_DOUBLE_COUNTING_CORRECTION_2
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsPAWDoubleCounting2NameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsPAWDoubleCounting2NameEnum::PAW_DOUBLE_COUNTING_CORRECTION_2: json_output = "PAW_double-counting_correction_2"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsPAWDoubleCounting2NameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "PAW_double-counting_correction_2") {
        enum_value = TotalEnergyContributionsPAWDoubleCounting2NameEnum::PAW_DOUBLE_COUNTING_CORRECTION_2;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsPAWDoubleCounting2NameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsPAWDoubleCounting3NameEnum {
    PAW_DOUBLE_COUNTING_CORRECTION_3
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsPAWDoubleCounting3NameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsPAWDoubleCounting3NameEnum::PAW_DOUBLE_COUNTING_CORRECTION_3: json_output = "PAW_double-counting_correction_3"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsPAWDoubleCounting3NameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "PAW_double-counting_correction_3") {
        enum_value = TotalEnergyContributionsPAWDoubleCounting3NameEnum::PAW_DOUBLE_COUNTING_CORRECTION_3;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsPAWDoubleCounting3NameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsHartreeFockNameEnum {
    HARTREE_FOCK
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsHartreeFockNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsHartreeFockNameEnum::HARTREE_FOCK: json_output = "hartree_fock"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsHartreeFockNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "hartree_fock") {
        enum_value = TotalEnergyContributionsHartreeFockNameEnum::HARTREE_FOCK;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsHartreeFockNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsNameEnum {
    TOTAL_ENERGY_CONTRIBUTIONS
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsNameEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsNameEnum::TOTAL_ENERGY_CONTRIBUTIONS: json_output = "total_energy_contributions"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsNameEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "total_energy_contributions") {
        enum_value = TotalEnergyContributionsNameEnum::TOTAL_ENERGY_CONTRIBUTIONS;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsNameEnum: " + string_value);
    }
}

enum class TotalEnergyContributionsUnitsEnum {
    KCAL_MOL,
    KJ_MOL,
    EV,
    J_MOL,
    HARTREE,
    CM_1,
    RY,
    EV_ATOM
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsUnitsEnum& enum_value) {
    switch (enum_value) {
        case TotalEnergyContributionsUnitsEnum::KCAL_MOL: json_output = "kcal/mol"; break;
        case TotalEnergyContributionsUnitsEnum::KJ_MOL: json_output = "kJ/mol"; break;
        case TotalEnergyContributionsUnitsEnum::EV: json_output = "eV"; break;
        case TotalEnergyContributionsUnitsEnum::J_MOL: json_output = "J/mol"; break;
        case TotalEnergyContributionsUnitsEnum::HARTREE: json_output = "hartree"; break;
        case TotalEnergyContributionsUnitsEnum::CM_1: json_output = "cm-1"; break;
        case TotalEnergyContributionsUnitsEnum::RY: json_output = "Ry"; break;
        case TotalEnergyContributionsUnitsEnum::EV_ATOM: json_output = "eV/atom"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsUnitsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "kcal/mol") {
        enum_value = TotalEnergyContributionsUnitsEnum::KCAL_MOL;
    } else if (string_value == "kJ/mol") {
        enum_value = TotalEnergyContributionsUnitsEnum::KJ_MOL;
    } else if (string_value == "eV") {
        enum_value = TotalEnergyContributionsUnitsEnum::EV;
    } else if (string_value == "J/mol") {
        enum_value = TotalEnergyContributionsUnitsEnum::J_MOL;
    } else if (string_value == "hartree") {
        enum_value = TotalEnergyContributionsUnitsEnum::HARTREE;
    } else if (string_value == "cm-1") {
        enum_value = TotalEnergyContributionsUnitsEnum::CM_1;
    } else if (string_value == "Ry") {
        enum_value = TotalEnergyContributionsUnitsEnum::RY;
    } else if (string_value == "eV/atom") {
        enum_value = TotalEnergyContributionsUnitsEnum::EV_ATOM;
    } else {
        throw std::runtime_error("Unknown TotalEnergyContributionsUnitsEnum: " + string_value);
    }
}

/**
 * product of temperature and configurational entropy
 */
struct TotalEnergyContributionsTemperatureEntropy {
    std::optional<TotalEnergyContributionsTemperatureEntropyNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsTemperatureEntropy& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsTemperatureEntropy& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsTemperatureEntropyNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * non self-consitent energy based on an input charge density
 */
struct TotalEnergyContributionsHarrisFoulkes {
    std::optional<TotalEnergyContributionsHarrisFoulkesNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsHarrisFoulkes& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsHarrisFoulkes& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsHarrisFoulkesNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * smearing energy
 */
struct TotalEnergyContributionsSmearing {
    std::optional<TotalEnergyContributionsSmearingNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsSmearing& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsSmearing& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsSmearingNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * kinetic + pseudopotential energy
 */
struct TotalEnergyContributionsOneElectron {
    std::optional<TotalEnergyContributionsOneElectronNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsOneElectron& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsOneElectron& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsOneElectronNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * energy due to coulomb potential
 */
struct TotalEnergyContributionsHartree {
    std::optional<TotalEnergyContributionsHartreeNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsHartree& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsHartree& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsHartreeNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * exchange energy
 */
struct TotalEnergyContributionsExchange {
    std::optional<TotalEnergyContributionsExchangeNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsExchange& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsExchange& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsExchangeNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * exchange and correlation energy per particle
 */
struct TotalEnergyContributionsExchangeCorrelation {
    std::optional<TotalEnergyContributionsExchangeCorrelationNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsExchangeCorrelation& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsExchangeCorrelation& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsExchangeCorrelationNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * summation of interaction energies at long length scales due to coloumbic interactions
 */
struct TotalEnergyContributionsEwald {
    std::optional<TotalEnergyContributionsEwaldNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsEwald& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsEwald& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsEwaldNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * divergent electrostatic ion interaction in compensating electron gas
 */
struct TotalEnergyContributionsAlphaZ {
    std::optional<TotalEnergyContributionsAlphaZNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsAlphaZ& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsAlphaZ& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsAlphaZNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * kinetic energy of wavefunctions in the atomic limit
 */
struct TotalEnergyContributionsAtomicEnergy {
    std::optional<TotalEnergyContributionsAtomicEnergyNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsAtomicEnergy& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsAtomicEnergy& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsAtomicEnergyNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * sum of one electron energies of kinetic, electrostatic, and exchange correlation
 */
struct TotalEnergyContributionsEigenvalues {
    std::optional<TotalEnergyContributionsEigenvaluesNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsEigenvalues& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsEigenvalues& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsEigenvaluesNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * double counting correction 2
 */
struct TotalEnergyContributionsPAWDoubleCounting2 {
    std::optional<TotalEnergyContributionsPAWDoubleCounting2NameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsPAWDoubleCounting2& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsPAWDoubleCounting2& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsPAWDoubleCounting2NameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * double counting correction 3
 */
struct TotalEnergyContributionsPAWDoubleCounting3 {
    std::optional<TotalEnergyContributionsPAWDoubleCounting3NameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsPAWDoubleCounting3& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsPAWDoubleCounting3& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsPAWDoubleCounting3NameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 * hartree-fock contribution
 */
struct TotalEnergyContributionsHartreeFock {
    std::optional<TotalEnergyContributionsHartreeFockNameEnum> name;
    double value{};
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributionsHartreeFock& data) {
    json_output = nlohmann::json::object();
    if (data.name) {
        json_output["name"] = *data.name;
    }
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributionsHartreeFock& data) {
    if (json_input.contains("name")) {
        data.name = json_input.at("name").get<TotalEnergyContributionsHartreeFockNameEnum>();
    }
    json_input.at("value").get_to(data.value);
}

/**
 *
 * Schema: properties_directory/non_scalar/total_energy_contributions.json
 */
struct TotalEnergyContributions {
    // product of temperature and configurational entropy
    std::optional<TotalEnergyContributionsTemperatureEntropy> temperature_entropy;
    // non self-consitent energy based on an input charge density
    std::optional<TotalEnergyContributionsHarrisFoulkes> harris_foulkes;
    // smearing energy
    std::optional<TotalEnergyContributionsSmearing> smearing;
    // kinetic + pseudopotential energy
    std::optional<TotalEnergyContributionsOneElectron> one_electron;
    // energy due to coulomb potential
    std::optional<TotalEnergyContributionsHartree> hartree;
    // exchange energy
    std::optional<TotalEnergyContributionsExchange> exchange;
    // exchange and correlation energy per particle
    std::optional<TotalEnergyContributionsExchangeCorrelation> exchange_correlation;
    // summation of interaction energies at long length scales due to coloumbic interactions
    std::optional<TotalEnergyContributionsEwald> ewald;
    // divergent electrostatic ion interaction in compensating electron gas
    std::optional<TotalEnergyContributionsAlphaZ> alpha_z;
    // kinetic energy of wavefunctions in the atomic limit
    std::optional<TotalEnergyContributionsAtomicEnergy> atomic_energy;
    // sum of one electron energies of kinetic, electrostatic, and exchange correlation
    std::optional<TotalEnergyContributionsEigenvalues> eigenvalues;
    // double counting correction 2
    std::optional<TotalEnergyContributionsPAWDoubleCounting2> p_a_w_double_counting2;
    // double counting correction 3
    std::optional<TotalEnergyContributionsPAWDoubleCounting3> p_a_w_double_counting3;
    // hartree-fock contribution
    std::optional<TotalEnergyContributionsHartreeFock> hartree_fock;
    TotalEnergyContributionsNameEnum name;
    std::optional<TotalEnergyContributionsUnitsEnum> units;
};

inline void to_json(nlohmann::json& json_output, const TotalEnergyContributions& data) {
    json_output = nlohmann::json::object();
    if (data.temperature_entropy) {
        json_output["temperatureEntropy"] = *data.temperature_entropy;
    }
    if (data.harris_foulkes) {
        json_output["harris_foulkes"] = *data.harris_foulkes;
    }
    if (data.smearing) {
        json_output["smearing"] = *data.smearing;
    }
    if (data.one_electron) {
        json_output["one_electron"] = *data.one_electron;
    }
    if (data.hartree) {
        json_output["hartree"] = *data.hartree;
    }
    if (data.exchange) {
        json_output["exchange"] = *data.exchange;
    }
    if (data.exchange_correlation) {
        json_output["exchange_correlation"] = *data.exchange_correlation;
    }
    if (data.ewald) {
        json_output["ewald"] = *data.ewald;
    }
    if (data.alpha_z) {
        json_output["alphaZ"] = *data.alpha_z;
    }
    if (data.atomic_energy) {
        json_output["atomicEnergy"] = *data.atomic_energy;
    }
    if (data.eigenvalues) {
        json_output["eigenvalues"] = *data.eigenvalues;
    }
    if (data.p_a_w_double_counting2) {
        json_output["PAWDoubleCounting2"] = *data.p_a_w_double_counting2;
    }
    if (data.p_a_w_double_counting3) {
        json_output["PAWDoubleCounting3"] = *data.p_a_w_double_counting3;
    }
    if (data.hartree_fock) {
        json_output["hartreeFock"] = *data.hartree_fock;
    }
    json_output["name"] = data.name;
    if (data.units) {
        json_output["units"] = *data.units;
    }
}

inline void from_json(const nlohmann::json& json_input, TotalEnergyContributions& data) {
    if (json_input.contains("temperatureEntropy")) {
        data.temperature_entropy = json_input.at("temperatureEntropy").get<TotalEnergyContributionsTemperatureEntropy>();
    }
    if (json_input.contains("harris_foulkes")) {
        data.harris_foulkes = json_input.at("harris_foulkes").get<TotalEnergyContributionsHarrisFoulkes>();
    }
    if (json_input.contains("smearing")) {
        data.smearing = json_input.at("smearing").get<TotalEnergyContributionsSmearing>();
    }
    if (json_input.contains("one_electron")) {
        data.one_electron = json_input.at("one_electron").get<TotalEnergyContributionsOneElectron>();
    }
    if (json_input.contains("hartree")) {
        data.hartree = json_input.at("hartree").get<TotalEnergyContributionsHartree>();
    }
    if (json_input.contains("exchange")) {
        data.exchange = json_input.at("exchange").get<TotalEnergyContributionsExchange>();
    }
    if (json_input.contains("exchange_correlation")) {
        data.exchange_correlation = json_input.at("exchange_correlation").get<TotalEnergyContributionsExchangeCorrelation>();
    }
    if (json_input.contains("ewald")) {
        data.ewald = json_input.at("ewald").get<TotalEnergyContributionsEwald>();
    }
    if (json_input.contains("alphaZ")) {
        data.alpha_z = json_input.at("alphaZ").get<TotalEnergyContributionsAlphaZ>();
    }
    if (json_input.contains("atomicEnergy")) {
        data.atomic_energy = json_input.at("atomicEnergy").get<TotalEnergyContributionsAtomicEnergy>();
    }
    if (json_input.contains("eigenvalues")) {
        data.eigenvalues = json_input.at("eigenvalues").get<TotalEnergyContributionsEigenvalues>();
    }
    if (json_input.contains("PAWDoubleCounting2")) {
        data.p_a_w_double_counting2 = json_input.at("PAWDoubleCounting2").get<TotalEnergyContributionsPAWDoubleCounting2>();
    }
    if (json_input.contains("PAWDoubleCounting3")) {
        data.p_a_w_double_counting3 = json_input.at("PAWDoubleCounting3").get<TotalEnergyContributionsPAWDoubleCounting3>();
    }
    if (json_input.contains("hartreeFock")) {
        data.hartree_fock = json_input.at("hartreeFock").get<TotalEnergyContributionsHartreeFock>();
    }
    json_input.at("name").get_to(data.name);
    if (json_input.contains("units")) {
        data.units = json_input.at("units").get<TotalEnergyContributionsUnitsEnum>();
    }
}

} // namespace q3::generated
