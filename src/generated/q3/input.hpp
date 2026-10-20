#pragma once

// @generated from input.json — DO NOT EDIT

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace q3::generated {

enum class InputModelDispersionEnum {
    NONE,
    D3BJ
};

inline void to_json(nlohmann::json& json_output, const InputModelDispersionEnum& enum_value) {
    switch (enum_value) {
        case InputModelDispersionEnum::NONE: json_output = "none"; break;
        case InputModelDispersionEnum::D3BJ: json_output = "d3bj"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputModelDispersionEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "none") {
        enum_value = InputModelDispersionEnum::NONE;
    } else if (string_value == "d3bj") {
        enum_value = InputModelDispersionEnum::D3BJ;
    } else {
        throw std::runtime_error("Unknown InputModelDispersionEnum: " + string_value);
    }
}

enum class InputMethodScfDiagonalizationEnum {
    DAVIDSON,
    DIRECT
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfDiagonalizationEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfDiagonalizationEnum::DAVIDSON: json_output = "davidson"; break;
        case InputMethodScfDiagonalizationEnum::DIRECT: json_output = "direct"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfDiagonalizationEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "davidson") {
        enum_value = InputMethodScfDiagonalizationEnum::DAVIDSON;
    } else if (string_value == "direct") {
        enum_value = InputMethodScfDiagonalizationEnum::DIRECT;
    } else {
        throw std::runtime_error("Unknown InputMethodScfDiagonalizationEnum: " + string_value);
    }
}

enum class InputMethodScfNumberOfSpinComponentsEnum {
    V_1,
    V_2
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfNumberOfSpinComponentsEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfNumberOfSpinComponentsEnum::V_1: json_output = 1; break;
        case InputMethodScfNumberOfSpinComponentsEnum::V_2: json_output = 2; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfNumberOfSpinComponentsEnum& enum_value) {
    const auto integer_value = json_input.get<int64_t>();
    if (integer_value == 1) {
        enum_value = InputMethodScfNumberOfSpinComponentsEnum::V_1;
    } else if (integer_value == 2) {
        enum_value = InputMethodScfNumberOfSpinComponentsEnum::V_2;
    } else {
        throw std::runtime_error("Unknown InputMethodScfNumberOfSpinComponentsEnum value: " + std::to_string(integer_value));
    }
}

enum class InputMethodScfSmearingTypeEnum {
    NONE,
    FIXED,
    GAUSSIAN,
    GAUSS,
    METHFESSEL_PAXTON,
    M_P,
    MP,
    MARZARI_VANDERBILT,
    COLD,
    M_V,
    MV,
    FERMI_DIRAC,
    F_D,
    FD
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfSmearingTypeEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfSmearingTypeEnum::NONE: json_output = "none"; break;
        case InputMethodScfSmearingTypeEnum::FIXED: json_output = "fixed"; break;
        case InputMethodScfSmearingTypeEnum::GAUSSIAN: json_output = "gaussian"; break;
        case InputMethodScfSmearingTypeEnum::GAUSS: json_output = "gauss"; break;
        case InputMethodScfSmearingTypeEnum::METHFESSEL_PAXTON: json_output = "methfessel-paxton"; break;
        case InputMethodScfSmearingTypeEnum::M_P: json_output = "m-p"; break;
        case InputMethodScfSmearingTypeEnum::MP: json_output = "mp"; break;
        case InputMethodScfSmearingTypeEnum::MARZARI_VANDERBILT: json_output = "marzari-vanderbilt"; break;
        case InputMethodScfSmearingTypeEnum::COLD: json_output = "cold"; break;
        case InputMethodScfSmearingTypeEnum::M_V: json_output = "m-v"; break;
        case InputMethodScfSmearingTypeEnum::MV: json_output = "mv"; break;
        case InputMethodScfSmearingTypeEnum::FERMI_DIRAC: json_output = "fermi-dirac"; break;
        case InputMethodScfSmearingTypeEnum::F_D: json_output = "f-d"; break;
        case InputMethodScfSmearingTypeEnum::FD: json_output = "fd"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfSmearingTypeEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "none") {
        enum_value = InputMethodScfSmearingTypeEnum::NONE;
    } else if (string_value == "fixed") {
        enum_value = InputMethodScfSmearingTypeEnum::FIXED;
    } else if (string_value == "gaussian") {
        enum_value = InputMethodScfSmearingTypeEnum::GAUSSIAN;
    } else if (string_value == "gauss") {
        enum_value = InputMethodScfSmearingTypeEnum::GAUSS;
    } else if (string_value == "methfessel-paxton") {
        enum_value = InputMethodScfSmearingTypeEnum::METHFESSEL_PAXTON;
    } else if (string_value == "m-p") {
        enum_value = InputMethodScfSmearingTypeEnum::M_P;
    } else if (string_value == "mp") {
        enum_value = InputMethodScfSmearingTypeEnum::MP;
    } else if (string_value == "marzari-vanderbilt") {
        enum_value = InputMethodScfSmearingTypeEnum::MARZARI_VANDERBILT;
    } else if (string_value == "cold") {
        enum_value = InputMethodScfSmearingTypeEnum::COLD;
    } else if (string_value == "m-v") {
        enum_value = InputMethodScfSmearingTypeEnum::M_V;
    } else if (string_value == "mv") {
        enum_value = InputMethodScfSmearingTypeEnum::MV;
    } else if (string_value == "fermi-dirac") {
        enum_value = InputMethodScfSmearingTypeEnum::FERMI_DIRAC;
    } else if (string_value == "f-d") {
        enum_value = InputMethodScfSmearingTypeEnum::F_D;
    } else if (string_value == "fd") {
        enum_value = InputMethodScfSmearingTypeEnum::FD;
    } else {
        throw std::runtime_error("Unknown InputMethodScfSmearingTypeEnum: " + string_value);
    }
}

enum class InputMethodScfMixingModeEnum {
    BROYDEN,
    LINEAR,
    ANDERSON
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfMixingModeEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfMixingModeEnum::BROYDEN: json_output = "broyden"; break;
        case InputMethodScfMixingModeEnum::LINEAR: json_output = "linear"; break;
        case InputMethodScfMixingModeEnum::ANDERSON: json_output = "anderson"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfMixingModeEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "broyden") {
        enum_value = InputMethodScfMixingModeEnum::BROYDEN;
    } else if (string_value == "linear") {
        enum_value = InputMethodScfMixingModeEnum::LINEAR;
    } else if (string_value == "anderson") {
        enum_value = InputMethodScfMixingModeEnum::ANDERSON;
    } else {
        throw std::runtime_error("Unknown InputMethodScfMixingModeEnum: " + string_value);
    }
}

enum class InputMethodScfMixingPreconditioningEnum {
    DEFAULT,
    KERKER,
    NONE
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfMixingPreconditioningEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfMixingPreconditioningEnum::DEFAULT: json_output = "default"; break;
        case InputMethodScfMixingPreconditioningEnum::KERKER: json_output = "kerker"; break;
        case InputMethodScfMixingPreconditioningEnum::NONE: json_output = "none"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfMixingPreconditioningEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "default") {
        enum_value = InputMethodScfMixingPreconditioningEnum::DEFAULT;
    } else if (string_value == "kerker") {
        enum_value = InputMethodScfMixingPreconditioningEnum::KERKER;
    } else if (string_value == "none") {
        enum_value = InputMethodScfMixingPreconditioningEnum::NONE;
    } else {
        throw std::runtime_error("Unknown InputMethodScfMixingPreconditioningEnum: " + string_value);
    }
}

enum class InputMethodScfHubbardProjectorsEnum {
    ATOMIC,
    ORTHO_ATOMIC
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfHubbardProjectorsEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfHubbardProjectorsEnum::ATOMIC: json_output = "atomic"; break;
        case InputMethodScfHubbardProjectorsEnum::ORTHO_ATOMIC: json_output = "ortho-atomic"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfHubbardProjectorsEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "atomic") {
        enum_value = InputMethodScfHubbardProjectorsEnum::ATOMIC;
    } else if (string_value == "ortho-atomic") {
        enum_value = InputMethodScfHubbardProjectorsEnum::ORTHO_ATOMIC;
    } else {
        throw std::runtime_error("Unknown InputMethodScfHubbardProjectorsEnum: " + string_value);
    }
}

enum class InputMethodScfHubbardStartingEigenvalueSpinEnum {
    V_1,
    V_2
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfHubbardStartingEigenvalueSpinEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfHubbardStartingEigenvalueSpinEnum::V_1: json_output = 1; break;
        case InputMethodScfHubbardStartingEigenvalueSpinEnum::V_2: json_output = 2; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfHubbardStartingEigenvalueSpinEnum& enum_value) {
    const auto integer_value = json_input.get<int64_t>();
    if (integer_value == 1) {
        enum_value = InputMethodScfHubbardStartingEigenvalueSpinEnum::V_1;
    } else if (integer_value == 2) {
        enum_value = InputMethodScfHubbardStartingEigenvalueSpinEnum::V_2;
    } else {
        throw std::runtime_error("Unknown InputMethodScfHubbardStartingEigenvalueSpinEnum value: " + std::to_string(integer_value));
    }
}

enum class InputMethodScfKgridShiftsItemEnum {
    V_0,
    V_1
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfKgridShiftsItemEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfKgridShiftsItemEnum::V_0: json_output = 0; break;
        case InputMethodScfKgridShiftsItemEnum::V_1: json_output = 1; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfKgridShiftsItemEnum& enum_value) {
    const auto integer_value = json_input.get<int64_t>();
    if (integer_value == 0) {
        enum_value = InputMethodScfKgridShiftsItemEnum::V_0;
    } else if (integer_value == 1) {
        enum_value = InputMethodScfKgridShiftsItemEnum::V_1;
    } else {
        throw std::runtime_error("Unknown InputMethodScfKgridShiftsItemEnum value: " + std::to_string(integer_value));
    }
}

enum class InputMethodScfKgridIbzReductionEnum {
    INTERNAL,
    SPGLIB
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfKgridIbzReductionEnum& enum_value) {
    switch (enum_value) {
        case InputMethodScfKgridIbzReductionEnum::INTERNAL: json_output = "internal"; break;
        case InputMethodScfKgridIbzReductionEnum::SPGLIB: json_output = "spglib"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfKgridIbzReductionEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "internal") {
        enum_value = InputMethodScfKgridIbzReductionEnum::INTERNAL;
    } else if (string_value == "spglib") {
        enum_value = InputMethodScfKgridIbzReductionEnum::SPGLIB;
    } else {
        throw std::runtime_error("Unknown InputMethodScfKgridIbzReductionEnum: " + string_value);
    }
}

enum class InputControlCalculationEnum {
    SCF,
    SELF_CONSISTENT_FIELD,
    NSCF,
    NON_SELF_CONSISTENT_FIELD,
    BANDS,
    BAND_STRUCTURE,
    RELAX,
    RELAXATION,
    DOS,
    DENSITY_OF_STATES,
    INTERPOLATION
};

inline void to_json(nlohmann::json& json_output, const InputControlCalculationEnum& enum_value) {
    switch (enum_value) {
        case InputControlCalculationEnum::SCF: json_output = "scf"; break;
        case InputControlCalculationEnum::SELF_CONSISTENT_FIELD: json_output = "self_consistent_field"; break;
        case InputControlCalculationEnum::NSCF: json_output = "nscf"; break;
        case InputControlCalculationEnum::NON_SELF_CONSISTENT_FIELD: json_output = "non_self_consistent_field"; break;
        case InputControlCalculationEnum::BANDS: json_output = "bands"; break;
        case InputControlCalculationEnum::BAND_STRUCTURE: json_output = "band_structure"; break;
        case InputControlCalculationEnum::RELAX: json_output = "relax"; break;
        case InputControlCalculationEnum::RELAXATION: json_output = "relaxation"; break;
        case InputControlCalculationEnum::DOS: json_output = "dos"; break;
        case InputControlCalculationEnum::DENSITY_OF_STATES: json_output = "density_of_states"; break;
        case InputControlCalculationEnum::INTERPOLATION: json_output = "interpolation"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlCalculationEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "scf") {
        enum_value = InputControlCalculationEnum::SCF;
    } else if (string_value == "self_consistent_field") {
        enum_value = InputControlCalculationEnum::SELF_CONSISTENT_FIELD;
    } else if (string_value == "nscf") {
        enum_value = InputControlCalculationEnum::NSCF;
    } else if (string_value == "non_self_consistent_field") {
        enum_value = InputControlCalculationEnum::NON_SELF_CONSISTENT_FIELD;
    } else if (string_value == "bands") {
        enum_value = InputControlCalculationEnum::BANDS;
    } else if (string_value == "band_structure") {
        enum_value = InputControlCalculationEnum::BAND_STRUCTURE;
    } else if (string_value == "relax") {
        enum_value = InputControlCalculationEnum::RELAX;
    } else if (string_value == "relaxation") {
        enum_value = InputControlCalculationEnum::RELAXATION;
    } else if (string_value == "dos") {
        enum_value = InputControlCalculationEnum::DOS;
    } else if (string_value == "density_of_states") {
        enum_value = InputControlCalculationEnum::DENSITY_OF_STATES;
    } else if (string_value == "interpolation") {
        enum_value = InputControlCalculationEnum::INTERPOLATION;
    } else {
        throw std::runtime_error("Unknown InputControlCalculationEnum: " + string_value);
    }
}

enum class InputControlVerbosityEnum {
    DEFAULT,
    HIGH
};

inline void to_json(nlohmann::json& json_output, const InputControlVerbosityEnum& enum_value) {
    switch (enum_value) {
        case InputControlVerbosityEnum::DEFAULT: json_output = "default"; break;
        case InputControlVerbosityEnum::HIGH: json_output = "high"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlVerbosityEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "default") {
        enum_value = InputControlVerbosityEnum::DEFAULT;
    } else if (string_value == "high") {
        enum_value = InputControlVerbosityEnum::HIGH;
    } else {
        throw std::runtime_error("Unknown InputControlVerbosityEnum: " + string_value);
    }
}

enum class InputControlNonlocalStressDerivativeMethodEnum {
    ANALYTICAL,
    NUMERICAL
};

inline void to_json(nlohmann::json& json_output, const InputControlNonlocalStressDerivativeMethodEnum& enum_value) {
    switch (enum_value) {
        case InputControlNonlocalStressDerivativeMethodEnum::ANALYTICAL: json_output = "analytical"; break;
        case InputControlNonlocalStressDerivativeMethodEnum::NUMERICAL: json_output = "numerical"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlNonlocalStressDerivativeMethodEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "analytical") {
        enum_value = InputControlNonlocalStressDerivativeMethodEnum::ANALYTICAL;
    } else if (string_value == "numerical") {
        enum_value = InputControlNonlocalStressDerivativeMethodEnum::NUMERICAL;
    } else {
        throw std::runtime_error("Unknown InputControlNonlocalStressDerivativeMethodEnum: " + string_value);
    }
}

enum class InputControlIsolatedSystemCorrectionEnum {
    NONE,
    MAKOV_PAYNE,
    MARTYNA_TUCKERMAN
};

inline void to_json(nlohmann::json& json_output, const InputControlIsolatedSystemCorrectionEnum& enum_value) {
    switch (enum_value) {
        case InputControlIsolatedSystemCorrectionEnum::NONE: json_output = "none"; break;
        case InputControlIsolatedSystemCorrectionEnum::MAKOV_PAYNE: json_output = "makov-payne"; break;
        case InputControlIsolatedSystemCorrectionEnum::MARTYNA_TUCKERMAN: json_output = "martyna-tuckerman"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlIsolatedSystemCorrectionEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "none") {
        enum_value = InputControlIsolatedSystemCorrectionEnum::NONE;
    } else if (string_value == "makov-payne") {
        enum_value = InputControlIsolatedSystemCorrectionEnum::MAKOV_PAYNE;
    } else if (string_value == "martyna-tuckerman") {
        enum_value = InputControlIsolatedSystemCorrectionEnum::MARTYNA_TUCKERMAN;
    } else {
        throw std::runtime_error("Unknown InputControlIsolatedSystemCorrectionEnum: " + string_value);
    }
}

enum class InputControlMeanFieldExportEnum {
    NONE,
    BERKELEYGW
};

inline void to_json(nlohmann::json& json_output, const InputControlMeanFieldExportEnum& enum_value) {
    switch (enum_value) {
        case InputControlMeanFieldExportEnum::NONE: json_output = "none"; break;
        case InputControlMeanFieldExportEnum::BERKELEYGW: json_output = "berkeleygw"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlMeanFieldExportEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "none") {
        enum_value = InputControlMeanFieldExportEnum::NONE;
    } else if (string_value == "berkeleygw") {
        enum_value = InputControlMeanFieldExportEnum::BERKELEYGW;
    } else {
        throw std::runtime_error("Unknown InputControlMeanFieldExportEnum: " + string_value);
    }
}

enum class InputControlInterpolationMethodEnum {
    WANNIER
};

inline void to_json(nlohmann::json& json_output, const InputControlInterpolationMethodEnum& enum_value) {
    switch (enum_value) {
        case InputControlInterpolationMethodEnum::WANNIER: json_output = "wannier"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlInterpolationMethodEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "wannier") {
        enum_value = InputControlInterpolationMethodEnum::WANNIER;
    } else {
        throw std::runtime_error("Unknown InputControlInterpolationMethodEnum: " + string_value);
    }
}

enum class InputControlInterpolationDisentanglementEnum {
    NONE,
    PROJECTABILITY
};

inline void to_json(nlohmann::json& json_output, const InputControlInterpolationDisentanglementEnum& enum_value) {
    switch (enum_value) {
        case InputControlInterpolationDisentanglementEnum::NONE: json_output = "none"; break;
        case InputControlInterpolationDisentanglementEnum::PROJECTABILITY: json_output = "projectability"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlInterpolationDisentanglementEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "none") {
        enum_value = InputControlInterpolationDisentanglementEnum::NONE;
    } else if (string_value == "projectability") {
        enum_value = InputControlInterpolationDisentanglementEnum::PROJECTABILITY;
    } else {
        throw std::runtime_error("Unknown InputControlInterpolationDisentanglementEnum: " + string_value);
    }
}

enum class InputControlInterpolationLocalizationEnum {
    AUTO,
    NONE,
    MAXIMAL
};

inline void to_json(nlohmann::json& json_output, const InputControlInterpolationLocalizationEnum& enum_value) {
    switch (enum_value) {
        case InputControlInterpolationLocalizationEnum::AUTO: json_output = "auto"; break;
        case InputControlInterpolationLocalizationEnum::NONE: json_output = "none"; break;
        case InputControlInterpolationLocalizationEnum::MAXIMAL: json_output = "maximal"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlInterpolationLocalizationEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "auto") {
        enum_value = InputControlInterpolationLocalizationEnum::AUTO;
    } else if (string_value == "none") {
        enum_value = InputControlInterpolationLocalizationEnum::NONE;
    } else if (string_value == "maximal") {
        enum_value = InputControlInterpolationLocalizationEnum::MAXIMAL;
    } else {
        throw std::runtime_error("Unknown InputControlInterpolationLocalizationEnum: " + string_value);
    }
}

enum class InputControlInterpolationExportFormatsItemEnum {
    WANNIER90
};

inline void to_json(nlohmann::json& json_output, const InputControlInterpolationExportFormatsItemEnum& enum_value) {
    switch (enum_value) {
        case InputControlInterpolationExportFormatsItemEnum::WANNIER90: json_output = "wannier90"; break;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlInterpolationExportFormatsItemEnum& enum_value) {
    const auto& string_value = json_input.get<std::string>();
    if (string_value == "wannier90") {
        enum_value = InputControlInterpolationExportFormatsItemEnum::WANNIER90;
    } else {
        throw std::runtime_error("Unknown InputControlInterpolationExportFormatsItemEnum: " + string_value);
    }
}

/**
 */
struct InputModel {
    // Model type (e.g. 'dft'). Accepted but not consumed — Kohn-Sham DFT is the only supported model
    std::optional<std::string> type;
    // Model subtype (e.g. 'gga'). Accepted but not consumed
    std::optional<std::string> subtype;
    // Exchange-correlation functional, matched case-insensitively and recorded lower-cased: 'PZ' (alias 'LDA'), 'PBE', 'PBEsol', 'BLYP', 'revPBE', 'RPBE'; 'libxc:<X>:<C>' where each half is a LibXC id or a LibXC name (e.g. 'libxc:117:130' or 'libxc:gga_x_rpbe:gga_c_pbe'); or QE's 'XC-iii-jjj-kkk-lll' form. Unrecognized values throw. Omitted, the run uses the functional the pseudopotentials state, as pw.x does without input_dft; datasets that disagree, or state one Q3 does not map, are refused
    std::optional<std::string> functional;
    // Semi-empirical dispersion correction, lower-cased on parse: 'd3bj' = DFT-D3 with Becke-Johnson damping
    std::optional<InputModelDispersionEnum> dispersion;
};

inline void to_json(nlohmann::json& json_output, const InputModel& data) {
    json_output = nlohmann::json::object();
    if (data.type) {
        json_output["type"] = *data.type;
    }
    if (data.subtype) {
        json_output["subtype"] = *data.subtype;
    }
    if (data.functional) {
        json_output["functional"] = *data.functional;
    }
    if (data.dispersion) {
        json_output["dispersion"] = *data.dispersion;
    }
}

inline void from_json(const nlohmann::json& json_input, InputModel& data) {
    if (json_input.contains("type")) {
        data.type = json_input.at("type").get<std::string>();
    }
    if (json_input.contains("subtype")) {
        data.subtype = json_input.at("subtype").get<std::string>();
    }
    if (json_input.contains("functional")) {
        data.functional = json_input.at("functional").get<std::string>();
    }
    if (json_input.contains("dispersion")) {
        data.dispersion = json_input.at("dispersion").get<InputModelDispersionEnum>();
    }
}

/**
 */
struct InputMethodCutoffs {
    // Plane-wave kinetic energy cutoff for wavefunctions in Rydberg
    std::optional<double> wavefunction;
    // Plane-wave kinetic energy cutoff for the charge density in Rydberg. 0 = auto-derived from the wavefunction cutoff
    std::optional<double> density;
};

inline void to_json(nlohmann::json& json_output, const InputMethodCutoffs& data) {
    json_output = nlohmann::json::object();
    if (data.wavefunction) {
        json_output["wavefunction"] = *data.wavefunction;
    }
    if (data.density) {
        json_output["density"] = *data.density;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodCutoffs& data) {
    if (json_input.contains("wavefunction")) {
        data.wavefunction = json_input.at("wavefunction").get<double>();
    }
    if (json_input.contains("density")) {
        data.density = json_input.at("density").get<double>();
    }
}

/**
 * Davidson eigensolver settings
 */
struct InputMethodScfDavidson {
    // Davidson working basis size as a multiple of the band count. The solver holds two arrays of this many vectors, so it is the dominant term in its memory. Defaults to 2; 4 restarts less often at twice the footprint, and was the default before 2026-09-13. A non-self-consistent run uses 4 unless this is set: a single pass at 2 can return a spectrum with an eigenvalue missing, so setting it lower there is a memory choice the run warns about.
    std::optional<int64_t> basis_expansion_factor;
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfDavidson& data) {
    json_output = nlohmann::json::object();
    if (data.basis_expansion_factor) {
        json_output["basis_expansion_factor"] = *data.basis_expansion_factor;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfDavidson& data) {
    if (json_input.contains("basis_expansion_factor")) {
        data.basis_expansion_factor = json_input.at("basis_expansion_factor").get<int64_t>();
    }
}

/**
 */
struct InputMethodScfSmearing {
    // Occupation smearing scheme, with QE's aliases: none/fixed (step function), gaussian, methfessel-paxton, marzari-vanderbilt (cold), fermi-dirac
    std::optional<InputMethodScfSmearingTypeEnum> type;
    // Smearing broadening (QE degauss) in Rydberg
    std::optional<double> degauss;
    // Hermite expansion order N of Methfessel-Paxton smearing (Phys. Rev. B 40, 3616); valid only with a Methfessel-Paxton type
    std::optional<int64_t> order;
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfSmearing& data) {
    json_output = nlohmann::json::object();
    if (data.type) {
        json_output["type"] = *data.type;
    }
    if (data.degauss) {
        json_output["degauss"] = *data.degauss;
    }
    if (data.order) {
        json_output["order"] = *data.order;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfSmearing& data) {
    if (json_input.contains("type")) {
        data.type = json_input.at("type").get<InputMethodScfSmearingTypeEnum>();
    }
    if (json_input.contains("degauss")) {
        data.degauss = json_input.at("degauss").get<double>();
    }
    if (json_input.contains("order")) {
        data.order = json_input.at("order").get<int64_t>();
    }
}

/**
 */
struct InputMethodScfMixing {
    // Density mixing algorithm: 'broyden' (G-space quasi-Newton), 'linear' (plain G-space), or 'anderson' (real-space Pulay/DIIS)
    std::optional<InputMethodScfMixingModeEnum> mode;
    // Mixing parameter (0 = no mixing, 1 = full replacement)
    std::optional<double> beta;
    // Number of previous iterations kept for Broyden/Anderson mixing
    std::optional<int64_t> ndim;
    // Density mixing preconditioning. 'default' resolves to 'kerker' when smearing is enabled and 'none' otherwise
    std::optional<InputMethodScfMixingPreconditioningEnum> preconditioning;
    // Thomas-Fermi screening wavevector for Kerker preconditioning, in inverse bohr
    std::optional<double> q0;
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfMixing& data) {
    json_output = nlohmann::json::object();
    if (data.mode) {
        json_output["mode"] = *data.mode;
    }
    if (data.beta) {
        json_output["beta"] = *data.beta;
    }
    if (data.ndim) {
        json_output["ndim"] = *data.ndim;
    }
    if (data.preconditioning) {
        json_output["preconditioning"] = *data.preconditioning;
    }
    if (data.q0) {
        json_output["q0"] = *data.q0;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfMixing& data) {
    if (json_input.contains("mode")) {
        data.mode = json_input.at("mode").get<InputMethodScfMixingModeEnum>();
    }
    if (json_input.contains("beta")) {
        data.beta = json_input.at("beta").get<double>();
    }
    if (json_input.contains("ndim")) {
        data.ndim = json_input.at("ndim").get<int64_t>();
    }
    if (json_input.contains("preconditioning")) {
        data.preconditioning = json_input.at("preconditioning").get<InputMethodScfMixingPreconditioningEnum>();
    }
    if (json_input.contains("q0")) {
        data.q0 = json_input.at("q0").get<double>();
    }
}

/**
 */
struct InputMethodScfHubbardItem {
    // Element symbol matching a species label in the material basis
    std::string species;
    // Orbital shell label (e.g. '3d', 'D', '4f') matching a UPF PP_CHI label, or a bare orbital letter selecting on angular momentum
    std::string orbital_label;
    // Hubbard U parameter in electron volts (converted to Rydberg internally)
    double u_ev{};
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfHubbardItem& data) {
    json_output = nlohmann::json::object();
    json_output["species"] = data.species;
    json_output["orbital_label"] = data.orbital_label;
    json_output["u_ev"] = data.u_ev;
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfHubbardItem& data) {
    json_input.at("species").get_to(data.species);
    json_input.at("orbital_label").get_to(data.orbital_label);
    json_input.at("u_ev").get_to(data.u_ev);
}

/**
 */
struct InputMethodScfHubbardStartingEigenvalue {
    // Species symbol, matched as written in the structure
    std::string species;
    // 1-based eigenvalue index (ascending order, 1 = lowest occupation)
    int64_t eigenvalue{};
    // Spin channel: 1 or 2
    InputMethodScfHubbardStartingEigenvalueSpinEnum spin;
    // Target occupation in [0, 1] (per-spin convention)
    double value{};
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfHubbardStartingEigenvalue& data) {
    json_output = nlohmann::json::object();
    json_output["species"] = data.species;
    json_output["eigenvalue"] = data.eigenvalue;
    json_output["spin"] = data.spin;
    json_output["value"] = data.value;
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfHubbardStartingEigenvalue& data) {
    json_input.at("species").get_to(data.species);
    json_input.at("eigenvalue").get_to(data.eigenvalue);
    json_input.at("spin").get_to(data.spin);
    json_input.at("value").get_to(data.value);
}

/**
 */
struct InputMethodScfKgrid {
    // Monkhorst-Pack grid dimensions [n1, n2, n3]
    std::optional<std::vector<int64_t>> dimensions;
    // K-point grid shifts [s1, s2, s3], each 0 or 1
    std::optional<std::vector<InputMethodScfKgridShiftsItemEnum>> shifts;
    // Irreducible-Brillouin-zone reduction: 'internal' (Q3 native) or 'spglib' (external library)
    std::optional<InputMethodScfKgridIbzReductionEnum> ibz_reduction;
};

inline void to_json(nlohmann::json& json_output, const InputMethodScfKgrid& data) {
    json_output = nlohmann::json::object();
    if (data.dimensions) {
        json_output["dimensions"] = *data.dimensions;
    }
    if (data.shifts) {
        json_output["shifts"] = *data.shifts;
    }
    if (data.ibz_reduction) {
        json_output["ibz_reduction"] = *data.ibz_reduction;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScfKgrid& data) {
    if (json_input.contains("dimensions")) {
        data.dimensions = json_input.at("dimensions").get<std::vector<int64_t>>();
    }
    if (json_input.contains("shifts")) {
        data.shifts = json_input.at("shifts").get<std::vector<InputMethodScfKgridShiftsItemEnum>>();
    }
    if (json_input.contains("ibz_reduction")) {
        data.ibz_reduction = json_input.at("ibz_reduction").get<InputMethodScfKgridIbzReductionEnum>();
    }
}

/**
 * SCF settings. Spelled either 'scf' or 'self_consistent_field'; both are the same block. The abbreviation 'nspin' is deliberately absent — the parser rejects it and points at number_of_spin_components.
 */
struct InputMethodScf {
    // SCF convergence threshold in Rydberg
    std::optional<double> convergence_threshold;
    // Maximum number of SCF iterations
    std::optional<int64_t> max_iterations;
    // Eigensolver: 'davidson' (iterative) or 'direct' (dense LAPACK; incompatible with DFT+U)
    std::optional<InputMethodScfDiagonalizationEnum> diagonalization;
    // Davidson eigensolver settings
    std::optional<InputMethodScfDavidson> davidson;
    // Number of Kohn-Sham states to compute. 0 = auto-determined from electrons + empty states
    std::optional<int64_t> number_of_bands;
    // Net cell charge in electrons removed (QE tot_charge convention): positive = cationic (electrons removed), negative = anionic (electrons added)
    std::optional<double> total_charge;
    // Number of spin components: 1 = unpolarized, 2 = collinear spin-polarized (LSDA)
    std::optional<InputMethodScfNumberOfSpinComponentsEnum> number_of_spin_components;
    // Initial magnetization per species, keyed by the species symbol as written in the material basis (so Fe1/Fe2 AFM sublattices are separable). QE convention: values in [-1, 1], negative = opposite sublattice
    std::optional<std::map<std::string, double>> starting_magnetization;
    // Initial magnetization per atom, one value per atom of the material basis in order (QE convention, [-1, 1]). Lets atoms of one species start antiparallel (same-species antiferromagnet). Exclusive with starting_magnetization
    std::optional<std::vector<double>> atomic_starting_magnetization;
    std::optional<InputMethodScfSmearing> smearing;
    std::optional<InputMethodScfMixing> mixing;
    // DFT+U (Dudarev) corrections per species/orbital shell. Empty = no Hubbard correction
    std::optional<std::vector<InputMethodScfHubbardItem>> hubbard;
    // Projector convention: 'atomic' (raw pseudo-atomic UPF orbitals, QE HUBBARD {atomic}) or 'ortho-atomic' (Lowdin-orthogonalized, QE HUBBARD {ortho-atomic})
    std::optional<InputMethodScfHubbardProjectorsEnum> hubbard_projectors;
    // First-iteration occupation-eigenvalue overrides (QE starting_ns_eigenvalue analog). Empty = no seeding
    std::optional<std::vector<InputMethodScfHubbardStartingEigenvalue>> hubbard_starting_eigenvalues;
    // SCF iterations for which the seeded occupation matrix is re-imposed (QE mixing_fixed_ns). 0 = seed the first iteration only
    std::optional<int64_t> hubbard_fixed_occupation_iterations;
    std::optional<InputMethodScfKgrid> kgrid;
};

inline void to_json(nlohmann::json& json_output, const InputMethodScf& data) {
    json_output = nlohmann::json::object();
    if (data.convergence_threshold) {
        json_output["convergence_threshold"] = *data.convergence_threshold;
    }
    if (data.max_iterations) {
        json_output["max_iterations"] = *data.max_iterations;
    }
    if (data.diagonalization) {
        json_output["diagonalization"] = *data.diagonalization;
    }
    if (data.davidson) {
        json_output["davidson"] = *data.davidson;
    }
    if (data.number_of_bands) {
        json_output["number_of_bands"] = *data.number_of_bands;
    }
    if (data.total_charge) {
        json_output["total_charge"] = *data.total_charge;
    }
    if (data.number_of_spin_components) {
        json_output["number_of_spin_components"] = *data.number_of_spin_components;
    }
    if (data.starting_magnetization) {
        json_output["starting_magnetization"] = *data.starting_magnetization;
    }
    if (data.atomic_starting_magnetization) {
        json_output["atomic_starting_magnetization"] = *data.atomic_starting_magnetization;
    }
    if (data.smearing) {
        json_output["smearing"] = *data.smearing;
    }
    if (data.mixing) {
        json_output["mixing"] = *data.mixing;
    }
    if (data.hubbard) {
        json_output["hubbard"] = *data.hubbard;
    }
    if (data.hubbard_projectors) {
        json_output["hubbard_projectors"] = *data.hubbard_projectors;
    }
    if (data.hubbard_starting_eigenvalues) {
        json_output["hubbard_starting_eigenvalues"] = *data.hubbard_starting_eigenvalues;
    }
    if (data.hubbard_fixed_occupation_iterations) {
        json_output["hubbard_fixed_occupation_iterations"] = *data.hubbard_fixed_occupation_iterations;
    }
    if (data.kgrid) {
        json_output["kgrid"] = *data.kgrid;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodScf& data) {
    if (json_input.contains("convergence_threshold")) {
        data.convergence_threshold = json_input.at("convergence_threshold").get<double>();
    }
    if (json_input.contains("max_iterations")) {
        data.max_iterations = json_input.at("max_iterations").get<int64_t>();
    }
    if (json_input.contains("diagonalization")) {
        data.diagonalization = json_input.at("diagonalization").get<InputMethodScfDiagonalizationEnum>();
    }
    if (json_input.contains("davidson")) {
        data.davidson = json_input.at("davidson").get<InputMethodScfDavidson>();
    }
    if (json_input.contains("number_of_bands")) {
        data.number_of_bands = json_input.at("number_of_bands").get<int64_t>();
    }
    if (json_input.contains("total_charge")) {
        data.total_charge = json_input.at("total_charge").get<double>();
    }
    if (json_input.contains("number_of_spin_components")) {
        data.number_of_spin_components = json_input.at("number_of_spin_components").get<InputMethodScfNumberOfSpinComponentsEnum>();
    }
    if (json_input.contains("starting_magnetization")) {
        data.starting_magnetization = json_input.at("starting_magnetization").get<std::map<std::string, double>>();
    }
    if (json_input.contains("atomic_starting_magnetization")) {
        data.atomic_starting_magnetization = json_input.at("atomic_starting_magnetization").get<std::vector<double>>();
    }
    if (json_input.contains("smearing")) {
        data.smearing = json_input.at("smearing").get<InputMethodScfSmearing>();
    }
    if (json_input.contains("mixing")) {
        data.mixing = json_input.at("mixing").get<InputMethodScfMixing>();
    }
    if (json_input.contains("hubbard")) {
        data.hubbard = json_input.at("hubbard").get<std::vector<InputMethodScfHubbardItem>>();
    }
    if (json_input.contains("hubbard_projectors")) {
        data.hubbard_projectors = json_input.at("hubbard_projectors").get<InputMethodScfHubbardProjectorsEnum>();
    }
    if (json_input.contains("hubbard_starting_eigenvalues")) {
        data.hubbard_starting_eigenvalues = json_input.at("hubbard_starting_eigenvalues").get<std::vector<InputMethodScfHubbardStartingEigenvalue>>();
    }
    if (json_input.contains("hubbard_fixed_occupation_iterations")) {
        data.hubbard_fixed_occupation_iterations = json_input.at("hubbard_fixed_occupation_iterations").get<int64_t>();
    }
    if (json_input.contains("kgrid")) {
        data.kgrid = json_input.at("kgrid").get<InputMethodScfKgrid>();
    }
}

/**
 */
struct InputMethodBandStructure {
    // Number of bands for a band-structure run; overrides method.scf.number_of_bands
    std::optional<int64_t> number_of_bands;
};

inline void to_json(nlohmann::json& json_output, const InputMethodBandStructure& data) {
    json_output = nlohmann::json::object();
    if (data.number_of_bands) {
        json_output["number_of_bands"] = *data.number_of_bands;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethodBandStructure& data) {
    if (json_input.contains("number_of_bands")) {
        data.number_of_bands = json_input.at("number_of_bands").get<int64_t>();
    }
}

/**
 */
struct InputMethod {
    // Method type (e.g. 'pseudopotential'). Accepted but not consumed — planewave-pseudopotential is the only supported method
    std::optional<std::string> type;
    // Free-form dataset label (e.g. 'norm-conserving', 'ultrasoft', 'paw'). Accepted but not consumed — the actual dataset kind is read from the pseudopotential files
    std::optional<std::string> subtype;
    // Element symbol to pseudopotential file path, resolved relative to the manifest's directory
    std::optional<std::map<std::string, std::string>> pseudopotentials;
    std::optional<InputMethodCutoffs> cutoffs;
    // SCF settings. Spelled either 'scf' or 'self_consistent_field'; both are the same block. The abbreviation 'nspin' is deliberately absent — the parser rejects it and points at number_of_spin_components.
    std::optional<InputMethodScf> scf;
    // SCF settings. Spelled either 'scf' or 'self_consistent_field'; both are the same block. The abbreviation 'nspin' is deliberately absent — the parser rejects it and points at number_of_spin_components.
    std::optional<InputMethodScf> self_consistent_field;
    std::optional<InputMethodBandStructure> band_structure;
};

inline void to_json(nlohmann::json& json_output, const InputMethod& data) {
    json_output = nlohmann::json::object();
    if (data.type) {
        json_output["type"] = *data.type;
    }
    if (data.subtype) {
        json_output["subtype"] = *data.subtype;
    }
    if (data.pseudopotentials) {
        json_output["pseudopotentials"] = *data.pseudopotentials;
    }
    if (data.cutoffs) {
        json_output["cutoffs"] = *data.cutoffs;
    }
    if (data.scf) {
        json_output["scf"] = *data.scf;
    }
    if (data.self_consistent_field) {
        json_output["self_consistent_field"] = *data.self_consistent_field;
    }
    if (data.band_structure) {
        json_output["band_structure"] = *data.band_structure;
    }
}

inline void from_json(const nlohmann::json& json_input, InputMethod& data) {
    if (json_input.contains("type")) {
        data.type = json_input.at("type").get<std::string>();
    }
    if (json_input.contains("subtype")) {
        data.subtype = json_input.at("subtype").get<std::string>();
    }
    if (json_input.contains("pseudopotentials")) {
        data.pseudopotentials = json_input.at("pseudopotentials").get<std::map<std::string, std::string>>();
    }
    if (json_input.contains("cutoffs")) {
        data.cutoffs = json_input.at("cutoffs").get<InputMethodCutoffs>();
    }
    if (json_input.contains("scf")) {
        data.scf = json_input.at("scf").get<InputMethodScf>();
    }
    if (json_input.contains("self_consistent_field")) {
        data.self_consistent_field = json_input.at("self_consistent_field").get<InputMethodScf>();
    }
    if (json_input.contains("band_structure")) {
        data.band_structure = json_input.at("band_structure").get<InputMethodBandStructure>();
    }
}

/**
 * Band-structure interpolation from a small model built out of a finished calculation. On an NSCF run (calculation: nscf) the block asks the engine to build the model and save it as <prefix>.save/interpolation.q3ip; on calculation: interpolation it names the saved model that q3_utility interpolate evaluates along control.band_path / control.kpoints_list and on dos_mesh. Each key belongs to exactly one of those two steps and is REFUSED on the other rather than ignored: grid, bands, projection_centers, projection_decay_bohr and export_formats are read when the record is captured, because they decide what goes into it; dos_mesh, disentanglement, localization, projectability_minimum, projectability_maximum and frozen_window_above_fermi_ev are read when the model is evaluated, because the record deliberately stores the untouched window so one capture serves several answers. method is given on both
 */
struct InputControlInterpolation {
    // How the model is built. 'wannier': Wannier functions in the one-shot projection gauge from the interface matrices of a full-grid NSCF run
    std::optional<InputControlInterpolationMethodEnum> method;
    // [n1, n2, n3], the full, unreduced k-point grid control.kpoints_list enumerates. Required on the NSCF build step, and stated explicitly because an explicit k-point list carries no grid shape of its own
    std::optional<std::vector<int64_t>> grid;
    // [first, last], the 1-based inclusive band range the model is built from. Absent means the lowest bands, which is what an isolated valence manifold needs. Give a range when the manifold does not start at band 1 -- a pseudopotential carrying semicore states puts them below it. The range must hold exactly as many bands as there are trial orbitals, and the run must compute at least 'last' bands
    std::optional<std::vector<int64_t>> bands;
    // How the N states the gauge is built from are chosen out of the band window. 'none' requires the window to be exactly those N states, which is what an isolated manifold gives. 'projectability' classifies every band by its projectability and energy and then chooses the smoothest subspace across the zone, which is what an entangled manifold -- a transition metal's d and s bands -- needs
    std::optional<InputControlInterpolationDisentanglementEnum> disentanglement;
    // How the gauge inside the chosen subspace is fixed. Minimising the Marzari-Vanderbilt spread makes the Wannier functions as localised as the mesh allows, so H(R) decays quickly enough to interpolate BETWEEN the mesh points; the one-shot projection gauge is exact AT them whatever it does. 'auto' minimises when the record carries the neighbour overlaps M_mn(k,b) the spread is defined through, and keeps the projection gauge when it does not -- a record imported from another code's projections and eigenvalues alone carries none. 'maximal' demands it and makes such a record an error. 'none' always keeps the projection gauge, which is useful to isolate what localisation contributes
    std::optional<InputControlInterpolationLocalizationEnum> localization;
    // Disentanglement: a band whose projectability falls below this is discarded from the window entirely, unless the frozen energy window kept it
    std::optional<double> projectability_minimum;
    // Disentanglement: a band whose projectability reaches this is frozen, kept exactly and never rotated by the optimisation
    std::optional<double> projectability_maximum;
    // Disentanglement: energy above the Fermi level, in electronvolts, below which a band is frozen whatever its projectability -- short of the singular-gauge rule: a band in the window whose projectability is below projectability_minimum is one the trial orbitals cannot represent, and it is excluded from the freeze and reported rather than frozen, because freezing it makes the gauge singular. Not redundant with projectability_maximum: a metal's Fermi-surface states are the least projectable ones, and discarding them leaves a hole the mesh-point round trip cannot see. 'auto', the default, applies the published 2 eV for atomic projectors to metal and insulator alike: it is the window the band distance that judges the model is measured on, and the two must coincide -- silicon at 12x12x12 gives 220 meV with the occupied manifold alone frozen and 14 meV at E_F + 2 eV, the whole difference being the lowest conduction band. For an insulator the Fermi level the engine reports sits in the gap, so the window reaches the conduction states the distance weighs and no further. Whether the system is a metal (a band crossing the Fermi level) is decided and reported either way. Any real number overrides the rule, and a value below the whole manifold freezes nothing by energy
    std::optional<nlohmann::json> frozen_window_above_fermi_ev;
    // Evaluation step: the spread minimiser's iteration budget. Far more than a well-posed case needs (copper converges in about sixty); the retry ladder doubles it after a stall, which is the one reason it is a key
    std::optional<int64_t> localization_maximum_iterations;
    // Evaluation step: the subspace iteration's budget. A disentanglement that exhausts it is an error, never a subspace returned as if it had converged; the error prints the last five gauge-invariant spreads
    std::optional<int64_t> disentanglement_maximum_iterations;
    // Evaluation step: the relative change in the gauge-invariant spread the subspace iteration must stay below, over five iterations, to count as converged. The automatic workflow's retry ladder loosens it for a disentanglement that exhausted its budget, to a margin above the change among the spreads that failure printed
    std::optional<double> disentanglement_convergence_tolerance;
    // Build step: per species, the 0-based indices of the pseudo-atomic shells to use as trial orbitals, in the order the pseudopotential file lists its atomic wavefunctions. A species not named keeps every shell its pseudopotential does not mark unusable (a negative occupation, or a zero-norm radial function). Refused together with projection_centers. The automatic workflow always writes it, naming the shells its rule keeps: every occupied shell, and the empty ones that are not a second channel of an l the atom has in valence
    std::optional<std::map<std::string, std::vector<int64_t>>> trial_shells;
    // Wannier: centres of the s-type trial orbitals, in crystal (fractional) coordinates. The count fixes the number of Wannier functions, so it must not exceed the band count. Empty falls back to the pseudopotentials' atom-centred atomic orbitals
    std::optional<std::vector<std::vector<double>>> projection_centers;
    // Wannier: decay length of those trial orbitals in bohr (Wannier90's zona is its reciprocal)
    std::optional<double> projection_decay_bohr;
    // Interoperability formats written beside the native record on the NSCF build step. 'wannier90': <prefix>.mmn/.amn/.eig/.nnkp for wannier90.x and the tools built on it
    std::optional<std::vector<InputControlInterpolationExportFormatsItemEnum>> export_formats;
    // Evaluation step: the dense uniform k-mesh the interpolated density of states is sampled on, binned with control.dos_broadening. Omit for no DOS
    std::optional<std::vector<int64_t>> dos_mesh;
};

inline void to_json(nlohmann::json& json_output, const InputControlInterpolation& data) {
    json_output = nlohmann::json::object();
    if (data.method) {
        json_output["method"] = *data.method;
    }
    if (data.grid) {
        json_output["grid"] = *data.grid;
    }
    if (data.bands) {
        json_output["bands"] = *data.bands;
    }
    if (data.disentanglement) {
        json_output["disentanglement"] = *data.disentanglement;
    }
    if (data.localization) {
        json_output["localization"] = *data.localization;
    }
    if (data.projectability_minimum) {
        json_output["projectability_minimum"] = *data.projectability_minimum;
    }
    if (data.projectability_maximum) {
        json_output["projectability_maximum"] = *data.projectability_maximum;
    }
    if (data.frozen_window_above_fermi_ev) {
        json_output["frozen_window_above_fermi_ev"] = *data.frozen_window_above_fermi_ev;
    }
    if (data.localization_maximum_iterations) {
        json_output["localization_maximum_iterations"] = *data.localization_maximum_iterations;
    }
    if (data.disentanglement_maximum_iterations) {
        json_output["disentanglement_maximum_iterations"] = *data.disentanglement_maximum_iterations;
    }
    if (data.disentanglement_convergence_tolerance) {
        json_output["disentanglement_convergence_tolerance"] = *data.disentanglement_convergence_tolerance;
    }
    if (data.trial_shells) {
        json_output["trial_shells"] = *data.trial_shells;
    }
    if (data.projection_centers) {
        json_output["projection_centers"] = *data.projection_centers;
    }
    if (data.projection_decay_bohr) {
        json_output["projection_decay_bohr"] = *data.projection_decay_bohr;
    }
    if (data.export_formats) {
        json_output["export_formats"] = *data.export_formats;
    }
    if (data.dos_mesh) {
        json_output["dos_mesh"] = *data.dos_mesh;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControlInterpolation& data) {
    if (json_input.contains("method")) {
        data.method = json_input.at("method").get<InputControlInterpolationMethodEnum>();
    }
    if (json_input.contains("grid")) {
        data.grid = json_input.at("grid").get<std::vector<int64_t>>();
    }
    if (json_input.contains("bands")) {
        data.bands = json_input.at("bands").get<std::vector<int64_t>>();
    }
    if (json_input.contains("disentanglement")) {
        data.disentanglement = json_input.at("disentanglement").get<InputControlInterpolationDisentanglementEnum>();
    }
    if (json_input.contains("localization")) {
        data.localization = json_input.at("localization").get<InputControlInterpolationLocalizationEnum>();
    }
    if (json_input.contains("projectability_minimum")) {
        data.projectability_minimum = json_input.at("projectability_minimum").get<double>();
    }
    if (json_input.contains("projectability_maximum")) {
        data.projectability_maximum = json_input.at("projectability_maximum").get<double>();
    }
    if (json_input.contains("frozen_window_above_fermi_ev")) {
        data.frozen_window_above_fermi_ev = json_input.at("frozen_window_above_fermi_ev").get<nlohmann::json>();
    }
    if (json_input.contains("localization_maximum_iterations")) {
        data.localization_maximum_iterations = json_input.at("localization_maximum_iterations").get<int64_t>();
    }
    if (json_input.contains("disentanglement_maximum_iterations")) {
        data.disentanglement_maximum_iterations = json_input.at("disentanglement_maximum_iterations").get<int64_t>();
    }
    if (json_input.contains("disentanglement_convergence_tolerance")) {
        data.disentanglement_convergence_tolerance = json_input.at("disentanglement_convergence_tolerance").get<double>();
    }
    if (json_input.contains("trial_shells")) {
        data.trial_shells = json_input.at("trial_shells").get<std::map<std::string, std::vector<int64_t>>>();
    }
    if (json_input.contains("projection_centers")) {
        data.projection_centers = json_input.at("projection_centers").get<std::vector<std::vector<double>>>();
    }
    if (json_input.contains("projection_decay_bohr")) {
        data.projection_decay_bohr = json_input.at("projection_decay_bohr").get<double>();
    }
    if (json_input.contains("export_formats")) {
        data.export_formats = json_input.at("export_formats").get<std::vector<InputControlInterpolationExportFormatsItemEnum>>();
    }
    if (json_input.contains("dos_mesh")) {
        data.dos_mesh = json_input.at("dos_mesh").get<std::vector<int64_t>>();
    }
}

/**
 */
struct InputControl {
    // Type of calculation to perform. The short spellings map to the canonical long ones on parse. 'interpolation' names an input that q3_utility interpolate evaluates from a saved model (see control.interpolation); the engine itself refuses it, as it does the Python-orchestrated band_structure/relaxation/density_of_states
    std::optional<InputControlCalculationEnum> calculation;
    // Output directory
    std::optional<std::string> outdir;
    // Prefix for output files
    std::optional<std::string> prefix;
    // How much the run prints: 'default' reports the physics and the SCF trajectory, 'high' adds the per-iteration and per-phase diagnostics (Davidson micro-iterations, timing breakdowns, mixer internals). Quantum ESPRESSO's verbosity maps onto this: 'high'/'debug' select 'high', 'low'/'medium'/'minimal'/'default' select 'default'. An explicit --log-level overrides it
    std::optional<InputControlVerbosityEnum> verbosity;
    // If true, use no symmetry at all: the k-point mesh is not reduced, and the density, forces and stress are not symmetrized (Quantum ESPRESSO nosym + noinv)
    std::optional<bool> disable_symmetry;
    // Method for nonlocal stress derivatives: 'analytical' (Lagrange polynomial differentiation, matches QE) or 'numerical' (finite-difference)
    std::optional<InputControlNonlocalStressDerivativeMethodEnum> nonlocal_stress_derivative_method;
    // Number of k-point pool groups for MPI parallelization. 0 = choose automatically from the irreducible k-point count. An explicit --pools flag overrides this
    std::optional<int64_t> number_of_kpoint_pools;
    // If true, use FFTW_ESTIMATE for bit-reproducible runs at a performance cost. If false (default), use FFTW_MEASURE
    std::optional<bool> use_deterministic_fft_planning;
    // If true, size the wavefunction FFT grid exactly, for 4 x ecutwfc as pw.x does, at 1.4-2.3x the solver time on the silicon and copper examples. If false (default), use the 2 x ecutwfc grid: faster, but V(r)psi(r) and the band density alias, which costs up to mRy against pw.x on hard pseudopotentials at low cutoffs. QE-format input always uses the exact grid
    std::optional<bool> use_exact_wavefunction_grid;
    // Isolated (charged) system correction: 'makov-payne' (QE assume_isolated='makov-payne', cubic lattices only) or 'martyna-tuckerman' (post-SCF evaluation, any lattice)
    std::optional<InputControlIsolatedSystemCorrectionEnum> isolated_system_correction;
    // Mean-field export for many-body perturbation theory: 'berkeleygw' writes WFN/RHO/vxc.dat
    std::optional<InputControlMeanFieldExportEnum> mean_field_export;
    // High-symmetry path for band-structure calculations. Each entry is either a bare label or a {label, coordinates} map
    std::optional<std::vector<nlohmann::json>> band_path;
    // Number of k-points sampled per band-path segment
    std::optional<int64_t> band_path_points_per_segment;
    // Explicit k-point list for NSCF calculations. Takes precedence over method.scf.kgrid when non-empty
    std::optional<std::vector<std::vector<double>>> kpoints_list;
    // Gaussian broadening for the density of states in Rydberg
    std::optional<double> dos_broadening;
    // DOS energy range lower bound in Rydberg. Set to 0 to auto-determine from band energies
    std::optional<double> dos_energy_min;
    // DOS energy range upper bound in Rydberg. Set to 0 to auto-determine from band energies
    std::optional<double> dos_energy_max;
    // Number of energy grid points for the DOS
    std::optional<int64_t> dos_energy_steps;
    // Compute atomic-orbital (ortho-atomic/Lowdin) projections during an NSCF run. Opt-in: adds a Lowdin pass per k-point
    std::optional<bool> compute_projections;
    // Band-structure interpolation from a small model built out of a finished calculation. On an NSCF run (calculation: nscf) the block asks the engine to build the model and save it as <prefix>.save/interpolation.q3ip; on calculation: interpolation it names the saved model that q3_utility interpolate evaluates along control.band_path / control.kpoints_list and on dos_mesh. Each key belongs to exactly one of those two steps and is REFUSED on the other rather than ignored: grid, bands, projection_centers, projection_decay_bohr and export_formats are read when the record is captured, because they decide what goes into it; dos_mesh, disentanglement, localization, projectability_minimum, projectability_maximum and frozen_window_above_fermi_ev are read when the model is evaluated, because the record deliberately stores the untouched window so one capture serves several answers. method is given on both
    std::optional<InputControlInterpolation> interpolation;
    // Maximum ionic relaxation steps
    std::optional<int64_t> relax_max_steps;
    // Force convergence threshold for relaxation in Ry/bohr (max |force|)
    std::optional<double> relax_force_threshold;
    // Energy convergence threshold for relaxation in Rydberg (energy change between steps)
    std::optional<double> relax_energy_threshold;
};

inline void to_json(nlohmann::json& json_output, const InputControl& data) {
    json_output = nlohmann::json::object();
    if (data.calculation) {
        json_output["calculation"] = *data.calculation;
    }
    if (data.outdir) {
        json_output["outdir"] = *data.outdir;
    }
    if (data.prefix) {
        json_output["prefix"] = *data.prefix;
    }
    if (data.verbosity) {
        json_output["verbosity"] = *data.verbosity;
    }
    if (data.disable_symmetry) {
        json_output["disable_symmetry"] = *data.disable_symmetry;
    }
    if (data.nonlocal_stress_derivative_method) {
        json_output["nonlocal_stress_derivative_method"] = *data.nonlocal_stress_derivative_method;
    }
    if (data.number_of_kpoint_pools) {
        json_output["number_of_kpoint_pools"] = *data.number_of_kpoint_pools;
    }
    if (data.use_deterministic_fft_planning) {
        json_output["use_deterministic_fft_planning"] = *data.use_deterministic_fft_planning;
    }
    if (data.use_exact_wavefunction_grid) {
        json_output["use_exact_wavefunction_grid"] = *data.use_exact_wavefunction_grid;
    }
    if (data.isolated_system_correction) {
        json_output["isolated_system_correction"] = *data.isolated_system_correction;
    }
    if (data.mean_field_export) {
        json_output["mean_field_export"] = *data.mean_field_export;
    }
    if (data.band_path) {
        json_output["band_path"] = *data.band_path;
    }
    if (data.band_path_points_per_segment) {
        json_output["band_path_points_per_segment"] = *data.band_path_points_per_segment;
    }
    if (data.kpoints_list) {
        json_output["kpoints_list"] = *data.kpoints_list;
    }
    if (data.dos_broadening) {
        json_output["dos_broadening"] = *data.dos_broadening;
    }
    if (data.dos_energy_min) {
        json_output["dos_energy_min"] = *data.dos_energy_min;
    }
    if (data.dos_energy_max) {
        json_output["dos_energy_max"] = *data.dos_energy_max;
    }
    if (data.dos_energy_steps) {
        json_output["dos_energy_steps"] = *data.dos_energy_steps;
    }
    if (data.compute_projections) {
        json_output["compute_projections"] = *data.compute_projections;
    }
    if (data.interpolation) {
        json_output["interpolation"] = *data.interpolation;
    }
    if (data.relax_max_steps) {
        json_output["relax_max_steps"] = *data.relax_max_steps;
    }
    if (data.relax_force_threshold) {
        json_output["relax_force_threshold"] = *data.relax_force_threshold;
    }
    if (data.relax_energy_threshold) {
        json_output["relax_energy_threshold"] = *data.relax_energy_threshold;
    }
}

inline void from_json(const nlohmann::json& json_input, InputControl& data) {
    if (json_input.contains("calculation")) {
        data.calculation = json_input.at("calculation").get<InputControlCalculationEnum>();
    }
    if (json_input.contains("outdir")) {
        data.outdir = json_input.at("outdir").get<std::string>();
    }
    if (json_input.contains("prefix")) {
        data.prefix = json_input.at("prefix").get<std::string>();
    }
    if (json_input.contains("verbosity")) {
        data.verbosity = json_input.at("verbosity").get<InputControlVerbosityEnum>();
    }
    if (json_input.contains("disable_symmetry")) {
        data.disable_symmetry = json_input.at("disable_symmetry").get<bool>();
    }
    if (json_input.contains("nonlocal_stress_derivative_method")) {
        data.nonlocal_stress_derivative_method = json_input.at("nonlocal_stress_derivative_method").get<InputControlNonlocalStressDerivativeMethodEnum>();
    }
    if (json_input.contains("number_of_kpoint_pools")) {
        data.number_of_kpoint_pools = json_input.at("number_of_kpoint_pools").get<int64_t>();
    }
    if (json_input.contains("use_deterministic_fft_planning")) {
        data.use_deterministic_fft_planning = json_input.at("use_deterministic_fft_planning").get<bool>();
    }
    if (json_input.contains("use_exact_wavefunction_grid")) {
        data.use_exact_wavefunction_grid = json_input.at("use_exact_wavefunction_grid").get<bool>();
    }
    if (json_input.contains("isolated_system_correction")) {
        data.isolated_system_correction = json_input.at("isolated_system_correction").get<InputControlIsolatedSystemCorrectionEnum>();
    }
    if (json_input.contains("mean_field_export")) {
        data.mean_field_export = json_input.at("mean_field_export").get<InputControlMeanFieldExportEnum>();
    }
    if (json_input.contains("band_path")) {
        data.band_path = json_input.at("band_path").get<std::vector<nlohmann::json>>();
    }
    if (json_input.contains("band_path_points_per_segment")) {
        data.band_path_points_per_segment = json_input.at("band_path_points_per_segment").get<int64_t>();
    }
    if (json_input.contains("kpoints_list")) {
        data.kpoints_list = json_input.at("kpoints_list").get<std::vector<std::vector<double>>>();
    }
    if (json_input.contains("dos_broadening")) {
        data.dos_broadening = json_input.at("dos_broadening").get<double>();
    }
    if (json_input.contains("dos_energy_min")) {
        data.dos_energy_min = json_input.at("dos_energy_min").get<double>();
    }
    if (json_input.contains("dos_energy_max")) {
        data.dos_energy_max = json_input.at("dos_energy_max").get<double>();
    }
    if (json_input.contains("dos_energy_steps")) {
        data.dos_energy_steps = json_input.at("dos_energy_steps").get<int64_t>();
    }
    if (json_input.contains("compute_projections")) {
        data.compute_projections = json_input.at("compute_projections").get<bool>();
    }
    if (json_input.contains("interpolation")) {
        data.interpolation = json_input.at("interpolation").get<InputControlInterpolation>();
    }
    if (json_input.contains("relax_max_steps")) {
        data.relax_max_steps = json_input.at("relax_max_steps").get<int64_t>();
    }
    if (json_input.contains("relax_force_threshold")) {
        data.relax_force_threshold = json_input.at("relax_force_threshold").get<double>();
    }
    if (json_input.contains("relax_energy_threshold")) {
        data.relax_energy_threshold = json_input.at("relax_energy_threshold").get<double>();
    }
}

/**
 * Q3 YAML input manifest — the user-facing input file format read by YAMLInputParser (src/io/input/parser/yaml.hpp). Every object here that Q3 owns end to end is closed (additionalProperties: false); those key sets are code-generated into src/generated/q3/input.hpp and are what the parser rejects unknown keys against, so this file is the single source of truth for which keys a manifest may contain.
 *
 * Schema: input.json
 */
struct Input {
    // Crystal structure: either a path to an external ESSE material JSON (resolved relative to the manifest's directory) or an inline map of the same shape. Deliberately not a closed object — an ESSE material legitimately carries fields Q3 does not consume
    nlohmann::json material;
    std::optional<InputModel> model;
    std::optional<InputMethod> method;
    std::optional<InputControl> control;
};

inline void to_json(nlohmann::json& json_output, const Input& data) {
    json_output = nlohmann::json::object();
    json_output["material"] = data.material;
    if (data.model) {
        json_output["model"] = *data.model;
    }
    if (data.method) {
        json_output["method"] = *data.method;
    }
    if (data.control) {
        json_output["control"] = *data.control;
    }
}

inline void from_json(const nlohmann::json& json_input, Input& data) {
    json_input.at("material").get_to(data.material);
    if (json_input.contains("model")) {
        data.model = json_input.at("model").get<InputModel>();
    }
    if (json_input.contains("method")) {
        data.method = json_input.at("method").get<InputMethod>();
    }
    if (json_input.contains("control")) {
        data.control = json_input.at("control").get<InputControl>();
    }
}

/**
 * Key allow-lists taken straight from input.json: one entry per object
 * the schema closes with "additionalProperties": false, keyed by the dotted
 * path to it ("" is the document root, "[]" a step through an array's items).
 *
 * This is what makes the schema the single source of truth for which keys an
 * input file may carry: a parser validates against these lists instead of
 * against hand-written ones, so a key added here reaches the parser by
 * regeneration rather than by someone remembering to edit two places.
 */
namespace input_schema {

inline const std::map<std::string, std::vector<std::string>>& allowed_keys_by_path() {
    static const std::map<std::string, std::vector<std::string>> registry = {
        {"",
         {
             "control",
             "material",
             "method",
             "model"
         }},
        {"control",
         {
             "band_path",
             "band_path_points_per_segment",
             "calculation",
             "compute_projections",
             "disable_symmetry",
             "dos_broadening",
             "dos_energy_max",
             "dos_energy_min",
             "dos_energy_steps",
             "interpolation",
             "isolated_system_correction",
             "kpoints_list",
             "mean_field_export",
             "nonlocal_stress_derivative_method",
             "number_of_kpoint_pools",
             "outdir",
             "prefix",
             "relax_energy_threshold",
             "relax_force_threshold",
             "relax_max_steps",
             "use_deterministic_fft_planning",
             "use_exact_wavefunction_grid",
             "verbosity"
         }},
        {"control.band_path[]",
         {
             "coordinates",
             "label"
         }},
        {"control.interpolation",
         {
             "bands",
             "disentanglement",
             "disentanglement_convergence_tolerance",
             "disentanglement_maximum_iterations",
             "dos_mesh",
             "export_formats",
             "frozen_window_above_fermi_ev",
             "grid",
             "localization",
             "localization_maximum_iterations",
             "method",
             "projectability_maximum",
             "projectability_minimum",
             "projection_centers",
             "projection_decay_bohr",
             "trial_shells"
         }},
        {"method",
         {
             "band_structure",
             "cutoffs",
             "pseudopotentials",
             "scf",
             "self_consistent_field",
             "subtype",
             "type"
         }},
        {"method.band_structure",
         {
             "number_of_bands"
         }},
        {"method.cutoffs",
         {
             "density",
             "wavefunction"
         }},
        {"method.scf",
         {
             "atomic_starting_magnetization",
             "convergence_threshold",
             "davidson",
             "diagonalization",
             "hubbard",
             "hubbard_fixed_occupation_iterations",
             "hubbard_projectors",
             "hubbard_starting_eigenvalues",
             "kgrid",
             "max_iterations",
             "mixing",
             "number_of_bands",
             "number_of_spin_components",
             "smearing",
             "starting_magnetization",
             "total_charge"
         }},
        {"method.scf.davidson",
         {
             "basis_expansion_factor"
         }},
        {"method.scf.hubbard[]",
         {
             "orbital_label",
             "species",
             "u_ev"
         }},
        {"method.scf.hubbard_starting_eigenvalues[]",
         {
             "eigenvalue",
             "species",
             "spin",
             "value"
         }},
        {"method.scf.kgrid",
         {
             "dimensions",
             "ibz_reduction",
             "shifts"
         }},
        {"method.scf.mixing",
         {
             "beta",
             "mode",
             "ndim",
             "preconditioning",
             "q0"
         }},
        {"method.scf.smearing",
         {
             "degauss",
             "order",
             "type"
         }},
        {"method.self_consistent_field",
         {
             "atomic_starting_magnetization",
             "convergence_threshold",
             "davidson",
             "diagonalization",
             "hubbard",
             "hubbard_fixed_occupation_iterations",
             "hubbard_projectors",
             "hubbard_starting_eigenvalues",
             "kgrid",
             "max_iterations",
             "mixing",
             "number_of_bands",
             "number_of_spin_components",
             "smearing",
             "starting_magnetization",
             "total_charge"
         }},
        {"method.self_consistent_field.davidson",
         {
             "basis_expansion_factor"
         }},
        {"method.self_consistent_field.hubbard[]",
         {
             "orbital_label",
             "species",
             "u_ev"
         }},
        {"method.self_consistent_field.hubbard_starting_eigenvalues[]",
         {
             "eigenvalue",
             "species",
             "spin",
             "value"
         }},
        {"method.self_consistent_field.kgrid",
         {
             "dimensions",
             "ibz_reduction",
             "shifts"
         }},
        {"method.self_consistent_field.mixing",
         {
             "beta",
             "mode",
             "ndim",
             "preconditioning",
             "q0"
         }},
        {"method.self_consistent_field.smearing",
         {
             "degauss",
             "order",
             "type"
         }},
        {"model",
         {
             "dispersion",
             "functional",
             "subtype",
             "type"
         }}
    };
    return registry;
}

/**
 * Keys allowed at `schema_path`. Throws if the schema closes no object at
 * that path — a caller asking about a path the schema does not own is a bug
 * in the caller, not bad input.
 */
inline const std::vector<std::string>& allowed_keys(const std::string& schema_path) {
    const auto& registry = allowed_keys_by_path();
    const auto entry = registry.find(schema_path);
    if (entry == registry.end())
        throw std::runtime_error("input.json closes no object at path '" + schema_path +
                                 "'");
    return entry->second;
}

} // namespace input_schema

} // namespace q3::generated
