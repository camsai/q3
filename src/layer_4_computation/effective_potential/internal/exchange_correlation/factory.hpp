#pragma once

/**
 * XCFunctionalFactory — creates XC functional instances from string names.
 *
 * Maps common names (PZ, PBE, PBEsol, BLYP, revPBE, RPBE) to LibXC functional
 * ids, and also supports "libxc:X:C" -- where each half is either a LibXC id or
 * the name LibXC knows the functional by -- and the QE LibXC format
 * "XC-000I-000I-101L-130L". Every form is matched case-insensitively.
 */

#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <xc.h>
#include <xc_funcs.h>

#include "layer_4_computation/effective_potential/internal/exchange_correlation/init.hpp"
#include "layer_4_computation/effective_potential/internal/exchange_correlation/libxc.hpp"

namespace q3 {

/**
 * Factory for XC functionals.
 * Supports: "PZ", "LDA", "PBE", "PBEsol", "BLYP", "revPBE", "RPBE",
 * "libxc:X:C" (ids or LibXC names), or QE's "XC-iii-jjj-kkk-lll" form.
 */
class XCFunctionalFactory {
  public:
    /**
     * Create an XC functional by name.
     * number_of_spin_components: 1=unpolarized, 2=spin-polarized
     */
    static std::unique_ptr<XCFunctional> create(const std::string& name,
                                                int number_of_spin_components = 1) {

        /**
         * Match on a case-folded copy throughout. A parsed input arrives
         * lower-cased, but a configuration built in code can spell the name
         * any way. Before the input was normalized, the YAML parser
         * upper-cased `model.functional`, so `libxc:117:130` reached here as
         * `LIBXC:117:130` and never matched the lower-case prefix test below,
         * and each alias had to be spelled out twice to cover the two cases
         * anyone happened to try.
         */
        const std::string lowered = to_lower(name);

        // Map common names to LibXC IDs
        int exchange_functional_id = -1;
        int correlation_functional_id = -1;
        if (lowered == "pz" || lowered == "sla+pz" || lowered == "lda") {
            exchange_functional_id = XC_LDA_X;
            correlation_functional_id = XC_LDA_C_PZ;
        } else if (lowered == "pbe") {
            exchange_functional_id = XC_GGA_X_PBE;
            correlation_functional_id = XC_GGA_C_PBE;
        } else if (lowered == "pbesol") {
            exchange_functional_id = XC_GGA_X_PBE_SOL;
            correlation_functional_id = XC_GGA_C_PBE_SOL;
        } else if (lowered == "blyp") {
            exchange_functional_id = XC_GGA_X_B88;
            correlation_functional_id = XC_GGA_C_LYP;
        } else if (lowered == "revpbe") {
            exchange_functional_id = XC_GGA_X_PBE_R;
            correlation_functional_id = XC_GGA_C_PBE;
        } else if (lowered == "rpbe") {
            /**
             * Hammer-Hansen-Norskov exchange with PBE correlation -- the pair
             * `libxc:117:130` spells out by number. Hammer, Hansen and Norskov,
             * Phys. Rev. B 59, 7413 (1999).
             */
            exchange_functional_id = XC_GGA_X_RPBE;
            correlation_functional_id = XC_GGA_C_PBE;
        }

        if (exchange_functional_id >= 0 && correlation_functional_id >= 0) {
            return std::make_unique<XC_LibXC>(
                exchange_functional_id, correlation_functional_id, number_of_spin_components, name);
        }

        // Generic "libxc:X:C", each half either a LibXC id or a LibXC name
        if (lowered.size() >= 6 && lowered.substr(0, 6) == "libxc:") {
            auto separator_position = lowered.find(':', 6);
            if (separator_position == std::string::npos) {
                throw std::runtime_error(
                    "LibXC format: 'libxc:EXCHANGE:CORRELATION', where each half is either a "
                    "LibXC id or a LibXC name (e.g. 'libxc:117:130' or "
                    "'libxc:gga_x_rpbe:gga_c_pbe'); got '" +
                    name + "'");
            }
            exchange_functional_id =
                resolve_libxc_component(lowered.substr(6, separator_position - 6), name);
            correlation_functional_id =
                resolve_libxc_component(lowered.substr(separator_position + 1), name);
            return std::make_unique<XC_LibXC>(
                exchange_functional_id, correlation_functional_id, number_of_spin_components, name);
        }

        // QE LibXC format: "XC-xxxx-xxxx-xxxx-xxxx"
        if (name.size() >= 22 && lowered.substr(0, 3) == "xc-" && name[2] == '-' &&
            name[7] == '-' && name[12] == '-' && name[17] == '-') {
            int local_density_exchange_index = 0;
            int local_density_correlation_index = 0;
            int generalized_gradient_exchange_index = 0;
            int generalized_gradient_correlation_index = 0;
            try {
                local_density_exchange_index = std::stoi(name.substr(3, 3));
                local_density_correlation_index = std::stoi(name.substr(8, 3));
                generalized_gradient_exchange_index = std::stoi(name.substr(13, 3));
                generalized_gradient_correlation_index = std::stoi(name.substr(18, 3));
            } catch (const std::invalid_argument&) {
                // Fall through to the unknown-functional error below.
            } catch (const std::out_of_range&) {
                // Fall through to the unknown-functional error below.
            }

            if (local_density_exchange_index > 0) {
                exchange_functional_id = local_density_exchange_index;
            } else if (generalized_gradient_exchange_index > 0) {
                exchange_functional_id = generalized_gradient_exchange_index;
            }

            if (local_density_correlation_index > 0) {
                correlation_functional_id = local_density_correlation_index;
            } else if (generalized_gradient_correlation_index > 0) {
                correlation_functional_id = generalized_gradient_correlation_index;
            }

            if (exchange_functional_id > 0 && correlation_functional_id > 0) {
                return std::make_unique<XC_LibXC>(exchange_functional_id,
                                                  correlation_functional_id,
                                                  number_of_spin_components,
                                                  name);
            }
        }

        throw std::runtime_error(
            "Unknown XC functional: " + name +
            ". Try: PZ, PBE, PBEsol, BLYP, revPBE, RPBE (case-insensitive); or "
            "'libxc:EXCHANGE:CORRELATION' with LibXC ids or names, e.g. 'libxc:117:130' or "
            "'libxc:gga_x_rpbe:gga_c_pbe'; or QE's LibXC form XC-xxxx-xxxx-xxxx-xxxx");
    }

  private:
    static std::string to_lower(std::string text) {
        for (auto& character : text)
            character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
        return text;
    }

    /**
     * One half of a `libxc:` specification, as a LibXC functional id.
     *
     * Accepts either the number or the name LibXC itself knows the functional
     * by. Requiring the number meant reading the id out of `xc_funcs.h` to say
     * something LibXC will answer to directly, and `libxc:117:130` records
     * nothing about what it selects -- `libxc:gga_x_rpbe:gga_c_pbe` is the same
     * functional and says so.
     *
     * `xc_functional_get_number` returns -1 for a name it does not know, which
     * is reported here rather than handed to `xc_func_init` as an id.
     */
    static int resolve_libxc_component(const std::string& component,
                                       const std::string& whole_name) {
        if (component.empty())
            throw std::runtime_error("Empty exchange or correlation field in XC functional '" +
                                     whole_name + "'");

        bool is_number = component.find_first_not_of("0123456789") == std::string::npos;
        if (is_number) {
            try {
                return std::stoi(component);
            } catch (const std::exception&) {
                throw std::runtime_error("XC functional '" + whole_name + "': '" + component +
                                         "' is not a usable LibXC id");
            }
        }

        int resolved = xc_functional_get_number(component.c_str());
        if (resolved < 0)
            throw std::runtime_error("XC functional '" + whole_name +
                                     "': LibXC does not know a functional named '" + component +
                                     "'");
        return resolved;
    }
};

} // namespace q3
