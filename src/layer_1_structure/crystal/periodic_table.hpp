#pragma once

/**
 * PeriodicTable: provides atomic number lookup from element symbol.
 *
 * Maps standard element symbols (e.g. "Si", "Fe") to their atomic numbers (Z).
 * Handles symbols with trailing digits by stripping non-alphabetic characters.
 */

#include <cctype>
#include <map>
#include <string>

namespace q3 {

/**
 * Provides element symbol to atomic number (Z) mapping.
 */
class PeriodicTable {
  public:
    /**
     * Return the atomic number for a given element symbol.
     * Strips trailing digits (e.g. "Si1" → "Si").
     * Returns 0 if the element is not found.
     */
    static int atomic_number_by_element_symbol(const std::string& symbol) {
        // Strip trailing digits (e.g. "Si1" → "Si")
        std::string clean;
        for (char character : symbol) {
            if (std::isalpha(character))
                clean += character;
        }

        const auto& table = symbol_to_atomic_number_table();
        auto iterator = table.find(clean);
        if (iterator != table.end())
            return iterator->second;
        return 0;
    }

    /**
     * Return the element symbol for a given atomic number, or an empty string
     * when there is no such element.
     *
     * The inverse of the lookup above, which a file format that identifies its
     * atoms by Z rather than by symbol needs on the way back in — a cube file
     * does (see io/volumetric/cube_reader.hpp).
     */
    static const std::string& element_symbol_by_atomic_number(int atomic_number) {
        static const std::map<int, std::string> symbol_table = [] {
            std::map<int, std::string> inverted;
            for (const auto& [symbol, number] : symbol_to_atomic_number_table())
                inverted.emplace(number, symbol);
            return inverted;
        }();
        static const std::string no_such_element;

        auto iterator = symbol_table.find(atomic_number);
        return iterator == symbol_table.end() ? no_such_element : iterator->second;
    }

  private:
    /**
     * The one element list, shared by both lookups so that neither direction
     * can know an element the other does not.
     */
    static const std::map<std::string, int>& symbol_to_atomic_number_table() {
        static const std::map<std::string, int> atomic_number_table = {
            {"H", 1},    {"He", 2},   {"Li", 3},   {"Be", 4},   {"B", 5},    {"C", 6},
            {"N", 7},    {"O", 8},    {"F", 9},    {"Ne", 10},  {"Na", 11},  {"Mg", 12},
            {"Al", 13},  {"Si", 14},  {"P", 15},   {"S", 16},   {"Cl", 17},  {"Ar", 18},
            {"K", 19},   {"Ca", 20},  {"Sc", 21},  {"Ti", 22},  {"V", 23},   {"Cr", 24},
            {"Mn", 25},  {"Fe", 26},  {"Co", 27},  {"Ni", 28},  {"Cu", 29},  {"Zn", 30},
            {"Ga", 31},  {"Ge", 32},  {"As", 33},  {"Se", 34},  {"Br", 35},  {"Kr", 36},
            {"Rb", 37},  {"Sr", 38},  {"Y", 39},   {"Zr", 40},  {"Nb", 41},  {"Mo", 42},
            {"Tc", 43},  {"Ru", 44},  {"Rh", 45},  {"Pd", 46},  {"Ag", 47},  {"Cd", 48},
            {"In", 49},  {"Sn", 50},  {"Sb", 51},  {"Te", 52},  {"I", 53},   {"Xe", 54},
            {"Cs", 55},  {"Ba", 56},  {"La", 57},  {"Ce", 58},  {"Pr", 59},  {"Nd", 60},
            {"Pm", 61},  {"Sm", 62},  {"Eu", 63},  {"Gd", 64},  {"Tb", 65},  {"Dy", 66},
            {"Ho", 67},  {"Er", 68},  {"Tm", 69},  {"Yb", 70},  {"Lu", 71},  {"Hf", 72},
            {"Ta", 73},  {"W", 74},   {"Re", 75},  {"Os", 76},  {"Ir", 77},  {"Pt", 78},
            {"Au", 79},  {"Hg", 80},  {"Tl", 81},  {"Pb", 82},  {"Bi", 83},  {"Po", 84},
            {"At", 85},  {"Rn", 86},  {"Fr", 87},  {"Ra", 88},  {"Ac", 89},  {"Th", 90},
            {"Pa", 91},  {"U", 92},   {"Np", 93},  {"Pu", 94},  {"Am", 95},  {"Cm", 96},
            {"Bk", 97},  {"Cf", 98},  {"Es", 99},  {"Fm", 100}, {"Md", 101}, {"No", 102},
            {"Lr", 103}, {"Rf", 104}, {"Db", 105}, {"Sg", 106}, {"Bh", 107}, {"Hs", 108},
            {"Mt", 109}, {"Ds", 110}, {"Rg", 111}, {"Cn", 112}, {"Nh", 113}, {"Fl", 114},
            {"Mc", 115}, {"Lv", 116}, {"Ts", 117}, {"Og", 118},
        };
        return atomic_number_table;
    }
};

} // namespace q3
