#pragma once

/**
 * Wannier90FormatReader — reads the Wannier90 interface files any plane-wave
 * code can produce (`.mmn`, `.amn`, `.eig`, `.nnkp`) into a
 * WannierInputs.
 *
 * This is what lets the localization engine be validated on matrices produced
 * by ANOTHER code before it is ever run on Q3's own: feeding Quantum
 * ESPRESSO's `pw2wannier90.x` output through this reader and into the
 * minimizer establishes the minimizer's correctness independently of whether
 * Q3's own overlap kernels are right. The two failure modes are then
 * separable, which they would not be if the engine only ever saw Q3 data.
 *
 * Formats follow the Wannier90 user guide (v3.1.0, 5 March 2020), chapter
 * "Interface with underlying electronic structure codes". Three conventions
 * are load-bearing and are checked rather than assumed:
 *
 * - Every index in these files is 1-based (Fortran origin); this reader
 *   converts on the way in and range-checks first, so a writer's off-by-one
 *   surfaces as a thrown error rather than a silently shifted matrix.
 * - A `.mmn` block stores its num_bands^2 elements with the FIRST index m
 *   varying fastest, i.e. column-major. Reading them row-major would
 *   transpose every block -- for a near-Hermitian overlap that survives
 *   inspection, so it must be got right here rather than noticed later.
 * - `.eig` holds eigenvalues in ELECTRON VOLTS while Q3 works in Rydberg;
 *   this reader converts on the way in, since WannierInputs's field
 *   is documented as Rydberg.
 *
 * The mirror-image writers live in writer.hpp; a round-trip
 * through both is one of this pair's tests, but the reader is deliberately
 * written to the format spec rather than to the writer, so a shared
 * misreading of the spec cannot cancel out between them.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"

namespace q3 {

namespace internal::wannier90_format {

/**
 * A header's declared element count must be reachable from the file's size:
 * every element is one text line of at least `minimum_bytes_per_element`
 * bytes, so a header claiming more than the file could hold is corrupt, and
 * would otherwise become a huge allocation before the first element is read.
 */
inline void require_plausible_element_count(const std::string& file_path,
                                            long long elements,
                                            std::uintmax_t minimum_bytes_per_element) {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(file_path, error);
    if (!error && static_cast<long double>(elements) * minimum_bytes_per_element >
                      static_cast<long double>(size)) {
        throw std::runtime_error(file_path + ": the header declares " + std::to_string(elements) +
                                 " elements, more than a " + std::to_string(size) +
                                 "-byte file can hold");
    }
}

} // namespace internal::wannier90_format

class Wannier90FormatReader {
  public:
    /**
     * One (k, b) pair's neighbour identity, as `.mmn`/`.nnkp` record it: the
     * physical k-point that is the periodic image of k+b, and the integer
     * reciprocal-lattice shift carrying that image to the actual k+b.
     */
    struct NeighborEntry {
        int neighbor_kpoint_index = 0;
        IntegerVector3D periodic_image_shift;
    };

    struct OverlapData {
        int number_of_bands = 0;
        int number_of_kpoints = 0;
        int number_of_neighbors = 0;
        /**
         * [kpoint_index][neighbor_index] -> one column-major M block of
         * number_of_bands^2 entries, matching WannierInputs's own
         * storage convention.
         */
        std::vector<std::vector<std::vector<Complex>>> overlaps;
        std::vector<std::vector<NeighborEntry>> neighbors;
    };

    struct ProjectionData {
        int number_of_bands = 0;
        int number_of_kpoints = 0;
        int number_of_wannier_functions = 0;
        /// [kpoint_index] -> column-major A block, band index varying fastest.
        std::vector<std::vector<Complex>> projections;
    };

    struct EigenvalueData {
        int number_of_bands = 0;
        int number_of_kpoints = 0;
        /// [kpoint_index][band_index], converted to Rydberg on the way in.
        std::vector<std::vector<Real>> eigenvalues_rydberg;
    };

    struct NeighborListData {
        /// Rows are a1, a2, a3 in Angstrom, as the `.nnkp` records them.
        Matrix3x3 real_lattice;
        Matrix3x3 reciprocal_lattice;
        /// Crystal (fractional) coordinates, in the file's own k-point order.
        std::vector<Vector3D> kpoints;
    };

    /// eV per Rydberg; `.eig` is eV, WannierInputs is Rydberg.
    static constexpr Real electron_volts_per_rydberg = constants::RY_TO_EV;

    /**
     * Read the `real_lattice`, `recip_lattice` and `kpoints` blocks of a
     * `.nnkp`. The neighbour table itself is not read here: the interpolation
     * path needs only the lattice (for the Wigner-Seitz construction) and the
     * k-point list (for the Fourier transform), while the neighbour table is
     * already carried by the `.mmn` whose overlaps it labels.
     */
    [[nodiscard]] static NeighborListData read_neighbor_list(const std::string& file_path) {
        std::ifstream stream(file_path);
        if (!stream) {
            throw std::runtime_error("cannot open neighbour list file: " + file_path);
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(stream, line)) {
            std::string trimmed = trim(line);
            if (!trimmed.empty()) {
                lines.push_back(trimmed);
            }
        }

        NeighborListData data;
        data.real_lattice = read_lattice_block(file_path, lines, "real_lattice");
        data.reciprocal_lattice = read_lattice_block(file_path, lines, "recip_lattice");

        std::vector<std::string> kpoint_body = extract_block(file_path, lines, "kpoints");
        if (kpoint_body.empty()) {
            throw std::runtime_error(file_path + ": 'kpoints' block is empty");
        }
        int declared_count = 0;
        {
            std::istringstream count_stream(kpoint_body[0]);
            if (!(count_stream >> declared_count) || declared_count <= 0) {
                throw std::runtime_error(file_path +
                                         ": 'kpoints' block must open with a positive count");
            }
        }
        if (static_cast<int>(kpoint_body.size()) - 1 != declared_count) {
            throw std::runtime_error(file_path + ": 'kpoints' block declares " +
                                     std::to_string(declared_count) + " k-points but holds " +
                                     std::to_string(kpoint_body.size() - 1) + " rows");
        }
        data.kpoints.reserve(declared_count);
        for (int index = 0; index < declared_count; index++) {
            std::istringstream row_stream(kpoint_body[index + 1]);
            Vector3D kpoint;
            if (!(row_stream >> kpoint.x >> kpoint.y >> kpoint.z)) {
                throw std::runtime_error(file_path +
                                         ": malformed k-point row: " + kpoint_body[index + 1]);
            }
            data.kpoints.push_back(kpoint);
        }
        return data;
    }

    [[nodiscard]] static OverlapData read_overlaps(const std::string& file_path) {
        std::ifstream stream(file_path);
        if (!stream) {
            throw std::runtime_error("cannot open overlap file: " + file_path);
        }
        std::string comment_line;
        std::getline(stream, comment_line);

        OverlapData data;
        if (!(stream >> data.number_of_bands >> data.number_of_kpoints >>
              data.number_of_neighbors)) {
            throw std::runtime_error(file_path +
                                     ": missing or malformed 'num_bands num_kpts nntot' header");
        }
        require_positive(file_path, "num_bands", data.number_of_bands);
        require_positive(file_path, "num_kpts", data.number_of_kpoints);
        require_positive(file_path, "nntot", data.number_of_neighbors);
        internal::wannier90_format::require_plausible_element_count(
            file_path,
            static_cast<long long>(data.number_of_kpoints) * data.number_of_neighbors *
                data.number_of_bands * data.number_of_bands,
            10);

        data.overlaps.assign(data.number_of_kpoints,
                             std::vector<std::vector<Complex>>(data.number_of_neighbors));
        data.neighbors.assign(data.number_of_kpoints,
                              std::vector<NeighborEntry>(data.number_of_neighbors));

        const long long block_element_count =
            static_cast<long long>(data.number_of_bands) * data.number_of_bands;
        for (int kpoint_index = 0; kpoint_index < data.number_of_kpoints; kpoint_index++) {
            for (int neighbor_index = 0; neighbor_index < data.number_of_neighbors;
                 neighbor_index++) {
                int block_kpoint = 0;
                int neighbor_kpoint = 0;
                IntegerVector3D shift;
                if (!(stream >> block_kpoint >> neighbor_kpoint >> shift.x >> shift.y >> shift.z)) {
                    throw std::runtime_error(
                        file_path + ": file ended before the declared " +
                        std::to_string(static_cast<long long>(data.number_of_kpoints) *
                                       data.number_of_neighbors) +
                        " (k,b) blocks were read");
                }
                /**
                 * Blocks must sweep k-points in order with nntot neighbours
                 * each -- the file carries no other statement of which block
                 * belongs to which k-point, so a writer that reordered them
                 * would silently mis-associate every overlap.
                 */
                if (block_kpoint != kpoint_index + 1) {
                    throw std::runtime_error(
                        file_path + ": blocks must sweep k-points in order; expected k-point " +
                        std::to_string(kpoint_index + 1) + ", found " +
                        std::to_string(block_kpoint));
                }
                data.neighbors[kpoint_index][neighbor_index].neighbor_kpoint_index = to_zero_based(
                    file_path, "neighbour k-point", neighbor_kpoint, data.number_of_kpoints);
                data.neighbors[kpoint_index][neighbor_index].periodic_image_shift = shift;

                std::vector<Complex>& block = data.overlaps[kpoint_index][neighbor_index];
                block.resize(block_element_count);
                /**
                 * m fastest in the file, and WannierInputs stores M
                 * column-major with the k-side band index fastest -- the same
                 * order, so the elements land in sequence with no transpose.
                 */
                for (int element_index = 0; element_index < block_element_count; element_index++) {
                    Real real_part = 0.0;
                    Real imaginary_part = 0.0;
                    if (!(stream >> real_part >> imaginary_part)) {
                        throw std::runtime_error(file_path +
                                                 ": file ended inside a (k,b) overlap block");
                    }
                    block[element_index] = Complex(real_part, imaginary_part);
                }
            }
        }
        return data;
    }

    [[nodiscard]] static ProjectionData read_projections(const std::string& file_path) {
        std::ifstream stream(file_path);
        if (!stream) {
            throw std::runtime_error("cannot open projection file: " + file_path);
        }
        std::string comment_line;
        std::getline(stream, comment_line);

        ProjectionData data;
        if (!(stream >> data.number_of_bands >> data.number_of_kpoints >>
              data.number_of_wannier_functions)) {
            throw std::runtime_error(file_path +
                                     ": missing or malformed 'num_bands num_kpts num_wann' header");
        }
        require_positive(file_path, "num_bands", data.number_of_bands);
        require_positive(file_path, "num_kpts", data.number_of_kpoints);
        require_positive(file_path, "num_wann", data.number_of_wannier_functions);
        internal::wannier90_format::require_plausible_element_count(
            file_path,
            static_cast<long long>(data.number_of_kpoints) * data.number_of_bands *
                data.number_of_wannier_functions,
            10);

        const std::size_t block_size =
            static_cast<std::size_t>(data.number_of_bands) * data.number_of_wannier_functions;
        data.projections.assign(data.number_of_kpoints, std::vector<Complex>(block_size));
        /**
         * Every element carries its own (m, n, k) triple, so nothing is
         * assumed about ordering: fill by index, then require that every slot
         * was written exactly once. A duplicated triple leaves another slot
         * silently at zero, which this catches.
         */
        std::vector<std::vector<bool>> written(data.number_of_kpoints,
                                               std::vector<bool>(block_size, false));

        const long long total_elements = static_cast<long long>(data.number_of_bands) *
                                         data.number_of_wannier_functions * data.number_of_kpoints;
        for (long long element_index = 0; element_index < total_elements; element_index++) {
            int band = 0;
            int wannier = 0;
            int kpoint = 0;
            Real real_part = 0.0;
            Real imaginary_part = 0.0;
            if (!(stream >> band >> wannier >> kpoint >> real_part >> imaginary_part)) {
                throw std::runtime_error(file_path + ": file ended before the declared " +
                                         std::to_string(total_elements) +
                                         " projection elements were read");
            }
            int band_index = to_zero_based(file_path, "band", band, data.number_of_bands);
            int wannier_index =
                to_zero_based(file_path, "Wannier", wannier, data.number_of_wannier_functions);
            int kpoint_index = to_zero_based(file_path, "k-point", kpoint, data.number_of_kpoints);

            std::size_t slot =
                static_cast<std::size_t>(wannier_index) * data.number_of_bands + band_index;
            if (written[kpoint_index][slot]) {
                throw std::runtime_error(
                    file_path + ": duplicate projection entry for band " + std::to_string(band) +
                    ", Wannier " + std::to_string(wannier) + ", k-point " + std::to_string(kpoint));
            }
            data.projections[kpoint_index][slot] = Complex(real_part, imaginary_part);
            written[kpoint_index][slot] = true;
        }
        return data;
    }

    /**
     * `.eig` carries no header, so the dimensions come from the indices
     * present and the index set must then be exactly the full rectangle --
     * a dropped line is an error, never a hole left at zero.
     */
    [[nodiscard]] static EigenvalueData read_eigenvalues(const std::string& file_path) {
        std::ifstream stream(file_path);
        if (!stream) {
            throw std::runtime_error("cannot open eigenvalue file: " + file_path);
        }

        struct Entry {
            int band = 0;
            int kpoint = 0;
            Real value = 0.0;
        };
        std::vector<Entry> entries;
        int maximum_band = 0;
        int maximum_kpoint = 0;

        std::string line;
        while (std::getline(stream, line)) {
            std::istringstream line_stream(line);
            Entry entry;
            if (!(line_stream >> entry.band >> entry.kpoint >> entry.value)) {
                if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
                    continue;
                }
                throw std::runtime_error(file_path +
                                         ": expected 'band kpoint eigenvalue', got: " + line);
            }
            /**
             * operator>> stops at the first character it cannot use and
             * reports success for the prefix it did use. A Fortran
             * D-exponent value such as -0.557929D+01 therefore reads as
             * -0.557929 -- wrong by a power of ten, with no error -- unless
             * the remainder of the line is checked.
             */
            std::string trailing;
            if (line_stream >> trailing) {
                throw std::runtime_error(
                    file_path + ": unexpected '" + trailing +
                    "' after the eigenvalue (Fortran D exponents are not a valid .eig "
                    "notation; this reader accepts E exponents and plain decimals): " +
                    line);
            }
            if (entry.band < 1 || entry.kpoint < 1) {
                throw std::runtime_error(
                    file_path + ": band and k-point indices are 1-based, got band " +
                    std::to_string(entry.band) + ", k-point " + std::to_string(entry.kpoint));
            }
            maximum_band = std::max(maximum_band, entry.band);
            maximum_kpoint = std::max(maximum_kpoint, entry.kpoint);
            entries.push_back(entry);
        }
        if (entries.empty()) {
            throw std::runtime_error(file_path + ": contains no eigenvalue lines");
        }

        /**
         * The table is sized from the largest indices seen, so check that the
         * lines present can fill it before allocating it: a single line
         * "1 2000000000 -5.0" would otherwise ask for a 100 GB table and die
         * of std::bad_alloc instead of this diagnosis.
         */
        if (static_cast<long long>(maximum_band) * maximum_kpoint !=
            static_cast<long long>(entries.size())) {
            throw std::runtime_error(
                file_path + ": " + std::to_string(entries.size()) +
                " eigenvalue lines cannot fill a " + std::to_string(maximum_kpoint) +
                " k-point by " + std::to_string(maximum_band) +
                " band table (the largest indices seen); the file is truncated or an index is "
                "wrong");
        }
        EigenvalueData data;
        data.number_of_bands = maximum_band;
        data.number_of_kpoints = maximum_kpoint;
        data.eigenvalues_rydberg.assign(maximum_kpoint, std::vector<Real>(maximum_band, 0.0));
        std::vector<std::vector<bool>> written(maximum_kpoint,
                                               std::vector<bool>(maximum_band, false));
        for (const Entry& entry : entries) {
            if (written[entry.kpoint - 1][entry.band - 1]) {
                throw std::runtime_error(file_path + ": duplicate eigenvalue for band " +
                                         std::to_string(entry.band) + ", k-point " +
                                         std::to_string(entry.kpoint));
            }
            data.eigenvalues_rydberg[entry.kpoint - 1][entry.band - 1] =
                entry.value / electron_volts_per_rydberg;
            written[entry.kpoint - 1][entry.band - 1] = true;
        }
        for (int kpoint_index = 0; kpoint_index < maximum_kpoint; kpoint_index++) {
            for (int band_index = 0; band_index < maximum_band; band_index++) {
                if (!written[kpoint_index][band_index]) {
                    throw std::runtime_error(
                        file_path + ": index range implies " + std::to_string(maximum_kpoint) +
                        " k-points x " + std::to_string(maximum_band) +
                        " bands, but the entry for band " + std::to_string(band_index + 1) +
                        ", k-point " + std::to_string(kpoint_index + 1) + " is absent");
                }
            }
        }
        return data;
    }

  private:
    static std::string trim(const std::string& text) {
        std::size_t first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        std::size_t last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1);
    }

    static std::string lowercase(std::string text) {
        for (char& character : text) {
            character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
        }
        return text;
    }

    /// Body lines of a `begin <name>` ... `end <name>` block.
    static std::vector<std::string> extract_block(const std::string& file_path,
                                                  const std::vector<std::string>& lines,
                                                  const std::string& name) {
        const std::string opening = "begin " + name;
        const std::string closing = "end " + name;
        std::size_t start = lines.size();
        for (std::size_t index = 0; index < lines.size(); index++) {
            if (lowercase(lines[index]) == opening) {
                start = index;
                break;
            }
        }
        if (start == lines.size()) {
            throw std::runtime_error(file_path + ": no 'begin " + name + "' block found");
        }
        for (std::size_t index = start + 1; index < lines.size(); index++) {
            if (lowercase(lines[index]) == closing) {
                return std::vector<std::string>(lines.begin() + start + 1, lines.begin() + index);
            }
        }
        throw std::runtime_error(file_path + ": 'begin " + name +
                                 "' block is never closed by 'end " + name + "'");
    }

    static Matrix3x3 read_lattice_block(const std::string& file_path,
                                        const std::vector<std::string>& lines,
                                        const std::string& name) {
        std::vector<std::string> body = extract_block(file_path, lines, name);
        if (body.size() != 3) {
            throw std::runtime_error(file_path + ": '" + name +
                                     "' block must hold exactly 3 rows, found " +
                                     std::to_string(body.size()));
        }
        Matrix3x3 lattice;
        for (int row = 0; row < 3; row++) {
            std::istringstream row_stream(body[row]);
            if (!(row_stream >> lattice[row].x >> lattice[row].y >> lattice[row].z)) {
                throw std::runtime_error(file_path + ": malformed '" + name +
                                         "' row: " + body[row]);
            }
        }
        return lattice;
    }

    static void require_positive(const std::string& file_path, const std::string& name, int value) {
        if (value <= 0) {
            throw std::runtime_error(file_path + ": " + name + " must be positive, got " +
                                     std::to_string(value));
        }
    }

    static int to_zero_based(const std::string& file_path,
                             const std::string& name,
                             int value,
                             int upper_bound) {
        if (value < 1 || value > upper_bound) {
            throw std::runtime_error(file_path + ": " + name + " index " + std::to_string(value) +
                                     " outside declared range 1.." + std::to_string(upper_bound));
        }
        return value - 1;
    }
};

} // namespace q3
