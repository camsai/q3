#pragma once

/**
 * BerkeleyGWCommonHeader — the 8 binary records shared by BerkeleyGW's WFN
 * and RHO mean-field files.
 *
 * Framing: Fortran `form='unformatted', access='sequential'` via
 * QEBinaryRecordFormat -- every record is `[int32 length][payload][int32
 * length]`, matching QE's own I/O. All reals are float64, all ints int32.
 *
 * Authority. The record order, contents and units are those BerkeleyGW
 * publishes: J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012),
 * Appendix 11.1, "Specification of file formats" (arXiv:1111.4429) -- "the
 * BerkeleyGW specification" below. What it leaves open -- record framing,
 * integer and real kinds, the width of the character fields, the count
 * repeated before each body block, the exact conventions of the symmetry
 * records -- follows Quantum ESPRESSO's pw2bgw.x converter, whose output
 * BerkeleyGW reads. Citations of the converter name the subroutine of
 * `PP/src/pw2bgw.f90` in QE 7.5 (release tag `qe-7.5`) -- write_wfng,
 * write_rhog, write_vxc_g, write_vxc_r -- and the record, never a line
 * number: the line numbers first cited here were read off a 2026-07-27
 * `develop` checkout and moved with that branch. The same convention holds
 * in wavefunction.hpp, charge_density.hpp and exchange_correlation.hpp.
 *
 * Arrays are stored column-major (Fortran order): the FIRST index varies
 * fastest, so an N x M array a[i][j] is laid out a[1][1], a[2][1], ...,
 * a[N][1], a[1][2], ... -- this is why `avec`/`bvec` (first index the
 * Cartesian component, second the lattice vector) flatten "vector-major"
 * (all 3 components of vector 1, then all 3 of vector 2, ...) while `mtrx`
 * (first index the row, second the column, third the operation) flattens
 * "column-major within each 3x3 block" (all 3 rows of column 1, then all 3
 * rows of column 2, ...) -- both are the SAME general rule, just applied to
 * arrays whose first index means something different. (The rotation matrix
 * the converter writes is itself the transpose of Q3's rotation, so mtrx
 * comes out as Q3's rotation flattened row by row; see record 6.)
 * Mixed-type records (2, 3, 8) are built as manual byte buffers rather than
 * plain C structs: a struct interleaving int32 and
 * double fields gets compiler-inserted alignment padding before each double
 * (Q3's own QE-format wavefunction checkpoint writer,
 * io/wavefunction/kohn_sham/planewave/checkpoint/qe.hpp:83-88, declares
 * exactly such a struct -- see docs/io_reference.md §4.8 on why the
 * mean-field record writes every scalar component-by-component instead).
 *
 * Records 1-8 (the BerkeleyGW specification's first eight header records;
 * write_wfng for WFN, and write_rhog for RHO, which is byte-identical except
 * that records 2-3 omit the k-point fields). Shapes in brackets, first index
 * fastest:
 *   1. stitle[32] sdate[32] stime[32]                    -- char, 96 bytes
 *   2. ns ng ntran cell_symmetry nat ecutrho
 *      {nk nbands ngkmax ecutwfc}                        -- mixed, k-point
 *                                                            fields in braces
 *   3. FFTgrid[3] {kgrid[3] kshift[3]}                    -- mixed
 *   4. celvol alat avec[3,3] adot[3,3]                    -- float64
 *   5. recvol blat bvec[3,3] bdot[3,3]                    -- float64
 *   6. mtrx[3,3,ntran]                                    -- int32
 *   7. tnp[3,ntran]                                       -- float64
 *   8. (apos[3], Z) per atom, for each of the nat atoms   -- mixed, interleaved
 *
 * The Kleinman-Bylander projector file VKB, which the specification does not
 * cover, carries the WFN header with one change: record 2's nbands is replaced
 * by the projector dimensions, `... nk nsp nkb nhm ngkmax ecutwfc` -- the
 * record BerkeleyGW 4.0's ParaBands reads (`pp_read_header` in
 * `MeanField/ParaBands/pseudopot_vkb.f90`). See the projector-dimensions
 * constructor below.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "io/checkpoint/qe.hpp"
#include "io/logger.hpp"
#include "io/mean_field_export/berkeleygw/cell_symmetry.hpp"
#include "io/mean_field_record/init.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_1_structure/symmetry/operation.hpp"
#include "layer_1_structure/symmetry/symmetry.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class BerkeleyGWCommonHeader {
  public:
    /**
     * @param stitle                 32-byte title, e.g. "WFN-Complex" or
     *                               "RHO-Complex" (space-padded/truncated to
     *                               32 bytes; pw2bgw.x's write_wfng and
     *                               write_rhog always pad with spaces, never
     *                               a NUL terminator).
     * @param include_kpoint_fields  true for WFN (record 2 carries
     *                               nk/nbands/ngkmax/ecutwfc and record 3
     *                               carries kgrid/kshift), false for RHO
     *                               (the BerkeleyGW specification's
     *                               wavefunction-only fields; write_rhog
     *                               records 2-3).
     */
    BerkeleyGWCommonHeader(const MeanFieldRecord& record,
                           std::string stitle,
                           bool include_kpoint_fields)
        : record_(record)
        , stitle_(std::move(stitle))
        , include_kpoint_fields_(include_kpoint_fields) {}

    /**
     * The VKB header: the WFN header with record 2's nbands replaced by
     * `projector_dimensions` = {nsp, nkb, nhm} -- the number of species, of
     * projector channels over all atoms, and the most channels any one atom
     * has.
     */
    BerkeleyGWCommonHeader(const MeanFieldRecord& record,
                           std::string stitle,
                           const std::array<int32_t, 3>& projector_dimensions)
        : record_(record)
        , stitle_(std::move(stitle))
        , include_kpoint_fields_(true)
        , projector_dimensions_(projector_dimensions) {}

    /// See the writers' own deleted rvalue constructors: this holds a reference.
    BerkeleyGWCommonHeader(const MeanFieldRecord&&, std::string, bool) = delete;
    BerkeleyGWCommonHeader(const MeanFieldRecord&&,
                           std::string,
                           const std::array<int32_t, 3>&) = delete;

    /**
     * kgrid and kshift for an explicit k-point list (crystal coordinates) --
     * the numbers pw2bgw.x takes as its wfng_nk and wfng_dk inputs for the
     * same points.
     *
     * Only a grid the list pins down is returned:
     *   1. the list unfolded by the crystal's rotations and time reversal,
     *      when that star fills a grid exactly -- a symmetry-reduced wedge, or
     *      a full grid, of a grid the rotations preserve;
     *   2. otherwise the list as given, when it is a full grid -- every
     *      point once, all with the same weight -- e.g. the WFNq list, the
     *      full 4x4x2 grid displaced by q0 = (0.001, 0, 0), which gives 4 4 2
     *      with shift (0.004, 0, 0);
     *   3. otherwise 0 0 0 / 0 0 0, what the record carried before.
     * A reduced list also fits coarser grids than its own, sometimes exactly:
     * the two-point wedge of the shifted 2x2x2 FCC grid is a complete 2x1x1
     * grid. Its weights are orbit sizes, though, not equal, which is what
     * keeps it out of case 2. Case 3 is then such a wedge of a grid the
     * rotations do not preserve -- a shifted FCC grid -- which no rule
     * recovers from the points alone.
     *
     * Weights are compared exactly: a full list is written with one weight
     * repeated, and the same scaling of equal numbers stays equal.
     */
    [[nodiscard]] static std::pair<std::array<int32_t, 3>, std::array<double, 3>>
    kpoint_grid_from_list(const std::vector<Vector3D>& kpoint_coordinates,
                          const std::vector<Real>& kpoint_weights,
                          const std::vector<SymmetryOperation>& symmetry_operations) {
        std::vector<Vector3D> star;
        star.reserve(2 * kpoint_coordinates.size() * (symmetry_operations.size() + 1));
        for (const Vector3D& kpoint : kpoint_coordinates) {
            star.push_back(kpoint);
            star.push_back(-kpoint);
            for (const auto& operation : symmetry_operations) {
                Vector3D rotated{0.0, 0.0, 0.0};
                for (int row = 0; row < 3; row++)
                    for (int column = 0; column < 3; column++)
                        rotated[row] += operation.reciprocal_rotation[row][column] * kpoint[column];
                star.push_back(rotated);
                star.push_back(-rotated);
            }
        }
        const auto star_grid = complete_kpoint_grid(star);
        if (star_grid.first[0] > 0)
            return star_grid;

        const auto listed_grid = complete_kpoint_grid(kpoint_coordinates);
        const auto& [dimensions, shifts] = listed_grid;
        const bool equal_weights =
            std::all_of(kpoint_weights.begin(), kpoint_weights.end(), [&](Real weight) {
                return weight == kpoint_weights.front();
            });
        if (equal_weights &&
            static_cast<int64_t>(kpoint_coordinates.size()) ==
                static_cast<int64_t>(dimensions[0]) * dimensions[1] * dimensions[2])
            return listed_grid;
        return {{0, 0, 0}, {0.0, 0.0, 0.0}};
    }

    /**
     * The smallest grid holding every point of `kpoint_coordinates`, if the
     * points fill it exactly (every grid point present); 0 0 0 / 0 0 0
     * otherwise. Per axis, the smallest n at which every n * k sits the same
     * distance from an integer, that common offset being the shift (in units
     * of the grid step, wrapped to (-0.5, 0.5]); an axis no grid of up to
     * `maximum_grid_dimension` divisions fits -- a band-structure path, say --
     * leaves the whole result unset.
     */
    [[nodiscard]] static std::pair<std::array<int32_t, 3>, std::array<double, 3>>
    complete_kpoint_grid(const std::vector<Vector3D>& kpoint_coordinates) {
        constexpr int maximum_grid_dimension = 1000;
        constexpr Real tolerance = math::tolerances::kpoint_grid_offset_tolerance;
        auto distance_from_integer = [](Real value) { return value - std::round(value); };
        const std::pair<std::array<int32_t, 3>, std::array<double, 3>> unset = {{0, 0, 0},
                                                                                {0.0, 0.0, 0.0}};

        std::array<int32_t, 3> dimensions = {0, 0, 0};
        std::array<double, 3> shifts = {0.0, 0.0, 0.0};
        if (kpoint_coordinates.empty())
            return unset;
        for (int axis = 0; axis < 3; axis++) {
            for (int grid_dimension = 1; grid_dimension <= maximum_grid_dimension;
                 grid_dimension++) {
                const Real offset =
                    distance_from_integer(grid_dimension * kpoint_coordinates.front()[axis]);
                const bool on_grid = std::all_of(
                    kpoint_coordinates.begin(), kpoint_coordinates.end(), [&](const Vector3D& k) {
                        return std::abs(distance_from_integer(grid_dimension * k[axis] - offset)) <=
                               tolerance;
                    });
                if (on_grid) {
                    dimensions[axis] = grid_dimension;
                    shifts[axis] = offset <= -0.5 + tolerance ? offset + 1.0 : offset;
                    break;
                }
            }
            if (dimensions[axis] == 0)
                return unset;
        }

        std::set<std::array<int64_t, 3>> occupied_points;
        for (const Vector3D& kpoint : kpoint_coordinates) {
            std::array<int64_t, 3> index{};
            for (int axis = 0; axis < 3; axis++) {
                const int64_t on_grid =
                    std::llround(dimensions[axis] * kpoint[axis] - shifts[axis]);
                index[axis] = ((on_grid % dimensions[axis]) + dimensions[axis]) % dimensions[axis];
            }
            occupied_points.insert(index);
        }
        if (static_cast<int64_t>(occupied_points.size()) !=
            static_cast<int64_t>(dimensions[0]) * dimensions[1] * dimensions[2])
            return unset;
        return {dimensions, shifts};
    }

    /// Write records 1 through 8 to `output_file`.
    void write(std::ofstream& output_file) const {
        const std::vector<SymmetryOperation>& symmetry_operations = record_.symmetry_operations;

        write_title_record(output_file);
        write_dimension_record(output_file, static_cast<int>(symmetry_operations.size()));
        write_grid_record(output_file);
        write_direct_lattice_record(output_file);
        write_reciprocal_lattice_record(output_file);
        write_symmetry_rotation_record(output_file, symmetry_operations);
        write_symmetry_translation_record(output_file, symmetry_operations);
        write_atom_record(output_file);
    }

  private:
    const MeanFieldRecord& record_;
    std::string stitle_;
    bool include_kpoint_fields_;
    /// Set for VKB only: nsp, nkb, nhm, written in place of nbands.
    std::optional<std::array<int32_t, 3>> projector_dimensions_;
    QEBinaryRecordFormat record_format_;

    /**
     * cell_symmetry: 0 for the cubic family, 1 for the hexagonal one. From the
     * Bravais index when the input named a lattice, which follows QE's ibrav
     * (see Crystal::set_from_lattice_type()), and from the lattice vectors
     * for a free lattice; see berkeleygw/cell_symmetry.hpp.
     */
    [[nodiscard]] int cell_symmetry() const {
        return berkeleygw_cell_symmetry(record_.bravais_lattice_index, record_.lattice_vectors);
    }

    /**
     * Metric tensor `dot[i][j] = vectors[i] . vectors[j] * scale^2` -- used
     * for both `adot` (direct lattice, a_i . a_j scaled by alat^2) and `bdot`
     * (reciprocal lattice, b_i . b_j scaled by (2*pi/alat)^2), the real- and
     * reciprocal-space metric tensors of the BerkeleyGW specification's
     * records 4 and 5 (a symmetric Gram matrix, so row-major vs.
     * column-major flattening are byte-identical either way).
     */
    [[nodiscard]] static Matrix3x3 metric_tensor(const Matrix3x3& vectors, Real scale) {
        Matrix3x3 result;
        Real scale_squared = scale * scale;
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                result[row][column] = vectors[row].dot(vectors[column]) * scale_squared;
            }
        }
        return result;
    }

    /**
     * Pad/truncate `text` to exactly `length` bytes with trailing spaces (Fortran CHARACTER
     * convention).
     */
    static void write_padded_string(char* destination, const std::string& text, size_t length) {
        size_t copy_length = std::min(text.size(), length);
        std::memcpy(destination, text.data(), copy_length);
        std::memset(destination + copy_length, ' ', length - copy_length);
    }

    /**
     * Record 1: stitle(32) sdate(32) stime(32) -- 96 bytes of padded chars.
     * sdate/stime mirror the "DD-Mon-YYYY"/"HH:MM:SS" formats pw2bgw.x writes;
     * BerkeleyGW does not parse these fields back, so exact QE locale
     * behavior is not required, only well-formed fixed-width padding.
     */
    void write_title_record(std::ofstream& output_file) const {
        std::vector<char> buffer(96, ' ');
        write_padded_string(buffer.data(), stitle_, 32);

        std::time_t now = std::time(nullptr);
        std::tm local_time{};
        localtime_r(&now, &local_time);
        char date_text[16];
        std::strftime(date_text, sizeof(date_text), "%d-%b-%Y", &local_time);
        char time_text[16];
        std::strftime(time_text, sizeof(time_text), "%H:%M:%S", &local_time);

        write_padded_string(buffer.data() + 32, date_text, 32);
        write_padded_string(buffer.data() + 64, time_text, 32);

        record_format_.write_record(
            output_file, buffer.data(), static_cast<int32_t>(buffer.size()));
    }

    /**
     * Record 2 (the BerkeleyGW specification's record 2; write_wfng for WFN,
     * write_rhog for RHO):
     *   WFN: ns ng ntran cell_symmetry nat ecutrho nk nbands ngkmax ecutwfc
     *   RHO: ns ng ntran cell_symmetry nat ecutrho
     * ecutrho/ecutwfc are already in Rydberg in Q3 (PlanewaveBasisConfig
     * docstring), the unit the BerkeleyGW specification gives for both cutoffs
     * -- no conversion.
     */
    void write_dimension_record(std::ofstream& output_file,
                                int number_of_symmetry_operations) const {
        int32_t number_of_spin_components = record_.number_of_spin_components;
        int32_t number_of_g_vectors = record_.number_of_g_vectors();
        int32_t number_of_symmetry_operations_i32 = number_of_symmetry_operations;
        int32_t cell_symmetry_value = cell_symmetry();
        int32_t number_of_atoms = record_.number_of_atoms();
        double charge_density_cutoff_ry = record_.charge_density_energy_cutoff;

        size_t fixed_size = 4 * sizeof(int32_t) + sizeof(int32_t) + sizeof(double);
        size_t kpoint_fields_size =
            include_kpoint_fields_ ? (3 * sizeof(int32_t) + sizeof(double)) : 0;
        if (projector_dimensions_) {
            kpoint_fields_size += 2 * sizeof(int32_t); // nsp nkb nhm in place of nbands
        }
        std::vector<char> buffer(fixed_size + kpoint_fields_size);

        size_t offset = 0;
        auto append = [&](const void* value, size_t size) {
            std::memcpy(buffer.data() + offset, value, size);
            offset += size;
        };
        append(&number_of_spin_components, sizeof(int32_t));
        append(&number_of_g_vectors, sizeof(int32_t));
        append(&number_of_symmetry_operations_i32, sizeof(int32_t));
        append(&cell_symmetry_value, sizeof(int32_t));
        append(&number_of_atoms, sizeof(int32_t));
        append(&charge_density_cutoff_ry, sizeof(double));

        if (include_kpoint_fields_) {
            int32_t number_of_kpoints = record_.number_of_kpoints();
            int32_t number_of_bands = record_.number_of_bands;
            const auto& plane_waves_per_kpoint = record_.plane_waves_per_kpoint;
            int32_t maximum_plane_waves = plane_waves_per_kpoint.empty()
                                              ? 0
                                              : *std::max_element(plane_waves_per_kpoint.begin(),
                                                                  plane_waves_per_kpoint.end());
            double wavefunction_cutoff_ry = record_.wavefunction_energy_cutoff;

            append(&number_of_kpoints, sizeof(int32_t));
            if (projector_dimensions_) {
                append(projector_dimensions_->data(), 3 * sizeof(int32_t));
            } else {
                append(&number_of_bands, sizeof(int32_t));
            }
            append(&maximum_plane_waves, sizeof(int32_t));
            append(&wavefunction_cutoff_ry, sizeof(double));
        }

        record_format_.write_record(
            output_file, buffer.data(), static_cast<int32_t>(buffer.size()));
    }

    /**
     * Record 3 (the BerkeleyGW specification's record 3; write_wfng for WFN,
     * write_rhog for RHO):
     *   WFN: FFTgrid[3] kgrid[3] kshift[3]
     *   RHO: FFTgrid[3]
     * FFTgrid is the DENSE (charge-density) FFT grid in pw2bgw.x's output, not
     * the smooth wavefunction grid -- Q3's analog is fft_rho(), not
     * fft_wave(). For a Monkhorst-Pack run kshift = 0.5 * (grid shift flag)
     * on each axis, matching what write_wfng writes from QE's 0/1 K_POINTS
     * offsets (Q3's KPointsGrid::grid_shifts() stores the same 0/1 shift-flag
     * convention). An explicit
     * k-point list -- how BerkeleyGW's WFN and WFNq are normally produced --
     * has no grid in the record, so kgrid/kshift are recovered from the
     * points themselves (kpoint_grid_from_list()); BerkeleyGW's epsilon
     * refuses a WFN whose kgrid is 0 0 0 ("kgrid out of bounds in wfn").
     */
    void write_grid_record(std::ofstream& output_file) const {
        int32_t fft_dimensions[3] = {record_.fft_grid_dimensions[0],
                                     record_.fft_grid_dimensions[1],
                                     record_.fft_grid_dimensions[2]};

        if (!include_kpoint_fields_) {
            record_format_.write_record(output_file, fft_dimensions, sizeof(fft_dimensions));
            return;
        }

        const auto& grid_dimensions = record_.kpoint_grid_dimensions;
        const auto& grid_shifts = record_.kpoint_grid_shifts;
        int32_t kpoint_grid_dimensions[3] = {
            grid_dimensions[0], grid_dimensions[1], grid_dimensions[2]};
        double kpoint_shifts[3] = {
            0.5 * grid_shifts[0], 0.5 * grid_shifts[1], 0.5 * grid_shifts[2]};
        if (grid_dimensions[0] == 0 && grid_dimensions[1] == 0 && grid_dimensions[2] == 0) {
            const auto [inferred_dimensions, inferred_shifts] = kpoint_grid_from_list(
                record_.kpoint_coordinates, record_.kpoint_weights, record_.symmetry_operations);
            for (int axis = 0; axis < 3; axis++) {
                kpoint_grid_dimensions[axis] = inferred_dimensions[axis];
                kpoint_shifts[axis] = inferred_shifts[axis];
            }
            if (inferred_dimensions[0] == 0)
                log::warn("     BerkeleyGW header: kgrid left at 0 0 0 -- the k-point list is "
                          "neither a complete grid nor the symmetry-reduced wedge of one, so "
                          "BerkeleyGW's epsilon will not accept this WFN. List every point of "
                          "the grid instead.");
        }

        std::vector<char> buffer(sizeof(fft_dimensions) + sizeof(kpoint_grid_dimensions) +
                                 sizeof(kpoint_shifts));
        size_t offset = 0;
        std::memcpy(buffer.data() + offset, fft_dimensions, sizeof(fft_dimensions));
        offset += sizeof(fft_dimensions);
        std::memcpy(buffer.data() + offset, kpoint_grid_dimensions, sizeof(kpoint_grid_dimensions));
        offset += sizeof(kpoint_grid_dimensions);
        std::memcpy(buffer.data() + offset, kpoint_shifts, sizeof(kpoint_shifts));

        record_format_.write_record(
            output_file, buffer.data(), static_cast<int32_t>(buffer.size()));
    }

    /**
     * Record 4: celvol alat avec[3,3] adot[3,3] -- all float64, so a flat
     * vector is safe (no interleaved int32 to trigger alignment padding).
     * avec flattens vector-major (the Cartesian component is the fast index,
     * i.e. all 3 components of lattice vector i before moving to vector
     * i+1) -- Q3's lattice_vectors() is already stored row-per-vector, so
     * writing rows in order is exactly this flattening. avec is in units of
     * alat (Crystal::lattice_vectors() docstring); adot is the metric tensor,
     * a_i . a_j scaled by alat^2 (metric_tensor()).
     */
    void write_direct_lattice_record(std::ofstream& output_file) const {
        const Matrix3x3& lattice_vectors = record_.lattice_vectors;
        Real lattice_parameter = record_.lattice_parameter;
        Matrix3x3 direct_metric = metric_tensor(lattice_vectors, lattice_parameter);

        std::vector<double> buffer;
        buffer.reserve(20);
        buffer.push_back(record_.cell_volume);
        buffer.push_back(lattice_parameter);
        for (int vector_index = 0; vector_index < 3; vector_index++) {
            buffer.push_back(lattice_vectors[vector_index].x);
            buffer.push_back(lattice_vectors[vector_index].y);
            buffer.push_back(lattice_vectors[vector_index].z);
        }
        for (int column = 0; column < 3; column++) {
            for (int row = 0; row < 3; row++) {
                buffer.push_back(direct_metric[row][column]);
            }
        }

        record_format_.write_record(
            output_file, buffer.data(), static_cast<int32_t>(buffer.size() * sizeof(double)));
    }

    /**
     * Record 5: recvol blat bvec(3,3) bdot(3,3) -- same shape as record 4,
     * for the reciprocal lattice. recvol = (2*pi)^3 / celvol, the Brillouin-zone
     * volume; blat = 2*pi/alat. bvec is in units of 2*pi/alat (Crystal's
     * reciprocal_lattice_vectors() docstring).
     */
    void write_reciprocal_lattice_record(std::ofstream& output_file) const {
        const Matrix3x3& reciprocal_lattice_vectors = record_.reciprocal_lattice_vectors;
        Real two_pi_over_lattice_parameter = record_.two_pi_over_lattice_parameter;
        Matrix3x3 reciprocal_metric =
            metric_tensor(reciprocal_lattice_vectors, two_pi_over_lattice_parameter);

        std::vector<double> buffer;
        buffer.reserve(20);
        buffer.push_back(8.0 * constants::pi * constants::pi * constants::pi / record_.cell_volume);
        buffer.push_back(two_pi_over_lattice_parameter);
        for (int vector_index = 0; vector_index < 3; vector_index++) {
            buffer.push_back(reciprocal_lattice_vectors[vector_index].x);
            buffer.push_back(reciprocal_lattice_vectors[vector_index].y);
            buffer.push_back(reciprocal_lattice_vectors[vector_index].z);
        }
        for (int column = 0; column < 3; column++) {
            for (int row = 0; row < 3; row++) {
                buffer.push_back(reciprocal_metric[row][column]);
            }
        }

        record_format_.write_record(
            output_file, buffer.data(), static_cast<int32_t>(buffer.size() * sizeof(double)));
    }

    /**
     * Record 6: mtrx(3,3,ntran) -- int32 rotation matrices in crystal
     * coordinates, carrying QE's crystal-coordinate rotation matrix S exactly
     * as write_wfng writes it: first index fastest, so within each 3x3 block
     * mtrx[1][1], mtrx[2][1], mtrx[3][1], mtrx[1][2], ... S is the TRANSPOSE
     * of Q3's real-space `rotation` (QE rotates crystal coordinates as
     * x'_j = sum_i S_ij x_i), so mtrx[k][j] = rotation[j][k] and the bytes are
     * `rotation` flattened row by row. The BerkeleyGW specification describes
     * mtrx only as rotation matrices in the reciprocal-lattice basis; the
     * exact convention -- transposition, and pairing with record 7 -- is
     * pinned by the comparisons that follow, not by the specification.
     * Checked operation by operation against QE 7.5's data-file-schema.xml and pw2bgw output for
     * wurtzite ZnO, diamond Si and trigonal Te. BerkeleyGW reads mtrx as the
     * same matrix QE uses, so writing `rotation` itself hands it R^T: the
     * group is unchanged, but each matrix is paired with the wrong record-7
     * translation, and epsilon stops with "G-vectors are falling outside of
     * the sphere in gmap" on any lattice where R^T is not R^-1.
     */
    void write_symmetry_rotation_record(
        std::ofstream& output_file,
        const std::vector<SymmetryOperation>& symmetry_operations) const {
        std::vector<int32_t> buffer;
        buffer.reserve(9 * symmetry_operations.size());
        for (const auto& operation : symmetry_operations) {
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    buffer.push_back(operation.rotation[row][column]);
                }
            }
        }
        record_format_.write_record(
            output_file, buffer.data(), static_cast<int32_t>(buffer.size() * sizeof(int32_t)));
    }

    /**
     * Record 7: tnp(3,ntran) -- float64 fractional translations in crystal
     * coordinates times 2*pi, NOT SymmetryOperation::fractional_translation
     * copied verbatim: the BerkeleyGW specification gives tnp in units of the
     * lattice vectors times 2*pi, and write_wfng derives it from QE's rotation
     * S and fractional translation f as (S^-1)^T f, reduced modulo 1, then
     * scaled by 2*pi. In Q3's terms S = rotation^T (record 6) and
     * f = -fractional_translation (QE
     * stores its translations with the opposite sign; the same check against
     * QE 7.5's XML confirms it for every operation), so
     *
     *   tnp / 2*pi = -rotation^-1 . fractional_translation
     *
     * -- the translation of the inverse operation {R|t}^-1 = {R^-1|-R^-1 t}.
     * Q3 already stores rotation^-1 transposed, as reciprocal_rotation =
     * (rotation^-1)^T (operation.hpp), so rotation^-1 . t contracts t with
     * reciprocal_rotation's FIRST index. The trigonal-Te test pins the sign
     * against pw2bgw.x's own output: there the 3_1 screw's c/3 and -c/3 differ.
     *
     * The result is then reduced to a canonical representative modulo 1. The
     * 1e-6 is added to BOTH comparison thresholds rather than forming a
     * symmetric dead-band, so the interval is (-0.5, 0.5] widened at the top
     * by 1e-6: an input already in [0.5, 0.5 + 1e-6) is left alone, and an
     * input of exactly -0.5 is mapped to +0.5. The effect is that a
     * translation sitting exactly on the boundary -- 0.5 is common, e.g.
     * glide planes and the diamond structure -- stays put instead of flipping
     * sign under rounding noise. Truncation toward zero rather than
     * round-to-nearest is what places the closed end at +0.5; pw2bgw.x makes
     * the same choice.
     */
    void write_symmetry_translation_record(
        std::ofstream& output_file,
        const std::vector<SymmetryOperation>& symmetry_operations) const {
        constexpr Real boundary_tolerance = 1.0e-6; // the same 1e-6 pw2bgw.x widens by
        std::vector<double> buffer;
        buffer.reserve(3 * symmetry_operations.size());
        for (const auto& operation : symmetry_operations) {
            Real rotated_translation[3];
            for (int component = 0; component < 3; component++) {
                Real sum = 0.0;
                for (int contracted = 0; contracted < 3; contracted++) {
                    sum -= static_cast<Real>(operation.reciprocal_rotation[contracted][component]) *
                           operation.fractional_translation[contracted];
                }
                rotated_translation[component] = sum;
            }
            for (int component = 0; component < 3; component++) {
                Real& value = rotated_translation[component];
                if (value >= boundary_tolerance + 0.5)
                    value -= std::trunc(value + 0.5);
                if (value < boundary_tolerance - 0.5)
                    value -= std::trunc(value - 0.5);
            }
            for (int component = 0; component < 3; component++) {
                buffer.push_back(rotated_translation[component] * constants::two_pi);
            }
        }
        record_format_.write_record(
            output_file, buffer.data(), static_cast<int32_t>(buffer.size() * sizeof(double)));
    }

    /**
     * Record 8: (apos(3,i), Z_i) for i=1..nat -- interleaved per atom (3
     * float64 then 1 int32, repeated), NOT a uniform array of one type, so a
     * manual byte buffer is required (a `struct{double[3];int32_t;}` would
     * get 4 bytes of trailing alignment padding per atom to keep the array
     * element size a multiple of 8, corrupting the interleaving for atom 2
     * onward). Positions are Cartesian in units of alat
     * (Crystal::convert_from_crystal_to_cartesian()); Z is the atomic number
     * (the BerkeleyGW specification's atomic numbers, as Crystal::atomic_number()
     * gives them, and as write_wfng writes them).
     */
    void write_atom_record(std::ofstream& output_file) const {
        int number_of_atoms = record_.number_of_atoms();
        std::vector<char> buffer(number_of_atoms * (3 * sizeof(double) + sizeof(int32_t)));

        size_t offset = 0;
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            const Vector3D& position = record_.atom_positions_cartesian[atom_index];
            double position_components[3] = {position.x, position.y, position.z};
            std::memcpy(buffer.data() + offset, position_components, sizeof(position_components));
            offset += sizeof(position_components);

            int32_t atomic_number = record_.atomic_numbers[atom_index];
            std::memcpy(buffer.data() + offset, &atomic_number, sizeof(atomic_number));
            offset += sizeof(atomic_number);
        }

        record_format_.write_record(
            output_file, buffer.data(), static_cast<int32_t>(buffer.size()));
    }
};

} // namespace q3
