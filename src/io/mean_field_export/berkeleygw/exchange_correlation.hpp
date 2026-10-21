#pragma once

/**
 * BerkeleyGWExchangeCorrelationFile — writes BerkeleyGW's `vxc.dat` mean-field file.
 *
 * Layout (the BerkeleyGW specification's vxc.dat section, with the column
 * widths of pw2bgw.x's `write_vxc_g`/`write_vxc_r`, which write byte-identical
 * text -- see header.hpp for both citations): per k-point, one ASCII header
 * line `kx ky kz ndiag noffdiag` -- three 13-wide fixed-point fields with 9
 * decimals, then two 8-wide integers (k in CRYSTAL coordinates; the two
 * integers are already multiplied by nspin -- nspin*ndiag, and nspin times
 * the number of off-diagonal band pairs) -- then `ndiag` diagonal lines
 * `spin band Re(Vxc) Im(Vxc)` -- two 8-wide integers, then two 15-wide
 * fixed-point fields with 9 decimals. Diagonal-only is in scope here, so
 * `noffdiag = 0` always and no off-diagonal lines (three 8-wide integers,
 * then the two values) are ever written. `ndiag` = the full band count: Q3
 * has no analog of pw2bgw.x's `diag_nmin`/`diag_nmax` partial-range input,
 * so every band is written, matching WFN's own energies/occupations records
 * (which are likewise unconditionally all-band). Matrix elements are in eV,
 * as the specification requires (pw2bgw.x converts each element from
 * Rydberg where it computes it), i.e. Q3's own `constants::RY_TO_EV`,
 * applied once at the very end.
 *
 * ── The physics: sidestepping the dense-grid transform ──────────────────
 *
 * pw2bgw.x computes `<psi_nk|Vxc|psi_nk>` by transforming psi_nk to real
 * space on the DENSE (charge-density) FFT grid, multiplying pointwise by
 * Vxc(r), and transforming back (write_vxc_g). Q3's only wavefunction
 * G<->R transform (`WavefunctionTransformer`, reached via
 * `PlanewaveBasis::g_to_r`/`r_to_g`) targets the coarser SMOOTH grid
 * (`fft_wave()`), not the dense one (`fft_rho()`) -- reproducing pw2bgw.x's
 * route literally would need a new G->R transform Q3 does not expose.
 *
 * That transform is avoidable entirely: pointwise multiplication in real
 * space is convolution in G-space, so the SAME matrix element is exactly a
 * G-space double sum over this band's own plane-wave coefficients c_nk and
 * the already-available G-space XC potential:
 *
 *     <psi_nk|Vxc|psi_nk> = sum_{G,G'} conj(c_nk(G)) * Vxc(G - G') * c_nk(G')
 *
 * where G, G' range over psi_nk's plane-wave G-set at this k-point and
 * Vxc(G) is `NormConservingPlanewaveKohnShamSCFState::exchange_correlation_potential_g`
 * (already on the charge-density G set -- see its assembly in
 * effective_potential/planewave/init.hpp's `assemble_effective_potential()`:
 * `exchange_correlation_potential_average = spin_averaged_potential -
 * hartree - local_ionic`, then `planewave_basis_.rho_r_to_g(...)`). Two
 * independent derivations confirm this needs no extra normalization
 * (no 1/N, no cell volume) beyond the final Ry->eV conversion:
 *
 *   1. Standard QE Bloch-function algebra: psi_nk(r) = sum_G c(G)
 *      e^{i(k+G).r} with the usual plane-wave normalization sum_G|c(G)|^2 =
 *      1 (so integral|psi|^2 dr = 1 over the cell), and Vxc(r) = sum_G
 *      Vxc(G) e^{iG.r}. Expanding <psi|Vxc|psi> = integral psi*(r) Vxc(r)
 *      psi(r) dr and using integral e^{iq.r} dr = Omega * delta_{q,0}
 *      collapses the two cell-volume factors (one from the integral, one
 *      from 1/Omega implicit in orthonormal c(G)) exactly, leaving the sum
 *      above with no leftover factor.
 *
 *   2. Q3's OWN Hamiltonian application (SerialPlanewaveHamiltonianApplicator,
 *      `apply_hamiltonian_batch()`'s "2. Local potential" step) computes
 *      H|psi> for the real local potential via UNNORMALIZED raw FFTs
 *      (`execute_inverse_raw_on_buffers()`/`execute_forward_on_buffers()`,
 *      see FFTGrid's `inverse_raw`/`forward` docs) with the potential
 *      PRE-SCALED by `1/total_grid_points` before the pointwise multiply
 *      (`update_potential()`'s `cached_scaled_potential_`). Working through
 *      that pipeline algebraically (raw-FFT round trip contributes exactly
 *      one factor of `total_grid_points`, canceling the pre-scaling)
 *      reproduces `H|psi>(G) = sum_G' V(G-G') psi(G')` with V(G) taken
 *      AS-IS from the stored G-space potential -- the identical formula and
 *      normalization used here, this time derived from what Q3 itself
 *      already does to apply a real potential to a wavefunction, not from a
 *      textbook.
 *
 * `MillerDifferenceIndex` below turns the per-(G,G') lookup "is Miller
 * difference (G-G') a stored charge-density G-vector, and if so at what
 * index" into an O(1) dense-array lookup, built ONCE per write_to_file()
 * call rather than once per band. A difference that lands outside the
 * stored charge-density G-set contributes zero (skipped, not
 * approximated) -- for the standard norm-conserving dual=4 cutoff ratio
 * (`PlanewaveBasis::generate()`'s own `4.0 *
 * energy_cutoff_wavefunction_` default) this never actually happens: by
 * the triangle inequality |G-G'| <= |G| + |G'| <= 2*sqrt(ecutwfc) =
 * sqrt(4*ecutwfc) = sqrt(ecutrho) for any two wavefunction G-vectors, so
 * every difference falls within the charge-density cutoff sphere. A
 * smaller dual ratio (user-configured `density` cutoff below 4x
 * `wavefunction`) can genuinely truncate some differences; write_to_file()
 * counts and reports the fraction (via `log::info`) precisely so that case
 * is visible rather than silently absorbed.
 *
 * Since Vxc(r) is real, Vxc(-G) = conj(Vxc(G)), which forces the DIAGONAL
 * matrix element above to be real in exact arithmetic; write_to_file()
 * computes it as complex, checks the imaginary part stays within
 * floating-point rounding of zero (see `imaginary_part_tolerance_ry`), and
 * throws rather than writing a suspect file if it does not -- a
 * non-negligible imaginary part means an index or conjugation bug, not
 * physics, and should never be silently written.
 *
 * ── Scope restrictions (both throw a clear, specific error) ─────────────
 *
 * - Spin-polarized (nspin>1): the only Vxc Q3 exposes on the SCF state,
 *   `exchange_correlation_potential_g`, is the SPIN-AVERAGED potential (see
 *   its assembly above) -- using it for both spin channels would silently
 *   give a wrong answer for any magnetized system. Mirrors
 *   `BerkeleyGWWavefunctionFile`'s own nspin>1 rejection exactly. (For an
 *   unpolarized run the average of one channel is itself, so this does not
 *   apply.)
 *
 * - Gamma-only (`PlanewaveBasis::is_gamma_only()`): a gamma-only run stores
 *   only HALF the plane-wave G-sphere per k-point and reconstructs the
 *   other half via c(-G) = conj(c(G)) wherever it actually needs psi(r) --
 *   see `SerialPlanewaveHamiltonianApplicator::apply_hamiltonian_batch()`'s
 *   explicit `+G`/`-G` packing before its own real-space multiply. The
 *   direct G-space double sum above, applied to only the STORED half,
 *   would silently omit exactly those reconstructed contributions. Adding
 *   the reconstruction correctly is possible in principle but out of scope
 *   here; throwing is the same honest choice already made for nspin>1
 *   rather than guessing. This is a NEW restriction relative to WFN/RHO
 *   (which pass gamma-only data through as-is, without needing this
 *   reconstruction) discovered while implementing this file, not part of
 *   the original two blockers.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "io/mean_field_record/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"

namespace q3 {

class BerkeleyGWExchangeCorrelationFile {
  public:
    explicit BerkeleyGWExchangeCorrelationFile(const MeanFieldRecord& record) : record_(record) {}
    /**
     * These writers keep a reference to the record, so binding one to a
     * temporary would dangle the moment the full expression ends. Deleting the
     * rvalue overload turns that into a compile error instead of a silent
     * read of freed memory -- callers must keep the record alive themselves,
     * which every real call site already does.
     */
    explicit BerkeleyGWExchangeCorrelationFile(const MeanFieldRecord&&) = delete;

    void write_to_file(const std::string& file_path) const {
        if (record_.number_of_spin_components > 1) {
            throw std::runtime_error(
                "BerkeleyGWExchangeCorrelationFile: spin-polarized (nspin>1) vxc.dat export is "
                "not supported -- Q3's exchange_correlation_potential_g is the SPIN-AVERAGED "
                "potential (see assemble_effective_potential() in "
                "effective_potential/planewave/init.hpp), so using it for both spin channels "
                "would silently give a wrong <psi|Vxc|psi> for any magnetized system. Use "
                "number_of_spin_components: 1, or omit mean_field_export, for a spin-polarized "
                "run. Cannot write '" +
                file_path + "'.");
        }
        if (record_.is_gamma_only) {
            throw std::runtime_error(
                "BerkeleyGWExchangeCorrelationFile: gamma-only vxc.dat export is not supported "
                "-- a gamma-only run stores only half the plane-wave G-sphere per k-point and "
                "reconstructs the other half via c(-G) = conj(c(G)) wherever real-space values "
                "are actually needed (see SerialPlanewaveHamiltonianApplicator's own +G/-G "
                "packing); the direct G-space convolution used here, applied to only the stored "
                "half, would silently omit those reconstructed contributions. Cannot write '" +
                file_path + "'.");
        }

        const auto& g_vectors = record_.charge_density_miller_indices;
        const auto& vxc_g = record_.exchange_correlation_potential_g;
        if (vxc_g.size() != g_vectors.size()) {
            throw std::runtime_error(
                "BerkeleyGWExchangeCorrelationFile: exchange_correlation_potential_g has " +
                std::to_string(vxc_g.size()) + " entries, expected " +
                std::to_string(g_vectors.size()) +
                " (one per charge-density G-vector) -- the SCF state does not carry a usable "
                "G-space Vxc (e.g. an incomplete or synthetic state). Cannot write '" +
                file_path + "'.");
        }

        MillerDifferenceIndex miller_lookup(g_vectors);
        Statistics statistics;
        std::vector<KPointBlock> blocks =
            compute_all_kpoint_blocks(g_vectors, vxc_g, miller_lookup, statistics);

        if (statistics.maximum_absolute_imaginary_part_ry > imaginary_part_tolerance_ry) {
            throw std::runtime_error(
                "BerkeleyGWExchangeCorrelationFile: computed <psi_nk|Vxc|psi_nk> has a "
                "non-negligible imaginary part (max |Im| = " +
                std::to_string(statistics.maximum_absolute_imaginary_part_ry) + " Ry, tolerance " +
                std::to_string(imaginary_part_tolerance_ry) +
                " Ry) -- Vxc(r) is real, so this diagonal matrix element must come out real to "
                "within floating-point rounding. This size of imaginary residual means an index "
                "or conjugation bug, not noise. Refusing to write a suspect '" +
                file_path + "'.");
        }

        std::ofstream output_file(file_path);
        if (!output_file.is_open()) {
            log::warn("BerkeleyGWExchangeCorrelationFile: cannot open {} for writing", file_path);
            return;
        }
        for (const auto& block : blocks) {
            write_kpoint_block(output_file, block);
        }

        double out_of_set_fraction =
            statistics.total_g_difference_pairs > 0
                ? static_cast<double>(statistics.out_of_set_g_difference_pairs) /
                      static_cast<double>(statistics.total_g_difference_pairs)
                : 0.0;
        log::info("     Wrote BerkeleyGW exchange-correlation file: {} ({} k-point(s), {} "
                  "band(s); out-of-set G-difference fraction {:.3e}, max |Im(Vxc)| {:.3e} eV)",
                  file_path,
                  blocks.size(),
                  record_.number_of_bands,
                  out_of_set_fraction,
                  statistics.maximum_absolute_imaginary_part_ry * constants::RY_TO_EV);
    }

  private:
    const MeanFieldRecord& record_;

    /**
     * Above this, a computed <psi_nk|Vxc|psi_nk>'s imaginary part is treated
     * as a bug rather than floating-point noise. A direct double sum of up
     * to ~1500^2 ~ 2.3e6 complex terms accumulates rounding error of order
     * sqrt(N)*epsilon*|typical term| -- roughly 1.5e3 * 1e-16 * O(1) ~
     * 1.5e-13 Ry in practice -- many orders of magnitude below this
     * threshold, so anything that trips it is a real defect.
     */
    static constexpr Real imaginary_part_tolerance_ry = 1.0e-6;

    struct Statistics {
        long total_g_difference_pairs = 0;
        long out_of_set_g_difference_pairs = 0;
        Real maximum_absolute_imaginary_part_ry = 0.0;
    };

    struct KPointBlock {
        Vector3D kpoint_crystal_coordinates;
        std::vector<Complex> diagonal_elements_ev; // band-major, band index 0..nbands-1
    };

    /**
     * Dense lookup from a Miller-index DIFFERENCE (G - G', for G, G' both
     * charge-density G-vectors) to that difference's position in the
     * charge-density G-vector list `PlanewaveBasis::g_vectors()`, or -1 if
     * the difference is not itself a stored charge-density G-vector. Built
     * ONCE per write_to_file() call (O(number_of_g_vectors)), then queried
     * O(number_of_plane_waves^2) times per band.
     *
     * Sized from the ACTUAL observed Miller range of g_vectors() rather
     * than, say, PlanewaveBasis::fft_rho()'s grid dimensions, so it needs
     * no assumption about how the charge-density G-sphere relates to its
     * enclosing FFT box. A difference of two Miller indices, each within
     * [min, max], ranges within [min-max, max-min] -- twice the extent of
     * the original range -- so the lookup box is sized to that wider
     * range. The ORIGINAL range is always a subset of it (needed so
     * populating the array from the original G-vectors never goes out of
     * bounds): G=(0,0,0) is always a stored charge-density G-vector (any
     * positive cutoff includes it, both in the general and gamma-only
     * cases), so 0 is always within [min, max] for every component, which
     * forces min <= 0 <= max and hence [min, max] subseteq [min-max,
     * max-min].
     */
    class MillerDifferenceIndex {
      public:
        explicit MillerDifferenceIndex(const std::vector<IntegerVector3D>& g_vectors) {
            if (g_vectors.empty()) {
                lookup_.assign(1, -1);
                return;
            }

            IntegerVector3D minimum_miller = g_vectors[0];
            IntegerVector3D maximum_miller = g_vectors[0];
            for (const auto& g_vector : g_vectors) {
                minimum_miller.x = std::min(minimum_miller.x, g_vector.x);
                minimum_miller.y = std::min(minimum_miller.y, g_vector.y);
                minimum_miller.z = std::min(minimum_miller.z, g_vector.z);
                maximum_miller.x = std::max(maximum_miller.x, g_vector.x);
                maximum_miller.y = std::max(maximum_miller.y, g_vector.y);
                maximum_miller.z = std::max(maximum_miller.z, g_vector.z);
            }

            minimum_difference_ = IntegerVector3D(minimum_miller.x - maximum_miller.x,
                                                  minimum_miller.y - maximum_miller.y,
                                                  minimum_miller.z - maximum_miller.z);
            maximum_difference_ = IntegerVector3D(maximum_miller.x - minimum_miller.x,
                                                  maximum_miller.y - minimum_miller.y,
                                                  maximum_miller.z - minimum_miller.z);

            range_y_ = maximum_difference_.y - minimum_difference_.y + 1;
            range_z_ = maximum_difference_.z - minimum_difference_.z + 1;
            int range_x = maximum_difference_.x - minimum_difference_.x + 1;
            lookup_.assign(static_cast<size_t>(range_x) * range_y_ * range_z_, -1);

            for (int g_index = 0; g_index < static_cast<int>(g_vectors.size()); g_index++) {
                const auto& miller = g_vectors[g_index];
                lookup_[flatten(miller.x, miller.y, miller.z)] = g_index;
            }
        }

        /// Position of Miller difference (dx,dy,dz) in the charge-density G-vector list, or -1.
        [[nodiscard]] int find(int dx, int dy, int dz) const {
            if (dx < minimum_difference_.x || dx > maximum_difference_.x ||
                dy < minimum_difference_.y || dy > maximum_difference_.y ||
                dz < minimum_difference_.z || dz > maximum_difference_.z) {
                return -1;
            }
            return lookup_[flatten(dx, dy, dz)];
        }

      private:
        [[nodiscard]] size_t flatten(int x, int y, int z) const {
            size_t x_index = static_cast<size_t>(x - minimum_difference_.x);
            size_t y_index = static_cast<size_t>(y - minimum_difference_.y);
            size_t z_index = static_cast<size_t>(z - minimum_difference_.z);
            return (x_index * static_cast<size_t>(range_y_) + y_index) *
                       static_cast<size_t>(range_z_) +
                   z_index;
        }

        IntegerVector3D minimum_difference_{0, 0, 0};
        IntegerVector3D maximum_difference_{0, 0, 0};
        int range_y_ = 1;
        int range_z_ = 1;
        std::vector<int> lookup_;
    };

    /**
     * <psi_nk|Vxc|psi_nk> = sum_{G,G'} conj(c_nk(G)) Vxc(G-G') c_nk(G'), in
     * Rydberg (see class comment for the derivation and normalization).
     * O(number_of_plane_waves^2) per band -- ~2.3e6 complex multiply-adds
     * for a 1500-plane-wave k-point, negligible next to the SCF cost that
     * already produced psi_nk and Vxc(G).
     */
    Complex compute_diagonal_matrix_element_ry(const std::vector<IntegerVector3D>& g_vectors,
                                               const std::vector<int>& g_indices_for_kpoint,
                                               int number_of_plane_waves,
                                               const Complex* coefficients,
                                               const std::vector<Complex>& vxc_g,
                                               const MillerDifferenceIndex& miller_lookup,
                                               Statistics& statistics) const {
        Complex matrix_element(0.0, 0.0);
        for (int bra_index = 0; bra_index < number_of_plane_waves; bra_index++) {
            const IntegerVector3D& bra_miller = g_vectors[g_indices_for_kpoint[bra_index]];
            Complex bra_coefficient_conjugate = std::conj(coefficients[bra_index]);
            for (int ket_index = 0; ket_index < number_of_plane_waves; ket_index++) {
                const IntegerVector3D& ket_miller = g_vectors[g_indices_for_kpoint[ket_index]];
                statistics.total_g_difference_pairs++;
                int vxc_index = miller_lookup.find(bra_miller.x - ket_miller.x,
                                                   bra_miller.y - ket_miller.y,
                                                   bra_miller.z - ket_miller.z);
                if (vxc_index < 0) {
                    statistics.out_of_set_g_difference_pairs++;
                    continue;
                }
                matrix_element +=
                    bra_coefficient_conjugate * vxc_g[vxc_index] * coefficients[ket_index];
            }
        }
        return matrix_element;
    }

    /// Compute every k-point's per-band diagonal matrix elements (eV), accumulating statistics.
    std::vector<KPointBlock>
    compute_all_kpoint_blocks(const std::vector<IntegerVector3D>& g_vectors,
                              const std::vector<Complex>& vxc_g,
                              const MillerDifferenceIndex& miller_lookup,
                              Statistics& statistics) const {
        const auto& kpoint_g_vector_indices = record_.kpoint_g_vector_indices;
        const auto& plane_waves_per_kpoint = record_.plane_waves_per_kpoint;
        int number_of_kpoints = record_.number_of_kpoints();
        int number_of_bands = record_.number_of_bands;

        std::vector<KPointBlock> blocks(number_of_kpoints);
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            blocks[kpoint_index].kpoint_crystal_coordinates =
                record_.kpoint_coordinates[kpoint_index];
            blocks[kpoint_index].diagonal_elements_ev.resize(number_of_bands);

            int number_of_plane_waves = plane_waves_per_kpoint[kpoint_index];
            const auto& g_indices_for_kpoint = kpoint_g_vector_indices[kpoint_index];

            /**
             * write_to_file() has already rejected nspin>1, so kpoint_state_index ==
             * kpoint_index unconditionally (spin_index is always 0) -- same as WFN's write_body().
             */
            for (int band_index = 0; band_index < number_of_bands; band_index++) {
                const Complex* coefficients =
                    record_.band_source->coefficients(kpoint_index, band_index);
                Complex element_ry = compute_diagonal_matrix_element_ry(g_vectors,
                                                                        g_indices_for_kpoint,
                                                                        number_of_plane_waves,
                                                                        coefficients,
                                                                        vxc_g,
                                                                        miller_lookup,
                                                                        statistics);
                statistics.maximum_absolute_imaginary_part_ry = std::max(
                    statistics.maximum_absolute_imaginary_part_ry, std::abs(element_ry.imag()));
                blocks[kpoint_index].diagonal_elements_ev[band_index] =
                    element_ry * constants::RY_TO_EV;
            }
        }
        return blocks;
    }

    /**
     * Fixed-width Fortran-F-edit-descriptor-style formatting: `width` total
     * characters, `decimal_places` digits after the decimal point,
     * right-justified with leading spaces -- matches `%*.*f`'s own
     * semantics exactly, so this is a thin, clearly-named wrapper. The
     * independent reader (tests/helpers_berkeleygw.py) parses by slicing
     * these exact fixed widths with NO separating whitespace assumed
     * between fields, so columns must match byte-for-byte.
     */
    static std::string format_fixed_width_real(Real value, int width, int decimal_places) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%*.*f", width, decimal_places, value);
        return std::string(buffer);
    }

    /// Fixed-width Fortran-I-edit-descriptor-style integer formatting (right-justified).
    static std::string format_fixed_width_int(int value, int width) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%*d", width, value);
        return std::string(buffer);
    }

    /**
     * One k-point block: header line `FORMAT (3F13.9, 2I8)` = `kx ky kz
     * ndiag noffdiag` (noffdiag always 0, diagonal-only scope), then one
     * `FORMAT (2I8, 2F15.9)` = `spin band Re(Vxc) Im(Vxc)` line per band
     * (spin always 1: write_to_file() has already rejected nspin>1).
     */
    void write_kpoint_block(std::ofstream& output_file, const KPointBlock& block) const {
        int number_of_bands = static_cast<int>(block.diagonal_elements_ev.size());

        output_file << format_fixed_width_real(block.kpoint_crystal_coordinates.x, 13, 9)
                    << format_fixed_width_real(block.kpoint_crystal_coordinates.y, 13, 9)
                    << format_fixed_width_real(block.kpoint_crystal_coordinates.z, 13, 9)
                    << format_fixed_width_int(number_of_bands, 8) << format_fixed_width_int(0, 8)
                    << "\n";

        for (int band_index = 0; band_index < number_of_bands; band_index++) {
            const Complex& element = block.diagonal_elements_ev[band_index];
            output_file << format_fixed_width_int(1, 8) << format_fixed_width_int(band_index + 1, 8)
                        << format_fixed_width_real(element.real(), 15, 9)
                        << format_fixed_width_real(element.imag(), 15, 9) << "\n";
        }
    }
};

} // namespace q3
