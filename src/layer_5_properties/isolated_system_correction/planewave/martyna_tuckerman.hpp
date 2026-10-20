#pragma once

/**
 * MartynaTuckermanCorrectionCalculator — reciprocal-space isolated-system
 * (finite-size electrostatic) correction, Martyna-Tuckerman kernel.
 *
 * G. J. Martyna and M. E. Tuckerman, "A reciprocal space based method for
 * treating long range interactions in ab initio and force-field-based
 * calculations in clusters," J. Chem. Phys. 110, 2810 (1999).
 *
 * *** SCOPE NOTICE — read before relying on this class ***
 * This is the POST-SCF, perturbative (non-self-consistent) evaluation of the
 * Martyna-Tuckerman correction: it is evaluated ONCE, on the density that
 * already converged under the ordinary PERIODIC Coulomb kernel. It is NOT
 * the same quantity as Quantum ESPRESSO's `assume_isolated='m-t'`, which is
 * self-consistent -- QE folds the MT-corrected local potential into the SCF
 * Hamiltonian on every iteration, so its density relaxes under the *corrected*
 * potential before its energy is ever evaluated. Q3's
 * isolated-system-correction interface (see init.hpp) is architecturally
 * post-SCF -- every scheme is evaluated once, after convergence, exactly
 * like MakovPayneCorrectionCalculator -- so
 * this class reports what the MT kernel difference contributes to the
 * ALREADY-CONVERGED (periodic-kernel) density, not what QE would converge to
 * under its own corrected potential. The two numbers coincide only to the
 * extent that MT's potential correction perturbs the self-consistent density
 * weakly (a reasonable expectation for compact, weakly-polarizable systems;
 * not guaranteed for strongly localized or highly charged ones). The
 * self-consistent version (feeding the correction back into V_eff every SCF
 * iteration) is tracked as future work -- see TODOs.md.
 *
 * Physical quantity computed: the difference between the classical
 * electrostatic energy of the system's TOTAL (ionic + electronic) charge
 * distribution evaluated with the Martyna-Tuckerman cutoff Coulomb kernel,
 * and the same quantity evaluated with the ordinary periodic (4*pi/G^2)
 * Coulomb kernel -- i.e. exactly the spurious periodic-image electrostatic
 * self-interaction energy that MakovPayneCorrectionCalculator's
 * monopole+dipole+quadrupole SERIES approximates, but summed EXACTLY here
 * (every multipole order at once, via a reciprocal-space kernel difference)
 * rather than via a truncated multipole expansion:
 *
 *   E_MT = (e^2 / 2) * Omega * sum_G |rho_total(G)|^2 * W(G)
 *
 * where:
 *   rho_total(G) = rho_ion(G) - rho_el(G)   (net charge density, G-space;
 *     positive sign = net positive/"hole" charge, matching
 *     MakovPayneCorrectionCalculator's net_charge convention)
 *   rho_ion(G) = (1/Omega) * sum_atoms Z_valence * exp(-i * 2*pi * G . tau)
 *     (ionic structure factor over the cell volume, same convention as
 *     compute_ewald())
 *   rho_el(G) = forward FFT of the spin-summed converged electron density
 *     (PlanewaveBasis::rho_r_to_g())
 *   W(G) = the MT kernel-difference function (Eq. 4.3 of the paper above;
 *     Quantum ESPRESSO calls the same quantity "wg_corr"): the
 *     reciprocal-space representation of "the truncated, erf-smoothed
 *     Coulomb potential, periodized by minimum-image wrapping into the
 *     simulation cell" MINUS "the ordinary periodic erf-smoothed Coulomb
 *     kernel," damped by a residual Gaussian
 *     filter for numerical convergence. The sum runs over ALL G including
 *     G=0 (unlike the ordinary Hartree/Ewald sums, which omit G=0 by
 *     convention) -- W(0) is finite by construction, which is exactly what
 *     lets this scheme handle a net cell charge without an external
 *     compensating-background assumption.
 *
 * W(G) construction, following Eq. 4.3 of Martyna & Tuckerman (1999) and its
 * surrounding discussion, cross-checked numerically against Quantum ESPRESSO:
 *   1. Pick the convergence parameter alpha (bohr^-2) by scanning DOWNWARD
 *      from 2.9 in steps of 0.1 and stopping at the first value whose
 *      erfc-tail truncation error at the charge-density G-cutoff is below
 *      math::tolerances::ewald_convergence_threshold (1e-7). That bound rises
 *      monotonically with alpha, so the descending scan returns the LARGEST
 *      admissible alpha on that grid (never 2.9 itself: the decrement
 *      precedes the first test), not the smallest one --
 *      structurally the same kind of erf/erfc-smoothed reciprocal-space
 *      search Q3's own Ewald sum performs, but NOT numerically identical to
 *      it (Q3's compute_ewald() search bounds a charge-scaled self-energy
 *      criterion with a `sqrt(2*alpha/pi)` factor; the MT search below uses
 *      `sqrt(2*alpha/(2*pi))` with no charge scaling) --
 *      select_convergence_parameters() below matches the MT criterion, which
 *      Quantum ESPRESSO uses as well, not compute_ewald()'s. beta = 0.5/alpha
 *      is a free filter-width choice, not a derived value; step 3 shows what
 *      it is coupled to.
 *   2. Evaluate, at every point of the SAME dense real-space grid the charge
 *      density lives on, evaluate_smoothed_coulomb_kernel_real_space(r) =
 *      erf(sqrt(alpha)*r)/r (r -> 0 limit: 2*sqrt(alpha/pi)), where r is the
 *      point's MINIMUM-IMAGE distance from the origin (the lattice's own
 *      periodicity, independent of where any atom sits). Forward-FFT that
 *      array (PlanewaveBasis::rho_r_to_g()) to obtain, at every G, the
 *      reciprocal-space coefficient of the periodized truncated kernel.
 *   3. W(G) = [Omega * Re(that coefficient) -
 *      evaluate_smoothed_coulomb_kernel_reciprocal_space(G^2)], multiplied
 *      by exp(-G^2*beta/4)^2, where the reciprocal-space kernel is the
 *      free-space FT of the real-space one for G != 0 (the paper's Eq. 4.3),
 *      and at G = 0 the one value that keeps the FILTERED total kernel
 *      continuous there: -4*pi*(1/(4*alpha) + 2*beta/4). Its first term is
 *      the analytic free-space integral of the complementary (erfc)
 *      short-range part, its second the O(G^2) coefficient the filter leaves
 *      behind against the 4*pi/G^2 pole -- so the value is tied to beta and
 *      to the filter's form, not to alpha alone. Both terms are derived in
 *      evaluate_smoothed_coulomb_kernel_reciprocal_space() below; Quantum
 *      ESPRESSO arrives at the same value.
 *
 * MINIMUM-IMAGE DISTANCE CAVEAT: step 2's "minimum-image distance from the
 * origin" wraps fractional coordinates into [-0.5, 0.5) (frac -= round(frac))
 * under the cell's OWN (possibly oblique) lattice vectors -- the same
 * approximation MakovPayneCorrectionCalculator already uses for its own
 * electronic dipole/quadrupole moments -- and is NOT the exact shortest
 * distance to the Bravais lattice (the Wigner-Seitz distance), which Quantum
 * ESPRESSO obtains by starting from this same fractional wrap and then
 * scanning neighboring images for a shorter one. The two coincide exactly
 * for orthogonal lattices (simple cubic, bravais_lattice_index 1), and for
 * ANY lattice inside the centered PARALLELEPIPED's inscribed sphere --
 * radius = half the smallest interplanar spacing, min_i(Omega/|a_j x a_k|),
 * which is strictly smaller than half the shortest lattice vector (the
 * Wigner-Seitz cell's own inscribed radius) whenever the primitive cell is
 * oblique: FCC 0.289*a vs 0.354*a, BCC 0.354*a vs 0.433*a. Outside that
 * smaller sphere the wrap can return a farther image than the true nearest
 * one (FCC example: a point at 0.30*a along the normal to the (a2,a3) plane
 * has fractional coordinate 0.52, so it wraps to a representative at
 * 0.494*a), i.e. near the cell boundary of an oblique primitive cell
 * (e.g. FCC/BCC, bravais_lattice_index 2/3) -- see the validity check below.
 *
 * VALIDITY ASSUMPTION ("density localized within half the cell"): the total
 * charge density must be negligible outside the region where the
 * parallelepiped minimum-image proxy above is exact. The check below uses
 * a sphere of radius = half the shortest lattice vector, centered on the
 * valence-charge-weighted ionic center x0 (same x0 convention as
 * MakovPayneCorrectionCalculator). This is checked cheaply -- folded into
 * the same real-space grid pass that builds the kernel array, so it adds no
 * extra asymptotic cost -- and logs a warning (NOT a throw, unlike
 * Makov-Payne's cubic-lattice gate) when a nontrivial fraction of the
 * density sits outside that sphere: (a) the check is itself a heuristic
 * proxy, not an exact geometric membership test against the true
 * Wigner-Seitz polyhedron, and (b) MT is legitimately lattice-general (no
 * lattice-type gate at all -- contrast MakovPayneCorrectionCalculator's
 * cubic-only throw). A fired warning means this correction's accuracy is
 * not guaranteed for that system, but the run is not blocked.
 *
 * RELATION TO QE's wg_corr_h()/wg_corr_ewald(): QE's own public API returns
 * TWO separate ENERGY numbers built on this same W(G) -- an electron-electron
 * Hartree-like correction (wg_corr_h(), using rho_el(G) alone) and an
 * ion-ion Ewald-like correction (wg_corr_ewald(), using rho_ion(G) alone) --
 * alongside a force correction (wg_corr_force()) that this class has no
 * counterpart for: the energy reported here carries no matching force term,
 * consistent with IsolatedSystemCorrectionCalculator reporting labeled energy
 * terms only (init.hpp) rather than feeding forces or relaxation.
 * The electron-ion CROSS term is not a third public number in QE: it is
 * captured only through wg_corr_loc(), a correction to the local potential
 * that feeds back into the SCF Hamiltonian and therefore into the
 * eigenvalue sum/double-counting bookkeeping -- exactly the self-consistent
 * machinery this class does not have (see the scope notice above). To still
 * capture that cross term -- needed for physical completeness, and for this
 * class's own "reduces to the Makov-Payne monopole term as the cell grows"
 * limit, see test_martyna_tuckerman.cpp -- this class sums
 * |rho_ion(G) - rho_el(G)|^2 * W(G) directly over the TOTAL charge, which
 * expands into wg_corr_ewald + wg_corr_h plus the (otherwise-missing)
 * electron-ion cross term: the mathematically complete, scheme-consistent
 * generalization of "the classical electrostatic energy of the whole
 * system's charge distribution, MT kernel minus periodic kernel" -- not
 * merely the sum of QE's own two public functions.
 *
 * Reported as a single term ("energy"), matching this interface's
 * scheme-agnostic design (IsolatedSystemCorrectionResult's docstring) --
 * unlike Makov-Payne, MT has no natural split into "orders," since the
 * reciprocal-space sum captures every multipole at once.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/math/phase_factor.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_5_properties/isolated_system_correction/planewave/init.hpp"

namespace q3 {

class MartynaTuckermanCorrectionCalculator : public IsolatedSystemCorrectionCalculator {
  public:
    [[nodiscard]] IsolatedSystemCorrectionResult
    compute(const NormConservingPlanewaveKohnShamSCFState& scf_state) const override {
        const Crystal& crystal_structure = *scf_state.crystal_structure;
        const PlanewaveBasis& planewave_basis = *scf_state.planewave_basis;

        Real cell_volume = crystal_structure.cell_volume();
        Real alat = crystal_structure.lattice_parameter();
        Real two_pi_over_lattice_parameter_squared =
            crystal_structure.two_pi_over_lattice_parameter_squared();

        auto [convergence_alpha, convergence_beta] =
            select_convergence_parameters(planewave_basis.charge_density_energy_cutoff());

        Matrix3x3 lattice_vectors = crystal_structure.lattice_vectors(); // alat units
        Matrix3x3 lattice_vectors_transpose = lattice_vectors.transpose();
        Matrix3x3 lattice_inverse_transpose = lattice_vectors.inverse().transpose();

        /**
         * x0 = valence-charge-weighted ionic center (alat units), same convention as
         * MakovPayneCorrectionCalculator.
         */
        int number_of_atoms = crystal_structure.number_of_atoms();
        Real total_valence_charge = 0.0;
        Vector3D ionic_center_alat{0.0, 0.0, 0.0};
        for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
            Real valence_charge =
                crystal_structure.valence_charge(crystal_structure.species_type_index(atom_index));
            total_valence_charge += valence_charge;
            ionic_center_alat +=
                valence_charge * crystal_structure.convert_from_crystal_to_cartesian(atom_index);
        }
        ionic_center_alat = ionic_center_alat / total_valence_charge;

        /**
         * Cheap locality-validity radius: half the shortest lattice vector, in Bohr.
         * That is exactly the region the parallelepiped proxy and the true
         * Wigner-Seitz distance share for an ORTHOGONAL cell; for an oblique one it
         * is optimistic (too large) -- the guaranteed-exact radius there is half the
         * smallest interplanar spacing -- so the warning can stay silent on a density
         * that already reaches points where the proxy is wrong. See the class
         * docstring's "VALIDITY ASSUMPTION" section.
         */
        Real shortest_lattice_vector_length_alat = std::min(
            {lattice_vectors[0].norm(), lattice_vectors[1].norm(), lattice_vectors[2].norm()});
        Real locality_radius_in_bohr = 0.5 * shortest_lattice_vector_length_alat * alat;

        const FFTGrid& density_grid = planewave_basis.fft_rho();
        int grid_dimension_1 = density_grid.grid_dimension_1();
        int grid_dimension_2 = density_grid.grid_dimension_2();
        int grid_dimension_3 = density_grid.grid_dimension_3();
        int local_grid_points = density_grid.local_grid_points();
        int number_of_spin_components = static_cast<int>(scf_state.charge_density_real.size());

        /**
         * Sweep this rank's part of the dense grid, asking the grid where each
         * local point sits. The local kernel and density arrays are exactly the
         * input layout required by the serial or distributed rho_r_to_g FFT.
         */
        std::vector<Real> kernel_real_space(local_grid_points);
        std::vector<Real> total_charge_density_real(local_grid_points, 0.0);
        Real charge_density_magnitude_within_locality_radius = 0.0;
        Real charge_density_magnitude_total = 0.0;

        for (int local_flat_index = 0; local_flat_index < local_grid_points; local_flat_index++) {
            int global_flat_index = density_grid.global_real_space_index(local_flat_index);
            int index_3 = global_flat_index % grid_dimension_3;
            int index_2 = (global_flat_index / grid_dimension_3) % grid_dimension_2;
            int index_1 = global_flat_index / (grid_dimension_3 * grid_dimension_2);
            Vector3D fractional_grid_point{static_cast<Real>(index_1) / grid_dimension_1,
                                           static_cast<Real>(index_2) / grid_dimension_2,
                                           static_cast<Real>(index_3) / grid_dimension_3};
            Vector3D grid_point_alat = lattice_vectors_transpose * fractional_grid_point;

            Real distance_from_origin_in_bohr = minimum_image_distance(grid_point_alat,
                                                                       Vector3D{0.0, 0.0, 0.0},
                                                                       lattice_vectors_transpose,
                                                                       lattice_inverse_transpose) *
                                                alat;
            kernel_real_space[local_flat_index] = evaluate_smoothed_coulomb_kernel_real_space(
                distance_from_origin_in_bohr, convergence_alpha);

            Real total_density_at_point = 0.0;
            for (int spin_index = 0; spin_index < number_of_spin_components; spin_index++) {
                total_density_at_point +=
                    scf_state.charge_density_real[spin_index][local_flat_index];
            }
            total_charge_density_real[local_flat_index] = total_density_at_point;

            Real distance_from_center_in_bohr = minimum_image_distance(grid_point_alat,
                                                                       ionic_center_alat,
                                                                       lattice_vectors_transpose,
                                                                       lattice_inverse_transpose) *
                                                alat;
            Real density_magnitude = std::abs(total_density_at_point);
            charge_density_magnitude_total += density_magnitude;
            if (distance_from_center_in_bohr <= locality_radius_in_bohr) {
                charge_density_magnitude_within_locality_radius += density_magnitude;
            }
        }

        std::vector<Real> locality_magnitudes = {
            charge_density_magnitude_within_locality_radius,
            charge_density_magnitude_total,
        };
        if (scf_state.execution_policy != nullptr) {
            scf_state.execution_policy->reduce_within_pool_vector(locality_magnitudes);
        }
        charge_density_magnitude_within_locality_radius = locality_magnitudes[0];
        charge_density_magnitude_total = locality_magnitudes[1];

        if (charge_density_magnitude_total > 0.0) {
            Real localized_fraction =
                charge_density_magnitude_within_locality_radius / charge_density_magnitude_total;
            if (localized_fraction < locality_warning_fraction_threshold) {
                log::warn(
                    "MartynaTuckermanCorrectionCalculator: only {:.1f}% of |charge density| lies "
                    "within the cheap Wigner-Seitz-proxy locality radius ({:.4f} bohr, half the "
                    "shortest lattice vector, centered on the ionic center) -- the "
                    "Martyna-Tuckerman post-SCF correction assumes the density is well localized "
                    "inside the cell (see this class's docstring); this system may violate that "
                    "assumption and the reported correction may not be accurate.",
                    localized_fraction * 100.0,
                    locality_radius_in_bohr);
            }
        }

        /**
         * Forward-FFT both the kernel-construction array and the (spin-summed) electron
         * density onto the same charge-density G-vector set (PlanewaveBasis::g_vectors()).
         */
        std::vector<Complex> kernel_g_space;
        planewave_basis.rho_r_to_g(kernel_real_space, kernel_g_space);
        std::vector<Complex> electron_density_g_space;
        planewave_basis.rho_r_to_g(total_charge_density_real, electron_density_g_space);

        int number_of_g_vectors = planewave_basis.number_of_g_vectors();
        int first_nonzero_g_index = planewave_basis.first_nonzero_g_index();
        bool is_gamma_only = planewave_basis.is_gamma_only();

        Real energy_sum = 0.0;
        for (int g_index = 0; g_index < number_of_g_vectors; g_index++) {
            const GVector& g_vector = planewave_basis.g_vectors()[g_index];
            Real g_squared_in_inverse_bohr_squared =
                g_vector.magnitude_squared * two_pi_over_lattice_parameter_squared;

            Complex ionic_structure_factor(0.0, 0.0);
            for (int atom_index = 0; atom_index < number_of_atoms; atom_index++) {
                Vector3D atomic_position_alat =
                    crystal_structure.convert_from_crystal_to_cartesian(atom_index);
                Real phase_angle = constants::two_pi * g_vector.g_cart.dot(atomic_position_alat);
                ionic_structure_factor += crystal_structure.valence_charge(
                                              crystal_structure.species_type_index(atom_index)) *
                                          compute_phase_factor(phase_angle);
            }
            Complex ionic_charge_density_g = ionic_structure_factor / cell_volume;

            Real reciprocal_space_kernel = evaluate_smoothed_coulomb_kernel_reciprocal_space(
                g_squared_in_inverse_bohr_squared, convergence_alpha, convergence_beta);
            Real gaussian_filter =
                std::exp(-g_squared_in_inverse_bohr_squared * convergence_beta / 4.0);
            Real kernel_difference =
                (cell_volume * kernel_g_space[g_index].real() - reciprocal_space_kernel) *
                gaussian_filter * gaussian_filter;

            Complex total_charge_density_g =
                ionic_charge_density_g - electron_density_g_space[g_index];

            /**
             * Gamma-only half-sphere: each G != 0 represents both G and -G, and
             * |rho(-G)|^2 = |rho(G)|^2 by Hermitian symmetry, so double the
             * contribution -- matches compute_ewald()/compute_hartree_potential()'s own
             * convention. G=0 (g_index < first_nonzero_g_index) is never doubled: it has
             * no separate -G partner.
             */
            Real gamma_factor = (is_gamma_only && g_index >= first_nonzero_g_index) ? 2.0 : 1.0;
            energy_sum += std::norm(total_charge_density_g) * kernel_difference * gamma_factor;
        }

        Real martyna_tuckerman_energy_in_ry =
            0.5 * constants::electron_charge_squared_rydberg * cell_volume * energy_sum;

        IsolatedSystemCorrectionResult result;
        result.applied = true;
        result.method = name();
        result.terms_in_ry = {{"energy", martyna_tuckerman_energy_in_ry}};
        return result;
    }

    [[nodiscard]] std::string name() const override {
        return "martyna-tuckerman";
    }

  private:
    /**
     * Fraction of |charge density| that must lie within the locality radius before
     * MartynaTuckermanCorrectionCalculator logs a validity warning. A heuristic
     * threshold (not derived from error-propagation analysis) -- see the class
     * docstring's "VALIDITY ASSUMPTION" section.
     */
    static constexpr Real locality_warning_fraction_threshold = 0.99;

    /**
     * Minimum-image displacement norm (alat units) of `point` from `reference`, under
     * the (possibly oblique) lattice given by `lattice_vectors_transpose` /
     * `lattice_inverse_transpose` -- wraps the fractional displacement into [-0.5, 0.5)
     * exactly like MakovPayneCorrectionCalculator's own electronic dipole/quadrupole
     * grid sum. See the class docstring's minimum-image-distance caveat for why this is
     * a proxy for, not identical to, the true Wigner-Seitz-cell distance.
     */
    [[nodiscard]] static Real minimum_image_distance(const Vector3D& point,
                                                     const Vector3D& reference,
                                                     const Matrix3x3& lattice_vectors_transpose,
                                                     const Matrix3x3& lattice_inverse_transpose) {
        Vector3D displacement = point - reference;
        Vector3D fractional_displacement = lattice_inverse_transpose * displacement;
        fractional_displacement.x -= std::round(fractional_displacement.x);
        fractional_displacement.y -= std::round(fractional_displacement.y);
        fractional_displacement.z -= std::round(fractional_displacement.z);
        Vector3D wrapped_displacement = lattice_vectors_transpose * fractional_displacement;
        return wrapped_displacement.norm();
    }

    /**
     * Select the Martyna-Tuckerman convergence parameter alpha (bohr^-2) and its
     * derived beta = 0.5/alpha, matching the MT convergence criterion as Quantum
     * ESPRESSO also applies it (NOT compute_ewald()'s differently-normalized search
     * -- see the class docstring): scanning downward from 2.9 in steps of 0.1, take
     * the first alpha reached -- i.e. the LARGEST one on that grid, since the bound
     * below rises monotonically with alpha -- whose erfc-tail truncation error at
     * the given charge-density energy cutoff is below
     * math::tolerances::ewald_convergence_threshold.
     *
     * The 2.9 start, the 0.1 step and beta's 0.5 factor are free choices of the
     * search schedule, the same values Quantum ESPRESSO uses, not quantities
     * derived here; any schedule meeting the same tolerance (different start or
     * step, bisection, or direct inversion of the bound) would serve equally.
     */
    [[nodiscard]] static std::pair<Real, Real>
    select_convergence_parameters(Real charge_density_energy_cutoff_in_ry) {
        Real convergence_alpha = 2.9;
        Real truncation_error_upper_bound = 1.0;
        while (truncation_error_upper_bound > math::tolerances::ewald_convergence_threshold) {
            convergence_alpha -= 0.1;
            if (convergence_alpha <= 0.0) {
                throw std::runtime_error(
                    "MartynaTuckermanCorrectionCalculator: could not find a convergence "
                    "parameter alpha for the given charge-density cutoff (the same "
                    "failure mode Quantum ESPRESSO reports here; should not happen for "
                    "any physically reasonable cutoff)");
            }
            truncation_error_upper_bound = constants::electron_charge_squared_rydberg *
                                           std::sqrt(2.0 * convergence_alpha / constants::two_pi) *
                                           std::erfc(std::sqrt(charge_density_energy_cutoff_in_ry /
                                                               (4.0 * convergence_alpha)));
        }
        Real convergence_beta = 0.5 / convergence_alpha;
        return {convergence_alpha, convergence_beta};
    }

    /// The MT smoothed kernel erf(sqrt(alpha)*r)/r, r -> 0 limit 2*sqrt(alpha/pi).
    [[nodiscard]] static Real evaluate_smoothed_coulomb_kernel_real_space(Real distance_in_bohr,
                                                                          Real convergence_alpha) {
        if (distance_in_bohr > constants::epsilon_1e_minus_6) {
            return std::erf(std::sqrt(convergence_alpha) * distance_in_bohr) / distance_in_bohr;
        }
        return 2.0 / constants::sqrt_pi * std::sqrt(convergence_alpha);
    }

    /**
     * Free-space Fourier transform of evaluate_smoothed_coulomb_kernel_real_space()
     * for G != 0 (the paper's Eq. 4.3), and at G = 0 the value that keeps the
     * FILTERED total kernel continuous.
     *
     * Derivation, in kernel units: with A(G) the (finite) G-space coefficient of
     * the periodized real-space kernel, the total kernel for G != 0 is
     *   v(G) = 4*pi/G^2 + [A(G) - 4*pi*exp(-G^2/(4*alpha))/G^2] * exp(-G^2*beta/2).
     * Expanding 4*pi*exp(-G^2/(4*alpha))/G^2 = 4*pi/G^2 - pi/alpha + O(G^2) and
     * exp(-G^2*beta/2) = 1 - G^2*beta/2 + O(G^4), the poles cancel and the filter
     * converts the pole into a finite term:
     *   lim_{G->0} v(G) = A(0) + pi/alpha + 2*pi*beta.
     * Returning -4*pi*(1/(4*alpha) + 2*beta/4) = -(pi/alpha + 2*pi*beta) at G = 0
     * makes W(0) = A(0) + pi/alpha + 2*pi*beta reproduce exactly that limit. The
     * first term is the analytic free-space integral of the erfc short-range part
     * (integral of erfc(sqrt(alpha)*r)/r over all space = pi/alpha); the second is
     * the filter's own O(G^2) coefficient. The value therefore moves with beta and
     * with the filter's form, not with alpha alone. Quantum ESPRESSO arrives at the
     * same value.
     */
    [[nodiscard]] static Real evaluate_smoothed_coulomb_kernel_reciprocal_space(
        Real g_squared_in_inverse_bohr_squared, Real convergence_alpha, Real convergence_beta) {
        if (g_squared_in_inverse_bohr_squared > constants::epsilon_1e_minus_6) {
            return constants::four_pi *
                   std::exp(-g_squared_in_inverse_bohr_squared / (4.0 * convergence_alpha)) /
                   g_squared_in_inverse_bohr_squared;
        }
        return -constants::four_pi *
               (1.0 / (4.0 * convergence_alpha) + 2.0 * convergence_beta / 4.0);
    }
};

} // namespace q3
