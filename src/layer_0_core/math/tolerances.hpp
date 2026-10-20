#pragma once

/**
 * Numerical tolerances and thresholds used throughout Q3.
 *
 * All values are defined as `inline constexpr` to ensure a single definition
 * across translation units. Separated into pure mathematical thresholds and
 * physics-specific thresholds.
 */

namespace q3 {
namespace math::tolerances {

/**
 * ─── Pure Mathematical Thresholds ────────────────────────────────────────────
 * These are numerical-analysis constants independent of any physical domain.
 */

/**
 * Threshold below which a vector magnitude is treated as zero.
 */
inline constexpr double zero_vector_magnitude_threshold = 1e-10;

/**
 * Threshold below which a floating-point value is treated as numerically zero.
 *  Used for quantities like overlap coefficients, azimuthal angles, etc.
 */
inline constexpr double numerical_zero_threshold = 1e-12;

/**
 * Minimum finite difference step for numerical derivatives.
 */
inline constexpr double minimum_finite_difference_step = 1e-10;

/**
 * Threshold below which a vector norm squared is treated as zero in iterative solvers.
 *  Used in Davidson diagonalizer for residual norm and orthogonalization checks.
 */
inline constexpr double iterative_solver_norm_threshold = 1e-14;

/**
 * Floor on the magnitude a relative change is measured against, so that an
 * objective that has reached exactly zero has a relative change at all.
 */
inline constexpr double relative_change_scale_floor = 1e-30;

/**
 * Minimum denominator value for preconditioner in iterative eigensolvers.
 */
inline constexpr double preconditioner_denominator_floor = 1e-4;

/**
 * Empty (unoccupied) bands converge to a looser tolerance than occupied bands:
 * their eigenvalues do not enter the total energy, so tight convergence is
 * wasted work. The effective empty-band tolerance is
 * max(empty_band_tolerance_relaxation_factor * tolerance, empty_band_tolerance_floor).
 *
 * Both numbers are iteration-path knobs, not answers: they decide when a band
 * stops being refined, never what it converges to, so no result in the test
 * suite distinguishes these values from neighbouring ones. The floor is what
 * stops the relaxed tolerance chasing the occupied one down as the SCF
 * tightens; the factor is small enough that a badly-converged empty band still
 * cannot pollute the subspace it spans.
 */
inline constexpr double empty_band_tolerance_relaxation_factor = 5.0;
inline constexpr double empty_band_tolerance_floor = 1e-5;

/**
 * Threshold below which a Gaunt coefficient is considered insignificant.
 */
inline constexpr double gaunt_coefficient_threshold = 1e-6;

/**
 * Minimum overlap-matrix eigenvalue accepted by Lowdin orthogonalization.
 * Below this the atomic-orbital overlap S is treated as singular (linearly
 * dependent orbitals) and S^{-1/2} would amplify noise by >= 1e5.
 */
inline constexpr double lowdin_minimum_overlap_eigenvalue = 1e-10;

/**
 * ─── Physics-Specific Thresholds ─────────────────────────────────────────────
 * These encode domain knowledge about DFT, plane-wave methods, or specific
 * physical quantities.
 */

/**
 * Whether to truncate radial integrations at a finite radius.
 */
inline constexpr bool use_radial_integration_truncation = true;

/**
 * Truncation radius (Bohr) for radial integration.
 */
inline constexpr double radial_integration_truncation_radius = 10.0;

/**
 * Threshold below which a D-matrix element is treated as zero.
 */
inline constexpr double d_matrix_element_threshold = 1e-15;

/**
 * Threshold below which a cell volume is treated as zero (uninitialized).
 */
inline constexpr double cell_volume_threshold = 1e-10;

/**
 * Default q-grid spacing (in atomic units) for radial Fourier transform interpolation tables.
 */
inline constexpr double default_q_grid_spacing = 0.01;

/**
 * Default number of q-points for atomic wavefunction interpolation tables.
 */
inline constexpr int default_number_of_q_points_for_atomic_wavefunctions = 2000;

/**
 * Threshold below which an occupation number is treated as zero (unoccupied).
 */
inline constexpr double occupation_threshold = 1e-10;

/**
 * Threshold below which a smearing width is treated as zero (insulating).
 */
inline constexpr double smearing_width_threshold = 1e-10;

/**
 * Rounding tolerance for energy cutoff comparisons (G-vector inclusion).
 */
inline constexpr double energy_cutoff_rounding_tolerance = 1e-8;

/**
 * Default symmetry precision for spglib operations (in Bohr).
 */
inline constexpr double default_symmetry_precision = 1e-5;

/**
 * How far a fractional translation component may sit from 1/n and still be read
 * as 1/n when the FFT grid is asked to be divisible by n.
 *
 * The translations come from spglib and are accurate to the symmetry precision
 * it was given, so the bound has to be at least that loose; it is two orders
 * looser so that tightening that precision never starts rejecting a genuine
 * fraction. What it must separate is coarser than either number by far: the
 * closest competing value for the smallest accepted denominator is 1/6 against
 * 1/5 or 1/7, 0.03 away, and the difference that motivated checking at all --
 * 0.375 read as 1/3 -- is 0.04 away.
 */
inline constexpr double fractional_translation_rounding_tolerance = 1e-3;

/**
 * Finite difference step size for numerical gradient computation in reciprocal space.
 */
inline constexpr double reciprocal_space_finite_difference_step = 1e-5;

/**
 * Threshold below which a spherical Bessel function argument is treated as zero.
 */
inline constexpr double bessel_argument_zero_threshold = 1e-8;

/**
 * Convergence threshold for Fermi energy bisection.
 */
inline constexpr double fermi_energy_bisection_convergence = 1e-12;

/**
 * Electron-count slack that still counts as "on the plateau" when bracketing
 * a gap to place E_F at its midpoint.
 *
 * Across a gap the smeared electron count equals the target only up to the
 * occupation tails, so the upper-edge bisection needs a tolerance rather than
 * an exact equality. Loose enough to span a real gap's residual tails. A
 * metal has a bracket too, this slack over its density of states, and the
 * midpoint of that one would leave half the slack in the cell as extra charge;
 * the calculator tells the two apart by comparing the bracket with the
 * smearing width, not by this value.
 */
inline constexpr double fermi_electron_count_plateau = 1e-6;

/**
 * How far a k-point weight sum may sit from the spin degeneracy and still count
 * as a Brillouin-zone sampling.
 *
 * Loose because the sum is built from symmetry multiplicities over a point
 * count and carries ordinary rounding; the case it exists to catch misses by
 * whole integers rather than by rounding, since a band path gives every point
 * weight 1 and its sum is the number of points.
 */
inline constexpr double kpoint_weight_sum = 1e-6;

/**
 * How far n * k may sit from the common grid offset, in units of the grid
 * step, and still count as a point of an n-division grid when a grid is
 * recovered from an explicit k-point list.
 *
 * Listed coordinates carry their printing noise, ~1e-10, which the largest
 * grid tried (1000 divisions) magnifies to ~1e-7. A point genuinely off an
 * n-division grid, k = p/m, misses it by at least 1/m of a step -- 1e-3 for
 * the finest grid tried. A q0 displacement of the whole list is common to
 * every point, so it moves the offset rather than the test.
 */
inline constexpr double kpoint_grid_offset_tolerance = 1e-6;

/**
 * Initial eigenvalue diagonalization tolerance for early SCF iterations.
 */
inline constexpr double initial_eigenvalue_tolerance = 1e-2;

/**
 * Floor for adaptive eigenvalue tolerance to prevent numerical issues.
 */
inline constexpr double minimum_eigenvalue_tolerance = 1e-13;

/**
 * Minimum electron density value to prevent division by zero in XC functionals.
 */
inline constexpr double minimum_electron_density = 1e-12;

/**
 * Ewald summation convergence check threshold.
 *
 * Also used by MartynaTuckermanCorrectionCalculator's convergence-parameter
 * search (select_convergence_parameters()), which bounds the same kind of
 * erf/erfc-smoothed reciprocal-space truncation error, at the same value
 * Quantum ESPRESSO uses when it sets up its own Martyna-Tuckerman correction.
 */
inline constexpr double ewald_convergence_threshold = 1e-7;

/**
 * Default threshold for energy consistency checks.
 */
inline constexpr double energy_consistency_threshold = 1e-8;

/**
 * Relative finite difference step scale for stress tensor gradient computation.
 */
inline constexpr double stress_finite_difference_scale = 1e-6;

/**
 * Threshold for the sum of |k_x| + |k_y| + |k_z| (in crystal coordinates)
 * below which a k-point is treated as the Gamma point.
 */
inline constexpr double kpoint_gamma_threshold = 1e-10;

/**
 * Smallest singular value of the trial-orbital projection A(k) below which
 * the orbitals are taken not to span the target manifold at that k-point: the
 * projection gauge A (A^dagger A)^(-1/2) is then singular, and is refused.
 */
inline constexpr double wannier_projection_singular_value_threshold = 1e-8;

/**
 * Modulus of a diagonal neighbour overlap M_nn(k,b) below which the Berry
 * phase of Wannier function n -- its centre and its spread -- is undefined:
 * the gauge has collapsed.
 */
inline constexpr double wannier_diagonal_overlap_threshold = 1e-10;

/**
 * How negative the gauge-invariant spread may come out, per Wannier function
 * in bohr squared, before it is read as overlaps that are not between
 * orthonormal states rather than as rounding. Omega_I cannot be negative.
 */
inline constexpr double wannier_negative_spread_tolerance = 1e-8;

/**
 * Relative width of a shell of lattice vectors: two |R| closer than this
 * share a shell when the decay of a real-space Hamiltonian is measured on
 * its outermost one.
 */
inline constexpr double lattice_shell_relative_width = 1e-6;

} // namespace math::tolerances
} // namespace q3
