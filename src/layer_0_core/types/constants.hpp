#pragma once

/**
 * Physical constants (SI, CGS, atomic units).
 *
 * The SI values are the NIST 2018 CODATA recommended values; four of them
 * (Planck, Boltzmann, elementary charge, electronvolt) are exact by the 2019
 * SI redefinition rather than measured. CODATA is the authority: any code
 * quoting the same revision quotes the same digits.
 *
 * Derived factors are one-line dimensional identities over those values --
 * there is only one way to convert Hartree to eV -- evaluated in IEEE-754
 * double. Agreement with another code's conversion factors is therefore a
 * consequence of agreeing on CODATA, not of arranging the arithmetic in any
 * particular way, and nothing in the test suite gates a last-ulp difference.
 */

#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {
namespace constants {

// ─── Mathematical constants ─────────────────────────────────────────────────
inline constexpr Real pi = 3.14159265358979323846;
inline constexpr Real two_pi = 2.0 * pi;  // 2π
inline constexpr Real four_pi = 4.0 * pi; // 4π
inline constexpr Real sqrt_pi = 1.77245385090551602729;
inline constexpr Real sqrt2 = 1.41421356237309504880;
inline constexpr Real degrees_to_radians = pi / 180.0;

// ─── Rydberg atomic units ───────────────────────────────────────────────────
inline constexpr Real electron_charge_squared_rydberg = 2.0; // e² in Ry units
inline constexpr Real spin_degeneracy = 2.0;                 // number of spins per level

// ─── Physics defaults ───────────────────────────────────────────────────────
/**
 * Starting magnetization applied to every species when a spin-polarized (LSDA)
 * run gives no starting_magnetization at all: large enough to break spin
 * symmetry so a magnetic ground state can be found, small enough not to bias
 * the SCF toward an artificial high-moment minimum. Explicit user values —
 * including explicit zeros — are always honored instead.
 */
inline constexpr Real default_starting_magnetization = 0.2;

/**
 * How many non-monotone iterations of the DFT+U occupation matrix it takes
 * before the SCF starts damping it.
 *
 * Damping perturbs the converged answer at first order in its own lag, so it
 * is switched on by evidence rather than by default and a monotonically
 * converging run must never trip it. Three tolerates the isolated backward
 * step a well-behaved run can take while the wavefunctions settle, and is
 * reached almost immediately by a genuine oscillation, which alternates every
 * iteration.
 */
inline constexpr int hubbard_damping_trigger_iterations = 3;

/**
 * How many Davidson iterations a non-self-consistent pass allows each k-point.
 *
 * A self-consistent run re-diagonalizes every iteration, so the solver's own
 * short cap only defers work to the next one. A non-self-consistent pass
 * diagonalizes once and keeps whatever a capped solve leaves unconverged:
 * copper's fourteen-band 4x4x4 capture takes 17 of the solver's 20 iterations
 * at a basis four times the band count, and ran into them at twice, with up
 * to seven bands unconverged at 48 of 140 k-point solves. The cap bounds a
 * pathological solve; a converging one exits as soon as every band meets the
 * tolerance, and one that does not is reported.
 */
inline constexpr int non_self_consistent_field_maximum_davidson_iterations = 200;

/**
 * The smallest Davidson working basis, as a multiple of the band count, that a
 * non-self-consistent pass runs with unless the deck names a factor itself.
 *
 * The pass seeds each k-point from its neighbour's vectors and keeps what the
 * solver returns. With a basis only twice the band count Davidson can settle
 * on eigenpairs that skip a state the seed lacks, and every residual test still
 * passes: copper's sixteen-band NSCF against direct diagonalization of the same
 * Hamiltonian has band 16 1.18 eV high at one k-point at a factor of 2 -- a
 * state skipped -- and more than 1 meV off at 15 of 64, against 1.2 meV at one
 * k-point at 4. A self-consistent run re-diagonalizes every iteration and
 * recovers, so the smaller default stays there; a deck that names a smaller
 * factor keeps it for the pass too, and is warned.
 */
inline constexpr int non_self_consistent_field_minimum_davidson_basis_expansion_factor = 4;

/**
 * Clamp on the Anderson extrapolation coefficient for the DFT+U occupation
 * matrix.
 *
 * The two-point secant is exact only where the residual map is locally linear.
 * Away from that it can ask for an arbitrarily long step, so the coefficient is
 * bounded: inside the bound Anderson does its work, at the bound the update is
 * still a damped step in the descent direction rather than a leap.
 */
inline constexpr Real hubbard_anderson_theta_limit = 0.9;

/**
 * Smallest |r - r_prev|^2 for which an Anderson secant step is trusted.
 *
 * Two nearly identical residuals make the secant denominator vanish and the
 * extrapolation coefficient explode; below this the step degrades to plain
 * damping, which is slower but never wild. Kept beside the other DFT+U mixing
 * knobs so tuning the occupation mixer means reading one place, not two.
 */
inline constexpr Real hubbard_anderson_secant_minimum_denominator = 1e-24;

/**
 * ─── Physical constants, SI (NIST 2018 CODATA) ─────────────────────────────
 * Identical to the values Quantum ESPRESSO uses.
 */
inline constexpr Real PLANCK_CONSTANT_SI = 6.62607015e-34;     // J·s
inline constexpr Real BOLTZMANN_CONSTANT_SI = 1.380649e-23;    // J/K
inline constexpr Real ELECTRON_CHARGE_SI = 1.602176634e-19;    // C
inline constexpr Real ELECTRONVOLT_SI = 1.602176634e-19;       // J
inline constexpr Real ELECTRON_MASS_SI = 9.1093837015e-31;     // kg
inline constexpr Real HARTREE_SI = 4.3597447222071e-18;        // J
inline constexpr Real RYDBERG_SI = HARTREE_SI / 2.0;           // J
inline constexpr Real BOHR_RADIUS_SI = 0.529177210903e-10;     // m
inline constexpr Real ATOMIC_MASS_UNIT_SI = 1.66053906660e-27; // kg
inline constexpr Real SPEED_OF_LIGHT_SI = 2.99792458e+8;       // m/s

/**
 * ─── Derived unit conversion factors ────────────────────────────────────────
 * Each is a dimensional identity over the SI values above, evaluated in
 * IEEE-754 double at compile time. The names in parentheses are the
 * corresponding Quantum ESPRESSO constants, given only so a reader can
 * cross-check a value against a published number; they are labels, not the
 * source of the definitions.
 */

/// Hartree → eV (AUTOEV)
inline constexpr Real HARTREE_TO_EV = HARTREE_SI / ELECTRONVOLT_SI;

/// Bohr radius in Ångströms (BOHR_RADIUS_SI * 1e10; BOHR_RADIUS_ANGSTROM)
inline constexpr Real BOHR_RADIUS_ANGSTROM = BOHR_RADIUS_SI * 1.0e10;

/// Ångström → Bohr (ANGSTROM_AU)
inline constexpr Real ANGSTROM_TO_BOHR = 1.0 / BOHR_RADIUS_ANGSTROM;

/// AMU → Rydberg atomic units (AMU_RY)
inline constexpr Real ATOMIC_MASS_UNIT_RY = ATOMIC_MASS_UNIT_SI / ELECTRON_MASS_SI / 2.0;

/// Rydberg → kbar (RY_KBAR): 10 * AU_GPA / 2
inline constexpr Real RYDBERG_TO_KILOBAR =
    10.0 * (HARTREE_SI / (BOHR_RADIUS_SI * BOHR_RADIUS_SI * BOHR_RADIUS_SI) / 1.0e9) / 2.0;

/// Boltzmann constant in Rydberg units (K_BOLTZMANN_RY)
inline constexpr Real BOLTZMANN_CONSTANT_RY = BOLTZMANN_CONSTANT_SI / RYDBERG_SI;

// ─── Tolerance constants (matching Quantum ESPRESSO) ────────────────────────
inline constexpr Real epsilon_1e_minus_4 = 1.0e-4;
inline constexpr Real epsilon_1e_minus_6 = 1.0e-6;
inline constexpr Real epsilon_1e_minus_8 = 1.0e-8;
inline constexpr Real epsilon_1e_minus_12 = 1.0e-12;
inline constexpr Real epsilon_1e_minus_14 = 1.0e-14;
inline constexpr Real epsilon_1e_minus_16 = 1.0e-16;

// ─── Backward-compatible aliases ────────────────────────────────────────────
inline constexpr Real RY_TO_EV = HARTREE_TO_EV / 2.0;

} // namespace constants
} // namespace q3
