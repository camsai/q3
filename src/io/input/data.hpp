#pragma once

/**
 * InputData: parsed input configuration for a Q3 calculation.
 *
 * Holds all user-specified parameters read from input files, including
 * theory selection, convergence settings, plane-wave cutoffs, k-point grid
 * dimensions, crystal structure data, and output control options.
 */

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_4_computation/interpolator/input_data/specification.hpp"

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/init.hpp"
#include "layer_4_computation/effective_potential/internal/hubbard/hubbard.hpp"

namespace q3 {

/// Parsed input data — holds raw values read from input files.
struct InputData {
    // Theory selection
    std::string theory = "kohn_sham";

    // SCF convergence
    Real convergence_threshold = 1e-6;
    int maximum_iterations = 100;

    // Kohn-Sham
    /**
     * Empty when the input names no functional. parse_input() then fills in
     * the one the pseudopotentials state, as pw.x does without input_dft; the
     * former default, "PZ", ran PBE datasets as LDA without a word.
     */
    std::string exchange_correlation_functional;
    int number_of_spin_components = 1;
    /**
     * Initial magnetization per species, parallel to `species` (QE
     * starting_magnetization(i) convention: values in [-1, 1], negative =
     * opposite sublattice for AFM). May be shorter than `species`; missing
     * entries mean 0. Empty = unset (spin-polarized runs then use a default).
     */
    std::vector<Real> starting_magnetization;
    /**
     * Initial magnetization per ATOM, parallel to `atoms` (same [-1, 1]
     * convention). Empty = unset. When set it overrides
     * `starting_magnetization` atom by atom, so atoms of one species can start
     * with different moments (a same-species antiferromagnet); such atoms are
     * also kept symmetry-inequivalent (see site_symmetry_classes_from_input).
     */
    std::vector<Real> atomic_starting_magnetization;
    std::string mixing_mode = "broyden";
    Real mixing_beta = 0.7;
    int mixing_history_size = 8;
    /**
     * Davidson working basis size as a multiple of the band count. Two arrays
     * of this many vectors dominate the solver's memory, so this is the one
     * deck key that trades restart frequency against footprint.
     */
    int davidson_basis_expansion_factor = 2;
    /**
     * Whether the deck named method.scf.davidson.basis_expansion_factor itself.
     * A non-self-consistent pass raises the default to its own floor, but keeps
     * a factor the deck asked for.
     */
    bool davidson_basis_expansion_factor_set_by_deck = false;
    std::string mixing_preconditioning = "default";
    Real mixing_q0 = 1.0;
    std::string smearing = "none";
    Real smearing_width = 0.0;
    int methfessel_paxton_order = 1;            // Hermite order N of Methfessel-Paxton smearing
    std::string dispersion_correction = "none"; // "none" or "d3bj"
    /**
     * Net cell charge in electrons removed (QE tot_charge convention):
     * positive = cationic (electrons removed), negative = anionic (electrons added).
     */
    Real total_charge = 0.0;
    // DFT+U (Dudarev). Empty `hubbard` = no Hubbard correction.
    std::vector<HubbardSpecification> hubbard;
    /**
     * Projector convention for the Hubbard manifold: "atomic" (raw pseudo-atomic UPF
     * orbitals, QE's HUBBARD {atomic} card) or "ortho-atomic" (Lowdin-orthogonalized
     * over the FULL system-wide atomic-orbital set, then the Hubbard manifold's columns
     * selected out -- QE's HUBBARD {ortho-atomic} card). A single joint diagonalization
     * over the whole system, so this is a per-run choice, not settable per manifold.
     */
    std::string hubbard_projector_type = "atomic";

    /**
     * First-iteration occupation-eigenvalue overrides (drop-in
     * `starting_ns_eigenvalue`, YAML `hubbard_starting_eigenvalues`).
     * Empty = no seeding. Indices are 0-based here; parsers convert.
     */
    std::vector<HubbardStartingEigenvalue> hubbard_starting_eigenvalues;

    /**
     * SCF iterations for which the seeded occupation matrix is held in place
     * (drop-in `mixing_fixed_ns`). 0 = seed the first iteration and let
     * self-consistency take over immediately.
     */
    int hubbard_fixed_occupation_iterations = 0;

    // Planewave
    Real wavefunction_energy_cutoff = 30.0;
    Real charge_density_energy_cutoff = 0.0;
    int number_of_bands = 0;
    std::string diagonalization_method = "davidson";

    // Pseudopotential / calculation type
    std::string calculation = "self_consistent_field";
    std::vector<std::pair<std::string, Vector3D>> band_path;
    int band_path_points_per_segment = 20;

    /**
     * Explicit k-point list for NSCF calculations (calculation == "non_self_consistent_field").
     * Each entry is (crystal-coordinate k-vector, weight).
     * Takes precedence over kpoint_grid when non-empty.
     */
    std::vector<std::pair<Vector3D, Real>> kpoints_list;

    // System
    int bravais_lattice_index = 0;
    Real cell_dimension_1 = 0.0; // lattice constant in bohr
    /**
     * The remaining cell dimensions, in the same convention as
     * `get_lattice_vectors_from_bravais_index`: b/a, c/a, and three cosines
     * whose meaning depends on the Bravais index. Zero for every lattice that
     * does not need them (all cubic ones), which is why the default is 0 rather
     * than 1.
     */
    Real cell_dimension_2 = 0.0;
    Real cell_dimension_3 = 0.0;
    Real cell_dimension_4 = 0.0;
    Real cell_dimension_5 = 0.0;
    Real cell_dimension_6 = 0.0;
    /**
     * Explicit lattice vectors (rows a1,a2,a3 in units of cell_dimension_1).
     * Set for "free" lattices (bravais_lattice_index == 0): YAML general cells
     * (hexagonal, slab, triclinic, explicit vectors) and QE CELL_PARAMETERS.
     */
    Matrix3x3 lattice_vectors{};
    bool has_explicit_lattice_vectors = false;
    int number_of_atoms = 0;
    int number_of_species = 0;

    // Species
    std::vector<AtomSpecies> species;

    // Atoms (crystal coords)
    std::vector<Atom> atoms;

    // K-points
    int kpoint_grid_dimension_1 = 1, kpoint_grid_dimension_2 = 1, kpoint_grid_dimension_3 = 1;
    int kpoint_grid_shift_1 = 0, kpoint_grid_shift_2 = 0, kpoint_grid_shift_3 = 0;

    // Pseudopotential directory
    std::string pseudopotential_directory = "./";

    /**
     * Canonical name of the registered input format this run was read from
     * ("yaml", "qe", or a name a build-time add-on registers), recorded by
     * parse_input(). Empty for an InputData assembled in code.
     *
     * The finished run is handed back to that format for the companion
     * outputs its own conventions expect (io/output/companion_outputs.hpp),
     * so those files are written only beside a run that was read from it.
     */
    std::string source_format;

    // Output control
    std::string output_directory = "./";
    std::string prefix = "q3";
    /**
     * How much the run says about itself on stdout: "default" (the physics and
     * the SCF trajectory) or "high" (that, plus the per-iteration and per-phase
     * diagnostics -- Davidson micro-iterations, timing breakdowns, mixer
     * internals -- which are logged at debug level). Quantum ESPRESSO's
     * `verbosity` maps onto this; an explicit --log-level overrides it.
     */
    std::string verbosity = "default";
    /**
     * true = use no symmetry at all: the k-point mesh is not reduced, and the density,
     * forces and stress are not symmetrized (Quantum ESPRESSO's nosym + noinv). A shifted
     * mesh then gives the raw-mesh energy rather than the symmetrized-mesh one.
     */
    bool disable_symmetry = false;
    std::string ibz_reduction_method = "internal"; // "internal" or "spglib"

    /**
     * Stress tensor options
     * "analytical" (default) = differentiate Lagrange interpolation polynomial (matches QE)
     * "numerical" = finite-difference derivatives of beta projectors and Ylm
     */
    std::string nonlocal_stress_derivative_method = "analytical";

    /**
     * Isolated (charged) system correction: "none" (default) or "makov-payne" (QE's
     * assume_isolated='makov-payne' analog). Cubic lattices only (see
     * MakovPayneCorrectionCalculator); opt-in since it's meaningless for genuinely
     * periodic bulk systems and only appropriate for a net-charged cell modeling an
     * isolated defect/molecule.
     */
    std::string isolated_system_correction = "none";

    /**
     * Mean-field export for an external many-body-perturbation-theory code:
     * "none" (default) or "berkeleygw" (writes WFN/RHO/vxc.dat -- see
     * create_mean_field_exporter()). BerkeleyGW normally consumes an NSCF run
     * with many empty (conduction) bands rather than the SCF run itself, but
     * the export is wired into both flavors unconditionally (Null object for
     * "none"), writing into output_directory.
     */
    std::string mean_field_export = "none";

    // DOS parameters (used when calculation == "density_of_states")
    Real dos_broadening = 0.02;   // Gaussian broadening in Ry
    Real dos_energy_min = -100.0; // Energy range lower bound in Ry (auto-set if 0)
    Real dos_energy_max = 100.0;  // Energy range upper bound in Ry (auto-set if 0)
    int dos_energy_steps = 2000;  // Number of energy grid points

    /**
     * Atomic-orbital (ortho-atomic/Löwdin) projections, used when
     * calculation == "non_self_consistent_field". Opt-in: adds a Löwdin
     * orthogonalization + projection pass per k-point on top of plain NSCF.
     */
    bool compute_projections = false;

    /**
     * `control.interpolation`: a build request on an NSCF run, an evaluation
     * request on calculation == "interpolation" (see the struct's header).
     */
    InterpolationSpecification interpolation;

    /**
     * Start the SCF from the charge density a previous run saved in
     * output_directory/prefix.save (pw.x startingpot = 'file'), rather than
     * from overlapping atoms. A relaxation driven step by step sets it, so
     * each geometry starts from the last one's density. A density that does
     * not fit this run's grid, or a missing one, falls back to the atoms.
     */
    bool start_from_saved_charge_density = false;

    /**
     * True when the input asked for a relaxation ('relax'/'vc-relax'), which
     * runs here as one SCF at the input's geometry -- a step of the relaxation
     * q3.workflows drives. The relaxation parameters below are then reported
     * in the results' provenance, for that driver to read.
     */
    bool is_relaxation_step = false;

    // Relaxation parameters (reported when is_relaxation_step)
    int relax_maximum_steps = 100;
    Real relax_force_threshold = 1.0e-3;  // Convergence threshold for max |force| in Ry/bohr
    Real relax_energy_threshold = 1.0e-4; // Convergence threshold for energy change in Ry

    /**
     * Parallelization
     * Number of k-point pool groups. Each pool handles a subset of k-points.
     * 0 = choose automatically from the irreducible k-point count.
     */
    int number_of_kpoint_pools = 0;

    /**
     * FFT planning determinism. false (default) = FFTW_MEASURE: each process
     * times several transform algorithms at startup and keeps the fastest,
     * which is the better-performing choice but makes runs whose SCF
     * trajectory is sensitive to Allreduce/summation-order noise (e.g. a
     * metallic system with a soft, nearly-degenerate magnetic moment under
     * hot smearing) vary slightly from run to run. true = FFTW_ESTIMATE: a
     * fixed heuristic plan with no runtime timing, giving bit-reproducible
     * repeated runs at some performance cost. See TODOs.md §2.3.
     */
    bool use_deterministic_fft_planning = false;

    /**
     * Wavefunction FFT grid size. false (default) = the 2 x ecutwfc grid: faster,
     * but approximate -- V(r)ψ(r) and |ψ(r)|² alias, which is harmless where the
     * wavefunctions have little weight near the cutoff and up to mRy off pw.x on
     * hard pseudopotentials at low cutoffs. true = the exact grid, sized for
     * 4 x ecutwfc as pw.x sizes it, at 1.4x to 2.3x the solver time on the silicon
     * and copper examples. YAML `control.use_exact_wavefunction_grid`; the
     * QE-format reader always sets it, so a pw.x deck runs on pw.x's grid.
     */
    bool use_exact_wavefunction_grid = false;

    /**
     * Store a "free" lattice (bravais_lattice_index 0) from explicit lattice
     * vectors given in bohr. Single source for both input formats that build
     * an explicit cell: the YAML general-lattice path and the QE
     * CELL_PARAMETERS bohr/angstrom card.
     *
     * @param vectors_bohr           Rows a1,a2,a3 in bohr.
     * @param preset_lattice_parameter  If > 0, used as alat directly (QE's
     *        celldm(1)-takes-precedence rule when CELL_PARAMETERS coexists
     *        with an explicit celldm). If <= 0 (the YAML case: there is no
     *        prior alat), alat is derived as |a1| instead.
     */
    void set_free_lattice_vectors(const Matrix3x3& vectors_bohr,
                                  Real preset_lattice_parameter = 0.0) {
        Real lattice_parameter =
            (preset_lattice_parameter > 0.0) ? preset_lattice_parameter : vectors_bohr[0].norm();
        if (lattice_parameter <= 0.0)
            throw std::runtime_error("Lattice vector a1 must have nonzero length");
        bravais_lattice_index = 0;
        cell_dimension_1 = lattice_parameter;
        for (int row = 0; row < 3; ++row) {
            lattice_vectors[row] = vectors_bohr[row] * (1.0 / lattice_parameter);
        }
        has_explicit_lattice_vectors = true;
    }
};

} // namespace q3
