#pragma once

/**
 * QEInputParser — Quantum ESPRESSO pw.x input file parser.
 *
 * Reads QE-style .in files with &control, &system, &electrons, &ions and &cell
 * namelists and the ATOMIC_SPECIES, ATOMIC_POSITIONS, K_POINTS, CELL_PARAMETERS
 * and HUBBARD cards.
 *
 * The policy on anything else is the one every input reader follows, for one
 * reason: a drop-in replacement that silently computes different
 * physics than the code it replaces is worse than one that refuses to run.
 *
 *   supported          mapped onto the Q3 equivalent
 *   recognized but
 *     unimplemented    HARD ERROR naming the key and the reason
 *   unknown            HARD ERROR naming the key
 *   physics-neutral    accepted and ignored, with a note in the log
 *
 * This parser ignored everything it did not recognize until 2026-08-08, which
 * is how a `HUBBARD` card came to be dropped without a word: a PAW+U input ran
 * to convergence and reported the no-U energy.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/input/parser/init.hpp"
#include "io/logger.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_1_structure/crystal/lattice.hpp"

namespace q3 {

namespace qe_input {

/**
 * Keys that steer output, restart or parallel layout and cannot move a
 * converged result. Accepted with a note.
 *
 * `nstep` sits here for the SCF path only — the relax path picks it up as
 * `relax_maximum_steps`, so it is honoured where it means something and inert
 * where it does not.
 */
inline const std::set<std::string>& physics_neutral_keys() {
    static const std::set<std::string> keys = {
        // &CONTROL: what gets written, how loudly, and for how long.
        "title",
        "iprint",
        "disk_io",
        "wf_collect",
        "max_seconds",
        "tprnfor",
        "tstress",
        "lkpoint_dir",
        "wfcdir",
        // &ELECTRONS: they steer how the SCF walks, never where it converges.
        "startingwfc",
        "diago_thr_init",
        "diago_david_ndim",
        "diago_full_acc",
        "diago_cg_maxiter",
        "adaptive_thr",
        "conv_thr_init",
        "conv_thr_multi",
    };
    return keys;
}

/**
 * Keys Q3 recognizes but does not implement, each with the reason. The reason
 * is the point: it tells the user whether to wait for a feature or change the
 * input.
 */
inline const std::map<std::string, std::string>& unimplemented_keys() {
    static const std::map<std::string, std::string> keys = {
        {"noncolin", "noncollinear magnetism is not implemented"},
        {"tqr",
         "real-space augmentation-charge treatment is an approximation with its own "
         "convergence behaviour, not a speed switch"},
        {"nr1",
         "an explicitly forced FFT grid is not implemented; Q3 sizes its grids from the "
         "cutoffs, and a grid that differs from QE's changes the result"},
        {"nr2", "see nr1"},
        {"nr3", "see nr1"},
        {"nr1s", "see nr1"},
        {"nr2s", "see nr1"},
        {"nr3s", "see nr1"},
        {"la2f", "electron-phonon coupling is not implemented"},
        {"twochem", "separate electron and hole chemical potentials are not implemented"},
        {"nelec_cond", "see twochem"},
        {"degauss_cond", "see twochem"},
        {"tbeta_smoothing",
         "smoothing the beta projectors changes the pseudopotential and is "
         "not implemented"},
        {"tq_smoothing", "see tbeta_smoothing"},
        {"xdm_a1", "see xdm"},
        {"xdm_a2", "see xdm"},
        {"ts_vdw_isolated", "see ts_vdw"},
        {"ts_vdw_econv_thr", "see ts_vdw"},
        {"london_rcut", "see london"},
        {"london_s8", "see london"},
        {"london_c6", "see london"},
        {"real_space",
         "real-space beta projection is an approximation with its own convergence "
         "behaviour, not a speed switch, so it is not silently ignored"},
        {"nosym_evc",
         "symmetrizing the k-point set without symmetrizing the calculation is not "
         "implemented; use nosym"},
        {"lspinorb", "spin-orbit coupling requires noncollinear magnetism, which Q3 does not have"},
        {"angle1", "noncollinear magnetization directions require noncollinear magnetism"},
        {"angle2", "noncollinear magnetization directions require noncollinear magnetism"},
        {"constrained_magnetization", "magnetization constraints are not implemented"},
        {"tot_magnetization",
         "fixed total magnetization is not implemented; use starting_magnetization, which sets "
         "the initial guess only"},
        {"exx_fraction", "hybrid functionals are not implemented"},
        {"ecutfock", "hybrid functionals are not implemented"},
        {"screening_parameter", "screened hybrids are not implemented"},
        {"exxdiv_treatment", "hybrid functionals are not implemented"},
        {"nqx1", "hybrid functionals are not implemented"},
        {"nqx2", "hybrid functionals are not implemented"},
        {"nqx3", "hybrid functionals are not implemented"},
        {"lda_plus_u",
         "the legacy namelist DFT+U syntax is not read; use the HUBBARD card "
         "(`HUBBARD (atomic)` / `U <species>-<shell> <eV>`), which names the shell "
         "explicitly instead of deriving it from the element"},
        {"lda_plus_u_kind", "see lda_plus_u"},
        {"hubbard_u", "see lda_plus_u"},
        {"hubbard_j0", "Q3 implements the Dudarev U only"},
        {"hubbard_alpha", "Q3 implements the Dudarev U only"},
        {"hubbard_beta", "Q3 implements the Dudarev U only"},
        {"u_projection_type", "see lda_plus_u; the HUBBARD card carries the projector flavour"},
        {"nbnd_cond",
         "conduction-band counting for two-chemical-potential runs is not implemented"},
        {"nelec", "explicit electron counts are not implemented; use tot_charge"},
        {"tefield", "sawtooth external fields are not implemented"},
        {"dipfield", "the dipole correction is not implemented"},
        {"edir", "see tefield"},
        {"eamp", "see tefield"},
        {"eopreg", "see tefield"},
        {"emaxpos", "see tefield"},
        {"lelfield", "finite electric fields are not implemented"},
        {"lberry", "Berry-phase polarization is not implemented"},
        {"gdir", "see lberry"},
        {"nppstr", "see lberry"},
        {"esm_bc", "effective screening medium boundaries are not implemented"},
        {"london", "the London dispersion correction is not mapped; Q3 implements D3(BJ)"},
        {"london_s6", "see london"},
        {"xdm", "the XDM dispersion correction is not implemented"},
        {"ts_vdw", "the Tkatchenko-Scheffler correction is not implemented"},
        {"nosym_but_time_reversal", "partial symmetry control is not implemented; use nosym"},
        {"noinv",
         "time-reversal control is not implemented separately from nosym, which disables both"},
        {"force_symmorphic", "restricting to symmorphic operations is not implemented"},
        {"use_all_frac", "fractional-translation control is not implemented"},
        {"one_atom_occupations", "per-atom occupation constraints are not implemented"},
        {"press",
         "target-pressure relaxation is a Python workflow, not an engine setting: "
         "q3.workflows.vc_relax.run_vc_relax_workflow(target_pressure_gpa=press/10). "
         "The engine itself relaxes to zero stress"},
        {"press_conv_thr", "see press"},
        {"cell_dofree", "constrained cell relaxation is not implemented"},
        {"cell_factor", "the variable-cell basis-enlargement factor is not implemented"},
        {"wmass", "cell dynamics is not implemented"},
    };
    return keys;
}

/**
 * pw.x's ionic-step settings that Q3 does not use, each with what Q3 does instead.
 *
 * A relaxation runs as one SCF per geometry, driven by q3.workflows with ASE's
 * BFGS. These steer pw.x's own optimizer and extrapolation -- the path to the
 * minimum, never the minimum itself -- so a deck carrying them is read, and the
 * log says each one is not used and why.
 */
inline const std::map<std::string, std::string>& unused_ionic_step_keys() {
    static const std::map<std::string, std::string> keys = {
        {"pot_extrapolation", "each geometry starts from the previous one's saved density"},
        {"wfc_extrapolation", "each geometry starts its wavefunctions afresh"},
        {"trust_radius_max", "ASE's BFGS limits a step with its own maxstep"},
        {"trust_radius_min", "ASE's BFGS limits a step with its own maxstep"},
        {"trust_radius_ini", "ASE's BFGS limits a step with its own maxstep"},
        {"bfgs_ndim", "ASE's BFGS keeps the full Hessian"},
        {"upscale", "each SCF converges to the input's conv_thr"},
    };
    return keys;
}

/**
 * Keys whose QE default switches the feature off, and the value that means
 * off.
 *
 * Real inputs state defaults explicitly all the time, and `noinv = .false.` is
 * a request for exactly what Q3 already does — refusing it would be a false
 * refusal. `.false.` is compared as a Fortran logical (`.F.`, `F` and `false`
 * all count); any other token is compared literally.
 */
inline const std::map<std::string, std::string>& keys_inert_at_default() {
    static const std::map<std::string, std::string> keys = {
        {"noncolin", ".false."},
        {"lspinorb", ".false."},
        {"tefield", ".false."},
        {"dipfield", ".false."},
        {"lelfield", ".false."},
        {"lberry", ".false."},
        {"london", ".false."},
        {"xdm", ".false."},
        {"ts_vdw", ".false."},
        {"real_space", ".false."},
        {"tqr", ".false."},
        {"twochem", ".false."},
        {"tbeta_smoothing", ".false."},
        {"tq_smoothing", ".false."},
        {"one_atom_occupations", ".false."},
        {"la2f", ".false."},
        {"lda_plus_u", ".false."},
        {"noinv", ".false."},
        {"nosym_evc", ".false."},
        {"force_symmorphic", ".false."},
        {"use_all_frac", ".false."},
        {"constrained_magnetization", "none"},
        {"tot_magnetization", "-10000"},
        {"esm_bc", "pbc"},
        {"pot_extrapolation", "atomic"},
        {"wfc_extrapolation", "none"},
    };
    return keys;
}

/**
 * Keys this parser reads, by namelist.
 *
 * This is the one table that has to be kept in step with the dispatch below by
 * hand. Drift is self-announcing in the safe direction: forgetting to list a
 * newly handled key makes the parser refuse an input it can actually run, which
 * fails loudly on the next test run rather than producing a wrong number.
 */
inline const std::map<std::string, std::set<std::string>>& honoured_keys_by_namelist() {
    static const std::map<std::string, std::set<std::string>> keys = {
        {"control",
         {"calculation",
          "pseudo_dir",
          "prefix",
          "outdir",
          "verbosity",
          "npool",
          "nstep",
          "etot_conv_thr",
          "forc_conv_thr",
          "restart_mode"}},
        {"system",
         {
             // Cell
             "ibrav",
             "celldm",
             "a",
             "b",
             "c",
             "cosab",
             "cosac",
             "cosbc",
             // Contents and basis
             "nat",
             "ntyp",
             "ecutwfc",
             "ecutrho",
             "nbnd",
             "tot_charge",
             // Model
             "input_dft",
             "vdw_corr",
             "dftd3_version",
             "dftd3_threebody",
             "assume_isolated",
             // Occupations and spin
             "occupations",
             "smearing",
             "degauss",
             "nspin",
             "starting_magnetization",
             "starting_ns_eigenvalue",
             "number_of_spin_components", // rejected with guidance, not read
                                          // Symmetry
             "nosym",
         }},
        {"electrons",
         {"mixing_fixed_ns",
          "conv_thr",
          "mixing_beta",
          "mixing_mode",
          "mixing_ndim",
          "electron_maxstep",
          "diagonalization",
          "scf_must_converge",
          "startingpot"}},
        {"ions", {"ion_dynamics"}},
        {"cell", {"cell_dynamics"}},
    };
    return keys;
}

} // namespace qe_input

class QEInputParser : public InputParser {
  public:
    InputData parse(const std::string& filename) const override {
        InputData data;
        /**
         * A pw.x deck runs on pw.x's wavefunction grid, sized for 4 x ecutwfc. pw.x has
         * no switch for it, so neither does this reader. See
         * InputData::use_exact_wavefunction_grid.
         */
        data.use_exact_wavefunction_grid = true;
        /**
         * pw.x mixes with mixing_mode = 'plain' unless the deck says otherwise,
         * and 'plain' is Broyden with no preconditioning. Q3's own default
         * ("default") adds Kerker whenever smearing is on, which is a different
         * mixer: on a metallic slab it damps exactly the long-wavelength charge
         * that converges slowest, and a Ni(111) slab that pw.x converges in 24
         * iterations took Q3 85. A deck that names `TF` or `local-TF` still gets
         * its preconditioner below.
         */
        data.mixing_preconditioning = "none";
        /// Raw 1-based starting_ns_eigenvalue (m, ispin, ityp, f), resolved to symbols post-parse.
        std::vector<RawStartingNsEigenvalue> raw_starting_ns_eigenvalues;
        std::ifstream input_file(filename);
        if (!input_file.is_open())
            throw std::runtime_error("Cannot open input file: " + filename);

        std::string line;
        std::string section;

        /**
         * CELL_PARAMETERS card state (applied after the whole file is read,
         * because celldm(1) may appear before or after the card). The units
         * string is empty for a card that names none; what that means depends
         * on whether a lattice parameter is given, so it is resolved then too.
         */
        Matrix3x3 cell_parameters_vectors{};
        std::string cell_parameters_units;
        int cell_parameters_row = 0;
        bool has_cell_parameters = false;

        /**
         * A/B/C/cosAB/cosAC/cosBC state, converted to celldm after the file is
         * read (which celldm each cosine becomes depends on ibrav).
         */
        Real lattice_constant_a = 0.0, lattice_constant_b = 0.0, lattice_constant_c = 0.0;
        Real cosine_ab = 0.0, cosine_ac = 0.0, cosine_bc = 0.0;
        bool has_crystallographic_constants = false;

        /**
         * ATOMIC_POSITIONS units option. The rows are stored as read and
         * converted after the whole file is read, because the cell they may be
         * Cartesian in can be given after the card.
         */
        std::string atomic_positions_units = "crystal";

        /**
         * Whether the deck states the two DFT-D3 sub-options itself. pw.x's
         * defaults for them are not what Q3 implements, so a D3 deck that
         * leaves either one out is refused once the whole file is read --
         * QE does not require them to follow vdw_corr. A stated value Q3
         * does not implement is refused where it is read.
         */
        bool has_dftd3_version = false;
        bool has_dftd3_threebody = false;

        while (std::getline(input_file, line)) {
            /**
             * Drop trailing comments before anything reads the line: a key scan
             * that sees `celldm(1) = 10.2 ! a = 5 in the old cell` would find a
             * key `a` in the prose.
             */
            line = strip_comment(line);
            std::string trimmed = trim(line);
            if (trimmed.empty())
                continue;

            std::string lower = to_lower(trimmed);

            /**
             * Namelist headers. Every `&name` is matched, not only the ones Q3
             * reads: an unrecognized header used to leave `section` pointing at
             * the previous namelist, so `&IONS`'s keys were scanned as
             * `&ELECTRONS`'s and silently found nothing.
             *
             * Fortran lets assignments share the header's line, and the closing
             * `/` too: `&ELECTRONS conv_thr = 1.0d-10, mixing_beta = 0.3 /`. The
             * rest of the line therefore goes on below as an ordinary namelist
             * line. It used to be thrown away with the header, so a deck written
             * that way ran on Q3's defaults for every key on those lines.
             */
            if (lower[0] == '&') {
                size_t name_end =
                    lower.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_", 1);
                if (name_end == std::string::npos)
                    name_end = lower.size();
                section = lower.substr(1, name_end - 1);
                // `&END` is Fortran's older namelist terminator, equivalent to `/`.
                if (section == "end") {
                    section = "";
                    continue;
                }
                if (qe_input::honoured_keys_by_namelist().count(section) == 0)
                    throw std::runtime_error(
                        "QE namelist '&" + section +
                        "' is not supported. Q3 reads &control, &system, &electrons, &ions and "
                        "&cell, and refuses the rest rather than ignoring the physics in them.");
                line = trimmed.substr(name_end);
                trimmed = trim(line);
                if (trimmed.empty())
                    continue;
                lower = to_lower(trimmed);
            }

            /**
             * The `/` that closes a namelist need not stand alone either: it may
             * follow the line's last assignment (`ecutwfc = 30.0 /`). Left in
             * place, it became part of that value -- `30.0 /`, or `'relax' /`
             * for a string key. It is cut off here, and the namelist closes at
             * the bottom of the loop, once the assignments before it are read.
             */
            bool line_closes_namelist = false;
            if (qe_input::honoured_keys_by_namelist().count(section) != 0) {
                line_closes_namelist = strip_namelist_terminator(line, section);
                trimmed = trim(line);
                if (trimmed.empty()) {
                    section = "";
                    continue;
                }
                lower = to_lower(trimmed);
            }
            if (lower[0] == '/') {
                section = "";
                continue;
            }

            // Detect card sections.
            if (lower.find("atomic_species") != std::string::npos) {
                section = "atomic_species";
                continue;
            }
            if (lower.find("atomic_positions") != std::string::npos) {
                section = "atomic_positions";
                atomic_positions_units = read_atomic_positions_units(lower);
                continue;
            }
            if (lower.find("k_points") != std::string::npos) {
                section = "k_points";
                continue;
            }
            /**
             * QE's remaining cards.
             *
             * They must be *recognized* even when unsupported: an unrecognized
             * card header leaves `section` pointing at whatever came before,
             * and the card's body is then parsed as that section's data. A
             * HUBBARD card following K_POINTS was silently dropped — a PAW+U
             * input ran and returned the no-U energy.
             */
            /**
             * A card header never contains '='. Without that guard the
             * `occupations` *key* in &SYSTEM matches the `OCCUPATIONS` *card*
             * and a perfectly ordinary input is refused.
             */
            bool looks_like_card = lower.find('=') == std::string::npos;

            if (looks_like_card && lower.rfind("hubbard", 0) == 0) {
                section = "hubbard";
                /**
                 * `HUBBARD (atomic)` / `(ortho-atomic)`. QE's other projector
                 * flavours (`wf`, `pseudo`, `norm-atomic`) have no Q3
                 * equivalent and are refused rather than silently substituted.
                 */
                if (lower.find("ortho-atomic") != std::string::npos)
                    data.hubbard_projector_type = "ortho-atomic";
                else if (lower.find("atomic") != std::string::npos)
                    data.hubbard_projector_type = "atomic";
                else
                    throw std::runtime_error(
                        "QE HUBBARD card projector type in '" + trimmed +
                        "' is not supported; Q3 implements (atomic) and (ortho-atomic)");
                continue;
            }
            for (const char* unsupported_card : {"occupations",
                                                 "constraints",
                                                 "atomic_forces",
                                                 "atomic_velocities",
                                                 "additional_k_points",
                                                 "solvents",
                                                 "hubbard_v",
                                                 "total_charge"}) {
                if (looks_like_card && lower.rfind(unsupported_card, 0) == 0)
                    throw std::runtime_error(
                        std::string("QE card '") + trimmed +
                        "' is not supported. Q3 refuses cards it cannot honour rather than "
                        "ignoring them, because an ignored card produces a converged result "
                        "for a different calculation than the one requested.");
            }

            if (lower.find("cell_parameters") != std::string::npos) {
                section = "cell_parameters";
                cell_parameters_row = 0;
                cell_parameters_units = read_cell_parameters_units(lower);
                continue;
            }

            std::string value;

            /**
             * Classify every key on the line before any of it is applied, so a
             * refusal names the offending key rather than failing later on a
             * missing value.
             */
            if (qe_input::honoured_keys_by_namelist().count(section) != 0)
                classify_namelist_keys(trimmed, section);

            if (section == "control") {
                if (extract_key_value(line, "pseudo_dir", value))
                    data.pseudopotential_directory = value;
                if (extract_key_value(line, "calculation", value)) {
                    /**
                     * QE's own vocabulary, mapped to Q3's flavor names. Passing
                     * it through verbatim meant a real QE input file failed on
                     * its very first field with "unsupported calculation type
                     * 'scf'" — the parser could read the file but not run it.
                     */
                    std::string requested = strip_quotes(value);
                    if (requested == "scf")
                        data.calculation = "self_consistent_field";
                    else if (requested == "nscf" || requested == "bands")
                        data.calculation = "non_self_consistent_field";
                    else if (requested == "relax")
                        data.calculation = "relaxation";
                    else if (requested == "vc-relax")
                        data.calculation = "variable_cell_relaxation";
                    else
                        data.calculation = requested;
                }
                if (extract_key_value(line, "prefix", value))
                    data.prefix = value;
                if (extract_key_value(line, "outdir", value))
                    data.output_directory = value;
                /**
                 * QE's six spellings collapse onto Q3's two, along the same
                 * line QE itself draws: 'high' and 'debug' raise iverbosity
                 * above zero, the other four leave it there. Read rather than
                 * ignored, so a deck that asked pw.x for the detailed output
                 * gets the detailed output here too.
                 */
                if (extract_key_value(line, "verbosity", value)) {
                    std::string requested = to_lower(strip_quotes(value));
                    if (requested == "high" || requested == "debug")
                        data.verbosity = "high";
                    else if (requested == "low" || requested == "medium" ||
                             requested == "minimal" || requested == "default")
                        data.verbosity = "default";
                    else
                        throw std::runtime_error(
                            "verbosity='" + requested +
                            "' is not a Quantum ESPRESSO verbosity; use 'high', 'debug', "
                            "'default', 'low', 'medium' or 'minimal'");
                }
                /**
                 * QE-vocabulary exception: 'npool' mirrors QE's own flag name so
                 * QE-style inputs stay portable. QE itself only accepts -npool on
                 * the command line; accepting it in &control is a Q3 extension.
                 * Everywhere else use the full name: number_of_kpoint_pools.
                 */
                if (extract_key_value(line, "npool", value))
                    data.number_of_kpoint_pools = std::stoi(value);
                /**
                 * QE's default, stated explicitly in most real inputs. Only
                 * 'from_scratch' means anything here — Q3 cannot resume a QE
                 * run's saved state.
                 */
                if (extract_key_value(line, "restart_mode", value)) {
                    std::string requested = to_lower(strip_quotes(value));
                    if (requested != "from_scratch")
                        throw std::runtime_error(
                            "restart_mode='" + requested +
                            "' is not supported; Q3 cannot resume a QE run's saved state");
                }
                /**
                 * Relaxation controls. They are inert for `calculation='scf'`,
                 * so they are applied unconditionally rather than deferred
                 * until the calculation type is known — which QE does not
                 * require to appear first.
                 */
                if (extract_key_value(line, "nstep", value))
                    data.relax_maximum_steps = std::stoi(value);
                if (extract_key_value(line, "forc_conv_thr", value))
                    data.relax_force_threshold = fortran_to_double(value);
                if (extract_key_value(line, "etot_conv_thr", value))
                    data.relax_energy_threshold = fortran_to_double(value);
            } else if (section == "system") {
                if (extract_key_value(line, "ibrav", value))
                    data.bravais_lattice_index = std::stoi(value);
                /**
                 * celldm(2..6) carry the axial ratios and cell angles. Dropping
                 * them left every non-cubic Bravais index — hexagonal,
                 * tetragonal, orthorhombic, monoclinic, triclinic — being built
                 * with c/a = 0, i.e. a collapsed cell.
                 */
                for (int index = 1; index <= 6; index++) {
                    if (!extract_key_value(line, "celldm(" + std::to_string(index) + ")", value))
                        continue;
                    Real dimension = fortran_to_double(value);
                    switch (index) {
                        case 1:
                            data.cell_dimension_1 = dimension;
                            break;
                        case 2:
                            data.cell_dimension_2 = dimension;
                            break;
                        case 3:
                            data.cell_dimension_3 = dimension;
                            break;
                        case 4:
                            data.cell_dimension_4 = dimension;
                            break;
                        case 5:
                            data.cell_dimension_5 = dimension;
                            break;
                        default:
                            data.cell_dimension_6 = dimension;
                            break;
                    }
                }
                /**
                 * QE's alternative crystallographic form. Recorded raw and
                 * converted after the file is read, because the celldm(4..6)
                 * each cosine maps onto depends on ibrav, which may appear
                 * later in the namelist.
                 */
                for (const auto& [key, target] :
                     {std::pair<const char*, Real*>{"A", &lattice_constant_a},
                      {"B", &lattice_constant_b},
                      {"C", &lattice_constant_c},
                      {"cosAB", &cosine_ab},
                      {"cosAC", &cosine_ac},
                      {"cosBC", &cosine_bc}}) {
                    if (extract_key_value(line, key, value)) {
                        *target = fortran_to_double(value);
                        has_crystallographic_constants = true;
                    }
                }
                if (extract_key_value(line, "nat", value))
                    data.number_of_atoms = std::stoi(value);
                if (extract_key_value(line, "ntyp", value))
                    data.number_of_species = std::stoi(value);
                if (extract_key_value(line, "ecutwfc", value))
                    data.wavefunction_energy_cutoff = fortran_to_double(value);
                if (extract_key_value(line, "ecutrho", value))
                    data.charge_density_energy_cutoff = fortran_to_double(value);
                if (extract_key_value(line, "nbnd", value))
                    data.number_of_bands = std::stoi(value);
                if (extract_key_value(line, "input_dft", value))
                    data.exchange_correlation_functional = strip_quotes(value);
                if (extract_key_value(line, "nspin", value)) {
                    data.number_of_spin_components = std::stoi(value);
                    if (data.number_of_spin_components != 1 && data.number_of_spin_components != 2)
                        throw std::runtime_error(
                            "nspin must be 1 (unpolarized) or 2 (collinear LSDA), got " + value +
                            " (noncollinear nspin=4 is not supported)");
                }
                /**
                 * YAML-manifest vocabulary exception: mirrors nspin's own guard above
                 * (the YAML parser rejects 'nspin' with a message pointing back here) --
                 * reject the manifest key here so a mistyped namelist input cannot
                 * silently fall through and run unpolarized.
                 */
                if (extract_key_value(line, "number_of_spin_components", value))
                    throw std::runtime_error(
                        "'number_of_spin_components' is not a QE namelist key; use nspin instead");
                if (extract_key_value(line, "starting_magnetization", value)) {
                    /**
                     * Per-species: starting_magnetization(i) = m. An unindexed
                     * occurrence means species 1 (Fortran scalar-assignment rule).
                     */
                    int species_index = 1;
                    size_t key_position = to_lower(line).find("starting_magnetization");
                    size_t parenthesis_position = line.find('(', key_position);
                    size_t equals_position = line.find('=', key_position);
                    if (parenthesis_position != std::string::npos &&
                        equals_position != std::string::npos &&
                        parenthesis_position < equals_position) {
                        try {
                            species_index = std::stoi(line.substr(parenthesis_position + 1));
                        } catch (const std::exception&) {
                            throw std::runtime_error(
                                "Cannot parse starting_magnetization species index in line: " +
                                trim(line));
                        }
                        if (species_index < 1)
                            throw std::runtime_error(
                                "starting_magnetization index must be >= 1, got " +
                                std::to_string(species_index));
                    }
                    if (static_cast<int>(data.starting_magnetization.size()) < species_index)
                        data.starting_magnetization.resize(species_index, 0.0);
                    data.starting_magnetization[species_index - 1] = fortran_to_double(value);
                }
                /**
                 * starting_ns_eigenvalue(m,ispin,ityp) = f.
                 *
                 * Scanned for *every* occurrence on the line rather than via
                 * `extract_key_value`, which returns only the first match: a
                 * seeding deck naturally packs several directives onto one
                 * namelist line, and silently keeping just one of them would
                 * steer the SCF into a minimum the user did not ask for --
                 * the precise failure seeding exists to rule out.
                 *
                 * All three indices are required. The Fortran scalar
                 * shorthand would leave which eigenvalue of which species to
                 * guesswork, and a seed aimed at the wrong manifold selects a
                 * different solution, not a slightly-off number. Species
                 * resolution to a symbol happens post-parse, since ntyp and
                 * ATOMIC_SPECIES may follow this line.
                 */
                std::string lower_line = to_lower(line);
                const std::string seeding_key = "starting_ns_eigenvalue";
                for (size_t key_position = lower_line.find(seeding_key);
                     key_position != std::string::npos;
                     key_position = lower_line.find(seeding_key, key_position + 1)) {
                    if (!is_whole_key_at(lower_line, key_position, seeding_key.size()))
                        continue;
                    size_t parenthesis_position = line.find('(', key_position);
                    size_t closing_position = line.find(')', key_position);
                    size_t equals_position = line.find('=', key_position);
                    if (parenthesis_position == std::string::npos ||
                        closing_position == std::string::npos ||
                        equals_position == std::string::npos ||
                        equals_position < parenthesis_position)
                        throw std::runtime_error(
                            "starting_ns_eigenvalue needs explicit (m,ispin,ityp) indices, got: " +
                            trim(line));
                    std::string index_list = line.substr(
                        parenthesis_position + 1, closing_position - parenthesis_position - 1);
                    std::vector<int> indices;
                    std::stringstream index_stream(index_list);
                    std::string index_token;
                    while (std::getline(index_stream, index_token, ','))
                        try {
                            indices.push_back(std::stoi(trim(index_token)));
                        } catch (const std::exception&) {
                            throw std::runtime_error(
                                "Cannot parse starting_ns_eigenvalue indices in line: " +
                                trim(line));
                        }
                    if (indices.size() != 3)
                        throw std::runtime_error(
                            "starting_ns_eigenvalue needs exactly (m,ispin,ityp), got " +
                            std::to_string(indices.size()) + " indices in: " + trim(line));

                    // This directive's value ends at the next top-level comma or comment.
                    size_t value_start = equals_position + 1;
                    while (value_start < line.size() &&
                           (line[value_start] == ' ' || line[value_start] == '\t'))
                        value_start++;
                    size_t value_end = line.find_first_of(",!\n\r", value_start);
                    if (value_end == std::string::npos)
                        value_end = line.size();
                    Real eigenvalue =
                        fortran_to_double(trim(line.substr(value_start, value_end - value_start)));

                    if (indices[0] < 1 || indices[1] < 1 || indices[1] > 2 || indices[2] < 1)
                        throw std::runtime_error(
                            "starting_ns_eigenvalue indices must be m >= 1, ispin in {1,2}, "
                            "ityp >= 1, got: " +
                            trim(line));
                    if (eigenvalue < 0.0 || eigenvalue > 1.0)
                        throw std::runtime_error(
                            "starting_ns_eigenvalue value must be in [0, 1], got " +
                            std::to_string(eigenvalue));
                    raw_starting_ns_eigenvalues.push_back(
                        {indices[0], indices[1], indices[2], eigenvalue});
                }
                if (extract_key_value(line, "degauss", value))
                    data.smearing_width = fortran_to_double(value);
                if (extract_key_value(line, "tot_charge", value))
                    data.total_charge = fortran_to_double(value);
                if (extract_key_value(line, "smearing", value))
                    data.smearing = strip_quotes(value);
                if (extract_key_value(line, "occupations", value)) {
                    std::string occupations = strip_quotes(value);
                    if (to_lower(occupations) == "smearing" && data.smearing == "none") {
                        data.smearing = "gaussian";
                    }
                }
                if (extract_key_value(line, "vdw_corr", value)) {
                    std::string vdw_corr = to_lower(strip_quotes(value));
                    if (vdw_corr == "dft-d3" || vdw_corr == "grimme-d3") {
                        /**
                         * Only the Becke-Johnson-damped two-body term is
                         * implemented -- what D3BJDispersionCalculator computes,
                         * and what pw.x runs for dftd3_version = 4,
                         * dftd3_threebody = .false.. Neither is pw.x's default,
                         * so the deck has to state both; see
                         * refuse_dftd3_defaults_q3_does_not_implement().
                         */
                        data.dispersion_correction = "d3bj";
                    } else if (vdw_corr == "none" || vdw_corr.empty()) {
                        data.dispersion_correction = "none";
                    } else {
                        /**
                         * pw.x's DFT-D2, Tkatchenko-Scheffler, MBD and XDM.
                         * These used to fall through silently and run with no
                         * dispersion correction at all.
                         */
                        throw std::runtime_error(
                            "vdw_corr = '" + vdw_corr +
                            "' is not supported; the only dispersion correction Q3 implements "
                            "is DFT-D3 with Becke-Johnson damping ('dft-d3' or 'grimme-d3')");
                    }
                }
                /**
                 * The two D3 sub-options Q3's implementation pins. They used to
                 * be ignored, which meant `vdw_corr='dft-d3'` silently produced
                 * the BJ-damped two-body energy whatever the file asked for.
                 */
                if (extract_key_value(line, "dftd3_version", value)) {
                    if (std::stoi(value) != 4)
                        throw std::runtime_error(
                            "dftd3_version = " + value +
                            " is not supported; Q3 implements version 4 (Becke-Johnson damping)");
                    has_dftd3_version = true;
                }
                if (extract_key_value(line, "dftd3_threebody", value)) {
                    if (fortran_logical_is_true(value))
                        throw std::runtime_error(
                            "dftd3_threebody = .true. is not supported; Q3 implements the "
                            "two-body D3(BJ) term only");
                    has_dftd3_threebody = true;
                }
                if (extract_key_value(line, "nosym", value))
                    data.disable_symmetry = fortran_logical_is_true(value);
                if (extract_key_value(line, "assume_isolated", value)) {
                    /**
                     * QE's spelling of the corrections Q3 has; its remaining
                     * options ('esm', '2D') are different physics, not aliases.
                     */
                    std::string requested = to_lower(strip_quotes(value));
                    if (requested == "makov-payne" || requested == "m-p" || requested == "mp")
                        data.isolated_system_correction = "makov-payne";
                    else if (requested == "martyna-tuckerman" || requested == "m-t" ||
                             requested == "mt")
                        data.isolated_system_correction = "martyna-tuckerman";
                    else if (requested == "none")
                        data.isolated_system_correction = "none";
                    else
                        throw std::runtime_error(
                            "assume_isolated='" + requested +
                            "' is not supported; Q3 implements 'makov-payne' and "
                            "'martyna-tuckerman'");
                }
            } else if (section == "ions") {
                if (extract_key_value(line, "ion_dynamics", value)) {
                    /**
                     * Q3 relaxes with ASE's BFGS. QE's other ion_dynamics
                     * values are molecular dynamics or damped dynamics, which
                     * follow a different trajectory and stop somewhere else.
                     */
                    std::string requested = to_lower(strip_quotes(value));
                    if (requested != "bfgs")
                        throw std::runtime_error(
                            "ion_dynamics='" + requested +
                            "' is not supported; Q3's relaxation is BFGS, and the damped-dynamics "
                            "and molecular-dynamics paths are not implemented");
                }
            } else if (section == "cell") {
                if (extract_key_value(line, "cell_dynamics", value)) {
                    std::string requested = to_lower(strip_quotes(value));
                    if (requested != "bfgs")
                        throw std::runtime_error(
                            "cell_dynamics='" + requested +
                            "' is not supported; Q3's variable-cell relaxation is BFGS on an ASE "
                            "cell filter");
                }
            } else if (section == "electrons") {
                if (extract_key_value(line, "conv_thr", value))
                    data.convergence_threshold = fortran_to_double(value);
                if (extract_key_value(line, "startingpot", value)) {
                    /// pw.x's two starts: overlapping atoms, or the saved density.
                    static const std::map<std::string, bool> starts_from_saved_density = {
                        {"atomic", false}, {"file", true}};
                    const std::string start = to_lower(strip_quotes(value));
                    const auto found = starts_from_saved_density.find(start);
                    if (found == starts_from_saved_density.end())
                        throw std::runtime_error("startingpot='" + start +
                                                 "' is not supported; use 'atomic' or 'file'");
                    data.start_from_saved_charge_density = found->second;
                }
                if (extract_key_value(line, "mixing_beta", value))
                    data.mixing_beta = fortran_to_double(value);
                if (extract_key_value(line, "mixing_mode", value)) {
                    /**
                     * QE's vocabulary, translated the way `calculation` is.
                     * `plain` is QE's *default* and appears in most real
                     * inputs; storing it verbatim meant Q3 died in the mixer
                     * factory with "Unknown mixer mode: 'plain'" on files QE
                     * runs without comment. It keeps the unpreconditioned
                     * default set at the top of parse().
                     *
                     * `TF`/`local-TF` are Thomas-Fermi preconditioners on the
                     * same Broyden mixing, and Q3 now has both: `TF` is the
                     * average (global) Thomas-Fermi screening, which is exactly
                     * what Kerker is, and `local-TF` builds the screening from
                     * the local density. Both map onto Broyden with the matching
                     * preconditioner rather than onto a different fixed point.
                     */
                    std::string requested = to_lower(strip_quotes(value));
                    if (requested == "plain") {
                        data.mixing_mode = "broyden";
                    } else if (requested == "tf") {
                        log::info("     QE input requests mixing_mode='{}'; using Broyden with "
                                  "Kerker preconditioning, which is that same average "
                                  "Thomas-Fermi screening",
                                  strip_quotes(value));
                        data.mixing_mode = "broyden";
                        data.mixing_preconditioning = "kerker";
                    } else if (requested == "local-tf") {
                        data.mixing_mode = "broyden";
                        data.mixing_preconditioning = "local-tf";
                    } else {
                        data.mixing_mode = requested;
                    }
                }
                if (extract_key_value(line, "mixing_ndim", value))
                    data.mixing_history_size = std::stoi(value);
                if (extract_key_value(line, "electron_maxstep", value))
                    data.maximum_iterations = std::stoi(value);
                /**
                 * QE's default is .true.; .false. asks it to carry on with an
                 * unconverged density, which Q3 has no equivalent for and must
                 * not silently upgrade to a converged run.
                 */
                if (extract_key_value(line, "scf_must_converge", value) &&
                    !fortran_logical_is_true(value))
                    throw std::runtime_error(
                        "scf_must_converge = .false. is not supported; Q3 always requires the "
                        "SCF to converge before reporting a result");
                if (extract_key_value(line, "mixing_fixed_ns", value)) {
                    /**
                     * Parsed through the same tolerant numeric path as every
                     * other value in this reader: a bare std::stoi throws
                     * std::invalid_argument without naming the key, which is
                     * the one thing this parser exists not to do.
                     */
                    Real fixed_iterations = fortran_to_double(value);
                    if (fixed_iterations < 0.0 || fixed_iterations != std::floor(fixed_iterations))
                        throw std::runtime_error(
                            "mixing_fixed_ns must be a non-negative whole number, got '" + value +
                            "'");
                    data.hubbard_fixed_occupation_iterations = static_cast<int>(fixed_iterations);
                }
                if (extract_key_value(line, "diagonalization", value)) {
                    /**
                     * QE's iterative solvers select an algorithm, not physics: both
                     * converge to the same fixed point. `cg` is mapped to Davidson with
                     * a notice rather than refused, because every QE PAW reference has
                     * to be generated with `cg` (its Davidson aborts in `cegterg` on PAW
                     * datasets), and refusing it would make exactly those inputs
                     * unreadable.
                     */
                    std::string requested = strip_quotes(value);
                    if (requested == "cg" || requested == "ppcg" || requested == "paro") {
                        log::info("     QE input requests diagonalization='{}'; using Davidson "
                                  "(same fixed point, different iteration path)",
                                  requested);
                        data.diagonalization_method = "davidson";
                    } else {
                        data.diagonalization_method = requested;
                    }
                }
            } else if (section == "atomic_species") {
                std::istringstream stream(trimmed);
                AtomSpecies species;
                stream >> species.symbol;
                if (!species.symbol.empty()) {
                    extract_real(stream, species.mass, trimmed, "ATOMIC_SPECIES");
                    stream >> species.pseudopotential_file;
                    data.species.push_back(species);
                }
            } else if (section == "atomic_positions") {
                std::istringstream stream(trimmed);
                std::string symbol;
                Real x_coordinate = 0.0, y_coordinate = 0.0, z_coordinate = 0.0;
                stream >> symbol;
                if (!symbol.empty()) {
                    extract_real(stream, x_coordinate, trimmed, "ATOMIC_POSITIONS");
                    extract_real(stream, y_coordinate, trimmed, "ATOMIC_POSITIONS");
                    extract_real(stream, z_coordinate, trimmed, "ATOMIC_POSITIONS");
                    const std::array<bool, 3> movable_components =
                        read_position_flags(stream, trimmed);
                    int species_index = -1;
                    for (int index = 0; index < static_cast<int>(data.species.size()); index++) {
                        if (data.species[index].symbol == symbol) {
                            species_index = index;
                            break;
                        }
                    }
                    if (species_index >= 0) {
                        Atom atom;
                        atom.species_index = species_index;
                        atom.position = {x_coordinate, y_coordinate, z_coordinate};
                        atom.movable_components = movable_components;
                        data.atoms.push_back(atom);
                    }
                }
            } else if (section == "k_points") {
                std::istringstream stream(trimmed);
                stream >> data.kpoint_grid_dimension_1 >> data.kpoint_grid_dimension_2 >>
                    data.kpoint_grid_dimension_3 >> data.kpoint_grid_shift_1 >>
                    data.kpoint_grid_shift_2 >> data.kpoint_grid_shift_3;
                section = "";
            } else if (section == "hubbard") {
                /**
                 * `U  <species>-<shell>  <value in eV>`, e.g. `U Ni-3d 4.0`.
                 * Only the U term: QE's `J`, `V` and `B` parameters describe
                 * interactions Q3's Dudarev implementation does not have, so
                 * they are refused rather than dropped.
                 */
                std::istringstream card_stream(trimmed);
                std::string parameter, target;
                Real value = 0.0;
                card_stream >> parameter >> target;
                std::string parameter_lower = to_lower(parameter);
                if (parameter_lower != "u")
                    throw std::runtime_error(
                        "QE HUBBARD parameter '" + parameter +
                        "' is not supported; Q3 implements the Dudarev U only");
                auto separator = target.find('-');
                if (separator == std::string::npos)
                    throw std::runtime_error("QE HUBBARD entry '" + trimmed +
                                             "' is not of the form U <species>-<shell> <eV>");
                /**
                 * Read the value only once the line is known to be a `U` entry:
                 * QE's `V` takes two targets, so extracting a number here first
                 * would report the second target as "not a number" and bury the
                 * accurate refusal above.
                 */
                extract_real(card_stream, value, trimmed, "HUBBARD");
                data.hubbard.push_back({target.substr(0, separator),
                                        target.substr(separator + 1),
                                        value * 2.0 / constants::HARTREE_TO_EV});
            } else if (section == "cell_parameters") {
                if (cell_parameters_row < 3) {
                    std::istringstream stream(trimmed);
                    for (int column = 0; column < 3; column++) {
                        extract_real(stream,
                                     cell_parameters_vectors[cell_parameters_row][column],
                                     trimmed,
                                     "CELL_PARAMETERS");
                    }
                    cell_parameters_row++;
                    if (cell_parameters_row == 3) {
                        has_cell_parameters = true;
                        section = "";
                    }
                }
            }

            if (line_closes_namelist)
                section = "";
        }

        if (data.dispersion_correction == "d3bj")
            refuse_dftd3_defaults_q3_does_not_implement(has_dftd3_version, has_dftd3_threebody);

        /**
         * A/B/C are resolved before CELL_PARAMETERS, because the alat that an
         * `(alat)` card is expressed in may have been given in that form.
         */
        if (has_crystallographic_constants) {
            apply_crystallographic_constants(data,
                                             lattice_constant_a,
                                             lattice_constant_b,
                                             lattice_constant_c,
                                             cosine_ab,
                                             cosine_ac,
                                             cosine_bc);
        }

        if (has_cell_parameters) {
            apply_cell_parameters(data, cell_parameters_vectors, cell_parameters_units);
        }

        // Positions last: converting Cartesian rows needs the finished cell.
        convert_atomic_positions_to_crystal(data, atomic_positions_units);

        /**
         * starting_magnetization(i) indices were only bounds-checked to be >= 1
         * while streaming (ntyp may appear after them in the namelist); validate
         * against the species count now that the whole file is read.
         */
        if (static_cast<int>(data.starting_magnetization.size()) > data.number_of_species &&
            data.number_of_species > 0) {
            throw std::runtime_error("starting_magnetization index " +
                                     std::to_string(data.starting_magnetization.size()) +
                                     " exceeds the number of species (ntyp = " +
                                     std::to_string(data.number_of_species) + ")");
        }

        /**
         * starting_ns_eigenvalue species indices, resolvable only now that
         * ATOMIC_SPECIES is read: convert 1-based (m, ispin, ityp) to the
         * symbol-keyed, 0-based form the SCF consumes. ityp beyond ntyp is a
         * hard error like any other out-of-range species reference.
         */
        for (const auto& raw_directive : raw_starting_ns_eigenvalues) {
            if (raw_directive.species_index > static_cast<int>(data.species.size()))
                throw std::runtime_error("starting_ns_eigenvalue species index " +
                                         std::to_string(raw_directive.species_index) +
                                         " exceeds the number of species (" +
                                         std::to_string(data.species.size()) + ")");
            data.hubbard_starting_eigenvalues.push_back(make_hubbard_starting_eigenvalue(
                data.species[static_cast<size_t>(raw_directive.species_index - 1)].symbol,
                raw_directive.eigenvalue_index,
                raw_directive.spin_index,
                raw_directive.value,
                "starting_ns_eigenvalue"));
        }

        return data;
    }

  private:
    /// Raw 1-based starting_ns_eigenvalue tuple; symbol resolution happens post-parse.
    struct RawStartingNsEigenvalue {
        int eigenvalue_index;
        int spin_index;
        int species_index;
        Real value;
    };

    /**
     * Convert a CELL_PARAMETERS card into InputData explicit lattice vectors
     * (bravais_lattice_index 0, rows in units of alat).
     *
     * Units handling matches pw.x: 'alat' rows are already in units of celldm(1)
     * (which must be present); 'bohr'/'angstrom' rows are absolute, and alat is
     * taken from celldm(1) when given, |a1| otherwise. A card that names no
     * units -- `units_named` empty -- is alat when celldm(1) or A is given and
     * bohr when neither is, which is how pw.x reads that deprecated spelling
     * (checked black-box against pw.x 7.6). It used to be read as bohr
     * regardless, so a deck that set celldm(1) to one angstrom in bohr and
     * wrote its rows in angstrom ran a cell 1.89 times too small on every axis.
     */
    static void apply_cell_parameters(InputData& data,
                                      const Matrix3x3& raw_vectors,
                                      const std::string& units_named) {
        if (data.bravais_lattice_index != 0)
            throw std::runtime_error("CELL_PARAMETERS requires ibrav = 0, got ibrav = " +
                                     std::to_string(data.bravais_lattice_index));

        std::string units = units_named;
        if (units.empty()) {
            bool has_lattice_parameter = data.cell_dimension_1 > 0.0;
            units = has_lattice_parameter ? "alat" : "bohr";
            log::info("     QE input: CELL_PARAMETERS names no units; reading its rows as {} "
                      "because {} lattice parameter is given, as pw.x does with this deprecated "
                      "spelling",
                      units,
                      has_lattice_parameter ? "a" : "no");
        }

        if (units == "alat") {
            if (data.cell_dimension_1 <= 0.0)
                throw std::runtime_error("CELL_PARAMETERS (alat) requires celldm(1) to be set");
            data.lattice_vectors = raw_vectors;
            data.has_explicit_lattice_vectors = true;
            return;
        }

        Real length_to_bohr = (units == "angstrom") ? constants::ANGSTROM_TO_BOHR : 1.0;
        Matrix3x3 vectors_bohr;
        for (int row = 0; row < 3; ++row) {
            vectors_bohr[row] = raw_vectors[row] * length_to_bohr;
        }

        /**
         * celldm(1) takes precedence when set (QE convention); InputData::
         * set_free_lattice_vectors derives alat = |a1| otherwise — the same
         * rule the YAML general-lattice path uses, single-sourced.
         */
        try {
            data.set_free_lattice_vectors(vectors_bohr, data.cell_dimension_1);
        } catch (const std::runtime_error&) {
            throw std::runtime_error("CELL_PARAMETERS: lattice vector a1 has zero length");
        }
    }

    /**
     * The units option on a CELL_PARAMETERS header line: `alat`, `bohr` or `angstrom`,
     * written bare or in parentheses or braces, or empty for a card that names none.
     * Which of the three an empty option means depends on whether the deck gives a
     * lattice parameter, so apply_cell_parameters resolves it once the whole file is
     * read. An option that is none of the three is refused by name rather than read as
     * one of them.
     */
    static std::string read_cell_parameters_units(const std::string& lower_header) {
        const std::string card_name = "cell_parameters";
        std::string option;
        for (char character : lower_header.substr(lower_header.find(card_name) + card_name.size()))
            if (std::string("(){}").find(character) == std::string::npos &&
                !std::isspace(static_cast<unsigned char>(character)))
                option += character;

        if (option.empty() || option == "alat" || option == "bohr" || option == "angstrom")
            return option;
        throw std::runtime_error("QE CELL_PARAMETERS option '" + option +
                                 "' is not recognized; the card takes alat, bohr or angstrom");
    }

    /**
     * The units option on an ATOMIC_POSITIONS header line.
     *
     * pw.x's input documentation lists five -- `crystal`, `alat`, `bohr`, `angstrom` and
     * `crystal_sg` -- and a card that names none is read as `alat`. Decks write the option
     * bare or in parentheses or braces. QE deprecates the option-less spelling but still
     * accepts it, so it is honoured here with a note rather than refused.
     *
     * The option used to be thrown away and every card read as crystal coordinates. The
     * angstrom deck in camsai/q3#18 therefore ran a structure with two O-O pairs 1.08 Å
     * apart, and its total energy missed pw.x's by 2.4 Ry.
     */
    static std::string read_atomic_positions_units(const std::string& lower_header) {
        const std::string card_name = "atomic_positions";
        std::string option;
        for (char character : lower_header.substr(lower_header.find(card_name) + card_name.size()))
            if (std::string("(){}").find(character) == std::string::npos &&
                !std::isspace(static_cast<unsigned char>(character)))
                option += character;

        if (option.empty()) {
            log::info("     QE input: ATOMIC_POSITIONS names no units; reading its rows as alat, "
                      "as pw.x does with this deprecated spelling");
            return "alat";
        }
        if (option == "crystal" || option == "alat" || option == "bohr" || option == "angstrom")
            return option;
        if (option == "crystal_sg")
            throw std::runtime_error(
                "QE ATOMIC_POSITIONS crystal_sg is not supported: it lists only the "
                "symmetry-inequivalent atoms and relies on space_group to generate the rest, "
                "which Q3 does not do. List every atom under ATOMIC_POSITIONS crystal instead.");
        throw std::runtime_error("QE ATOMIC_POSITIONS option '" + option +
                                 "' is not recognized; the card takes crystal, alat, bohr, "
                                 "angstrom or crystal_sg");
    }

    /**
     * Rewrite ATOMIC_POSITIONS rows as the crystal coordinates InputData stores.
     *
     * `alat`, `bohr` and `angstrom` rows are Cartesian: in units of the lattice parameter,
     * in bohr, and in angstrom. Each row r becomes crystal coordinates f by solving
     * r = f A, where the rows of A are the lattice vectors in units of alat -- the same cell
     * the Crystal is later built from: the CELL_PARAMETERS vectors when that card is
     * present, the ibrav cell otherwise. alat is celldm(1) (or A), or |a1| for a
     * CELL_PARAMETERS card in bohr or angstrom, which is the value pw.x's input
     * documentation gives it and the one apply_cell_parameters() stores.
     */
    static void convert_atomic_positions_to_crystal(InputData& data, const std::string& units) {
        if (units == "crystal" || data.atoms.empty())
            return;

        bool has_cell = data.has_explicit_lattice_vectors ||
                        (data.bravais_lattice_index != 0 && data.cell_dimension_1 > 0.0);
        if (!has_cell)
            throw std::runtime_error(
                "QE ATOMIC_POSITIONS " + units +
                " rows are Cartesian and need the cell to become crystal coordinates, but the "
                "input defines none: ibrav = 0 needs a CELL_PARAMETERS card, and any other ibrav "
                "needs celldm(1) or A");

        Matrix3x3 lattice_vectors =
            data.has_explicit_lattice_vectors
                ? data.lattice_vectors
                : get_lattice_vectors_from_bravais_index(data.bravais_lattice_index,
                                                         data.cell_dimension_1,
                                                         data.cell_dimension_2,
                                                         data.cell_dimension_3,
                                                         data.cell_dimension_4,
                                                         data.cell_dimension_5,
                                                         data.cell_dimension_6);
        if (std::abs(lattice_vectors.determinant()) < math::tolerances::cell_volume_threshold)
            throw std::runtime_error("QE ATOMIC_POSITIONS " + units +
                                     " rows cannot be converted: the cell has zero volume");

        Real length_to_alat = 1.0;
        if (units == "bohr")
            length_to_alat = 1.0 / data.cell_dimension_1;
        else if (units == "angstrom")
            length_to_alat = constants::ANGSTROM_TO_BOHR / data.cell_dimension_1;

        Matrix3x3 inverse_lattice = lattice_vectors.inverse();
        for (Atom& atom : data.atoms) {
            Vector3D cartesian = atom.position * length_to_alat;
            Vector3D fractional{0.0, 0.0, 0.0};
            for (int column = 0; column < 3; column++)
                fractional[column] = cartesian[0] * inverse_lattice[0][column] +
                                     cartesian[1] * inverse_lattice[1][column] +
                                     cartesian[2] * inverse_lattice[2][column];
            atom.position = fractional;
        }
    }

    static std::string to_lower(std::string text) {
        std::transform(text.begin(), text.end(), text.begin(), ::tolower);
        return text;
    }

    static std::string trim(const std::string& text) {
        size_t first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            return "";
        size_t last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1);
    }

    static std::string strip_quotes(const std::string& text) {
        std::string result = text;
        if (result.size() >= 2 && result.front() == '\'' && result.back() == '\'')
            result = result.substr(1, result.size() - 2);
        if (result.size() >= 2 && result.front() == '"' && result.back() == '"')
            result = result.substr(1, result.size() - 2);
        return result;
    }

    /// Convert Fortran-style "d" exponent to "e" for C++ parsing.
    static double fortran_to_double(const std::string& text) {
        std::string fixed = text;
        for (auto& character : fixed) {
            if (character == 'd' || character == 'D')
                character = 'e';
        }
        return std::stod(fixed);
    }

    /**
     * Read one whitespace-delimited real from a card line, accepting Fortran's
     * `d`/`D` exponent marker.
     *
     * The namelist keys have gone through fortran_to_double() all along; the
     * cards read straight into a `Real` with `>>`, which is not the same thing.
     * `>>` stops at the `d` in `2.0d0`: the first field on the line silently
     * becomes 2.0, every field after it fails to convert, and C++11 then sets
     * each of those to zero. A CELL_PARAMETERS block written in double-precision
     * literals -- the ordinary way to write one when the digits matter, as they
     * do for phonons -- parsed as a degenerate cell, and the run continued to
     * completion reporting `cell volume = 0.0000` instead of stopping.
     *
     * ATOMIC_SPECIES failed more quietly still: `Ni 58.6934d0 Ni.UPF` left `d0`
     * as the next token, so the mass lost its exponent and `d0` was taken as the
     * pseudopotential filename.
     *
     * A token that is present but not a number is refused rather than silently
     * becoming zero, which is what produced the degenerate cell above.
     */
    static void extract_real(std::istringstream& stream,
                             Real& value,
                             const std::string& line,
                             const std::string& card_name) {
        std::string token;
        if (!(stream >> token))
            throw std::runtime_error("QE " + card_name + " line '" + line +
                                     "' ends before all of its numbers were read");
        try {
            std::string fixed = token;
            for (auto& character : fixed) {
                if (character == 'd' || character == 'D')
                    character = 'e';
            }
            size_t characters_consumed = 0;
            value = std::stod(fixed, &characters_consumed);
            if (characters_consumed != fixed.size())
                throw std::invalid_argument("trailing characters");
        } catch (const std::exception&) {
            throw std::runtime_error("QE " + card_name + " line '" + line + "': '" + token +
                                     "' is not a number");
        }
    }

    /**
     * The optional `if_pos` flags after an ATOMIC_POSITIONS row's coordinates.
     *
     * pw.x reads three integers there, one per coordinate: 0 holds that coordinate fixed while
     * the ions move, and 1, the default, lets it move. They do not change an SCF; they are kept
     * on the atom and written to the results' material as ESSE basis.constraints, where the
     * relaxation q3.workflows drives reads them and holds those coordinates fixed. Anything
     * else after the coordinates is not a set of flags, and is refused rather than dropped.
     */
    static std::array<bool, 3> read_position_flags(std::istringstream& stream,
                                                   const std::string& line) {
        std::vector<std::string> trailing_tokens;
        std::string token;
        while (stream >> token)
            trailing_tokens.push_back(token);
        if (trailing_tokens.empty())
            return {true, true, true};

        std::string trailing_text;
        for (const std::string& trailing_token : trailing_tokens)
            trailing_text += (trailing_text.empty() ? "" : " ") + trailing_token;
        bool are_flags =
            trailing_tokens.size() == 3 &&
            std::all_of(trailing_tokens.begin(),
                        trailing_tokens.end(),
                        [](const std::string& flag) { return flag == "0" || flag == "1"; });
        if (!are_flags)
            throw std::runtime_error("QE ATOMIC_POSITIONS line '" + line + "' has '" +
                                     trailing_text +
                                     "' after its coordinates, which is not three if_pos flags "
                                     "(each 0 or 1)");
        return {trailing_tokens[0] == "1", trailing_tokens[1] == "1", trailing_tokens[2] == "1"};
    }

    /**
     * Refuse a DFT-D3 deck that leaves dftd3_version or dftd3_threebody to pw.x.
     *
     * pw.x defaults them to dftd3_version = 3, which is zero damping, and
     * dftd3_threebody = .true. (system_defaults in QE's Modules/read_namelists.f90;
     * INPUT_PW documents the same). Q3 implements the two-body term with Becke-Johnson
     * damping, which pw.x runs only when the deck states dftd3_version = 4 and
     * dftd3_threebody = .false.. A deck that says `vdw_corr = 'dft-d3'` and nothing more
     * therefore asks pw.x for zero-damped D3 with the three-body term, and Q3 used to run
     * it as two-body D3(BJ) and report success.
     *
     * Each key left out is named with the default pw.x takes for it and the line that
     * asks for what Q3 computes, so the refusal says how to fix the deck.
     */
    static void refuse_dftd3_defaults_q3_does_not_implement(bool has_dftd3_version,
                                                            bool has_dftd3_threebody) {
        std::string explanation;
        std::string lines_to_add;
        if (!has_dftd3_version) {
            explanation += " The input does not state dftd3_version, so pw.x uses its default, "
                           "dftd3_version = 3, which is zero damping.";
            lines_to_add += "dftd3_version = 4";
        }
        if (!has_dftd3_threebody) {
            explanation += " The input does not state dftd3_threebody, so pw.x uses its default, "
                           "dftd3_threebody = .true., which adds the three-body term.";
            lines_to_add +=
                std::string(lines_to_add.empty() ? "" : " and ") + "dftd3_threebody = .false.";
        }
        if (lines_to_add.empty())
            return;
        throw std::runtime_error("QE input selects DFT-D3 with vdw_corr, and Q3 implements only "
                                 "its two-body term with Becke-Johnson damping." +
                                 explanation + " To run two-body D3(BJ) in both codes, add " +
                                 lines_to_add + " to &SYSTEM.");
    }

    /// Position of the first `target` outside a quoted string, or npos.
    static size_t first_unquoted_position(const std::string& line, char target) {
        char quote = '\0';
        for (size_t position = 0; position < line.size(); position++) {
            char character = line[position];
            if (quote != '\0') {
                if (character == quote)
                    quote = '\0';
            } else if (character == '\'' || character == '"') {
                quote = character;
            } else if (character == target) {
                return position;
            }
        }
        return std::string::npos;
    }

    /// Remove a trailing `!` comment, respecting quoted strings.
    static std::string strip_comment(const std::string& line) {
        return line.substr(0, first_unquoted_position(line, '!'));
    }

    /**
     * Remove the `/` that closes the namelist from a line inside it, and say whether the
     * line had one.
     *
     * Fortran ends namelist input at the first slash outside a quoted string, so that is the
     * one taken: a slash inside quotes is part of a value (`pseudo_dir = './pseudo/'`), and
     * there is no unquoted value a slash can belong to.
     *
     * Fortran then skips the rest of the line, so anything written after the slash would be
     * dropped without a word -- by pw.x as well. It is refused instead, the way this parser
     * refuses every other piece of input it would otherwise drop.
     */
    static bool strip_namelist_terminator(std::string& line, const std::string& section) {
        size_t slash_position = first_unquoted_position(line, '/');
        if (slash_position == std::string::npos)
            return false;
        std::string after_slash = trim(line.substr(slash_position + 1));
        if (!after_slash.empty())
            throw std::runtime_error("QE namelist line '" + trim(line) + "' has '" + after_slash +
                                     "' after the '/' that closes &" + section +
                                     ". Fortran stops reading the namelist at that slash and "
                                     "skips the rest of the line, so Q3 refuses the line rather "
                                     "than drop that text without a word.");
        line = line.substr(0, slash_position);
        return true;
    }

    /**
     * The key names assigned on one namelist line.
     *
     * Fortran allows several assignments per line separated by commas, and QE's
     * own examples use that form freely (`ibrav= 2, celldm(1) =10.2, nat= 2`),
     * so this splits rather than assuming one key per line. Array indices are
     * dropped: `celldm(3)` and `starting_magnetization(2)` classify under their
     * base names.
     */
    static std::vector<std::pair<std::string, std::string>>
    namelist_keys_on_line(const std::string& line) {
        std::vector<std::pair<std::string, std::string>> keys;
        std::vector<std::string> fragments;
        std::string current;
        char quote = '\0';
        /**
         * Commas inside parentheses are index separators, not entry
         * separators: `starting_ns_eigenvalue(5,2,1) = 1.d0` is one
         * assignment. Only a top-level comma splits.
         */
        int parenthesis_depth = 0;
        for (char character : line) {
            if (quote != '\0') {
                if (character == quote)
                    quote = '\0';
            } else if (character == '\'' || character == '"') {
                quote = character;
            } else if (character == '(') {
                parenthesis_depth++;
            } else if (character == ')') {
                parenthesis_depth = std::max(0, parenthesis_depth - 1);
            } else if (character == ',' && parenthesis_depth == 0) {
                fragments.push_back(current);
                current.clear();
                continue;
            }
            current += character;
        }
        fragments.push_back(current);

        for (const std::string& fragment : fragments) {
            std::string trimmed_fragment = trim(fragment);
            if (trimmed_fragment.empty())
                continue;
            size_t equals_position = trimmed_fragment.find('=');
            if (equals_position == std::string::npos)
                throw std::runtime_error(
                    "QE namelist entry '" + trimmed_fragment +
                    "' is not an assignment. Fortran's array-continuation form "
                    "(`starting_magnetization = 0.5, 0.3`) is not read; give each element its "
                    "own indexed assignment.");
            std::string key = to_lower(trim(trimmed_fragment.substr(0, equals_position)));
            size_t parenthesis_position = key.find('(');
            if (parenthesis_position != std::string::npos)
                key = key.substr(0, parenthesis_position);
            if (!key.empty())
                keys.emplace_back(
                    key,
                    to_lower(strip_quotes(trim(trimmed_fragment.substr(equals_position + 1)))));
        }
        return keys;
    }

    /**
     * Whether a key's value is the QE default that leaves the feature off.
     *
     * Numeric defaults are compared as numbers so that `-10000`, `-10000.0` and
     * `-1.0d4` all read alike; everything else is compared literally, with
     * `.false.` going through the Fortran-logical spellings.
     */
    static bool is_default_value(const std::string& default_token, const std::string& value) {
        if (default_token == ".false.")
            return !fortran_logical_is_true(value);
        try {
            size_t consumed = 0;
            Real default_number = std::stod(default_token, &consumed);
            if (consumed == default_token.size())
                return std::abs(fortran_to_double(value) - default_number) < 1e-9;
        } catch (const std::exception&) {
            // Not numeric: fall through to the literal comparison.
        }
        return value == default_token;
    }

    /**
     * Sort every key on a namelist line into honoured / neutral / unimplemented
     * / unknown, throwing on the last two.
     */
    static void classify_namelist_keys(const std::string& line, const std::string& section) {
        const auto& honoured = qe_input::honoured_keys_by_namelist().at(section);
        for (const auto& [key, value] : namelist_keys_on_line(line)) {
            if (honoured.count(key) != 0)
                continue;
            if (qe_input::physics_neutral_keys().count(key) != 0) {
                log::info("     QE input: {} accepted and ignored (does not affect results)", key);
                continue;
            }
            auto unused_ionic_step = qe_input::unused_ionic_step_keys().find(key);
            if (unused_ionic_step != qe_input::unused_ionic_step_keys().end()) {
                log::info("     QE input: {} is not used ({}); it changes how a relaxation "
                          "reaches its minimum, not the minimum",
                          key,
                          unused_ionic_step->second);
                continue;
            }
            /**
             * A feature switched off explicitly asks for what Q3 already does,
             * so it is accepted silently — no note, because there is nothing
             * for the user to check.
             */
            auto inert = qe_input::keys_inert_at_default().find(key);
            if (inert != qe_input::keys_inert_at_default().end() &&
                is_default_value(inert->second, value))
                continue;
            auto unimplemented = qe_input::unimplemented_keys().find(key);
            if (unimplemented != qe_input::unimplemented_keys().end())
                throw std::runtime_error("QE namelist key '" + key + "' in &" + section +
                                         " is not supported: " + unimplemented->second +
                                         ". Remove it or use Q3's own input format.");
            /**
             * A key Q3 reads elsewhere is a misplacement, not an unknown key,
             * and saying which namelist it belongs in is more useful than
             * saying it was never heard of.
             */
            for (const auto& [other_section, other_keys] : qe_input::honoured_keys_by_namelist())
                if (other_section != section && other_keys.count(key) != 0)
                    throw std::runtime_error("QE namelist key '" + key + "' belongs to &" +
                                             other_section + ", not &" + section);
            throw std::runtime_error(
                "QE namelist key '" + key + "' in &" + section +
                " is not recognized. Q3 refuses unknown keys rather than ignoring them, because "
                "ignoring one that changes physics silently produces a result that does not "
                "match the input.");
        }
    }

    /**
     * QE's crystallographic cell form (A, B, C in angstrom and the three
     * cosines) converted to celldm, following QE's own A/B/C-to-celldm
     * convention — including which celldm each cosine lands in, which depends
     * on the Bravais index.
     */
    static void apply_crystallographic_constants(InputData& data,
                                                 Real length_a,
                                                 Real length_b,
                                                 Real length_c,
                                                 Real cosine_ab,
                                                 Real cosine_ac,
                                                 Real cosine_bc) {
        if (length_a <= 0.0)
            throw std::runtime_error("QE input gives A = " + std::to_string(length_a) +
                                     "; the lattice parameter A must be positive");
        if (data.cell_dimension_1 > 0.0)
            throw std::runtime_error("QE input gives both celldm(1) and A; they are alternative "
                                     "spellings of the same quantity and QE refuses the pair too");

        data.cell_dimension_1 = length_a * constants::ANGSTROM_TO_BOHR;
        data.cell_dimension_2 = (length_b > 0.0) ? length_b / length_a : 0.0;
        data.cell_dimension_3 = (length_c > 0.0) ? length_c / length_a : 0.0;

        int index = data.bravais_lattice_index;
        if (index == 14 || index == 0) {
            data.cell_dimension_4 = cosine_bc;
            data.cell_dimension_5 = cosine_ac;
            data.cell_dimension_6 = cosine_ab;
        } else if (index == -12 || index == -13) {
            data.cell_dimension_5 = cosine_ac;
        } else if (index == 5 || index == -5 || index == 12 || index == 13) {
            data.cell_dimension_4 = cosine_ab;
        }
    }

    /// Whether a Fortran logical reads as true (`.true.`, `T`, `.t.`).
    static bool fortran_logical_is_true(const std::string& value) {
        std::string lower = to_lower(trim(strip_quotes(value)));
        return lower == ".true." || lower == "t" || lower == ".t." || lower == "true";
    }

    /**
     * Whether `position` starts a whole namelist key rather than the tail of a
     * longer one.
     *
     * Substring matching is not good enough here: `conv_thr` occurs inside both
     * `etot_conv_thr` and `forc_conv_thr`, and the single-letter key `A` occurs
     * inside nearly everything. A key is preceded by a separator and followed
     * by `=`, whitespace, or its array index.
     */
    static bool is_whole_key_at(const std::string& lower_line, size_t position, size_t length) {
        if (position > 0) {
            char before = lower_line[position - 1];
            if (before != ' ' && before != '\t' && before != ',')
                return false;
        }
        size_t after_position = position + length;
        if (after_position >= lower_line.size())
            return false;
        char after = lower_line[after_position];
        return after == '=' || after == ' ' || after == '\t' || after == '(';
    }

    /// Extract value from "key = value" or "key=value" line.
    static bool
    extract_key_value(const std::string& line, const std::string& key, std::string& value) {
        std::string lower_line = to_lower(line);
        std::string lower_key = to_lower(key);
        size_t position = lower_line.find(lower_key);
        while (position != std::string::npos &&
               !is_whole_key_at(lower_line, position, lower_key.size()))
            position = lower_line.find(lower_key, position + 1);
        if (position == std::string::npos)
            return false;

        size_t equals_position = line.find('=', position);
        if (equals_position == std::string::npos)
            return false;

        size_t start = equals_position + 1;
        while (start < line.size() && line[start] == ' ')
            start++;
        size_t end = line.find_first_of(",!\n\r", start);
        if (end == std::string::npos)
            end = line.size();
        value = trim(line.substr(start, end - start));
        value = strip_quotes(value);
        return true;
    }
};

} // namespace q3
