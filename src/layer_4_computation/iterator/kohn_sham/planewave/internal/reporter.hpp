#pragma once

/**
 * PlanewaveKohnShamSCFReporter — all human-readable SCF output for a plane-wave KS calculation.
 *
 * This class (and its companion data structs) is the single location for every
 * log::info / log::warn call that describes SCF progress, convergence, eigenvalues, and the
 * final energy breakdown.
 *
 * Design: the reporter is completely decoupled from the iterator type. The iterator
 * populates a plain data struct and passes it to a static report function. No iterator
 * header is included here, so there is no circular dependency.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/math/tolerances.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_1_structure/reciprocal_space/kpoint.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/internal/memory_estimate.hpp"
#include "layer_4_computation/state/internal/energies/kohn_sham/planewave/init.hpp"

namespace q3 {

// ─── Data-transfer structs ───────────────────────────────────────────────────

/**
 * All data needed to produce the post-initialization setup summary log block.
 */
struct PlanewaveKohnShamSCFSetupSummary {
    // Crystal geometry
    Real lattice_parameter;
    Real cell_volume;
    int number_of_atoms;

    // Plane-wave basis
    Real wavefunction_energy_cutoff;
    Real charge_density_energy_cutoff;
    int fft_rho_grid_dimension_1;
    int fft_rho_grid_dimension_2;
    int fft_rho_grid_dimension_3;
    int fft_wave_grid_dimension_1;
    int fft_wave_grid_dimension_2;
    int fft_wave_grid_dimension_3;
    int number_of_g_vectors;
    int number_of_kpoints;
    int max_plane_waves_per_kpoint;

    // Electronic structure
    int number_of_spin_components;
    Real number_of_electrons;
    int number_of_bands;

    // Method / solver choices
    std::string xc_functional_name;
    std::string diagonalizer_name;
    /// Davidson working basis as a multiple of the band count; 0 when not iterative.
    int davidson_basis_expansion_factor = 0;
    std::string mixer_name;
    Real mixing_beta;

    // Energies
    Real ewald_energy;

    // Symmetry
    bool density_symmetrization_active;
    int number_of_shells;
    int number_of_symmetry_operations;

    /**
     * Parallel topology, as it ended up rather than as it was requested. Set
     * only for MPI runs; number_of_processes stays 1 in serial, which is what
     * suppresses the block below.
     */
    int number_of_processes = 1;
    int number_of_kpoint_pools = 1;
    /// Ranks sharing a k-point pool, however that pool divides its work.
    int band_group_size = 1;
    bool replicated_plane_wave_fallback_active = false;
    /// True when those ranks divide the bands rather than the reciprocal-space coefficients.
    bool bands_are_distributed = false;
};

/**
 * All data needed to produce the post-convergence report block
 * (magnetization, eigenvalues, and final energies).
 *
 * The `energies` pointer is non-owning. The iterator that owns the energies
 * must outlive this struct.
 */
struct PlanewaveKohnShamSCFConvergenceSummary {
    int number_of_spin_components;
    /// The reporting rank's pool share under MPI, not the irreducible total.
    int number_of_kpoints;
    int number_of_bands;
    /// Set from the execution policy so the report can say whose k-points it lists.
    int number_of_kpoint_pools = 1;

    const PlanewaveKohnShamEnergies* energies; // non-owning

    /// eigenvalues[kpoint_state_index][band_index] in Rydberg.
    std::vector<std::vector<Real>> eigenvalues;
    /// k-points in the same order as eigenvalues rows.
    std::vector<KPoint> kpoints;

    /// Globally integrated spin-density difference in Bohr magnetons per cell.
    Real total_magnetization = 0.0;
    /// Globally integrated absolute spin-density difference in Bohr magnetons per cell.
    Real absolute_magnetization = 0.0;
};

// ─── Reporter ────────────────────────────────────────────────────────────────

/**
 * Stateless reporter: all methods are static and write to the global logger.
 */
class PlanewaveKohnShamSCFReporter {
  public:
    /**
     * Print the estimated per-rank memory footprint, term by term.
     *
     * Printed after the pseudopotential summary rather than with the rest of
     * the setup block, because the projector count it needs does not exist
     * until the pseudopotential basis has been built.
     *
     * The figures are what the code will allocate, not what it ought to: terms
     * whose labels say "copy" are redundant today and are named so a reader can
     * see the difference between the footprint and the requirement.
     *
     * A dozen lines of arithmetic nobody reads on a run that fits, so it rides
     * at debug with the rest of the diagnostics; the measured peak at the end
     * of the run still prints, and is what a reader who did not ask for this
     * compares against.
     */
    static void report_memory_estimate(const PlanewaveMemoryEstimate& estimate) {
        log::debug("     Estimated memory per rank (arrays only, replicated when bands "
                   "are distributed):");
        for (const auto& term : estimate.steady_state) {
            log::debug(
                "       {:<38} {:<32} = {}", term.label, term.formula, format_bytes(term.bytes));
        }
        log::debug("       {:<38} {:<32} = {}",
                   "steady state",
                   "",
                   format_bytes(estimate.steady_state_bytes));
        for (const auto& term : estimate.transient) {
            log::debug("       {:<38} {:<32} = {} (transient)",
                       term.label,
                       term.formula,
                       format_bytes(term.bytes));
        }
        log::debug("       {:<38} {:<32} ~ {}",
                   "estimated peak per rank",
                   "",
                   format_bytes(estimate.peak_bytes));
        /**
         * This is what the run allocates, which is not the same as what it
         * keeps resident, and the difference goes both ways. Large transient
         * blocks freed during setup are commonly retained by the allocator
         * rather than returned, which pushes the resident set above this
         * figure. Against that, arrays are sized for the largest basis the
         * solver may reach, and a run that converges before filling one never
         * touches all of it. Compare with the measured peak at the end of the
         * run rather than trusting either number alone.
         */
        log::debug("       (arrays as allocated; compare with the measured peak at end of run)");
    }

    /**
     * Print the one-time setup summary that appears after initialization.
     */
    static void report_setup_summary(const PlanewaveKohnShamSCFSetupSummary& summary) {
        log::info("");
        log::info("     Q3 - Plane-wave DFT SCF Solver");
        log::info("     ================================");
        log::info("");
        log::info("     lattice parameter (alat)  = {:.4f} a.u.", summary.lattice_parameter);
        log::info("     cell volume               = {:.4f} (a.u.)^3", summary.cell_volume);
        log::info("     ecutwfc                   = {:.4f} Ry", summary.wavefunction_energy_cutoff);
        log::info("     ecutrho                   = {:.4f} Ry",
                  summary.charge_density_energy_cutoff);
        log::info("     FFT grid (rho)            = {} x {} x {}",
                  summary.fft_rho_grid_dimension_1,
                  summary.fft_rho_grid_dimension_2,
                  summary.fft_rho_grid_dimension_3);
        log::info("     FFT grid (wave)           = {} x {} x {}",
                  summary.fft_wave_grid_dimension_1,
                  summary.fft_wave_grid_dimension_2,
                  summary.fft_wave_grid_dimension_3);
        log::info("     number of G-vectors       = {}", summary.number_of_g_vectors);
        // The irreducible total; the iterator assembles it across pools.
        log::info("     number of k-points        = {}", summary.number_of_kpoints);
        log::info("     number of atoms           = {}", summary.number_of_atoms);
        if (summary.number_of_spin_components == 2) {
            log::info("     spin polarization         = LSDA (number_of_spin_components=2)");
        }
        log::info("     number of electrons       = {:.4f}", summary.number_of_electrons);
        log::info("     number of bands           = {}", summary.number_of_bands);
        /**
         * How the ranks were actually arranged, and whether G-vectors ended up
         * distributed. Without this the log cannot distinguish a run that used
         * band groups from one that silently fell back to replicated G-vectors,
         * which changes both the performance and which properties are available.
         */
        if (summary.number_of_processes > 1) {
            /**
             * Ranks per pool is derived rather than taken from band_group_size,
             * because the replicated-G fallback sets the band group size to 1
             * (see the basis rebuild in the iterator) while the ranks stay in
             * their pools. Printing that value made the parenthetical fail to
             * factor the rank count -- "128 (16 pool(s) x 1 rank(s) per pool)"
             * -- and contradict the topology line logged at startup. The
             * G-vector line below is what carries the fallback.
             */
            const int ranks_per_pool =
                (summary.number_of_kpoint_pools > 0)
                    ? summary.number_of_processes / summary.number_of_kpoint_pools
                    : summary.number_of_processes;
            log::info("     MPI ranks                 = {} ({} pool(s) x {} rank(s) per pool)",
                      summary.number_of_processes,
                      summary.number_of_kpoint_pools,
                      ranks_per_pool);
            if (summary.bands_are_distributed) {
                /**
                 * Replicated here is the point rather than a fallback: each rank
                 * holds every G-vector precisely so that it can own whole bands,
                 * which is what keeps its transforms local and leaves stress and
                 * export available.
                 */
                log::info("     G-vector distribution     = replicated (bands divided across the "
                          "pool's ranks)");
            } else if (summary.replicated_plane_wave_fallback_active) {
                log::info("     G-vector distribution     = replicated (fallback: a rank's FFT "
                          "slab held no plane waves; every rank in a pool repeats "
                          "the whole solve)");
            } else if (summary.band_group_size > 1) {
                log::info(
                    "     G-vector distribution     = slab-distributed across the band group");
            } else {
                log::info("     G-vector distribution     = replicated (one rank per pool)");
            }
        }
        log::info("");
        log::info("     XC functional             = {}", summary.xc_functional_name);
        log::info("     diagonalization           = {}", summary.diagonalizer_name);
        log::info("     mixing mode               = {} (beta={:.2f})",
                  summary.mixer_name,
                  summary.mixing_beta);
        log::info("");
        log::info("     max PW per k-point        = {}", summary.max_plane_waves_per_kpoint);
        if (summary.davidson_basis_expansion_factor > 0) {
            log::info("     Davidson basis            = {} x {} bands = {} vectors",
                      summary.davidson_basis_expansion_factor,
                      summary.number_of_bands,
                      summary.davidson_basis_expansion_factor * summary.number_of_bands);
        }
        log::info("");
        log::info("     Ewald energy              = {:14.9f} Ry", summary.ewald_energy);
        log::info("");
        if (summary.density_symmetrization_active) {
            log::info(
                "     Density symmetrization    = {} shells from {} G-vectors ({} symmetry ops)",
                summary.number_of_shells,
                summary.number_of_g_vectors,
                summary.number_of_symmetry_operations);
        }
    }

    /**
     * Print the first SCF iteration line, and its energy breakdown.
     *
     * Only the first line belongs to the SCF trajectory a reader follows; the
     * four components below it are a one-off diagnostic for a first iteration
     * that landed somewhere unexpected, so they ride at debug with the rest of
     * the diagnostics.
     */
    static void report_first_iteration(Real total_energy,
                                       const PlanewaveKohnShamEnergies& energies) {
        log::info("     iteration # {:3d}     Etot = {:14.9f} Ry", 1, total_energy);
        log::debug("       band energy                    = {:.8f}", energies.band_energy);
        log::debug("       double counting correction     = {:.8f}",
                   energies.double_counting_correction);
        log::debug("       Hartree energy                 = {:.8f}", energies.hartree_energy);
        log::debug("       XC energy                      = {:.8f}",
                   energies.exchange_correlation_energy);
    }

    /**
     * Print the full post-convergence block: magnetization (if spin-polarized),
     * eigenvalues (first ≤10 k-point states), and the final energy breakdown.
     */
    static void report_convergence_summary(const PlanewaveKohnShamSCFConvergenceSummary& summary) {
        report_magnetization(summary);
        report_eigenvalues(summary);
        report_final_energies(*summary.energies);
    }

  private:
    /**
     * Render a byte count in the unit a reader can compare against a node.
     *
     * MiB below a gibibyte, GiB above it: a 54-atom deck and a 432-atom deck
     * differ by three orders of magnitude and neither reads well in the
     * other's unit.
     */
    static std::string format_bytes(std::uint64_t bytes) {
        char buffer[32];
        double gibibytes = static_cast<double>(bytes) / 1073741824.0;
        if (gibibytes < 1.0) {
            std::snprintf(buffer, sizeof(buffer), "%8.1f MiB", gibibytes * 1024.0);
        } else {
            std::snprintf(buffer, sizeof(buffer), "%8.2f GiB", gibibytes);
        }
        return std::string(buffer);
    }

    static void report_magnetization(const PlanewaveKohnShamSCFConvergenceSummary& summary) {
        if (summary.number_of_spin_components != 2) {
            return;
        }
        /**
         * Both of QE's numbers, because the total alone cannot describe an
         * antiferromagnet: ∫(ρ↑−ρ↓) is zero for a converged AFM state *and*
         * for one that collapsed to non-magnetic, so reporting only the total
         * left "did the moments survive?" unanswerable from Q3's own output.
         * ∫|ρ↑−ρ↓| separates them — chromium's AFM state carries 2.60 Bohr
         * mag/cell of absolute magnetization at zero total.
         */
        log::info("");
        log::info("     total magnetization = {:.4f} Bohr mag/cell", summary.total_magnetization);
        log::info("     absolute magnetization = {:.4f} Bohr mag/cell",
                  summary.absolute_magnetization);
    }

    static void report_eigenvalues(const PlanewaveKohnShamSCFConvergenceSummary& summary) {
        int number_of_kpoint_states = summary.number_of_kpoints * summary.number_of_spin_components;
        /**
         * Every k-point, every time. This used to print min(n, 10) of whatever
         * the reporting rank happened to hold, so a 60-k-point run showed ten
         * in serial and its pool's eight under MPI, with the labels restarting
         * at k_0 in each pool. The iterator now assembles the global grid
         * before calling this, so the indices below are the irreducible ones
         * and the count matches the IBZ reduction reported at startup.
         */
        int number_of_states_to_print = number_of_kpoint_states;
        log::section("Eigenvalues (eV), " + std::to_string(number_of_kpoint_states) +
                     " k-point state(s):");
        for (int kpoint_index_state = 0; kpoint_index_state < number_of_states_to_print;
             ++kpoint_index_state) {
            int spin_index = kpoint_index_state / summary.number_of_kpoints;
            int kpoint_index = kpoint_index_state % summary.number_of_kpoints;
            std::string spin_label = (summary.number_of_spin_components == 2)
                                         ? ((spin_index == 0) ? " (up)" : " (dn)")
                                         : "";
            std::string line = "     E_k = ( ";
            for (int band_index = 0; band_index < summary.number_of_bands; ++band_index) {
                char buffer[32];
                std::snprintf(buffer,
                              sizeof(buffer),
                              "%.4f",
                              summary.eigenvalues[kpoint_index_state][band_index] *
                                  constants::RY_TO_EV);
                line += buffer;
                if (band_index < summary.number_of_bands - 1) {
                    line += ", ";
                }
            }
            line += " )";
            const KPoint& kpoint = summary.kpoints[kpoint_index];
            log::info("     k_{} = ({:.4f}, {:.4f}, {:.4f}){}",
                      kpoint_index,
                      kpoint.coordinates.x,
                      kpoint.coordinates.y,
                      kpoint.coordinates.z,
                      spin_label);
            log::info("{}", line);
            if (kpoint_index_state + 1 < number_of_states_to_print) {
                log::info("");
            }
        }
    }

    static void report_final_energies(const PlanewaveKohnShamEnergies& energies) {
        log::section("Final Energies (Ry):");
        log::info("     total energy            = {:14.9f} Ry", energies.total());
        log::info("     band energy             = {:14.9f}", energies.band_energy);
        log::info("     Hartree energy          = {:14.9f}", energies.hartree_energy);
        log::info("     XC energy               = {:14.9f}", energies.exchange_correlation_energy);
        if (energies.ewald_energy != 0.0) {
            log::info("     Ewald energy            = {:14.9f}", energies.ewald_energy);
        }
        if (energies.dispersion_energy() != 0.0) {
            log::info("     Dispersion (D3-BJ)      = {:14.9f}", energies.dispersion_energy());
        }
        if (energies.hubbard_energy_contribution() != 0.0) {
            log::info("     DFT+U (Hubbard)         = {:14.9f}",
                      energies.hubbard_energy_contribution());
        }
        /**
         * Printed with the same label QE uses, deliberately: it is the first
         * rung of the PAW failure ladder, and a mismatched total is diagnosed
         * by putting these two lines side by side before anything else.
         */
        if (energies.one_center_energy_contribution() != 0.0) {
            log::info("     one-center paw contrib. = {:14.9f}",
                      energies.one_center_energy_contribution());
        }
        if (std::abs(energies.smearing_entropy_contribution) >
            math::tolerances::zero_vector_magnitude_threshold) {
            log::info("     smearing contribution   = {:14.9f}",
                      energies.smearing_entropy_contribution);
            log::info("     E_one_electron + E_TS   = {:14.9f}",
                      energies.band_energy + energies.double_counting_correction);
        }
        log::info("");
    }
};

} // namespace q3
