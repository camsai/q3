#pragma once

/**
 * Closed-form estimate of a plane-wave run's memory footprint, per rank.
 *
 * Printing one before the run starts is how a user learns that a deck will not
 * fit without spending an allocation finding out. Q3 had none, which is how a
 * 432-atom example came to carry a comment saying it needed 45 GB per rank when
 * it needs more than twice that.
 *
 * The estimate is deliberately arithmetic on the dimensions the setup already
 * knows, not a sum over live allocations: it has to be printable before the
 * solver has allocated anything. Its accuracy is therefore a claim about the
 * code that can be checked against the measured resident set, and the
 * checkpoint probes in ProfilingCounters exist to check it.
 *
 * Every term below is replicated on every rank when bands are distributed,
 * because that mode keeps the whole G-sphere on each rank by construction.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace q3 {

/// One line of the estimate: what it is, how it is sized, and how large it is.
struct PlanewaveMemoryEstimateTerm {
    std::string label;
    std::string formula;
    std::uint64_t bytes = 0;
};

/**
 * The dimensions an estimate needs, gathered from the basis, the
 * pseudopotential and the solver configuration.
 */
struct PlanewaveMemoryEstimateInputs {
    int number_of_bands = 0;
    int max_plane_waves_per_kpoint = 0;
    int number_of_g_vectors = 0;
    int number_of_projectors = 0;
    int number_of_atomic_orbitals = 0;
    int number_of_atoms = 0;
    int number_of_spin_components = 1;
    int local_number_of_kpoint_states = 1;
    int mixing_history_depth = 0;
    std::int64_t fft_rho_grid_points = 0;
    std::int64_t fft_wave_grid_points = 0;
    int davidson_basis_expansion_factor = 2;
    /// False for ultrasoft and PAW, where S is not the identity and is stored.
    bool overlap_is_identity = false;
    /// False for the direct solver, which holds no Davidson workspace.
    bool diagonalizer_is_davidson = true;
};

/**
 * A footprint estimate: the terms that persist, the terms that only spike, and
 * the totals a reader actually acts on.
 */
struct PlanewaveMemoryEstimate {
    std::vector<PlanewaveMemoryEstimateTerm> steady_state;
    std::vector<PlanewaveMemoryEstimateTerm> transient;
    std::uint64_t steady_state_bytes = 0;
    /// What the solver workspace alone contributes to the steady state.
    std::uint64_t solver_bytes = 0;
    /// The tallest point of setup, before the solver workspace exists.
    std::uint64_t setup_peak_bytes = 0;
    std::uint64_t peak_bytes = 0;
    int maximum_basis_size = 0;

    static PlanewaveMemoryEstimate compute(const PlanewaveMemoryEstimateInputs& inputs);
};

inline PlanewaveMemoryEstimate
PlanewaveMemoryEstimate::compute(const PlanewaveMemoryEstimateInputs& inputs) {
    const std::uint64_t complex_size = 16;
    const std::uint64_t real_size = 8;
    const std::uint64_t bands = static_cast<std::uint64_t>(inputs.number_of_bands);
    const std::uint64_t plane_waves = static_cast<std::uint64_t>(inputs.max_plane_waves_per_kpoint);
    const std::uint64_t g_vectors = static_cast<std::uint64_t>(inputs.number_of_g_vectors);
    const std::uint64_t projectors = static_cast<std::uint64_t>(inputs.number_of_projectors);
    const std::uint64_t spins = static_cast<std::uint64_t>(inputs.number_of_spin_components);

    PlanewaveMemoryEstimate estimate;

    /**
     * The Davidson basis is sized by the GLOBAL band count, never by a rank's
     * local range: band distribution splits the loops over these arrays, not
     * the arrays themselves.
     */
    int expanded = inputs.davidson_basis_expansion_factor * inputs.number_of_bands;
    int clamped =
        expanded < inputs.max_plane_waves_per_kpoint ? expanded : inputs.max_plane_waves_per_kpoint;
    estimate.maximum_basis_size =
        clamped > inputs.number_of_bands ? clamped : inputs.number_of_bands;
    const std::uint64_t basis = static_cast<std::uint64_t>(estimate.maximum_basis_size);

    auto add_steady = [&estimate](const std::string& label,
                                  const std::string& formula,
                                  std::uint64_t bytes,
                                  bool belongs_to_solver = false) {
        if (bytes == 0) {
            return;
        }
        estimate.steady_state.push_back({label, formula, bytes});
        estimate.steady_state_bytes += bytes;
        if (belongs_to_solver) {
            estimate.solver_bytes += bytes;
        }
    };
    auto add_transient =
        [&estimate](const std::string& label, const std::string& formula, std::uint64_t bytes) {
            if (bytes == 0) {
                return;
            }
            estimate.transient.push_back({label, formula, bytes});
        };

    if (inputs.diagonalizer_is_davidson) {
        add_steady("Davidson basis V, HV",
                   "2 x M x npw x 16",
                   2 * basis * plane_waves * complex_size,
                   true);
        /**
         * S|V> is stored only when S is not the identity. For norm-conserving
         * runs it would be a verbatim copy of the basis, so the solver aliases
         * the two and this term disappears.
         */
        if (!inputs.overlap_is_identity) {
            add_steady(
                "Davidson basis SV", "M x npw x 16", basis * plane_waves * complex_size, true);
        }
        /**
         * One band-sized buffer, borrowed for the two rotations that cannot be
         * done in place. The Ritz vectors, H|Ritz>, S|Ritz> and the residual
         * array are gone: correction vectors are built straight into the free
         * basis columns.
         */
        add_steady("rotation scratch", "nb x npw x 16", bands * plane_waves * complex_size, true);
        if (!inputs.overlap_is_identity) {
            add_steady(
                "Ritz overlap vectors", "nb x npw x 16", bands * plane_waves * complex_size, true);
        }
        /**
         * H and S in the subspace, the eigenvector matrix, and the copy of S the
         * eigensolver is allowed to destroy; the Hamiltonian's copy is the
         * eigenvector buffer itself. At Gamma the solver packs to real in place
         * rather than into copies of its own.
         */
        add_steady("subspace matrices", "4 x M^2 x 16", 4 * basis * basis * complex_size, true);
    }

    add_steady("wavefunctions",
               "nkstates x nb x npw x 16",
               static_cast<std::uint64_t>(inputs.local_number_of_kpoint_states) * bands *
                   plane_waves * complex_size);
    add_steady("nonlocal projectors", "nkb x npw x 16", projectors * plane_waves * complex_size);
    /**
     * The D matrix is block-diagonal by atom and is stored that way. As a dense
     * nkb x nkb it was 967 MB on the 432-atom deck, of which 2 MB was nonzero.
     * The estimate does not carry the per-atom channel count, so it uses the
     * average, nkb^2 / nat.
     */
    if (inputs.number_of_atoms > 0) {
        add_steady("block-diagonal D matrix",
                   "nkb^2 / nat x 16",
                   projectors * projectors / static_cast<std::uint64_t>(inputs.number_of_atoms) *
                       complex_size);
    }
    add_steady("Broyden history",
               "3 x ndim x nspin x ngm x 16",
               3 * static_cast<std::uint64_t>(inputs.mixing_history_depth) * spins * g_vectors *
                   complex_size);
    add_steady("G-vector table", "ngm x 80", g_vectors * 80);
    add_steady("FFT buffers",
               "2 x (Nrho + 2 x Nwave) x 16",
               2 *
                   (static_cast<std::uint64_t>(inputs.fft_rho_grid_points) +
                    2 * static_cast<std::uint64_t>(inputs.fft_wave_grid_points)) *
                   complex_size);
    add_steady("density and potential",
               "(4 + 3 x nspin) x Nrho x 8",
               (4 + 3 * spins) * static_cast<std::uint64_t>(inputs.fft_rho_grid_points) *
                   real_size);

    /**
     * The atomic-orbital start holds five arrays of max(natwfc, nb) columns at
     * once. It is a spike, not a resident cost, but it is the tallest point of
     * a small run and it is what an undersized allocation dies in.
     */
    std::uint64_t starting_columns = static_cast<std::uint64_t>(
        inputs.number_of_atomic_orbitals > inputs.number_of_bands ? inputs.number_of_atomic_orbitals
                                                                  : inputs.number_of_bands);
    add_transient("wavefunction initialization",
                  "5 x max(natwfc, nb) x npw x 16",
                  5 * starting_columns * plane_waves * complex_size);

    /**
     * The setup spike and the solver workspace never coexist: the initializer's
     * arrays are gone by the time the first solve allocates. So the peak is the
     * taller of the two phases, not their sum.
     */
    std::uint64_t largest_transient = 0;
    for (const auto& term : estimate.transient) {
        if (term.bytes > largest_transient) {
            largest_transient = term.bytes;
        }
    }
    estimate.setup_peak_bytes =
        estimate.steady_state_bytes - estimate.solver_bytes + largest_transient;
    estimate.peak_bytes = estimate.steady_state_bytes > estimate.setup_peak_bytes
                              ? estimate.steady_state_bytes
                              : estimate.setup_peak_bytes;
    return estimate;
}

} // namespace q3
