#pragma once

/**
 * export_and_record_mean_field() — the single call both flavors make.
 *
 * Does two things for a finished run:
 *
 *   1. runs the requested exporter ("none" resolves to the Null object, so no
 *      call site branches on the format name); and
 *   2. when an export was actually requested, also drops the format-agnostic
 *      MeanFieldRecord into `<prefix>.save/mean-field.q3mf`, so the same run
 *      can later be converted to a *different* format with `q3_utility convert`
 *      instead of being recomputed.
 *
 * The record is deliberately NOT written when `mean_field_export` is "none".
 * It duplicates the wavefunctions (which `wfc*.dat` already stores, in a
 * write-only QE-compatible format with no reader), so writing it
 * unconditionally would roughly double every run's output for users who never
 * convert anything. Tying it to "the user asked for a mean-field export" costs
 * nothing to everyone else, needs no new control key, and still covers the case
 * the converter exists for -- exporting to a second format later. Teaching the
 * converter to read `wfc*.dat` and `charge-density.dat` directly would retire
 * the duplication entirely; that is the natural follow-up, and needs a reader
 * for those two formats first.
 *
 * A failure to write the record must not lose the export the user asked for, so
 * it is logged rather than thrown: WFN/RHO/vxc.dat are already on disk by then.
 */

#include <string>
#include <utility>

#include "io/input/data.hpp"
#include "io/logger.hpp"
#include "io/mean_field_export/factory.hpp"
#include "io/mean_field_record/checkpoint.hpp"
#include "io/mean_field_record/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/init.hpp"

namespace q3 {

inline void export_and_record_mean_field(const MeanFieldRecord& record,
                                         const InputData& input,
                                         const std::string& directory) {
    create_mean_field_exporter(input.mean_field_export)->export_to_directory(record, directory);

    if (input.mean_field_export.empty() || input.mean_field_export == "none")
        return;

    const std::string record_path = directory + "/" + input.prefix + ".save/mean-field.q3mf";
    try {
        write_mean_field_record(record, record_path);
        log::info("     Wrote mean-field record: {}", record_path);
    } catch (const std::exception& error) {
        log::warn("     mean-field record not written ({}); the exported files above are "
                  "unaffected, but `q3_utility convert` will not be able to re-convert this run.",
                  error.what());
    }
}

/**
 * Topology-aware entry point -- the one both flavors call.
 *
 * The exporter serializes the state it can see; it has no gather step. Under
 * `q3_mpi`, EVERY multi-rank topology distributes that state, so no single
 * rank holds a complete mean field. Both halves were established empirically
 * by running a 2-rank silicon export (2x2x2 shifted grid, 2 IBZ k-points) and
 * reading the files back:
 *
 *   - K-point pools (number_of_pools() > 1) distribute k-points: each pool
 *     holds only its own. Measured: a 2-pool run wrote a WELL-FORMED WFN with
 *     1 of 2 k-points and its weight renormalized to 1.0 -- exactly the shape
 *     a BerkeleyGW consumer cannot detect.
 *
 *   - Band groups (group_size() > 1) distribute the plane-wave coefficients
 *     themselves: each rank stores a G-slab of every band, not the full
 *     sphere. Measured: rank 0 held 809 of silicon's 1546 plane waves at
 *     k-point 1, and the pre-guard export wrote those 809-coefficient slabs
 *     into full-length records with both ranks interleaving onto the same
 *     file -- corrupted Fortran framing, with the run still exiting 0.
 *
 * So the export happens only when one rank holds everything: exactly one pool
 * AND exactly one rank in it. Serial policies report precisely that, so the
 * serial binaries take the writing path unconditionally and there is exactly
 * one code path to reason about. Everything else refuses loudly. The checkpoint
 * is the way around it: with one rank per pool every rank holds its k-points
 * whole, so a pooled SCF writes the same `<prefix>.save/` a serial one does
 * (see write_checkpoints()), and a serial non-self-consistent run restores it
 * and exports. The export itself still has no gather, which is why it does not
 * follow the checkpoint onto pools.
 *
 * The refusal is decided from topology data that is IDENTICAL on every rank,
 * before any collective operation. That is load-bearing: `build_record` runs
 * a rho_r_to_g FFT that is a pool-collective under MPI, so any structure in
 * which ranks disagree about reaching it deadlocks (measured, the first
 * version of this guard did).
 *
 * `build_record` is a callable so refusing topologies skip the record
 * assembly entirely rather than building a record nobody may write.
 */
template <typename RecordBuilder>
inline void export_and_record_mean_field_under_topology(
    const execution::PlanewaveExecutionPolicy& execution_policy,
    RecordBuilder&& build_record,
    const InputData& input,
    const std::string& directory) {
    const bool one_rank_holds_the_complete_mean_field =
        execution_policy.one_process_owns_complete_state();

    if (!one_rank_holds_the_complete_mean_field) {
        const bool export_was_requested =
            !input.mean_field_export.empty() && input.mean_field_export != "none";
        if (execution_policy.is_root() && export_was_requested) {
            log::error("mean_field_export is not supported under MPI: k-point pools hold only "
                       "their own k-points and band groups hold only a G-vector slab of each "
                       "band, so no rank sees the complete mean field. Nothing was exported. "
                       "Request it from a serial non_self_consistent_field run that restores "
                       "the checkpoint (written when every rank has its own k-point pool), or "
                       "run the calculation with a serial binary.");
        }
        return;
    }

    export_and_record_mean_field(std::forward<RecordBuilder>(build_record)(), input, directory);
}

} // namespace q3
