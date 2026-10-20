#pragma once

/**
 * Flavor — abstract interface for a Q3 calculation mode.
 *
 * A Flavor encapsulates a complete calculation workflow (SCF, NSCF, etc.)
 * and returns a FlavorResult with all computed quantities as raw types.
 */

#include "io/input/data.hpp"
#include "io/output/material/init.hpp"
#include "io/provenance/factory.hpp"
#include "layer_4_computation/flavors/result.hpp"

namespace q3 {

class Flavor {
  public:
    virtual ~Flavor() = default;

    /**
     * Execute the calculation described by the input and return results,
     * stamped with the provenance of the run that produced them.
     *
     * Not virtual, and the reason is the stamp. Provenance attached at the call
     * sites instead would mean every present and future caller had to remember
     * to attach it, and the one that forgot would emit a results file
     * indistinguishable from an unattributable one. Here it is structural: a
     * flavor cannot run without being attributed.
     *
     * Re-reading the pseudopotential files to identify them costs milliseconds
     * against a calculation that has already finished, which is the right trade
     * for a subclass never having to carry provenance plumbing of its own.
     */
    FlavorResult run(const InputData& input) {
        FlavorResult result = run_calculation(input);
        result.provenance = build_provenance_json(input);
        result.material = build_material_json(input);
        return result;
    }

  protected:
    /**
     * Execute the calculation described by the input and return results.
     *
     * Implementations must be self-contained: they set up, run, and finalize
     * the calculation entirely. The caller receives a fully-populated
     * FlavorResult, minus the provenance run() adds.
     */
    virtual FlavorResult run_calculation(const InputData& input) = 0;
};

} // namespace q3
