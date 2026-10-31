#pragma once

/**
 * Flavor — abstract interface for a Q3 calculation mode.
 *
 * A Flavor encapsulates a complete calculation workflow (SCF, NSCF, etc.)
 * and returns its results as results.json's own struct, generated::Results.
 */

#include "io/input/data.hpp"
#include "io/output/material/init.hpp"
#include "io/output/results.hpp"
#include "io/provenance/factory.hpp"
#include "layer_5_properties/band_structure/band_edges.hpp"

namespace q3 {

class Flavor {
  public:
    virtual ~Flavor() = default;

    /**
     * Execute the calculation described by the input and return results,
     * stamped with the provenance of the run that produced them, the material
     * it ran on, and the band edges and gap its eigenvalues give.
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
    generated::Results run(const InputData& input) {
        generated::Results results = run_calculation(input);
        if (nlohmann::json provenance = build_provenance_json(input); !provenance.is_null()) {
            results.provenance = std::move(provenance);
        }
        if (nlohmann::json material = build_material_json(input); !material.is_null()) {
            results.structure = generated::ResultsStructure{std::move(material)};
        }
        add_band_edge_records(results);
        return results;
    }

  protected:
    /**
     * Execute the calculation described by the input and return results.
     *
     * Implementations must be self-contained: they set up, run, and finalize
     * the calculation entirely. The caller receives fully-populated results,
     * minus what run() adds.
     */
    virtual generated::Results run_calculation(const InputData& input) = 0;
};

} // namespace q3
