#pragma once

/**
 * create_quantum_state_interpolator — the evaluation-side factory: an
 * InterpolationRecord and the evaluation manifest's specification in, an
 * interpolator of the electronic state and the report of its construction
 * out.
 *
 * The record names its method, so the caller -- the utility that evaluates a
 * saved model, or a test -- never branches on it, and the specification is
 * the method-agnostic block every method reads its own knobs from. Adding a
 * method is one payload member on the record, one construction function
 * under method/, one report member on the construction, and one case here;
 * every consumer of QuantumStateInterpolator then serves it unchanged. Mirrors
 * create_mean_field_exporter(), minus the Null object: there is no "no interpolation" interpolator
 * to hand out, since a record only exists when one was asked for.
 *
 * The two axes meet here and only here. The method builds a model from the
 * record; the target evaluates that model. For every method that yields a
 * local-orbital H(R) -- Wannier now, projected atomic orbitals and Shirley's
 * basis later -- the evaluator is the same one, which is why it is selected
 * here rather than owned by the method.
 *
 * Capture, the opposite direction, is deliberately not in this file: it needs
 * the plane-wave SCF iterator, and evaluating a saved model must not. See
 * capture.hpp.
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "layer_4_computation/interpolator/construction.hpp"
#include "layer_4_computation/interpolator/input_data/record.hpp"
#include "layer_4_computation/interpolator/input_data/specification.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction.hpp"
#include "layer_4_computation/interpolator/method/wannier/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/init.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/local_orbital.hpp"

namespace q3 {

/**
 * A specification that says nothing builds the record as the manifest's
 * defaults ask: an isolated manifold, localised where the record can support
 * it.
 */
[[nodiscard]] inline QuantumStateInterpolatorConstruction
create_quantum_state_interpolator(const InterpolationRecord& record,
                                  const InterpolationSpecification& specification = {}) {
    if (record.method == "wannier") {
        WannierConstruction construction =
            WannierHamiltonianBuilder(WannierConstructionSettings::from(specification))
                .build(record.wannier, record.real_lattice_bohr);
        QuantumStateInterpolatorConstruction result;
        result.interpolator = std::make_unique<LocalOrbitalQuantumStateInterpolator>(
            std::move(construction.hamiltonian), construction.report.description());
        result.wannier = std::move(construction.report);
        return result;
    }
    throw std::runtime_error("create_quantum_state_interpolator: unknown interpolation method '" +
                             record.method + "'. Supported: wannier");
}

} // namespace q3
