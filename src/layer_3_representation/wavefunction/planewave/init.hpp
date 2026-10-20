#pragma once

/**
 * Plane-wave Kohn-Sham wavefunction.
 *
 * A wavefunction represented as coefficients on a G-vector grid
 * (plane-wave expansion). Inherits from ReciprocalSpaceWavefunction
 * for the abstract interface and OnGridFunctionMixin for the
 * grid-based storage.
 *
 * ψ(r) = Σ_G c_G · e^{i(k+G)·r}
 *
 * The coefficients c_G are Complex values stored at each G-vector
 * point in the GVectorGrid.
 */

#include <stdexcept>

#include "layer_0_core/math/on_grid_function_mixin.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_2_quantum_system/core/wavefunction/reciprocal_space.hpp"
#include "layer_3_representation/basis/planewave/internal/g_vectors_grid.hpp"

namespace q3 {

/// A single wavefunction in the plane-wave basis: c_G coefficients on a GVectorGrid.
class PlanewaveWavefunction : public ReciprocalSpaceWavefunction,
                              public OnGridFunctionMixin<GVectorsGrid, Complex> {
  public:
    PlanewaveWavefunction() = default;

    [[nodiscard]] int number_of_components() const override {
        return 1;
    }

    Complex evaluate(const ReciprocalSpaceVector& g_vector) const override {
        (void)g_vector;
        throw std::runtime_error("PlanewaveWavefunction::evaluate() is not yet implemented. "
                                 "Use index-based access instead.");
    }

    /// Number of plane-wave coefficients.
    [[nodiscard]] int number_of_plane_waves() const {
        return number_of_grid_points();
    }

    /// Raw coefficient data pointer (for BLAS/LAPACK interop).
    Complex* coefficient_data() {
        return values_.data();
    }
    [[nodiscard]] const Complex* coefficient_data() const {
        return values_.data();
    }

    /// Resize the coefficient storage.
    void resize(int number_of_plane_waves) {
        values_.resize(number_of_plane_waves, Complex(0.0, 0.0));
    }
};

/// Pseudopotential plane-wave Kohn-Sham wavefunction (same as base PW for now).
using NormConservingPlanewaveWavefunction = PlanewaveWavefunction;

/// Ultrasoft pseudopotential plane-wave Kohn-Sham wavefunction (same as base PW for now).
using UltrasoftPlanewaveWavefunction = NormConservingPlanewaveWavefunction;

} // namespace q3
