#pragma once

/**
 * SpreadTerms — Omega split into the part no gauge can change and the parts a gauge
 * can, with the centre and the spread of every function.
 */

#include <vector>

#include "layer_0_core/types/init.hpp"

namespace q3 {

struct [[nodiscard]] SpreadTerms {
    Real gauge_invariant_bohr_squared = 0.0;
    Real diagonal_bohr_squared = 0.0;
    Real off_diagonal_bohr_squared = 0.0;
    /// One centre per Wannier function, in bohr, in the same Cartesian frame as b.
    std::vector<Vector3D> centers_bohr;
    /**
     * One spread per Wannier function, Omega_n = <r^2>_n - |<r>_n|^2 in bohr
     * squared (MV equations 31 and 32 in finite differences: <r^2>_n is
     * (1/N_k) sum_kb w_b [(1 - |M_nn|^2) + (Im ln M_nn)^2]). They sum to the
     * total, which is how the split above and this one are checked against
     * each other. The three totals say how localised the set is; these say
     * which function is not, and a driver that has no reference bands to
     * compare against judges a model by their spread across functions.
     */
    std::vector<Real> spreads_bohr_squared;

    [[nodiscard]] Real total_bohr_squared() const {
        return gauge_invariant_bohr_squared + diagonal_bohr_squared + off_diagonal_bohr_squared;
    }
    /// The only part this gauge can move; what the minimiser actually descends.
    [[nodiscard]] Real minimizable_bohr_squared() const {
        return diagonal_bohr_squared + off_diagonal_bohr_squared;
    }
};

} // namespace q3
