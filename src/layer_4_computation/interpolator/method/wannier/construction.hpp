#pragma once

/**
 * WannierConstruction — what WannierHamiltonianBuilder produces: the
 * real-space Hamiltonian, and the numbers a driver needs to decide whether to
 * keep it.
 */

#include "layer_3_representation/local_orbital/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"

namespace q3 {

struct [[nodiscard]] WannierConstruction {
    LocalOrbitalHamiltonian hamiltonian;
    WannierConstructionReport report;
};

} // namespace q3
