#pragma once

/**
 * LocalOrbitalQuantumStateInterpolator — evaluates any model that is a
 * local-orbital Hamiltonian: Bloch sum to H(k), then diagonalize.
 *
 * Named for the model it reads, not for the method that built it. Wannier
 * functions, projected atomic orbitals and Shirley's optimal basis all produce
 * an H(R) of exactly this shape, so they share this evaluator; a method that
 * produces no Hamiltonian -- a Fourier star-function fit of the eigenvalues --
 * gets its own file beside this one.
 *
 * It owns its Hamiltonian rather than referencing one, so an interpolator
 * outlives whatever built it. That matters: the model is built inside the
 * factory and the caller keeps only the interface.
 *
 * The eigenproblem is the dense generalized one the plane-wave side already
 * solves for ultrasoft pseudopotentials, so it goes through the same
 * SubspaceDiagonalizer rather than a second LAPACK wrapper. S(k) is the
 * identity for an orthonormal basis, which the Hamiltonian returns for a model
 * with no overlap blocks, and Wannier functions are orthonormal by
 * construction.
 */

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/execution/serial.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_3_representation/local_orbital/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/subspace/factory.hpp"
#include "layer_4_computation/interpolator/target/quantum_state/init.hpp"

namespace q3 {

class LocalOrbitalQuantumStateInterpolator : public QuantumStateInterpolator {
  public:
    explicit LocalOrbitalQuantumStateInterpolator(LocalOrbitalHamiltonian hamiltonian,
                                                  std::string method_description = "local-orbital")
        : hamiltonian_(std::move(hamiltonian))
        , method_description_(std::move(method_description))
        , diagonalizer_(SubspaceDiagonalizerFactory::create(serial_execution_policy)) {}

    [[nodiscard]] int number_of_bands() const override {
        return hamiltonian_.number_of_orbitals();
    }

    [[nodiscard]] std::vector<Real> eigenvalues_at(const Vector3D& kpoint) const override {
        const int dimension = hamiltonian_.number_of_orbitals();
        std::vector<Complex> hamiltonian_at_kpoint = hamiltonian_.hamiltonian_at_kpoint(kpoint);
        std::vector<Complex> overlap_at_kpoint = hamiltonian_.overlap_at_kpoint(kpoint);
        std::vector<Real> eigenvalues(static_cast<std::size_t>(dimension), 0.0);
        /// zhegv wants somewhere to put the vectors even when only values are read back.
        std::vector<Complex> eigenvectors(static_cast<std::size_t>(dimension) * dimension,
                                          Complex(0.0, 0.0));

        const int status = diagonalizer_->solve(dimension,
                                                hamiltonian_at_kpoint.data(),
                                                overlap_at_kpoint.data(),
                                                eigenvalues.data(),
                                                eigenvectors.data());
        if (status != 0) {
            throw std::runtime_error(
                "LocalOrbitalQuantumStateInterpolator::eigenvalues_at: the eigensolver failed with "
                "status " +
                std::to_string(status) +
                "; a positive status above the basis size means the overlap S(k) is not positive "
                "definite, i.e. the model's orbitals are linearly dependent at this k-point");
        }
        return eigenvalues;
    }

    [[nodiscard]] std::string description() const override {
        return method_description_ + ": " + std::to_string(hamiltonian_.number_of_orbitals()) +
               " orbitals on " + std::to_string(hamiltonian_.lattice_vectors().size()) +
               " lattice vectors";
    }

    /**
     * The total Bloch-sum weight. For a Hamiltonian from a k -> R transform it
     * must equal the number of k-points the transform ran over (the
     * Wigner-Seitz degeneracy identity), which is the one check on the model a
     * consumer can make without knowing how it was built.
     */
    [[nodiscard]] std::map<std::string, Real> diagnostics() const override {
        Real total_weight = 0.0;
        for (const LocalOrbitalLatticeVector& vector : hamiltonian_.lattice_vectors()) {
            total_weight += vector.bloch_sum_weight;
        }
        return {{TOTAL_BLOCH_SUM_WEIGHT, total_weight}};
    }

    /// The name that diagnostic goes by, which a results file gives a key of its own.
    static constexpr const char* TOTAL_BLOCH_SUM_WEIGHT = "total_bloch_sum_weight";

    /// The model itself, for diagnostics (decay of H(R), the R set) and tests.
    [[nodiscard]] const LocalOrbitalHamiltonian& hamiltonian() const {
        return hamiltonian_;
    }

  private:
    inline static const execution::SerialExecutionPolicy serial_execution_policy{};

    LocalOrbitalHamiltonian hamiltonian_;
    std::string method_description_;
    std::unique_ptr<SubspaceDiagonalizer> diagonalizer_;
};

} // namespace q3
