#pragma once

/**
 * LocalOrbitalHamiltonian — the Hamiltonian of a periodic system expressed in
 * a basis of localized orbitals: one N x N block H(R) per lattice vector R,
 * plus the overlap S(R) of the orbitals when they are not orthonormal.
 *
 * This is the second REPRESENTATION beside the plane-wave one, and it is
 * named for the basis, not for how the orbitals were obtained. Wannier
 * functions, numerical atomic orbitals, Gaussian-type orbitals and
 * Slater-Koster tight-binding parameters all produce an object of exactly
 * this shape, and everything that consumes it -- Bloch sums, interpolated
 * bands, densities of states, transport -- is the same for all of them.
 * Nothing here knows about plane waves, gauges, or k-meshes.
 *
 * Two conventions are fixed by this class:
 *
 *  - k-points are CRYSTAL (fractional) coordinates, so the Bloch phase is
 *    e^{+2 pi i k.R} with the integer translation R directly; no Cartesian
 *    conversion is ever needed.
 *  - Each lattice vector carries a weight applied in the Bloch sum. For a
 *    Hamiltonian obtained by real-space integrals the weight is 1. For one
 *    obtained by Fourier transform from a k-mesh, vectors on the boundary
 *    of the Wigner-Seitz supercell are shared between periodic images and
 *    carry 1/degeneracy; keeping that weight with the vector rather than
 *    folded into the block is what lets the same class hold both kinds.
 *
 * The overlap is optional. Wannier functions are orthonormal by
 * construction, so S(R) = delta_{R,0} I and the Bloch sum returns the
 * identity; a non-orthogonal basis sets its S(R) blocks and the eigenproblem
 * downstream becomes the generalized one, H C = e S C -- the same shape the
 * plane-wave representation already solves for ultrasoft pseudopotentials.
 */

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/lattice_fourier_transform.hpp"

namespace q3 {

/// One lattice vector of the Hamiltonian's real-space domain.
struct [[nodiscard]] LocalOrbitalLatticeVector {
    IntegerVector3D translation;
    /// Cartesian R in whatever length unit the caller used; decay diagnostics only.
    Vector3D cartesian;
    /**
     * Weight applied in the Bloch sum: 1 for a block obtained from real-space
     * integrals, 1/degeneracy for a Wigner-Seitz boundary vector shared by
     * that many periodic images.
     */
    Real bloch_sum_weight = 1.0;
};

class LocalOrbitalHamiltonian {
  public:
    /**
     * Fix the basis size and the real-space domain. Any blocks previously
     * set are discarded, since their shape and order are no longer defined.
     */
    void set_domain(int number_of_orbitals,
                    std::vector<LocalOrbitalLatticeVector> lattice_vectors) {
        if (number_of_orbitals <= 0) {
            throw std::invalid_argument(
                "LocalOrbitalHamiltonian::set_domain: number_of_orbitals must be positive");
        }
        if (lattice_vectors.empty()) {
            throw std::invalid_argument(
                "LocalOrbitalHamiltonian::set_domain: the domain must hold at least one lattice "
                "vector");
        }
        number_of_orbitals_ = number_of_orbitals;
        lattice_vectors_ = std::move(lattice_vectors);
        hamiltonian_blocks_.clear();
        overlap_blocks_.clear();

        /**
         * The Bloch sum takes the translations and weights as parallel arrays.
         * Unpacking them once here rather than per call keeps the dense-mesh
         * sampling path -- tens of thousands of calls -- allocation-free.
         */
        translations_.clear();
        bloch_sum_weights_.clear();
        translations_.reserve(lattice_vectors_.size());
        bloch_sum_weights_.reserve(lattice_vectors_.size());
        for (const LocalOrbitalLatticeVector& entry : lattice_vectors_) {
            translations_.push_back(entry.translation);
            bloch_sum_weights_.push_back(entry.bloch_sum_weight);
        }
    }

    /**
     * H(R): one column-major number_of_orbitals x number_of_orbitals block
     * per lattice vector, in the domain's order.
     */
    void set_hamiltonian(std::vector<std::vector<Complex>> blocks) {
        require_block_shapes(blocks, "set_hamiltonian");
        hamiltonian_blocks_ = std::move(blocks);
    }

    /**
     * S(R), same layout as the Hamiltonian blocks. Not setting it means the
     * orbitals are orthonormal and S(k) is the identity for every k.
     */
    void set_overlap(std::vector<std::vector<Complex>> blocks) {
        require_block_shapes(blocks, "set_overlap");
        overlap_blocks_ = std::move(blocks);
    }

    [[nodiscard]] bool has_nontrivial_overlap() const {
        return !overlap_blocks_.empty();
    }

    [[nodiscard]] int number_of_orbitals() const {
        return number_of_orbitals_;
    }

    [[nodiscard]] const std::vector<LocalOrbitalLatticeVector>& lattice_vectors() const {
        return lattice_vectors_;
    }

    [[nodiscard]] const std::vector<std::vector<Complex>>& hamiltonian_blocks() const {
        return hamiltonian_blocks_;
    }

    [[nodiscard]] const std::vector<std::vector<Complex>>& overlap_blocks() const {
        return overlap_blocks_;
    }

    /**
     * H(k) = sum_R w_R e^{+2 pi i k.R} H(R), column-major, for `kpoint` in
     * crystal coordinates. `kpoint` need not lie on any mesh -- evaluating
     * this anywhere is what the representation is for.
     */
    [[nodiscard]] std::vector<Complex> hamiltonian_at_kpoint(const Vector3D& kpoint) const {
        if (hamiltonian_blocks_.empty()) {
            throw std::runtime_error(
                "LocalOrbitalHamiltonian::hamiltonian_at_kpoint: no Hamiltonian blocks set");
        }
        return bloch_sum(hamiltonian_blocks_, kpoint);
    }

    /// S(k) by the same Bloch sum, or the identity when the basis is orthonormal.
    [[nodiscard]] std::vector<Complex> overlap_at_kpoint(const Vector3D& kpoint) const {
        if (overlap_blocks_.empty()) {
            std::vector<Complex> identity(static_cast<std::size_t>(number_of_orbitals_) *
                                              number_of_orbitals_,
                                          Complex(0.0, 0.0));
            for (int index = 0; index < number_of_orbitals_; index++) {
                identity[static_cast<std::size_t>(index) * number_of_orbitals_ + index] =
                    Complex(1.0, 0.0);
            }
            return identity;
        }
        return bloch_sum(overlap_blocks_, kpoint);
    }

    /**
     * Largest |H(R)| over the blocks whose lattice vector is at least
     * `minimum_distance` from the origin in Cartesian length -- the direct
     * measure of how fast the Hamiltonian decays, and therefore of how far
     * a Bloch sum truncated to this domain can be trusted between the
     * k-points it was built from.
     */
    [[nodiscard]] Real maximum_hamiltonian_magnitude_beyond(Real minimum_distance) const {
        if (hamiltonian_blocks_.size() != lattice_vectors_.size()) {
            throw std::runtime_error("LocalOrbitalHamiltonian::maximum_hamiltonian_magnitude_"
                                     "beyond: no Hamiltonian blocks set");
        }
        Real worst = 0.0;
        for (std::size_t vector_index = 0; vector_index < lattice_vectors_.size(); vector_index++) {
            if (lattice_vectors_[vector_index].cartesian.norm() < minimum_distance) {
                continue;
            }
            for (const Complex& element : hamiltonian_blocks_[vector_index]) {
                worst = std::max(worst, std::abs(element));
            }
        }
        return worst;
    }

  private:
    [[nodiscard]] std::vector<Complex> bloch_sum(const std::vector<std::vector<Complex>>& blocks,
                                                 const Vector3D& kpoint) const {
        return lattice_fourier_transform::to_kpoint(
            kpoint, blocks, translations_, bloch_sum_weights_);
    }

    void require_block_shapes(const std::vector<std::vector<Complex>>& blocks,
                              const char* what) const {
        if (lattice_vectors_.empty()) {
            throw std::runtime_error(std::string("LocalOrbitalHamiltonian::") + what +
                                     ": set_domain() has not been called");
        }
        if (blocks.size() != lattice_vectors_.size()) {
            throw std::invalid_argument(std::string("LocalOrbitalHamiltonian::") + what + ": " +
                                        std::to_string(blocks.size()) + " blocks were given for " +
                                        std::to_string(lattice_vectors_.size()) +
                                        " lattice vectors");
        }
        const std::size_t block_size =
            static_cast<std::size_t>(number_of_orbitals_) * number_of_orbitals_;
        for (std::size_t vector_index = 0; vector_index < blocks.size(); vector_index++) {
            if (blocks[vector_index].size() != block_size) {
                throw std::invalid_argument(std::string("LocalOrbitalHamiltonian::") + what +
                                            ": block " + std::to_string(vector_index) + " holds " +
                                            std::to_string(blocks[vector_index].size()) +
                                            " entries, not " + std::to_string(number_of_orbitals_) +
                                            " x " + std::to_string(number_of_orbitals_));
            }
        }
    }

    int number_of_orbitals_ = 0;
    std::vector<LocalOrbitalLatticeVector> lattice_vectors_;
    /// The domain unpacked for lattice_fourier_transform::to_kpoint; see set_domain.
    std::vector<IntegerVector3D> translations_;
    std::vector<Real> bloch_sum_weights_;
    std::vector<std::vector<Complex>> hamiltonian_blocks_;
    std::vector<std::vector<Complex>> overlap_blocks_;
};

} // namespace q3
