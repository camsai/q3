#pragma once

/**
 * PseudopotentialOverlapOperator — norm-conserving overlap: S = I.
 *
 * Norm-conserving pseudopotential_set leave the plane-wave basis orthonormal, so the
 * overlap is the identity and the eigenproblem is standard (zheev). This serves as
 * the base for pseudopotential overlap operators; the ultrasoft subclass adds the
 * augmentation overlap on top. The crystal, quantum system snapshot, and
 * Kleinman-Bylander projectors are held here so subclasses can use them without
 * re-plumbing call sites.
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "io/logger.hpp"
#include "layer_0_core/execution/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_0_core/types/matrix.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/pseudopotential/planewave/kleinman_bylander_projector_set.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/init.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/overlap/overlap_block_gpu_copied.hpp"

namespace q3 {

class PseudopotentialOverlapOperator : public OverlapOperator {
  public:
    PseudopotentialOverlapOperator(
        const Crystal& crystal_structure,
        const NormConservingPlanewaveKohnShamQuantumSystem& quantum_system,
        const KleinmanBylanderProjectorSet& kleinman_bylander_projector_set,
        const execution::ExecutionPolicy& execution_policy)
        : crystal_structure_(crystal_structure)
        , quantum_system_(quantum_system)
        , kleinman_bylander_projector_set_(kleinman_bylander_projector_set)
        , execution_policy_(execution_policy) {}

    void apply(int number_of_plane_waves,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        for (int g_vector_index = 0; g_vector_index < number_of_plane_waves; g_vector_index++) {
            overlap_wavefunction[g_vector_index] = wavefunction[g_vector_index];
        }
    }

    [[nodiscard]] bool is_identity() const override {
        return true;
    }

    /// S = 1: the overlap block is a device copy of the vectors.
    [[nodiscard]] std::unique_ptr<GPUOverlapBlock>
    create_gpu_overlap_block(execution::gpu::device::GPUDeviceBackend& device_backend,
                             int number_of_plane_waves,
                             int number_of_vectors,
                             const Complex* vectors,
                             Complex* overlap_vectors) const override {
        return std::make_unique<CopiedGPUOverlapBlock>(
            device_backend,
            static_cast<std::size_t>(number_of_plane_waves) * number_of_vectors,
            vectors,
            overlap_vectors);
    }

    /**
     * Build the dense overlap matrix S for the norm-conserving case.
     *
     * For norm-conserving pseudopotentials, the plane-wave basis is orthonormal
     * so the overlap matrix is the identity: S = I.
     */
    void build_dense_matrix(int number_of_plane_waves,
                            std::vector<Complex>& overlap_matrix) const override {
        overlap_matrix.assign(number_of_plane_waves * number_of_plane_waves, Complex(0.0, 0.0));
        for (int index = 0; index < number_of_plane_waves; index++) {
            overlap_matrix[index + index * number_of_plane_waves] = Complex(1.0, 0.0);
        }
    }

  protected:
    const Crystal& crystal_structure_;
    const NormConservingPlanewaveKohnShamQuantumSystem& quantum_system_;
    const KleinmanBylanderProjectorSet& kleinman_bylander_projector_set_;
    const execution::ExecutionPolicy& execution_policy_;
};

} // namespace q3
