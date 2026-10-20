#pragma once

/**
 * WannierHamiltonianBuilder — the Wannier method: captured interface
 * matrices to a local-orbital Hamiltonian H(R), with a report of how.
 *
 * Four steps, and only the middle two are Wannier's own. The inputs are
 * checked and their neighbour geometry resolved once
 * (WannierConstructionInputs). A SubspaceSelection chooses the N states the
 * gauge is built from -- the window as it stands, or a subspace
 * disentangled out of a wider one -- and hands them over as frames in the
 * projection gauge. A GaugeChoice fixes the gauge inside them: kept, or
 * rotated to minimise the spread. Rotating the eigenvalues by the final
 * frames gives H^W(k) = C^dagger E C at each mesh point, and transforming
 * that to real space is a lattice Fourier sum that belongs to nobody in
 * particular, so it is layer 1's (lattice_fourier_transform), taken over the
 * Wigner-Seitz supercell of the k-mesh so that boundary vectors carry their
 * degeneracy weight.
 *
 * The builder holds the settings and nothing else; which strategy runs is
 * decided by the two factories, so no step here branches on what was asked
 * for. Nothing here touches a basis either: it works on any inputs of the
 * right shape -- Q3's own, or a set imported from another code's
 * Wannier90-format files -- which is what lets the method be validated
 * against Quantum ESPRESSO's matrices independently of Q3's plane-wave
 * kernels.
 */

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "layer_0_core/logging/init.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix.hpp"
#include "layer_0_core/math/linear_algebra/dense_matrix_product.hpp"
#include "layer_0_core/math/linear_algebra/matrix_operation.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/reciprocal_space/lattice_fourier_transform.hpp"
#include "layer_1_structure/reciprocal_space/wigner_seitz_supercell.hpp"
#include "layer_3_representation/local_orbital/hamiltonian_decay.hpp"
#include "layer_3_representation/local_orbital/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction.hpp"
#include "layer_4_computation/interpolator/method/wannier/construction_inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/factory.hpp"
#include "layer_4_computation/interpolator/method/wannier/gauge/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/inputs.hpp"
#include "layer_4_computation/interpolator/method/wannier/internal/band_classification.hpp"
#include "layer_4_computation/interpolator/method/wannier/report/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/settings/init.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/factory.hpp"
#include "layer_4_computation/interpolator/method/wannier/subspace/init.hpp"

namespace q3 {

class WannierHamiltonianBuilder {
  public:
    /// The defaults are an isolated manifold in the projection gauge: what a record alone supports.
    explicit WannierHamiltonianBuilder(WannierConstructionSettings settings = {})
        : settings_(std::move(settings)) {}

    /**
     * `real_lattice` rows are a1, a2, a3 in any consistent length unit; only
     * the Wigner-Seitz geometry depends on it, and that is unit-free.
     */
    [[nodiscard]] WannierConstruction build(const WannierInputs& record,
                                            const Matrix3x3& real_lattice) const {
        const WannierConstructionInputs inputs(record);
        const std::unique_ptr<SubspaceSelection> selection = create_subspace_selection(settings_);
        const std::unique_ptr<GaugeChoice> choice = create_gauge_choice(settings_, inputs);

        const WannierSubspace subspace = selection->choose(inputs);
        const WannierGauge gauge = choice->choose(inputs, subspace);

        WannierConstruction construction;
        construction.report.disentanglement = subspace.report;
        construction.report.spread = gauge.spread;
        construction.hamiltonian =
            real_space_hamiltonian(inputs, subspace.classification, gauge.frames, real_lattice);
        construction.report.hamiltonian_decay =
            HamiltonianDecay::measured_on(construction.hamiltonian);
        return construction;
    }

  private:
    /**
     * H^W(k) = C(k)^dagger E(k) C(k), one N x N block per k-point, summed to
     * H(R) over the Wigner-Seitz supercell of the k-mesh.
     */
    [[nodiscard]] static LocalOrbitalHamiltonian
    real_space_hamiltonian(const WannierConstructionInputs& inputs,
                           const BandClassification& classification,
                           const std::vector<DenseMatrix<Complex>>& frames,
                           const Matrix3x3& real_lattice) {
        const WannierInputs& record = inputs.record();
        const int number_of_kpoints = inputs.number_of_kpoints();

        WignerSeitzSupercell supercell;
        supercell.compute(real_lattice,
                          record.kpoint_grid_dimensions[0],
                          record.kpoint_grid_dimensions[1],
                          record.kpoint_grid_dimensions[2]);
        log::info("Wigner-Seitz supercell: {} lattice vectors, total weight {:.6f} (expected {})",
                  supercell.number_of_lattice_vectors(),
                  supercell.total_weight(),
                  number_of_kpoints);

        std::vector<LocalOrbitalLatticeVector> domain;
        std::vector<IntegerVector3D> translations;
        domain.reserve(supercell.lattice_vectors().size());
        translations.reserve(supercell.lattice_vectors().size());
        for (const WignerSeitzLatticeVector& entry : supercell.lattice_vectors()) {
            domain.push_back({entry.lattice_translation,
                              entry.cartesian,
                              1.0 / static_cast<Real>(entry.degeneracy)});
            translations.push_back(entry.lattice_translation);
        }

        std::vector<std::vector<Complex>> hamiltonian_at_kpoints;
        hamiltonian_at_kpoints.reserve(static_cast<std::size_t>(number_of_kpoints));
        for (int kpoint_index = 0; kpoint_index < number_of_kpoints; kpoint_index++) {
            const KPointBandClassification& entry =
                classification.kpoints[static_cast<std::size_t>(kpoint_index)];
            std::vector<Real> kept_eigenvalues;
            kept_eigenvalues.reserve(entry.kept_bands.size());
            for (const int band : entry.kept_bands) {
                kept_eigenvalues.push_back(record.eigenvalues_rydberg[kpoint_index][band]);
            }
            const DenseMatrix<Complex>& frame = frames[static_cast<std::size_t>(kpoint_index)];
            hamiltonian_at_kpoints.push_back(multiply(MatrixOperation::ConjugateTranspose,
                                                      frame,
                                                      MatrixOperation::None,
                                                      frame.scaled_rows(kept_eigenvalues))
                                                 .raw());
        }

        LocalOrbitalHamiltonian hamiltonian;
        hamiltonian.set_domain(record.number_of_trial_orbitals, std::move(domain));
        hamiltonian.set_hamiltonian(lattice_fourier_transform::to_real_space(
            record.kpoint_coordinates, hamiltonian_at_kpoints, translations));
        return hamiltonian;
    }

    WannierConstructionSettings settings_;
};

} // namespace q3
