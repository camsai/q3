#pragma once

/**
 * GPUStructureFactorCalculator — the phase sums of StructureFactorCalculator on a device.
 *
 * Both sums are a matrix of phases times a small matrix of weights. A kernel forms the
 * phases E[j, a] = e^{∓iG_j·τ_a} on the device, for a block of the G-vectors at a time,
 * and the device BLAS multiplies:
 *
 *   structure factors:  S (G-vectors × sets) = E (G-vectors × atoms) · Q (atoms × sets)
 *   gradient sums:      Y (atoms × 3·sets)   = Eᵀ (atoms × G-vectors) · X (G-vectors × 3·sets)
 *
 * with X[j, 3k + α] = G_{j,α} · phase_weights[k][j], and F[a][α] = Im Y[a, 3·set_of_atom[a] + α].
 * What goes up per call is the G-vectors' Cartesian components and the weights; what comes
 * down is S, or Y, which are small. The phases, one per atom and G-vector and the only part
 * whose cost grows with both, never exist on the host.
 *
 * The blocks keep the phase matrix within a fixed number of elements, whatever the cell. The
 * workspaces are kept between calls and shared, so calls are serialized.
 */

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_0_core/execution/gpu/device/internal/buffer.hpp"
#include "layer_0_core/math/linear_algebra/gpu/blas/init.hpp"
#include "layer_0_core/types/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/basis/planewave/structure_factor_calculator.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

namespace q3 {

/// Forms the phases on a device and multiplies them there; see the file comment.
class GPUStructureFactorCalculator : public StructureFactorCalculator {
  public:
    /// Largest phase matrix formed at once by default, in elements (256 MiB of complex doubles).
    static constexpr std::size_t default_phase_block_elements = std::size_t{1} << 24;

    /**
     * @param crystal_structure  The atoms; positions are read once, here.
     * @param planewave_basis    Whose G-vectors the indices name. Must outlive this object.
     * @param device_backend     The device the sums run on. Must not be null.
     * @param device_blas        The device BLAS of that backend. Must outlive this object.
     * @param phase_block_elements  Largest phase matrix formed at once, in elements.
     */
    GPUStructureFactorCalculator(
        const Crystal& crystal_structure,
        const PlanewaveBasis& planewave_basis,
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend,
        const execution::gpu::blas::GPUBLAS& device_blas,
        std::size_t phase_block_elements = default_phase_block_elements)
        : planewave_basis_(planewave_basis)
        , number_of_atoms_(crystal_structure.number_of_atoms())
        , phase_block_elements_(std::max<std::size_t>(1, phase_block_elements))
        , device_backend_(require(std::move(device_backend)))
        , device_blas_(device_blas)
        , device_positions_(3 * static_cast<std::size_t>(number_of_atoms_), device_backend_)
        , device_g_cartesian_(0, device_backend_)
        , device_phases_(0, device_backend_)
        , device_weights_(0, device_backend_)
        , device_products_(0, device_backend_) {
        std::vector<Real> positions(3 * static_cast<std::size_t>(number_of_atoms_));
        for (int atom_index = 0; atom_index < number_of_atoms_; atom_index++) {
            const Vector3D position =
                crystal_structure.convert_from_crystal_to_cartesian(atom_index);
            for (int direction = 0; direction < 3; direction++) {
                positions[3 * static_cast<std::size_t>(atom_index) + direction] =
                    position[direction];
            }
        }
        if (!positions.empty()) {
            device_backend_->copy(
                device_positions_.data(), positions.data(), positions.size() * sizeof(Real));
        }
    }

    /// See StructureFactorCalculator::structure_factors(); formed on the device.
    [[nodiscard]] std::vector<std::vector<Complex>>
    structure_factors(const std::vector<int>& g_vector_indices,
                      const std::vector<std::vector<Real>>& atom_weights) const override {
        const std::size_t number_of_sets = atom_weights.size();
        const std::size_t number_of_g_vectors = g_vector_indices.size();
        std::vector<std::vector<Complex>> result(
            number_of_sets, std::vector<Complex>(number_of_g_vectors, Complex(0.0, 0.0)));
        if (number_of_sets == 0 || number_of_g_vectors == 0 || number_of_atoms_ == 0) {
            return result;
        }
        // Q, atoms × sets, column-major.
        std::vector<Complex> weights(static_cast<std::size_t>(number_of_atoms_) * number_of_sets);
        for (std::size_t set = 0; set < number_of_sets; set++) {
            if (atom_weights[set].size() != static_cast<std::size_t>(number_of_atoms_)) {
                throw std::invalid_argument(
                    "GPUStructureFactorCalculator: a set of atom weights must have one entry "
                    "per atom");
            }
            for (int atom_index = 0; atom_index < number_of_atoms_; atom_index++) {
                weights[set * static_cast<std::size_t>(number_of_atoms_) + atom_index] =
                    Complex(atom_weights[set][static_cast<std::size_t>(atom_index)], 0.0);
            }
        }
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        ensure_capacity(device_weights_, weights.size());
        device_backend_->copy(
            device_weights_.data(), weights.data(), weights.size() * sizeof(Complex));

        std::vector<Complex> block_result;
        for_each_block(number_of_g_vectors, [&](std::size_t first, std::size_t count) {
            form_phases(g_vector_indices, first, count, -1.0);
            ensure_capacity(device_products_, count * number_of_sets);
            device_blas_.gemm('N',
                              'N',
                              static_cast<int>(count),
                              static_cast<int>(number_of_sets),
                              number_of_atoms_,
                              Complex(1.0, 0.0),
                              device_phases_.data(),
                              static_cast<int>(count),
                              device_weights_.data(),
                              number_of_atoms_,
                              Complex(0.0, 0.0),
                              device_products_.data(),
                              static_cast<int>(count));
            device_backend_->synchronize();
            block_result.resize(count * number_of_sets);
            device_backend_->copy(block_result.data(),
                                  device_products_.data(),
                                  block_result.size() * sizeof(Complex));
            for (std::size_t set = 0; set < number_of_sets; set++) {
                std::copy(block_result.begin() + static_cast<std::ptrdiff_t>(set * count),
                          block_result.begin() + static_cast<std::ptrdiff_t>((set + 1) * count),
                          result[set].begin() + static_cast<std::ptrdiff_t>(first));
            }
        });
        return result;
    }

    /// See StructureFactorCalculator::gradient_phase_sums(); formed on the device.
    [[nodiscard]] std::vector<Vector3D>
    gradient_phase_sums(const std::vector<int>& g_vector_indices,
                        const std::vector<std::vector<Complex>>& phase_weights,
                        const std::vector<int>& set_of_atom) const override {
        if (set_of_atom.size() != static_cast<std::size_t>(number_of_atoms_)) {
            throw std::invalid_argument(
                "GPUStructureFactorCalculator: set_of_atom must have one entry per atom");
        }
        const std::size_t number_of_sets = phase_weights.size();
        const std::size_t number_of_g_vectors = g_vector_indices.size();
        for (const auto& weights : phase_weights) {
            if (weights.size() != number_of_g_vectors) {
                throw std::invalid_argument(
                    "GPUStructureFactorCalculator: a set of phase weights must have one entry "
                    "per G-vector");
            }
        }
        std::vector<Vector3D> result(static_cast<std::size_t>(number_of_atoms_),
                                     Vector3D{0.0, 0.0, 0.0});
        if (number_of_sets == 0 || number_of_g_vectors == 0 || number_of_atoms_ == 0) {
            return result;
        }
        const std::size_t columns = 3 * number_of_sets;
        const auto& g_vectors = planewave_basis_.g_vectors();
        std::lock_guard<std::mutex> lock(workspace_mutex_);
        // Y, atoms × 3·sets, accumulated over the blocks on the device.
        ensure_capacity(device_products_, static_cast<std::size_t>(number_of_atoms_) * columns);
        std::vector<Complex> gradient_weights;
        for_each_block(number_of_g_vectors, [&](std::size_t first, std::size_t count) {
            form_phases(g_vector_indices, first, count, +1.0);
            // X, G-vectors × 3·sets, column-major.
            gradient_weights.assign(count * columns, Complex(0.0, 0.0));
            for (std::size_t set = 0; set < number_of_sets; set++) {
                for (std::size_t entry = 0; entry < count; entry++) {
                    const Vector3D& g_cartesian =
                        g_vectors[static_cast<std::size_t>(g_vector_indices[first + entry])].g_cart;
                    const Complex weight = phase_weights[set][first + entry];
                    for (std::size_t direction = 0; direction < 3; direction++) {
                        gradient_weights[(3 * set + direction) * count + entry] =
                            g_cartesian[static_cast<int>(direction)] * weight;
                    }
                }
            }
            ensure_capacity(device_weights_, gradient_weights.size());
            device_backend_->copy(device_weights_.data(),
                                  gradient_weights.data(),
                                  gradient_weights.size() * sizeof(Complex));
            device_blas_.gemm('T',
                              'N',
                              number_of_atoms_,
                              static_cast<int>(columns),
                              static_cast<int>(count),
                              Complex(1.0, 0.0),
                              device_phases_.data(),
                              static_cast<int>(count),
                              device_weights_.data(),
                              static_cast<int>(count),
                              first == 0 ? Complex(0.0, 0.0) : Complex(1.0, 0.0),
                              device_products_.data(),
                              number_of_atoms_);
            // The weights are rewritten for the next block once this product has read them.
            device_backend_->synchronize();
        });
        std::vector<Complex> products(static_cast<std::size_t>(number_of_atoms_) * columns);
        device_backend_->copy(
            products.data(), device_products_.data(), products.size() * sizeof(Complex));
        for (int atom_index = 0; atom_index < number_of_atoms_; atom_index++) {
            const int set = set_of_atom[static_cast<std::size_t>(atom_index)];
            if (set < 0) {
                continue;
            }
            for (int direction = 0; direction < 3; direction++) {
                result[static_cast<std::size_t>(atom_index)][direction] =
                    products[(3 * static_cast<std::size_t>(set) + direction) * number_of_atoms_ +
                             atom_index]
                        .imag();
            }
        }
        return result;
    }

  private:
    [[nodiscard]] static std::shared_ptr<execution::gpu::device::GPUDeviceBackend>
    require(std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend) {
        if (!device_backend) {
            throw std::invalid_argument(
                "GPUStructureFactorCalculator: the device backend must not be null");
        }
        return device_backend;
    }

    /// Calls `body(first, count)` for consecutive blocks of the G-vectors.
    template <typename Body> void for_each_block(std::size_t number_of_g_vectors, Body body) const {
        const std::size_t block = std::max<std::size_t>(
            1, phase_block_elements_ / static_cast<std::size_t>(number_of_atoms_));
        for (std::size_t first = 0; first < number_of_g_vectors; first += block) {
            body(first, std::min(block, number_of_g_vectors - first));
        }
    }

    /// The phases e^{sign·iG_j·τ_a} of the block's G-vectors, into device_phases_.
    void form_phases(const std::vector<int>& g_vector_indices,
                     std::size_t first,
                     std::size_t count,
                     Real sign) const {
        const auto& g_vectors = planewave_basis_.g_vectors();
        std::vector<Real> g_cartesian(3 * count);
        for (std::size_t entry = 0; entry < count; entry++) {
            const Vector3D& g_vector =
                g_vectors[static_cast<std::size_t>(g_vector_indices[first + entry])].g_cart;
            for (int direction = 0; direction < 3; direction++) {
                g_cartesian[3 * entry + static_cast<std::size_t>(direction)] = g_vector[direction];
            }
        }
        ensure_capacity(device_g_cartesian_, g_cartesian.size());
        device_backend_->copy(
            device_g_cartesian_.data(), g_cartesian.data(), g_cartesian.size() * sizeof(Real));
        ensure_capacity(device_phases_, count * static_cast<std::size_t>(number_of_atoms_));
        diagonalizer::planewave::gpu::launch_form_structure_phases(static_cast<int>(count),
                                                                   number_of_atoms_,
                                                                   device_g_cartesian_.data(),
                                                                   device_positions_.data(),
                                                                   sign,
                                                                   device_phases_.data());
    }

    template <typename Element>
    void ensure_capacity(execution::gpu::device::GPUDeviceBuffer<Element>& buffer,
                         std::size_t elements) const {
        if (buffer.size() < elements) {
            buffer = execution::gpu::device::GPUDeviceBuffer<Element>(elements, device_backend_);
        }
    }

    const PlanewaveBasis& planewave_basis_;
    int number_of_atoms_;
    std::size_t phase_block_elements_;
    std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend_;
    const execution::gpu::blas::GPUBLAS& device_blas_;
    mutable std::mutex workspace_mutex_;
    execution::gpu::device::GPUDeviceBuffer<Real> device_positions_;
    mutable execution::gpu::device::GPUDeviceBuffer<Real> device_g_cartesian_;
    mutable execution::gpu::device::GPUDeviceBuffer<Complex> device_phases_;
    mutable execution::gpu::device::GPUDeviceBuffer<Complex> device_weights_;
    mutable execution::gpu::device::GPUDeviceBuffer<Complex> device_products_;
};

} // namespace q3
