#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_0_core/execution/gpu/device/init.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/gpu/init.hpp"
#include "layer_4_computation/charge_density/planewave/internal/accumulator.hpp"
#include "layer_4_computation/charge_density/planewave/internal/accumulator_gpu.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/init.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/gpu_fft_grid_cpu_backed.hpp"

using namespace q3;

namespace {

using test_support::CPUBackedGPUDeviceBackend;
using test_support::CPUBackedGPUFFTGrid;

/// Two-atom silicon after one SCF iteration on a shifted 2x2x2 grid: complex bands at several k.
struct ConvergedSilicon {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    KPointsGrid kpoint_grid;
    std::unique_ptr<KohnShamSCFIterator> scf_iterator;
    PlanewaveKohnShamSCFIterator* planewave_iterator = nullptr;

    ConvergedSilicon() {
        pseudopotential_set.push_back(
            UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));
        crystal_structure.set_from_lattice_type(2, 10.263048);
        AtomSpecies si_species{"Si", 28.08, pseudopotential_set[0].valence_charge, "Si.pz-vbc.UPF"};
        crystal_structure.add_species(si_species);
        crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
        crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
        crystal_structure.compute_derived_quantities();
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);

        KohnShamSCFIteratorConfig config;
        config.crystal_structure = &crystal_structure;
        config.number_of_spin_components = 1;
        config.kpoint_grid = &kpoint_grid;
        config.pseudopotential_set = &pseudopotential_set;
        config.wavefunction_energy_cutoff = 16.0;
        config.maximum_iterations = 1;
        scf_iterator = create_kohn_sham_iterator_from_config(config);
        scf_iterator->run();
        planewave_iterator = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator.get());
    }
};

} // namespace

// ── The device accumulator reproduces the host accumulator ──────────

TEST(GPUChargeDensityAccumulator, MatchesTheHostAccumulator) {
    ConvergedSilicon silicon;
    ASSERT_NE(silicon.planewave_iterator, nullptr);
    const auto& basis = silicon.planewave_iterator->planewave_basis();
    const auto& quantum_system = silicon.planewave_iterator->quantum_system();
    int number_of_bands = silicon.planewave_iterator->number_of_bands();
    Real inverse_cell_volume = 1.0 / silicon.crystal_structure.cell_volume();

    std::vector<std::vector<Real>> host_density;
    PlanewaveChargeDensityAccumulator host_accumulator(basis, silicon.kpoint_grid, quantum_system);
    host_accumulator.accumulate(number_of_bands, 1, inverse_cell_volume, host_density);

    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUFFTGrid device_transform(device_backend);
    device_transform.init(basis.fft_wave().grid_dimension_1(),
                          basis.fft_wave().grid_dimension_2(),
                          basis.fft_wave().grid_dimension_3());
    std::vector<std::vector<Real>> device_density;
    GPUPlanewaveChargeDensityAccumulator device_accumulator(
        basis, silicon.kpoint_grid, quantum_system, device_transform, device_backend);
    device_accumulator.accumulate(number_of_bands, 1, inverse_cell_volume, device_density);

    ASSERT_EQ(device_density.size(), host_density.size());
    ASSERT_EQ(device_density[0].size(), host_density[0].size());
    Real largest_density = 0.0;
    Real largest_difference = 0.0;
    for (std::size_t grid_index = 0; grid_index < host_density[0].size(); ++grid_index) {
        largest_density = std::max(largest_density, std::abs(host_density[0][grid_index]));
        largest_difference =
            std::max(largest_difference,
                     std::abs(device_density[0][grid_index] - host_density[0][grid_index]));
    }
    ASSERT_GT(largest_density, 0.0);
    EXPECT_LT(largest_difference, 1e-12 * largest_density);
}

// ── Without an initialized device transform the host path runs ──────

TEST(GPUChargeDensityAccumulator, FallsBackToTheHostWhenTheTransformIsNotInitialized) {
    ConvergedSilicon silicon;
    ASSERT_NE(silicon.planewave_iterator, nullptr);
    const auto& basis = silicon.planewave_iterator->planewave_basis();
    const auto& quantum_system = silicon.planewave_iterator->quantum_system();
    int number_of_bands = silicon.planewave_iterator->number_of_bands();
    Real inverse_cell_volume = 1.0 / silicon.crystal_structure.cell_volume();

    std::vector<std::vector<Real>> host_density;
    PlanewaveChargeDensityAccumulator host_accumulator(basis, silicon.kpoint_grid, quantum_system);
    host_accumulator.accumulate(number_of_bands, 1, inverse_cell_volume, host_density);

    auto device_backend = std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUFFTGrid device_transform(device_backend); // never initialized: zero grid points
    std::vector<std::vector<Real>> device_density;
    GPUPlanewaveChargeDensityAccumulator device_accumulator(
        basis, silicon.kpoint_grid, quantum_system, device_transform, device_backend);
    device_accumulator.accumulate(number_of_bands, 1, inverse_cell_volume, device_density);

    ASSERT_EQ(device_density.size(), host_density.size());
    EXPECT_EQ(device_density[0], host_density[0]);
}
