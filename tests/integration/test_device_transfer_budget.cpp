/**
 * What an SCF iteration moves between host and device, counted on a device
 * path whose device is the host.
 *
 * The device eigensolver, the device Hamiltonian and the device density
 * accumulator run here through host stand-ins, so every GPUDeviceBackend::copy
 * they make is counted although no byte leaves the host. None of the defects
 * this guards against changes a result: host code touching device memory, or
 * the same data crossing more often than the algorithm needs, only shows up as
 * time on a real device, which is why no other test sees it.
 *
 * The band coefficients and the Kleinman-Bylander projectors are counted apart
 * from everything else, by where the host end of each copy points. The bands
 * do not cross at all during a run: the starting rotation writes them on the
 * device, the eigensolver seeds from them and writes its result there, and the
 * density is summed from them there. They used to cross three times per
 * iteration (up for the seed, up again for the density, down after the solve),
 * one copy per band. The host gets them when it asks, one copy per k-point
 * state. The projectors cross once per k-point for the whole run: each
 * k-point's set owns its device copy and knows it is current.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/pseudopotential/planewave/band_projection_calculator_serial.hpp"
#include "layer_4_computation/charge_density/planewave/internal/accumulator_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/davidson_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/band_projection_calculator_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/projector_set_builder_gpu.hpp"
#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/projector_strain_overlap_calculator_gpu.hpp"
#include "layer_4_computation/iterator/kohn_sham/factory.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/pseudopotential/init.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_5_properties/forces/planewave/pseudopotential/internal/nonlocal.hpp"
#include "layer_5_properties/stress_tensor/planewave/contributions.hpp"
#include "layer_5_properties/stress_tensor/planewave/pseudopotential/internal/nonlocal.hpp"
#include "utils/gpu_blas_cpu_backed.hpp"
#include "utils/gpu_device_backend_cpu_backed.hpp"
#include "utils/gpu_eigensolver_cpu_backed.hpp"
#include "utils/gpu_fft_grid_cpu_backed.hpp"

using namespace q3;
using test_support::CPUBackedGPUBLAS;
using test_support::CPUBackedGPUDeviceBackend;
using test_support::CPUBackedGPUEigensolver;
using test_support::CPUBackedGPUFFTGrid;

namespace {

/// Bands kept per k-point state; two silicon atoms fill four of them.
constexpr int number_of_bands = 8;

/// One copy between host memory and a device buffer, as the backend was asked to make it.
struct Crossing {
    const void* host_end = nullptr;
    std::size_t size_bytes = 0;
    bool is_upload = false;
};

/// What one run moved, with the band coefficients counted apart from the rest.
struct TransferReport {
    std::uint64_t band_bytes_uploaded = 0;
    std::uint64_t band_bytes_downloaded = 0;
    std::uint64_t band_upload_copies = 0;
    std::uint64_t band_download_copies = 0;
    std::uint64_t projector_bytes_uploaded = 0;
    std::uint64_t projector_upload_copies = 0;
    std::uint64_t other_bytes_uploaded = 0;
    std::uint64_t other_bytes_downloaded = 0;
    /// Size of all the band blocks together, and how many k-point states they belong to.
    std::uint64_t band_block_bytes = 0;
    int number_of_kpoint_states = 0;
    /// Size of all the projector sets together, one per k-point.
    std::uint64_t projector_set_bytes = 0;
    int number_of_kpoints = 0;
    Real total_energy = 0.0;
};

/// How a device run of this test builds its projector sets and how much memory it reports.
struct DeviceOptions {
    /// Form the projector sets on the device instead of uploading host-built ones.
    bool builds_projector_sets_on_device = false;
    /// What the stand-in device reports as its memory; zero is "unknown", so no budget.
    std::size_t device_memory_bytes = 0;
    /// Solve the Davidson subspace eigenproblem with the device eigensolver.
    bool solves_subspace_on_device = false;
};

/**
 * Two-atom silicon on a shifted 2x2x2 mesh, run for a fixed number of SCF
 * iterations, on the host or on the device path built from the host stand-ins
 * the way the GPU entry points build it from a vendor's backends.
 */
class SiliconRun {
  public:
    SiliconRun(int number_of_iterations,
               bool on_device,
               bool off_site_atom = false,
               DeviceOptions device_options = {})
        : device_options_(device_options) {
        device_backend_->set_total_memory_bytes(device_options.device_memory_bytes);
        pseudopotential_set_.push_back(
            UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Si.pz-vbc.UPF"));
        crystal_structure_.set_from_lattice_type(2, 10.263048);
        AtomSpecies si_species{
            "Si", 28.08, pseudopotential_set_[0].valence_charge, "Si.pz-vbc.UPF"};
        crystal_structure_.add_species(si_species);
        crystal_structure_.add_atom({0, {0.0, 0.0, 0.0}});
        // Off its site, the second atom feels a force and the cell is strained unevenly.
        crystal_structure_.add_atom(
            {0, off_site_atom ? Vector3D(0.27, 0.24, 0.25) : Vector3D(0.25, 0.25, 0.25)});
        crystal_structure_.compute_derived_quantities();
        kpoint_grid_.generate(2, 2, 2, 1, 1, 1);

        KohnShamSCFIteratorConfig config;
        config.crystal_structure = &crystal_structure_;
        config.number_of_spin_components = 1;
        config.kpoint_grid = &kpoint_grid_;
        config.pseudopotential_set = &pseudopotential_set_;
        config.wavefunction_energy_cutoff = 16.0;
        config.number_of_bands = number_of_bands;
        config.maximum_iterations = number_of_iterations;
        // Never reached, so the run performs exactly the iterations asked for.
        config.convergence_threshold = 1.0e-30;
        scf_iterator_ = create_kohn_sham_iterator_from_config(config);
        planewave_iterator_ = dynamic_cast<PlanewaveKohnShamSCFIterator*>(scf_iterator_.get());
        pseudopotential_iterator_ =
            dynamic_cast<PlanewavePseudopotentialKohnShamSCFIterator*>(scf_iterator_.get());
        if (on_device) {
            run_on_the_device_path();
        }
        scf_iterator_->run();
        // What follows the run -- this test's own reads included -- is not the run's traffic.
        device_backend_->set_copy_observer({});
        totals_after_run_ = device_backend_->transfer_totals();
    }

    [[nodiscard]] TransferReport report() const {
        const PlanewaveKohnShamQuantumSystem& quantum_system =
            planewave_iterator_->quantum_system();
        TransferReport transfer_report;
        transfer_report.total_energy = scf_iterator_->total_energy();
        transfer_report.number_of_kpoint_states = quantum_system.number_of_kpoint_states();

        /**
         * The band blocks' host addresses, read after the run: asking for a
         * host view brings the bands down from the device, which is why the
         * recording stopped when the run did.
         */
        std::vector<const char*> block_begins;
        std::vector<const char*> block_ends;
        for (int state = 0; state < quantum_system.number_of_kpoint_states(); ++state) {
            const std::size_t block_bytes =
                static_cast<std::size_t>(
                    quantum_system.number_of_plane_waves_for_kpoint_state(state)) *
                quantum_system.number_of_bands() * sizeof(Complex);
            block_begins.push_back(reinterpret_cast<const char*>(quantum_system.band_block(state)));
            block_ends.push_back(block_begins.back() + block_bytes);
            transfer_report.band_block_bytes += block_bytes;
        }

        // The projector sets' host addresses: the sets the eigensolver read are still built.
        std::vector<const char*> set_begins;
        std::vector<const char*> set_ends;
        transfer_report.number_of_kpoints = kpoint_grid_.number_of_kpoints();
        for (int kpoint_index = 0; kpoint_index < kpoint_grid_.number_of_kpoints();
             ++kpoint_index) {
            const KleinmanBylanderProjectorSet& projector_set =
                pseudopotential_iterator_->projector_set(kpoint_index);
            const std::size_t set_bytes =
                static_cast<std::size_t>(projector_set.number_of_projectors()) *
                projector_set.number_of_plane_waves() * sizeof(Complex);
            set_begins.push_back(reinterpret_cast<const char*>(projector_set.host()));
            set_ends.push_back(set_begins.back() + set_bytes);
            transfer_report.projector_set_bytes += set_bytes;
        }

        for (const Crossing& crossing : crossings_) {
            const char* host_end = static_cast<const char*>(crossing.host_end);
            bool is_band_data = false;
            for (std::size_t block = 0; block < block_begins.size(); ++block) {
                is_band_data = is_band_data ||
                               (host_end >= block_begins[block] && host_end < block_ends[block]);
            }
            bool is_projector_data = false;
            for (std::size_t set = 0; set < set_begins.size(); ++set) {
                is_projector_data =
                    is_projector_data || (host_end >= set_begins[set] && host_end < set_ends[set]);
            }
            if (is_projector_data && crossing.is_upload) {
                transfer_report.projector_bytes_uploaded += crossing.size_bytes;
                ++transfer_report.projector_upload_copies;
            } else if (is_band_data && crossing.is_upload) {
                transfer_report.band_bytes_uploaded += crossing.size_bytes;
                ++transfer_report.band_upload_copies;
            } else if (is_band_data) {
                transfer_report.band_bytes_downloaded += crossing.size_bytes;
                ++transfer_report.band_download_copies;
            } else if (crossing.is_upload) {
                transfer_report.other_bytes_uploaded += crossing.size_bytes;
            } else {
                transfer_report.other_bytes_downloaded += crossing.size_bytes;
            }
        }
        return transfer_report;
    }

    /// What the backend itself counted during the run, to compare with the crossings recorded.
    [[nodiscard]] execution::gpu::device::DeviceTransferTotals backend_totals() const {
        return totals_after_run_;
    }

    /**
     * Ask for every k-point state's bands on the host, as the first host reader after a device
     * run does, and return what that moved.
     */
    [[nodiscard]] execution::gpu::device::DeviceTransferTotals bring_bands_to_the_host() const {
        const execution::gpu::device::DeviceTransferTotals before =
            device_backend_->transfer_totals();
        const PlanewaveKohnShamQuantumSystem& quantum_system =
            planewave_iterator_->quantum_system();
        for (int state = 0; state < quantum_system.number_of_kpoint_states(); ++state) {
            (void)quantum_system.band_block(state);
        }
        return device_backend_->transfer_totals().since(before);
    }

    /// The run's projector sets, as the iterator holds them after the run.
    [[nodiscard]] KleinmanBylanderProjectorSetStore& projector_set_store() const {
        return pseudopotential_iterator_->projector_set_store();
    }

    /// The size of the largest k-point's projector set.
    [[nodiscard]] std::size_t largest_projector_set_bytes() const {
        std::size_t largest = 0;
        const int number_of_projectors = pseudopotential_iterator_->norm_conserving_quantum_system()
                                             .pseudopotential_basis()
                                             .total_number_of_projectors();
        for (int kpoint_index = 0; kpoint_index < kpoint_grid_.number_of_kpoints();
             ++kpoint_index) {
            largest = std::max(
                largest,
                static_cast<std::size_t>(number_of_projectors) *
                    planewave_iterator_->planewave_basis().number_of_plane_waves_by_kpoint_index(
                        kpoint_index) *
                    sizeof(Complex));
        }
        return largest;
    }

    /// The state a flavor hands to forces and stress, carrying this run's projection calculator.
    [[nodiscard]] NormConservingPlanewaveKohnShamSCFState converged_state() const {
        return pseudopotential_iterator_->result();
    }

    /// The nonlocal force on each atom, from a state of this run.
    [[nodiscard]] std::vector<Vector3D>
    nonlocal_forces(const NormConservingPlanewaveKohnShamSCFState& state) const {
        return NonlocalPseudopotentialForceCalculator(
                   crystal_structure_, planewave_iterator_->planewave_basis(), kpoint_grid_, state)
            .compute();
    }

    /// The nonlocal stress, from a state of this run.
    [[nodiscard]] StressTensorContributions
    nonlocal_stress(const NormConservingPlanewaveKohnShamSCFState& state) const {
        StressTensorContributions contributions;
        NonlocalPseudopotentialStressTensorCalculator(crystal_structure_,
                                                      planewave_iterator_->planewave_basis(),
                                                      kpoint_grid_,
                                                      state,
                                                      contributions)
            .compute();
        return contributions;
    }

    /// Record every crossing from now on into `crossings`, until stop_recording().
    void record_crossings_into(std::vector<Crossing>& crossings) {
        device_backend_->set_copy_observer(
            [this, &crossings](const void* destination, const void* source, std::size_t size) {
                const bool destination_is_device = device_backend_->is_device_memory(destination);
                const bool source_is_device = device_backend_->is_device_memory(source);
                if (destination_is_device && !source_is_device) {
                    crossings.push_back({source, size, true});
                } else if (source_is_device && !destination_is_device) {
                    crossings.push_back({destination, size, false});
                }
            });
    }
    void stop_recording() {
        device_backend_->set_copy_observer({});
    }

  private:
    void run_on_the_device_path() {
        device_backend_->set_copy_observer(
            [this](const void* destination, const void* source, std::size_t size_bytes) {
                const bool destination_is_device = device_backend_->is_device_memory(destination);
                const bool source_is_device = device_backend_->is_device_memory(source);
                if (destination_is_device && !source_is_device) {
                    crossings_.push_back({source, size_bytes, true});
                } else if (source_is_device && !destination_is_device) {
                    crossings_.push_back({destination, size_bytes, false});
                }
            });
        auto device_davidson = std::make_unique<GPUPlanewaveDavidsonDiagonalizer>(
            PlanewaveDavidsonDiagonalizer::default_serial_execution_policy,
            device_blas_,
            &device_transform_,
            device_eigensolver_,
            device_backend_);
        device_davidson->set_solves_subspace_on_device(device_options_.solves_subspace_on_device);
        planewave_iterator_->set_diagonalizer(std::move(device_davidson));
        const gpu::GPUFFTGrid& device_transform = device_transform_;
        std::shared_ptr<execution::gpu::device::GPUDeviceBackend> device_backend = device_backend_;
        planewave_iterator_->set_charge_density_accumulator_factory(
            [&device_transform,
             device_backend](const PlanewaveBasis& basis,
                             const KPointsGrid& kpoint_grid,
                             const PlanewaveKohnShamQuantumSystem& quantum_system) {
                return std::make_unique<GPUPlanewaveChargeDensityAccumulator>(
                    basis, kpoint_grid, quantum_system, device_transform, device_backend);
            });
        planewave_iterator_->set_band_projection_calculator(
            std::make_shared<GPUBandProjectionCalculator>(device_backend_, device_blas_));
        const CPUBackedGPUBLAS& strain_blas = device_blas_;
        planewave_iterator_->set_projector_strain_overlap_calculator_factory(
            [device_backend,
             &strain_blas](const Crystal& crystal_structure,
                           const PlanewaveBasis& planewave_basis,
                           const NormConservingPlanewaveBasis& pseudopotential_basis) {
                return std::make_unique<GPUProjectorStrainOverlapCalculator>(crystal_structure,
                                                                             planewave_basis,
                                                                             pseudopotential_basis,
                                                                             device_backend,
                                                                             strain_blas);
            });
        if (device_options_.builds_projector_sets_on_device) {
            const CPUBackedGPUBLAS& device_blas = device_blas_;
            planewave_iterator_->set_projector_set_builder_factory(
                [device_backend,
                 &device_blas](const Crystal& crystal_structure,
                               const PlanewaveBasis& planewave_basis,
                               const NormConservingPlanewaveBasis& pseudopotential_basis) {
                    return std::make_unique<GPUProjectorSetBuilder>(crystal_structure,
                                                                    planewave_basis,
                                                                    pseudopotential_basis,
                                                                    device_backend,
                                                                    device_blas);
                });
        }
    }

    DeviceOptions device_options_;
    PseudopotentialSet pseudopotential_set_;
    Crystal crystal_structure_;
    KPointsGrid kpoint_grid_;
    std::shared_ptr<CPUBackedGPUDeviceBackend> device_backend_ =
        std::make_shared<CPUBackedGPUDeviceBackend>();
    CPUBackedGPUBLAS device_blas_{device_backend_};
    CPUBackedGPUEigensolver device_eigensolver_{device_backend_};
    CPUBackedGPUFFTGrid device_transform_{device_backend_};
    std::vector<Crossing> crossings_;
    execution::gpu::device::DeviceTransferTotals totals_after_run_;
    std::unique_ptr<KohnShamSCFIterator> scf_iterator_;
    PlanewaveKohnShamSCFIterator* planewave_iterator_ = nullptr;
    PlanewavePseudopotentialKohnShamSCFIterator* pseudopotential_iterator_ = nullptr;
};

/// The traffic of the iterations a longer run performs beyond a shorter one.
struct IterationTraffic {
    TransferReport shorter_run = SiliconRun(2, /*on_device=*/true).report();
    TransferReport longer_run = SiliconRun(4, /*on_device=*/true).report();
    std::uint64_t number_of_iterations = 2;

    [[nodiscard]] std::uint64_t band_block_bytes() const {
        return longer_run.band_block_bytes;
    }
    [[nodiscard]] std::uint64_t number_of_kpoint_states() const {
        return static_cast<std::uint64_t>(longer_run.number_of_kpoint_states);
    }
};

} // namespace

// ─── The device path computes what the host path computes ───────────────────

TEST(DeviceTransferBudget, TheDevicePathOnHostStandInsReproducesTheHostEnergy) {
    SiliconRun host_run(4, /*on_device=*/false);
    SiliconRun device_run(4, /*on_device=*/true);

    EXPECT_NEAR(device_run.report().total_energy, host_run.report().total_energy, 1.0e-9);
}

TEST(DeviceTransferBudget, AHostRunCrossesNothing) {
    SiliconRun host_run(2, /*on_device=*/false);

    EXPECT_EQ(host_run.backend_totals().host_to_device_copies, 0u);
    EXPECT_EQ(host_run.backend_totals().device_to_host_copies, 0u);
}

TEST(DeviceTransferBudget, TheBackendsTotalsAreTheCrossingsItWasAskedToMake) {
    SiliconRun device_run(2, /*on_device=*/true);
    const TransferReport transfer_report = device_run.report();

    EXPECT_GT(transfer_report.other_bytes_uploaded, 0u);
    EXPECT_EQ(device_run.backend_totals().host_to_device_bytes,
              transfer_report.band_bytes_uploaded + transfer_report.projector_bytes_uploaded +
                  transfer_report.other_bytes_uploaded);
    EXPECT_EQ(device_run.backend_totals().device_to_host_bytes,
              transfer_report.band_bytes_downloaded + transfer_report.other_bytes_downloaded);
}

// ─── The bands stay on the device ───────────────────────────────────────────

TEST(DeviceTransferBudget, ARunMovesNoBandDataEitherWay) {
    IterationTraffic traffic;

    for (const TransferReport* run : {&traffic.shorter_run, &traffic.longer_run}) {
        ASSERT_GT(run->band_block_bytes, 0u);
        EXPECT_EQ(run->band_bytes_uploaded, 0u);
        EXPECT_EQ(run->band_upload_copies, 0u);
        EXPECT_EQ(run->band_bytes_downloaded, 0u);
        EXPECT_EQ(run->band_download_copies, 0u);
    }
}

TEST(DeviceTransferBudget, TheFirstHostRequestAfterARunIsOneDownloadPerKpointState) {
    SiliconRun device_run(2, /*on_device=*/true);

    const execution::gpu::device::DeviceTransferTotals first_request =
        device_run.bring_bands_to_the_host();
    const TransferReport transfer_report = device_run.report();

    EXPECT_EQ(first_request.device_to_host_bytes, transfer_report.band_block_bytes);
    EXPECT_EQ(first_request.device_to_host_copies,
              static_cast<std::uint64_t>(transfer_report.number_of_kpoint_states));
    EXPECT_EQ(first_request.host_to_device_copies, 0u);

    // Both copies are current now: asking again moves nothing.
    const execution::gpu::device::DeviceTransferTotals second_request =
        device_run.bring_bands_to_the_host();
    EXPECT_EQ(second_request.device_to_host_copies, 0u);
    EXPECT_EQ(second_request.host_to_device_copies, 0u);
}

// ─── The projectors ─────────────────────────────────────────────────────────

TEST(DeviceTransferBudget, EachKpointsProjectorSetIsUploadedOnceForTheWholeRun) {
    IterationTraffic traffic;

    for (const TransferReport* run : {&traffic.shorter_run, &traffic.longer_run}) {
        ASSERT_GT(run->projector_set_bytes, 0u);
        EXPECT_EQ(run->projector_bytes_uploaded, run->projector_set_bytes);
        EXPECT_EQ(run->projector_upload_copies, static_cast<std::uint64_t>(run->number_of_kpoints));
    }
}

// ─── Everything else an iteration moves ─────────────────────────────────────

TEST(DeviceTransferBudget, TrafficOtherThanBandsStaysUnderOneAndAHalfBandBlocksEachWay) {
    IterationTraffic traffic;
    /**
     * Subspace matrices, eigenvalues, potentials and index tables: 0.93 band
     * blocks up and 1.12 down per iteration when this was written. The budget
     * leaves room for the eigensolver to take a few more or fewer steps, and
     * none for one more bulk crossing of anything the size of the bands.
     */
    const std::uint64_t budget = traffic.number_of_iterations * traffic.band_block_bytes() * 3 / 2;

    // Kept in the test report, so that a run's log says what the traffic was and not only "ok".
    RecordProperty("band_block_bytes", std::to_string(traffic.band_block_bytes()));
    RecordProperty("other_bytes_uploaded_per_iteration",
                   std::to_string((traffic.longer_run.other_bytes_uploaded -
                                   traffic.shorter_run.other_bytes_uploaded) /
                                  traffic.number_of_iterations));
    RecordProperty("other_bytes_downloaded_per_iteration",
                   std::to_string((traffic.longer_run.other_bytes_downloaded -
                                   traffic.shorter_run.other_bytes_downloaded) /
                                  traffic.number_of_iterations));

    EXPECT_LT(traffic.longer_run.other_bytes_uploaded - traffic.shorter_run.other_bytes_uploaded,
              budget);
    EXPECT_LT(traffic.longer_run.other_bytes_downloaded -
                  traffic.shorter_run.other_bytes_downloaded,
              budget);
}

// ─── Forces and stress after the run ────────────────────────────────────────

TEST(DeviceTransferBudget, NonlocalForcesAndStressThroughTheDeviceMatchTheHostProducts) {
    SiliconRun device_run(3, /*on_device=*/true, /*off_site_atom=*/true);
    NormConservingPlanewaveKohnShamSCFState device_state = device_run.converged_state();
    NormConservingPlanewaveKohnShamSCFState host_state = device_run.converged_state();
    host_state.band_projections = std::make_shared<SerialBandProjectionCalculator>();
    host_state.projector_strain_overlap_calculator_factory = {};

    const std::vector<Vector3D> device_forces = device_run.nonlocal_forces(device_state);
    const std::vector<Vector3D> host_forces = device_run.nonlocal_forces(host_state);
    const StressTensorContributions device_stress = device_run.nonlocal_stress(device_state);
    const StressTensorContributions host_stress = device_run.nonlocal_stress(host_state);

    ASSERT_EQ(device_forces.size(), host_forces.size());
    Real largest_force = 0.0;
    for (std::size_t atom = 0; atom < host_forces.size(); ++atom) {
        for (int direction = 0; direction < 3; ++direction) {
            largest_force = std::max(largest_force, std::abs(host_forces[atom][direction]));
            EXPECT_NEAR(device_forces[atom][direction], host_forces[atom][direction], 1.0e-11);
        }
    }
    // The comparison is of something: the off-site atom is pushed.
    EXPECT_GT(largest_force, 1.0e-3);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            EXPECT_NEAR(
                device_stress.nonlocal[row][column], host_stress.nonlocal[row][column], 1.0e-11);
        }
    }
    EXPECT_GT(std::abs(host_stress.nonlocal[0][0]), 1.0e-6);
}

TEST(DeviceTransferBudget, ForcesAndStressUploadEachStatesBandsOnceAndBringNoneDown) {
    SiliconRun device_run(2, /*on_device=*/true, /*off_site_atom=*/true);
    NormConservingPlanewaveKohnShamSCFState state = device_run.converged_state();
    const PlanewaveKohnShamQuantumSystem& snapshot = state.quantum_system();
    std::vector<const char*> block_begins;
    std::vector<const char*> block_ends;
    for (int kpoint_state = 0; kpoint_state < snapshot.number_of_kpoint_states(); ++kpoint_state) {
        const std::size_t block_bytes =
            static_cast<std::size_t>(
                snapshot.number_of_plane_waves_for_kpoint_state(kpoint_state)) *
            snapshot.number_of_bands() * sizeof(Complex);
        block_begins.push_back(reinterpret_cast<const char*>(snapshot.band_block(kpoint_state)));
        block_ends.push_back(block_begins.back() + block_bytes);
    }

    std::vector<Crossing> crossings;
    device_run.record_crossings_into(crossings);
    (void)device_run.nonlocal_forces(state);
    (void)device_run.nonlocal_stress(state);
    device_run.stop_recording();

    std::uint64_t band_block_uploads = 0;
    std::uint64_t band_block_downloads = 0;
    for (const Crossing& crossing : crossings) {
        const char* host_end = static_cast<const char*>(crossing.host_end);
        for (std::size_t block = 0; block < block_begins.size(); ++block) {
            if (host_end >= block_begins[block] && host_end < block_ends[block]) {
                ++(crossing.is_upload ? band_block_uploads : band_block_downloads);
            }
        }
    }
    /**
     * The state holds a host copy of the bands, so each k-point state's block goes up once,
     * for the first product that needs it, and serves every product after it: the four of
     * the force and each block of the stress. Nothing of the bands comes back.
     */
    EXPECT_EQ(band_block_uploads, static_cast<std::uint64_t>(snapshot.number_of_kpoint_states()));
    EXPECT_EQ(band_block_downloads, 0u);
}

// ─── Projector sets formed on the device ────────────────────────────────────

TEST(DeviceTransferBudget, WithTheDeviceBuilderNoProjectorSetIsUploadedOrHeldOnTheHost) {
    DeviceOptions device_built;
    device_built.builds_projector_sets_on_device = true;
    SiliconRun host_built_run(3, /*on_device=*/true);
    SiliconRun device_built_run(3, /*on_device=*/true, /*off_site_atom=*/false, device_built);

    // Read before report(), which asks for the sets on the host to find their addresses.
    KleinmanBylanderProjectorSetStore& sets = device_built_run.projector_set_store();
    std::size_t sets_on_the_device = 0;
    for (int kpoint_index = 0; kpoint_index < sets.number_of_kpoints(); ++kpoint_index) {
        ASSERT_TRUE(sets.is_built(kpoint_index));
        EXPECT_FALSE(sets.at(kpoint_index).host_is_allocated());
        sets_on_the_device += sets.device_bytes(kpoint_index);
    }
    const TransferReport host_built = host_built_run.report();
    const TransferReport device_built_report = device_built_run.report();

    EXPECT_EQ(sets_on_the_device, host_built.projector_set_bytes);
    // The same projectors, so the same run: the two builders agree to the last bit.
    EXPECT_EQ(device_built_report.total_energy, host_built.total_energy);
    /**
     * What goes up instead of the sets is their factors: per plane wave of a k-point, the
     * real and imaginary part of a structure factor for each of the two atoms, 32 bytes, and
     * a form factor for each of the four channels, held as a complex number, 64 bytes, where
     * the set itself is eight complex projectors, 128. With two atoms that saves a quarter
     * of the upload; the saving grows with the cell, since a set grows with the atoms times
     * the channels and the factors with their sum. (Counted from the backend's totals: a
     * device-built set has no host address during the run to recognize its crossings by.)
     */
    EXPECT_EQ(host_built_run.backend_totals().host_to_device_bytes -
                  device_built_run.backend_totals().host_to_device_bytes,
              host_built.projector_set_bytes / 4);
    EXPECT_EQ(device_built_run.backend_totals().device_to_host_bytes,
              host_built_run.backend_totals().device_to_host_bytes);
    RecordProperty("projector_set_bytes", std::to_string(host_built.projector_set_bytes));
    RecordProperty("uploaded_with_host_built_sets",
                   std::to_string(host_built_run.backend_totals().host_to_device_bytes));
    RecordProperty("uploaded_with_device_built_sets",
                   std::to_string(device_built_run.backend_totals().host_to_device_bytes));
}

TEST(DeviceTransferBudget, ADeviceTooSmallForEverySetStillGivesTheSameEnergy) {
    SiliconRun unlimited_run(3, /*on_device=*/true);
    const std::size_t one_set = unlimited_run.largest_projector_set_bytes();
    for (bool builds_on_device : {false, true}) {
        DeviceOptions small_device;
        small_device.builds_projector_sets_on_device = builds_on_device;
        // The solver budgets half of what the device reports: room for two sets of eight.
        small_device.device_memory_bytes = 2 * 2 * one_set;
        SiliconRun small_run(3, /*on_device=*/true, /*off_site_atom=*/false, small_device);

        KleinmanBylanderProjectorSetStore& sets = small_run.projector_set_store();
        EXPECT_EQ(sets.device_memory_budget(), 2 * one_set);
        EXPECT_LE(sets.device_bytes(), sets.device_memory_budget());
        EXPECT_GT(sets.device_bytes(), 0u);
        EXPECT_EQ(small_run.report().total_energy, unlimited_run.report().total_energy)
            << "builds_on_device=" << builds_on_device;
    }
}

// ─── The subspace eigenproblem solved on the device ─────────────────────────

TEST(DeviceTransferBudget, SolvingTheSubspaceOnTheDeviceGivesTheHostEnergyAndBringsLessDown) {
    DeviceOptions device_solve;
    device_solve.solves_subspace_on_device = true;
    SiliconRun host_run(4, /*on_device=*/false);
    SiliconRun host_solve_run(4, /*on_device=*/true);
    SiliconRun device_solve_run(4, /*on_device=*/true, /*off_site_atom=*/false, device_solve);

    // Another eigensolver path orders its sums differently; the physics is the same.
    EXPECT_NEAR(device_solve_run.report().total_energy, host_run.report().total_energy, 1.0e-9);
    /**
     * A host solve brings both subspace matrices down for every Davidson iteration. A
     * device solve brings down the eigenvectors, one matrix, and the eigenvalues.
     */
    EXPECT_LT(device_solve_run.backend_totals().device_to_host_bytes,
              host_solve_run.backend_totals().device_to_host_bytes);
    EXPECT_EQ(device_solve_run.backend_totals().host_to_device_bytes,
              host_solve_run.backend_totals().host_to_device_bytes);
}
