#include <gtest/gtest.h>

#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/fft_grid_serial.hpp"
#include "layer_3_representation/basis/planewave/internal/fft/pair.hpp"

using namespace q3;

/**
 * Minimal serial factory: the FFTGridFactory seam lets this layer-3 test run
 * without any layer-4 execution policy.
 */
class SerialFFTGridFactoryForTest : public FFTGridFactory {
  public:
    std::unique_ptr<FFTGrid> create_fft_grid() const override {
        return std::make_unique<SerialFFTGrid>();
    }
};

static const SerialFFTGridFactoryForTest serial_fft_grid_factory;

/**
 * Helper to build a minimal FCC silicon crystal for FFT pair tests.
 */
static Crystal make_silicon_crystal() {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 10.263);
    AtomSpecies silicon{"Si", 28.08, 4.0, "Si.pz-vbc.UPF"};
    crystal_structure.add_species(silicon);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.add_atom({0, {0.25, 0.25, 0.25}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

// ── After initialization, both grids should have positive dimensions ──
TEST(PlanewaveFFTPair, InitializationProducesValidGridDimensions) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveFFTPair fft_pair;
    Real energy_cutoff_charge_density = 64.0;
    Real energy_cutoff_wavefunction = 16.0;
    fft_pair.initialize(
        serial_fft_grid_factory, silicon, energy_cutoff_charge_density, energy_cutoff_wavefunction);

    EXPECT_GT(fft_pair.fft_rho().grid_dimension_1(), 0);
    EXPECT_GT(fft_pair.fft_rho().grid_dimension_2(), 0);
    EXPECT_GT(fft_pair.fft_rho().grid_dimension_3(), 0);
    EXPECT_GT(fft_pair.fft_wave().grid_dimension_1(), 0);
    EXPECT_GT(fft_pair.fft_wave().grid_dimension_2(), 0);
    EXPECT_GT(fft_pair.fft_wave().grid_dimension_3(), 0);
}

// ── The charge density grid should be at least as large as the wave grid ──
TEST(PlanewaveFFTPair, ChargeDensityGridIsAtLeastAsLargeAsWaveGrid) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveFFTPair fft_pair;
    fft_pair.initialize(serial_fft_grid_factory, silicon, 64.0, 16.0);

    EXPECT_GE(fft_pair.fft_rho().grid_dimension_1(), fft_pair.fft_wave().grid_dimension_1());
    EXPECT_GE(fft_pair.fft_rho().grid_dimension_2(), fft_pair.fft_wave().grid_dimension_2());
    EXPECT_GE(fft_pair.fft_rho().grid_dimension_3(), fft_pair.fft_wave().grid_dimension_3());
}

// ── The wave grid is capped at the rho grid even when 2*ecutwfc > ecutrho ──
TEST(PlanewaveFFTPair, WaveGridCappedAtRhoGridWhenCutoffExceedsDensity) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveFFTPair fft_pair;
    /**
     * Set wavefunction cutoff higher than half the charge density cutoff
     * so that 2*ecutwfc > ecutrho. The wave grid should still not exceed rho.
     */
    Real energy_cutoff_charge_density = 40.0;
    Real energy_cutoff_wavefunction = 30.0;
    fft_pair.initialize(
        serial_fft_grid_factory, silicon, energy_cutoff_charge_density, energy_cutoff_wavefunction);

    EXPECT_LE(fft_pair.fft_wave().total_grid_points(), fft_pair.fft_rho().total_grid_points());
}

// ── Both const and non-const accessors return consistent references ──
TEST(PlanewaveFFTPair, ConstAndNonConstAccessorsAreConsistent) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveFFTPair fft_pair;
    fft_pair.initialize(serial_fft_grid_factory, silicon, 64.0, 16.0);

    FFTGrid& rho_mutable = fft_pair.fft_rho();
    const FFTGrid& rho_const = static_cast<const PlanewaveFFTPair&>(fft_pair).fft_rho();
    EXPECT_EQ(rho_mutable.grid_dimension_1(), rho_const.grid_dimension_1());

    FFTGrid& wave_mutable = fft_pair.fft_wave();
    const FFTGrid& wave_const = static_cast<const PlanewaveFFTPair&>(fft_pair).fft_wave();
    EXPECT_EQ(wave_mutable.grid_dimension_1(), wave_const.grid_dimension_1());
}

// ── Total grid points match product of dimensions ──
TEST(PlanewaveFFTPair, TotalGridPointsMatchDimensionProduct) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveFFTPair fft_pair;
    fft_pair.initialize(serial_fft_grid_factory, silicon, 64.0, 16.0);

    EXPECT_EQ(fft_pair.fft_rho().total_grid_points(),
              fft_pair.fft_rho().grid_dimension_1() * fft_pair.fft_rho().grid_dimension_2() *
                  fft_pair.fft_rho().grid_dimension_3());
    EXPECT_EQ(fft_pair.fft_wave().total_grid_points(),
              fft_pair.fft_wave().grid_dimension_1() * fft_pair.fft_wave().grid_dimension_2() *
                  fft_pair.fft_wave().grid_dimension_3());
}

// ── Wavefunction grid size: 2 x ecutwfc by default, exact on request ──

/**
 * The default wavefunction grid is sized for 2 x ecutwfc, on which V·ψ and |ψ|² alias; on
 * a hard deck at a low cutoff that put Q3 9.5 mRy off pw.x. The exact grid holds the
 * 4 x ecutwfc sphere, as pw.x's does, and every QE-format deck runs on it. For this cell at
 * 16/128 Ry, pw.x 7.6 prints
 *     Dense  grid:  3308 G-vectors  FFT dimensions: ( 32, 32, 32)
 *     Smooth grid:  1139 G-vectors  FFT dimensions: ( 20, 20, 20)
 */
TEST(PlanewaveFFTPair, ExactWavefunctionGridIsTheSmoothGridPwxBuilds) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveFFTPair fft_pair;
    fft_pair.initialize(serial_fft_grid_factory, silicon, 128.0, 16.0, true, true);

    EXPECT_EQ(fft_pair.fft_rho().grid_dimension_1(), 32);
    EXPECT_EQ(fft_pair.fft_rho().grid_dimension_2(), 32);
    EXPECT_EQ(fft_pair.fft_rho().grid_dimension_3(), 32);
    EXPECT_EQ(fft_pair.fft_wave().grid_dimension_1(), 20);
    EXPECT_EQ(fft_pair.fft_wave().grid_dimension_2(), 20);
    EXPECT_EQ(fft_pair.fft_wave().grid_dimension_3(), 20);
}

TEST(PlanewaveFFTPair, DefaultWavefunctionGridIsSizedForTwiceTheCutoff) {
    Crystal silicon = make_silicon_crystal();
    PlanewaveFFTPair reduced;
    reduced.initialize(serial_fft_grid_factory, silicon, 128.0, 16.0);

    int factors[3];
    FFTGridSizer::compute_fft_factors(silicon, factors);
    int expected[3];
    FFTGridSizer::calculate_grid_dimensions(32.0,
                                            silicon.two_pi_over_lattice_parameter_squared(),
                                            silicon.lattice_vectors(),
                                            silicon.reciprocal_lattice_vectors(),
                                            expected[0],
                                            expected[1],
                                            expected[2],
                                            factors);
    EXPECT_EQ(reduced.fft_wave().grid_dimension_1(), expected[0]);
    EXPECT_EQ(reduced.fft_wave().grid_dimension_2(), expected[1]);
    EXPECT_EQ(reduced.fft_wave().grid_dimension_3(), expected[2]);
    EXPECT_LT(reduced.fft_wave().total_grid_points(), 20 * 20 * 20);
    // The density grid does not depend on the choice.
    EXPECT_EQ(reduced.fft_rho().grid_dimension_1(), 32);
}

TEST(PlanewaveFFTPair, ExactWavefunctionGridIsTheDensityGridAtFourTimesTheCutoff) {
    // ecutrho = 4 x ecutwfc, the default, is where pw.x runs a single grid.
    Crystal silicon = make_silicon_crystal();
    PlanewaveFFTPair fft_pair;
    fft_pair.initialize(serial_fft_grid_factory, silicon, 64.0, 16.0, true, true);

    EXPECT_EQ(fft_pair.fft_wave().grid_dimension_1(), fft_pair.fft_rho().grid_dimension_1());
    EXPECT_EQ(fft_pair.fft_wave().grid_dimension_2(), fft_pair.fft_rho().grid_dimension_2());
    EXPECT_EQ(fft_pair.fft_wave().grid_dimension_3(), fft_pair.fft_rho().grid_dimension_3());
}
