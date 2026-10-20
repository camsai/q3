#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "io/pseudopotential/upf_reader.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "layer_5_properties/projections/planewave/init.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * S = I + strength |beta><beta|: the shape of an ultrasoft/PAW augmentation
 * overlap with a single projector (Vanderbilt, PRB 41, 7892 (1990)), standing
 * in for a live ultrasoft SCF state so this test needs no converged run. What
 * it exercises is the calculator's use of whatever S it is handed, which is
 * the whole of what this class contributes to the S metric.
 */
class SingleProjectorOverlapOperator : public OverlapOperator {
  public:
    SingleProjectorOverlapOperator(std::vector<Complex> projector, Real strength)
        : projector_(std::move(projector)), strength_(strength) {}

    void apply(int number_of_plane_waves,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        Complex projector_overlap(0.0, 0.0);
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            projector_overlap += std::conj(projector_[g_index]) * wavefunction[g_index];
        }
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            overlap_wavefunction[g_index] =
                wavefunction[g_index] + strength_ * projector_[g_index] * projector_overlap;
        }
    }

    void build_dense_matrix(int number_of_plane_waves,
                            std::vector<Complex>& overlap_matrix) const override {
        overlap_matrix.assign(static_cast<std::size_t>(number_of_plane_waves) *
                                  number_of_plane_waves,
                              Complex(0.0, 0.0));
        for (int column = 0; column < number_of_plane_waves; column++) {
            for (int row = 0; row < number_of_plane_waves; row++) {
                Complex value = (row == column) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
                value += strength_ * projector_[row] * std::conj(projector_[column]);
                overlap_matrix[static_cast<std::size_t>(row) + column * number_of_plane_waves] =
                    value;
            }
        }
    }

  private:
    std::vector<Complex> projector_;
    Real strength_;
};

/// S = I written out as an operator, to separate "no operator" from "S is the identity".
class IdentityOverlapOperator : public OverlapOperator {
  public:
    void apply(int number_of_plane_waves,
               const Complex* wavefunction,
               Complex* overlap_wavefunction) const override {
        for (int g_index = 0; g_index < number_of_plane_waves; g_index++) {
            overlap_wavefunction[g_index] = wavefunction[g_index];
        }
    }

    [[nodiscard]] bool is_identity() const override {
        return true;
    }

    void build_dense_matrix(int number_of_plane_waves,
                            std::vector<Complex>& overlap_matrix) const override {
        overlap_matrix.assign(static_cast<std::size_t>(number_of_plane_waves) *
                                  number_of_plane_waves,
                              Complex(0.0, 0.0));
        for (int index = 0; index < number_of_plane_waves; index++) {
            overlap_matrix[static_cast<std::size_t>(index) + index * number_of_plane_waves] =
                Complex(1.0, 0.0);
        }
    }
};

PseudopotentialSet load_nickel_pseudopotential() {
    PseudopotentialSet pseudopotential_set;
    pseudopotential_set.push_back(
        UPFReader::read(std::string(EXAMPLES_DIR) + "/data/pseudopotentials/nc/Ni.upf"));
    return pseudopotential_set;
}

Crystal make_nickel_fcc(const PseudopotentialSet& pseudopotential_set) {
    Crystal crystal_structure;
    crystal_structure.set_from_lattice_type(2, 6.65);
    AtomSpecies nickel{"Ni", 58.69, pseudopotential_set[0].valence_charge, "Ni.upf"};
    crystal_structure.add_species(nickel);
    crystal_structure.add_atom({0, {0.0, 0.0, 0.0}});
    crystal_structure.compute_derived_quantities();
    return crystal_structure;
}

struct Fixture {
    PseudopotentialSet pseudopotential_set;
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::vector<int> g_vector_indices;
    int number_of_plane_waves = 0;

    Fixture()
        : pseudopotential_set(load_nickel_pseudopotential())
        , crystal_structure(make_nickel_fcc(pseudopotential_set)) {
        basis.generate(serial_policy, crystal_structure, 30.0, 120.0);
        kpoint_grid.generate(2, 2, 2, 1, 1, 1);
        std::vector<Real> kinetic_energy_squared;
        basis.compute_gk(kpoint_grid[0].coordinates,
                         crystal_structure,
                         g_vector_indices,
                         kinetic_energy_squared);
        number_of_plane_waves = static_cast<int>(g_vector_indices.size());
    }

    /// The calculator's own atomic orbitals, before any orthogonalization.
    [[nodiscard]] std::vector<std::vector<Complex>> build_atomic_orbitals() const {
        auto atomic_wavefunction_table = tabulate_atomic_wavefunctions_in_q_space(
            pseudopotential_set, crystal_structure.cell_volume(), 0.01, 900);
        std::vector<std::vector<Complex>> orbitals;
        build_atomic_orbitals_g_space(0,
                                      number_of_plane_waves,
                                      g_vector_indices,
                                      crystal_structure,
                                      kpoint_grid,
                                      basis,
                                      pseudopotential_set,
                                      atomic_wavefunction_table,
                                      0.01,
                                      900,
                                      orbitals);
        return orbitals;
    }
};

/**
 * A quantum system whose bands are exactly `bands`, so the projections of
 * those synthetic "converged" states are analytically known.
 */
PlanewaveKohnShamQuantumSystem
build_quantum_system_from_bands(const Fixture& fixture,
                                const std::vector<std::vector<Complex>>& bands) {
    int number_of_bands = static_cast<int>(bands.size());
    PlanewaveKohnShamQuantumSystem quantum_system(fixture.kpoint_grid,
                                                  fixture.basis,
                                                  number_of_bands,
                                                  /*number_of_spin_components_value=*/1,
                                                  /*number_of_electrons=*/1.0);
    quantum_system.allocate_electronic_states(
        std::vector<int>(fixture.kpoint_grid.number_of_kpoints(), fixture.number_of_plane_waves));
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        std::memcpy(quantum_system.wavefunction_band_data(0, band_index),
                    bands[band_index].data(),
                    fixture.number_of_plane_waves * sizeof(Complex));
    }
    return quantum_system;
}

std::vector<std::vector<Real>> project(const Fixture& fixture,
                                       const PlanewaveWavefunctionProjectionCalculator& calculator,
                                       const PlanewaveKohnShamQuantumSystem& quantum_system,
                                       int number_of_bands,
                                       const OverlapOperator* overlap_operator) {
    return calculator.compute_projections_at_kpoint(0,
                                                    0,
                                                    fixture.number_of_plane_waves,
                                                    number_of_bands,
                                                    fixture.g_vector_indices,
                                                    fixture.crystal_structure,
                                                    fixture.kpoint_grid,
                                                    fixture.basis,
                                                    fixture.pseudopotential_set,
                                                    quantum_system,
                                                    overlap_operator);
}

/**
 * A normalized beta function to build the augmentation overlap from: the
 * fixture's own first atomic orbital, so the augmentation term is guaranteed
 * to reach the orbital set rather than living in a corner of the plane-wave
 * sphere the atomic orbitals barely occupy. That is also how a real beta
 * function sits relative to them -- both are localized on the same atom.
 */
std::vector<Complex> make_projector(const std::vector<std::vector<Complex>>& orbitals) {
    std::vector<Complex> projector = orbitals[0];
    Real norm = 0.0;
    for (const Complex& component : projector) {
        norm += std::norm(component);
    }
    Real scale = 1.0 / std::sqrt(norm);
    for (Complex& component : projector) {
        component *= scale;
    }
    return projector;
}

} // namespace

/**
 * ── A converged state equal to one ortho-atomic orbital projects onto ──
 * ── itself with weight 1 and onto every other orbital with weight 0.  ──
 *
 * This exploits the already-proven orthonormality of the Löwdin basis
 * (test_lowdin_orthogonalization.cpp) as ground truth: feeding the
 * calculator's own orthogonalized orbitals back in as synthetic "converged
 * wavefunctions" and checking the round-trip projections come out as the
 * identity matrix exercises both the orbital reconstruction and the
 * projection formula together.
 */
TEST(PlanewaveWavefunctionProjectionCalculator, KnownOrbitalsProjectAsIdentityMatrix) {
    Fixture fixture;

    PlanewaveWavefunctionProjectionCalculator calculator(fixture.crystal_structure,
                                                         fixture.pseudopotential_set);
    ASSERT_EQ(calculator.number_of_orbitals(), 10);

    /**
     * Independently reproduce the calculator's own ortho-atomic orbitals to
     * build synthetic "converged" wavefunctions from them.
     */
    std::vector<std::vector<Complex>> reference_orbitals = fixture.build_atomic_orbitals();
    lowdin_orthogonalize_orbitals(reference_orbitals);

    int number_of_bands = calculator.number_of_orbitals(); // one band per orbital, for this test
    PlanewaveKohnShamQuantumSystem quantum_system =
        build_quantum_system_from_bands(fixture, reference_orbitals);

    auto projections = project(fixture, calculator, quantum_system, number_of_bands, nullptr);

    ASSERT_EQ(static_cast<int>(projections.size()), number_of_bands);
    Real tolerance = 1e-8;
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        ASSERT_EQ(static_cast<int>(projections[band_index].size()),
                  calculator.number_of_orbitals());
        for (int orbital_index = 0; orbital_index < calculator.number_of_orbitals();
             orbital_index++) {
            Real expected = (band_index == orbital_index) ? 1.0 : 0.0;
            EXPECT_NEAR(projections[band_index][orbital_index], expected, tolerance)
                << "band " << band_index << " orbital " << orbital_index;
        }
    }
}

/**
 * ── Norm-conserving results do not move when S is written out ─────────
 *
 * A norm-conserving run passes no overlap operator at all, and one that
 * passes an explicit S = I must get bit-for-bit the same weights. This is
 * the local half of the claim the parity matrix makes globally: the S metric
 * changes nothing where S is the identity.
 */
TEST(PlanewaveWavefunctionProjectionCalculator, ExplicitIdentityOverlapMatchesNoOverlapExactly) {
    Fixture fixture;

    PlanewaveWavefunctionProjectionCalculator calculator(fixture.crystal_structure,
                                                         fixture.pseudopotential_set);
    std::vector<std::vector<Complex>> bands = fixture.build_atomic_orbitals();
    lowdin_orthogonalize_orbitals(bands);
    int number_of_bands = static_cast<int>(bands.size());
    PlanewaveKohnShamQuantumSystem quantum_system = build_quantum_system_from_bands(fixture, bands);

    auto without_operator = project(fixture, calculator, quantum_system, number_of_bands, nullptr);
    IdentityOverlapOperator identity;
    auto with_identity = project(fixture, calculator, quantum_system, number_of_bands, &identity);

    ASSERT_EQ(with_identity.size(), without_operator.size());
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        ASSERT_EQ(with_identity[band_index].size(), without_operator[band_index].size());
        for (int orbital_index = 0; orbital_index < calculator.number_of_orbitals();
             orbital_index++) {
            EXPECT_EQ(with_identity[band_index][orbital_index],
                      without_operator[band_index][orbital_index])
                << "band " << band_index << " orbital " << orbital_index;
        }
    }
}

/**
 * ── The S metric, and what taking it in the wrong metric costs ────────
 *
 * With a non-identity S, the states that are orthonormal are the S-orthonormal
 * ones, so bands built from S-metric Löwdin orbitals must again project as the
 * identity matrix -- each band's weights summing to exactly 1, which is the
 * per-band form of the sum rule the Löwdin charges integrate into the valence
 * electron count. The same bands projected in the plain metric do not: that
 * deviation is precisely the charge a PDOS taken in the wrong metric loses or
 * invents.
 */
TEST(PlanewaveWavefunctionProjectionCalculator, SMetricProjectionsSatisfyThePerBandSumRule) {
    Fixture fixture;
    std::vector<std::vector<Complex>> bands = fixture.build_atomic_orbitals();
    SingleProjectorOverlapOperator overlap_operator(make_projector(bands), 4.0);

    PlanewaveWavefunctionProjectionCalculator calculator(fixture.crystal_structure,
                                                         fixture.pseudopotential_set);

    lowdin_orthogonalize_orbitals(
        bands, compute_orbital_overlap_matrix_in_s_metric(bands, &overlap_operator));
    int number_of_bands = static_cast<int>(bands.size());
    PlanewaveKohnShamQuantumSystem quantum_system = build_quantum_system_from_bands(fixture, bands);

    auto projections =
        project(fixture, calculator, quantum_system, number_of_bands, &overlap_operator);
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        Real band_sum = 0.0;
        for (int orbital_index = 0; orbital_index < calculator.number_of_orbitals();
             orbital_index++) {
            Real expected = (band_index == orbital_index) ? 1.0 : 0.0;
            EXPECT_NEAR(projections[band_index][orbital_index], expected, 1e-8)
                << "band " << band_index << " orbital " << orbital_index;
            band_sum += projections[band_index][orbital_index];
        }
        EXPECT_NEAR(band_sum, 1.0, 1e-8) << "band " << band_index;
    }

    auto wrong_metric_projections =
        project(fixture, calculator, quantum_system, number_of_bands, nullptr);
    Real largest_band_sum_error = 0.0;
    for (int band_index = 0; band_index < number_of_bands; band_index++) {
        Real band_sum = 0.0;
        for (int orbital_index = 0; orbital_index < calculator.number_of_orbitals();
             orbital_index++) {
            band_sum += wrong_metric_projections[band_index][orbital_index];
        }
        largest_band_sum_error = std::max(largest_band_sum_error, std::abs(band_sum - 1.0));
    }
    EXPECT_GT(largest_band_sum_error, 1e-3)
        << "the plain metric happened to satisfy the sum rule too, so this fixture no longer "
           "distinguishes the two metrics";
}

// ── Orbital labels are exposed and match enumerate_atomic_orbitals ─────

TEST(PlanewaveWavefunctionProjectionCalculator, OrbitalLabelsMatchEnumeration) {
    auto pseudopotential_set = load_nickel_pseudopotential();
    Crystal crystal_structure = make_nickel_fcc(pseudopotential_set);

    PlanewaveWavefunctionProjectionCalculator calculator(crystal_structure, pseudopotential_set);
    auto expected_labels = enumerate_atomic_orbitals(crystal_structure, pseudopotential_set);

    ASSERT_EQ(calculator.orbital_labels().size(), expected_labels.size());
    for (size_t i = 0; i < expected_labels.size(); i++) {
        EXPECT_EQ(calculator.orbital_labels()[i].shell_label, expected_labels[i].shell_label);
        EXPECT_EQ(calculator.orbital_labels()[i].angular_momentum,
                  expected_labels[i].angular_momentum);
        EXPECT_EQ(calculator.orbital_labels()[i].magnetic_quantum_number,
                  expected_labels[i].magnetic_quantum_number);
    }
}
