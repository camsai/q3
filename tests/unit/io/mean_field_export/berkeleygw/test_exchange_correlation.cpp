#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/mean_field_export/berkeleygw/exchange_correlation.hpp"
#include "io/mean_field_record/from_scf_state.hpp"
#include "layer_0_core/types/constants.hpp"
#include "layer_1_structure/crystal/crystal.hpp"
#include "layer_1_structure/reciprocal_space/kpoints_grid.hpp"
#include "layer_3_representation/basis/planewave/init.hpp"
#include "layer_3_representation/quantum_system/kohn_sham/planewave/norm_conserving/init.hpp"
#include "layer_4_computation/iterator/kohn_sham/planewave/execution/serial.hpp"
#include "layer_4_computation/state/kohn_sham/planewave/norm_conserving/init.hpp"
#include "utils/temporary_path.hpp"

using namespace q3;

namespace {

execution::PlanewaveSerialExecutionPolicy serial_policy;

/**
 * Minimal single-atom, single-(unshifted-Gamma)-k-point fixture -- small
 * enough that hand-computed matrix elements stay tractable, but still a
 * real (non-gamma-only, since is_gamma_only defaults to false regardless
 * of the k-point sampling itself) charge-density G-set with plenty of
 * plane waves at k=(0,0,0) to pick distinct G-vectors from.
 */
void populate_synthetic_state(
    Crystal& crystal_structure,
    PlanewaveBasis& basis,
    KPointsGrid& kpoint_grid,
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem>& quantum_system,
    NormConservingPlanewaveKohnShamSCFState& scf_state,
    int number_of_bands = 1) {
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}};
    crystal_structure = Crystal::from_input_data(1, 10.0, species, atoms);

    basis.generate(serial_policy, crystal_structure, 8.0, 32.0); // dual = 4 (production default)
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(number_of_bands);
    quantum_system->set_number_of_electrons(4.0);
    quantum_system->number_of_spin_components = 1;
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;
}

std::string make_temporary_path(const std::string& suffix) {
    return test_support::make_temporary_path(suffix);
}

/// Linear search for a Miller index among the (unsorted-by-Miller) charge-density G-vectors.
int find_global_g_vector_index(const std::vector<GVector>& g_vectors,
                               const IntegerVector3D& miller) {
    for (int index = 0; index < static_cast<int>(g_vectors.size()); index++) {
        if (g_vectors[index].miller == miller) {
            return index;
        }
    }
    return -1;
}

/// One parsed `FORMAT (2I8, 2F15.9)` diagonal-matrix-element line: spin band Re Im.
struct ParsedDiagonalLine {
    int spin = 0;
    int band = 0;
    double real_part = 0.0;
    double imaginary_part = 0.0;
};

/**
 * Independently re-implements the fixed-width slicing that
 * tests/helpers_berkeleygw.py's read_berkeleygw_exchange_correlation_file()
 * uses (widths (8,8,15,15), no separating whitespace assumed) -- written
 * fresh here rather than calling the writer's own formatting helpers, so
 * this is a genuine check of write_to_file()'s actual byte output, not a
 * tautology against itself.
 */
ParsedDiagonalLine parse_diagonal_line(const std::string& line) {
    if (line.size() < 46) {
        throw std::runtime_error("diagonal line too short: '" + line + "'");
    }
    ParsedDiagonalLine parsed;
    parsed.spin = std::stoi(line.substr(0, 8));
    parsed.band = std::stoi(line.substr(8, 8));
    parsed.real_part = std::stod(line.substr(16, 15));
    parsed.imaginary_part = std::stod(line.substr(31, 15));
    return parsed;
}

} // namespace

// ── Scope restrictions ───────────────────────────────────────────────────

TEST(BerkeleyGWExchangeCorrelationFile, RejectsSpinPolarizedStateBeforeWritingAnything) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);
    quantum_system->number_of_spin_components = 2;

    std::string path = make_temporary_path(".vxc_spin_polarized");
    try {
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWExchangeCorrelationFile(record).write_to_file(path);
        FAIL() << "expected write_to_file to throw for nspin>1";
    } catch (const std::runtime_error& error) {
        std::string message = error.what();
        EXPECT_NE(message.find("spin"), std::string::npos) << message;
    }

    // No partial/corrupt file should be left behind.
    EXPECT_FALSE(std::ifstream(path).good());
    std::remove(path.c_str());
}

TEST(BerkeleyGWExchangeCorrelationFile, RejectsGammaOnlyStateBeforeWritingAnything) {
    Crystal crystal_structure;
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}};
    crystal_structure = Crystal::from_input_data(1, 10.0, species, atoms);

    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 8.0, 32.0, /*is_gamma_only=*/true);
    KPointsGrid kpoint_grid;
    kpoint_grid.generate(1, 1, 1, 0, 0, 0);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);

    auto quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(1);
    quantum_system->set_number_of_electrons(4.0);
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;
    scf_state.exchange_correlation_potential_g.assign(basis.g_vectors().size(), Complex(0.0, 0.0));

    std::string path = make_temporary_path(".vxc_gamma_only");
    try {
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWExchangeCorrelationFile(record).write_to_file(path);
        FAIL() << "expected write_to_file to throw for gamma-only";
    } catch (const std::runtime_error& error) {
        std::string message = error.what();
        EXPECT_NE(message.find("gamma-only"), std::string::npos) << message;
    }
    EXPECT_FALSE(std::ifstream(path).good());
    std::remove(path.c_str());
}

TEST(BerkeleyGWExchangeCorrelationFile, ThrowsWhenExchangeCorrelationPotentialGIsMissing) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);
    // scf_state.exchange_correlation_potential_g deliberately left empty/unpopulated.

    std::string path = make_temporary_path(".vxc_missing_data");
    try {
        const MeanFieldRecord record = build_mean_field_record(scf_state);
        BerkeleyGWExchangeCorrelationFile(record).write_to_file(path);
        FAIL() << "expected write_to_file to throw when Vxc(G) is unavailable";
    } catch (const std::runtime_error& error) {
        std::string message = error.what();
        EXPECT_NE(message.find("exchange_correlation_potential_g"), std::string::npos) << message;
    }
    EXPECT_FALSE(std::ifstream(path).good());
    std::remove(path.c_str());
}

// ── Physics: G-space convolution correctness ─────────────────────────────

/**
 * V_xc(r) = V0 (constant, i.e. only Vxc(G=0) is nonzero) makes
 * <psi|Vxc|psi> = V0 * sum_G|c(G)|^2 = V0 exactly for ANY normalized psi --
 * every G != G' term vanishes (Vxc(G-G') = 0 for G != G'), and every G = G'
 * term contributes |c(G)|^2 * V0 regardless of which G it is. This pins
 * down both the overall normalization convention and the G=0 self-term
 * handling. Uses two distinct nonzero coefficients (not just c(G=0)=1) so
 * the "different G's still only see Vxc(0)" part of the claim is actually
 * exercised, not merely the single-plane-wave special case.
 */
TEST(BerkeleyGWExchangeCorrelationFile, ConstantPotentialGivesV0ForNormalizedWavefunction) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    const auto& g_vectors = basis.g_vectors();
    const auto& g_indices = basis.kpoint_g_vector_indices()[0];
    ASSERT_GE(g_indices.size(), 2u) << "need at least 2 plane waves at Gamma for this test";

    const Real v0_ry = 0.37;
    int gamma_index = find_global_g_vector_index(g_vectors, IntegerVector3D(0, 0, 0));
    ASSERT_GE(gamma_index, 0) << "G=(0,0,0) must always be a stored charge-density G-vector";
    scf_state.exchange_correlation_potential_g.assign(g_vectors.size(), Complex(0.0, 0.0));
    scf_state.exchange_correlation_potential_g[gamma_index] = Complex(v0_ry, 0.0);

    // 0.6^2 + 0.8^2 = 1.0 exactly -- an exact Pythagorean normalization.
    Complex* coefficients = quantum_system->wavefunction_band_data(0, 0);
    coefficients[0] = Complex(0.6, 0.0);
    coefficients[1] = Complex(0.0, 0.8);
    for (size_t index = 2; index < g_indices.size(); index++) {
        coefficients[index] = Complex(0.0, 0.0);
    }

    std::string path = make_temporary_path(".vxc_constant_potential");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWExchangeCorrelationFile(record).write_to_file(path);

    std::ifstream file(path);
    std::string header_line;
    ASSERT_TRUE(std::getline(file, header_line));
    std::string diagonal_line;
    ASSERT_TRUE(std::getline(file, diagonal_line));
    ParsedDiagonalLine parsed = parse_diagonal_line(diagonal_line);

    EXPECT_EQ(parsed.spin, 1);
    EXPECT_EQ(parsed.band, 1);
    EXPECT_NEAR(parsed.real_part, v0_ry * constants::RY_TO_EV, 1.0e-6);
    EXPECT_NEAR(parsed.imaginary_part, 0.0, 1.0e-6);

    std::remove(path.c_str());
}

/**
 * A single nonzero Vxc(G) beyond G=0, computed by hand. With coefficients
 * c_a = c_b = 1/sqrt(2) on two distinct plane waves G_a, G_b (all others
 * zero), Vxc(0) = 0, and Vxc(G_a - G_b) = V, Vxc(G_b - G_a) = conj(V) (the
 * Hermiticity Vxc(r) real requires), only the cross terms survive:
 *   <psi|Vxc|psi> = conj(c_a) V c_b + conj(c_b) conj(V) c_a = 2*Re(conj(c_a) V c_b)
 * With c_a = c_b = 1/sqrt(2) (real) and V = 0.5 + 0.1i:
 *   conj(c_a) V c_b = 0.5 * (0.5 + 0.1i) = 0.25 + 0.05i => 2*Re(...) = 0.5 Ry exactly.
 */
TEST(BerkeleyGWExchangeCorrelationFile, OneNonzeroOffGammaVxcComponentMatchesHandComputedValue) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    const auto& g_vectors = basis.g_vectors();
    const auto& g_indices = basis.kpoint_g_vector_indices()[0];
    ASSERT_GE(g_indices.size(), 2u);

    IntegerVector3D miller_a = g_vectors[g_indices[0]].miller;
    IntegerVector3D miller_b = g_vectors[g_indices[1]].miller;
    IntegerVector3D difference(
        miller_a.x - miller_b.x, miller_a.y - miller_b.y, miller_a.z - miller_b.z);
    IntegerVector3D negated_difference(-difference.x, -difference.y, -difference.z);
    ASSERT_FALSE(difference == IntegerVector3D(0, 0, 0))
        << "two distinct plane waves must have distinct Miller indices";

    int difference_index = find_global_g_vector_index(g_vectors, difference);
    int negated_difference_index = find_global_g_vector_index(g_vectors, negated_difference);
    ASSERT_GE(difference_index, 0) << "G_a - G_b must fall within the dual=4 charge-density G-set";
    ASSERT_GE(negated_difference_index, 0);

    Complex v(0.5, 0.1);
    scf_state.exchange_correlation_potential_g.assign(g_vectors.size(), Complex(0.0, 0.0));
    scf_state.exchange_correlation_potential_g[difference_index] = v;
    scf_state.exchange_correlation_potential_g[negated_difference_index] = std::conj(v);

    Complex* coefficients = quantum_system->wavefunction_band_data(0, 0);
    Real inverse_sqrt2 = 1.0 / std::sqrt(2.0);
    coefficients[0] = Complex(inverse_sqrt2, 0.0);
    coefficients[1] = Complex(inverse_sqrt2, 0.0);
    for (size_t index = 2; index < g_indices.size(); index++) {
        coefficients[index] = Complex(0.0, 0.0);
    }

    std::string path = make_temporary_path(".vxc_hand_computed");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWExchangeCorrelationFile(record).write_to_file(path);

    std::ifstream file(path);
    std::string header_line;
    ASSERT_TRUE(std::getline(file, header_line));
    std::string diagonal_line;
    ASSERT_TRUE(std::getline(file, diagonal_line));
    ParsedDiagonalLine parsed = parse_diagonal_line(diagonal_line);

    Real expected_ry = 0.5;
    EXPECT_NEAR(parsed.real_part, expected_ry * constants::RY_TO_EV, 1.0e-6);
    EXPECT_NEAR(parsed.imaginary_part, 0.0, 1.0e-6);

    std::remove(path.c_str());
}

/**
 * Vxc(r) real forces Vxc(-G) = conj(Vxc(G)), which in turn forces every
 * DIAGONAL <psi_nk|Vxc|psi_nk> to be real -- an algebraic identity that
 * holds for ANY coefficients (normalized or not) and ANY Hermitian-
 * symmetric G-space potential, not just the hand-picked single-component
 * cases above. Builds a Hermitian-symmetric Vxc(G) across the WHOLE
 * charge-density G-set (real part even in G, imaginary part odd in G) and
 * a wavefunction spread over several plane waves with no analytically
 * convenient answer, then checks only the imaginary part -- a real defect
 * (index or conjugation bug) would show up as an O(0.1-1) eV-scale
 * residual, not floating-point noise.
 */
TEST(BerkeleyGWExchangeCorrelationFile, ImaginaryPartIsNegligibleForGenericHermitianPotential) {
    Crystal crystal_structure;
    PlanewaveBasis basis;
    KPointsGrid kpoint_grid;
    std::shared_ptr<NormConservingPlanewaveKohnShamQuantumSystem> quantum_system;
    NormConservingPlanewaveKohnShamSCFState scf_state;
    populate_synthetic_state(crystal_structure, basis, kpoint_grid, quantum_system, scf_state);

    const auto& g_vectors = basis.g_vectors();
    scf_state.exchange_correlation_potential_g.resize(g_vectors.size());
    for (size_t index = 0; index < g_vectors.size(); index++) {
        const IntegerVector3D& miller = g_vectors[index].miller;
        Real real_part = 0.2 / (1.0 + miller.x * miller.x + miller.y * miller.y +
                                miller.z * miller.z); // even in G: function of |G|^2 only
        Real imaginary_part =
            0.01 * (miller.x + 2.0 * miller.y + 3.0 * miller.z); // odd in G: linear in components
        scf_state.exchange_correlation_potential_g[index] = Complex(real_part, imaginary_part);
    }

    const auto& g_indices = basis.kpoint_g_vector_indices()[0];
    int number_of_plane_waves = static_cast<int>(g_indices.size());
    ASSERT_GE(number_of_plane_waves, 6);
    Complex* coefficients = quantum_system->wavefunction_band_data(0, 0);
    Real norm_squared = 0.0;
    for (int index = 0; index < number_of_plane_waves; index++) {
        Complex value(std::sin(0.3 * index + 0.1), 0.3 * std::cos(0.2 * index + 0.7));
        coefficients[index] = value;
        norm_squared += std::norm(value);
    }
    Real scale = 1.0 / std::sqrt(norm_squared);
    for (int index = 0; index < number_of_plane_waves; index++) {
        coefficients[index] *= scale;
    }

    std::string path = make_temporary_path(".vxc_hermitian_imag_check");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWExchangeCorrelationFile(record).write_to_file(path);

    std::ifstream file(path);
    std::string header_line;
    ASSERT_TRUE(std::getline(file, header_line));
    std::string diagonal_line;
    ASSERT_TRUE(std::getline(file, diagonal_line));
    ParsedDiagonalLine parsed = parse_diagonal_line(diagonal_line);

    EXPECT_NEAR(parsed.imaginary_part, 0.0, 1.0e-6)
        << "a real Vxc(r) forces the diagonal matrix element to be real; a residual this size "
           "indicates an index or conjugation bug";

    std::remove(path.c_str());
}

// ── Exact on-disk formatting ──────────────────────────────────────────────

/**
 * Compares the FULL first two lines of a known, hand-constructed case
 * against LITERAL expected strings (independently computed via Python's
 * identical IEEE-754 `%f`/`%d` formatting, not by calling back into the
 * writer's own formatting helpers), pinning down `FORMAT (3F13.9, 2I8)` and
 * `FORMAT (2I8, 2F15.9)`'s exact column widths byte-for-byte -- the
 * independent reader (tests/helpers_berkeleygw.py) parses by fixed width
 * with no separator, so an off-by-one column here would silently corrupt
 * every downstream field.
 */
TEST(BerkeleyGWExchangeCorrelationFile, FileContentsMatchExactFixedWidthColumnsForKnownValue) {
    Crystal crystal_structure;
    std::vector<AtomSpecies> species = {AtomSpecies{"Si", 28.0855, 4.0, "Si.upf"}};
    std::vector<Atom> atoms = {Atom{0, {0.0, 0.0, 0.0}}};
    crystal_structure = Crystal::from_input_data(1, 10.0, species, atoms);

    PlanewaveBasis basis;
    basis.generate(serial_policy, crystal_structure, 8.0, 32.0);
    KPointsGrid kpoint_grid;
    /**
     * dim=2, shift=1 along x puts kpoints at exactly x = +0.25 and x = -0.25
     * (KPointsGrid::generate()'s own formula: 0/2 + 0.5/2 = 0.25, folded 1/2+0.5/2=0.75 -> -0.25),
     * both exactly representable in binary floating point -- a clean, hand-verifiable F13.9 case.
     */
    kpoint_grid.generate(2, 1, 1, 1, 0, 0);
    basis.initialize_kpoint_data(crystal_structure, kpoint_grid);
    ASSERT_NEAR(kpoint_grid[0].coordinates.x, 0.25, 1.0e-12);
    ASSERT_NEAR(kpoint_grid[0].coordinates.y, 0.0, 1.0e-12);
    ASSERT_NEAR(kpoint_grid[0].coordinates.z, 0.0, 1.0e-12);

    auto quantum_system = std::make_shared<NormConservingPlanewaveKohnShamQuantumSystem>();
    quantum_system->set_kpoint_grid(&kpoint_grid);
    quantum_system->set_basis(&basis);
    quantum_system->set_number_of_bands(1);
    quantum_system->set_number_of_electrons(2.0);
    quantum_system->allocate_electronic_states(basis.kpoint_number_of_plane_waves());

    NormConservingPlanewaveKohnShamSCFState scf_state;
    scf_state.crystal_structure = &crystal_structure;
    scf_state.kpoint_grid = &kpoint_grid;
    scf_state.planewave_basis = &basis;
    scf_state.quantum_system_snapshot_ = quantum_system;

    const auto& g_vectors = basis.g_vectors();
    int gamma_index = find_global_g_vector_index(g_vectors, IntegerVector3D(0, 0, 0));
    ASSERT_GE(gamma_index, 0);
    scf_state.exchange_correlation_potential_g.assign(g_vectors.size(), Complex(0.0, 0.0));
    /**
     * Vxc(G=0) = 1.0 Ry exactly, c(G=0) = 1.0 -- <psi|Vxc|psi> = 1.0 Ry = RY_TO_EV eV exactly,
     * by the same "only G=G' survives" argument as
     * ConstantPotentialGivesV0ForNormalizedWavefunction.
     */
    scf_state.exchange_correlation_potential_g[gamma_index] = Complex(1.0, 0.0);

    const auto& g_indices_k0 = basis.kpoint_g_vector_indices()[0];
    int local_gamma_index = -1;
    for (int index = 0; index < static_cast<int>(g_indices_k0.size()); index++) {
        if (g_vectors[g_indices_k0[index]].miller == IntegerVector3D(0, 0, 0)) {
            local_gamma_index = index;
            break;
        }
    }
    ASSERT_GE(local_gamma_index, 0);
    Complex* coefficients_k0 = quantum_system->wavefunction_band_data(0, 0);
    for (int index = 0; index < static_cast<int>(g_indices_k0.size()); index++) {
        coefficients_k0[index] = Complex(0.0, 0.0);
    }
    coefficients_k0[local_gamma_index] = Complex(1.0, 0.0);

    /**
     * k-point 1 is irrelevant to what this test checks (only the first two lines, for k-point
     * 0), but must still be a valid normalized state for write_to_file() to succeed at all.
     */
    const auto& g_indices_k1 = basis.kpoint_g_vector_indices()[1];
    Complex* coefficients_k1 = quantum_system->wavefunction_band_data(1, 0);
    for (int index = 0; index < static_cast<int>(g_indices_k1.size()); index++) {
        coefficients_k1[index] = Complex(0.0, 0.0);
    }
    coefficients_k1[0] = Complex(1.0, 0.0);

    std::string path = make_temporary_path(".vxc_fixed_width");
    const MeanFieldRecord record = build_mean_field_record(scf_state);
    BerkeleyGWExchangeCorrelationFile(record).write_to_file(path);

    std::ifstream file(path);
    std::string header_line;
    ASSERT_TRUE(std::getline(file, header_line));
    std::string diagonal_line;
    ASSERT_TRUE(std::getline(file, diagonal_line));

    /**
     * Independently computed (Python, same IEEE-754 %f/%d semantics as C's printf):
     *   "%13.9f" % 0.25 == "  0.250000000"; "%13.9f" % 0.0 == "  0.000000000"
     *   "%8d" % 1 == "       1"; "%8d" % 0 == "       0"
     *   RY_TO_EV = 13.605693122994017; "%15.9f" % that == "   13.605693123"
     *   "%15.9f" % 0.0 == "    0.000000000"
     */
    std::string expected_header_line = "  0.250000000  0.000000000  0.000000000       1       0";
    std::string expected_diagonal_line = "       1       1   13.605693123    0.000000000";

    EXPECT_EQ(header_line, expected_header_line);
    EXPECT_EQ(diagonal_line, expected_diagonal_line);

    std::remove(path.c_str());
}
