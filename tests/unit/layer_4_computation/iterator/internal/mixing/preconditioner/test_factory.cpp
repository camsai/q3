#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "layer_4_computation/iterator/internal/mixing/preconditioner/factory.hpp"

using namespace q3;

namespace {

PreconditionerEnvironment make_environment() {
    PreconditionerEnvironment environment;
    environment.g_squared = {0.0, 1.0, 4.0, 9.0};
    environment.q0_squared = 1.0;
    environment.first_nonzero_g_index = 1;
    environment.to_reciprocal_space = [](const std::vector<Real>& real_space,
                                         std::vector<Complex>& reciprocal) {
        reciprocal.assign(real_space.size(), Complex(0.0, 0.0));
        for (std::size_t index = 0; index < real_space.size(); ++index) {
            reciprocal[index] = Complex(real_space[index], 0.0);
        }
    };
    environment.to_real_space = [](const std::vector<Complex>& reciprocal,
                                   std::vector<Real>& real_space) {
        real_space.assign(reciprocal.size(), 0.0);
        for (std::size_t index = 0; index < reciprocal.size(); ++index) {
            real_space[index] = reciprocal[index].real();
        }
    };
    return environment;
}

} // namespace

TEST(PreconditionerFactory, CreatesEachSupportedName) {
    auto environment = make_environment();
    EXPECT_EQ(PreconditionerFactory::create("none", environment)->name(), "none");
    EXPECT_EQ(PreconditionerFactory::create("kerker", environment)->name(), "kerker");
    EXPECT_EQ(PreconditionerFactory::create("local-tf", environment)->name(), "local-tf");
}

/**
 * An unknown name must throw. Before the preconditioner was an object, the
 * iterator tested `== "kerker"` and quietly ran unpreconditioned on anything
 * else, so a typo cost a slow run and produced no diagnostic.
 */
TEST(PreconditionerFactory, UnknownNameThrows) {
    auto environment = make_environment();
    EXPECT_THROW(PreconditionerFactory::create("kerkar", environment), std::runtime_error);
    EXPECT_THROW(PreconditionerFactory::create("", environment), std::runtime_error);
    EXPECT_THROW(PreconditionerFactory::create("default", environment), std::runtime_error);
}

/// "default" is resolved by the SCF iterator before it ever reaches the factory.
TEST(PreconditionerFactory, IsSupportedMatchesWhatCreateAccepts) {
    EXPECT_TRUE(PreconditionerFactory::is_supported("none"));
    EXPECT_TRUE(PreconditionerFactory::is_supported("kerker"));
    EXPECT_TRUE(PreconditionerFactory::is_supported("local-tf"));
    EXPECT_FALSE(PreconditionerFactory::is_supported("default"));
    EXPECT_FALSE(PreconditionerFactory::is_supported("kerkar"));
}

/// The identity leaves every component alone, G = 0 included.
TEST(PreconditionerFactory, NoneIsTheIdentity) {
    auto environment = make_environment();
    auto preconditioner = PreconditionerFactory::create("none", environment);
    std::vector<Complex> residual = {{1.0, 2.0}, {3.0, -1.0}, {0.5, 0.25}, {-2.0, 0.0}};
    std::vector<Complex> original = residual;
    preconditioner->apply(residual);
    for (std::size_t index = 0; index < residual.size(); ++index) {
        EXPECT_DOUBLE_EQ(residual[index].real(), original[index].real());
        EXPECT_DOUBLE_EQ(residual[index].imag(), original[index].imag());
    }
}
