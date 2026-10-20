/**
 * Registry and NamedFactory: name lookup, the one-entry-per-name rule, and
 * registration order.
 */

#include "utils/named_factory.hpp"
#include "utils/registry.hpp"
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace q3;

namespace {

/// A product a NamedFactory can build, carrying which builder made it.
struct Widget {
    std::string made_by;
};

using WidgetFactory = NamedFactory<Widget>;

std::unique_ptr<WidgetFactory> widget_factory(std::vector<std::string> names) {
    const std::string builder_name = names.front();
    return std::make_unique<WidgetFactory>(std::move(names), [builder_name] {
        return std::make_unique<Widget>(Widget{builder_name});
    });
}

} // namespace

TEST(RegistryTest, FindsAnEntryByAnyOfItsNames) {
    Registry<WidgetFactory> registry;
    registry.add(widget_factory({"alpha", "first"}));
    ASSERT_NE(registry.find("first"), nullptr);
    EXPECT_EQ(registry.find("first")->create()->made_by, "alpha");
    EXPECT_EQ(registry.find("missing"), nullptr);
}

TEST(RegistryTest, RefusesASecondEntryUnderATakenName) {
    Registry<WidgetFactory> registry;
    registry.add(widget_factory({"alpha"}));
    EXPECT_THROW(registry.add(widget_factory({"beta", "alpha"})), std::logic_error);
    EXPECT_EQ(registry.entries().size(), 1u);
}

TEST(RegistryTest, KeepsRegistrationOrderAndJoinsCanonicalNames) {
    Registry<WidgetFactory> registry;
    registry.add(widget_factory({"gamma", "third"}));
    registry.add(widget_factory({"alpha"}));
    EXPECT_EQ(registry.canonical_names(", "), "gamma, alpha");
    EXPECT_EQ(registry.entries().front()->names().front(), "gamma");
}

TEST(RegistryTest, AnEmptyRegistryJoinsToNothing) {
    const Registry<WidgetFactory> registry;
    EXPECT_EQ(registry.canonical_names("|"), "");
}
