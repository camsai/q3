#pragma once

/**
 * ChargeDensityCheckpoint — abstract base for writing charge density to disk.
 *
 * Theory-agnostic: knows only that a charge density exists and can be serialized.
 * Subclasses add theory-specific context (KS energies, PW basis, etc.)
 */

#include "io/mixins/binary_serializable.hpp"

namespace q3 {

class ChargeDensityCheckpoint : public BinarySerializable {
  public:
    ~ChargeDensityCheckpoint() override = default;
};

} // namespace q3
