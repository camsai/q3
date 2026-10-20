#pragma once

/**
 * WavefunctionCheckpoint — abstract base for writing wavefunctions to disk.
 *
 * Theory-agnostic: knows only that wavefunctions exist and can be serialized.
 * Subclasses add theory-specific context.
 */

#include "io/mixins/binary_serializable.hpp"

namespace q3 {

class WavefunctionCheckpoint : public BinarySerializable {
  public:
    ~WavefunctionCheckpoint() override = default;
};

} // namespace q3
