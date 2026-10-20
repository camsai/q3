#pragma once

/**
 * KohnShamWavefunctionCheckpoint — Kohn-Sham level wavefunction checkpoint.
 *
 * Knows about KS-level concepts but not about the basis representation.
 */

#include "io/wavefunction/init.hpp"

namespace q3 {

class KohnShamWavefunctionCheckpoint : public WavefunctionCheckpoint {
  public:
    ~KohnShamWavefunctionCheckpoint() override = default;
};

} // namespace q3
