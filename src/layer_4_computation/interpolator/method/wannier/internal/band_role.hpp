#pragma once

/**
 * BandRole — what the classifier does with one band at one k-point: keep it
 * exactly, let it compete for the room the frozen bands leave, or throw it
 * away.
 */

namespace q3 {

enum class BandRole { Frozen, Candidate, Discarded };

} // namespace q3
