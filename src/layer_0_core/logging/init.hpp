#pragma once

/**
 * Logging re-export for core layers.
 *
 * Layers 1–3 should include this header instead of "io/logger.hpp"
 * so that the logging dependency flows through layer_0_core rather
 * than reaching across into the IO layer.
 */

#include "io/logger.hpp"
