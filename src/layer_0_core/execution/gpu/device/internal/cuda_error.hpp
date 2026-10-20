#pragma once

/**
 * CUDA error handling utilities.
 *
 * WARNING: This file includes <cuda_runtime.h> and <cublas_v2.h>.
 * It MUST NOT be included from shared (vendor-neutral) headers.
 * Only CUDA-specific translation units (cuda.cpp, cuda.cu) should
 * include this file. This enforces AGENTS.md §6.1: serial baseline
 * files must compile without CUDA toolchain.
 *
 * Error logic is in inline functions; macros exist solely to capture
 * __FILE__ and __LINE__ for diagnostic messages.
 */

#include "layer_0_core/logging/init.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace q3 {
namespace execution {
namespace gpu {
namespace cuda {

/**
 * Check a CUDA runtime API call and throw on error.
 *
 * @param error  Return value from a cuda* function.
 * @param file   Source file name (from __FILE__).
 * @param line   Source line number (from __LINE__).
 * @throws std::runtime_error  If error != cudaSuccess.
 */
inline void check_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        q3::log::error("CUDA error at {}:{}: {} (code {})",
                       file,
                       line,
                       cudaGetErrorString(error),
                       static_cast<int>(error));
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
    }
}

/**
 * Check a cuBLAS API call and throw on error.
 *
 * @param status  Return value from a cublas* function.
 * @param file    Source file name (from __FILE__).
 * @param line    Source line number (from __LINE__).
 * @throws std::runtime_error  If status != CUBLAS_STATUS_SUCCESS.
 */
inline void check_cublas_error(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        q3::log::error("cuBLAS error at {}:{}: code {}", file, line, static_cast<int>(status));
        throw std::runtime_error("cuBLAS error");
    }
}

} // namespace cuda
} // namespace gpu
} // namespace execution
} // namespace q3

/**
 * Check a CUDA runtime API call, logging file and line on error.
 *
 * Usage:
 *   Q3_CUDA_CHECK(cudaMalloc(&pointer, size));
 */
#define Q3_CUDA_CHECK(call) q3::execution::gpu::cuda::check_error((call), __FILE__, __LINE__)

/**
 * Check a cuBLAS API call, logging file and line on error.
 *
 * Usage:
 *   Q3_CUBLAS_CHECK(cublasCreate(&handle));
 */
#define Q3_CUBLAS_CHECK(call)                                                                      \
    q3::execution::gpu::cuda::check_cublas_error((call), __FILE__, __LINE__)
