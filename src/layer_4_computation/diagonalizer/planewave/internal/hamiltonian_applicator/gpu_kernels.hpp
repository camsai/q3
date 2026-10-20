#pragma once

/**
 * GPU kernels for Phase 5b: Scatter, gather, kinetic energy, potential multiply,
 * residual computation, and preconditioning.
 *
 * Declares the vendor-neutral kernel launch functions, which are specialized
 * in side-by-side backend-specific source files (CUDA, HIP, SYCL, Serial).
 */

#include <complex>
#include <vector>

#include "layer_0_core/types/single_precision.hpp"

namespace q3 {
namespace diagonalizer {
namespace planewave {
namespace gpu {

void launch_apply_kinetic_energy(int number_of_vectors,
                                 int number_of_plane_waves,
                                 const double* kinetic_energy,
                                 const std::complex<double>* psi,
                                 std::complex<double>* hpsi);

void launch_apply_kinetic_energy_single(int number_of_vectors,
                                        int number_of_plane_waves,
                                        const float* kinetic_energy,
                                        const std::complex<float>* psi,
                                        std::complex<float>* hpsi);

void launch_scatter_g_to_fft_grid(int number_of_vectors,
                                  int number_of_plane_waves,
                                  int local_smooth_grid_points,
                                  const int* g_to_fft_mapping,
                                  const int* minus_g_to_fft_mapping,
                                  bool is_gamma_only,
                                  const std::complex<double>* psi,
                                  std::complex<double>* fft_workspace);

void launch_scatter_g_to_fft_grid_single(int number_of_vectors,
                                         int number_of_plane_waves,
                                         int local_smooth_grid_points,
                                         const int* g_to_fft_mapping,
                                         const int* minus_g_to_fft_mapping,
                                         bool is_gamma_only,
                                         const std::complex<float>* psi,
                                         std::complex<float>* fft_workspace);

void launch_scatter_g_to_fft_grid_mixed(int number_of_vectors,
                                        int number_of_plane_waves,
                                        int local_smooth_grid_points,
                                        const int* g_to_fft_mapping,
                                        const int* minus_g_to_fft_mapping,
                                        bool is_gamma_only,
                                        const std::complex<double>* psi,
                                        std::complex<float>* fft_workspace);

void launch_multiply_potential(int number_of_vectors,
                               int local_smooth_grid_points,
                               const double* potential,
                               std::complex<double>* fft_workspace);

void launch_multiply_potential_single(int number_of_vectors,
                                      int local_smooth_grid_points,
                                      const float* potential,
                                      std::complex<float>* fft_workspace);

void launch_gather_fft_grid_to_g(int number_of_vectors,
                                 int number_of_plane_waves,
                                 int local_smooth_grid_points,
                                 const int* g_to_fft_mapping,
                                 const std::complex<double>* fft_workspace,
                                 std::complex<double>* hpsi);

void launch_gather_fft_grid_to_g_single(int number_of_vectors,
                                        int number_of_plane_waves,
                                        int local_smooth_grid_points,
                                        const int* g_to_fft_mapping,
                                        const std::complex<float>* fft_workspace,
                                        std::complex<float>* hpsi);

void launch_gather_fft_grid_to_g_mixed(int number_of_vectors,
                                       int number_of_plane_waves,
                                       int local_smooth_grid_points,
                                       const int* g_to_fft_mapping,
                                       const std::complex<float>* fft_workspace,
                                       std::complex<double>* hpsi);

void launch_compute_residuals(int number_of_bands,
                              int number_of_plane_waves,
                              const double* eigenvalues,
                              const std::complex<double>* hamiltonian_ritz_vectors,
                              const std::complex<double>* overlap_ritz_vectors,
                              std::complex<double>* residual_wavefunctions);

void launch_apply_preconditioner(int number_of_bands,
                                 int number_of_plane_waves,
                                 const double* kinetic_energy_squared,
                                 const double* eigenvalues,
                                 const bool* band_converged,
                                 double floor_value,
                                 std::complex<double>* residual_wavefunctions);

} // namespace gpu
} // namespace planewave
} // namespace diagonalizer
} // namespace q3
