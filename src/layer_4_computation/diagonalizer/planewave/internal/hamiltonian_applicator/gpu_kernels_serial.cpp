#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace q3 {
namespace diagonalizer {
namespace planewave {
namespace gpu {

void launch_apply_kinetic_energy(int number_of_vectors,
                                 int number_of_plane_waves,
                                 const double* kinetic_energy,
                                 const std::complex<double>* psi,
                                 std::complex<double>* hpsi) {
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int offset = vector_index * number_of_plane_waves + g_index;
            hpsi[offset] = kinetic_energy[g_index] * psi[offset];
        }
    }
}

void launch_apply_kinetic_energy_single(int number_of_vectors,
                                        int number_of_plane_waves,
                                        const float* kinetic_energy,
                                        const std::complex<float>* psi,
                                        std::complex<float>* hpsi) {
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int offset = vector_index * number_of_plane_waves + g_index;
            hpsi[offset] = kinetic_energy[g_index] * psi[offset];
        }
    }
}

void launch_scatter_g_to_fft_grid(int number_of_vectors,
                                  int number_of_plane_waves,
                                  int local_smooth_grid_points,
                                  const int* g_to_fft_mapping,
                                  const int* minus_g_to_fft_mapping,
                                  bool is_gamma_only,
                                  const std::complex<double>* psi,
                                  std::complex<double>* fft_workspace) {
    std::fill_n(fft_workspace,
                static_cast<std::size_t>(number_of_vectors) * local_smooth_grid_points,
                std::complex<double>(0.0, 0.0));
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        const std::complex<double>* psi_vec = &psi[vector_index * number_of_plane_waves];
        std::complex<double>* workspace_vec =
            &fft_workspace[vector_index * local_smooth_grid_points];
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int fft_index = g_to_fft_mapping[g_index];
            if (fft_index >= 0) {
                workspace_vec[fft_index] = psi_vec[g_index];
            }
        }
        if (is_gamma_only && minus_g_to_fft_mapping != nullptr) {
            for (int g_index = 1; g_index < number_of_plane_waves; ++g_index) {
                int fft_minus = minus_g_to_fft_mapping[g_index];
                if (fft_minus >= 0) {
                    workspace_vec[fft_minus] = std::conj(psi_vec[g_index]);
                }
            }
        }
    }
}

void launch_scatter_g_to_fft_grid_single(int number_of_vectors,
                                         int number_of_plane_waves,
                                         int local_smooth_grid_points,
                                         const int* g_to_fft_mapping,
                                         const int* minus_g_to_fft_mapping,
                                         bool is_gamma_only,
                                         const std::complex<float>* psi,
                                         std::complex<float>* fft_workspace) {
    std::fill_n(fft_workspace,
                static_cast<std::size_t>(number_of_vectors) * local_smooth_grid_points,
                std::complex<float>(0.0F, 0.0F));
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        const std::complex<float>* psi_vec = &psi[vector_index * number_of_plane_waves];
        std::complex<float>* workspace_vec =
            &fft_workspace[vector_index * local_smooth_grid_points];
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int fft_index = g_to_fft_mapping[g_index];
            if (fft_index >= 0) {
                workspace_vec[fft_index] = psi_vec[g_index];
            }
        }
        if (is_gamma_only && minus_g_to_fft_mapping != nullptr) {
            for (int g_index = 1; g_index < number_of_plane_waves; ++g_index) {
                int fft_minus = minus_g_to_fft_mapping[g_index];
                if (fft_minus >= 0) {
                    workspace_vec[fft_minus] = std::conj(psi_vec[g_index]);
                }
            }
        }
    }
}

void launch_scatter_g_to_fft_grid_mixed(int number_of_vectors,
                                        int number_of_plane_waves,
                                        int local_smooth_grid_points,
                                        const int* g_to_fft_mapping,
                                        const int* minus_g_to_fft_mapping,
                                        bool is_gamma_only,
                                        const std::complex<double>* psi,
                                        std::complex<float>* fft_workspace) {
    std::fill_n(fft_workspace,
                static_cast<std::size_t>(number_of_vectors) * local_smooth_grid_points,
                std::complex<float>(0.0F, 0.0F));
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        const std::complex<double>* psi_vec = &psi[vector_index * number_of_plane_waves];
        std::complex<float>* workspace_vec =
            &fft_workspace[vector_index * local_smooth_grid_points];
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int fft_index = g_to_fft_mapping[g_index];
            if (fft_index >= 0) {
                workspace_vec[fft_index] =
                    std::complex<float>(static_cast<float>(psi_vec[g_index].real()),
                                        static_cast<float>(psi_vec[g_index].imag()));
            }
        }
        if (is_gamma_only && minus_g_to_fft_mapping != nullptr) {
            for (int g_index = 1; g_index < number_of_plane_waves; ++g_index) {
                int fft_minus = minus_g_to_fft_mapping[g_index];
                if (fft_minus >= 0) {
                    workspace_vec[fft_minus] =
                        std::conj(std::complex<float>(static_cast<float>(psi_vec[g_index].real()),
                                                      static_cast<float>(psi_vec[g_index].imag())));
                }
            }
        }
    }
}

void launch_multiply_potential(int number_of_vectors,
                               int local_smooth_grid_points,
                               const double* potential,
                               std::complex<double>* fft_workspace) {
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        std::complex<double>* workspace_vec =
            &fft_workspace[vector_index * local_smooth_grid_points];
        for (int grid_index = 0; grid_index < local_smooth_grid_points; ++grid_index) {
            workspace_vec[grid_index] *= potential[grid_index];
        }
    }
}

void launch_multiply_potential_single(int number_of_vectors,
                                      int local_smooth_grid_points,
                                      const float* potential,
                                      std::complex<float>* fft_workspace) {
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        std::complex<float>* workspace_vec =
            &fft_workspace[vector_index * local_smooth_grid_points];
        for (int grid_index = 0; grid_index < local_smooth_grid_points; ++grid_index) {
            workspace_vec[grid_index] *= potential[grid_index];
        }
    }
}

void launch_gather_fft_grid_to_g(int number_of_vectors,
                                 int number_of_plane_waves,
                                 int local_smooth_grid_points,
                                 const int* g_to_fft_mapping,
                                 const std::complex<double>* fft_workspace,
                                 std::complex<double>* hpsi) {
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        std::complex<double>* hpsi_vec = &hpsi[vector_index * number_of_plane_waves];
        const std::complex<double>* workspace_vec =
            &fft_workspace[vector_index * local_smooth_grid_points];
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int fft_index = g_to_fft_mapping[g_index];
            if (fft_index >= 0) {
                hpsi_vec[g_index] += workspace_vec[fft_index];
            }
        }
    }
}

void launch_gather_fft_grid_to_g_single(int number_of_vectors,
                                        int number_of_plane_waves,
                                        int local_smooth_grid_points,
                                        const int* g_to_fft_mapping,
                                        const std::complex<float>* fft_workspace,
                                        std::complex<float>* hpsi) {
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        std::complex<float>* hpsi_vec = &hpsi[vector_index * number_of_plane_waves];
        const std::complex<float>* workspace_vec =
            &fft_workspace[vector_index * local_smooth_grid_points];
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int fft_index = g_to_fft_mapping[g_index];
            if (fft_index >= 0) {
                hpsi_vec[g_index] += workspace_vec[fft_index];
            }
        }
    }
}

void launch_gather_fft_grid_to_g_mixed(int number_of_vectors,
                                       int number_of_plane_waves,
                                       int local_smooth_grid_points,
                                       const int* g_to_fft_mapping,
                                       const std::complex<float>* fft_workspace,
                                       std::complex<double>* hpsi) {
    for (int vector_index = 0; vector_index < number_of_vectors; ++vector_index) {
        std::complex<double>* hpsi_vec = &hpsi[vector_index * number_of_plane_waves];
        const std::complex<float>* workspace_vec =
            &fft_workspace[vector_index * local_smooth_grid_points];
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int fft_index = g_to_fft_mapping[g_index];
            if (fft_index >= 0) {
                hpsi_vec[g_index] +=
                    std::complex<double>(static_cast<double>(workspace_vec[fft_index].real()),
                                         static_cast<double>(workspace_vec[fft_index].imag()));
            }
        }
    }
}

void launch_compute_residuals(int number_of_bands,
                              int number_of_plane_waves,
                              const double* eigenvalues,
                              const std::complex<double>* hamiltonian_ritz_vectors,
                              const std::complex<double>* overlap_ritz_vectors,
                              std::complex<double>* residual_wavefunctions) {
    for (int band_index = 0; band_index < number_of_bands; ++band_index) {
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            int offset = band_index * number_of_plane_waves + g_index;
            residual_wavefunctions[offset] = hamiltonian_ritz_vectors[offset] -
                                             eigenvalues[band_index] * overlap_ritz_vectors[offset];
        }
    }
}

void launch_apply_preconditioner(int number_of_bands,
                                 int number_of_plane_waves,
                                 const double* kinetic_energy_squared,
                                 const double* eigenvalues,
                                 const bool* band_converged,
                                 double floor_value,
                                 std::complex<double>* residual_wavefunctions) {
    for (int band_index = 0; band_index < number_of_bands; ++band_index) {
        if (band_converged[band_index]) {
            continue;
        }
        std::complex<double>* residual =
            &residual_wavefunctions[band_index * number_of_plane_waves];
        for (int g_index = 0; g_index < number_of_plane_waves; ++g_index) {
            double value_x = kinetic_energy_squared[g_index] - eigenvalues[band_index];
            double denominator =
                0.5 * (1.0 + value_x + std::sqrt(1.0 + (value_x - 1.0) * (value_x - 1.0)));
            if (denominator < floor_value) {
                denominator = floor_value;
            }
            residual[g_index] /= denominator;
        }
    }
}

} // namespace gpu
} // namespace planewave
} // namespace diagonalizer
} // namespace q3
