#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

namespace q3 {
namespace diagonalizer {
namespace planewave {
namespace gpu {

// ─── HIP Kernels ──────────────────────────────────────────────────────────

__global__ void apply_kinetic_energy_kernel(int total_elements,
                                            int number_of_plane_waves,
                                            const double* kinetic_energy,
                                            const hipDoubleComplex* psi,
                                            hipDoubleComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int g_index = index % number_of_plane_waves;
        double ke = kinetic_energy[g_index];
        hipDoubleComplex p = psi[index];
        hpsi[index] = make_hipDoubleComplex(ke * p.x, ke * p.y);
    }
}

__global__ void apply_kinetic_energy_single_kernel(int total_elements,
                                                   int number_of_plane_waves,
                                                   const float* kinetic_energy,
                                                   const hipComplex* psi,
                                                   hipComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int g_index = index % number_of_plane_waves;
        float ke = kinetic_energy[g_index];
        hipComplex p = psi[index];
        hpsi[index] = make_hipFloatComplex(ke * p.x, ke * p.y);
    }
}

__global__ void scatter_g_to_fft_grid_kernel(int total_elements,
                                             int number_of_plane_waves,
                                             int local_smooth_grid_points,
                                             const int* g_to_fft_mapping,
                                             const int* minus_g_to_fft_mapping,
                                             bool is_gamma_only,
                                             const hipDoubleComplex* psi,
                                             hipDoubleComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        hipDoubleComplex val = psi[index];
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            fft_workspace[vector_index * local_smooth_grid_points + fft_index] = val;
        }
        if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
            int fft_minus = minus_g_to_fft_mapping[g_index];
            if (fft_minus >= 0) {
                fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                    make_hipDoubleComplex(val.x, -val.y);
            }
        }
    }
}

__global__ void scatter_g_to_fft_grid_single_kernel(int total_elements,
                                                    int number_of_plane_waves,
                                                    int local_smooth_grid_points,
                                                    const int* g_to_fft_mapping,
                                                    const int* minus_g_to_fft_mapping,
                                                    bool is_gamma_only,
                                                    const hipComplex* psi,
                                                    hipComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        hipComplex val = psi[index];
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            fft_workspace[vector_index * local_smooth_grid_points + fft_index] = val;
        }
        if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
            int fft_minus = minus_g_to_fft_mapping[g_index];
            if (fft_minus >= 0) {
                fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                    make_hipFloatComplex(val.x, -val.y);
            }
        }
    }
}

__global__ void scatter_g_to_fft_grid_mixed_kernel(int total_elements,
                                                   int number_of_plane_waves,
                                                   int local_smooth_grid_points,
                                                   const int* g_to_fft_mapping,
                                                   const int* minus_g_to_fft_mapping,
                                                   bool is_gamma_only,
                                                   const hipDoubleComplex* psi,
                                                   hipComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        hipDoubleComplex val = psi[index];
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            fft_workspace[vector_index * local_smooth_grid_points + fft_index] =
                make_hipFloatComplex((float)val.x, (float)val.y);
        }
        if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
            int fft_minus = minus_g_to_fft_mapping[g_index];
            if (fft_minus >= 0) {
                fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                    make_hipFloatComplex((float)val.x, -(float)val.y);
            }
        }
    }
}

__global__ void multiply_potential_kernel(int total_elements,
                                          int local_smooth_grid_points,
                                          const double* potential,
                                          hipDoubleComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int grid_index = index % local_smooth_grid_points;
        double pot = potential[grid_index];
        hipDoubleComplex val = fft_workspace[index];
        fft_workspace[index] = make_hipDoubleComplex(val.x * pot, val.y * pot);
    }
}

__global__ void multiply_potential_single_kernel(int total_elements,
                                                 int local_smooth_grid_points,
                                                 const float* potential,
                                                 hipComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int grid_index = index % local_smooth_grid_points;
        float pot = potential[grid_index];
        hipComplex val = fft_workspace[index];
        fft_workspace[index] = make_hipFloatComplex(val.x * pot, val.y * pot);
    }
}

__global__ void gather_fft_grid_to_g_kernel(int total_elements,
                                            int number_of_plane_waves,
                                            int local_smooth_grid_points,
                                            const int* g_to_fft_mapping,
                                            const hipDoubleComplex* fft_workspace,
                                            hipDoubleComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            hipDoubleComplex val =
                fft_workspace[vector_index * local_smooth_grid_points + fft_index];
            hipDoubleComplex current = hpsi[index];
            hpsi[index] = make_hipDoubleComplex(current.x + val.x, current.y + val.y);
        }
    }
}

__global__ void gather_fft_grid_to_g_single_kernel(int total_elements,
                                                   int number_of_plane_waves,
                                                   int local_smooth_grid_points,
                                                   const int* g_to_fft_mapping,
                                                   const hipComplex* fft_workspace,
                                                   hipComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            hipComplex val = fft_workspace[vector_index * local_smooth_grid_points + fft_index];
            hipComplex current = hpsi[index];
            hpsi[index] = make_hipFloatComplex(current.x + val.x, current.y + val.y);
        }
    }
}

__global__ void gather_fft_grid_to_g_mixed_kernel(int total_elements,
                                                  int number_of_plane_waves,
                                                  int local_smooth_grid_points,
                                                  const int* g_to_fft_mapping,
                                                  const hipComplex* fft_workspace,
                                                  hipDoubleComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            hipComplex val = fft_workspace[vector_index * local_smooth_grid_points + fft_index];
            hipDoubleComplex current = hpsi[index];
            hpsi[index] =
                make_hipDoubleComplex(current.x + (double)val.x, current.y + (double)val.y);
        }
    }
}

__global__ void compute_residuals_kernel(int total_elements,
                                         int number_of_plane_waves,
                                         const double* eigenvalues,
                                         const hipDoubleComplex* hamiltonian_ritz_vectors,
                                         const hipDoubleComplex* overlap_ritz_vectors,
                                         hipDoubleComplex* residual_wavefunctions) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int band_index = index / number_of_plane_waves;
        double eig = eigenvalues[band_index];
        hipDoubleComplex h_ritz = hamiltonian_ritz_vectors[index];
        hipDoubleComplex s_ritz = overlap_ritz_vectors[index];
        residual_wavefunctions[index] =
            make_hipDoubleComplex(h_ritz.x - eig * s_ritz.x, h_ritz.y - eig * s_ritz.y);
    }
}

__global__ void apply_preconditioner_kernel(int total_elements,
                                            int number_of_plane_waves,
                                            const double* kinetic_energy_squared,
                                            const double* eigenvalues,
                                            const bool* band_converged,
                                            double floor_value,
                                            hipDoubleComplex* residual_wavefunctions) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int band_index = index / number_of_plane_waves;
        if (band_converged[band_index]) {
            return;
        }
        int g_index = index % number_of_plane_waves;
        double value_x = kinetic_energy_squared[g_index] - eigenvalues[band_index];
        double denominator = 0.5 * (1.0 + value_x + sqrt(1.0 + (value_x - 1.0) * (value_x - 1.0)));
        if (denominator < floor_value) {
            denominator = floor_value;
        }
        hipDoubleComplex res = residual_wavefunctions[index];
        residual_wavefunctions[index] =
            make_hipDoubleComplex(res.x / denominator, res.y / denominator);
    }
}

// ─── HIP Launchers ──────────────────────────────────────────────────────────

void launch_apply_kinetic_energy(int number_of_vectors,
                                 int number_of_plane_waves,
                                 const double* kinetic_energy,
                                 const std::complex<double>* psi,
                                 std::complex<double>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(apply_kinetic_energy_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       kinetic_energy,
                       reinterpret_cast<const hipDoubleComplex*>(psi),
                       reinterpret_cast<hipDoubleComplex*>(hpsi));
}

void launch_apply_kinetic_energy_single(int number_of_vectors,
                                        int number_of_plane_waves,
                                        const float* kinetic_energy,
                                        const std::complex<float>* psi,
                                        std::complex<float>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(apply_kinetic_energy_single_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       kinetic_energy,
                       reinterpret_cast<const hipComplex*>(psi),
                       reinterpret_cast<hipComplex*>(hpsi));
}

void launch_scatter_g_to_fft_grid(int number_of_vectors,
                                  int number_of_plane_waves,
                                  int local_smooth_grid_points,
                                  const int* g_to_fft_mapping,
                                  const int* minus_g_to_fft_mapping,
                                  bool is_gamma_only,
                                  const std::complex<double>* psi,
                                  std::complex<double>* fft_workspace) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    hipMemset(fft_workspace,
              0,
              number_of_vectors * local_smooth_grid_points * sizeof(std::complex<double>));
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(scatter_g_to_fft_grid_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       local_smooth_grid_points,
                       g_to_fft_mapping,
                       minus_g_to_fft_mapping,
                       is_gamma_only,
                       reinterpret_cast<const hipDoubleComplex*>(psi),
                       reinterpret_cast<hipDoubleComplex*>(fft_workspace));
}

void launch_scatter_g_to_fft_grid_single(int number_of_vectors,
                                         int number_of_plane_waves,
                                         int local_smooth_grid_points,
                                         const int* g_to_fft_mapping,
                                         const int* minus_g_to_fft_mapping,
                                         bool is_gamma_only,
                                         const std::complex<float>* psi,
                                         std::complex<float>* fft_workspace) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    hipMemset(fft_workspace,
              0,
              number_of_vectors * local_smooth_grid_points * sizeof(std::complex<float>));
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(scatter_g_to_fft_grid_single_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       local_smooth_grid_points,
                       g_to_fft_mapping,
                       minus_g_to_fft_mapping,
                       is_gamma_only,
                       reinterpret_cast<const hipComplex*>(psi),
                       reinterpret_cast<hipComplex*>(fft_workspace));
}

void launch_scatter_g_to_fft_grid_mixed(int number_of_vectors,
                                        int number_of_plane_waves,
                                        int local_smooth_grid_points,
                                        const int* g_to_fft_mapping,
                                        const int* minus_g_to_fft_mapping,
                                        bool is_gamma_only,
                                        const std::complex<double>* psi,
                                        std::complex<float>* fft_workspace) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    hipMemset(fft_workspace,
              0,
              number_of_vectors * local_smooth_grid_points * sizeof(std::complex<float>));
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(scatter_g_to_fft_grid_mixed_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       local_smooth_grid_points,
                       g_to_fft_mapping,
                       minus_g_to_fft_mapping,
                       is_gamma_only,
                       reinterpret_cast<const hipDoubleComplex*>(psi),
                       reinterpret_cast<hipComplex*>(fft_workspace));
}

void launch_multiply_potential(int number_of_vectors,
                               int local_smooth_grid_points,
                               const double* potential,
                               std::complex<double>* fft_workspace) {
    int total_elements = number_of_vectors * local_smooth_grid_points;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(multiply_potential_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       local_smooth_grid_points,
                       potential,
                       reinterpret_cast<hipDoubleComplex*>(fft_workspace));
}

void launch_multiply_potential_single(int number_of_vectors,
                                      int local_smooth_grid_points,
                                      const float* potential,
                                      std::complex<float>* fft_workspace) {
    int total_elements = number_of_vectors * local_smooth_grid_points;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(multiply_potential_single_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       local_smooth_grid_points,
                       potential,
                       reinterpret_cast<hipComplex*>(fft_workspace));
}

void launch_gather_fft_grid_to_g(int number_of_vectors,
                                 int number_of_plane_waves,
                                 int local_smooth_grid_points,
                                 const int* g_to_fft_mapping,
                                 const std::complex<double>* fft_workspace,
                                 std::complex<double>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(gather_fft_grid_to_g_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       local_smooth_grid_points,
                       g_to_fft_mapping,
                       reinterpret_cast<const hipDoubleComplex*>(fft_workspace),
                       reinterpret_cast<hipDoubleComplex*>(hpsi));
}

void launch_gather_fft_grid_to_g_single(int number_of_vectors,
                                        int number_of_plane_waves,
                                        int local_smooth_grid_points,
                                        const int* g_to_fft_mapping,
                                        const std::complex<float>* fft_workspace,
                                        std::complex<float>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(gather_fft_grid_to_g_single_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       local_smooth_grid_points,
                       g_to_fft_mapping,
                       reinterpret_cast<const hipComplex*>(fft_workspace),
                       reinterpret_cast<hipComplex*>(hpsi));
}

void launch_gather_fft_grid_to_g_mixed(int number_of_vectors,
                                       int number_of_plane_waves,
                                       int local_smooth_grid_points,
                                       const int* g_to_fft_mapping,
                                       const std::complex<float>* fft_workspace,
                                       std::complex<double>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(gather_fft_grid_to_g_mixed_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       local_smooth_grid_points,
                       g_to_fft_mapping,
                       reinterpret_cast<const hipComplex*>(fft_workspace),
                       reinterpret_cast<hipDoubleComplex*>(hpsi));
}

void launch_compute_residuals(int number_of_bands,
                              int number_of_plane_waves,
                              const double* eigenvalues,
                              const std::complex<double>* hamiltonian_ritz_vectors,
                              const std::complex<double>* overlap_ritz_vectors,
                              std::complex<double>* residual_wavefunctions) {
    int total_elements = number_of_bands * number_of_plane_waves;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(compute_residuals_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       eigenvalues,
                       reinterpret_cast<const hipDoubleComplex*>(hamiltonian_ritz_vectors),
                       reinterpret_cast<const hipDoubleComplex*>(overlap_ritz_vectors),
                       reinterpret_cast<hipDoubleComplex*>(residual_wavefunctions));
}

void launch_apply_preconditioner(int number_of_bands,
                                 int number_of_plane_waves,
                                 const double* kinetic_energy_squared,
                                 const double* eigenvalues,
                                 const bool* band_converged,
                                 double floor_value,
                                 std::complex<double>* residual_wavefunctions) {
    int total_elements = number_of_bands * number_of_plane_waves;
    if (total_elements == 0)
        return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    hipLaunchKernelGGL(apply_preconditioner_kernel,
                       blocks,
                       threads,
                       0,
                       0,
                       total_elements,
                       number_of_plane_waves,
                       kinetic_energy_squared,
                       eigenvalues,
                       band_converged,
                       floor_value,
                       reinterpret_cast<hipDoubleComplex*>(residual_wavefunctions));
}

} // namespace gpu
} // namespace planewave
} // namespace diagonalizer
} // namespace q3
