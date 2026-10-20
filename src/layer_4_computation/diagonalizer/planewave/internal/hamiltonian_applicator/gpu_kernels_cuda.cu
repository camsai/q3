#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace q3 {
namespace diagonalizer {
namespace planewave {
namespace gpu {

// ─── CUDA Kernels ──────────────────────────────────────────────────────────

__global__ void apply_kinetic_energy_kernel(
    int total_elements,
    int number_of_plane_waves,
    const double* kinetic_energy,
    const cuDoubleComplex* psi,
    cuDoubleComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int g_index = index % number_of_plane_waves;
        double ke = kinetic_energy[g_index];
        cuDoubleComplex p = psi[index];
        hpsi[index] = make_cuDoubleComplex(ke * cuCreal(p), ke * cuCimag(p));
    }
}

__global__ void apply_kinetic_energy_single_kernel(
    int total_elements,
    int number_of_plane_waves,
    const float* kinetic_energy,
    const cuComplex* psi,
    cuComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int g_index = index % number_of_plane_waves;
        float ke = kinetic_energy[g_index];
        cuComplex p = psi[index];
        hpsi[index] = make_cuFloatComplex(ke * cuCrealf(p), ke * cuCimagf(p));
    }
}

__global__ void scatter_g_to_fft_grid_kernel(
    int total_elements,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const int* minus_g_to_fft_mapping,
    bool is_gamma_only,
    const cuDoubleComplex* psi,
    cuDoubleComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        cuDoubleComplex val = psi[index];
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            fft_workspace[vector_index * local_smooth_grid_points + fft_index] = val;
        }
        if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
            int fft_minus = minus_g_to_fft_mapping[g_index];
            if (fft_minus >= 0) {
                fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                    make_cuDoubleComplex(cuCreal(val), -cuCimag(val));
            }
        }
    }
}

__global__ void scatter_g_to_fft_grid_single_kernel(
    int total_elements,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const int* minus_g_to_fft_mapping,
    bool is_gamma_only,
    const cuComplex* psi,
    cuComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        cuComplex val = psi[index];
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            fft_workspace[vector_index * local_smooth_grid_points + fft_index] = val;
        }
        if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
            int fft_minus = minus_g_to_fft_mapping[g_index];
            if (fft_minus >= 0) {
                fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                    make_cuFloatComplex(cuCrealf(val), -cuCimagf(val));
            }
        }
    }
}

__global__ void scatter_g_to_fft_grid_mixed_kernel(
    int total_elements,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const int* minus_g_to_fft_mapping,
    bool is_gamma_only,
    const cuDoubleComplex* psi,
    cuComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        cuDoubleComplex val = psi[index];
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            fft_workspace[vector_index * local_smooth_grid_points + fft_index] =
                make_cuFloatComplex((float)cuCreal(val), (float)cuCimag(val));
        }
        if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
            int fft_minus = minus_g_to_fft_mapping[g_index];
            if (fft_minus >= 0) {
                fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                    make_cuFloatComplex((float)cuCreal(val), -(float)cuCimag(val));
            }
        }
    }
}

__global__ void multiply_potential_kernel(
    int total_elements,
    int local_smooth_grid_points,
    const double* potential,
    cuDoubleComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int grid_index = index % local_smooth_grid_points;
        double pot = potential[grid_index];
        cuDoubleComplex val = fft_workspace[index];
        fft_workspace[index] = make_cuDoubleComplex(cuCreal(val) * pot, cuCimag(val) * pot);
    }
}

__global__ void multiply_potential_single_kernel(
    int total_elements,
    int local_smooth_grid_points,
    const float* potential,
    cuComplex* fft_workspace) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int grid_index = index % local_smooth_grid_points;
        float pot = potential[grid_index];
        cuComplex val = fft_workspace[index];
        fft_workspace[index] = make_cuFloatComplex(cuCrealf(val) * pot, cuCimagf(val) * pot);
    }
}

__global__ void gather_fft_grid_to_g_kernel(
    int total_elements,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const cuDoubleComplex* fft_workspace,
    cuDoubleComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            cuDoubleComplex val = fft_workspace[vector_index * local_smooth_grid_points + fft_index];
            cuDoubleComplex current = hpsi[index];
            hpsi[index] = make_cuDoubleComplex(cuCreal(current) + cuCreal(val), cuCimag(current) + cuCimag(val));
        }
    }
}

__global__ void gather_fft_grid_to_g_single_kernel(
    int total_elements,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const cuComplex* fft_workspace,
    cuComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            cuComplex val = fft_workspace[vector_index * local_smooth_grid_points + fft_index];
            cuComplex current = hpsi[index];
            hpsi[index] = make_cuFloatComplex(cuCrealf(current) + cuCrealf(val), cuCimagf(current) + cuCimagf(val));
        }
    }
}

__global__ void gather_fft_grid_to_g_mixed_kernel(
    int total_elements,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const cuComplex* fft_workspace,
    cuDoubleComplex* hpsi) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int vector_index = index / number_of_plane_waves;
        int g_index = index % number_of_plane_waves;
        int fft_index = g_to_fft_mapping[g_index];
        if (fft_index >= 0) {
            cuComplex val = fft_workspace[vector_index * local_smooth_grid_points + fft_index];
            cuDoubleComplex current = hpsi[index];
            hpsi[index] = make_cuDoubleComplex(
                cuCreal(current) + (double)cuCrealf(val),
                cuCimag(current) + (double)cuCimagf(val));
        }
    }
}

__global__ void compute_residuals_kernel(
    int total_elements,
    int number_of_plane_waves,
    const double* eigenvalues,
    const cuDoubleComplex* hamiltonian_ritz_vectors,
    const cuDoubleComplex* overlap_ritz_vectors,
    cuDoubleComplex* residual_wavefunctions) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        int band_index = index / number_of_plane_waves;
        double eig = eigenvalues[band_index];
        cuDoubleComplex h_ritz = hamiltonian_ritz_vectors[index];
        cuDoubleComplex s_ritz = overlap_ritz_vectors[index];
        residual_wavefunctions[index] = make_cuDoubleComplex(
            cuCreal(h_ritz) - eig * cuCreal(s_ritz),
            cuCimag(h_ritz) - eig * cuCimag(s_ritz));
    }
}

__global__ void apply_preconditioner_kernel(
    int total_elements,
    int number_of_plane_waves,
    const double* kinetic_energy_squared,
    const double* eigenvalues,
    const bool* band_converged,
    double floor_value,
    cuDoubleComplex* residual_wavefunctions) {
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
        cuDoubleComplex res = residual_wavefunctions[index];
        residual_wavefunctions[index] = make_cuDoubleComplex(
            cuCreal(res) / denominator,
            cuCimag(res) / denominator);
    }
}

// ─── CUDA Launchers ──────────────────────────────────────────────────────────

void launch_apply_kinetic_energy(
    int number_of_vectors,
    int number_of_plane_waves,
    const double* kinetic_energy,
    const std::complex<double>* psi,
    std::complex<double>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    apply_kinetic_energy_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        kinetic_energy,
        reinterpret_cast<const cuDoubleComplex*>(psi),
        reinterpret_cast<cuDoubleComplex*>(hpsi));
}

void launch_apply_kinetic_energy_single(
    int number_of_vectors,
    int number_of_plane_waves,
    const float* kinetic_energy,
    const std::complex<float>* psi,
    std::complex<float>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    apply_kinetic_energy_single_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        kinetic_energy,
        reinterpret_cast<const cuComplex*>(psi),
        reinterpret_cast<cuComplex*>(hpsi));
}

void launch_scatter_g_to_fft_grid(
    int number_of_vectors,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const int* minus_g_to_fft_mapping,
    bool is_gamma_only,
    const std::complex<double>* psi,
    std::complex<double>* fft_workspace) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0) return;
    cudaMemset(fft_workspace, 0, number_of_vectors * local_smooth_grid_points * sizeof(std::complex<double>));
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    scatter_g_to_fft_grid_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        local_smooth_grid_points,
        g_to_fft_mapping,
        minus_g_to_fft_mapping,
        is_gamma_only,
        reinterpret_cast<const cuDoubleComplex*>(psi),
        reinterpret_cast<cuDoubleComplex*>(fft_workspace));
}

void launch_scatter_g_to_fft_grid_single(
    int number_of_vectors,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const int* minus_g_to_fft_mapping,
    bool is_gamma_only,
    const std::complex<float>* psi,
    std::complex<float>* fft_workspace) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0) return;
    cudaMemset(fft_workspace, 0, number_of_vectors * local_smooth_grid_points * sizeof(std::complex<float>));
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    scatter_g_to_fft_grid_single_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        local_smooth_grid_points,
        g_to_fft_mapping,
        minus_g_to_fft_mapping,
        is_gamma_only,
        reinterpret_cast<const cuComplex*>(psi),
        reinterpret_cast<cuComplex*>(fft_workspace));
}

void launch_scatter_g_to_fft_grid_mixed(
    int number_of_vectors,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const int* minus_g_to_fft_mapping,
    bool is_gamma_only,
    const std::complex<double>* psi,
    std::complex<float>* fft_workspace) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0) return;
    cudaMemset(fft_workspace, 0, number_of_vectors * local_smooth_grid_points * sizeof(std::complex<float>));
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    scatter_g_to_fft_grid_mixed_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        local_smooth_grid_points,
        g_to_fft_mapping,
        minus_g_to_fft_mapping,
        is_gamma_only,
        reinterpret_cast<const cuDoubleComplex*>(psi),
        reinterpret_cast<cuComplex*>(fft_workspace));
}

void launch_multiply_potential(
    int number_of_vectors,
    int local_smooth_grid_points,
    const double* potential,
    std::complex<double>* fft_workspace) {
    int total_elements = number_of_vectors * local_smooth_grid_points;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    multiply_potential_kernel<<<blocks, threads>>>(
        total_elements,
        local_smooth_grid_points,
        potential,
        reinterpret_cast<cuDoubleComplex*>(fft_workspace));
}

void launch_multiply_potential_single(
    int number_of_vectors,
    int local_smooth_grid_points,
    const float* potential,
    std::complex<float>* fft_workspace) {
    int total_elements = number_of_vectors * local_smooth_grid_points;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    multiply_potential_single_kernel<<<blocks, threads>>>(
        total_elements,
        local_smooth_grid_points,
        potential,
        reinterpret_cast<cuComplex*>(fft_workspace));
}

void launch_gather_fft_grid_to_g(
    int number_of_vectors,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const std::complex<double>* fft_workspace,
    std::complex<double>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gather_fft_grid_to_g_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        local_smooth_grid_points,
        g_to_fft_mapping,
        reinterpret_cast<const cuDoubleComplex*>(fft_workspace),
        reinterpret_cast<cuDoubleComplex*>(hpsi));
}

void launch_gather_fft_grid_to_g_single(
    int number_of_vectors,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const std::complex<float>* fft_workspace,
    std::complex<float>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gather_fft_grid_to_g_single_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        local_smooth_grid_points,
        g_to_fft_mapping,
        reinterpret_cast<const cuComplex*>(fft_workspace),
        reinterpret_cast<cuComplex*>(hpsi));
}

void launch_gather_fft_grid_to_g_mixed(
    int number_of_vectors,
    int number_of_plane_waves,
    int local_smooth_grid_points,
    const int* g_to_fft_mapping,
    const std::complex<float>* fft_workspace,
    std::complex<double>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gather_fft_grid_to_g_mixed_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        local_smooth_grid_points,
        g_to_fft_mapping,
        reinterpret_cast<const cuComplex*>(fft_workspace),
        reinterpret_cast<cuDoubleComplex*>(hpsi));
}

void launch_compute_residuals(
    int number_of_bands,
    int number_of_plane_waves,
    const double* eigenvalues,
    const std::complex<double>* hamiltonian_ritz_vectors,
    const std::complex<double>* overlap_ritz_vectors,
    std::complex<double>* residual_wavefunctions) {
    int total_elements = number_of_bands * number_of_plane_waves;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    compute_residuals_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        eigenvalues,
        reinterpret_cast<const cuDoubleComplex*>(hamiltonian_ritz_vectors),
        reinterpret_cast<const cuDoubleComplex*>(overlap_ritz_vectors),
        reinterpret_cast<cuDoubleComplex*>(residual_wavefunctions));
}

void launch_apply_preconditioner(
    int number_of_bands,
    int number_of_plane_waves,
    const double* kinetic_energy_squared,
    const double* eigenvalues,
    const bool* band_converged,
    double floor_value,
    std::complex<double>* residual_wavefunctions) {
    int total_elements = number_of_bands * number_of_plane_waves;
    if (total_elements == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    apply_preconditioner_kernel<<<blocks, threads>>>(
        total_elements,
        number_of_plane_waves,
        kinetic_energy_squared,
        eigenvalues,
        band_converged,
        floor_value,
        reinterpret_cast<cuDoubleComplex*>(residual_wavefunctions));
}

} // namespace gpu
} // namespace planewave
} // namespace diagonalizer
} // namespace q3
