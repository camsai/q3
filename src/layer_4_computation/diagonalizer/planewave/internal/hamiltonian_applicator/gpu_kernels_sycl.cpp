#include "layer_4_computation/diagonalizer/planewave/internal/hamiltonian_applicator/gpu_kernels.hpp"

#include <cmath>
#include <sycl/sycl.hpp>

namespace q3 {
namespace diagonalizer {
namespace planewave {
namespace gpu {

static sycl::queue get_sycl_queue() {
    static sycl::queue queue{sycl::default_selector_v};
    return queue;
}

void launch_apply_kinetic_energy(int number_of_vectors,
                                 int number_of_plane_waves,
                                 const double* kinetic_energy,
                                 const std::complex<double>* psi,
                                 std::complex<double>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    auto queue = get_sycl_queue();
    queue
        .parallel_for(sycl::range<1>(total_elements),
                      [=](sycl::id<1> idx) {
                          int index = idx[0];
                          int g_index = index % number_of_plane_waves;
                          hpsi[index] = kinetic_energy[g_index] * psi[index];
                      })
        .wait();
}

void launch_apply_kinetic_energy_single(int number_of_vectors,
                                        int number_of_plane_waves,
                                        const float* kinetic_energy,
                                        const std::complex<float>* psi,
                                        std::complex<float>* hpsi) {
    int total_elements = number_of_vectors * number_of_plane_waves;
    if (total_elements == 0)
        return;
    auto queue = get_sycl_queue();
    queue
        .parallel_for(sycl::range<1>(total_elements),
                      [=](sycl::id<1> idx) {
                          int index = idx[0];
                          int g_index = index % number_of_plane_waves;
                          hpsi[index] = kinetic_energy[g_index] * psi[index];
                      })
        .wait();
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
    auto queue = get_sycl_queue();
    queue
        .memset(fft_workspace,
                0,
                number_of_vectors * local_smooth_grid_points * sizeof(std::complex<double>))
        .wait();
    queue
        .parallel_for(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int index = idx[0];
                int vector_index = index / number_of_plane_waves;
                int g_index = index % number_of_plane_waves;
                std::complex<double> val = psi[index];
                int fft_index = g_to_fft_mapping[g_index];
                if (fft_index >= 0) {
                    fft_workspace[vector_index * local_smooth_grid_points + fft_index] = val;
                }
                if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
                    int fft_minus = minus_g_to_fft_mapping[g_index];
                    if (fft_minus >= 0) {
                        fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                            std::conj(val);
                    }
                }
            })
        .wait();
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
    auto queue = get_sycl_queue();
    queue
        .memset(fft_workspace,
                0,
                number_of_vectors * local_smooth_grid_points * sizeof(std::complex<float>))
        .wait();
    queue
        .parallel_for(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int index = idx[0];
                int vector_index = index / number_of_plane_waves;
                int g_index = index % number_of_plane_waves;
                std::complex<float> val = psi[index];
                int fft_index = g_to_fft_mapping[g_index];
                if (fft_index >= 0) {
                    fft_workspace[vector_index * local_smooth_grid_points + fft_index] = val;
                }
                if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
                    int fft_minus = minus_g_to_fft_mapping[g_index];
                    if (fft_minus >= 0) {
                        fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                            std::conj(val);
                    }
                }
            })
        .wait();
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
    auto queue = get_sycl_queue();
    queue
        .memset(fft_workspace,
                0,
                number_of_vectors * local_smooth_grid_points * sizeof(std::complex<float>))
        .wait();
    queue
        .parallel_for(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int index = idx[0];
                int vector_index = index / number_of_plane_waves;
                int g_index = index % number_of_plane_waves;
                std::complex<double> val = psi[index];
                std::complex<float> val_single(static_cast<float>(val.real()),
                                               static_cast<float>(val.imag()));
                int fft_index = g_to_fft_mapping[g_index];
                if (fft_index >= 0) {
                    fft_workspace[vector_index * local_smooth_grid_points + fft_index] = val_single;
                }
                if (is_gamma_only && g_index > 0 && minus_g_to_fft_mapping != nullptr) {
                    int fft_minus = minus_g_to_fft_mapping[g_index];
                    if (fft_minus >= 0) {
                        fft_workspace[vector_index * local_smooth_grid_points + fft_minus] =
                            std::conj(val_single);
                    }
                }
            })
        .wait();
}

void launch_multiply_potential(int number_of_vectors,
                               int local_smooth_grid_points,
                               const double* potential,
                               std::complex<double>* fft_workspace) {
    int total_elements = number_of_vectors * local_smooth_grid_points;
    if (total_elements == 0)
        return;
    auto queue = get_sycl_queue();
    queue
        .parallel_for(sycl::range<1>(total_elements),
                      [=](sycl::id<1> idx) {
                          int index = idx[0];
                          int grid_index = index % local_smooth_grid_points;
                          fft_workspace[index] *= potential[grid_index];
                      })
        .wait();
}

void launch_multiply_potential_single(int number_of_vectors,
                                      int local_smooth_grid_points,
                                      const float* potential,
                                      std::complex<float>* fft_workspace) {
    int total_elements = number_of_vectors * local_smooth_grid_points;
    if (total_elements == 0)
        return;
    auto queue = get_sycl_queue();
    queue
        .parallel_for(sycl::range<1>(total_elements),
                      [=](sycl::id<1> idx) {
                          int index = idx[0];
                          int grid_index = index % local_smooth_grid_points;
                          fft_workspace[index] *= potential[grid_index];
                      })
        .wait();
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
    auto queue = get_sycl_queue();
    queue
        .parallel_for(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int index = idx[0];
                int vector_index = index / number_of_plane_waves;
                int g_index = index % number_of_plane_waves;
                int fft_index = g_to_fft_mapping[g_index];
                if (fft_index >= 0) {
                    hpsi[index] +=
                        fft_workspace[vector_index * local_smooth_grid_points + fft_index];
                }
            })
        .wait();
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
    auto queue = get_sycl_queue();
    queue
        .parallel_for(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int index = idx[0];
                int vector_index = index / number_of_plane_waves;
                int g_index = index % number_of_plane_waves;
                int fft_index = g_to_fft_mapping[g_index];
                if (fft_index >= 0) {
                    hpsi[index] +=
                        fft_workspace[vector_index * local_smooth_grid_points + fft_index];
                }
            })
        .wait();
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
    auto queue = get_sycl_queue();
    queue
        .parallel_for(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int index = idx[0];
                int vector_index = index / number_of_plane_waves;
                int g_index = index % number_of_plane_waves;
                int fft_index = g_to_fft_mapping[g_index];
                if (fft_index >= 0) {
                    std::complex<float> val =
                        fft_workspace[vector_index * local_smooth_grid_points + fft_index];
                    hpsi[index] += std::complex<double>(static_cast<double>(val.real()),
                                                        static_cast<double>(val.imag()));
                }
            })
        .wait();
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
    auto queue = get_sycl_queue();
    queue
        .parallel_for(sycl::range<1>(total_elements),
                      [=](sycl::id<1> idx) {
                          int index = idx[0];
                          int band_index = index / number_of_plane_waves;
                          residual_wavefunctions[index] =
                              hamiltonian_ritz_vectors[index] -
                              eigenvalues[band_index] * overlap_ritz_vectors[index];
                      })
        .wait();
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
    auto queue = get_sycl_queue();
    queue
        .parallel_for(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int index = idx[0];
                int band_index = index / number_of_plane_waves;
                if (band_converged[band_index]) {
                    return;
                }
                int g_index = index % number_of_plane_waves;
                double value_x = kinetic_energy_squared[g_index] - eigenvalues[band_index];
                double denominator =
                    0.5 * (1.0 + value_x + std::sqrt(1.0 + (value_x - 1.0) * (value_x - 1.0)));
                if (denominator < floor_value) {
                    denominator = floor_value;
                }
                residual_wavefunctions[index] /= denominator;
            })
        .wait();
}

} // namespace gpu
} // namespace planewave
} // namespace diagonalizer
} // namespace q3
