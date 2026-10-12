#include "neuralforge/tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace neuralforge {
namespace {

void validate_matmul_shapes(const Tensor& a, const Tensor& b,
                            std::size_t& M, std::size_t& K, std::size_t& N) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul currently supports only 2-D tensors");
    }
    M = a.shape()[0];
    K = a.shape()[1];
    const auto K2 = b.shape()[0];
    N = b.shape()[1];
    if (K != K2) {
        throw std::invalid_argument("matmul shape mismatch");
    }
}

} // namespace

Tensor matmul_naive(const Tensor& a, const Tensor& b) {
    std::size_t M = 0, K = 0, N = 0;
    validate_matmul_shapes(a, b, M, K, N);

    Tensor result({M, N}, 0.0f);
    const float* A = a.data();
    const float* B = b.data();
    float* C = result.data();

    // i-k-j: stream B along rows while A is reused across the inner j loop.
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float aik = A[i * K + k];
            for (std::size_t j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }
    return result;
}

Tensor matmul_blocked(const Tensor& a, const Tensor& b, std::size_t block_size) {
    std::size_t M = 0, K = 0, N = 0;
    validate_matmul_shapes(a, b, M, K, N);

    if (block_size == 0) {
        throw std::invalid_argument("matmul_blocked requires block_size > 0");
    }

    // Small products: blocking overhead is not worth it; use the tight naive loop.
    if (M * K * N < block_size * block_size * block_size) {
        return matmul_naive(a, b);
    }

    Tensor result({M, N}, 0.0f);
    const float* A = a.data();
    const float* B = b.data();
    float* C = result.data();

    // Tile over (i, k, j). Within each tile use i-k-j so the B panel stays hot in L1.
    // Default block_size=64 → three 64×64 float panels ≈ 48 KiB, fits typical L1D.
    for (std::size_t i0 = 0; i0 < M; i0 += block_size) {
        const std::size_t i_max = std::min(i0 + block_size, M);
        for (std::size_t k0 = 0; k0 < K; k0 += block_size) {
            const std::size_t k_max = std::min(k0 + block_size, K);
            for (std::size_t j0 = 0; j0 < N; j0 += block_size) {
                const std::size_t j_max = std::min(j0 + block_size, N);
                for (std::size_t i = i0; i < i_max; ++i) {
                    for (std::size_t k = k0; k < k_max; ++k) {
                        const float aik = A[i * K + k];
                        for (std::size_t j = j0; j < j_max; ++j) {
                            C[i * N + j] += aik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    return matmul_blocked(a, b);
}

} // namespace neuralforge
