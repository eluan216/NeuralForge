#include "neuralforge/tensor.hpp"

#include <algorithm>
#include <limits>
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

// Multiply a*b into out. Returns false if the product would overflow size_t.
[[nodiscard]] bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

// True iff M*K*N < block_size^3 without overflowing intermediate products.
// On overflow of the left-hand volume, the problem is treated as large (false).
// On overflow of the right-hand threshold, the threshold is treated as infinite (true
// only when the left-hand volume is finite — which it is if we reach the compare).
[[nodiscard]] bool volume_below_block_cube(std::size_t M, std::size_t K, std::size_t N,
                                           std::size_t block_size) noexcept {
    std::size_t mk = 0;
    if (!checked_mul(M, K, mk)) {
        return false;  // M*K already overflows → volume is huge
    }
    std::size_t volume = 0;
    if (!checked_mul(mk, N, volume)) {
        return false;  // M*K*N overflows → volume is huge
    }

    std::size_t bb = 0;
    if (!checked_mul(block_size, block_size, bb)) {
        // block_size^2 overflows → block_size^3 is even larger than any finite volume
        return true;
    }
    std::size_t threshold = 0;
    if (!checked_mul(bb, block_size, threshold)) {
        return true;  // block_size^3 overflows → threshold is huge
    }
    return volume < threshold;
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
    // Comparison is overflow-safe (see volume_below_block_cube).
    if (volume_below_block_cube(M, K, N, block_size)) {
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
