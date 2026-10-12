#include "neuralforge/tensor.hpp"

#include <stdexcept>

namespace neuralforge {

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Currently only support rank-2 matrices (baseline for Weeks 1–2)
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul currently supports only 2-D tensors");
    }

    const auto M = a.shape()[0];
    const auto K = a.shape()[1];
    const auto K2 = b.shape()[0];
    const auto N = b.shape()[1];

    if (K != K2) {
        throw std::invalid_argument(
            "matmul shape mismatch: A is (" + std::to_string(M) + "," +
            std::to_string(K) + ") but B is (" + std::to_string(K2) + "," +
            std::to_string(N) + ")");
    }

    Tensor result({M, N}, 0.0f);

    // Naive triple-loop – intentional baseline.
    // Later weeks will replace / accelerate this with blocked, SIMD, or
    // multi-threaded implementations and measure the difference.
    const float* A = a.data();
    const float* B = b.data();
    float* C = result.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const float aik = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    return result;
}

} // namespace neuralforge
