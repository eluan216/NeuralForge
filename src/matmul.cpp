#include "neuralforge/tensor.hpp"
#include <stdexcept>
#include <string>

namespace neuralforge {
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul currently supports only 2-D tensors");
    }
    const auto M = a.shape()[0];
    const auto K = a.shape()[1];
    const auto K2 = b.shape()[0];
    const auto N = b.shape()[1];
    if (K != K2) {
        throw std::invalid_argument("matmul shape mismatch");
    }
    Tensor result({M, N}, 0.0f);
    const float* A = a.data();
    const float* B = b.data();
    float* C = result.data();
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float aik = A[i * K + k];
            for (std::size_t j = 0; j < N; ++j) C[i * N + j] += aik * B[k * N + j];
        }
    }
    return result;
}
} // namespace neuralforge
