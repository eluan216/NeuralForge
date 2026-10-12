#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "neuralforge/tensor.hpp"

using namespace neuralforge;

namespace {

Tensor random_matrix(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor t({rows, cols});
    for (std::size_t i = 0; i < t.size(); ++i) {
        t.at(i) = dist(rng);
    }
    return t;
}

void expect_tensors_near(const Tensor& a, const Tensor& b, float abs_tol = 1e-4f,
                         float rel_tol = 1e-4f) {
    ASSERT_EQ(a.shape(), b.shape());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float x = a.at(i);
        const float y = b.at(i);
        const float tol = abs_tol + rel_tol * std::max(std::fabs(x), std::fabs(y));
        EXPECT_NEAR(x, y, tol) << "index " << i;
    }
}

} // namespace

TEST(MatMulBlocked, MatchesNaiveOnIdentity) {
    Tensor identity{{1.f, 0.f}, {0.f, 1.f}};
    Tensor a{{2.f, 3.f}, {4.f, 5.f}};
    expect_tensors_near(matmul_blocked(a, identity), matmul_naive(a, identity));
}

TEST(MatMulBlocked, MatchesNaiveOnHandExample) {
    Tensor a{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    Tensor b{{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}};
    const auto naive = matmul_naive(a, b);
    const auto blocked = matmul_blocked(a, b, 2);
    expect_tensors_near(naive, blocked, 1e-5f, 1e-5f);
    EXPECT_FLOAT_EQ(blocked.at(0, 0), 58.f);
    EXPECT_FLOAT_EQ(blocked.at(1, 1), 154.f);
}

TEST(MatMulBlocked, MatchesNaiveRandomSquare) {
    std::mt19937 rng(42);
    for (std::size_t n : {1u, 3u, 7u, 16u, 33u, 64u, 65u}) {
        auto A = random_matrix(n, n, rng);
        auto B = random_matrix(n, n, rng);
        expect_tensors_near(matmul_naive(A, B), matmul_blocked(A, B, 16));
        expect_tensors_near(matmul_naive(A, B), matmul_blocked(A, B, 64));
    }
}

TEST(MatMulBlocked, MatchesNaiveRandomRectangular) {
    std::mt19937 rng(7);
    const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> shapes = {
        {1, 16, 3}, {8, 16, 3}, {32, 17, 9}, {64, 128, 32}, {100, 50, 75}};
    for (const auto& [M, K, N] : shapes) {
        auto A = random_matrix(M, K, rng);
        auto B = random_matrix(K, N, rng);
        expect_tensors_near(matmul_naive(A, B), matmul_blocked(A, B, 32));
    }
}

TEST(MatMulBlocked, DefaultMatmulUsesBlockedPath) {
    // Public matmul() must stay numerically consistent with the blocked kernel.
    std::mt19937 rng(99);
    auto A = random_matrix(48, 48, rng);
    auto B = random_matrix(48, 48, rng);
    expect_tensors_near(matmul(A, B), matmul_blocked(A, B));
}

TEST(MatMulBlocked, RejectsZeroBlockSize) {
    Tensor a = Tensor::ones({2, 2});
    Tensor b = Tensor::ones({2, 2});
    EXPECT_THROW(matmul_blocked(a, b, 0), std::invalid_argument);
}
