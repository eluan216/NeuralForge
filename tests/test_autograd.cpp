#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "neuralforge/autograd.hpp"

using namespace neuralforge;

namespace {

void expect_tensors_near(const Tensor& a, const Tensor& b, float tol = 2e-2f) {
    ASSERT_EQ(a.shape(), b.shape());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a.at(i), b.at(i), tol) << "index " << i;
    }
}

} // namespace

TEST(Autograd, SumGradientMatchesFiniteDifference) {
    Variable x(Tensor{{1.f, -2.f}, {3.f, 0.5f}}, true);
    auto loss = sum(x);
    loss.backward();

    auto fd = finite_difference_grad(x, [](const Variable& v) { return sum(v); });
    expect_tensors_near(x.grad(), fd, 1e-3f);
}

TEST(Autograd, ReluGradientMatchesFiniteDifference) {
    // Avoid the non-differentiable kink at 0 for FD comparison.
    Variable x(Tensor{{1.f, -2.f}, {3.f, 0.5f}}, true);
    auto loss = sum(relu(x));
    x.zero_grad();
    loss.backward();

    auto fd = finite_difference_grad(x, [](const Variable& v) { return sum(relu(v)); });
    expect_tensors_near(x.grad(), fd, 1e-3f);
}

TEST(Autograd, MatMulGradientMatchesFiniteDifference) {
    Variable a(Tensor{{0.5f, -1.f}, {1.5f, 2.f}}, true);
    Variable b(Tensor{{1.f, 0.5f}, {-0.5f, 2.f}}, false);

    auto loss = sum(matmul(a, b));
    a.zero_grad();
    loss.backward();

    auto fd = finite_difference_grad(a, [&](const Variable& va) {
        Variable bb(b.data(), false);
        return sum(matmul(va, bb));
    });
    expect_tensors_near(a.grad(), fd, 2e-2f);
}

TEST(Autograd, MatMulGradientWrtB) {
    Variable a(Tensor{{0.5f, -1.f}, {1.5f, 2.f}}, false);
    Variable b(Tensor{{1.f, 0.5f}, {-0.5f, 2.f}}, true);

    auto loss = sum(matmul(a, b));
    b.zero_grad();
    loss.backward();

    auto fd = finite_difference_grad(b, [&](const Variable& vb) {
        Variable aa(a.data(), false);
        return sum(matmul(aa, vb));
    });
    expect_tensors_near(b.grad(), fd, 2e-2f);
}

TEST(Autograd, BiasAddBroadcastGradient) {
    Variable x(Tensor{{1.f, 2.f}, {3.f, 4.f}}, true);
    Variable bias(Tensor::zeros({2}), true);
    bias.data().at(0) = 0.5f;
    bias.data().at(1) = -1.f;

    auto loss = sum(add(x, bias));
    x.zero_grad();
    bias.zero_grad();
    loss.backward();

    // ∂L/∂x = 1 for each element; ∂L/∂bias_j = sum over batch
    EXPECT_FLOAT_EQ(x.grad().at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(x.grad().at(1, 1), 1.f);
    EXPECT_FLOAT_EQ(bias.grad().at(0), 2.f);
    EXPECT_FLOAT_EQ(bias.grad().at(1), 2.f);
}

TEST(Autograd, TinyMlpPathMatchesFiniteDifference) {
    // y = sum(relu(x @ W + b))
    Variable x(Tensor{{0.2f, -0.3f, 0.5f}}, true);  // 1×3
    Variable W(Tensor{{0.1f, -0.2f}, {0.3f, 0.4f}, {-0.5f, 0.6f}}, true);  // 3×2
    Variable b(Tensor{{0.05f, -0.1f}}, true);  // 2

    auto hidden = relu(add(matmul(x, W), b));
    auto loss = sum(hidden);
    x.zero_grad();
    W.zero_grad();
    b.zero_grad();
    loss.backward();

    auto fd_W = finite_difference_grad(W, [&](const Variable& Ww) {
        Variable xx(x.data(), false);
        Variable bb(b.data(), false);
        return sum(relu(add(matmul(xx, Ww), bb)));
    });
    expect_tensors_near(W.grad(), fd_W, 3e-2f);

    auto fd_b = finite_difference_grad(b, [&](const Variable& bb) {
        Variable xx(x.data(), false);
        Variable Ww(W.data(), false);
        return sum(relu(add(matmul(xx, Ww), bb)));
    });
    expect_tensors_near(b.grad(), fd_b, 3e-2f);
}

TEST(Autograd, BackwardRequiresScalar) {
    Variable x(Tensor{{1.f, 2.f}}, true);
    EXPECT_THROW(x.backward(), std::invalid_argument);
}
