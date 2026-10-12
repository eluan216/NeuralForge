#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <vector>

#include "neuralforge/autograd.hpp"
#include "neuralforge/export_model.hpp"
#include "neuralforge/loss.hpp"
#include "neuralforge/model.hpp"
#include "neuralforge/optim.hpp"

using namespace neuralforge;
namespace fs = std::filesystem;

TEST(Training, MseRejectsShapeMismatch) {
    Variable pred(Tensor{{1.f, 2.f}}, true);
    Variable tgt(Tensor{{1.f}}, false);
    EXPECT_THROW(mse_loss(pred, tgt), std::invalid_argument);
}

TEST(Training, MseGradientMatchesFiniteDifference) {
    Variable pred(Tensor{{0.5f, -1.0f}, {2.0f, 0.25f}}, true);
    Variable tgt(Tensor{{1.0f, -1.5f}, {1.5f, 0.0f}}, false);

    auto loss = mse_loss(pred, tgt);
    pred.zero_grad();
    loss.backward();

    auto fd = finite_difference_grad(pred, [&](const Variable& p) {
        Variable t(tgt.data(), false);
        return mse_loss(p, t);
    });

    ASSERT_EQ(pred.grad().shape(), fd.shape());
    for (std::size_t i = 0; i < fd.size(); ++i) {
        EXPECT_NEAR(pred.grad().at(i), fd.at(i), 2e-2f) << "i=" << i;
    }
}

TEST(Training, MseKnownValue) {
    // mean((1-3)^2 + (2-2)^2) = mean(4 + 0) = 2
    Variable pred(Tensor{{1.f, 2.f}}, true);
    Variable tgt(Tensor{{3.f, 2.f}}, false);
    auto loss = mse_loss(pred, tgt);
    EXPECT_FLOAT_EQ(loss.data().at(0), 2.0f);
}

TEST(Training, SgdRejectsInvalidLearningRate) {
    EXPECT_THROW(SGD(0.0f), std::invalid_argument);
    EXPECT_THROW(SGD(-0.1f), std::invalid_argument);
    EXPECT_THROW(SGD(std::nanf("")), std::invalid_argument);
}

TEST(Training, SgdExactParameterUpdate) {
    // θ=1, g=0.5, lr=0.1 → θ' = 1 - 0.05 = 0.95
    Variable theta(Tensor{{1.0f}}, true);
    theta.grad() = Tensor{{0.5f}};
    // grad() returns ref but may be empty — set via zero_grad then assign
    theta.zero_grad();
    theta.grad().at(0) = 0.5f;

    std::vector<Variable> params{theta};
    SGD opt(0.1f);
    opt.step(params);
    EXPECT_FLOAT_EQ(theta.data().at(0), 0.95f);
}

TEST(Training, SgdRejectsNonFiniteGrad) {
    Variable theta(Tensor{{1.0f}}, true);
    theta.zero_grad();
    theta.grad().at(0) = std::numeric_limits<float>::quiet_NaN();
    std::vector<Variable> params{theta};
    SGD opt(0.1f);
    EXPECT_THROW(opt.step(params), std::runtime_error);
}

TEST(Training, TinyLinearRegressionLossDecreases) {
    // Fit y ≈ 2x + 1 with a single linear layer (no hidden).
    // pred = x @ W + b, W shape (1,1), b shape (1).
    Variable W(Tensor{{0.0f}}, true);  // start far from 2
    Variable b(Tensor{{0.0f}}, true);

    // Dataset: x in {-1,0,1,2}, y = 2x+1
    const Tensor X{{-1.f}, {0.f}, {1.f}, {2.f}};
    const Tensor Y{{-1.f}, {1.f}, {3.f}, {5.f}};

    SGD opt(0.05f);
    std::vector<Variable> params{W, b};

    float prev_loss = 1e9f;
    for (int epoch = 0; epoch < 200; ++epoch) {
        Variable x(X, false);
        Variable y(Y, false);
        auto pred = add(matmul(x, W), b);
        auto loss = mse_loss(pred, y);
        opt.zero_grad(params);
        loss.backward();
        opt.step(params);

        const float L = loss.data().at(0);
        if (epoch > 0 && epoch % 50 == 0) {
            EXPECT_LT(L, prev_loss) << "epoch " << epoch;
            prev_loss = L;
        }
        if (epoch == 0) prev_loss = L;
    }

    EXPECT_NEAR(W.data().at(0), 2.0f, 0.15f);
    EXPECT_NEAR(b.data().at(0), 1.0f, 0.15f);
    // Final loss should be small
    Variable x(X, false);
    Variable y(Y, false);
    auto final_loss = mse_loss(add(matmul(x, W), b), y);
    EXPECT_LT(final_loss.data().at(0), 0.05f);
}

TEST(Training, TinyMlpExportLoadsInInference) {
    // One hidden unit MLP on 2-D input → 1 output; train briefly, export, reload.
    Variable W1(Tensor{{0.3f, -0.2f}, {0.1f, 0.4f}}, true);  // 2×2
    Variable b1(Tensor{{0.0f, 0.0f}}, true);
    Variable W2(Tensor{{0.5f}, {-0.3f}}, true);  // 2×1
    Variable b2(Tensor{{0.0f}}, true);

    const Tensor X{{0.5f, -0.5f}, {-0.5f, 0.5f}, {1.0f, 1.0f}, {-1.0f, -1.0f}};
    const Tensor Y{{0.2f}, {0.2f}, {1.0f}, {-1.0f}};

    SGD opt(0.1f);
    std::vector<Variable> params{W1, b1, W2, b2};

    float first = 0.0f, last = 0.0f;
    for (int epoch = 0; epoch < 80; ++epoch) {
        Variable x(X, false);
        Variable y(Y, false);
        auto h = relu(add(matmul(x, W1), b1));
        auto pred = add(matmul(h, W2), b2);
        auto loss = mse_loss(pred, y);
        if (epoch == 0) first = loss.data().at(0);
        opt.zero_grad(params);
        loss.backward();
        opt.step(params);
        if (epoch == 79) last = loss.data().at(0);
    }
    EXPECT_LT(last, first);

    const auto path = fs::temp_directory_path() / "nf_train_export.nfmodel";
    export_mlp_model(path, W1.data(), b1.data(), W2.data(), b2.data());

    auto model = MLPModel::load(path);
    EXPECT_EQ(model.input_features(), 2u);
    EXPECT_EQ(model.hidden_features(), 2u);
    EXPECT_EQ(model.output_features(), 1u);

    // Inference path runs without throwing on a batch matching the training inputs.
    Tensor batch = X;
    auto probs = model.predict_proba(batch);
    EXPECT_EQ(probs.shape()[0], 4u);
    EXPECT_EQ(probs.shape()[1], 1u);

    fs::remove(path);
}
