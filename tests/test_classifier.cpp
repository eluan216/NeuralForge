#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <random>
#include <vector>

#include "neuralforge/autograd.hpp"
#include "neuralforge/export_model.hpp"
#include "neuralforge/loss.hpp"
#include "neuralforge/model.hpp"
#include "neuralforge/nn_ops.hpp"
#include "neuralforge/optim.hpp"

using namespace neuralforge;
namespace fs = std::filesystem;

namespace {

// Deterministic 3-class Gaussian clusters (matches the PyTorch toy setup).
void make_clusters(std::size_t n_per_class, Tensor& x_out, std::vector<std::size_t>& y_out,
                   unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.55f);
    const float centers[3][2] = {{-2.0f, -1.5f}, {2.0f, -1.0f}, {0.0f, 2.0f}};
    const std::size_t n = n_per_class * 3;
    x_out = Tensor({n, 2});
    y_out.resize(n);
    std::size_t idx = 0;
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < n_per_class; ++i) {
            x_out.at(idx, 0) = centers[c][0] + noise(rng);
            x_out.at(idx, 1) = centers[c][1] + noise(rng);
            y_out[idx] = c;
            ++idx;
        }
    }
}

float accuracy(const Tensor& logits, const std::vector<std::size_t>& y) {
    std::size_t correct = 0;
    const auto batch = logits.shape()[0];
    const auto classes = logits.shape()[1];
    for (std::size_t i = 0; i < batch; ++i) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < classes; ++j) {
            if (logits.at(i, j) > logits.at(i, best)) best = j;
        }
        if (best == y[i]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(batch);
}

} // namespace

TEST(Classifier, CrossEntropyKnownValue) {
    // logits = [[2, 0]], target = 0 → -log(softmax) ≈ -log(0.8808) ≈ 0.1269
    Variable logits(Tensor{{2.0f, 0.0f}}, true);
    auto loss = cross_entropy_loss(logits, {0});
    EXPECT_NEAR(loss.data().at(0), 0.126928f, 1e-4f);
}

TEST(Classifier, CrossEntropyGradientMatchesFiniteDifference) {
    Variable logits(Tensor{{1.0f, -0.5f, 0.2f}, {0.0f, 2.0f, -1.0f}}, true);
    const std::vector<std::size_t> targets{0, 1};

    auto loss = cross_entropy_loss(logits, targets);
    logits.zero_grad();
    loss.backward();

    auto fd = finite_difference_grad(logits, [&](const Variable& z) {
        return cross_entropy_loss(z, targets);
    }, 1e-3f);

    ASSERT_EQ(logits.grad().shape(), fd.shape());
    for (std::size_t i = 0; i < fd.size(); ++i) {
        EXPECT_NEAR(logits.grad().at(i), fd.at(i), 3e-2f) << "i=" << i;
    }
}

TEST(Classifier, CrossEntropyRejectsBadTargets) {
    Variable logits(Tensor{{1.0f, 0.0f}}, true);
    EXPECT_THROW(cross_entropy_loss(logits, {2}), std::invalid_argument);
    EXPECT_THROW(cross_entropy_loss(logits, {0, 1}), std::invalid_argument);
}

TEST(Classifier, AdamRejectsInvalidHyperparams) {
    EXPECT_THROW(Adam(-0.1f), std::invalid_argument);
    EXPECT_THROW(Adam(1e-3f, 1.0f), std::invalid_argument);
    EXPECT_THROW(Adam(1e-3f, 0.9f, 1.0f), std::invalid_argument);
    EXPECT_THROW(Adam(1e-3f, 0.9f, 0.999f, 0.0f), std::invalid_argument);
}

TEST(Classifier, AdamExactFirstStep) {
    // θ=0, g=1, lr=0.1, β1=0.9, β2=0.999, ε=1e-8
    // m=0.1, v=0.001, m̂=1, v̂=1 → θ' = 0 - 0.1 * 1 / (1+ε) ≈ -0.1
    Variable theta(Tensor::zeros({1}), true);
    theta.zero_grad();
    theta.grad().at(0) = 1.0f;

    std::vector<Variable> params{theta};
    Adam opt(0.1f, 0.9f, 0.999f, 1e-8f);
    opt.step(params);
    EXPECT_EQ(opt.timestep(), 1u);
    EXPECT_NEAR(theta.data().at(0), -0.1f, 1e-5f);
}

TEST(Classifier, AdamRejectsNonFiniteWithoutPartialUpdate) {
    Variable a(Tensor::full({1}, 1.0f), true);
    Variable b(Tensor::full({1}, 2.0f), true);
    a.zero_grad();
    b.zero_grad();
    a.grad().at(0) = 0.5f;
    b.grad().at(0) = std::numeric_limits<float>::quiet_NaN();

    const float a_before = a.data().at(0);
    const float b_before = b.data().at(0);
    std::vector<Variable> params{a, b};
    Adam opt(0.1f);
    EXPECT_THROW(opt.step(params), std::runtime_error);
    EXPECT_FLOAT_EQ(a.data().at(0), a_before);
    EXPECT_FLOAT_EQ(b.data().at(0), b_before);
    EXPECT_EQ(opt.timestep(), 0u);  // validation failed before increment... wait we increment after validate
}

TEST(Classifier, TrainExportAndInfer) {
    Tensor x_train, x_val;
    std::vector<std::size_t> y_train, y_val;
    make_clusters(80, x_train, y_train, /*seed=*/7);
    make_clusters(20, x_val, y_val, /*seed=*/99);

    // 2 → 16 → 3 MLP, small random init
    std::mt19937 rng(123);
    std::normal_distribution<float> n01(0.0f, 0.1f);
    Tensor w1t({2, 16}), b1t({16}, 0.0f), w2t({16, 3}), b2t({3}, 0.0f);
    for (std::size_t i = 0; i < w1t.size(); ++i) w1t.at(i) = n01(rng);
    for (std::size_t i = 0; i < w2t.size(); ++i) w2t.at(i) = n01(rng);

    Variable W1(w1t, true), b1(b1t, true), W2(w2t, true), b2(b2t, true);
    std::vector<Variable> params{W1, b1, W2, b2};
    Adam opt(0.05f);

    float first_loss = 0.0f, last_loss = 0.0f;
    for (int epoch = 0; epoch < 120; ++epoch) {
        Variable x(x_train, false);
        auto h = relu(add(matmul(x, W1), b1));
        auto logits = add(matmul(h, W2), b2);
        auto loss = cross_entropy_loss(logits, y_train);
        if (epoch == 0) first_loss = loss.data().at(0);
        opt.zero_grad(params);
        loss.backward();
        opt.step(params);
        if (epoch == 119) last_loss = loss.data().at(0);
    }
    EXPECT_LT(last_loss, first_loss);

    // Held-out accuracy (separate from training loss)
    Variable xv(x_val, false);
    auto h_val = relu(add(matmul(xv, W1), b1));
    auto logits_val = add(matmul(h_val, W2), b2);
    const float val_acc = accuracy(logits_val.data(), y_val);
    EXPECT_GT(val_acc, 0.85f);

    // Export and reload through the inference path
    const auto path = fs::temp_directory_path() / "nf_classifier_train.nfmodel";
    export_mlp_model(path, W1.data(), b1.data(), W2.data(), b2.data());
    auto model = MLPModel::load(path);
    EXPECT_EQ(model.input_features(), 2u);
    EXPECT_EQ(model.hidden_features(), 16u);
    EXPECT_EQ(model.output_features(), 3u);

    // Class predictions from inference path should match autograd argmax on val set
    auto proba = model.predict_proba(x_val);
    auto pred_classes = model.predict(x_val);
    ASSERT_EQ(pred_classes.size(), y_val.size());
    std::size_t matches = 0;
    for (std::size_t i = 0; i < y_val.size(); ++i) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < 3; ++j) {
            if (logits_val.data().at(i, j) > logits_val.data().at(i, best)) best = j;
        }
        EXPECT_EQ(pred_classes[i], best);
        if (pred_classes[i] == y_val[i]) ++matches;
        // Probabilities finite and sum ≈ 1
        float s = 0.0f;
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_TRUE(std::isfinite(proba.at(i, j)));
            s += proba.at(i, j);
        }
        EXPECT_NEAR(s, 1.0f, 1e-4f);
    }
    const float loaded_acc = static_cast<float>(matches) / static_cast<float>(y_val.size());
    EXPECT_NEAR(loaded_acc, val_acc, 1e-5f);

    fs::remove(path);
}
