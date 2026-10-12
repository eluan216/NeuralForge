#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "neuralforge/model.hpp"
#include "neuralforge/nn_ops.hpp"
#include "neuralforge/tensor.hpp"

using namespace neuralforge;
namespace fs = std::filesystem;

namespace {

fs::path write_temp_model(const std::string& body, const std::string& name) {
    const auto path = fs::temp_directory_path() / name;
    std::ofstream out(path);
    out << body;
    return path;
}

std::string valid_identity_model() {
    return "NEURALFORGE_MLP 1\n"
           "2 2 2\n"
           "1 0 0 1\n"
           "0 0\n"
           "1 0 0 1\n"
           "0 0\n";
}

} // namespace

// ---------------------------------------------------------------------------
// Tensor / matmul zero dimensions
// ---------------------------------------------------------------------------

TEST(Reliability, ZeroDimTensorIsEmpty) {
    Tensor t({0, 4});
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.shape()[0], 0u);
    EXPECT_EQ(t.shape()[1], 4u);
}

TEST(Reliability, MatMulEmptyBatchProducesEmptyOutput) {
    Tensor a({0, 3}, 1.0f);
    Tensor b = Tensor::ones({3, 5});
    auto c = matmul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<std::size_t>{0, 5}));
    EXPECT_TRUE(c.empty());
}

TEST(Reliability, MatMulEmptyInnerDimProducesZeros) {
    // (2×0) @ (0×3) → (2×3) of zeros (no multiply-adds).
    Tensor a({2, 0});
    Tensor b({0, 3});
    auto c = matmul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(c.size(), 6u);
    for (std::size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c.at(i), 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Dense / ReLU / Softmax edge cases
// ---------------------------------------------------------------------------

TEST(Reliability, DenseEmptyBatch) {
    Tensor input({0, 2});
    Tensor weights{{1.f, 0.f}, {0.f, 1.f}};
    Tensor bias = Tensor::zeros({2});
    auto out = dense(input, weights, bias);
    EXPECT_EQ(out.shape(), (std::vector<std::size_t>{0, 2}));
}

TEST(Reliability, SoftmaxEmptyBatch) {
    Tensor logits({0, 3});
    auto probs = softmax_rows(logits);
    EXPECT_EQ(probs.shape(), (std::vector<std::size_t>{0, 3}));
    EXPECT_TRUE(probs.empty());
}

TEST(Reliability, SoftmaxRejectsZeroColumns) {
    Tensor logits({2, 0});
    EXPECT_THROW(softmax_rows(logits), std::invalid_argument);
}

TEST(Reliability, SoftmaxRejectsNonFinite) {
    Tensor logits{{1.f, std::numeric_limits<float>::quiet_NaN()}};
    EXPECT_THROW(softmax_rows(logits), std::invalid_argument);
    Tensor infs{{1.f, std::numeric_limits<float>::infinity()}};
    EXPECT_THROW(softmax_rows(infs), std::invalid_argument);
}

TEST(Reliability, SoftmaxExtremeButFiniteLogits) {
    // Large dynamic range; stable softmax should still sum to ~1.
    Tensor logits{{-1.0e5f, 0.0f, 1.0e5f}};
    auto probs = softmax_rows(logits);
    double sum = 0.0;
    for (std::size_t c = 0; c < 3; ++c) sum += probs.at(0, c);
    EXPECT_NEAR(sum, 1.0, 1e-5);
    EXPECT_NEAR(probs.at(0, 2), 1.0f, 1e-5f);
}

TEST(Reliability, ReluOnEmptyTensor) {
    Tensor empty({0, 4});
    auto out = relu(empty);
    EXPECT_TRUE(out.empty());
}

// ---------------------------------------------------------------------------
// Model inference: empty batch + non-finite inputs
// ---------------------------------------------------------------------------

TEST(Reliability, PredictEmptyBatch) {
    const auto path = write_temp_model(valid_identity_model(), "nf_rel_empty_batch.nfmodel");
    auto model = MLPModel::load(path);
    Tensor inputs({0, 2});
    auto probs = model.predict_proba(inputs);
    EXPECT_EQ(probs.shape(), (std::vector<std::size_t>{0, 2}));
    auto classes = model.predict(inputs);
    EXPECT_TRUE(classes.empty());
    fs::remove(path);
}

TEST(Reliability, PredictRejectsNonFiniteInput) {
    const auto path = write_temp_model(valid_identity_model(), "nf_rel_nan_in.nfmodel");
    auto model = MLPModel::load(path);
    Tensor bad{{1.0f, std::numeric_limits<float>::quiet_NaN()}};
    EXPECT_THROW(model.predict_proba(bad), std::invalid_argument);
    Tensor inf_in{{std::numeric_limits<float>::infinity(), 0.0f}};
    EXPECT_THROW(model.predict(inf_in), std::invalid_argument);
    fs::remove(path);
}

// ---------------------------------------------------------------------------
// Model load: resource limits (reject before allocating huge weights)
// ---------------------------------------------------------------------------

TEST(Reliability, OversizedFeatureDimThrows) {
    // 70000 > kMaxFeatureDim (65536)
    const auto path = write_temp_model("NEURALFORGE_MLP 1\n70000 2 2\n", "nf_rel_big_in.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

TEST(Reliability, ParameterBudgetThrows) {
    // 4096*4096*2 weight matrices far exceed kMaxParameterElements (16M).
    const auto path =
        write_temp_model("NEURALFORGE_MLP 1\n4096 4096 4096\n", "nf_rel_budget.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

TEST(Reliability, DimensionAtLimitStillRejectedIfProductTooLarge) {
    // Each axis under 65536 but product of weights exceeds budget.
    const auto path =
        write_temp_model("NEURALFORGE_MLP 1\n8192 8192 4\n", "nf_rel_product.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}
