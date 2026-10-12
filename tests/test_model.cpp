#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "neuralforge/model.hpp"

namespace fs = std::filesystem;
using namespace neuralforge;

namespace {

fs::path write_temp_model(const std::string& contents, const std::string& name) {
    const fs::path path = fs::temp_directory_path() / name;
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

// Minimal valid 2→2→2 model with known arithmetic for prediction checks.
// W1 = [[1, 0], [0, 1]], b1 = [0, 0]
// W2 = [[1, 0], [0, 1]], b2 = [0, 0]
// Forward for input [1, 2]:
//   h = ReLU([1, 2]) = [1, 2]
//   logits = [1, 2]
//   softmax ≈ [0.268941, 0.731059]
std::string valid_identity_model() {
    return
        "NEURALFORGE_MLP 1\n"
        "2 2 2\n"
        "1 0 0 1\n"   // W1 row-major (2x2)
        "0 0\n"       // b1
        "1 0 0 1\n"   // W2 row-major (2x2)
        "0 0\n";      // b2
}

} // namespace

// ---------------------------------------------------------------------------
// Valid load
// ---------------------------------------------------------------------------

TEST(ModelLoader, LoadsValidMinimalModel) {
    const auto path = write_temp_model(valid_identity_model(), "nf_valid_min.nfmodel");
    auto model = MLPModel::load(path);
    EXPECT_EQ(model.input_features(), 2u);
    EXPECT_EQ(model.hidden_features(), 2u);
    EXPECT_EQ(model.output_features(), 2u);
    fs::remove(path);
}

TEST(ModelLoader, LoadsShippedToyClassifier) {
#ifdef NEURALFORGE_SOURCE_DIR
    const fs::path path = fs::path(NEURALFORGE_SOURCE_DIR) / "models" / "toy_classifier.nfmodel";
    ASSERT_TRUE(fs::exists(path)) << "Expected sample model at " << path;
    auto model = MLPModel::load(path);
    EXPECT_EQ(model.input_features(), 2u);
    EXPECT_EQ(model.hidden_features(), 16u);
    EXPECT_EQ(model.output_features(), 3u);
#else
    GTEST_SKIP() << "NEURALFORGE_SOURCE_DIR not defined";
#endif
}

// ---------------------------------------------------------------------------
// Missing / unreadable file
// ---------------------------------------------------------------------------

TEST(ModelLoader, MissingFileThrows) {
    EXPECT_THROW(MLPModel::load("/nonexistent/path/no_such_model.nfmodel"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Malformed headers
// ---------------------------------------------------------------------------

TEST(ModelLoader, BadMagicThrows) {
    const auto path = write_temp_model("WRONG_MAGIC 1\n2 2 2\n", "nf_bad_magic.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

TEST(ModelLoader, UnsupportedVersionThrows) {
    const auto path = write_temp_model("NEURALFORGE_MLP 99\n2 2 2\n", "nf_bad_ver.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

TEST(ModelLoader, EmptyFileThrows) {
    const auto path = write_temp_model("", "nf_empty.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

TEST(ModelLoader, ZeroDimensionThrows) {
    const auto path = write_temp_model("NEURALFORGE_MLP 1\n0 2 2\n", "nf_zero_dim.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

TEST(ModelLoader, NegativeDimensionThrows) {
    const auto path = write_temp_model("NEURALFORGE_MLP 1\n2 -1 2\n", "nf_neg_dim.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

// ---------------------------------------------------------------------------
// Truncated weights
// ---------------------------------------------------------------------------

TEST(ModelLoader, TruncatedWeightsThrows) {
    // Header claims 2x2 W1 but only one value provided.
    const auto path = write_temp_model(
        "NEURALFORGE_MLP 1\n"
        "2 2 2\n"
        "1.0\n",  // truncated W1
        "nf_trunc.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

TEST(ModelLoader, TrailingExtraTokensThrows) {
    const auto path = write_temp_model(
        valid_identity_model() + "999\n",
        "nf_trailing.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

// ---------------------------------------------------------------------------
// Non-finite parameters
// ---------------------------------------------------------------------------

TEST(ModelLoader, NonFiniteWeightThrows) {
    const auto path = write_temp_model(
        "NEURALFORGE_MLP 1\n"
        "2 2 2\n"
        "1 0 nan 1\n"
        "0 0\n"
        "1 0 0 1\n"
        "0 0\n",
        "nf_nan.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

TEST(ModelLoader, InfiniteBiasThrows) {
    const auto path = write_temp_model(
        "NEURALFORGE_MLP 1\n"
        "2 2 2\n"
        "1 0 0 1\n"
        "0 inf\n"
        "1 0 0 1\n"
        "0 0\n",
        "nf_inf.nfmodel");
    EXPECT_THROW(MLPModel::load(path), std::runtime_error);
    fs::remove(path);
}

// ---------------------------------------------------------------------------
// Incorrect input shapes at prediction time
// ---------------------------------------------------------------------------

TEST(ModelLoader, WrongFeatureCountThrows) {
    const auto path = write_temp_model(valid_identity_model(), "nf_shape.nfmodel");
    auto model = MLPModel::load(path);
    Tensor bad({1, 3}, 0.0f);  // expects 2 features
    EXPECT_THROW(model.predict_proba(bad), std::invalid_argument);
    EXPECT_THROW(model.predict(bad), std::invalid_argument);
    fs::remove(path);
}

TEST(ModelLoader, Rank1InputThrows) {
    const auto path = write_temp_model(valid_identity_model(), "nf_rank1.nfmodel");
    auto model = MLPModel::load(path);
    Tensor bad = Tensor::zeros({2});  // rank-1, not (batch, features)
    EXPECT_THROW(model.predict_proba(bad), std::invalid_argument);
    fs::remove(path);
}

// ---------------------------------------------------------------------------
// Prediction results (deterministic hand-crafted model)
// ---------------------------------------------------------------------------

TEST(ModelLoader, PredictProbaSumsToOne) {
    const auto path = write_temp_model(valid_identity_model(), "nf_pred.nfmodel");
    auto model = MLPModel::load(path);

    Tensor inputs{{1.f, 2.f}, {-1.f, 0.5f}};
    const Tensor probs = model.predict_proba(inputs);

    ASSERT_EQ(probs.shape(), (std::vector<std::size_t>{2, 2}));
    for (std::size_t row = 0; row < 2; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < 2; ++col) {
            EXPECT_GE(probs.at(row, col), 0.f);
            EXPECT_LE(probs.at(row, col), 1.f);
            sum += probs.at(row, col);
        }
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
    fs::remove(path);
}

TEST(ModelLoader, PredictProbaMatchesHandCalculation) {
    const auto path = write_temp_model(valid_identity_model(), "nf_hand.nfmodel");
    auto model = MLPModel::load(path);

    // input [1, 2] → logits [1, 2] after identity + ReLU → softmax
    Tensor inputs{{1.f, 2.f}};
    const Tensor probs = model.predict_proba(inputs);

    const float e1 = std::exp(1.f);
    const float e2 = std::exp(2.f);
    const float denom = e1 + e2;
    EXPECT_NEAR(probs.at(0, 0), e1 / denom, 1e-5);
    EXPECT_NEAR(probs.at(0, 1), e2 / denom, 1e-5);

    const auto classes = model.predict(inputs);
    ASSERT_EQ(classes.size(), 1u);
    EXPECT_EQ(classes[0], 1u);  // class 1 has higher logit
    fs::remove(path);
}

TEST(ModelLoader, PredictArgmaxOnToyClassifier) {
#ifdef NEURALFORGE_SOURCE_DIR
    const fs::path path = fs::path(NEURALFORGE_SOURCE_DIR) / "models" / "toy_classifier.nfmodel";
    ASSERT_TRUE(fs::exists(path));
    auto model = MLPModel::load(path);

    // Center of class-0 cluster should predict class 0 with high confidence.
    Tensor inputs{{-2.f, -1.5f}};
    const auto classes = model.predict(inputs);
    ASSERT_EQ(classes.size(), 1u);
    EXPECT_EQ(classes[0], 0u);

    const Tensor probs = model.predict_proba(inputs);
    EXPECT_GT(probs.at(0, 0), 0.9f);
#else
    GTEST_SKIP() << "NEURALFORGE_SOURCE_DIR not defined";
#endif
}
