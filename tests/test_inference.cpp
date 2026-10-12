#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "neuralforge/nn_ops.hpp"

using namespace neuralforge;

TEST(Dense, ComputesBatchForwardPass) {
    // X (2x2) * W (2x3) + b (3)
    Tensor x{{1.f, 2.f}, {3.f, 4.f}};
    Tensor w{{1.f, 0.f, -1.f}, {2.f, 1.f, 0.f}};
    Tensor b = Tensor::zeros({3});

    const Tensor y = dense(x, w, b);
    EXPECT_EQ(y.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_FLOAT_EQ(y.at(0, 0), 5.f);
    EXPECT_FLOAT_EQ(y.at(0, 1), 2.f);
    EXPECT_FLOAT_EQ(y.at(0, 2), -1.f);
    EXPECT_FLOAT_EQ(y.at(1, 0), 11.f);
    EXPECT_FLOAT_EQ(y.at(1, 1), 4.f);
    EXPECT_FLOAT_EQ(y.at(1, 2), -3.f);
}

TEST(Dense, RejectsIncompatibleDimensions) {
    Tensor x = Tensor::ones({2, 3});
    Tensor weights = Tensor::ones({2, 4});
    Tensor bias = Tensor::zeros({4});
    EXPECT_THROW(dense(x, weights, bias), std::invalid_argument);
}

TEST(ReLU, ClipsNegativeValues) {
    Tensor x{{-2.f, 0.f, 3.f}};
    const Tensor y = relu(x);
    EXPECT_FLOAT_EQ(y.at(0, 0), 0.f);
    EXPECT_FLOAT_EQ(y.at(0, 1), 0.f);
    EXPECT_FLOAT_EQ(y.at(0, 2), 3.f);
    EXPECT_FLOAT_EQ(x.at(0, 0), -2.f); // input remains unchanged
}

TEST(Softmax, EachRowSumsToOne) {
    Tensor logits{{1000.f, 1001.f, 999.f}, {0.f, 0.f, 0.f}};
    const Tensor probs = softmax_rows(logits);
    for (std::size_t row = 0; row < probs.shape()[0]; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < probs.shape()[1]; ++col) {
            EXPECT_GE(probs.at(row, col), 0.f);
            EXPECT_LE(probs.at(row, col), 1.f);
            sum += probs.at(row, col);
        }
        EXPECT_NEAR(sum, 1.0, 1e-6);
    }
}

TEST(Softmax, RejectsNonFiniteLogits) {
    Tensor logits{{0.f, std::numeric_limits<float>::infinity()}};
    EXPECT_THROW(softmax_rows(logits), std::invalid_argument);
}
