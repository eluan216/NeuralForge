#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include "neuralforge/tensor.hpp"

using namespace neuralforge;

TEST(Tensor, DefaultConstructedIsEmpty) {
    Tensor t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.ndim(), 0u);
}

TEST(Tensor, ZerosCreatesCorrectShapeAndValues) {
    auto t = Tensor::zeros({2, 3});
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(t.size(), 6u);
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t.at(i), 0.0f);
    }
}

TEST(Tensor, OnesWorks) {
    auto t = Tensor::ones({4});
    EXPECT_EQ(t.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(t.at(i), 1.0f);
    }
}

TEST(Tensor, InitializerList2D) {
    Tensor t{{1.f, 2.f, 3.f},
             {4.f, 5.f, 6.f}};
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(t.at(0, 2), 3.f);
    EXPECT_FLOAT_EQ(t.at(1, 1), 5.f);
}

TEST(Tensor, CopyHasIndependentStorage) {
    Tensor original{{1.f, 2.f}, {3.f, 4.f}};
    Tensor copied = original;
    copied.at(0, 0) = 99.f;
    EXPECT_FLOAT_EQ(original.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(copied.at(0, 0), 99.f);
}

TEST(Tensor, CopyAssignmentHasIndependentStorage) {
    Tensor original{{1.f, 2.f}};
    Tensor copied;
    copied = original;
    copied.at(0, 1) = 42.f;
    EXPECT_FLOAT_EQ(original.at(0, 1), 2.f);
}

TEST(Tensor, ZeroDimensionProducesEmptyTensor) {
    Tensor t({2, 0, 4});
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
}

TEST(Tensor, ShapeOverflowThrows) {
    using S = std::size_t;
    EXPECT_THROW(Tensor({std::numeric_limits<S>::max(), 2}), std::overflow_error);
}

TEST(Tensor, OutOfRangeThrows) {
    auto t = Tensor::zeros({2, 2});
    EXPECT_THROW(t.at(4), std::out_of_range);
    EXPECT_THROW(t.at(2, 0), std::out_of_range);
    EXPECT_THROW(t.at(0, 2), std::out_of_range);
}

TEST(MatMul, Identity) {
    Tensor identity{{1.f, 0.f}, {0.f, 1.f}};
    Tensor a{{2.f, 3.f}, {4.f, 5.f}};
    auto c = matmul(a, identity);
    EXPECT_FLOAT_EQ(c.at(0, 0), 2.f);
    EXPECT_FLOAT_EQ(c.at(0, 1), 3.f);
    EXPECT_FLOAT_EQ(c.at(1, 0), 4.f);
    EXPECT_FLOAT_EQ(c.at(1, 1), 5.f);
}

TEST(MatMul, Simple2x3By3x2) {
    Tensor a{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    Tensor b{{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}};
    auto c = matmul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(c.at(0, 0), 58.f);
    EXPECT_FLOAT_EQ(c.at(0, 1), 64.f);
    EXPECT_FLOAT_EQ(c.at(1, 0), 139.f);
    EXPECT_FLOAT_EQ(c.at(1, 1), 154.f);
}

TEST(MatMul, ShapeMismatchThrows) {
    Tensor a{{1.f, 2.f}};
    Tensor b{{1.f, 2.f, 3.f}};
    EXPECT_THROW(matmul(a, b), std::invalid_argument);
}

TEST(MatMul, RankMismatchThrows) {
    Tensor a = Tensor::zeros({2, 2, 2});
    Tensor b = Tensor::zeros({2, 2});
    EXPECT_THROW(matmul(a, b), std::invalid_argument);
}
