#include <gtest/gtest.h>

#include "neuralforge/tensor.hpp"

using namespace neuralforge;

// ---------------------------------------------------------------------------
// Construction & shape
// ---------------------------------------------------------------------------

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

TEST(Tensor, OutOfRangeThrows) {
    auto t = Tensor::zeros({2, 2});
    EXPECT_THROW(t.at(4), std::out_of_range);
    EXPECT_THROW(t.at(2, 0), std::out_of_range);
    EXPECT_THROW(t.at(0, 2), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Matrix multiplication correctness
// ---------------------------------------------------------------------------

TEST(MatMul, Identity) {
    Tensor I{{1.f, 0.f},
             {0.f, 1.f}};
    Tensor A{{2.f, 3.f},
             {4.f, 5.f}};

    auto C = matmul(A, I);
    EXPECT_FLOAT_EQ(C.at(0, 0), 2.f);
    EXPECT_FLOAT_EQ(C.at(0, 1), 3.f);
    EXPECT_FLOAT_EQ(C.at(1, 0), 4.f);
    EXPECT_FLOAT_EQ(C.at(1, 1), 5.f);
}

TEST(MatMul, Simple2x3_3x2) {
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[7,  8],
    //      [9, 10],
    //      [11,12]]
    // C = [[58, 64],
    //      [139,154]]
    Tensor A{{1.f, 2.f, 3.f},
             {4.f, 5.f, 6.f}};
    Tensor B{{7.f, 8.f},
             {9.f, 10.f},
             {11.f, 12.f}};

    auto C = matmul(A, B);
    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_FLOAT_EQ(C.at(0, 0), 58.f);
    EXPECT_FLOAT_EQ(C.at(0, 1), 64.f);
    EXPECT_FLOAT_EQ(C.at(1, 0), 139.f);
    EXPECT_FLOAT_EQ(C.at(1, 1), 154.f);
}

TEST(MatMul, ShapeMismatchThrows) {
    Tensor A{{1.f, 2.f}};
    Tensor B{{1.f, 2.f, 3.f}};
    EXPECT_THROW(matmul(A, B), std::invalid_argument);
}

TEST(MatMul, RankMismatchThrows) {
    Tensor A = Tensor::zeros({2, 2, 2});
    Tensor B = Tensor::zeros({2, 2});
    EXPECT_THROW(matmul(A, B), std::invalid_argument);
}
