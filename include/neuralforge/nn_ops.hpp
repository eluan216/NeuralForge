#pragma once

#include "neuralforge/tensor.hpp"

namespace neuralforge {

// Dense layer for row-major batches:
// input: (batch, in_features), weights: (in_features, out_features),
// bias: (out_features,), output: (batch, out_features).
[[nodiscard]] Tensor dense(const Tensor& input,
                           const Tensor& weights,
                           const Tensor& bias);

// Element-wise rectified linear unit: max(0, x).
[[nodiscard]] Tensor relu(const Tensor& input);

// Numerically stable softmax applied independently to each row of a rank-2 tensor.
[[nodiscard]] Tensor softmax_rows(const Tensor& logits);

} // namespace neuralforge
