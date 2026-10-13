#pragma once

#include <cstddef>
#include <vector>

#include "neuralforge/autograd.hpp"

namespace neuralforge {

/// Mean squared error: mean_i (prediction_i - target_i)^2.
[[nodiscard]] Variable mse_loss(const Variable& prediction, const Variable& target);

/// Multiclass cross-entropy from **logits** (not probabilities).
///
/// logits: shape (batch, num_classes). targets[i] ∈ [0, num_classes).
/// Stable log-sum-exp per row; mean over batch. Scalar for backward().
[[nodiscard]] Variable cross_entropy_loss(const Variable& logits,
                                          const std::vector<std::size_t>& targets);

} // namespace neuralforge
