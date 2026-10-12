#pragma once

#include "neuralforge/autograd.hpp"

namespace neuralforge {

/// Mean squared error: mean_i (prediction_i - target_i)^2.
/// Returns a scalar Variable compatible with backward().
/// Targets are treated as constants (no gradient through targets).
/// Shapes must match; non-finite values are rejected.
[[nodiscard]] Variable mse_loss(const Variable& prediction, const Variable& target);

} // namespace neuralforge
