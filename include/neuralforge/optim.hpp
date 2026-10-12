#pragma once

#include <vector>

#include "neuralforge/autograd.hpp"

namespace neuralforge {

/// Stochastic gradient descent: θ ← θ − lr · ∇θ L
///
/// Only updates Variables with requires_grad() and a populated grad buffer.
/// Rejects non-positive learning rates and non-finite gradients.
class SGD {
public:
    explicit SGD(float learning_rate);

    [[nodiscard]] float learning_rate() const noexcept { return lr_; }

    /// Apply one update step to each parameter in-place.
    void step(const std::vector<Variable>& parameters);

    /// Zero gradients of all listed parameters.
    void zero_grad(const std::vector<Variable>& parameters);

private:
    float lr_;
};

} // namespace neuralforge
