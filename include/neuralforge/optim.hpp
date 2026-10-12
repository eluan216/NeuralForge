#pragma once

#include <vector>

#include "neuralforge/autograd.hpp"

namespace neuralforge {

/// Stochastic gradient descent: θ ← θ − lr · ∇θ L
class SGD {
public:
    explicit SGD(float learning_rate);

    [[nodiscard]] float learning_rate() const noexcept { return lr_; }

    void step(std::vector<Variable>& parameters);
    void zero_grad(std::vector<Variable>& parameters);

private:
    float lr_;
};

} // namespace neuralforge
