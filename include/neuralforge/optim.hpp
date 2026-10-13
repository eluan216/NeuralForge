#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "neuralforge/autograd.hpp"
#include "neuralforge/tensor.hpp"

namespace neuralforge {

class SGD {
public:
    explicit SGD(float learning_rate);
    [[nodiscard]] float learning_rate() const noexcept { return lr_; }
    void step(std::vector<Variable>& parameters);
    void zero_grad(std::vector<Variable>& parameters);

private:
    float lr_;
};

/// Adam with bias-corrected moments. State is keyed by parameter identity
/// and persists across step() calls (do not reconstruct Adam each batch).
class Adam {
public:
    explicit Adam(float learning_rate = 1e-3f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float eps = 1e-8f);

    [[nodiscard]] float learning_rate() const noexcept { return lr_; }
    [[nodiscard]] float beta1() const noexcept { return beta1_; }
    [[nodiscard]] float beta2() const noexcept { return beta2_; }
    [[nodiscard]] float eps() const noexcept { return eps_; }
    [[nodiscard]] std::size_t timestep() const noexcept { return t_; }

    void step(std::vector<Variable>& parameters);
    void zero_grad(std::vector<Variable>& parameters);

private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    std::size_t t_{0};

    struct MomentState {
        Tensor m;
        Tensor v;
    };
    std::unordered_map<const void*, MomentState> state_;
};

} // namespace neuralforge
