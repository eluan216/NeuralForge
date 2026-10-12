#pragma once

#include <filesystem>
#include <vector>
#include "neuralforge/tensor.hpp"

namespace neuralforge {

// Versioned, dependency-free model format for a two-layer MLP.
// Architecture: Dense -> ReLU -> Dense -> Softmax.
class MLPModel {
public:
    static MLPModel load(const std::filesystem::path& path);
    [[nodiscard]] Tensor predict_proba(const Tensor& inputs) const;
    [[nodiscard]] std::vector<std::size_t> predict(const Tensor& inputs) const;
    [[nodiscard]] std::size_t input_features() const noexcept { return input_features_; }
    [[nodiscard]] std::size_t hidden_features() const noexcept { return hidden_features_; }
    [[nodiscard]] std::size_t output_features() const noexcept { return output_features_; }

private:
    std::size_t input_features_{};
    std::size_t hidden_features_{};
    std::size_t output_features_{};
    Tensor w1_, b1_, w2_, b2_;
};

} // namespace neuralforge
