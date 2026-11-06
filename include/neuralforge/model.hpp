#pragma once

#include <filesystem>
#include <iosfwd>
#include <array>
#include <span>
#include <cstdint>
#include <vector>
#include "neuralforge/tensor.hpp"

namespace neuralforge {

class InferenceWorkspace {
 friend class MLPModel;
 std::size_t input_{},hidden_{},output_{};
 std::vector<float> scratch_;
public:
 [[nodiscard]] std::size_t scratch_bytes() const noexcept {return scratch_.size()*sizeof(float);}
};
// Versioned, dependency-free model format for a two-layer MLP.
// Architecture: Dense -> ReLU -> Dense -> Softmax.
class MLPModel {
public:
    static MLPModel from_parameters(const std::array<Tensor,4>& parameters);
    [[nodiscard]] std::array<Tensor,4> parameters() const {return {w1_,b1_,w2_,b2_};}
    [[nodiscard]] InferenceWorkspace prepare() const;
    /// Success path performs no C++ heap allocations. Buffers must not overlap.
    /// Share immutable models across threads; use one workspace per concurrent call.
    void predict_proba_into(std::span<const float> inputs,std::size_t batch,
                            std::span<float> outputs,InferenceWorkspace& workspace) const;
    /// Stable content identifier, not a cryptographic signature.
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    static MLPModel load(const std::filesystem::path& path);
    static MLPModel read(std::istream& stream);
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
