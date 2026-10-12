#include "neuralforge/nn_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace neuralforge {

Tensor dense(const Tensor& input, const Tensor& weights, const Tensor& bias) {
    if (input.ndim() != 2 || weights.ndim() != 2 || bias.ndim() != 1) {
        throw std::invalid_argument("dense expects input/weights rank 2 and bias rank 1");
    }
    if (input.shape()[1] != weights.shape()[0]) {
        throw std::invalid_argument("dense input feature count does not match weights");
    }
    if (weights.shape()[1] != bias.shape()[0]) {
        throw std::invalid_argument("dense output feature count does not match bias");
    }

    Tensor output = matmul(input, weights);
    const auto batch = output.shape()[0];
    const auto features = output.shape()[1];
    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t col = 0; col < features; ++col) {
            output.at(row, col) += bias.at(col);
        }
    }
    return output;
}

Tensor relu(const Tensor& input) {
    Tensor output(input.shape());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output.at(i) = std::max(0.0f, input.at(i));
    }
    return output;
}

Tensor softmax_rows(const Tensor& logits) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument("softmax_rows expects a rank-2 tensor");
    }
    const auto rows = logits.shape()[0];
    const auto cols = logits.shape()[1];
    if (cols == 0) {
        throw std::invalid_argument("softmax_rows requires at least one column");
    }

    Tensor output(logits.shape(), 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t col = 0; col < cols; ++col) {
            const float value = logits.at(row, col);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("softmax_rows requires finite logits");
            }
            max_value = std::max(max_value, value);
        }

        double denominator = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const double exponent = std::exp(
                static_cast<double>(logits.at(row, col) - max_value));
            output.at(row, col) = static_cast<float>(exponent);
            denominator += exponent;
        }
        for (std::size_t col = 0; col < cols; ++col) {
            output.at(row, col) = static_cast<float>(
                static_cast<double>(output.at(row, col)) / denominator);
        }
    }
    return output;
}

} // namespace neuralforge
