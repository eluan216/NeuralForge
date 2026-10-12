#include "neuralforge/model.hpp"
#include "neuralforge/nn_ops.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace neuralforge {
namespace {

// Guards against hostile or corrupted model headers that would attempt to
// allocate multi-gigabyte weight tensors before any parameter is read.
constexpr std::size_t kMaxFeatureDim = 65'536;           // per axis
constexpr std::size_t kMaxParameterElements = 16'777'216; // ~64 MiB of float

[[nodiscard]] bool checked_mul_size(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

std::size_t read_dimension(std::istream& in, const char* label) {
    long long value{};
    if (!(in >> value) || value <= 0 ||
        static_cast<unsigned long long>(value) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("Invalid model dimension: ") + label);
    }
    const auto dim = static_cast<std::size_t>(value);
    if (dim > kMaxFeatureDim) {
        throw std::runtime_error(std::string("Model dimension exceeds safety limit: ") + label);
    }
    return dim;
}

void enforce_parameter_budget(std::size_t input, std::size_t hidden, std::size_t output) {
    std::size_t w1 = 0, w2 = 0, total = 0;
    if (!checked_mul_size(input, hidden, w1) || !checked_mul_size(hidden, output, w2)) {
        throw std::runtime_error("Model parameter count overflows size_t");
    }
    // total = w1 + hidden + w2 + output
    if (w1 > kMaxParameterElements || w2 > kMaxParameterElements) {
        throw std::runtime_error("Model weight matrix exceeds safety limit");
    }
    total = w1;
    if (total > kMaxParameterElements - hidden) {
        throw std::runtime_error("Model parameter count exceeds safety limit");
    }
    total += hidden;
    if (total > kMaxParameterElements - w2) {
        throw std::runtime_error("Model parameter count exceeds safety limit");
    }
    total += w2;
    if (total > kMaxParameterElements - output) {
        throw std::runtime_error("Model parameter count exceeds safety limit");
    }
}

void read_values(std::istream& in, Tensor& tensor, const char* label) {
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        float value{};
        if (!(in >> value) || !std::isfinite(value)) {
            throw std::runtime_error(std::string("Missing or non-finite model parameter: ") + label);
        }
        tensor.at(i) = value;
    }
}

void require_finite_inputs(const Tensor& inputs) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (!std::isfinite(inputs.at(i))) {
            throw std::invalid_argument("Model input contains non-finite values");
        }
    }
}

} // namespace

MLPModel MLPModel::load(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open model file: " + path.string());
    std::string magic;
    int version{};
    if (!(in >> magic >> version) || magic != "NEURALFORGE_MLP" || version != 1) {
        throw std::runtime_error("Unsupported or malformed NeuralForge model header");
    }

    MLPModel model;
    model.input_features_ = read_dimension(in, "input");
    model.hidden_features_ = read_dimension(in, "hidden");
    model.output_features_ = read_dimension(in, "output");
    enforce_parameter_budget(model.input_features_, model.hidden_features_, model.output_features_);

    model.w1_ = Tensor({model.input_features_, model.hidden_features_});
    model.b1_ = Tensor({model.hidden_features_});
    model.w2_ = Tensor({model.hidden_features_, model.output_features_});
    model.b2_ = Tensor({model.output_features_});
    read_values(in, model.w1_, "w1");
    read_values(in, model.b1_, "b1");
    read_values(in, model.w2_, "w2");
    read_values(in, model.b2_, "b2");
    std::string trailing;
    if (in >> trailing) throw std::runtime_error("Unexpected trailing data in model file");
    return model;
}

Tensor MLPModel::predict_proba(const Tensor& inputs) const {
    if (inputs.ndim() != 2 || inputs.shape()[1] != input_features_) {
        throw std::invalid_argument("Model input must have shape (batch, " +
                                    std::to_string(input_features_) + ")");
    }
    // batch == 0 is valid: returns an empty (0, C) probability tensor.
    require_finite_inputs(inputs);
    return softmax_rows(dense(relu(dense(inputs, w1_, b1_)), w2_, b2_));
}

std::vector<std::size_t> MLPModel::predict(const Tensor& inputs) const {
    const Tensor probabilities = predict_proba(inputs);
    std::vector<std::size_t> result(probabilities.shape()[0]);
    for (std::size_t row = 0; row < probabilities.shape()[0]; ++row) {
        std::size_t best = 0;
        for (std::size_t col = 1; col < output_features_; ++col) {
            if (probabilities.at(row, col) > probabilities.at(row, best)) best = col;
        }
        result[row] = best;
    }
    return result;
}

} // namespace neuralforge
