#include "neuralforge/model.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Dataset text format:
//   n_samples n_features
//   x0 x1 ... x_{F-1} label
//   ...
// Labels are integer class indices in [0, output_features).

namespace {

constexpr std::size_t kMaxSamples = 1'000'000;
constexpr std::size_t kMaxFeatures = 65'536;

[[nodiscard]] bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

struct Dataset {
    std::size_t n_features{};
    std::vector<float> features;  // row-major [n_samples * n_features]
    std::vector<std::size_t> labels;
};

Dataset load_dataset(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open dataset: " + path.string());
    }

    unsigned long long n_samples_ull = 0;
    unsigned long long n_features_ull = 0;
    if (!(in >> n_samples_ull >> n_features_ull) || n_samples_ull == 0 || n_features_ull == 0) {
        throw std::runtime_error("Invalid dataset header (expected n_samples n_features)");
    }
    if (n_samples_ull > kMaxSamples) {
        throw std::runtime_error("Dataset sample count exceeds safety limit");
    }
    if (n_features_ull > kMaxFeatures) {
        throw std::runtime_error("Dataset feature count exceeds safety limit");
    }

    const std::size_t n_samples = static_cast<std::size_t>(n_samples_ull);
    const std::size_t n_features = static_cast<std::size_t>(n_features_ull);

    std::size_t element_count = 0;
    if (!checked_mul(n_samples, n_features, element_count)) {
        throw std::runtime_error("Dataset n_samples * n_features overflows size_t");
    }

    Dataset ds;
    ds.n_features = n_features;
    try {
        ds.features.resize(element_count);
        ds.labels.resize(n_samples);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("Dataset allocation failed (too large)");
    }

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            float value = 0.0f;
            if (!(in >> value)) {
                throw std::runtime_error("Truncated features at sample " + std::to_string(i));
            }
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite feature at sample " + std::to_string(i));
            }
            ds.features[i * n_features + j] = value;
        }
        long long label = -1;
        if (!(in >> label) || label < 0) {
            throw std::runtime_error("Invalid label at sample " + std::to_string(i));
        }
        ds.labels[i] = static_cast<std::size_t>(label);
    }
    return ds;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: neuralforge_eval_accuracy <model.nfmodel> <dataset.txt>\n"
                  << "Dataset: first line 'n_samples n_features', then feature rows + integer label.\n";
        return 2;
    }
    try {
        const auto model = neuralforge::MLPModel::load(argv[1]);
        const Dataset ds = load_dataset(argv[2]);
        if (ds.n_features != model.input_features()) {
            throw std::runtime_error("Dataset feature count does not match model");
        }

        const std::size_t n = ds.labels.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (ds.labels[i] >= model.output_features()) {
                throw std::runtime_error(
                    "Label out of range at sample " + std::to_string(i) +
                    " (label=" + std::to_string(ds.labels[i]) +
                    ", classes=" + std::to_string(model.output_features()) + ")");
            }
        }

        neuralforge::Tensor inputs({n, ds.n_features});
        for (std::size_t i = 0; i < ds.features.size(); ++i) {
            inputs.at(i) = ds.features[i];
        }

        const auto t0 = std::chrono::steady_clock::now();
        const auto predicted = model.predict(inputs);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (predicted.size() != n) {
            throw std::runtime_error("Predict returned unexpected batch size");
        }

        std::size_t correct = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (predicted[i] >= model.output_features()) {
                throw std::runtime_error("Predicted class out of range");
            }
            if (predicted[i] == ds.labels[i]) ++correct;
        }

        const double acc = static_cast<double>(correct) / static_cast<double>(n);
        const double throughput = (ms > 0.0) ? (n / (ms / 1000.0)) : 0.0;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "samples=" << n << " correct=" << correct
                  << " accuracy=" << acc << " latency_ms=" << ms
                  << " throughput_samples_per_s=" << throughput << '\n';

        std::cout << "predictions=";
        for (std::size_t i = 0; i < n; ++i) {
            if (i) std::cout << ',';
            std::cout << predicted[i];
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NeuralForge eval error: " << e.what() << '\n';
        return 1;
    }
}
