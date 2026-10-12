#include "neuralforge/model.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    std::size_t n_samples = 0;
    std::size_t n_features = 0;
    if (!(in >> n_samples >> n_features) || n_samples == 0 || n_features == 0) {
        throw std::runtime_error("Invalid dataset header (expected n_samples n_features)");
    }
    Dataset ds;
    ds.n_features = n_features;
    ds.features.resize(n_samples * n_features);
    ds.labels.resize(n_samples);
    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            if (!(in >> ds.features[i * n_features + j])) {
                throw std::runtime_error("Truncated features at sample " + std::to_string(i));
            }
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
        neuralforge::Tensor inputs({n, ds.n_features});
        for (std::size_t i = 0; i < ds.features.size(); ++i) {
            inputs.at(i) = ds.features[i];
        }

        const auto t0 = std::chrono::steady_clock::now();
        const auto predicted = model.predict(inputs);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::size_t correct = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (predicted[i] >= model.output_features()) {
                throw std::runtime_error("Predicted class out of range");
            }
            if (predicted[i] == ds.labels[i]) ++correct;
        }

        const double acc = static_cast<double>(correct) / static_cast<double>(n);
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "samples=" << n << " correct=" << correct
                  << " accuracy=" << acc << " latency_ms=" << ms
                  << " throughput_samples_per_s=" << (n / (ms / 1000.0)) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NeuralForge eval error: " << e.what() << '\n';
        return 1;
    }
}
