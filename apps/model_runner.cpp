#include "neuralforge/model.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: neuralforge_model_runner <model.nfmodel> [x1 x2 ...]\n"
                  << "If no features are supplied, a small two-row smoke-test batch is used.\n";
        return 2;
    }
    try {
        const auto model = neuralforge::MLPModel::load(argv[1]);
        std::vector<float> values;
        for (int i = 2; i < argc; ++i) values.push_back(std::stof(argv[i]));
        std::size_t batch = 1;
        if (values.empty()) {
            if (model.input_features() != 2) throw std::runtime_error("Default smoke test expects a 2-feature model; pass feature values explicitly");
            values = {-1.0f, -0.8f, 1.0f, 0.8f};
            batch = 2;
        } else {
            if (values.size() != model.input_features()) throw std::runtime_error("Pass exactly input_features values for one sample");
        }
        neuralforge::Tensor inputs({batch, model.input_features()});
        for (std::size_t i = 0; i < values.size(); ++i) inputs.at(i) = values[i];
        const auto probabilities = model.predict_proba(inputs);
        const auto classes = model.predict(inputs);
        std::cout << std::fixed << std::setprecision(6);
        for (std::size_t row = 0; row < batch; ++row) {
            std::cout << "sample=" << row << " class=" << classes[row] << " probabilities=[";
            for (std::size_t col = 0; col < model.output_features(); ++col) {
                if (col) std::cout << ", ";
                std::cout << probabilities.at(row, col);
            }
            std::cout << "]\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "NeuralForge model error: " << e.what() << '\n';
        return 1;
    }
}
