#include <iostream>

#include "neuralforge/nn_ops.hpp"

int main() {
    using namespace neuralforge;

    // Tiny deterministic example: batch of two samples, two input features,
    // three output classes. These are demonstration parameters, not trained weights.
    Tensor inputs{{1.f, 2.f}, {3.f, 4.f}};
    Tensor weights{{0.2f, -0.1f, 0.4f}, {0.3f, 0.5f, -0.2f}};
    Tensor bias = Tensor::zeros({3});

    const Tensor logits = dense(inputs, weights, bias);
    const Tensor probabilities = softmax_rows(logits);

    std::cout << "NeuralForge CPU inference demo\n";
    std::cout << "Logits: " << logits.to_string() << '\n';
    std::cout << "Probabilities: " << probabilities.to_string() << '\n';
    std::cout << "Note: demonstration weights are deterministic, not trained.\n";
    return 0;
}
