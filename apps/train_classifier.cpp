#include "neuralforge/autograd.hpp"
#include "neuralforge/export_model.hpp"
#include "neuralforge/loss.hpp"
#include "neuralforge/model.hpp"
#include "neuralforge/optim.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace neuralforge;

namespace {

void make_clusters(std::size_t n_per_class, Tensor& x_out, std::vector<std::size_t>& y_out,
                   unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.55f);
    const float centers[3][2] = {{-2.0f, -1.5f}, {2.0f, -1.0f}, {0.0f, 2.0f}};
    const std::size_t n = n_per_class * 3;
    x_out = Tensor({n, 2});
    y_out.resize(n);
    std::size_t idx = 0;
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < n_per_class; ++i) {
            x_out.at(idx, 0) = centers[c][0] + noise(rng);
            x_out.at(idx, 1) = centers[c][1] + noise(rng);
            y_out[idx] = c;
            ++idx;
        }
    }
}

float accuracy(const Tensor& logits, const std::vector<std::size_t>& y) {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < logits.shape()[1]; ++j) {
            if (logits.at(i, j) > logits.at(i, best)) best = j;
        }
        if (best == y[i]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(y.size());
}

} // namespace

int main(int argc, char** argv) {
    std::string out_path = "models/cpp_trained_classifier.nfmodel";
    int epochs = 150;
    unsigned seed = 7;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--output" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (a == "--help") {
            std::cout << "Usage: neuralforge_train_classifier [--output path] [--epochs N] [--seed S]\n";
            return 0;
        }
    }

    try {
        Tensor x_train, x_val;
        std::vector<std::size_t> y_train, y_val;
        make_clusters(120, x_train, y_train, seed);
        make_clusters(40, x_val, y_val, seed + 1);

        std::mt19937 rng(seed + 42);
        std::normal_distribution<float> n01(0.0f, 0.1f);
        Tensor w1t({2, 16}), b1t({16}, 0.0f), w2t({16, 3}), b2t({3}, 0.0f);
        for (std::size_t i = 0; i < w1t.size(); ++i) w1t.at(i) = n01(rng);
        for (std::size_t i = 0; i < w2t.size(); ++i) w2t.at(i) = n01(rng);

        Variable W1(w1t, true), b1(b1t, true), W2(w2t, true), b2(b2t, true);
        std::vector<Variable> params{W1, b1, W2, b2};
        Adam opt(0.05f);

        float train_loss = 0.0f;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            Variable x(x_train, false);
            auto h = relu(add(matmul(x, W1), b1));
            auto logits = add(matmul(h, W2), b2);
            auto loss = cross_entropy_loss(logits, y_train);
            opt.zero_grad(params);
            loss.backward();
            opt.step(params);
            train_loss = loss.data().at(0);
        }

        Variable xv(x_val, false);
        auto h_val = relu(add(matmul(xv, W1), b1));
        auto logits_val = add(matmul(h_val, W2), b2);
        const float val_acc = accuracy(logits_val.data(), y_val);

        export_mlp_model(out_path, W1.data(), b1.data(), W2.data(), b2.data());

        // Verify reload
        auto model = MLPModel::load(out_path);
        auto pred = model.predict(x_val);
        std::size_t correct = 0;
        for (std::size_t i = 0; i < pred.size(); ++i)
            if (pred[i] == y_val[i]) ++correct;
        const float loaded_acc =
            static_cast<float>(correct) / static_cast<float>(pred.size());

        std::cout << "train_samples=" << y_train.size()
                  << " val_samples=" << y_val.size()
                  << " epochs=" << epochs << '\n';
        std::cout << "final_train_loss=" << train_loss << '\n';
        std::cout << "val_accuracy=" << val_acc << '\n';
        std::cout << "loaded_model_val_accuracy=" << loaded_acc << '\n';
        std::cout << "exported_model=" << out_path << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "train_classifier error: " << e.what() << '\n';
        return 1;
    }
}
