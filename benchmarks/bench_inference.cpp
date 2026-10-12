#include <benchmark/benchmark.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "neuralforge/model.hpp"
#include "neuralforge/tensor.hpp"

using namespace neuralforge;

namespace {

// Resolve path to the sample model. Override with NEURALFORGE_MODEL_PATH.
std::filesystem::path model_path() {
    if (const char* env = std::getenv("NEURALFORGE_MODEL_PATH")) {
        return env;
    }
#ifdef NEURALFORGE_SOURCE_DIR
    return std::filesystem::path(NEURALFORGE_SOURCE_DIR) / "models" / "toy_classifier.nfmodel";
#else
    return "models/toy_classifier.nfmodel";
#endif
}

const MLPModel& shared_model() {
    static const MLPModel model = MLPModel::load(model_path());
    return model;
}

Tensor make_batch(std::size_t batch, std::size_t features) {
    // Deterministic pseudo-random-ish values in [-1, 1] so results are stable.
    Tensor t({batch, features});
    for (std::size_t i = 0; i < t.size(); ++i) {
        t.at(i) = static_cast<float>((static_cast<int>(i % 17) - 8) / 8.0);
    }
    return t;
}

} // namespace

// Full forward: Dense → ReLU → Dense → Softmax (predict_proba)
static void BM_Inference_Proba(benchmark::State& state) {
    const auto& model = shared_model();
    const std::size_t batch = static_cast<std::size_t>(state.range(0));
    const Tensor inputs = make_batch(batch, model.input_features());

    for (auto _ : state) {
        auto probs = model.predict_proba(inputs);
        benchmark::DoNotOptimize(probs.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch));
    state.counters["batch"] = static_cast<double>(batch);
    state.counters["samples_per_sec"] = benchmark::Counter(
        static_cast<double>(batch),
        benchmark::Counter::kIsIterationInvariantRate);
}

// Argmax path (predict)
static void BM_Inference_Predict(benchmark::State& state) {
    const auto& model = shared_model();
    const std::size_t batch = static_cast<std::size_t>(state.range(0));
    const Tensor inputs = make_batch(batch, model.input_features());

    for (auto _ : state) {
        auto classes = model.predict(inputs);
        benchmark::DoNotOptimize(classes.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * batch));
    state.counters["batch"] = static_cast<double>(batch);
}

BENCHMARK(BM_Inference_Proba)
    ->Name("Inference/PredictProba")
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Inference_Predict)
    ->Name("Inference/Predict")
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->Arg(256)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
