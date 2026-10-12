#include <benchmark/benchmark.h>

#include "neuralforge/tensor.hpp"

using namespace neuralforge;

static void BM_MatMul(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));

    auto A = Tensor::ones({static_cast<std::size_t>(N),
                           static_cast<std::size_t>(N)});
    auto B = Tensor::ones({static_cast<std::size_t>(N),
                           static_cast<std::size_t>(N)});

    for (auto _ : state) {
        auto C = matmul(A, B);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    // Report items processed (matrix multiplications)
    state.SetItemsProcessed(state.iterations());
    // Rough FLOPs: 2 * N^3
    state.SetComplexityN(N);
}

BENCHMARK(BM_MatMul)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Complexity(benchmark::oNCubed);

BENCHMARK_MAIN();
