#include <benchmark/benchmark.h>

#include <cstddef>

#include "neuralforge/tensor.hpp"

using namespace neuralforge;

// ---------------------------------------------------------------------------
// Square matmul: C = A @ B with A, B ∈ R^{N×N}
// Baseline triple-loop kernel — measure before any optimization.
// ---------------------------------------------------------------------------
static void BM_MatMul_Square(benchmark::State& state) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    auto A = Tensor::ones({N, N});
    auto B = Tensor::ones({N, N});

    for (auto _ : state) {
        auto C = matmul(A, B);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.counters["N"] = static_cast<double>(N);
    state.counters["FLOPs"] = benchmark::Counter(
        static_cast<double>(2ull * N * N * N),
        benchmark::Counter::kIsIterationInvariantRate);
    state.SetComplexityN(static_cast<int64_t>(N));
}

// ---------------------------------------------------------------------------
// Rectangular: A (M×K) @ B (K×N)
// Ranges: range(0)=M, range(1)=K, range(2)=N
// ---------------------------------------------------------------------------
static void BM_MatMul_Rectangular(benchmark::State& state) {
    const std::size_t M = static_cast<std::size_t>(state.range(0));
    const std::size_t K = static_cast<std::size_t>(state.range(1));
    const std::size_t N = static_cast<std::size_t>(state.range(2));
    auto A = Tensor::ones({M, K});
    auto B = Tensor::ones({K, N});

    for (auto _ : state) {
        auto C = matmul(A, B);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.counters["M"] = static_cast<double>(M);
    state.counters["K"] = static_cast<double>(K);
    state.counters["N"] = static_cast<double>(N);
    state.counters["FLOPs"] = benchmark::Counter(
        static_cast<double>(2ull * M * K * N),
        benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_MatMul_Square)
    ->Name("MatMul/Square")
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity(benchmark::oNCubed);

// Typical MLP-shaped products: (batch × in) @ (in × out)
BENCHMARK(BM_MatMul_Rectangular)
    ->Name("MatMul/Rect")
    ->Args({1, 16, 3})       // single-sample toy classifier dense
    ->Args({8, 16, 3})
    ->Args({32, 16, 3})
    ->Args({128, 16, 3})
    ->Args({1, 256, 256})
    ->Args({32, 256, 256})
    ->Args({64, 512, 512})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
