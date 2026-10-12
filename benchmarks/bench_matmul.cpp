#include <benchmark/benchmark.h>

#include <cstddef>

#include "neuralforge/tensor.hpp"

using namespace neuralforge;

static void BM_MatMul_Naive_Square(benchmark::State& state) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    auto A = Tensor::ones({N, N});
    auto B = Tensor::ones({N, N});

    for (auto _ : state) {
        auto C = matmul_naive(A, B);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.counters["N"] = static_cast<double>(N);
    state.counters["FLOPs"] = benchmark::Counter(
        static_cast<double>(2ull * N * N * N),
        benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_MatMul_Blocked_Order(benchmark::State& state) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const auto order = state.range(1) == 0 ? MatMulTileOrder::Ikj : MatMulTileOrder::Ijk;
    auto A = Tensor::ones({N, N});
    auto B = Tensor::ones({N, N});

    for (auto _ : state) {
        auto C = matmul_blocked(A, B, 64, order);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.counters["N"] = static_cast<double>(N);
    state.counters["FLOPs"] = benchmark::Counter(
        static_cast<double>(2ull * N * N * N),
        benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_MatMul_Default_Square(benchmark::State& state) {
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
}

static void BM_MatMul_Rect_Order(benchmark::State& state) {
    const std::size_t M = static_cast<std::size_t>(state.range(0));
    const std::size_t K = static_cast<std::size_t>(state.range(1));
    const std::size_t N = static_cast<std::size_t>(state.range(2));
    const auto order = state.range(3) == 0 ? MatMulTileOrder::Ikj : MatMulTileOrder::Ijk;
    auto A = Tensor::ones({M, K});
    auto B = Tensor::ones({K, N});

    for (auto _ : state) {
        auto C = matmul_blocked(A, B, 64, order);
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

BENCHMARK(BM_MatMul_Naive_Square)
    ->Name("MatMul/Naive/Square")
    ->Arg(64)->Arg(128)->Arg(256)->Arg(512)
    ->Unit(benchmark::kMicrosecond);

// range(0)=N, range(1)=0 Ikj / 1 Ijk
BENCHMARK(BM_MatMul_Blocked_Order)
    ->Name("MatMul/Blocked/Ikj/Square")
    ->Args({64, 0})->Args({128, 0})->Args({256, 0})->Args({512, 0})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatMul_Blocked_Order)
    ->Name("MatMul/Blocked/Ijk/Square")
    ->Args({64, 1})->Args({128, 1})->Args({256, 1})->Args({512, 1})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatMul_Default_Square)
    ->Name("MatMul/Default/Square")
    ->Arg(64)->Arg(128)->Arg(256)->Arg(512)
    ->Unit(benchmark::kMicrosecond);

// Args: M, K, N, order(0=Ikj, 1=Ijk)
BENCHMARK(BM_MatMul_Rect_Order)
    ->Name("MatMul/Blocked/Ikj/Rect")
    ->Args({32, 256, 256, 0})
    ->Args({64, 512, 512, 0})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatMul_Rect_Order)
    ->Name("MatMul/Blocked/Ijk/Rect")
    ->Args({32, 256, 256, 1})
    ->Args({64, 512, 512, 1})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
