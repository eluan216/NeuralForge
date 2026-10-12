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

// range(0)=N, range(1)=block_size, range(2)=0 Ikj / 1 Ijk
static void BM_MatMul_Blocked_Square(benchmark::State& state) {
    const std::size_t N = static_cast<std::size_t>(state.range(0));
    const std::size_t bs = static_cast<std::size_t>(state.range(1));
    const auto order = state.range(2) == 0 ? MatMulTileOrder::Ikj : MatMulTileOrder::Ijk;
    auto A = Tensor::ones({N, N});
    auto B = Tensor::ones({N, N});

    for (auto _ : state) {
        auto C = matmul_blocked(A, B, bs, order);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.counters["N"] = static_cast<double>(N);
    state.counters["block"] = static_cast<double>(bs);
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

// range(0)=M, range(1)=K, range(2)=N, range(3)=block_size, range(4)=0 Ikj / 1 Ijk
static void BM_MatMul_Blocked_Rect(benchmark::State& state) {
    const std::size_t M = static_cast<std::size_t>(state.range(0));
    const std::size_t K = static_cast<std::size_t>(state.range(1));
    const std::size_t N = static_cast<std::size_t>(state.range(2));
    const std::size_t bs = static_cast<std::size_t>(state.range(3));
    const auto order = state.range(4) == 0 ? MatMulTileOrder::Ikj : MatMulTileOrder::Ijk;
    auto A = Tensor::ones({M, K});
    auto B = Tensor::ones({K, N});

    for (auto _ : state) {
        auto C = matmul_blocked(A, B, bs, order);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
    state.counters["M"] = static_cast<double>(M);
    state.counters["K"] = static_cast<double>(K);
    state.counters["N"] = static_cast<double>(N);
    state.counters["block"] = static_cast<double>(bs);
    state.counters["FLOPs"] = benchmark::Counter(
        static_cast<double>(2ull * M * K * N),
        benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_MatMul_Naive_Square)
    ->Name("MatMul/Naive/Square")
    ->Arg(64)->Arg(128)->Arg(256)->Arg(512)
    ->Unit(benchmark::kMicrosecond);

// Phase 5: block-size sweep for Ijk (primary candidate order)
BENCHMARK(BM_MatMul_Blocked_Square)
    ->Name("MatMul/Blocked/Ijk/Square")
    ->Args({64, 16, 1})->Args({64, 32, 1})->Args({64, 64, 1})->Args({64, 128, 1})
    ->Args({128, 16, 1})->Args({128, 32, 1})->Args({128, 64, 1})->Args({128, 128, 1})
    ->Args({256, 16, 1})->Args({256, 32, 1})->Args({256, 64, 1})->Args({256, 128, 1})
    ->Args({512, 16, 1})->Args({512, 32, 1})->Args({512, 64, 1})->Args({512, 128, 1})
    ->Unit(benchmark::kMicrosecond);

// Cross-check: same sizes with Ikj at block 32 and 64
BENCHMARK(BM_MatMul_Blocked_Square)
    ->Name("MatMul/Blocked/Ikj/Square")
    ->Args({128, 32, 0})->Args({128, 64, 0})
    ->Args({256, 32, 0})->Args({256, 64, 0})
    ->Args({512, 32, 0})->Args({512, 64, 0})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatMul_Default_Square)
    ->Name("MatMul/Default/Square")
    ->Arg(64)->Arg(128)->Arg(256)->Arg(512)
    ->Unit(benchmark::kMicrosecond);

// Rectangular: (batch×in)@ (in×out)-like and larger panels
BENCHMARK(BM_MatMul_Blocked_Rect)
    ->Name("MatMul/Blocked/Ijk/Rect")
    ->Args({32, 256, 256, 16, 1})->Args({32, 256, 256, 32, 1})
    ->Args({32, 256, 256, 64, 1})->Args({32, 256, 256, 128, 1})
    ->Args({64, 512, 512, 16, 1})->Args({64, 512, 512, 32, 1})
    ->Args({64, 512, 512, 64, 1})->Args({64, 512, 512, 128, 1})
    ->Args({128, 256, 64, 16, 1})->Args({128, 256, 64, 32, 1})
    ->Args({128, 256, 64, 64, 1})->Args({128, 256, 64, 128, 1})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatMul_Blocked_Rect)
    ->Name("MatMul/Blocked/Ikj/Rect")
    ->Args({32, 256, 256, 32, 0})->Args({32, 256, 256, 64, 0})
    ->Args({64, 512, 512, 32, 0})->Args({64, 512, 512, 64, 0})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
