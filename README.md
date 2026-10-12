# NeuralForge

**A C++20 CPU inference runtime with a real PyTorch-to-C++ model pipeline.**

NeuralForge demonstrates tensor ownership, matrix multiplication, neural-network primitives, model parameter serialization, inference, and numerical validation against a PyTorch reference. It is an engineering learning project; performance claims should only be made after reproducible benchmarks.

## Current capabilities

- Core tensor abstraction with value semantics and shape overflow checks
- **Cache-blocked matmul** (`Ijk` tile order, default `block_size=64`) with naive kernel for A/B checks
- Dense, ReLU, and numerically stable row-wise Softmax
- Versioned `NEURALFORGE_MLP 1` model file format
- C++ loader and prediction runner for Dense → ReLU → Dense → Softmax
- PyTorch training/export with stratified train/validation split
- C++/PyTorch numerical parity checker
- GitHub Actions CI (GCC + Clang) with pinned FetchContent dependencies
- Google Benchmark suites (naive vs blocked, tile order, **block-size sweep**)

## Quick start

```bash
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # correctness
```

Performance is a **separate** step from `ctest`:

```bash
./build/bin/neuralforge_bench \
  --benchmark_filter='MatMul/Blocked/Ijk/(Square|Rect)' \
  --benchmark_repetitions=5 --benchmark_min_time=0.15s \
  --benchmark_report_aggregates_only=true
```

## Phase 5 — block-size tuning (summary)

Sweep of tile sizes **16 / 32 / 64 / 128** with `MatMulTileOrder::Ijk` on square and rectangular shapes (5 repetitions, Release, g++ 13.3).

| Finding | Detail |
|---------|--------|
| Best general tile | **64** on all square sizes and most rectangular shapes |
| Worst | 16 (consistent) |
| 128 | Only competitive when the problem already fits one tile |
| Default change? | **No** — keep `block_size=64` and `Ijk` |

Full table and methodology: [`benchmarks/results/phase5_block_size.md`](benchmarks/results/phase5_block_size.md).

Numbers are from one measurement host; reproduce before treating them as authoritative.

## Continuous integration

Every push and PR to `main` runs GCC 13 + Clang 18 builds, the unit test suite, and smoke demos. CI validates **correctness**, not benchmark rankings.

## Architecture

```
Python / PyTorch training → versioned .nfmodel file → C++ MLPModel loader
    → Tensor / MatMul (blocked Ijk, tile 64) / Dense / ReLU / Softmax → class probabilities
```

## Roadmap

| Phase | Status |
|-------|--------|
| Tensor + MatMul + tests | ✅ |
| Dense / ReLU / Softmax | ✅ |
| PyTorch train → export → C++ inference + parity | ✅ |
| Model-loader tests + train/val evaluation | ✅ |
| GitHub Actions CI | ✅ |
| Baseline benchmarks | ✅ |
| Cache-blocked matmul + tile-order A/B | ✅ |
| Block-size tuning (default 64 retained) | ✅ |
| SIMD / threads (only if justified) | Deferred |
| REST API, Docker | Later |

## License

MIT
