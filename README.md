# NeuralForge

**A C++20 CPU inference runtime with a real PyTorch-to-C++ model pipeline.**

NeuralForge demonstrates tensor ownership, matrix multiplication, neural-network primitives, model parameter serialization, inference, and numerical validation against a PyTorch reference. It is an engineering learning project; performance claims should only be made after reproducible benchmarks.

## Current capabilities

- Core tensor abstraction with value semantics and shape overflow checks
- **Cache-blocked matmul** (default) with naive kernel retained for A/B measurement
- Dense, ReLU, and numerically stable row-wise Softmax
- Versioned `NEURALFORGE_MLP 1` model file format
- C++ loader and prediction runner for Dense → ReLU → Dense → Softmax
- PyTorch training/export with stratified train/validation split
- C++/PyTorch numerical parity checker
- GitHub Actions CI (GCC + Clang) with pinned FetchContent dependencies
- Google Benchmark suites for matmul (naive vs blocked) and inference latency

## Quick start

```bash
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Matmul A/B benchmarks

```bash
./build/bin/neuralforge_bench --benchmark_filter='MatMul/(Naive|Blocked)/Square' \
  --benchmark_repetitions=5 --benchmark_min_time=0.2s --benchmark_report_aggregates_only=true
```

Or use the full baseline script:

```bash
chmod +x scripts/run_baselines.sh
./scripts/run_baselines.sh --repetitions 5
```

## Phase 4 — measured matmul optimization

| Size | Naive (CPU mean) | Blocked 64 (CPU mean) | Speedup |
|------|------------------|------------------------|---------|
| 64×64 | ~55 µs | ~34 µs | **~1.6×** |
| 128×128 | ~391 µs | ~270 µs | **~1.4×** |
| 256×256 | ~3.2 ms | ~2.7 ms | **~1.2×** |
| 512×512 | ~24.5 ms | ~24.0 ms | ~1.0× (noise) |

**Machine:** Intel Xeon Platinum 8481C @ 2.70 GHz, 2 cores, L1D 48 KiB, g++ 13.3 `-O3 -DNDEBUG`, 5 repetitions.

- Default `matmul()` uses **cache-blocked** tiling (`block_size=64`).
- `matmul_naive()` is kept for correctness tests and fair comparison.
- Numerical agreement is tested on random square/rectangular shapes within abs/rel tolerance `1e-4`.
- **SIMD and multithreading were not added** — measurements show clear gains from blocking at mid sizes, but 512×512 and the tiny MLP panels (2×16, 16×3) do not yet justify the complexity. Those come next only if profiling shows a bottleneck.

## Continuous integration

Every push and PR to `main` runs GCC 13 + Clang 18 builds, the full test suite (41 cases), and smoke demos.

## Architecture

```
Python / PyTorch training → versioned .nfmodel file → C++ MLPModel loader
    → Tensor / MatMul (blocked) / Dense / ReLU / Softmax → class probabilities
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
| Cache-blocked matmul (measured) | ✅ |
| SIMD / threads (only if justified) | Deferred |
| REST API, Docker | Later |

## License

MIT
