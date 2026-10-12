# NeuralForge

**A C++20 CPU inference runtime with a real PyTorch-to-C++ model pipeline.**

NeuralForge demonstrates tensor ownership, matrix multiplication, neural-network primitives, model parameter serialization, inference, and numerical validation against a PyTorch reference. It is an engineering learning project; performance claims should only be made after reproducible benchmarks.

## Current capabilities

- Core tensor abstraction with value semantics and shape overflow checks
- Cache-blocked matmul (`Ijk`, default `block_size=64`) with naive kernel for A/B checks
- Dense, ReLU, and numerically stable row-wise Softmax
- Versioned `NEURALFORGE_MLP 1` model file format with load-time resource limits
- C++ loader, prediction runner, and **held-out accuracy evaluator**
- PyTorch training/export with stratified train/validation split
- C++/PyTorch numerical parity checker
- **Evaluation suite** — accuracy (PyTorch vs C++), parity, batch latency/throughput
- GitHub Actions CI (GCC + Clang) with pinned FetchContent dependencies

## Quick start

```bash
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # correctness (not performance)
```

### Inference evaluation suite

```bash
python python/eval_suite.py --model models/toy_classifier.nfmodel
# optional: retrain with a fixed seed, then evaluate
python python/eval_suite.py --model models/toy_classifier.nfmodel --retrain --seed 42
```

Reports:

1. **Held-out accuracy** — same weights, same validation set, PyTorch vs `neuralforge_eval_accuracy`
2. **Numerical parity** — max absolute error on probe samples (tolerance `2e-5`)
3. **Latency / throughput** — `Inference/PredictProba` across batch sizes (Google Benchmark)

Standalone C++ accuracy tool:

```bash
./build/bin/neuralforge_eval_accuracy models/toy_classifier.nfmodel path/to/val_set.txt
# val_set.txt: first line `n_samples n_features`, then `x0 … xF-1 label` per row
```

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
| Model-loader tests + reliability hardening | ✅ |
| GitHub Actions CI | ✅ |
| Cache-blocked matmul + block-size tuning | ✅ |
| Inference evaluation suite | ✅ |
| SIMD / threads (only if justified) | Deferred |
| REST API, Docker | Later |

## License

MIT
