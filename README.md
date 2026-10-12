# NeuralForge

**A C++20 CPU inference runtime with a real PyTorch-to-C++ model pipeline.**

NeuralForge demonstrates tensor ownership, matrix multiplication, neural-network primitives, model parameter serialization, inference, and numerical validation against a PyTorch reference. It is an engineering learning project; performance claims should only be made after reproducible benchmarks.

## Current capabilities

- Core tensor abstraction with value semantics and shape overflow checks
- Baseline row-major matrix multiplication
- Dense, ReLU, and numerically stable row-wise Softmax
- Versioned `NEURALFORGE_MLP 1` model file format
- C++ loader and prediction runner for Dense → ReLU → Dense → Softmax
- PyTorch training/export with stratified train/validation split and dual accuracy metrics
- C++/PyTorch numerical parity checker
- GitHub Actions CI (GCC + Clang) with pinned FetchContent dependencies
- **Baseline performance benchmarks** for matmul and inference (sizes × batch × repetitions + env metadata)

## Quick start

Build the C++ engine:

```bash
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use the included sample model immediately:

```bash
./build/bin/neuralforge_model_runner models/toy_classifier.nfmodel
./build/bin/neuralforge_model_runner models/toy_classifier.nfmodel -2.0 -1.5
```

### Baseline benchmarks (before optimizations)

```bash
# Build with benchmarks (default ON)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEURALFORGE_BUILD_BENCHMARKS=ON
cmake --build build -j

# One-shot: records compiler, hardware, build flags, and repeated-run stats
chmod +x scripts/run_baselines.sh
./scripts/run_baselines.sh                  # default: 5 repetitions, 0.2s min time
./scripts/run_baselines.sh --repetitions 10
```

Results land in `benchmarks/results/baseline_<UTC-timestamp>/`:

| File | Contents |
|------|----------|
| `env.txt` | CPU model, cores, memory, compiler version, CMake cache flags, repetition settings |
| `matmul.json` / `matmul.txt` | Square (16…512) and rectangular matmul latency + FLOP rate |
| `inference.json` / `inference.txt` | `predict_proba` / `predict` latency for batch sizes 1…512 |

Google Benchmark reports mean / median / stddev across repetitions. These numbers are the **unoptimized baseline** against which future kernel changes must be compared.

Train a fresh model (requires Python 3.10+ and `python -m pip install torch`):

```bash
python python/train_export.py --output models/toy_classifier.nfmodel
python python/validate_parity.py models/toy_classifier.nfmodel build/bin/neuralforge_model_runner
```

## Continuous integration

Every push and pull request to `main` runs `.github/workflows/ci.yml`:

- **Matrix:** GCC 13 and Clang 18 on Ubuntu 24.04
- **Steps:** configure (Ninja) → build → `ctest` → model-runner and demo smoke tests
- **Dependencies:** GoogleTest and Google Benchmark at **pinned commit SHAs**

## Architecture

```
Python / PyTorch training → versioned .nfmodel file → C++ MLPModel loader
    → Tensor / MatMul / Dense / ReLU / Softmax → class probabilities
```

## Model file format v1

Whitespace-separated text:

1. `NEURALFORGE_MLP 1`
2. `input_features hidden_features output_features`
3. `W1` flattened row-major `(input, hidden)`
4. `b1`
5. `W2` flattened row-major `(hidden, output)`
6. `b2`

## Project structure

```
NeuralForge/
├── .github/workflows/ci.yml
├── scripts/run_baselines.sh
├── benchmarks/
│   ├── bench_matmul.cpp
│   ├── bench_inference.cpp
│   └── results/
├── include/neuralforge/
├── src/
├── apps/
├── python/
├── models/
└── tests/
```

## Roadmap

| Phase | Status |
|-------|--------|
| Tensor + MatMul + tests | ✅ |
| Dense / ReLU / Softmax | ✅ |
| PyTorch train → export → C++ inference + parity | ✅ |
| Model-loader tests + train/val evaluation | ✅ |
| GitHub Actions CI (reproducible deps) | ✅ |
| Baseline matmul + inference benchmarks | ✅ |
| Measured optimizations (blocked/SIMD/threads) | Next |
| REST API, Docker | Later |

## Limitations

- Fixed architecture: Dense → ReLU → Dense → Softmax
- No ONNX yet (explicit format keeps layout auditable)
- CPU-only **baseline** triple-loop matmul (intentionally unoptimized until measured)

## License

MIT
