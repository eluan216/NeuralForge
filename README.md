# NeuralForge

**High-Performance AI Inference Engine** written in modern C++20.

NeuralForge is a lightweight CPU inference runtime that loads trained machine learning models, executes predictions efficiently, and provides measurable performance optimizations. It is designed as a portfolio-grade engineering project demonstrating systems programming + ML infrastructure skills relevant to FAANG ML Systems / Inference / Performance Engineering roles.

> Goal: Show that you understand both *how a model makes predictions* and *how to make those predictions fast, reliable, and production-ready*.

## Features (Roadmap)

- [x] Core tensor abstraction and matrix multiplication (Weeks 1–2)
- [x] Dense layer, ReLU, numerically stable softmax + inference primitives (Phase 1)
- [ ] Load and run a small neural network trained in PyTorch (Weeks 3–4)
- [ ] Batching, multithreading, memory reuse, and optimized kernels (Weeks 5–6)
- [ ] HTTP prediction API + Docker + CI + benchmarks (Weeks 7–8)

## Architecture (High Level)

```
Python (PyTorch) ──train──► model weights (ONNX / custom)
                                 │
                                 ▼
                    Model Loader (C++)
                                 │
                                 ▼
              NeuralForge Inference Engine
              (Tensors · MatMul · Dense · ReLU · Softmax)
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
               Optimizer                  Benchmarks
            (batch / threads)          (latency / throughput)
                    │
                    ▼
              Prediction API
```

## Quick Start

### Prerequisites
- CMake ≥ 3.20
- C++20 compiler (GCC 11+, Clang 14+, or MSVC 2022)
- GoogleTest & Google Benchmark (fetched automatically via CMake)

```bash
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bin/neuralforge_demo
./build/bin/neuralforge_inference_demo
./build/bin/neuralforge_bench
```

## Project Structure

```
NeuralForge/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/neuralforge/
│   ├── tensor.hpp          # Core Tensor (value semantics)
│   └── nn_ops.hpp          # dense, relu, softmax_rows
├── src/
│   ├── tensor.cpp
│   ├── matmul.cpp
│   └── nn_ops.cpp
├── tests/
│   ├── test_tensor.cpp
│   └── test_inference.cpp
├── benchmarks/
│   └── bench_matmul.cpp
└── apps/
    ├── main.cpp                 # MatMul demo
    └── inference_demo.cpp       # Dense + Softmax demo
```

## Phase 1 Highlights

- **Value-semantic Tensor** – independent storage on copy; overflow-safe shape sizing.
- **`dense`** – batch matrix-multiply + bias.
- **`relu`** – element-wise max(0, x).
- **`softmax_rows`** – numerically stable row-wise softmax (handles large logits).
- Full unit-test coverage for the new primitives.

## Development Roadmap

| Weeks | Focus |
|-------|-------|
| 1–2   | C++ foundations, Tensor, MatMul, unit tests, baseline benchmarks |
| Phase 1 | Dense / ReLU / Softmax + hardened Tensor |
| 3–4   | Train small classifier in PyTorch → export → pure-C++ inference |
| 5–6   | Batching, threads, memory pools, profiling & measurable gains |
| 7–8   | REST API, Docker, GitHub Actions, docs, demo video |

## Correctness & Performance Philosophy

- Every numerical result is compared against a trusted reference (PyTorch / ONNX Runtime) within a defined tolerance.
- All performance claims are backed by reproducible Google Benchmark results on the same hardware.
- No invented numbers. Modest, measured improvements are preferred over marketing claims.

## License

MIT

---

Built as a genuine engineering project — not a demo that cannot be defended in an interview.
