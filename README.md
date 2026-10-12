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

Train a fresh model (requires Python 3.10+ and `python -m pip install torch`):

```bash
python python/train_export.py --output models/toy_classifier.nfmodel
python python/validate_parity.py models/toy_classifier.nfmodel build/bin/neuralforge_model_runner
```

The trainer holds out a stratified validation set (default 20% per class) and prints both `training_accuracy` and `validation_accuracy`. The parity checker compares five samples against a PyTorch reference at an absolute tolerance of `2e-5`.

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

Export transposes PyTorch `Linear.weight` so C++ dense matches row-major layout.

## Project structure

```
NeuralForge/
├── include/neuralforge/
│   ├── tensor.hpp
│   ├── nn_ops.hpp
│   └── model.hpp
├── src/
│   ├── tensor.cpp, matmul.cpp, nn_ops.cpp, model.cpp
├── apps/
│   ├── main.cpp, inference_demo.cpp, model_runner.cpp
├── python/
│   ├── train_export.py
│   └── validate_parity.py
├── models/
│   └── toy_classifier.nfmodel
├── tests/
└── benchmarks/
```

## Roadmap

| Phase | Status |
|-------|--------|
| Tensor + MatMul + tests | ✅ |
| Dense / ReLU / Softmax | ✅ |
| PyTorch train → export → C++ inference + parity | ✅ |
| Model-loader tests + train/val evaluation | ✅ |
| Batching, threads, measured optimizations | Next |
| REST API, Docker, CI | Later |

## Limitations

- Fixed architecture: Dense → ReLU → Dense → Softmax
- No ONNX yet (explicit format keeps layout auditable)
- CPU-only baseline matmul

## License

MIT
