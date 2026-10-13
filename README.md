# NeuralForge

**A C++20 CPU runtime that can train a classifier with its own autograd engine and export it for inference.**

## Current capabilities

- Tensor + cache-blocked matmul, Dense / ReLU / Softmax
- Versioned `.nfmodel` load/export
- Reverse-mode autograd (`Variable`)
- Losses: MSE, **stable multiclass cross-entropy (logits)**
- Optimizers: SGD, **Adam** (persistent moments, bias correction)
- End-to-end **C++ classifier training** → export → inference reload
- PyTorch export path + evaluation suite + CI

## Quick start

```bash
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Train a classifier in pure C++

```bash
./build/bin/neuralforge_train_classifier \
  --output models/cpp_trained_classifier.nfmodel \
  --epochs 150 --seed 7
# prints train loss, held-out accuracy, and verifies reload
```

Training metrics (loss, val accuracy) and inference parity after export are reported separately.

## Roadmap

| Phase | Status |
|-------|--------|
| Inference + eval suite | ✅ |
| Autograd | ✅ |
| MSE + SGD | ✅ |
| Cross-entropy + Adam + C++ classifier | ✅ |
| Broader PyTorch parity / larger models | Next |
| Profile-driven SIMD / threads | Deferred |

## License

MIT
