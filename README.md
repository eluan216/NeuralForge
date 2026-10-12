# NeuralForge

**A C++20 CPU inference runtime with a growing training stack (autograd) and a PyTorch export pipeline.**

## Current capabilities

- Core tensor abstraction with value semantics and shape overflow checks
- Cache-blocked matmul (`Ijk`, default `block_size=64`)
- Dense, ReLU, and numerically stable row-wise Softmax
- Versioned `NEURALFORGE_MLP 1` model format with load-time resource limits
- C++ loader, prediction runner, held-out accuracy evaluator
- PyTorch training/export + numerical parity checker
- Evaluation suite (accuracy, per-example agreement, parity, latency)
- **Reverse-mode autograd** — `Variable` graph with `matmul`, `add` (bias broadcast), `relu`, `sum`; gradients checked against finite differences
- GitHub Actions CI (GCC + Clang)

## Quick start

```bash
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Autograd (Phase 4B)

```cpp
#include "neuralforge/autograd.hpp"
using namespace neuralforge;

Variable x(Tensor{{0.2f, -0.3f}}, true);       // requires_grad
Variable W(Tensor{{0.1f, 0.4f}, {-0.2f, 0.5f}}, true);
Variable b(Tensor{{0.05f, -0.1f}}, true);

auto y = sum(relu(add(matmul(x, W), b)));      // scalar loss
y.backward();
// W.grad(), b.grad(), x.grad() hold ∂y/∂param
```

Gradients for each op are validated in `tests/test_autograd.cpp` against central finite differences.

### Evaluation suite

```bash
python python/eval_suite.py --model models/toy_classifier.nfmodel
```

## Roadmap

| Phase | Status |
|-------|--------|
| Tensor + MatMul + tests | ✅ |
| Dense / ReLU / Softmax | ✅ |
| PyTorch train → export → C++ inference + parity | ✅ |
| Reliability + evaluation suite | ✅ |
| Cache-blocked matmul + block-size tuning | ✅ |
| Reverse-mode autograd (FD-validated) | ✅ |
| Training primitives (loss, SGD/Adam) | Next |
| End-to-end C++ training vs PyTorch | Later |
| SIMD / threads (profile-driven) | Deferred |

## License

MIT
