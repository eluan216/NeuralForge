# NeuralForge

**A C++20 CPU inference runtime with reverse-mode autograd and a first training loop.**

## Current capabilities

- Tensor + cache-blocked matmul, Dense / ReLU / Softmax
- Versioned `.nfmodel` load path with resource limits
- PyTorch export + parity + evaluation suite
- **Reverse-mode autograd** (`Variable`, matmul / add / relu / sum)
- **Training primitives**: MSE loss, SGD, `.nfmodel` export from trained weights
- GitHub Actions CI (GCC + Clang)

## Quick start

```bash
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Training (Phase 4C)

```cpp
#include "neuralforge/autograd.hpp"
#include "neuralforge/loss.hpp"
#include "neuralforge/optim.hpp"
#include "neuralforge/export_model.hpp"

Variable W(/* ... */, true), b(Tensor::zeros({out}), true);
SGD opt(0.05f);
std::vector<Variable> params{W, b};

for (int epoch = 0; epoch < 200; ++epoch) {
    auto pred = add(matmul(x, W), b);
    auto loss = mse_loss(pred, y);   // scalar
    opt.zero_grad(params);
    loss.backward();
    opt.step(params);
}
export_mlp_model("out.nfmodel", W1.data(), b1.data(), W2.data(), b2.data());
```

Tests cover finite-difference MSE gradients, exact SGD updates, decreasing loss on linear regression, and export → `MLPModel::load`.

## Roadmap

| Phase | Status |
|-------|--------|
| Inference stack + eval suite | ✅ |
| Reverse-mode autograd | ✅ |
| MSE + SGD + export training loop | ✅ |
| Cross-entropy + Adam | Next |
| End-to-end C++ train vs PyTorch classifier | Later |
| SIMD / threads (profile-driven) | Deferred |

## License

MIT
