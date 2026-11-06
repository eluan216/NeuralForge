# NeuralForge

[![CI](https://github.com/eluan216/NeuralForge/actions/workflows/ci.yml/badge.svg)](https://github.com/eluan216/NeuralForge/actions/workflows/ci.yml)
[![Comparative evidence](https://github.com/eluan216/NeuralForge/actions/workflows/verification.yml/badge.svg)](https://github.com/eluan216/NeuralForge/actions/workflows/verification.yml)
[![Real sensor evidence](https://github.com/eluan216/NeuralForge/actions/workflows/sensor-evidence.yml/badge.svg)](https://github.com/eluan216/NeuralForge/actions/workflows/sensor-evidence.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Small models. Local learning. Controlled updates.**

I’m building NeuralForge for C++ applications that need to run small classifiers locally and evaluate model updates before accepting them. My focus is CPU-only Linux applications and edge gateways: an inspectable runtime, explicit memory ownership, resumable training, and a reversible update path.

The core is **C++20**, with a **C11-compatible embedding API**. AVX2 and ARM NEON intrinsics accelerate the prepared inference path where supported. Python is used for export, parity verification and comparisons—not required by the deployed runtime.

> This is an early, deliberately narrow implementation: two-layer float32 MLP classifiers. The controlled-shift demo is synthetic; a separate participant-held-out benchmark uses recorded smartphone sensor features. I do not claim general ONNX compatibility, microcontroller readiness, hard real-time guarantees, or a performance advantage over other runtimes.

## What makes this release useful

| Capability | Current implementation |
| --- | --- |
| Prepared inference | Caller-owned input/output buffers and a reusable hidden-layer workspace; no C++ heap allocations on valid prepared calls |
| Hardware dispatch | Runtime-guarded AVX2 on GCC/Clang x86; NEON on AArch64; selectable scalar fallback |
| Native embedding | Opaque C handles, explicit lengths, thread-local errors and an installable CMake package |
| Local training | Reverse-mode autograd, stable logits cross-entropy, SGD and Adam |
| Checkpoint resume | Weights, Adam moments/timestep, semantic names and stored standardization; v1 identity checkpoints remain readable |
| Deployment bundles | One artifact carries feature order, class names, preprocessing and weights; the C API accepts raw feature vectors |
| Controlled updates | Recent-quality improvement, overall/per-class retention checks, resource budgets and explicit approval |
| Rollback and audit | One previous active model and an in-process event log, with an exported report in the demo |
| Evidence | GCC/Clang and native ARM64 CI; scalar sanitizers; allocation/lifetime tests; parser fuzz smoke; PyTorch/ONNX Runtime comparisons |

## Build and try it

Requires a C++20 compiler and CMake 3.20+. Tests and Google Benchmark are fetched only when their build options are enabled.

```sh
git clone https://github.com/eluan216/NeuralForge.git
cd NeuralForge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure --parallel 2

./build/bin/neuralforge_c_example models/toy_classifier.nfmodel
./build/bin/neuralforge_latency models/toy_classifier.nfmodel
./build/bin/neuralforge_adapt_demo --output artifacts/adaptation
```

The adaptation demo:
1. Trains an initial classifier on simulated two-channel sensor features.
2. Saves and reloads its complete training checkpoint.
3. Trains a separate candidate on labelled shifted examples plus replay data.
4. Evaluates untouched recent and reference holdouts.
5. Requires approval before promotion, rejects a deliberately bad candidate, and verifies rollback.

Read `artifacts/adaptation/report.json` for cohort results and the event sequence. The injected shift and generated data are reproducible simulations, not a hardware experiment or a client outcome.

[Adaptation contract and limitations →](docs/controlled-adaptation.md)

## Real sensor evidence

I also evaluate the native training/deployment path on **UCI Human Activity Recognition Using Smartphones**, separate from the simulated adaptation demo.

| Fixed participant-held-out protocol | Test accuracy | Macro-F1 |
| --- | ---: | ---: |
| NeuralForge C++ 561→32→6 MLP | 94.20% | 94.14% |
| Logistic regression baseline | 95.11% | 95.07% |
| Majority baseline | 18.22% | 5.14% |

The experiment fits on 16 participants, selects an epoch using five other training participants, and evaluates the official nine-participant test cohort: 2,947 rows. Additional standardization is fitted on the fitting participants only. The raw-input deployment bundle matches PyTorch's predicted classes on all test examples.

The linear baseline is stronger here; I report it rather than claiming the neural model is best. These are publisher-provided engineered features, not an implemented raw-signal DSP pipeline or industrial field validation. This experiment does not adapt to the test subjects.

[Protocol, pinned environment, full metrics and reproduction →](docs/real-sensor-evaluation.md)

## Prepare once, infer into your buffers

```cpp
#include <neuralforge/model.hpp>
#include <vector>

auto model = neuralforge::MLPModel::load("models/toy_classifier.nfmodel");
auto workspace = model.prepare();
std::vector<float> input(model.input_features(), 0.25f);
std::vector<float> probabilities(model.output_features());

// Reuse these buffers and workspace for subsequent calls.
model.predict_proba_into(input, 1, probabilities, workspace);
```

The workspace holds `hidden_features * sizeof(float)` scratch bytes, independently of batch size. Weights, vector/object bookkeeping and caller input/output buffers are additional memory. A shared model is immutable; use a separate workspace for every concurrent call.

The low-level MLP interface expects already-preprocessed values. For raw inputs, load an `InferenceArtifact` from a `.nfbundle`; it applies stored preprocessing and retains ordered feature/class names. Non-finite inputs, mismatched buffers, overlap and non-finite intermediate results are rejected. Outputs from a failed call must be discarded.

[Memory, threading and C API details →](docs/embedding.md)

## Embed without Python

```sh
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DNEURALFORGE_BUILD_TESTS=OFF -DNEURALFORGE_BUILD_BENCHMARKS=OFF
cmake --build build-core --parallel 2
cmake --install build-core --prefix "$PWD/install"

cmake -S examples/embedding -B consumer -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build consumer --parallel 2
./consumer/embed models/toy_classifier.nfmodel
```

Use `find_package(NeuralForge CONFIG REQUIRED)` and link `NeuralForge::neuralforge`. The core needs the platform C/C++ runtime, but no Python, model SDK or third-party numerical library. The [C example](examples/c_inference.c) demonstrates ownership and cleanup.

Set `-DBUILD_SHARED_LIBS=ON` for a shared library or ctypes integration. Set `-DNEURALFORGE_ENABLE_SIMD=OFF` for the explicit scalar kernel. No global `-march=native` is required.

## Measure before making performance claims

```sh
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DNEURALFORGE_BUILD_TESTS=OFF -DNEURALFORGE_BUILD_BENCHMARKS=OFF
cmake --build build-shared --parallel 2

python -m pip install numpy onnx onnxruntime
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python python/benchmark_compare.py \
  --library build-shared/lib/libneuralforge.so \
  --output artifacts/comparison.json
```

This compares identical float32 weights and inputs across NeuralForge, PyTorch CPU and ONNX Runtime: three model sizes, batches 1 and 32, one thread, numerical/class agreement, startup, latency percentiles and process peak RSS.

The Python benchmark includes each backend's Python/FFI boundary and reports differing output-buffer ownership. It is not an isolated kernel comparison. Native latency is measured separately. CI results depend on the host; no universal speedup, energy saving or device-memory claim follows from them.

[Benchmark methodology and reproduction →](docs/benchmarks.md)

## Training and checkpointing

The original pure C++ classifier example remains available:

```sh
./build/bin/neuralforge_train_classifier \
  --output artifacts/classifier.nfmodel --epochs 150 --seed 7
```

Create the output directory first. For resumable application training, use `TrainableMLP`, `train_batch`, `save_checkpoint` and `load_checkpoint`. Checkpoints preserve Adam state by parameter order, not transient pointer addresses. A test compares resumed and uninterrupted updates exactly within the same build.

A `.nfmodel` is an inference-only weight artifact. A `.nfbundle` adds preprocessing and semantic names and is directly loadable by the C API. Version 2 `.nfcheckpoint` files preserve weights, Adam state, names and standardization for raw-input training; version 1 identity checkpoints remain readable. `learner.artifact()` produces a deployable bundle without losing preprocessing. Signal processing before the published feature vector remains the application’s responsibility.

## Verification and supported scope

- Linux x86-64: GCC and Clang.
- Linux ARM64: native runner exercising the NEON path; not Raspberry Pi board certification.
- Scalar Debug build: AddressSanitizer, UndefinedBehaviorSanitizer and leak detection.
- Prepared C++/C calls: a standalone allocation counter checks 4,000 calls, including standardized bundle inference.
- Training: graph-lifetime, checkpoint resume and atomic Adam-failure tests.
- Deployment: approval, forgetting, class coverage, resource limits and rollback.
- Model parser: a 2,000-input libFuzzer smoke run; not an exhaustive security audit.
- Export: PyTorch parity and held-out agreement are checked separately from runtime timings.
- Real sensor protocol: official participant separation, training-only additional preprocessing, validation selection and an untouched official test cohort; a second native ARM64 job checks the x86-trained bundle without retraining.

Windows, macOS, bare-metal MCUs, GPU/NPU backends, quantization and physical-device power measurements are future work. This runtime currently supports Dense → ReLU → Dense → Softmax, not arbitrary networks or LLMs. The update gate is a library policy, not an authentication or safety-certification system.

## Repository map

```text
include/neuralforge/  Public C++ interfaces and the C API
src/                 Tensor math, autograd, SIMD dispatch, inference and learning
apps/                CLI examples, adaptation simulation and native timing
examples/            C embedding and installed-package consumer
python/              Export, parity and same-weight comparison tools
tests/               Unit, resource, ownership and workflow checks
fuzz/                In-memory model-loader harness
benchmarks/          Google Benchmark cases and historical measurements
docs/                Embedding, adaptation and evidence contracts
```

## Contribute

I welcome reproducible bug reports, independent benchmark runs and focused integration improvements. Read [CONTRIBUTING.md](CONTRIBUTING.md). Useful next work includes independent physical-device measurements, other attributed real sensor tasks, further checkpoint-format validation and additional output heads—each with tests and honest capability boundaries.

## License

[MIT](LICENSE).
