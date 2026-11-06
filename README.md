# NeuralForge

[![CI](https://github.com/eluan216/NeuralForge/actions/workflows/ci.yml/badge.svg)](https://github.com/eluan216/NeuralForge/actions/workflows/ci.yml)
[![Runtime comparisons](https://github.com/eluan216/NeuralForge/actions/workflows/verification.yml/badge.svg)](https://github.com/eluan216/NeuralForge/actions/workflows/verification.yml)
[![Sensor evaluation](https://github.com/eluan216/NeuralForge/actions/workflows/sensor-evidence.yml/badge.svg)](https://github.com/eluan216/NeuralForge/actions/workflows/sensor-evidence.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Small models. Local learning. Controlled updates.**

I’m **Oguma Eluanatein Odo**, and I’m building NeuralForge for applications that need small machine-learning models to run and learn locally.

I want the path from a feature vector to a prediction to be understandable: who owns the memory, which CPU kernel runs, how a model is evaluated, and what happens when an update goes wrong. NeuralForge brings those concerns together in an inspectable **C++20 core**, a **C11-compatible API**, and command-line tools.

My current focus is CPU-only Linux applications and edge gateways. Python supports data preparation, export and independent verification; deployed inference runs through the native library.

[Quick start](#try-it-locally) · [Documentation](docs/README.md) · [Measured results](#the-evidence-i-publish) · [Contributing](CONTRIBUTING.md)

## Why I’m building it

I’m interested in more than getting a model to produce a prediction. I want to keep its feature meanings and preprocessing intact, measure the resources it uses, resume training correctly, and evaluate a candidate before replacing an active model.

That is the direction I’m taking with NeuralForge: a small, testable workflow for local learning, with explicit ownership and a reversible update path.

## How you can use it

| Interface | What you can do |
| --- | --- |
| **Command line** | Run inference, train a classifier, inspect latency, or walk through the adaptation demo. |
| **C++ library** | Load a model or deployment bundle and infer directly into reusable application buffers. |
| **C API** | Embed the runtime in a C application or connect another language through FFI. My Python comparison tools use this API through `ctypes`. |

An application supplies features and decides what to do with the prediction:

```text
Application features → Stored preprocessing → Native inference → Class probabilities
```

Raw-sensor acquisition, feature extraction and application actions remain the integrating application's responsibility.

## Try it locally

You need a C++20 compiler and CMake 3.20+. The commands below target Linux. Test and Google Benchmark dependencies are fetched when their build options are enabled.

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

The adaptation demo trains an initial classifier, resumes a checkpoint, evaluates a separate candidate, requires approval to promote it, rejects a bad candidate, and verifies rollback. Its report is written to `artifacts/adaptation/report.json`.

I use generated sensor-like features for this demonstration. It is a simulation of the update workflow, separate from the recorded-data experiment below.

## What I’ve implemented

- **Prepared inference:** caller-owned buffers and reusable workspace, with no C++ heap allocations on valid prepared calls.
- **CPU kernels:** guarded AVX2 dispatch on supported GCC/Clang x86 builds, AArch64 NEON, and a scalar fallback.
- **Local training:** reverse-mode autograd, stable logits cross-entropy, SGD and resumable Adam.
- **Deployment bundles:** ordered feature names, class names, frozen standardization and weights travel together.
- **Controlled updates:** recent-quality improvement, overall/per-class retention checks, resource limits and explicit promotion approval.
- **Recovery:** one-level rollback restores the full artifact; an in-process audit records update decisions.
- **Native embedding:** opaque C handles, explicit buffer lengths, thread-local errors and an installable CMake package.

The current model architecture is **Dense → ReLU → Dense → Softmax**, using float32 values.

## Embed it in an application

My low-level C++ interface lets you prepare once and reuse your buffers:

```cpp
#include <neuralforge/model.hpp>
#include <vector>

int main() {
    auto model = neuralforge::MLPModel::load("models/toy_classifier.nfmodel");
    auto workspace = model.prepare();
    std::vector<float> input(model.input_features(), 0.25f);
    std::vector<float> probabilities(model.output_features());

    model.predict_proba_into(input, 1, probabilities, workspace);
}
```

This low-level interface expects preprocessed inputs. For raw feature vectors, use `InferenceArtifact::load` with a `.nfbundle`; it applies the stored preprocessing. I expose the same bundle-loading path through the C API.

A shared model is immutable. Give each concurrent inference its own workspace, and discard output from a failed call.

[Installation, the C example, ownership and memory details →](docs/embedding.md)

## The evidence I publish

I keep correctness, model quality and runtime performance separate.

On **UCI Human Activity Recognition Using Smartphones**, my fixed participant-held-out experiment produced:

| Model | Test accuracy | Macro-F1 |
| --- | ---: | ---: |
| NeuralForge C++ 561→32→6 MLP | 94.20% | 94.14% |
| Logistic regression baseline | 95.11% | 95.07% |
| Majority baseline | 18.22% | 5.14% |

I fit on 16 participants, select an epoch using five separate validation participants, and score the official nine-participant test cohort: 2,947 examples. Additional standardization is fitted only on fitting data. The deployed bundle matches PyTorch's predicted classes on every test example and transfers unchanged to hosted native ARM64 with zero class differences.

The linear baseline is stronger here, and I report that result. These are publisher-provided engineered smartphone features, not an industrial field trial or a raw-signal pipeline I implemented.

[Full protocol, metrics, attribution and reproduction →](docs/real-sensor-evaluation.md)

I also compare identical weights against PyTorch CPU and ONNX Runtime. Those measurements disclose Python/FFI overhead, thread settings and output-buffer ownership. Larger-batch results favor ONNX Runtime; I do not claim a universal speed advantage.

[Runtime benchmark methodology and commands →](docs/benchmarks.md)

## How I check changes

CI covers Linux GCC, Clang, hosted native ARM64 and a scalar sanitizer build. Checks include:

- Numerical agreement, C embedding and installed-package consumption.
- Training-graph lifetime, checkpoint resume and transactional Adam failures.
- Semantic schema checks, class retention, resource limits and rollback.
- Allocation counting across 4,000 prepared C/C++ calls.
- A bounded model/artifact parser fuzz run.
- Participant-separated sensor evaluation and x86-to-ARM64 artifact transfer.

Training, loading, error handling and audit logging allocate memory. The prepared-call allocation contract is narrower than a fixed-memory or hard-real-time guarantee.

## Find your way around

```text
include/neuralforge/  Public C++ interfaces and C API
src/                 Kernels, inference, autograd and local learning
apps/                Command-line examples and dataset trainer
examples/            C embedding and installed-package consumer
python/              Data preparation, export and evaluation tools
tests/               Unit, ownership, resource and workflow checks
fuzz/                Parser fuzz harness
benchmarks/          Microbenchmarks and recorded evidence
docs/                Setup, contracts, methodology and dataset attribution
```

Start with the [documentation index](docs/README.md). I keep generated builds, caches, checkpoints and experimental outputs out of the source tree through [the ignore rules](.gitignore).

## Where I’m taking it next

My next priorities are independent physical-device measurements, more attributed sensor tasks, stronger validation of model formats, and additional output heads where a concrete application needs them.

The supported evidence currently covers hosted Linux x86-64 and ARM64. Physical sensor boards, bare-metal microcontrollers, GPU/NPU backends, quantization and arbitrary ONNX models remain future work.

## Contribute

I welcome reproducible bug reports, independent measurements and focused integration improvements. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and include the build, inputs and evidence needed to reproduce your result.

You can also find me on [GitHub](https://github.com/eluan216).

## License

I release NeuralForge under the [MIT License](LICENSE). External datasets and dependencies retain their own licenses; the UCI evaluation includes [separate dataset attribution](docs/datasets/uci-har.md).
