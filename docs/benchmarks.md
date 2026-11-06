# Reproducible performance evidence

Do not infer market superiority from a matrix-multiplication microbenchmark. Historical [block-size measurements](../benchmarks/results/phase5_block_size.md) compare internal configurations on a specific host.

## Native prepared API

neuralforge_latency loads once, prepares once, warms 100 calls and records 2,000 batch-one calls. JSON includes selected backend, scratch storage and p50/p95/p99 latency. Timings include clock overhead. This is not an energy or hard real-time measurement.

## Same-weight comparison

python/benchmark_compare.py creates identical float32 MLP weights for:
- 8 → 32 → 3
- 64 → 128 → 8
- 256 → 256 → 16

Each uses batches 1 and 32. Inputs and weights use fixed seeds. Results must pass NumPy-reference probability agreement (rtol 2e-5, atol 2e-6) and predicted-class agreement before timing.

Backends:
- NeuralForge through ctypes and its prepared C ABI, reusing caller output.
- PyTorch CPU with inference_mode and one compute/inter-op thread.
- ONNX Runtime CPUExecutionProvider, one intra/inter-op thread, full graph optimization.

The script uses a new process per backend/shape. It records package versions, platform, CPU, thread policy, startup, 50 warmups, latency percentiles, numerical error and process peak RSS.

### Interpretation

These are Python API-boundary comparisons, not isolated native-kernel timings. Framework return buffers and FFI paths differ; the report discloses ownership. Startup includes backend import, model construction/load and configuration after shared NumPy fixture loading. Process peak RSS uses Linux /proc/self/status VmHWM for the worker exec address space, an OS-reported estimate. It includes Python, NumPy, the backend and allocator behavior, not just model memory. getrusage ru_maxrss is avoided because a fork parent’s larger pre-exec peak can carry over. Fixture files use synthetic weights; classification accuracy has not been established by these timings.

Shared CI hosts have scheduling/frequency noise and are not controlled hardware laboratories. Compare quality/resource trade-offs on your target device, repeat runs and retain raw evidence. Do not merge native-only latency with Python-boundary timings into an unlabeled speedup.

## Reproduce

From the repository root:

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

These commands target Linux; the Python dependencies are evaluation tooling, not deployed runtime requirements.

The Parity and comparative evidence workflow publishes:
- comparison.json and exact model/input fixtures
- source-commit.txt and compiler.txt
- native-latency.json

The workflow also runs the existing PyTorch export/held-out agreement suite separately from timing. CI artifacts retain host-specific measurements rather than putting a permanent fastest-runtime claim in the README.

## Independent hardware results

Useful contributions include host/CPU/OS/compiler details, build flags, backend/package versions, thread and precision settings, repetitions, raw reports and any changes to preprocessing. Native ARM64 CI is not a Raspberry Pi measurement. Energy must be measured with appropriate hardware before reporting it; this project currently provides no energy results.
