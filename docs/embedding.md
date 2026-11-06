# Native embedding and memory contract

The deployed runtime uses C++20; the public C header is consumable from C11 and C-FFI languages. This does not require extra language runtimes or a Python service.

## Ownership

Load an immutable MLPModel, call prepare once, and reuse caller-owned float32 input/output buffers. A workspace contains one hidden-layer scratch vector. It does not own model weights, input or output arrays. Memory reported by scratch_bytes excludes vector bookkeeping and allocator overhead.

predict_proba_into accepts flattened row-major input and output spans plus an explicit batch count. Lengths must equal batch × input_features and batch × output_features. Overflow, overlap, wrong workspace dimensions and non-finite input/intermediates are errors. A failed call may have partially written output; discard it.

The old allocating predict_proba API remains as a reference/convenience path. Only the prepared success path has the no-C++-heap-allocation contract. Training, loading, checkpoints, errors, model promotion and audit logging allocate memory. This is not a global fixed-memory or hard real-time guarantee.

## C ABI

See [c_api.h](../include/neuralforge/c_api.h) and [the C example](../examples/c_inference.c).

- ABI version: 1. Opaque model/workspace handles are created and destroyed by this library.
- Counts are float elements, not bytes.
- Check every nf_status before using output.
- nf_last_error returns a thread-local message; do not retain its pointer across subsequent calls.
- C++ exceptions are caught at the C boundary. Invalid pointers or insufficient caller-allocated memory cannot be validated by the library; callers must supply valid buffers.
- Models may be shared among threads. Each simultaneous inference requires a different workspace. Do not destroy handles while another call uses them.

The Python comparison uses this same C ABI through ctypes. Rust, Go or other language bindings can use the ABI without changing the core; they are not bundled in this release.

## Hardware kernels

The prepared dense operation uses a scaled-add kernel:
- GCC/Clang x86: AVX2 function compiled with a local target attribute, selected after CPU feature detection.
- AArch64: NEON intrinsics with scalar handling of vector tails.
- Otherwise, or with NEURALFORGE_ENABLE_SIMD=OFF: explicit scalar implementation.

The rest of the binary does not require -march=native. Compilers may still auto-vectorize scalar code. Small floating-point differences are expected across architectures and contraction settings; tests and comparisons use disclosed tolerances. No quantized, GPU, NPU or hand-written assembly backend is claimed.

## Install

From the repository root:

```sh
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
  -DNEURALFORGE_BUILD_TESTS=OFF -DNEURALFORGE_BUILD_BENCHMARKS=OFF
cmake --build build-core --parallel 2
cmake --install build-core --prefix "$PWD/install"

cmake -S examples/embedding -B consumer -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build consumer --parallel 2
./consumer/embed models/toy_classifier.nfmodel
```

Use `find_package(NeuralForge CONFIG REQUIRED)` and link `NeuralForge::neuralforge`. Disabling tests and benchmarks avoids their FetchContent dependencies. `BUILD_SHARED_LIBS` controls static/shared output. The installed consumer example is built and run in CI.

Supported execution evidence currently covers Linux x86-64 and hosted Linux ARM64. Standard-library filesystem use, dynamic loading and training allocations mean this is not a bare-metal microcontroller runtime.

## Allocation evidence

tests/test_allocations.cpp overrides ordinary and aligned C++ allocation operators, prepares both interfaces first, and checks 4,000 inference calls. It measures C++ new/new[] calls, not every possible operating-system or C-library allocation. Input/output storage is allocated outside the measured region.

## Deployable bundles and semantic metadata

InferenceArtifact couples weights with ordered feature/class names and per-feature mean/scale. The C API's nf_model_load accepts both legacy .nfmodel files (identity preprocessing and generated default names) and .nfbundle files (stored metadata and preprocessing).

For bundles, pass raw feature values in the declared order. Query nf_model_feature_name and nf_model_class_name to inspect meaning. Returned strings belong to the model and remain valid until destruction. nf_model_preprocessing_bytes reports the float mean/scale payload, excluding string/allocator bookkeeping.

An ArtifactWorkspace adds one input-width normalization vector only when standardization is nonidentity. Its scratch byte count includes that vector plus the dense hidden workspace. Valid raw-input calls do not allocate; scalar conversion uses the bundle's frozen statistics. The 4,000-call allocation check covers both C++ interfaces and both legacy/bundled C paths.

Bundle loading is capped at 16 MiB and validates names, dimensions, finite statistics, positive scales and embedded model consistency. Input/output metadata dimensions are capped at 4,096; bundled weights are capped at 262,144 parameters. Load-time peak memory includes parsing and model copies and is not equal to prepared inference scratch storage.

Bundle fingerprints include semantic names and preprocessing as well as model weights. They remain noncryptographic content IDs. AdaptiveDeployment now stores the full artifact, checks ordered feature/class equality before evaluation, and restores preprocessing/metadata during rollback.
