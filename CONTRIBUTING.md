# Contributing

I welcome focused improvements backed by reproducible evidence.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure --parallel 2
```

Use a separate build directory for sanitizers or scalar fallback. CI covers Linux GCC, Clang, native ARM64 and a scalar ASan/UBSan build. Verification dependencies belong to Python tooling, not the core runtime.

## Changes that help

- Real integration examples with explicit input, ownership and preprocessing contracts.
- Independent performance results following docs/benchmarks.md.
- Numerical parity and malformed-input regressions.
- Resource limits, checkpoint compatibility and reliable resume behavior.
- Additional model heads/operators with tests and a concrete use case.

Preserve the prepared inference success path's no-C++-heap-allocation contract. Keep ISA-specific instructions behind appropriate dispatch and a portable fallback. Do not add global -march=native to distributable builds.

Add regression tests for meaningful bugs. Never use evaluation holdouts to train an adaptation candidate. Keep simulations labeled; do not call a hosted ARM run a physical-device measurement. Avoid unsupported universal performance or production-safety claims.

C ABI ownership and format changes need explicit compatibility/version handling. Do not let C++ exceptions cross the C boundary. Generated checkpoints, model experiments, build trees and timing artifacts should remain outside committed source unless deliberately included as reviewed fixtures.

The core is MIT licensed. External datasets and dependencies retain their licences.
