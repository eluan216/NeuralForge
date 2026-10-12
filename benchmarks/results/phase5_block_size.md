# Phase 5 — Block-size tuning report

**Status:** measured on one cloud host; **not** independently verified by the project owner.
Treat numbers as environment-specific until reproduced.

## Experimental setup (held constant)

| Item | Value |
|------|-------|
| Compiler | g++ (Ubuntu 13.3.0) |
| Build | `CMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`) |
| Host | Linux x86_64, Intel Xeon (L1D 48 KiB, L2 2 MiB) |
| Tool | Google Benchmark 1.8.3 (pinned SHA) |
| Repetitions | 5 |
| Min time | 0.12–0.15 s per config |
| Metric | CPU mean (µs); lower is better |
| Tile order (primary) | `MatMulTileOrder::Ijk` |

Correctness is **not** established by these timings. Numerical agreement of blocked vs naive is covered by the `MatMulBlocked.*` unit tests (`ctest -R MatMulBlocked`).

Reproduce:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -R MatMulBlocked

./build/bin/neuralforge_bench \
  --benchmark_filter='MatMul/Blocked/Ijk/(Square|Rect)' \
  --benchmark_repetitions=5 --benchmark_min_time=0.15s \
  --benchmark_report_aggregates_only=true
```

## Square matrices — Ijk, CPU mean (µs)

| N | block 16 | block 32 | **block 64** | block 128 |
|---|----------|----------|--------------|-----------|
| 64 | 53.2 | 46.9 | **37.6** | 38.6 |
| 128 | 430 | 365 | **313** | 370 |
| 256 | 3507 | 2813 | **2639** | 2957 |
| 512 | 29977 | 25112 | **21204** | 24134 |

**Observation:** For every square size tested, **block_size = 64** is the best or tied-best Ijk setting. 16 is consistently worst; 128 helps only when the matrix already fits in a single tile (N ≤ 64).

## Rectangular matrices — Ijk, CPU mean (µs)

| Shape (M×K×N) | block 16 | block 32 | **block 64** | block 128 |
|---------------|----------|----------|--------------|-----------|
| 32×256×256 | 451 | 354 | **334** | 368 |
| 64×512×512 | 3656 | 3058 | **2617** | 2997 |
| 128×256×64 | 438 | 359 | 310 | **298** |

**Observation:** 64 generalizes well. The thin-output case (N=64) slightly prefers 128 (within ~4%); not enough alone to change the global default.

## Tile-order cross-check (block 32 vs 64)

Ikj at block 64 was competitive or faster on this particular run for some sizes. Earlier runs on a similar host preferred Ijk. **Order ranking is noisy across hosts/runs.** Both orders remain available via `MatMulTileOrder`; the production default stays `Ijk` until a multi-host study shows a stable winner.

Within each order, **block 64 still beat block 32** on the sampled points.

## Decision

| Parameter | Previous default | New default | Rationale |
|-----------|------------------|-------------|-----------|
| `block_size` | 64 | **64 (unchanged)** | Best Ijk setting on square and most rectangular shapes; repeated means, not a single trial |
| `MatMulTileOrder` | Ijk | **Ijk (unchanged)** | No stable multi-run evidence to flip; order remains selectable |

Do **not** promote block 16/32 (clearly slower) or 128 (rarely better, sometimes worse) without new data on the target deployment machine.

## Next milestones

1. Owner-side reproduction of this table under the commands above.
2. Optional: multi-host / multi-compiler confirmation before locking defaults in a release note.
3. Only then consider SIMD or threading if larger GEMMs dominate a profiled inference path.
