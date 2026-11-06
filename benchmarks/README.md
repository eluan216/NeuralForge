# Benchmarks and recorded evidence

I keep microbenchmarks, runtime comparisons and model-quality results separate so their meaning stays clear.

| Tool or result | Scope |
| --- | --- |
| [bench_matmul.cpp](bench_matmul.cpp) | Internal matrix-multiplication configurations |
| [bench_inference.cpp](bench_inference.cpp) | Google Benchmark inference cases |
| [run_baselines.sh](../scripts/run_baselines.sh) | Captures compiler/hardware metadata and raw Google Benchmark output |
| [Native latency CLI](../apps/latency.cpp) | Prepared C++ calls, including measurement overhead |
| [Python runtime comparison](../python/benchmark_compare.py) | Same-weight Python API-boundary comparison across three backends |
| [Recorded UCI report](results/uci_har_report.json) | Participant-held-out model quality, baselines, class metrics and provenance |
| [Historical block-size report](results/phase5_block_size.md) | Environment-specific internal tuning; not a competing-runtime comparison |

[Methodology, interpretation and reproduction commands](../docs/benchmarks.md).

## Generated output

Local baseline runs write timestamped directories under `benchmarks/results/baseline_*/`; those directories are ignored. Other experimental output should go under `artifacts/` or a build directory.

The committed JSON report is deliberate evidence with a documented source/protocol. I keep it separate from transient timing files and preserve the original measurement context.
