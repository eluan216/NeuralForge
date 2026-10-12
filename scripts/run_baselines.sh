#!/usr/bin/env bash
# Run baseline matmul + inference benchmarks and record environment metadata.
#
# Usage (from repo root, after a Release build with benchmarks enabled):
#   ./scripts/run_baselines.sh
#   ./scripts/run_baselines.sh --repetitions 10
#
# Outputs under benchmarks/results/baseline_<timestamp>/:
#   env.txt          — compiler, hardware, build settings
#   matmul.json      — Google Benchmark JSON (per-rep + aggregates)
#   inference.json   — Google Benchmark JSON (per-rep + aggregates)
#   matmul.txt       — human-readable console capture
#   inference.txt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
REPETITIONS="${REPETITIONS:-5}"
MIN_TIME="${MIN_TIME:-0.2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --min-time)    MIN_TIME="$2"; shift 2 ;;
    --build-dir)   BUILD_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

MATMUL_BIN="$BUILD_DIR/bin/neuralforge_bench"
INFER_BIN="$BUILD_DIR/bin/neuralforge_bench_inference"

if [[ ! -x "$MATMUL_BIN" ]]; then
  echo "Missing $MATMUL_BIN — configure with -DNEURALFORGE_BUILD_BENCHMARKS=ON and build." >&2
  exit 1
fi
if [[ ! -x "$INFER_BIN" ]]; then
  echo "Missing $INFER_BIN — rebuild after adding bench_inference." >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$REPO_ROOT/benchmarks/results/baseline_${STAMP}"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------
{
  echo "timestamp_utc=$STAMP"
  echo "hostname=$(hostname 2>/dev/null || echo unknown)"
  echo "uname=$(uname -a 2>/dev/null || true)"
  echo ""
  echo "### CPU"
  if [[ -r /proc/cpuinfo ]]; then
    grep -m1 'model name' /proc/cpuinfo || true
    grep -m1 'cpu cores' /proc/cpuinfo || true
    grep -c '^processor' /proc/cpuinfo | awk '{print "logical_cpus=" $1}'
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    sysctl -n hw.ncpu 2>/dev/null | awk '{print "logical_cpus=" $1}'
  fi
  echo ""
  echo "### Memory"
  if [[ -r /proc/meminfo ]]; then
    grep -E 'MemTotal|MemAvailable' /proc/meminfo || true
  fi
  echo ""
  echo "### Compiler / build"
  if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    grep -E 'CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER:|CMAKE_CXX_FLAGS|NEURALFORGE_' \
      "$BUILD_DIR/CMakeCache.txt" | grep -v '^//' || true
  fi
  if command -v "${CXX:-c++}" >/dev/null 2>&1; then
    echo "cxx_version=$("${CXX:-c++}" --version 2>/dev/null | head -1)"
  fi
  echo ""
  echo "### Benchmark settings"
  echo "repetitions=$REPETITIONS"
  echo "min_time_seconds=$MIN_TIME"
  echo "matmul_bin=$MATMUL_BIN"
  echo "inference_bin=$INFER_BIN"
  echo "model_path=${NEURALFORGE_MODEL_PATH:-$REPO_ROOT/models/toy_classifier.nfmodel}"
  echo "note=These are baseline measurements of the unoptimized triple-loop matmul and MLP forward path."
} > "$OUT_DIR/env.txt"

echo "Writing baseline results to $OUT_DIR"
cat "$OUT_DIR/env.txt"
echo ""

COMMON_FLAGS=(
  --benchmark_repetitions="$REPETITIONS"
  --benchmark_min_time="${MIN_TIME}s"
  --benchmark_display_aggregates_only=false
  --benchmark_report_aggregates_only=false
)

# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------
echo "=== MatMul baseline ==="
"$MATMUL_BIN" "${COMMON_FLAGS[@]}" \
  --benchmark_format=console 2>&1 | tee "$OUT_DIR/matmul.txt"
"$MATMUL_BIN" "${COMMON_FLAGS[@]}" \
  --benchmark_format=json --benchmark_out="$OUT_DIR/matmul.json" >/dev/null

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
echo ""
echo "=== Inference baseline ==="
export NEURALFORGE_MODEL_PATH="${NEURALFORGE_MODEL_PATH:-$REPO_ROOT/models/toy_classifier.nfmodel}"
"$INFER_BIN" "${COMMON_FLAGS[@]}" \
  --benchmark_format=console 2>&1 | tee "$OUT_DIR/inference.txt"
"$INFER_BIN" "${COMMON_FLAGS[@]}" \
  --benchmark_format=json --benchmark_out="$OUT_DIR/inference.json" >/dev/null

echo ""
echo "Baseline complete: $OUT_DIR"
echo "  env.txt  matmul.json  inference.json  matmul.txt  inference.txt"
