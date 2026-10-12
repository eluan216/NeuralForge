#!/usr/bin/env python3
"""NeuralForge inference evaluation suite.

Measures, in one run:
  1. Held-out accuracy — PyTorch reference vs C++ runtime on the same weights/data
  2. Numerical parity — max abs error on fixed probe samples
  3. Latency / throughput — Google Benchmark inference across batch sizes

Requires: torch, a built Release tree with neuralforge_eval_accuracy and
          neuralforge_bench_inference (and optionally neuralforge_model_runner).

Usage (from repo root):
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
  python python/eval_suite.py --model models/toy_classifier.nfmodel
  python python/eval_suite.py --model models/toy_classifier.nfmodel --retrain
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

# Reuse dataset construction from the training script when available.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))

from train_export import make_dataset, export_model  # noqa: E402


def load_nfmodel_as_torch(path: Path) -> tuple[nn.Sequential, int, int, int]:
    """Rebuild the PyTorch MLP from a NeuralForge v1 text model file."""
    tokens = path.read_text().split()
    if len(tokens) < 5 or tokens[0] != "NEURALFORGE_MLP" or tokens[1] != "1":
        raise SystemExit(f"Unsupported model header in {path}")
    n_in, n_hid, n_out = map(int, tokens[2:5])
    vals = list(map(float, tokens[5:]))
    need = n_in * n_hid + n_hid + n_hid * n_out + n_out
    if len(vals) != need:
        raise SystemExit(f"Parameter count mismatch in {path}: got {len(vals)} need {need}")

    def take(n: int) -> list[float]:
        nonlocal vals
        chunk, vals = vals[:n], vals[n:]
        return chunk

    w1 = torch.tensor(take(n_in * n_hid)).reshape(n_in, n_hid).T.contiguous()  # (hid, in)
    b1 = torch.tensor(take(n_hid))
    w2 = torch.tensor(take(n_hid * n_out)).reshape(n_hid, n_out).T.contiguous()
    b2 = torch.tensor(take(n_out))

    model = nn.Sequential(nn.Linear(n_in, n_hid), nn.ReLU(), nn.Linear(n_hid, n_out))
    with torch.no_grad():
        model[0].weight.copy_(w1)
        model[0].bias.copy_(b1)
        model[2].weight.copy_(w2)
        model[2].bias.copy_(b2)
    model.eval()
    return model, n_in, n_hid, n_out


def write_dataset(path: Path, x: torch.Tensor, y: torch.Tensor) -> None:
    n, f = x.shape
    lines = [f"{n} {f}"]
    for i in range(n):
        feats = " ".join(f"{v:.8g}" for v in x[i].tolist())
        lines.append(f"{feats} {int(y[i].item())}")
    path.write_text("\n".join(lines) + "\n")


def torch_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(dim=1) == y).float().mean().item())


def run_cpp_accuracy(eval_bin: Path, model: Path, dataset: Path) -> dict:
    out = subprocess.check_output([str(eval_bin), str(model), str(dataset)], text=True)
    # samples=N correct=C accuracy=A latency_ms=L throughput_samples_per_s=T
    m = re.search(
        r"samples=(\d+) correct=(\d+) accuracy=([0-9.eE+-]+) "
        r"latency_ms=([0-9.eE+-]+) throughput_samples_per_s=([0-9.eE+-]+)",
        out,
    )
    if not m:
        raise SystemExit(f"Could not parse C++ eval output:\n{out}")
    return {
        "samples": int(m.group(1)),
        "correct": int(m.group(2)),
        "accuracy": float(m.group(3)),
        "latency_ms": float(m.group(4)),
        "throughput": float(m.group(5)),
        "raw": out.strip(),
    }


def run_parity(parity_script: Path, model: Path, runner: Path) -> str:
    return subprocess.check_output(
        [sys.executable, str(parity_script), str(model), str(runner)], text=True
    ).strip()


def run_latency_bench(bench_bin: Path, model: Path, repetitions: int) -> str:
    env = {"NEURALFORGE_MODEL_PATH": str(model)}
    import os

    full_env = {**os.environ, **env}
    return subprocess.check_output(
        [
            str(bench_bin),
            "--benchmark_filter=Inference/PredictProba",
            f"--benchmark_repetitions={repetitions}",
            "--benchmark_min_time=0.1s",
            "--benchmark_report_aggregates_only=true",
        ],
        text=True,
        env=full_env,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=_REPO / "models" / "toy_classifier.nfmodel")
    p.add_argument("--build-dir", type=Path, default=_REPO / "build")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-class", type=int, default=400)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--retrain", action="store_true", help="Retrain and overwrite --model before eval")
    p.add_argument("--bench-repetitions", type=int, default=3)
    p.add_argument("--skip-bench", action="store_true")
    args = p.parse_args()

    eval_bin = args.build_dir / "bin" / "neuralforge_eval_accuracy"
    runner = args.build_dir / "bin" / "neuralforge_model_runner"
    bench_bin = args.build_dir / "bin" / "neuralforge_bench_inference"
    parity_script = _REPO / "python" / "validate_parity.py"

    for path, label in [
        (eval_bin, "neuralforge_eval_accuracy"),
        (runner, "neuralforge_model_runner"),
    ]:
        if not path.is_file():
            raise SystemExit(f"Missing {label} at {path}; build the project in Release first.")

    torch.manual_seed(args.seed)

    if args.retrain:
        from train_export import accuracy as train_acc  # local name clash avoidance

        x_train, y_train, x_val, y_val = make_dataset(args.n_per_class, args.val_fraction)
        model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(250):
            opt.zero_grad()
            loss_fn(model(x_train), y_train).backward()
            opt.step()
        model.eval()
        export_model(model, args.model)
        print(
            f"retrained model={args.model} "
            f"train_acc={train_acc(model, x_train, y_train):.4f} "
            f"val_acc={train_acc(model, x_val, y_val):.4f}"
        )
    else:
        # Fresh held-out set from the same generative process (seeded).
        _, _, x_val, y_val = make_dataset(args.n_per_class, args.val_fraction)

    torch_model, n_in, _, n_out = load_nfmodel_as_torch(args.model)
    if x_val.shape[1] != n_in:
        raise SystemExit("Validation feature width does not match model")

    # Clamp labels that might be out of range only if model classes differ (should not).
    if int(y_val.max()) >= n_out:
        raise SystemExit("Validation labels exceed model output classes")

    pt_acc = torch_accuracy(torch_model, x_val, y_val)

    with tempfile.TemporaryDirectory() as td:
        ds_path = Path(td) / "val_set.txt"
        write_dataset(ds_path, x_val, y_val)
        cpp = run_cpp_accuracy(eval_bin, args.model, ds_path)

    print("=== Held-out accuracy (same weights, same val set) ===")
    print(f"pytorch_accuracy={pt_acc:.6f}")
    print(f"cpp_accuracy={cpp['accuracy']:.6f} samples={cpp['samples']}")
    print(f"cpp_eval_latency_ms={cpp['latency_ms']:.4f} cpp_eval_throughput={cpp['throughput']:.1f}")
    acc_delta = abs(pt_acc - cpp["accuracy"])
    print(f"accuracy_abs_delta={acc_delta:.6f}")
    if acc_delta > 1e-6 and cpp["correct"] != int(round(pt_acc * cpp["samples"])):
        # Soft check: class predictions should match when weights match.
        print("WARNING: PyTorch and C++ accuracies differ; inspect parity next.")

    print("=== Numerical parity ===")
    print(run_parity(parity_script, args.model, runner))

    if not args.skip_bench:
        if not bench_bin.is_file():
            print("=== Latency bench skipped (binary missing; build with BENCHMARKS=ON) ===")
        else:
            print("=== Inference latency (PredictProba, aggregates) ===")
            print(run_latency_bench(bench_bin, args.model, args.bench_repetitions))


if __name__ == "__main__":
    main()
