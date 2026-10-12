#!/usr/bin/env python3
"""NeuralForge inference evaluation suite.

Measures, in one run:
  1. Held-out accuracy — PyTorch vs C++ on the same weights/data
  2. Per-example predicted-class agreement (not just matching accuracy rates)
  3. Numerical parity — max abs error; finite probabilities
  4. Latency / throughput — toy model and optional larger synthetic MLPs

Exits with status 0 only when every check passes.

Usage:
  python python/eval_suite.py --model models/toy_classifier.nfmodel
  python python/eval_suite.py --model models/toy_classifier.nfmodel --large-bench
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))

from train_export import make_dataset, export_model  # noqa: E402


class EvalFailure(Exception):
    pass


def load_nfmodel_as_torch(path: Path) -> tuple[nn.Sequential, int, int, int]:
    tokens = path.read_text().split()
    if len(tokens) < 5 or tokens[0] != "NEURALFORGE_MLP" or tokens[1] != "1":
        raise EvalFailure(f"Unsupported model header in {path}")
    n_in, n_hid, n_out = map(int, tokens[2:5])
    vals = list(map(float, tokens[5:]))
    need = n_in * n_hid + n_hid + n_hid * n_out + n_out
    if len(vals) != need:
        raise EvalFailure(f"Parameter count mismatch in {path}: got {len(vals)} need {need}")

    def take(n: int) -> list[float]:
        nonlocal vals
        chunk, vals = vals[:n], vals[n:]
        return chunk

    w1 = torch.tensor(take(n_in * n_hid)).reshape(n_in, n_hid).T.contiguous()
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


def torch_predictions(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise EvalFailure("PyTorch logits contain non-finite values")
        probs = torch.softmax(logits, dim=1)
        if not torch.isfinite(probs).all():
            raise EvalFailure("PyTorch probabilities contain non-finite values")
        if ((probs < 0) | (probs > 1)).any():
            raise EvalFailure("PyTorch probabilities outside [0, 1]")
        row_sums = probs.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            raise EvalFailure("PyTorch probability rows do not sum to 1")
        return logits.argmax(dim=1)


def torch_accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).float().mean().item())


def run_cpp_accuracy(eval_bin: Path, model: Path, dataset: Path) -> dict:
    proc = subprocess.run(
        [str(eval_bin), str(model), str(dataset)], text=True, capture_output=True
    )
    if proc.returncode != 0:
        raise EvalFailure(f"C++ eval failed (exit {proc.returncode}): {proc.stderr.strip()}")
    out = proc.stdout
    m = re.search(
        r"samples=(\d+) correct=(\d+) accuracy=([0-9.eE+-]+) "
        r"latency_ms=([0-9.eE+-]+) throughput_samples_per_s=([0-9.eE+-]+)",
        out,
    )
    if not m:
        raise EvalFailure(f"Could not parse C++ eval output:\n{out}")
    pred_m = re.search(r"predictions=([0-9,]+)", out)
    if not pred_m:
        raise EvalFailure(f"C++ eval did not emit predictions=...\n{out}")
    predictions = [int(v) for v in pred_m.group(1).split(",") if v != ""]
    return {
        "samples": int(m.group(1)),
        "correct": int(m.group(2)),
        "accuracy": float(m.group(3)),
        "latency_ms": float(m.group(4)),
        "throughput": float(m.group(5)),
        "predictions": predictions,
        "raw": out.strip(),
    }


def run_parity(parity_script: Path, model: Path, runner: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(parity_script), str(model), str(runner)],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise EvalFailure(
            f"Parity check failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    line = proc.stdout.strip()
    if "parity_passed=" not in line:
        raise EvalFailure(f"Unexpected parity output: {line}")
    return line


def run_latency_bench(bench_bin: Path, model: Path, repetitions: int) -> str:
    full_env = {**os.environ, "NEURALFORGE_MODEL_PATH": str(model)}
    proc = subprocess.run(
        [
            str(bench_bin),
            "--benchmark_filter=Inference/PredictProba",
            f"--benchmark_repetitions={repetitions}",
            "--benchmark_min_time=0.1s",
            "--benchmark_report_aggregates_only=true",
        ],
        text=True,
        capture_output=True,
        env=full_env,
    )
    if proc.returncode != 0:
        raise EvalFailure(f"Benchmark failed: {proc.stderr.strip()}")
    return proc.stdout


def write_random_nfmodel(path: Path, n_in: int, n_hid: int, n_out: int, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    w1 = (torch.randn(n_in, n_hid, generator=g) * 0.05).reshape(-1).tolist()
    b1 = (torch.randn(n_hid, generator=g) * 0.01).tolist()
    w2 = (torch.randn(n_hid, n_out, generator=g) * 0.05).reshape(-1).tolist()
    b2 = (torch.randn(n_out, generator=g) * 0.01).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NEURALFORGE_MLP 1\n{n_in} {n_hid} {n_out}\n")
        for values in (w1, b1, w2, b2):
            for i, value in enumerate(values):
                f.write(f"{float(value):.9g} ")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            f.write("\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=_REPO / "models" / "toy_classifier.nfmodel")
    p.add_argument("--build-dir", type=Path, default=_REPO / "build")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-class", type=int, default=400)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--bench-repetitions", type=int, default=3)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument(
        "--large-bench",
        action="store_true",
        help="Also benchmark synthetic larger MLPs (64→128→10, 256→256→20)",
    )
    args = p.parse_args()

    eval_bin = args.build_dir / "bin" / "neuralforge_eval_accuracy"
    runner = args.build_dir / "bin" / "neuralforge_model_runner"
    bench_bin = args.build_dir / "bin" / "neuralforge_bench_inference"
    parity_script = _REPO / "python" / "validate_parity.py"

    for path, label in [(eval_bin, "neuralforge_eval_accuracy"), (runner, "neuralforge_model_runner")]:
        if not path.is_file():
            print(f"Missing {label} at {path}; build Release first.", file=sys.stderr)
            return 2

    try:
        torch.manual_seed(args.seed)

        if args.retrain:
            from train_export import accuracy as train_acc

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
            _, _, x_val, y_val = make_dataset(args.n_per_class, args.val_fraction)

        torch_model, n_in, _, n_out = load_nfmodel_as_torch(args.model)
        if x_val.shape[1] != n_in:
            raise EvalFailure("Validation feature width does not match model")
        if int(y_val.max()) >= n_out or int(y_val.min()) < 0:
            raise EvalFailure("Validation labels outside model class range")

        pt_pred = torch_predictions(torch_model, x_val)
        pt_acc = torch_accuracy(pt_pred, y_val)

        with tempfile.TemporaryDirectory() as td:
            ds_path = Path(td) / "val_set.txt"
            write_dataset(ds_path, x_val, y_val)
            cpp = run_cpp_accuracy(eval_bin, args.model, ds_path)

        print("=== Held-out accuracy (same weights, same val set) ===")
        print(f"pytorch_accuracy={pt_acc:.6f}")
        print(f"cpp_accuracy={cpp['accuracy']:.6f} samples={cpp['samples']}")
        print(
            f"cpp_eval_latency_ms={cpp['latency_ms']:.4f} "
            f"cpp_eval_throughput={cpp['throughput']:.1f}"
        )

        if cpp["samples"] != len(pt_pred):
            raise EvalFailure(f"Sample count mismatch: torch={len(pt_pred)} cpp={cpp['samples']}")
        if len(cpp["predictions"]) != len(pt_pred):
            raise EvalFailure(
                f"Prediction vector length mismatch: torch={len(pt_pred)} cpp={len(cpp['predictions'])}"
            )

        mismatches = [
            i
            for i, (a, b) in enumerate(zip(pt_pred.tolist(), cpp["predictions"]))
            if int(a) != int(b)
        ]
        print(f"per_example_class_mismatches={len(mismatches)}")
        if mismatches:
            raise EvalFailure(
                f"Per-example class mismatch on {len(mismatches)} samples "
                f"(first indices: {mismatches[:10]})"
            )

        acc_delta = abs(pt_acc - cpp["accuracy"])
        print(f"accuracy_abs_delta={acc_delta:.6f}")
        if acc_delta > 1e-9:
            raise EvalFailure(f"Accuracy rates differ: delta={acc_delta}")

        print("=== Numerical parity ===")
        print(run_parity(parity_script, args.model, runner))

        if not args.skip_bench:
            if not bench_bin.is_file():
                print("=== Latency bench skipped (binary missing) ===")
            else:
                print("=== Inference latency — toy model (2→16→3) ===")
                print(
                    "Note: high samples/s on this tiny MLP does not generalize to larger models."
                )
                print(run_latency_bench(bench_bin, args.model, args.bench_repetitions))

                if args.large_bench:
                    with tempfile.TemporaryDirectory() as td:
                        for shape in ((64, 128, 10), (256, 256, 20)):
                            n_in_b, n_hid_b, n_out_b = shape
                            mpath = Path(td) / f"bench_{n_in_b}_{n_hid_b}_{n_out_b}.nfmodel"
                            write_random_nfmodel(mpath, n_in_b, n_hid_b, n_out_b)
                            print(
                                f"=== Inference latency — synthetic MLP "
                                f"{n_in_b}→{n_hid_b}→{n_out_b} ==="
                            )
                            print(run_latency_bench(bench_bin, mpath, args.bench_repetitions))

        print("=== eval_suite: ALL CHECKS PASSED ===")
        return 0

    except EvalFailure as e:
        print(f"=== eval_suite: FAILED ===\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
