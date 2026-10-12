#!/usr/bin/env python3
"""Write synthetic NeuralForge MLP models for latency scaling experiments.

These are random finite weights — not trained classifiers. Use them only to
measure how inference cost grows with width, not to report accuracy.

Usage:
  python python/generate_bench_models.py --out-dir models/bench
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def write_model(path: Path, n_in: int, n_hid: int, n_out: int, seed: int) -> None:
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
    print(f"wrote {path} shape={n_in}->{n_hid}->{n_out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("models/bench"))
    args = p.parse_args()
    shapes = [(64, 128, 10), (128, 256, 10), (256, 256, 20), (512, 512, 64)]
    for i, (n_in, n_hid, n_out) in enumerate(shapes):
        write_model(args.out_dir / f"mlp_{n_in}_{n_hid}_{n_out}.nfmodel", n_in, n_hid, n_out, seed=100 + i)


if __name__ == "__main__":
    main()
