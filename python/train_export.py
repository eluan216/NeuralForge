"""Train a tiny PyTorch MLP and export weights for the pure-C++ NeuralForge runtime.

Requires: python -m pip install torch
Usage: python python/train_export.py --output models/toy_classifier.nfmodel
"""
import argparse
from pathlib import Path
import random
import torch
from torch import nn


def make_dataset(n_per_class: int = 400):
    # Three reproducible Gaussian clusters; useful for exercising the full pipeline.
    centers = torch.tensor([[-2.0, -1.5], [2.0, -1.0], [0.0, 2.0]])
    xs, ys = [], []
    for label, center in enumerate(centers):
        xs.append(center + 0.55 * torch.randn(n_per_class, 2))
        ys.append(torch.full((n_per_class,), label, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    order = torch.randperm(len(y))
    return x[order], y[order]


def export_model(model: nn.Sequential, path: Path):
    first: nn.Linear = model[0]
    second: nn.Linear = model[2]
    # C++ dense expects (in_features, out_features), while PyTorch stores (out, in).
    w1 = first.weight.detach().cpu().T.contiguous().reshape(-1).tolist()
    b1 = first.bias.detach().cpu().reshape(-1).tolist()
    w2 = second.weight.detach().cpu().T.contiguous().reshape(-1).tolist()
    b2 = second.bias.detach().cpu().reshape(-1).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NEURALFORGE_MLP 1\n{first.in_features} {first.out_features} {second.out_features}\n")
        for values in (w1, b1, w2, b2):
            for i, value in enumerate(values):
                f.write(f"{value:.9g} ")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/toy_classifier.nfmodel")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    x, y = make_dataset()
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    output = Path(args.output)
    export_model(model, output)
    print(f"training_accuracy={accuracy:.4f}")
    print(f"exported_model={output}")
    print("Format: versioned plain-text NeuralForge MLP; parameters are trained, not hand-written.")


if __name__ == "__main__":
    main()
