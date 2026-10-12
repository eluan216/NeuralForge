"""Train a tiny PyTorch MLP and export weights for the pure-C++ NeuralForge runtime.

Requires: python -m pip install torch
Usage: python python/train_export.py --output models/toy_classifier.nfmodel

Reports both training and held-out validation accuracy so learning can be
distinguished from generalization on the synthetic Gaussian clusters.
"""
import argparse
from pathlib import Path
import random
import torch
from torch import nn


def make_dataset(n_per_class: int = 400, val_fraction: float = 0.2):
    """Build three Gaussian clusters and split into train / validation.

    The split is stratified by class so each set keeps the same class balance.
    Returns (x_train, y_train, x_val, y_val).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    centers = torch.tensor([[-2.0, -1.5], [2.0, -1.0], [0.0, 2.0]])
    train_xs, train_ys, val_xs, val_ys = [], [], [], []

    for label, center in enumerate(centers):
        points = center + 0.55 * torch.randn(n_per_class, 2)
        labels = torch.full((n_per_class,), label, dtype=torch.long)
        # Per-class shuffle then split so validation is held out, not mixed into training.
        order = torch.randperm(n_per_class)
        points, labels = points[order], labels[order]
        n_val = max(1, int(round(n_per_class * val_fraction)))
        val_xs.append(points[:n_val])
        val_ys.append(labels[:n_val])
        train_xs.append(points[n_val:])
        train_ys.append(labels[n_val:])

    def _cat_and_shuffle(xs, ys):
        x, y = torch.cat(xs), torch.cat(ys)
        order = torch.randperm(len(y))
        return x[order], y[order]

    x_train, y_train = _cat_and_shuffle(train_xs, train_ys)
    x_val, y_val = _cat_and_shuffle(val_xs, val_ys)
    return x_train, y_train, x_val, y_val


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


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
    parser = argparse.ArgumentParser(
        description="Train a toy MLP and export a NeuralForge .nfmodel with train/val metrics."
    )
    parser.add_argument("--output", default="models/toy_classifier.nfmodel")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of each class held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=400,
        help="Samples per class before the train/val split (default: 400)",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.n_per_class < 5:
        parser.error("--n-per-class must be at least 5")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    x_train, y_train, x_val, y_val = make_dataset(
        n_per_class=args.n_per_class, val_fraction=args.val_fraction
    )

    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    train_acc = accuracy(model, x_train, y_train)
    val_acc = accuracy(model, x_val, y_val)

    output = Path(args.output)
    export_model(model, output)

    print(f"train_samples={len(y_train)} val_samples={len(y_val)} val_fraction={args.val_fraction}")
    print(f"training_accuracy={train_acc:.4f}")
    print(f"validation_accuracy={val_acc:.4f}")
    print(f"exported_model={output}")
    print(
        "Note: validation_accuracy is measured on a stratified held-out set "
        "never used for gradient updates."
    )


if __name__ == "__main__":
    main()
