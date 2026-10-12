"""Compare C++ inference probabilities with a PyTorch reference from the same export.

Usage: python python/validate_parity.py models/toy_classifier.nfmodel build/bin/neuralforge_model_runner
Requires: python -m pip install torch
"""
import re
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: validate_parity.py MODEL.nfmodel PATH_TO_MODEL_RUNNER")
    model_path, runner = Path(sys.argv[1]), Path(sys.argv[2])
    tokens = model_path.read_text(encoding="utf-8").split()
    if len(tokens) < 5 or tokens[0] != "NEURALFORGE_MLP" or tokens[1] != "1":
        raise SystemExit("Invalid model format")
    n_in, n_hidden, n_out = map(int, tokens[2:5])
    values = torch.tensor([float(v) for v in tokens[5:]], dtype=torch.float32)
    expected = n_in*n_hidden + n_hidden + n_hidden*n_out + n_out
    if values.numel() != expected:
        raise SystemExit(f"Expected {expected} parameters, found {values.numel()}")
    offset = 0
    w1 = values[offset:offset+n_in*n_hidden].reshape(n_in, n_hidden); offset += n_in*n_hidden
    b1 = values[offset:offset+n_hidden]; offset += n_hidden
    w2 = values[offset:offset+n_hidden*n_out].reshape(n_hidden, n_out); offset += n_hidden*n_out
    b2 = values[offset:offset+n_out]
    cases = [[-2.0, -1.5], [2.0, -1.0], [0.0, 2.0], [0.2, -0.1], [1.3, 1.1]]
    max_error = 0.0
    for sample in cases:
        x = torch.tensor(sample, dtype=torch.float32)
        reference = F.softmax(F.relu(x @ w1 + b1) @ w2 + b2, dim=0)
        output = subprocess.check_output([str(runner), str(model_path), *map(str, sample)], text=True)
        match = re.search(r"probabilities=\[([^]]+)\]", output)
        if not match:
            raise SystemExit(f"Could not parse C++ output: {output}")
        actual = torch.tensor([float(v.strip()) for v in match.group(1).split(",")])
        error = torch.max(torch.abs(reference - actual)).item()
        max_error = max(max_error, error)
        if error > 2e-5:
            raise SystemExit(f"Parity failure for {sample}: max_abs_error={error:.8g}; output={output}")
    print(f"parity_passed={len(cases)} max_absolute_error={max_error:.8g} tolerance=2e-5")


if __name__ == "__main__":
    main()
