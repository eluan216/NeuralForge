#!/usr/bin/env python3
"""Same-weight CPU comparison at Python API boundaries; not a native-kernel speed claim."""
from __future__ import annotations
import argparse
import ctypes as ct
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

def versions():
    result = {}
    for package in ("numpy", "torch", "onnx", "onnxruntime"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result

def process_peak_kib():
    # getrusage().ru_maxrss can carry a heavy fork parent's peak across exec.
    # VmHWM belongs to this process's current address space instead.
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None

def worker(args):
    import numpy as np
    data = np.load(args.fixture)
    x, expected = data["x"], data["expected"]
    start = time.perf_counter()
    cleanup = lambda: None
    details = {}
    if args.backend == "neuralforge":
        lib = ct.CDLL(str(args.library.resolve()))
        handle, workspace = ct.c_void_p(), ct.c_void_p()
        ptr = ct.POINTER(ct.c_float)
        lib.nf_model_load.argtypes = [ct.c_char_p, ct.POINTER(ct.c_void_p)]
        lib.nf_workspace_create.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
        lib.nf_predict_proba.argtypes = [ct.c_void_p, ct.c_void_p, ptr, ct.c_size_t, ct.c_size_t, ptr, ct.c_size_t]
        lib.nf_model_parameter_bytes.argtypes = [ct.c_void_p]
        lib.nf_model_parameter_bytes.restype = ct.c_size_t
        lib.nf_workspace_destroy.restype = None
        lib.nf_model_destroy.restype = None
        lib.nf_workspace_destroy.argtypes = [ct.c_void_p]
        lib.nf_model_destroy.argtypes = [ct.c_void_p]
        lib.nf_workspace_bytes.argtypes = [ct.c_void_p]
        lib.nf_workspace_bytes.restype = ct.c_size_t
        lib.nf_backend.restype = ct.c_char_p
        lib.nf_last_error.restype = ct.c_char_p
        if lib.nf_model_load(str(args.fixture.with_suffix(".nfmodel")).encode(), ct.byref(handle)):
            raise RuntimeError(lib.nf_last_error().decode())
        if lib.nf_workspace_create(handle, ct.byref(workspace)):
            lib.nf_model_destroy(handle)
            raise RuntimeError(lib.nf_last_error().decode())
        y = np.empty_like(expected)
        xp, yp = x.ctypes.data_as(ptr), y.ctypes.data_as(ptr)
        def run():
            status = lib.nf_predict_proba(handle, workspace, xp, x.size, x.shape[0], yp, y.size)
            if status:
                raise RuntimeError(lib.nf_last_error().decode())
            return y
        def cleanup():
            lib.nf_workspace_destroy(workspace)
            lib.nf_model_destroy(handle)
        details = {"kernel": lib.nf_backend().decode(), "scratch_bytes": lib.nf_workspace_bytes(workspace),
                   "parameter_bytes": lib.nf_model_parameter_bytes(handle), "output_buffer": "caller-owned, reused"}
    elif args.backend == "onnxruntime":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(args.fixture.with_suffix(".onnx")),
                                      sess_options=options, providers=["CPUExecutionProvider"])
        def run():
            return session.run(None, {"x": x})[0]
        details = {"graph_optimization": "ORT_ENABLE_ALL", "output_buffer": "runtime-returned"}
    else:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], data["w1"].shape[1]),
                                    torch.nn.ReLU(), torch.nn.Linear(data["w2"].shape[0], expected.shape[1]),
                                    torch.nn.Softmax(dim=1))
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(data["w1"].T.copy()))
            model[0].bias.copy_(torch.from_numpy(data["b1"]))
            model[2].weight.copy_(torch.from_numpy(data["w2"].T.copy()))
            model[2].bias.copy_(torch.from_numpy(data["b2"]))
        model.eval()
        tensor = torch.from_numpy(x)
        def run():
            return model(tensor).numpy()
        # Keep the inference-mode context active across warmup and all measurements.
        context = torch.inference_mode()
        context.__enter__()
        cleanup = lambda: context.__exit__(None, None, None)
        details = {"inference_mode": True, "output_buffer": "runtime-returned"}
    startup_ms = (time.perf_counter() - start) * 1000
    try:
        actual = run()
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=2e-5, atol=2e-6):
            raise AssertionError(f"{args.backend}: numerical parity failed")
        if not np.array_equal(actual.argmax(1), expected.argmax(1)):
            raise AssertionError(f"{args.backend}: predicted-class agreement failed")
        max_error = float(np.max(np.abs(actual - expected)))
        for _ in range(50):
            run()
        timings = []
        for _ in range(args.iterations):
            t0 = time.perf_counter_ns()
            run()
            timings.append((time.perf_counter_ns() - t0) / 1000)
        print(json.dumps({"backend": args.backend, "fixture": args.fixture.stem, "batch": x.shape[0],
                          "iterations": args.iterations, "startup_ms": startup_ms,
                          "p50_us": float(np.percentile(timings, 50)),
                          "p95_us": float(np.percentile(timings, 95)),
                          "p99_us": float(np.percentile(timings, 99)),
                          "max_abs_error": max_error, "class_agreement": True,
                          "process_peak_rss_kib": process_peak_kib(),
                          "rss_method": "Linux /proc/self/status VmHWM estimate; current exec address space",
                          **details}))
    finally:
        cleanup()

def fixtures(directory):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    paths = []
    for n_in, hidden, n_out in ((8, 32, 3), (64, 128, 8), (256, 256, 16)):
        rng = np.random.default_rng(7)
        w1 = (rng.standard_normal((n_in, hidden)) * 0.1).astype("float32")
        b1 = (rng.standard_normal(hidden) * 0.01).astype("float32")
        w2 = (rng.standard_normal((hidden, n_out)) * 0.1).astype("float32")
        b2 = (rng.standard_normal(n_out) * 0.01).astype("float32")
        for batch in (1, 32):
            path = directory / f"mlp_{n_in}_{hidden}_{n_out}_b{batch}.npz"
            x = rng.standard_normal((batch, n_in)).astype("float32")
            logits = np.maximum(x @ w1 + b1, 0) @ w2 + b2
            expected = np.exp(logits - logits.max(axis=1, keepdims=True))
            expected /= expected.sum(axis=1, keepdims=True)
            np.savez(path, x=x, expected=expected, w1=w1, b1=b1, w2=w2, b2=b2)
            values = np.concatenate([t.reshape(-1) for t in (w1, b1, w2, b2)])
            path.with_suffix(".nfmodel").write_text(
                f"NEURALFORGE_MLP 1\n{n_in} {hidden} {n_out}\n" +
                " ".join(format(float(v), ".9g") for v in values) + "\n")
            nodes = [helper.make_node("MatMul", ["x", "w1"], ["a"]),
                     helper.make_node("Add", ["a", "b1"], ["b"]),
                     helper.make_node("Relu", ["b"], ["c"]),
                     helper.make_node("MatMul", ["c", "w2"], ["d"]),
                     helper.make_node("Add", ["d", "b2"], ["e"]),
                     helper.make_node("Softmax", ["e"], ["y"], axis=1)]
            graph = helper.make_graph(nodes, "same_weight_mlp",
                    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, n_in])],
                    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, n_out])],
                    [numpy_helper.from_array(t, name) for name, t in
                     (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2))])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
            onnx.checker.check_model(model)
            onnx.save(model, str(path.with_suffix(".onnx")))
            paths.append(path)
    return paths

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparison.json"))
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--backend", choices=("neuralforge", "torch", "onnxruntime"))
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if not 10 <= args.iterations <= 100000:
        parser.error("iterations must be between 10 and 100000")
    if args.backend:
        worker(args)
        return
    if not args.library.is_file():
        parser.error("library not found; build with -DBUILD_SHARED_LIBS=ON")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.parent / "fixtures"
    directory.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows = []
    for fixture in fixtures(directory):
        for backend in ("neuralforge", "torch", "onnxruntime"):
            command = [sys.executable, str(Path(__file__).resolve()), "--library", str(args.library.resolve()),
                       "--backend", backend, "--fixture", str(fixture.resolve()), "--iterations", str(args.iterations)]
            row = json.loads(subprocess.check_output(command, text=True, env=env))
            rows.append(row)
            print(json.dumps(row), flush=True)
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    report = {"scope": "Python API boundary; NeuralForge via ctypes, other backends via native Python APIs",
              "environment": {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu,
                              "python": platform.python_version(), "packages": versions(),
                              "thread_limit": 1, "precision": "float32"},
              "limitations": ["Synthetic weights and inputs: these are not model-accuracy results.",
                             "All timed calls include their Python/FFI overhead. Output-buffer ownership differs and is reported.",
                             "Peak RSS uses the kernel-reported VmHWM estimate for the worker exec image, including Python and imported dependencies.",
                             "Startup includes backend import/load/configuration after shared NumPy fixture loading.",
                             "CI timings are environment-specific, not independently reproduced device guarantees."],
              "results": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
if __name__ == "__main__":
    main()
