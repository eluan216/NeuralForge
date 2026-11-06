#!/usr/bin/env python3
"""Subject-disjoint UCI HAR evaluation. Training and validation are C++; official test is scored once."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

SOURCE_URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
EXPECTED_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
VALIDATION_SUBJECTS = (3, 15, 17, 26, 30)
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024

def acquire(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "uci-har-240.zip"
    if not path.exists():
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=90) as response, temporary.open("wb") as out:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("Dataset archive exceeds download limit")
                    out.write(chunk)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Dataset archive exceeds size limit")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if EXPECTED_SHA256 is not None and digest != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch; cached data must not be used")
    print(json.dumps({"source": SOURCE_URL, "archive_sha256": digest,
                      "pinned": EXPECTED_SHA256 is not None}), flush=True)
    return path, digest

def selected_zip(path):
    outer = zipfile.ZipFile(path)
    nested = [n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")]
    if nested:
        entry = outer.getinfo(nested[0])
        if entry.file_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Nested archive exceeds size limit")
        inner = zipfile.ZipFile(io.BytesIO(outer.read(entry)))
        outer.close()
        return inner
    return outer

def member(archive, suffix):
    matches = [n for n in archive.namelist() if n.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one dataset member: {suffix}")
    info = archive.getinfo(matches[0])
    if info.file_size > 100 * 1024 * 1024:
        raise ValueError("Dataset member exceeds size limit")
    return archive.read(info).decode("utf-8")


def subject_split(train_subjects, test_subjects):
    import numpy as np
    train_ids, test_ids = set(map(int, train_subjects)), set(map(int, test_subjects))
    validation_ids = set(VALIDATION_SUBJECTS)
    if train_ids & test_ids:
        raise ValueError("Official training and test subjects overlap")
    if not validation_ids < train_ids:
        raise ValueError("Fixed validation participants are missing or leave no fitting participants")
    validation = np.isin(train_subjects, VALIDATION_SUBJECTS)
    return ~validation, validation

def write_dataset(path, x, y, classes=6):
    import numpy as np
    if len(x) != len(y) or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix")
    np.savetxt(path, np.column_stack((x, y)), fmt=["%.9g"] * x.shape[1] + ["%d"],
               header=f"NF_DATASET 1 {len(y)} {x.shape[1]} {classes}", comments="")

def cpp_probabilities(library, bundle, x):
    import ctypes as ct
    import numpy as np
    lib = ct.CDLL(str(library.resolve()))
    pointer = ct.POINTER(ct.c_float)
    lib.nf_model_load.argtypes = [ct.c_char_p, ct.POINTER(ct.c_void_p)]
    lib.nf_model_inputs.argtypes = [ct.c_void_p]
    lib.nf_model_inputs.restype = ct.c_size_t
    lib.nf_model_outputs.argtypes = [ct.c_void_p]
    lib.nf_model_outputs.restype = ct.c_size_t
    lib.nf_model_feature_name.argtypes = [ct.c_void_p, ct.c_size_t]
    lib.nf_model_feature_name.restype = ct.c_char_p
    lib.nf_model_class_name.argtypes = [ct.c_void_p, ct.c_size_t]
    lib.nf_model_class_name.restype = ct.c_char_p
    lib.nf_workspace_create.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    lib.nf_predict_proba.argtypes = [ct.c_void_p, ct.c_void_p, pointer, ct.c_size_t, ct.c_size_t, pointer, ct.c_size_t]
    lib.nf_last_error.restype = ct.c_char_p
    lib.nf_model_destroy.argtypes = [ct.c_void_p]
    lib.nf_model_destroy.restype = None
    lib.nf_workspace_destroy.argtypes = [ct.c_void_p]
    lib.nf_workspace_destroy.restype = None
    handle, workspace = ct.c_void_p(), ct.c_void_p()
    if lib.nf_model_load(str(bundle.resolve()).encode(), ct.byref(handle)):
        raise RuntimeError(lib.nf_last_error().decode())
    try:
        ni, nc = lib.nf_model_inputs(handle), lib.nf_model_outputs(handle)
        if x.shape[1] != ni:
            raise ValueError("Raw dataset and deployed artifact disagree")
        feature_names = [lib.nf_model_feature_name(handle, i).decode() for i in range(ni)]
        class_names = [lib.nf_model_class_name(handle, i).decode() for i in range(nc)]
        if lib.nf_workspace_create(handle, ct.byref(workspace)):
            raise RuntimeError(lib.nf_last_error().decode())
        x = np.ascontiguousarray(x, dtype="float32")
        y = np.empty((len(x), nc), dtype="float32")
        status = lib.nf_predict_proba(handle, workspace, x.ctypes.data_as(pointer), x.size, len(x),
                                    y.ctypes.data_as(pointer), y.size)
        if status:
            raise RuntimeError(lib.nf_last_error().decode())
        if not np.isfinite(y).all() or not np.allclose(y.sum(axis=1), 1, atol=1e-5):
            raise AssertionError("Deployed artifact returned invalid probabilities")
        return y, feature_names, class_names
    finally:
        lib.nf_workspace_destroy(workspace)
        lib.nf_model_destroy(handle)

def metrics(labels, predictions, class_names):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    order = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=order, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "macro_f1": float(f1.mean()),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=order).tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(class_names)}}

def evaluate(args, path, digest):
    import csv
    import importlib.metadata
    import platform
    import shutil
    import subprocess
    import tempfile
    import warnings
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing experimental runs")
    output.mkdir(parents=True, exist_ok=True)
    with selected_zip(path) as archive:
        load = lambda name, dtype: np.loadtxt(io.StringIO(member(archive, name)), dtype=dtype)
        x_train = load("train/X_train.txt", "float32")
        y_train = load("train/y_train.txt", "int64") - 1
        subjects = load("train/subject_train.txt", "int64")
        test_subjects = load("test/subject_test.txt", "int64")
        if x_train.shape != (7352, 561) or len(test_subjects) != 2947:
            raise ValueError("Unexpected official dataset shape")
        fitting, validation = subject_split(subjects, test_subjects)
        feature_names = [f"{int(line.split(maxsplit=1)[0]):03d}:{line.split(maxsplit=1)[1]}"
                         for line in member(archive, "features.txt").splitlines()]
        class_names = [line.split(maxsplit=1)[1] for line in member(archive, "activity_labels.txt").splitlines()]
        if len(feature_names) != 561 or len(set(feature_names)) != 561 or len(class_names) != 6:
            raise ValueError("Invalid dataset schema")
        # Only fitting/validation participants are made available to the C++ trainer.
        with tempfile.TemporaryDirectory(prefix="nf-har-") as td:
            td = Path(td)
            write_dataset(td / "fit.nfdata", x_train[fitting], y_train[fitting])
            write_dataset(td / "validation.nfdata", x_train[validation], y_train[validation])
            (td / "schema.nfschema").write_text(
                "NEURALFORGE_SCHEMA 1 561 6\n" +
                "\n".join(json.dumps(name, ensure_ascii=False) for name in feature_names + class_names) + "\n")
            command = [str(args.trainer.resolve()), "--train", str(td / "fit.nfdata"),
                       "--validation", str(td / "validation.nfdata"), "--schema", str(td / "schema.nfschema"),
                       "--output", str(output), "--epochs", "40", "--hidden", "32", "--batch", "128",
                       "--learning-rate", "0.003", "--seed", "7"]
            subprocess.run(command, check=True)
        training = json.loads((output / "training.json").read_text())
        mean, scale = np.asarray(training["mean"], dtype="float32"), np.asarray(training["scale"], dtype="float32")
        # Test features/labels are read only after training and epoch selection have finished.
        x_test = load("test/X_test.txt", "float32")
        y_test = load("test/y_test.txt", "int64") - 1
    if x_test.shape != (2947, 561) or set(y_train) != set(range(6)) or set(y_test) != set(range(6)):
        raise ValueError("Unexpected official test matrix or class IDs")
    probability, deployed_features, deployed_classes = cpp_probabilities(args.library, output / "model.nfbundle", x_test)
    if deployed_features != feature_names or deployed_classes != class_names:
        raise AssertionError("Deployed semantic schema differs from source schema")
    prediction = probability.argmax(axis=1)
    normalized = lambda x: ((x.astype("float64") - mean.astype("float64")) / scale.astype("float64")).astype("float32")
    # Same-weight parity checks verify deployment/export correctness, not a second model's accuracy.
    import torch
    from eval_suite import load_nfmodel_as_torch
    torch.set_num_threads(1)
    torch_model, _, _, _ = load_nfmodel_as_torch(output / "weights.nfmodel")
    with torch.inference_mode():
        reference_probability = torch.softmax(torch_model(torch.from_numpy(normalized(x_test))), dim=1).numpy()
    if not np.allclose(probability, reference_probability, rtol=2e-4, atol=2e-5):
        raise AssertionError("Real-data deployment/export numerical parity failed")
    mismatches = int(np.count_nonzero(prediction != reference_probability.argmax(axis=1)))
    if mismatches:
        raise AssertionError(f"Real-data predicted-class parity failed: {mismatches}")
    fitting_scaled = normalized(x_train[fitting])
    test_scaled = normalized(x_test)
    majority = DummyClassifier(strategy="most_frequent").fit(fitting_scaled, y_train[fitting])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ConvergenceWarning)
        linear = LogisticRegression(C=1.0, max_iter=1000, random_state=7).fit(fitting_scaled, y_train[fitting])
    baseline_converged = not any(issubclass(w.category, ConvergenceWarning) for w in seen)
    models = {"neuralforge_cpp": metrics(y_test, prediction, class_names),
              "majority_baseline": metrics(y_test, majority.predict(test_scaled), class_names),
              "logistic_regression_baseline": metrics(y_test, linear.predict(test_scaled), class_names)}
    report = {"dataset": {"name": "UCI Human Activity Recognition Using Smartphones", "doi": "10.24432/C54S4K",
                           "url": SOURCE_URL, "archive_sha256": digest, "license": "CC BY 4.0"},
              "protocol": {"fit_subjects": sorted(set(map(int, subjects[fitting]))),
                           "validation_subjects": list(VALIDATION_SUBJECTS),
                           "official_test_subjects": sorted(set(map(int, test_subjects))),
                           "fitting_rows": int(fitting.sum()), "validation_rows": int(validation.sum()),
                           "official_test_rows": len(y_test), "additional_preprocessing": "training-only population mean/std; constant scale=1",
                           "model_selection": "earliest best validation macro-F1 over fixed 40 epochs; no test tuning",
                           "seed": 7, "hidden": 32, "learning_rate": 0.003, "batch": 128},
              "model_bundle_sha256": hashlib.sha256((output / "model.nfbundle").read_bytes()).hexdigest(),
              "selection": {"selected_epoch": training["selected_epoch"], "validation_macro_f1": training["validation_macro_f1"]},
              "models": models, "parity": {"max_abs_probability_error": float(np.max(np.abs(probability-reference_probability))),
                                          "class_mismatches": mismatches, "rtol": 2e-4, "atol": 2e-5},
              "baseline": {"logistic_C": 1.0, "logistic_max_iter": 1000, "converged": baseline_converged},
              "environment": {"platform": platform.platform(), "packages": {name: importlib.metadata.version(name)
                               for name in ("numpy", "scikit-learn", "torch")}},
              "limitations": ["Uses publisher-provided 561 engineered features, not a raw-signal or on-device DSP pipeline.",
                              "Subject-disjoint evaluation does not establish industrial performance or prospective deployment quality.",
                              "Additional normalization is fitted on fitting participants only; upstream published feature processing is outside this implementation.",
                              "No test subjects are used for local adaptation in this experiment.",
                              "The logistic baseline is fixed, without hyperparameter search; the neural model uses validation epoch selection."]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "test_predictions.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["test_row", "anonymous_subject_id", "true_class_1based", "predicted_class_1based"] +
                        ["probability_" + name for name in class_names])
        for i, (subject, actual, pred, probs) in enumerate(zip(test_subjects, y_test, prediction, probability)):
            writer.writerow([i, int(subject), int(actual)+1, int(pred)+1, *map(float, probs)])
    notice = Path(__file__).resolve().parents[1] / "docs" / "datasets" / "uci-har.md"
    if notice.is_file():
        shutil.copyfile(notice, output / "DATASET_NOTICE.md")
    print("REAL_SENSOR_REPORT=" + json.dumps(report), flush=True)

def verify_existing(args, path):
    import csv
    import platform
    import numpy as np
    directory = args.verify_existing
    report = json.loads((directory / "report.json").read_text())
    bundle = directory / "model.nfbundle"
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if digest != report["model_bundle_sha256"]:
        raise AssertionError("Transferred bundle checksum mismatch")
    training = json.loads((directory / "training.json").read_text())
    with selected_zip(path) as archive:
        x = np.loadtxt(io.StringIO(member(archive, "test/X_test.txt")), dtype="float32")
        labels = np.loadtxt(io.StringIO(member(archive, "test/y_test.txt")), dtype="int64") - 1
        subjects = np.loadtxt(io.StringIO(member(archive, "test/subject_test.txt")), dtype="int64")
    with (directory / "test_predictions.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(labels) or any(int(row["test_row"]) != i or int(row["true_class_1based"])-1 != int(labels[i]) or
                                     int(row["anonymous_subject_id"]) != int(subjects[i]) for i, row in enumerate(rows)):
        raise AssertionError("Cross-architecture test row alignment failed")
    expected = np.asarray([[float(row["probability_" + name]) for name in training["class_names"]] for row in rows], dtype="float32")
    actual, names, classes = cpp_probabilities(args.library, bundle, x)
    if names != training["feature_names"] or classes != training["class_names"]:
        raise AssertionError("Transferred semantic schema mismatch")
    mismatches = int(np.count_nonzero(actual.argmax(1) != expected.argmax(1)))
    if mismatches or not np.allclose(actual, expected, rtol=2e-4, atol=2e-5):
        raise AssertionError("Cross-architecture output agreement failed")
    result = {"machine": platform.machine(), "platform": platform.platform(), "bundle_sha256": digest,
              "source": "x86-64-trained artifact; no retraining on this runner", "test_rows": len(labels),
              "class_mismatches": mismatches, "max_abs_probability_error": float(np.max(np.abs(actual-expected))),
              "accuracy": float(np.mean(actual.argmax(1)==labels)), "rtol": 2e-4, "atol": 2e-5,
              "scope": "Hosted native ARM64 runtime, not a Raspberry Pi or physical sensor validation"}
    args.cross_output.parent.mkdir(parents=True, exist_ok=True)
    args.cross_output.write_text(json.dumps(result, indent=2) + "\n")
    print("CROSS_ARCHITECTURE_REPORT=" + json.dumps(result), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/uci-har"))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--verify-existing", type=Path)
    parser.add_argument("--cross-output", type=Path, default=Path("artifacts/arm64-parity.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/real-sensor"))
    args = parser.parse_args()
    path, digest = acquire(args.cache)
    if args.download_only:
        return
    if args.verify_existing is not None:
        if args.library is None or not args.library.is_file():
            parser.error("provide a native --library for bundle verification")
        verify_existing(args, path)
        return
    if args.trainer is None or args.library is None or not args.trainer.is_file() or not args.library.is_file():
        parser.error("provide --trainer and --library from a shared Release build")
    evaluate(args, path, digest)
if __name__ == "__main__":
    main()
