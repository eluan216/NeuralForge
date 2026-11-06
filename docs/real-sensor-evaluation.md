# Recorded sensor evaluation

This experiment tests **native C++ training and raw-input artifact deployment** on UCI's smartphone activity feature dataset. It is separate from the simulated drift/promotion demonstration.

## Measured result

At source commit `7694332495332ff1c94a1159c65f2231be3100ac`, [the evidence workflow](https://github.com/eluan216/NeuralForge/actions/runs/42287514518) produced:

| Model | Official test accuracy | Macro-F1 |
| --- | ---: | ---: |
| NeuralForge C++ MLP | 0.9419748897 | 0.9414421633 |
| Fixed logistic regression | 0.9511367492 | 0.9507325472 |
| Majority baseline | 0.1822192060 | 0.0513777268 |

The MLP has 561 inputs, 32 hidden units and six outputs. On 2,947 official test rows it has zero predicted-class disagreements with PyTorch using the same weights. Maximum probability difference was 2.3245811462402344e-6 (checks: rtol=2e-4, atol=2e-5).

The linear baseline is better on this protocol. These results establish a working recorded-feature evaluation path, not a neural-model superiority or industrial performance claim.

[Complete immutable report, confusion matrices and per-class scores](../benchmarks/results/uci_har_report.json).

## Data separation

The official training partition contains 21 participants. We reserve IDs 3, 15, 17, 26 and 30 for validation:
- Fitting: 16 participants, 5,540 rows.
- Validation: five participants, 1,812 rows.
- Official test: nine different participants, 2,947 rows.

The split is by participant, never random overlapping windows. The C++ trainer receives only fitting and validation files. It fits extra standardization on fitting data, trains 40 fixed epochs with Adam (learning rate 0.003, batch 128, seed 7), and keeps the earliest best validation macro-F1 epoch. In the recorded run it selected epoch 34.

The Python orchestrator reads official test features/labels only after model selection has completed. Test data are not used for training or adaptation. Fixed majority and C=1 logistic baselines use the same fitting participants; no baseline hyperparameter search is performed.

These controls apply to our additional preprocessing and training. The publisher's upstream feature engineering/preprocessing is provided data, not a raw-signal pipeline we recreate or audit.

## Reproduce on Linux

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DNEURALFORGE_BUILD_TESTS=OFF -DNEURALFORGE_BUILD_BENCHMARKS=OFF
cmake --build build --parallel 2

python -m pip install -r python/requirements-sensor.txt
python -m pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python python/sensor_benchmark.py \
  --trainer build/bin/neuralforge_train_dataset \
  --library build/lib/libneuralforge.so \
  --output artifacts/real-sensor
```

Use a new, empty output directory for each experiment. Acquisition verifies the pinned official archive checksum before parsing. Raw interchange files live in a temporary directory; the original archive lives in an ignored cache. The [protocol tests](../python/test_sensor_protocol.py) reject participant overlap or missing validation groups.

The recorded environment used GCC 13.3.0, Linux x86-64/glibc 2.39, NumPy 2.5.4, scikit-learn 1.9.1 and PyTorch 2.15.0+cpu. Same-build deterministic resume is tested; cross-platform bitwise training identity is not promised.

## Artifacts

The workflow publishes:
- model.nfbundle: ordered names, frozen preprocessing and weights; directly loadable through the C API.
- best.nfcheckpoint: resumable optimizer/model/schema/preprocessing state.
- weights.nfmodel: normalized-input weights for independent parity verification.
- training.json and history.csv: configuration, frozen statistics, validation history and selected epoch.
- report.json and test_predictions.csv: protocol, baselines, class metrics, confusion matrices and individual predictions.
- Dataset notice and source/compiler evidence.

Source data are [attributed separately under CC BY 4.0](datasets/uci-har.md); this is not a claim that the MIT software licence replaces dataset rights.

## Limits

This is a smartphone-activity classifier over engineered features. It is not an industrial maintenance, raw-sensor, temporal drift, physical-device energy, latency or safety benchmark. A single fixed split/run does not measure population-wide uncertainty. Additional tasks and prospective device trials remain necessary.

## Cross-architecture deployment check

After the x86-64 training/evaluation job finishes, a separate hosted native ARM64 job downloads that exact bundle and its saved x86 test probabilities. It verifies the bundle SHA-256, rebuilds only the runtime, loads the official test features again, and compares all 2,947 outputs against x86 without retraining or refitting preprocessing.

The check requires zero predicted-class differences and numerical agreement at rtol=2e-4, atol=2e-5. It writes arm64-parity.json with host, bundle hash, probability error and test accuracy. This tests transfer across native CPU architectures; it is not a Raspberry Pi board, live sensor, power or hard-real-time result.
