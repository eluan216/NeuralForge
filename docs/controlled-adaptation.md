# Controlled local adaptation

NeuralForge separates training from deployment. TrainableMLP owns independent trainable parameters and Adam state. snapshot returns a value-owned inference model. Copying TrainableMLP is disabled so candidate training cannot accidentally share mutable leaf variables.

## Checkpoints

Version 1 contains:
- Input/hidden/output dimensions and float32 weights.
- Adam learning rate, betas, epsilon, timestep and both moment tensors.
- Ordered feature and class names.
- An explicit identity-preprocessing marker.

Moments are rebound to new parameter identities on load. Checkpoints are capped at 16 MiB, 262,144 parameters and 4,096 elements per dimension. Names must be unique, nonempty and at most 128 characters. Invalid shapes, non-finite values, negative variances, unsupported formats and trailing data are rejected.

A staged write is renamed into place. Supported POSIX filesystems provide atomic visibility; no fsync/power-loss durability or cryptographic authenticity is promised. Use trusted paths and synchronize writers. Snapshot files do not capture external preprocessing, data-loader/RNG state or application state; exact-resume tests use deterministic batches under the same build.

Adam validates and stages all proposed parameter and moment updates before committing. Autograd output references are weak, preventing completed graphs from retaining their own outputs.

## Gate

AdaptiveDeployment evaluates a candidate internally against:
1. A labelled recent holdout: required accuracy improvement.
2. A labelled reference holdout: bounded overall accuracy loss and per-class recall loss.
3. Parameter-byte and prepared-scratch limits.
4. Explicit caller approval.

The gate checks input and output dimensions; callers must also check semantic feature order and class mapping from their checkpoint/application schema before evaluating an inference-only model. The reference cohort must cover every class. This prevents an unobserved class from silently receiving a passing retention result. Aggregate accuracy alone can still miss important subgroups; supply representative cohorts and add task-specific checks before any high-impact deployment.

Default policy: at least 20 samples per cohort, 0.02 absolute recent accuracy gain, at most 0.01 reference accuracy drop, at most 0.05 per-class recall drop, 4 MiB parameter storage and 256 KiB scratch. These defaults are illustrative thresholds, not a statistical guarantee or validated industrial policy.

A passing candidate without approval records awaiting_approval and leaves the active model untouched. A failed gate records rejected. An approved passing candidate replaces active and retains one previous model. Approved rollback restores that previous model once.

Events contain sequence, action, before/after content IDs and evaluation metrics. The library event log is in memory and grows with actions; persist it in your application if required. IDs use FNV-1a over dimensions and float bits and are not cryptographic signatures. The bool approval parameter is a caller policy signal, not authentication.

## Error monitoring

ErrorDriftMonitor compares classification error across two adjacent rolling labelled windows. Storage is bounded at construction; updates do not allocate. Its alert neither proves feature/concept drift nor triggers training or promotion. It requires outcomes/labels; it cannot detect accuracy loss from unlabelled input alone.

## Synthetic demonstration

neuralforge_adapt_demo generates independent initial-training, adaptation-training, reference-holdout and recent-holdout cohorts from fixed seeds. A known offset is injected into one simulated feature. Candidate training uses initial-data replay plus adaptation examples. No holdout is used for optimizer updates.

The demo verifies approval is necessary, promotes a qualifying candidate, rejects a deliberately bad candidate, and restores the initial model. It writes checkpoints, an inference-only model and report.json with cohort metrics and the event sequence.

This is a reproducible integration exercise, not a real sensor capture, natural-drift benchmark, predictive-maintenance claim or substitute for prospective evaluation. Callers remain responsible for preventing training/evaluation overlap and temporal/device leakage; the library cannot infer provenance from tensors.
