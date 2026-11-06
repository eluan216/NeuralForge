# Documentation

I organize these guides around the tasks someone needs to perform with NeuralForge. Start with the repository [README](../README.md) for the quick start and project direction.

| I want to… | Start here |
| --- | --- |
| Embed inference in C or C++ | [Native embedding and memory contract](embedding.md) |
| Train, resume, evaluate and roll back an update | [Controlled local adaptation](controlled-adaptation.md) |
| Reproduce the recorded sensor experiment | [Real sensor evaluation](real-sensor-evaluation.md) |
| Understand the source dataset and split | [UCI HAR attribution and protocol](datasets/uci-har.md) |
| Compare runtime performance | [Benchmark methodology](benchmarks.md) |
| Find microbenchmarks and saved evidence | [Benchmark index](../benchmarks/README.md) |
| Contribute a change | [Contribution guide](../CONTRIBUTING.md) |

## Artifact types

| File | Purpose |
| --- | --- |
| `.nfmodel` | Inference weights; inputs must already use the required preprocessing. |
| `.nfbundle` | Weights, ordered semantic names and frozen preprocessing for raw feature vectors. |
| `.nfcheckpoint` | Resumable model and Adam state, schema and standardization. Version 1 identity checkpoints remain readable. |
| `.nfdata` | Labelled feature interchange used by the dataset-training CLI and its fixtures. |

The C API loads both inference weights and deployment bundles. Checkpoints are loaded through TrainableMLP. The [embedding guide](embedding.md) describes ownership and error handling.

## Reading the evidence

The synthetic adaptation demo validates the update workflow. The UCI experiment evaluates recorded engineered sensor features. Runtime benchmarks measure execution overhead and resource use. These answer different questions and should not be treated as interchangeable proof.

Source code and current guides describe implemented behavior. Roadmap items in the root README identify work I have not yet validated.
