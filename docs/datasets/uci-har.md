# UCI HAR dataset attribution and evaluation protocol

Dataset: Human Activity Recognition Using Smartphones, UCI Machine Learning Repository, DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

Creators: Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto and Xavier Parra (2013). The current [UCI dataset page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) identifies the dataset as **CC BY 4.0**. This dataset's licence is separate from NeuralForge's MIT software licence.

The source contains accelerometer/gyroscope-derived features from 30 participants performing six activities. We use the publisher's 561-feature matrices. We do not claim to reproduce acquisition or raw-signal feature extraction.

Official archive:
https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip

SHA-256: `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.

## Fixed protocol

- Preserve the publisher's official participant-disjoint training/test partition.
- From official training participants, reserve IDs **3, 15, 17, 26 and 30** for validation; use the remaining training participants for fitting.
- Fit additional population mean/std only on fitting rows, with scale=1 for near-constant features.
- Train the classifier in C++20: 561→32→6, Adam learning rate 0.003, batch 128, seed 7, 40 epochs.
- Select the earliest best validation macro-F1 checkpoint.
- Only then read and score the official test feature/label matrices. No test data are supplied to the trainer or used for adaptation.
- Report accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, and every test prediction.
- Compare with a majority baseline and fixed C=1 logistic regression using the same fitting participants and additional preprocessing.
- Verify that the raw-input C artifact agrees with the same weights evaluated in PyTorch.

Activity labels are the publisher's WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING and LAYING. Feature names are prefixed with the original numeric index because the publisher's display names are not all unique.

This is a subject-held-out smartphone activity experiment, not an industrial predictive-maintenance or temporal online-adaptation result. The publisher's upstream engineered-feature preprocessing cannot be audited or undone by this pipeline. The extra standardization and model fitting are isolated to fitting participants.

## Artifacts and attribution

The source archive is downloaded into an ignored cache and checksum-verified. Raw train/validation interchange files exist only in a temporary directory. CI publishes selected model weights/bundle/checkpoint, training history, protocol/environment metadata, predictions and this notice. These derived artifacts retain dataset attribution; they are not a relicensing of the source data.

Reproduce with the commands in docs/real-sensor-evaluation.md. Report changes to this protocol explicitly rather than comparing incompatible scores.
