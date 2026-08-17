# Image Quality Assessment Model

An end-to-end no-reference image quality assessment project for predicting whether an image is suitable for downstream computer-vision processing. The planned pipeline uses KonIQ-10k, a pretrained EfficientNet-B0 regression model, and a FastAPI upload endpoint.

## Current stage

Stage 1 - project and repository setup - is complete. Dataset loading, model training, measured results, inference, and the API will be added in separate, understandable checkpoints. No training result is claimed before an experiment is actually run.

## Objective

Given one image, the model will predict an overall perceptual quality score. A documented threshold will convert that score into a suitability decision. Deterministic properties, such as the uploaded image's pixel dimensions, will be reported separately from the learned model prediction.

## Dataset decision: KonIQ-10k

[KonIQ-10k](https://arxiv.org/abs/1910.06180) is a large, public, no-reference image quality assessment dataset made from real-world photographs with authentic, mixed distortions. It is a practical fit because it is large enough for transfer learning while remaining manageable on a single development machine.

The local annotation file was inspected before model implementation:

| Property | Value |
| --- | --- |
| Labeled images | 10,073 |
| Image resolution in this copy | 512 x 384 pixels |
| Target | Continuous Mean Opinion Score (MOS) |
| Observed MOS range | 3.9118 to 88.3889 |
| Official training split | 7,058 images |
| Official validation split | 1,000 images |
| Official test split | 2,015 images |
| Rating columns | `c1` to `c5`, rating count, MOS, and score deviation |

The image directory contains 300 additional JPEG files that do not have annotation rows. The future dataset loader will treat the CSV as the source of truth and will ignore those files. It will also convert every image to RGB because one labeled JPEG in this local copy is grayscale.

### What the labels support

KonIQ-10k directly supports **overall image-quality regression** through MOS. The photographs can contain real combinations of blur, exposure, noise, compression, and other capture problems, so the overall score is relevant to the internship objective.

The supplied annotations do **not** identify individual defects. Therefore this project will not present `c1` to `c5` as defect classes and will not claim separate model predictions for:

- blur or motion blur;
- glare;
- darkness or overexposure;
- motion artifacts;
- occlusion;
- poor framing; or
- low resolution.

Low resolution can be checked from image dimensions with a transparent rule. Supporting the other defects as separate outputs would require a different dataset with verified per-defect labels or an additional labeled model.

## Planned machine-learning approach

1. Read labeled filenames and MOS values from the annotation CSV and use its official splits.
2. Resize inputs to 224 x 224 and apply ImageNet normalization for EfficientNet-B0.
3. Apply mild augmentation only to the training set.
4. Replace EfficientNet-B0's ImageNet classifier with a scalar regression head.
5. Train the new head while the feature extractor is frozen, then optionally fine-tune the last feature layers at a lower learning rate.
6. Select the checkpoint with the best validation loss and report MAE and RMSE on the untouched test split.
7. Expose the verified inference pipeline through `POST /analyze-quality`.

## Project structure

```text
ImageQualityAssessmentModel/
|-- api/                 # FastAPI application (added at the API stage)
|-- data/                # Dataset placement notes; image data is ignored by Git
|-- models/              # Checkpoint notes; trained weights are ignored by Git
|-- src/
|   `-- iqa/             # Dataset, model, training, evaluation, and inference code
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Local setup

Python 3.11 or newer is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

See [data/README.md](data/README.md) for the expected dataset layout. Dataset files, virtual environments, credentials, caches, generated results, and model weights are intentionally excluded from Git.

## Scope

This phase of the project ends with a working FastAPI integration and complete usage documentation. Docker and a separate automated test suite are intentionally deferred.
