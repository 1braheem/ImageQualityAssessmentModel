# Image Quality Assessment Model

An end-to-end no-reference image quality assessment project that predicts an image's overall perceptual quality and whether it is suitable for downstream computer-vision processing. It uses KonIQ-10k, a pretrained EfficientNet-B0 regressor, and a FastAPI image-upload endpoint.

The project is complete through FastAPI. Docker and a separate automated test suite are intentionally outside the current scope.

## What the system returns

For one JPEG, PNG, or WebP image, the API returns:

- `quality_score`: learned EfficientNet-B0 prediction between 0 and 1;
- `mos_equivalent`: the same prediction on the KonIQ 0-100 MOS scale;
- `model_check`: whether the learned score meets the documented 0.60 threshold;
- `resolution_check`: a separate deterministic 224 x 224 minimum-dimension rule; and
- `suitable`: true only when both checks pass.

The 0.60 threshold is a transparent project heuristic, not a separately trained suitability classifier. KonIQ-10k does not contain downstream-task suitability labels.

## Dataset decision: KonIQ-10k

[KonIQ-10k](https://arxiv.org/abs/1910.06180) contains diverse real-world photographs with authentic, mixed distortions and crowdsourced quality ratings. Its size makes it practical for transfer learning while still fitting on one development machine.

The local annotation file was inspected before implementation:

| Property | Value |
| --- | --- |
| Labeled images | 10,073 |
| Image resolution in this copy | 512 x 384 pixels |
| Target | Continuous Mean Opinion Score (MOS) |
| Observed MOS range | 3.9118 to 88.3889 |
| Official training split | 7,058 images |
| Official validation split | 1,000 images |
| Official test split | 2,015 images |
| Annotation fields | `c1`-`c5`, rating count, MOS, score deviation, and split |

The local image directory has 300 additional JPEGs without annotation rows. The loader uses the CSV as its only source of samples, so those files are ignored. All 10,073 labeled files were decoded successfully; one grayscale image is converted to RGB automatically.

### Label limitations

KonIQ-10k directly supports **overall quality regression** through MOS. The photographs can contain real combinations of blur, exposure, noise, compression, and other capture problems, so the overall score is relevant to the project objective.

The annotation fields `c1` to `c5` are distributions of human quality ratings. They are not defect classes. The project therefore does not invent separate predictions for blur, glare, darkness, overexposure, motion artifacts, occlusion, poor framing, or low resolution. Pixel dimensions are checked by a separate rule because all images in this dataset copy have already been standardized to 512 x 384.

## Preprocessing

The annotation CSV defines all samples and the official splits:

```text
CSV row -> Load labeled image -> Convert to RGB -> Resize to 224 x 224
        -> Convert to tensor -> ImageNet normalization -> EfficientNet-B0
```

- Training: the pipeline adds only a random horizontal flip.
- Validation: deterministic preprocessing for checkpoint selection.
- Test: deterministic preprocessing and no use until final evaluation.

Strong blur, color, brightness, or compression augmentation is intentionally avoided because it would change perceived quality without changing the MOS label.

## Model and transfer learning

EfficientNet-B0 was selected because it provides a strong accuracy/compute trade-off and has pretrained ImageNet features in `torchvision`.

```text
224 x 224 RGB image
        -> ImageNet-pretrained EfficientNet-B0 feature extractor
        -> Dropout
        -> Linear layer with one output
        -> Sigmoid
        -> Normalized quality score [0, 1]
```

Transfer learning reuses visual features learned from ImageNet instead of training a convolutional network from scratch. During the head phase, all feature blocks are frozen and only the new 1,281-parameter regression head is trained. The best head checkpoint is then restored, the final two feature blocks are unfrozen, and those blocks are fine-tuned at one-tenth the learning rate.

Training uses MSE loss and AdamW. The checkpoint with the lowest validation loss is saved to `models/efficientnet_b0_koniq10k.pt`; the last epoch is not automatically treated as the best model.

## Measured training results

This development run used an Apple M3 GPU through MPS, batch size 64, seed 42, three head epochs, and two fine-tuning epochs.

| Epoch | Phase | Train loss | Validation loss | Validation MAE | Validation RMSE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Head | 0.018290 | 0.013972 | 9.390 | 11.820 |
| 2 | Head | 0.012869 | 0.012856 | 8.873 | 11.338 |
| 3 | Head | 0.011999 | 0.012622 | 8.805 | 11.235 |
| 4 | Fine-tune | 0.010957 | 0.010725 | 7.921 | 10.356 |
| 5 | Fine-tune | 0.007926 | 0.009762 | 7.643 | 9.880 |

MAE and RMSE are reported in MOS points; losses are calculated on normalized `MOS / 100` targets.

## Test evaluation

The selected epoch-5 checkpoint was evaluated once on all 2,015 unseen official test images:

| Metric | Result |
| --- | ---: |
| MAE | 7.455 MOS points |
| RMSE | 9.674 MOS points |

Example predictions:

| Image | Actual MOS | Predicted MOS | Absolute error |
| --- | ---: | ---: | ---: |
| `10007357496.jpg` | 68.729 | 71.245 | 2.516 |
| `10020766793.jpg` | 81.506 | 75.970 | 5.536 |
| `10020891105.jpg` | 56.830 | 57.928 | 1.098 |
| `10022757465.jpg` | 71.015 | 69.570 | 1.446 |
| `10039534103.jpg` | 76.075 | 57.893 | 18.182 |

Largest observed errors include:

| Image | Actual MOS | Predicted MOS | Absolute error |
| --- | ---: | ---: | ---: |
| `11511265293.jpg` | 24.277 | 66.964 | 42.687 |
| `4980829250.jpg` | 26.729 | 65.868 | 39.140 |
| `8595948410.jpg` | 33.426 | 70.067 | 36.641 |
| `552629141.jpg` | 30.538 | 66.058 | 35.520 |
| `6098120607.jpg` | 23.799 | 57.911 | 34.113 |

The main failure pattern is regression toward average scores. Among test images below MOS 40, MAE is 11.879 and predictions are 9.732 points too high on average. The 42 rare test images above MOS 80 are underpredicted by 9.133 points on average. Severe artistic contrast, overexposure, blur, dark abstract content, and unusual composition appear in several large-error cases. More fine-tuning, score-balanced sampling, higher-resolution crops, and validation on downstream tasks are reasonable future experiments.

## Project structure

```text
ImageQualityAssessmentModel/
|-- api/
|   |-- main.py          # FastAPI routes and upload validation
|   |-- schemas.py       # Typed response models
|   `-- static/          # Local upload UI (HTML, CSS, and JavaScript)
|-- data/
|   `-- README.md        # Local dataset layout
|-- models/
|   `-- README.md        # Checkpoint notes; weights are ignored by Git
|-- src/iqa/
|   |-- data.py          # Dataset, transforms, splits, and DataLoaders
|   |-- model.py         # EfficientNet-B0 regressor and checkpoints
|   |-- train.py         # Head training and fine-tuning
|   |-- evaluate.py      # Test metrics and error examples
|   |-- metrics.py       # Streaming MAE and RMSE
|   `-- inference.py     # Single-image prediction and suitability logic
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Dataset files, virtual environments, credentials, caches, generated evaluation files, and model weights are excluded from Git.

## Installation

From the repository directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place KonIQ-10k at either `data/KonIQ-10k/` or `KonIQ-10k/`. See [data/README.md](data/README.md) for the required files.

Validate the dataset and one real preprocessed batch:

```bash
.venv/bin/python -m src.iqa.data --batch-size 4
```

Add `--verify-images` to decode every labeled file.

## Train the model

The command used for the measured MPS run was:

```bash
.venv/bin/python -m src.iqa.train \
  --batch-size 64 \
  --workers 4 \
  --head-epochs 3 \
  --finetune-epochs 2 \
  --device mps
```

For a CPU-only machine, start with:

```bash
.venv/bin/python -m src.iqa.train \
  --batch-size 16 \
  --workers 0 \
  --head-epochs 3 \
  --finetune-epochs 2 \
  --device cpu
```

The first run downloads the public ImageNet EfficientNet-B0 weights. The trained KonIQ checkpoint is local and intentionally not committed.

## Evaluate the checkpoint

```bash
.venv/bin/python -m src.iqa.evaluate \
  --batch-size 64 \
  --workers 4 \
  --device mps \
  --examples 5 \
  --worst 10 \
  --predictions-out artifacts/test_predictions.json
```

Use `--device cpu --workers 0` on a CPU-only system.

## Run FastAPI

The default API expects `models/efficientnet_b0_koniq10k.pt`. After training, start the server from the repository root:

```bash
source .venv/bin/activate
.venv/bin/python -m uvicorn api.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Open:

- Upload interface: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Readiness check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Use the upload interface

1. Open `http://127.0.0.1:8000/` while the server terminal remains running.
2. Drag an image onto the upload area, or click it and choose a JPEG, PNG, or WebP file.
3. Confirm the preview, then click **Analyze image quality**.
4. Read the 0-100 quality score, suitability decision, learned-model check, and separate resolution rule.

The page validates the basic file type and 10 MiB limit before sending the image to the local API. Server-side validation remains authoritative for corrupt, mismatched, or oversized images.

If the checkpoint is stored elsewhere, set an absolute path before starting:

```bash
export IQA_MODEL_PATH=/absolute/path/to/efficientnet_b0_koniq10k.pt
```

Analyze an image:

```bash
curl -X POST http://127.0.0.1:8000/analyze-quality \
  -F "file=@/absolute/path/to/image.jpg;type=image/jpeg"
```

Example response from a real test image:

```json
{
  "quality_score": 0.7124459147453308,
  "mos_equivalent": 71.24459147453308,
  "suitable": true,
  "model_check": {
    "threshold": 0.6,
    "passes": true
  },
  "resolution_check": {
    "minimum_width": 224,
    "minimum_height": 224,
    "passes": true
  },
  "image": {
    "width": 512,
    "height": 384,
    "format": "JPEG"
  }
}
```

The endpoint accepts JPEG, PNG, and WebP files up to 10 MiB. It returns clear 400, 413, 415, or 503 responses for empty/corrupt images, excessive sizes, unsupported types, or a missing checkpoint.

## Current limitations

- The model predicts overall subjective quality, not named defects.
- The suitability threshold is heuristic because the dataset has no downstream-CV suitability label.
- The resolution check is independent of the neural network.
- All training images in this copy are 512 x 384, so resolution diversity is not learned.
- A five-epoch transfer-learning run is a practical baseline, not an exhaustive hyperparameter study.
- Model weights are not stored in Git; a fresh clone must train a checkpoint or receive one through an appropriate artifact store.
