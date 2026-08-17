# Image Quality Assessment Model

An end-to-end no-reference image quality assessment project that predicts an image's overall perceptual quality and whether it is suitable for downstream computer-vision processing. It uses KonIQ-10k, a pretrained EfficientNet-B0 regressor, eight explainable image diagnostics, and a FastAPI image-upload endpoint with a local web interface.

The project is complete through FastAPI. Docker and a separate automated test suite are intentionally outside the current scope.

## What the system returns

For one JPEG, PNG, or WebP image, the API returns:

- `quality_score`: learned EfficientNet-B0 prediction between 0 and 1;
- `mos_equivalent`: the same prediction on the KonIQ 0-100 MOS scale;
- `model_check`: whether the learned score meets the documented 0.60 threshold;
- `resolution_check`: a separate deterministic 224 x 224 minimum-dimension rule;
- `suitable`: true only when the learned-model check and all eight diagnostics pass;
- `image`: the decoded file's original width, height, and detected format;
- `diagnostics`: eight named diagnostic or risk checks; and
- `diagnostics_summary`: the number of flagged checks and whether review is recommended.

The 0.60 threshold is a transparent project heuristic, not a separately trained suitability classifier. KonIQ-10k does not contain downstream-task suitability labels.

The suitability decision combines two clearly separated sources: `model_check` must pass and every entry in `diagnostics` must pass. The diagnostics therefore affect the final decision, but remain deterministic pixel measurements rather than outputs from the EfficientNet regression head. `resolution_check` exposes the original minimum-dimension rule directly, while `diagnostics.low_resolution` reports that evidence in the common diagnostic format. Because the two advisory risk heuristics participate in this conservative decision, a false risk flag can make `suitable` false; consumers should inspect the individual explanation and measurement rather than treat the Boolean as ground truth.

### Learned score versus deterministic diagnostics

There are two different kinds of analysis in one response:

1. **Learned overall quality:** EfficientNet-B0 produces one `quality_score`, trained against KonIQ-10k MOS. `mos_equivalent` is that same value expressed on a 0-100 scale, not a second prediction.
2. **Deterministic image signals:** ordinary pixel statistics and dimension rules produce the eight entries in `diagnostics`. Each entry is calculated directly from the uploaded image on every request; none is trained from KonIQ defect labels.

The separate `diagnostics_summary` object contains `flagged_count` and `review_recommended`. `flagged_count` is the number of diagnostics with `passes: false`; `review_recommended` is true when at least one is flagged.

Each entry under `diagnostics` contains:

- `score`: normalized severity from 0 to 1, where higher means more evidence of a problem;
- `passes`: whether the signal remains within its configured review threshold;
- `status`: short human-readable result text;
- `assessment_type`: `deterministic_signal` or `advisory_risk_heuristic`;
- `explanation`: what the result means and how cautiously to read it;
- `measured`: the underlying statistic or dimensions used by that check; and
- `thresholds`: the configured values used to turn the measurement into severity and Pass/Review output.

The UI multiplies `score` by 100 to present a 0-100 severity index and bar, then displays explicit **Pass** or **Review** text. It does not present the value as a percentage. A diagnostic score is not a probability, confidence value, MOS component, or quantity calibrated for comparison with another diagnostic.

A severity of exactly `0` is a valid computed result, not a missing or uninitialized value. Scores are threshold-normalized screening severities: measurements on the safe side of a configured “good” boundary intentionally clamp to zero. The accompanying `measured` values and `thresholds` show the raw evidence and mapping used. For `low_resolution`, zero specifically means both original dimensions meet or exceed their minimums; it does not mean the dimensions are zero or that overall image quality is perfect.

### Eight diagnostic signals and their limits

Six entries are presented as **deterministic detections**: a named, reproducible image measurement crosses a stated threshold. “Detected” here means the implemented signal fired; it does not mean a supervised defect classifier confirmed the cause. The first five are flagged at severity 0.50 or above; low resolution uses the exact stated minimum dimensions.

| API key | Output category | What is measured | Scientific limitation |
| --- | --- | --- | --- |
| `blur` | Detection | Variance of a four-neighbour luminance Laplacian, with mean gradient reported for context | Smooth backgrounds, shallow depth of field, illustrations, and intentional softness can resemble blur. |
| `glare` | Detection | Localized low-saturation pixels above 98.5% luminance, measured over an 8 x 8 grid | White objects, lamps, reflections, and legitimate highlights can trigger the same pixel pattern. |
| `darkness` | Detection | Mean luminance plus the fraction of pixels at or below 15% luminance | Night scenes and low-key photography may be intentionally dark; scene intent is unknown. |
| `overexposure` | Detection | Mean luminance plus the fraction of pixels above 95% luminance; clipped-channel fraction is also reported | Snow, paper, studio backgrounds, and other naturally white areas can resemble clipping. |
| `motion_artifacts` | Detection | Horizontal-versus-vertical gradient imbalance plus excess adjacent-pixel correlation along the smoother axis | Repeated textures, camera perspective, and strongly directional content can affect the signal; it does not prove camera or subject motion. |
| `low_resolution` | Detection | Original pixel width and height against the minimum rule | Dimensions are exact, but pixel count alone cannot establish useful detail, focus, or downstream-task performance. |

The remaining two entries are deliberately named **heuristic risks**, not detections:

| API key | Output category | What the heuristic indicates | Scientific limitation |
| --- | --- | --- | --- |
| `occlusion` | Risk indicator | Uniform, low-gradient tiles on a 6 x 6 grid, with extra weight for the center and extreme tones | There are no object boxes, masks, subject labels, or scene understanding; skies, walls, graphics, and shallow depth of field can resemble obstruction. |
| `poor_framing` | Risk indicator | Excess gradient detail in the outer 6% border and displacement of the gradient-energy centroid | There is no subject detector or aesthetic ground truth, so deliberate off-center composition or edge detail can trigger review. |

Diagnostic analysis applies EXIF orientation, converts the image to RGB, and limits the longest analysis side to 1,024 pixels for predictable computation; the low-resolution rule still uses the original oriented dimensions. All eight thresholds are practical project heuristics. Their defect-level precision and recall have not been measured on a labeled diagnostic benchmark. They are useful for transparent screening and debugging, not as medical, safety-critical, forensic, or photographic ground truth.

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

The annotation fields `c1` to `c5` are distributions of human quality ratings. They are not defect classes. EfficientNet-B0 is therefore trained to predict one overall MOS value only; it is **not** trained to classify blur, glare, darkness, overexposure, motion, low resolution, occlusion, or poor framing.

The named diagnostics returned alongside MOS are separate, deterministic measurements of the uploaded pixels. They do not come from KonIQ defect labels and must not be interpreted as eight additional neural-network predictions. This distinction also means the measured test MAE and RMSE below evaluate only overall MOS regression, not diagnostic accuracy.

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

The selected epoch-5 checkpoint was evaluated once on all 2,015 unseen official test images for its sole learned target, overall MOS:

| Metric | Result |
| --- | ---: |
| MAE | 7.455 MOS points |
| RMSE | 9.674 MOS points |

These are measured regression errors for the one-dimensional MOS prediction. They are not blur/glare/exposure detection accuracy and do not imply that EfficientNet-B0 received defect-classification training.

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
|   |-- inference.py     # Single-image prediction and model-quality logic
|   `-- diagnostics.py   # Eight deterministic signals and advisory risks
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
4. Read the 0-100 learned quality score, suitability decision, learned-model check, and separate resolution rule.
5. Review the eight diagnostic cards. Each shows severity from 0 to 100 (higher is worse), explicit **Pass** or **Review** text, a status and explanation, and the measured pixel statistic or dimensions.

Treat **Occlusion risk** and **Poor framing risk** as advisory prompts, not detected facts. The other six cards report deterministic threshold detections, but can still produce false positives or false negatives for the reasons documented above.

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

The response keeps the learned result and diagnostic measurements in separate fields. An abbreviated shape is:

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
  },
  "diagnostics": {
    "blur": {
      "score": 0.12,
      "passes": true,
      "status": "Pass: sufficient high-frequency detail detected",
      "assessment_type": "deterministic_signal",
      "explanation": "Low Laplacian variance indicates weak fine detail, although intentionally smooth scenes can also score poorly.",
      "measured": {
        "laplacian_variance": 0.002218,
        "mean_gradient": 0.028
      },
      "thresholds": {
        "review_score": 0.5,
        "bad_laplacian_variance": 0.00015,
        "good_laplacian_variance": 0.0025
      }
    },
    "glare": {},
    "darkness": {},
    "overexposure": {},
    "motion_artifacts": {},
    "occlusion": {},
    "poor_framing": {},
    "low_resolution": {}
  },
  "diagnostics_summary": {
    "flagged_count": 0,
    "review_recommended": false
  }
}
```

The seven empty objects above abbreviate the repeated check shape shown for `blur`; actual API responses populate every field for every check. Use `/docs` for the live, complete schema.

The endpoint accepts JPEG, PNG, and WebP files up to 10 MiB. It returns clear 400, 413, 415, or 503 responses for empty/corrupt images, excessive sizes, unsupported types, or a missing checkpoint.

## Current limitations

- EfficientNet-B0 predicts overall subjective quality, not named defects; its measured test MAE is 7.455 and RMSE is 9.674 MOS points.
- The suitability threshold is heuristic because the dataset has no downstream-CV suitability label.
- All eight diagnostics are independent of the neural-network output, and their thresholds have not been validated as defect classifiers on a labeled diagnostic benchmark.
- Diagnostic severity values are screening signals, not probabilities or mutually comparable scores.
- Occlusion and poor-framing outputs are advisory risks because the pipeline has no subject localization, segmentation, scene intent, or aesthetic labels.
- The resolution check is exact for pixel dimensions but does not prove that an image contains useful visual detail.
- All training images in this copy are 512 x 384, so resolution diversity is not learned.
- A five-epoch transfer-learning run is a practical baseline, not an exhaustive hyperparameter study.
- Model weights are not stored in Git; a fresh clone must train a checkpoint or receive one through an appropriate artifact store.
