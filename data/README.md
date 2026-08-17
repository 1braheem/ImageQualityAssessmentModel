# Dataset location

The project uses KonIQ-10k for overall image-quality regression. Dataset files are local-only and must not be committed.

Place the extracted files in this layout:

```text
data/
`-- KonIQ-10k/
    |-- 512x384/
    |   |-- 10004473376.jpg
    |   `-- ...
    `-- koniq10k_distributions_sets.csv
```

The annotation CSV is the source of truth. Only filenames listed in it will be loaded; unlisted JPEGs will be ignored. The expected columns are `image_name`, `MOS`, and `set`, with optional rating-distribution columns also present in the official file.

The loader also recognizes a root-level `KonIQ-10k/` directory and the `KONIQ10K_ROOT` environment variable. Validate the split counts and one preprocessed batch from the repository root with:

```bash
.venv/bin/python -m src.iqa.data --batch-size 4
```

Add `--verify-images` to decode all 10,073 labeled files.
