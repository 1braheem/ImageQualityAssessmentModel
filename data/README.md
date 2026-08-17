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

The current local download may remain at the repository root during setup because that path is also ignored. It can be moved under `data/` before the preprocessing stage.
