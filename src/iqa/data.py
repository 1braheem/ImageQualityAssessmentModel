"""KonIQ-10k loading and EfficientNet-B0 input preprocessing.

The annotation CSV is deliberately the only source of samples.  JPEG files that
are present in the image directory but absent from the CSV are never loaded.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CSV_FILENAME = "koniq10k_distributions_sets.csv"
IMAGE_DIRECTORY = "512x384"
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MOS_SCALE = 100.0
OFFICIAL_SPLITS = ("training", "validation", "test")
EXPECTED_SPLIT_COUNTS = {
    "training": 7_058,
    "validation": 1_000,
    "test": 2_015,
}

_SPLIT_ALIASES = {
    "train": "training",
    "training": "training",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}

ImageTransform = Callable[[Image.Image], Tensor]


class KonIQ10kError(RuntimeError):
    """Base exception for actionable KonIQ-10k data errors."""


class DatasetIntegrityError(KonIQ10kError):
    """Raised when the annotation file or local dataset is inconsistent."""


class DatasetImageError(KonIQ10kError):
    """Raised when a labeled image cannot be found or decoded."""


def canonical_split(split: str) -> str:
    """Return the official split name for a supported name or alias."""

    normalized = str(split).strip().lower()
    try:
        return _SPLIT_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(OFFICIAL_SPLITS)
        raise ValueError(f"Unknown split {split!r}; expected one of: {choices}") from exc


def _candidate_dataset_roots(base: Path) -> tuple[Path, ...]:
    """Return supported layouts for an explicit or inferred base path."""

    return (
        base,
        base / "data" / "KonIQ-10k",
        base / "KonIQ-10k",
    )


def _looks_like_dataset_root(path: Path) -> bool:
    return (path / CSV_FILENAME).is_file() and (path / IMAGE_DIRECTORY).is_dir()


def resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    """Resolve a KonIQ-10k directory in either supported project layout.

    Resolution order is an explicit argument, ``KONIQ10K_ROOT``, the current
    directory, and the repository containing this module.  Each base may itself
    be the dataset directory, a project with ``data/KonIQ-10k``, or a project
    with a root-level ``KonIQ-10k`` directory.
    """

    bases: list[Path] = []
    if dataset_root is not None:
        bases.append(Path(dataset_root).expanduser())
    else:
        environment_root = os.environ.get("KONIQ10K_ROOT")
        if environment_root:
            bases.append(Path(environment_root).expanduser())
        bases.extend((Path.cwd(), Path(__file__).resolve().parents[2]))

    checked: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        for candidate in _candidate_dataset_roots(base):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            checked.append(candidate)
            if _looks_like_dataset_root(candidate):
                return candidate

    checked_text = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "Could not locate KonIQ-10k. Expected both "
        f"{CSV_FILENAME!r} and a {IMAGE_DIRECTORY!r} directory in one of:\n"
        f"  - {checked_text}\n"
        "Pass dataset_root explicitly or set KONIQ10K_ROOT."
    )


def _csv_row_numbers(mask: pd.Series, limit: int = 10) -> str:
    """Format zero-based dataframe indexes as human-readable CSV row numbers."""

    rows = (mask[mask].index[:limit] + 2).tolist()
    return ", ".join(str(row) for row in rows)


def _read_annotations(dataset_root: Path) -> pd.DataFrame:
    csv_path = dataset_root / CSV_FILENAME
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DatasetIntegrityError(
            f"Could not read KonIQ-10k annotations at {csv_path}: {exc}"
        ) from exc

    required_columns = {"image_name", "MOS", "set"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise DatasetIntegrityError(
            f"Annotation CSV {csv_path} is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    names = frame["image_name"].astype("string")
    invalid_names = names.isna() | names.str.strip().eq("")
    if invalid_names.any():
        raise DatasetIntegrityError(
            "Annotation CSV contains empty image_name values at CSV row(s): "
            f"{_csv_row_numbers(invalid_names)}"
        )
    frame["image_name"] = names.str.strip()

    duplicate_names = frame["image_name"].duplicated(keep=False)
    if duplicate_names.any():
        examples = frame.loc[duplicate_names, "image_name"].head(10).tolist()
        raise DatasetIntegrityError(
            "Annotation CSV contains duplicate image_name values; examples: "
            + ", ".join(str(name) for name in examples)
        )

    mos = pd.to_numeric(frame["MOS"], errors="coerce")
    invalid_mos = mos.isna() | ~np.isfinite(mos) | (mos < 0.0) | (mos > MOS_SCALE)
    if invalid_mos.any():
        raise DatasetIntegrityError(
            f"MOS must be finite and between 0 and {MOS_SCALE:g}; invalid CSV "
            f"row(s): {_csv_row_numbers(invalid_mos)}"
        )
    frame["MOS"] = mos.astype("float32")

    split_names = frame["set"].astype("string").str.strip().str.lower()
    canonical_names = split_names.map(_SPLIT_ALIASES)
    invalid_splits = canonical_names.isna()
    if invalid_splits.any():
        examples = sorted(set(split_names[invalid_splits].astype(str).tolist()))
        raise DatasetIntegrityError(
            "Annotation CSV contains unsupported set values: " + ", ".join(examples)
        )
    frame["set"] = canonical_names

    return frame


def get_split_counts(dataset_root: str | Path | None = None) -> dict[str, int]:
    """Return sample counts read from the official CSV."""

    root = resolve_dataset_root(dataset_root)
    annotations = _read_annotations(root)
    counts = annotations["set"].value_counts()
    return {split: int(counts.get(split, 0)) for split in OFFICIAL_SPLITS}


def validate_official_split_counts(
    dataset_root: str | Path | None = None,
) -> dict[str, int]:
    """Validate the CSV against the published KonIQ-10k split sizes."""

    counts = get_split_counts(dataset_root)
    if counts != EXPECTED_SPLIT_COUNTS:
        raise DatasetIntegrityError(
            "Unexpected official split counts. "
            f"Expected {EXPECTED_SPLIT_COUNTS}, found {counts}."
        )
    return counts


def build_transform(split: str = "test") -> transforms.Compose:
    """Build 224x224 ImageNet preprocessing for an official dataset split.

    A horizontal flip is the only train-only augmentation.  It changes neither
    compression, sharpness, color, nor exposure, so it does not fabricate a new
    perceptual-quality label.
    """

    split = canonical_split(split)
    operations: list[Callable] = [
        transforms.Resize(
            IMAGE_SIZE,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    ]
    if split == "training":
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.extend(
        (
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
    )
    return transforms.Compose(operations)


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """Decode one image and detach an RGB copy from its file handle."""

    path = Path(image_path)
    if not path.is_file():
        raise DatasetImageError(f"Labeled image file is missing: {path}")
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DatasetImageError(f"Could not decode labeled image {path}: {exc}") from exc


def preprocess_image(
    image: Image.Image | str | Path,
    transform: ImageTransform | None = None,
) -> Tensor:
    """Force an image to RGB and apply deterministic inference preprocessing."""

    rgb_image = load_rgb_image(image) if isinstance(image, (str, Path)) else image.convert("RGB")
    selected_transform = transform or build_transform("test")
    tensor = selected_transform(rgb_image)
    if not isinstance(tensor, Tensor):
        raise TypeError("The image transform must return a torch.Tensor")
    return tensor


class KonIQ10kDataset(Dataset[dict[str, Tensor | str]]):
    """CSV-driven KonIQ-10k regression dataset.

    Each item contains ``image`` (normalized ``3x224x224`` tensor), ``target``
    (scalar ``MOS / 100``), ``mos`` (raw scalar MOS), and ``image_name``.
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        split: str = "training",
        transform: ImageTransform | None = None,
        *,
        validate_files: bool = True,
        verify_images: bool = False,
    ) -> None:
        self.root = resolve_dataset_root(dataset_root)
        self.images_dir = self.root / IMAGE_DIRECTORY
        self.split = canonical_split(split)
        self.transform = transform or build_transform(self.split)

        annotations = _read_annotations(self.root)
        self.annotations = annotations.loc[
            annotations["set"].eq(self.split), ["image_name", "MOS", "set"]
        ].reset_index(drop=True)
        if self.annotations.empty:
            raise DatasetIntegrityError(
                f"Annotation CSV has no rows for the {self.split!r} split"
            )

        if validate_files or verify_images:
            self._validate_labeled_files(verify_images=verify_images)

    def _validate_labeled_files(self, *, verify_images: bool) -> None:
        paths = [self.images_dir / name for name in self.annotations["image_name"]]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            examples = "\n  - ".join(str(path) for path in missing[:10])
            suffix = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
            raise DatasetIntegrityError(
                f"{len(missing)} labeled image(s) are missing from the "
                f"{self.split!r} split:\n  - {examples}{suffix}"
            )

        if not verify_images:
            return

        corrupt: list[tuple[Path, Exception]] = []
        for path in paths:
            try:
                # Conversion forces pixel decoding and also checks grayscale inputs.
                load_rgb_image(path).load()
            except DatasetImageError as exc:
                corrupt.append((path, exc))
                if len(corrupt) == 10:
                    break
        if corrupt:
            examples = "\n  - ".join(f"{path}: {error}" for path, error in corrupt)
            raise DatasetIntegrityError(
                "One or more labeled images could not be decoded:\n  - " + examples
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.annotations.iloc[index]
        image_name = str(row["image_name"])
        image_path = self.images_dir / image_name
        try:
            image = load_rgb_image(image_path)
        except DatasetImageError as exc:
            raise DatasetImageError(
                f"Failed to load {self.split!r} sample {index} ({image_name}): {exc}"
            ) from exc

        image_tensor = self.transform(image)
        if not isinstance(image_tensor, Tensor):
            raise TypeError(
                f"Transform for {image_name} returned {type(image_tensor).__name__}, "
                "not torch.Tensor"
            )

        mos = torch.tensor(float(row["MOS"]), dtype=torch.float32)
        return {
            "image": image_tensor,
            "target": mos / MOS_SCALE,
            "mos": mos,
            "image_name": image_name,
        }


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable data loading."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python inside a PyTorch DataLoader worker."""

    del worker_id  # PyTorch has already incorporated it into initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset_root: str | Path | None = None,
    split: str = "training",
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    shuffle: bool | None = None,
    validate_files: bool = True,
    verify_images: bool = False,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Create a reproducible DataLoader for one official split."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    split = canonical_split(split)
    seed_everything(seed)
    dataset = KonIQ10kDataset(
        dataset_root=dataset_root,
        split=split,
        validate_files=validate_files,
        verify_images=verify_images,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_shuffle = split == "training" if shuffle is None else shuffle
    selected_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    selected_persistent_workers = (
        num_workers > 0 if persistent_workers is None else persistent_workers
    )
    if selected_persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=selected_shuffle,
        num_workers=num_workers,
        pin_memory=selected_pin_memory,
        persistent_workers=selected_persistent_workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def create_dataloaders(
    dataset_root: str | Path | None = None,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    splits: Sequence[str] = OFFICIAL_SPLITS,
    validate_files: bool = True,
    verify_images: bool = False,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """Create loaders keyed by canonical split name.

    Separate, deterministically offset generators prevent iteration of one split
    from changing the sample or augmentation order of another split.
    """

    loaders: dict[str, DataLoader] = {}
    for offset, requested_split in enumerate(splits):
        split = canonical_split(requested_split)
        if split in loaders:
            raise ValueError(f"Duplicate requested split: {split}")
        loaders[split] = create_dataloader(
            dataset_root=dataset_root,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed + offset,
            validate_files=validate_files,
            verify_images=verify_images,
            pin_memory=pin_memory,
        )
    return loaders


def _validate_one_batch(loader: DataLoader) -> dict[str, object]:
    batch = next(iter(loader))
    images = batch["image"]
    targets = batch["target"]
    raw_mos = batch["mos"]
    if not isinstance(images, Tensor) or images.ndim != 4:
        raise DatasetIntegrityError("DataLoader did not produce a batched image tensor")
    if tuple(images.shape[1:]) != (3, *IMAGE_SIZE):
        raise DatasetIntegrityError(
            f"Expected image batches shaped [N, 3, 224, 224], found {tuple(images.shape)}"
        )
    if not torch.isfinite(images).all() or not torch.isfinite(targets).all():
        raise DatasetIntegrityError("A validation batch contains NaN or infinite values")
    if torch.any(targets < 0.0) or torch.any(targets > 1.0):
        raise DatasetIntegrityError("Normalized MOS target is outside [0, 1]")
    return {
        "image_shape": tuple(images.shape),
        "target_range": (float(targets.min()), float(targets.max())),
        "mos_range": (float(raw_mos.min()), float(raw_mos.max())),
        "first_image": batch["image_name"][0],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate official counts and load one preprocessed batch without training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="KonIQ-10k directory or project directory (auto-detected by default)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Decode every labeled image in all splits (slower)",
    )
    args = parser.parse_args(argv)

    try:
        root = resolve_dataset_root(args.data_root)
        counts = validate_official_split_counts(root)
        print(f"Dataset root: {root}")
        print("Official split counts:")
        for split in OFFICIAL_SPLITS:
            print(f"  {split}: {counts[split]}")

        if args.verify_images:
            for split in OFFICIAL_SPLITS:
                KonIQ10kDataset(root, split, verify_images=True)
            print("All labeled images decoded successfully.")

        loader = create_dataloader(
            root,
            "training",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        summary = _validate_one_batch(loader)
        print("Batch validation:")
        print(f"  image shape: {summary['image_shape']}")
        print(f"  normalized target range: {summary['target_range']}")
        print(f"  raw MOS range: {summary['mos_range']}")
        print(f"  first image: {summary['first_image']}")
        print("Dataset and preprocessing validation passed.")
    except (FileNotFoundError, KonIQ10kError, ValueError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    return 0


__all__ = [
    "CSV_FILENAME",
    "DatasetImageError",
    "DatasetIntegrityError",
    "EXPECTED_SPLIT_COUNTS",
    "IMAGE_DIRECTORY",
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "KonIQ10kDataset",
    "KonIQ10kError",
    "MOS_SCALE",
    "OFFICIAL_SPLITS",
    "build_transform",
    "canonical_split",
    "create_dataloader",
    "create_dataloaders",
    "get_split_counts",
    "load_rgb_image",
    "preprocess_image",
    "resolve_dataset_root",
    "seed_everything",
    "seed_worker",
    "validate_official_split_counts",
]


if __name__ == "__main__":
    raise SystemExit(main())
