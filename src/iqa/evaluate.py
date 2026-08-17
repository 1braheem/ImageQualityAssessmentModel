"""Evaluate a trained IQA checkpoint on the official KonIQ-10k test split."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .data import create_dataloader
from .metrics import RegressionAccumulator
from .model import load_iqa_checkpoint, normalized_to_mos
from .train import DEFAULT_CHECKPOINT, select_device, set_deterministic_seed


@dataclass(frozen=True)
class PredictionRecord:
    image_name: str
    actual_mos: float
    predicted_mos: float
    absolute_error: float


def evaluate(
    checkpoint_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    batch_size: int = 64,
    workers: int = 0,
    device_name: str = "auto",
    seed: int = 42,
) -> tuple[dict[str, float], list[PredictionRecord], dict[str, Any]]:
    """Return MOS-point test metrics, per-image predictions, and checkpoint data."""

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if workers < 0:
        raise ValueError("workers cannot be negative")

    set_deterministic_seed(seed)
    device = select_device(device_name)
    model, checkpoint = load_iqa_checkpoint(checkpoint_path, map_location=device)
    test_loader = create_dataloader(
        dataset_root=dataset_root,
        split="test",
        batch_size=batch_size,
        num_workers=workers,
        seed=seed,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    accumulator = RegressionAccumulator()
    records: list[PredictionRecord] = []
    with torch.inference_mode():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            predicted_mos = normalized_to_mos(model(images)).detach().cpu().float()
            actual_mos = batch["mos"].detach().cpu().float().view(-1)
            accumulator.update(predicted_mos, actual_mos)

            for image_name, actual, predicted in zip(
                batch["image_name"],
                actual_mos.tolist(),
                predicted_mos.tolist(),
                strict=True,
            ):
                records.append(
                    PredictionRecord(
                        image_name=str(image_name),
                        actual_mos=float(actual),
                        predicted_mos=float(predicted),
                        absolute_error=abs(float(predicted) - float(actual)),
                    )
                )

    return accumulator.compute(), records, checkpoint


def _print_records(title: str, records: list[PredictionRecord]) -> None:
    print(title)
    for record in records:
        print(
            f"  image={record.image_name} actual={record.actual_mos:.3f} "
            f"predicted={record.predicted_mos:.3f} "
            f"absolute_error={record.absolute_error:.3f}"
        )


def _write_predictions(
    output_path: Path,
    records: list[PredictionRecord],
    metrics: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(PredictionRecord.__annotations__))
            writer.writeheader()
            writer.writerows(rows)
    elif suffix == ".json":
        payload = {"metrics_mos": metrics, "predictions": rows}
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        raise ValueError("prediction output must end in .csv or .json")


def run_cli(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    print(f"device={device}")
    metrics, records, checkpoint = evaluate(
        args.checkpoint,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        device_name=args.device,
        seed=args.seed,
    )

    checkpoint_metrics = checkpoint.get("metrics", {})
    print(f"checkpoint={Path(args.checkpoint).expanduser().resolve()}")
    print(
        f"selected_epoch={checkpoint.get('epoch', 'unknown')} "
        f"phase={checkpoint.get('phase', 'unknown')} "
        f"validation_loss={checkpoint_metrics.get('val_loss', 'unknown')}"
    )
    print(f"official_test_samples={len(records)}")
    print(f"test_mae={metrics['mae']:.3f}_MOS")
    print(f"test_rmse={metrics['rmse']:.3f}_MOS")

    example_count = min(args.examples, len(records))
    worst_count = min(args.worst, len(records))
    if example_count:
        _print_records("example_predictions:", records[:example_count])
    if worst_count:
        worst_records = sorted(
            records, key=lambda record: record.absolute_error, reverse=True
        )[:worst_count]
        _print_records("worst_predictions:", worst_records)

    if args.predictions_out is not None:
        output_path = Path(args.predictions_out).expanduser()
        _write_predictions(output_path, records, metrics)
        print(f"predictions_written={output_path.resolve()}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the best IQA checkpoint on the official test split."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="KonIQ-10k directory (auto-detected under data/ when omitted).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Optional ignored artifact path ending in .csv or .json.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.examples < 0 or args.worst < 0:
        raise ValueError("examples and worst counts cannot be negative")
    run_cli(args)


if __name__ == "__main__":
    main()
