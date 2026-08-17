"""Train EfficientNet-B0 on the official KonIQ-10k train/validation splits."""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import KonIQ10kDataset, build_transform
from .metrics import RegressionAccumulator
from .model import IQAModel, load_iqa_checkpoint, make_checkpoint


DEFAULT_CHECKPOINT = Path("models/efficientnet_b0_koniq10k.pt")
MOS_SCALE = 100.0


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    mae: float
    rmse: float


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, then Apple MPS, then CPU unless a device is requested."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_training_loaders(
    dataset_root: str | Path | None,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """Create loaders using only the official training and validation rows."""

    training_dataset = KonIQ10kDataset(
        dataset_root=dataset_root,
        split="training",
        transform=build_transform("training"),
    )
    validation_dataset = KonIQ10kDataset(
        dataset_root=dataset_root,
        split="validation",
        transform=build_transform("validation"),
    )
    generator = torch.Generator().manual_seed(seed)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker if workers else None,
        "persistent_workers": workers > 0,
    }
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common,
    )
    return training_loader, validation_loader


def _keep_frozen_feature_blocks_in_eval(model: IQAModel) -> None:
    """Prevent running-stat updates in feature blocks that are fully frozen."""

    for block in model.network.features.children():
        parameters = tuple(block.parameters())
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            block.eval()


def run_epoch(
    model: IQAModel,
    loader: Iterable[Mapping[str, Any]],
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Run one training or validation epoch and return MOS-point metrics."""

    training = optimizer is not None
    if training:
        model.train()
        _keep_frozen_feature_blocks_in_eval(model)
    else:
        model.eval()

    total_loss = 0.0
    observation_count = 0
    metrics = RegressionAccumulator()

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets: Tensor = batch["target"].to(device, non_blocking=True).float().view(-1)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(images)
            loss = criterion(predictions, targets)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = targets.numel()
            total_loss += loss.detach().item() * batch_size
            observation_count += batch_size
            metrics.update(predictions, targets)

    if observation_count == 0:
        raise RuntimeError("the data loader did not yield any samples")

    point_metrics = metrics.compute(scale=MOS_SCALE)
    return EpochMetrics(
        loss=total_loss / observation_count,
        mae=point_metrics["mae"],
        rmse=point_metrics["rmse"],
    )


def _save_best_checkpoint(
    path: Path,
    model: IQAModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    phase: str,
    training_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = make_checkpoint(
        model,
        epoch=epoch,
        phase=phase,
        metrics={
            "train_loss": training_metrics.loss,
            "train_mae_mos": training_metrics.mae,
            "train_rmse_mos": training_metrics.rmse,
            "val_loss": validation_metrics.loss,
            "val_mae_mos": validation_metrics.mae,
            "val_rmse_mos": validation_metrics.rmse,
        },
        optimizer=optimizer,
        seed=seed,
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def train(args: argparse.Namespace) -> Path:
    """Execute head training and optional final-block fine-tuning."""

    if args.head_epochs < 0 or args.finetune_epochs < 0:
        raise ValueError("epoch counts cannot be negative")
    if args.head_epochs + args.finetune_epochs == 0:
        raise ValueError("at least one training epoch is required")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    set_deterministic_seed(args.seed)
    device = select_device(args.device)
    print(f"device={device}")

    training_loader, validation_loader = build_training_loaders(
        args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    print(
        f"official splits: training={len(training_loader.dataset)} "
        f"validation={len(validation_loader.dataset)}"
    )

    model = IQAModel(pretrained=args.pretrained, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    checkpoint_path = Path(args.checkpoint).expanduser()
    best_validation_loss = float("inf")
    best_checkpoint_saved = False
    global_epoch = 0

    phases = (
        ("head", args.head_epochs, args.head_lr),
        ("finetune", args.finetune_epochs, args.finetune_lr),
    )
    for phase, epochs, learning_rate in phases:
        if epochs == 0:
            continue

        if phase == "head":
            model.freeze_backbone()
        else:
            if best_checkpoint_saved:
                model, _ = load_iqa_checkpoint(
                    checkpoint_path, map_location=device
                )
                print("restored_best_head_checkpoint=true")
            model.unfreeze_last_feature_blocks(args.unfreeze_blocks)

        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(
            f"phase={phase} epochs={epochs} lr={learning_rate:g} "
            f"trainable_parameters={trainable_parameters:,}"
        )

        for phase_epoch in range(1, epochs + 1):
            global_epoch += 1
            training_metrics = run_epoch(
                model,
                training_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )
            validation_metrics = run_epoch(
                model,
                validation_loader,
                criterion=criterion,
                device=device,
            )
            improved = validation_metrics.loss < best_validation_loss
            if improved:
                best_validation_loss = validation_metrics.loss
                _save_best_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch=global_epoch,
                    phase=phase,
                    training_metrics=training_metrics,
                    validation_metrics=validation_metrics,
                    seed=args.seed,
                )
                best_checkpoint_saved = True

            marker = " saved_best" if improved else ""
            print(
                f"epoch={global_epoch} phase={phase} phase_epoch={phase_epoch}/{epochs} "
                f"train_loss={training_metrics.loss:.6f} "
                f"val_loss={validation_metrics.loss:.6f} "
                f"val_mae={validation_metrics.mae:.3f}_MOS "
                f"val_rmse={validation_metrics.rmse:.3f}_MOS{marker}"
            )

    print(f"best_checkpoint={checkpoint_path.resolve()}")
    return checkpoint_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet-B0 quality regressor on KonIQ-10k."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="KonIQ-10k directory (auto-detected under data/ when omitted).",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the backbone from torchvision ImageNet weights.",
    )
    return parser


def main() -> None:
    train(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
