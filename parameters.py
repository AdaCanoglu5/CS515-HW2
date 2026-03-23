"""Argument parsing and lightweight experiment configuration."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Container for CLI configuration with dict-style compatibility."""

    dataset: str
    data_dir: str
    num_workers: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    imagenet_mean: tuple[float, ...]
    imagenet_std: tuple[float, ...]
    model: str
    feature_size: int
    input_size: int
    hidden_sizes: list[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10
    dropout: float = 0.3
    vgg_depth: str = "16"
    resnet_layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    save_path: str = "best_model.pth"
    log_interval: int = 0
    mode: str = "both"
    run_name: str = "default_run"
    pretrained: bool = False
    transfer_mode: str = "none"
    freeze_backbone: bool = False
    teacher_checkpoint: str = ""
    distill_mode: str = "none"
    temperature: float = 4.0
    alpha: float = 0.7
    label_smoothing: float = 0.0
    output_dir: str = "artifacts"
    save_csv: bool = True
    csv_path: str = "artifacts/runs/default_run.csv"
    summary_path: str = "artifacts/summary.csv"
    flops_summary_path: str = "artifacts/flops_summary.csv"
    checkpoint_dir: str = "artifacts/checkpoints"
    run_dir: str = "artifacts/runs"
    val_split: float = 0.1

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_params() -> ExperimentConfig:
    """Parse CLI arguments into a small dataclass config."""

    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    parser.add_argument("--mode", choices=["train", "test", "both", "flops"], default="both")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="mlp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--input_size", type=int, default=32, help="Image resolution for CIFAR-style experiments.")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--transfer_mode",
        choices=["none", "resize_freeze", "modify_finetune"],
        default="none",
    )
    parser.add_argument("--freeze_backbone", type=_str2bool, default=False)
    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument(
        "--distill_mode",
        choices=["none", "kd", "teacher_trueclass_prob"],
        default="none",
    )
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--save_csv", type=_str2bool, default=True)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    parser.add_argument(
        "--resnet_layers",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)",
    )

    args = parser.parse_args()

    if args.dataset == "mnist":
        feature_size = 784
        input_size = 28
        mean, std = (0.1307,), (0.3081,)
    else:
        feature_size = 3072
        input_size = args.input_size
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    if args.transfer_mode == "resize_freeze":
        input_size = 224
        args.pretrained = True
    elif args.transfer_mode == "modify_finetune":
        input_size = 32
        args.pretrained = True

    run_name = args.run_name or f"{args.model}_{args.dataset}"
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    run_dir = output_dir / "runs"

    return ExperimentConfig(
        dataset=args.dataset,
        data_dir="./data",
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        imagenet_mean=(0.485, 0.456, 0.406),
        imagenet_std=(0.229, 0.224, 0.225),
        model=args.model,
        feature_size=feature_size,
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        num_classes=10,
        dropout=0.3,
        vgg_depth=args.vgg_depth,
        resnet_layers=args.resnet_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        seed=args.seed,
        device=args.device,
        save_path=str(checkpoint_dir / f"{run_name}.pth"),
        log_interval=args.log_interval,
        mode=args.mode,
        run_name=run_name,
        pretrained=args.pretrained,
        transfer_mode=args.transfer_mode,
        freeze_backbone=args.freeze_backbone,
        teacher_checkpoint=args.teacher_checkpoint,
        distill_mode=args.distill_mode,
        temperature=args.temperature,
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
        output_dir=str(output_dir),
        save_csv=args.save_csv,
        csv_path=str(run_dir / f"{run_name}.csv"),
        summary_path=str(output_dir / "summary.csv"),
        flops_summary_path=str(output_dir / "flops_summary.csv"),
        checkpoint_dir=str(checkpoint_dir),
        run_dir=str(run_dir),
        val_split=args.val_split,
    )
