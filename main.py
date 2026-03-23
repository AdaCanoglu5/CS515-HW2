"""Main entry point for training, testing, and FLOPs reporting."""

from __future__ import annotations

import csv
import random
import ssl
from pathlib import Path

import numpy as np
import torch

from models.CNN import MNIST_CNN, SimpleCNN
from models.MLP import MLP
from models.ResNet import BasicBlock, ResNet
from models.VGG import VGG
from models.mobilenet import MobileNetV2
from models.pretrained_resnet import build_pretrained_resnet18
from parameters import ExperimentConfig, get_params
from test import run_test
from train import ensure_output_dirs, run_training


ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """Seed all major random number generators for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params: ExperimentConfig) -> torch.nn.Module:
    """Build the requested model while preserving the existing project flow."""

    model_name = params["model"]
    dataset = params["dataset"]
    nc = params["num_classes"]

    if params["pretrained"]:
        if dataset != "cifar10" or model_name != "resnet":
            raise ValueError("Pretrained transfer learning is only supported for CIFAR-10 ResNet runs.")
        return build_pretrained_resnet18(
            num_classes=nc,
            transfer_mode=params["transfer_mode"],
            freeze_backbone=params["freeze_backbone"],
        )

    if model_name == "mlp":
        return MLP(
            input_size=params["feature_size"],
            hidden_sizes=params["hidden_sizes"],
            num_classes=nc,
            dropout=params["dropout"],
        )

    if model_name == "cnn":
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10 with vgg.")
        return VGG(dept=params["vgg_depth"], num_class=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10 with resnet.")
        return ResNet(BasicBlock, params["resnet_layers"], num_classes=nc)

    if model_name == "mobilenet":
        if dataset == "mnist":
            raise ValueError("MobileNetV2 is designed for 3-channel images; use cifar10 with mobilenet.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")


def write_flops_summary(params: ExperimentConfig, device: torch.device) -> Path:
    """Compute and save one FLOPs summary CSV for the required architectures."""

    try:
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        raise ImportError("ptflops is required for --mode flops. Please install it first.") from exc

    output_path = Path(params["flops_summary_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    models_to_profile = [
        ("ResNet", ResNet(BasicBlock, params["resnet_layers"], num_classes=params["num_classes"]), (3, 32, 32)),
        ("SimpleCNN", SimpleCNN(num_classes=params["num_classes"]), (3, 32, 32)),
        ("MobileNetV2", MobileNetV2(num_classes=params["num_classes"]), (3, 32, 32)),
    ]

    for arch_name, model, input_res in models_to_profile:
        model = model.to(device)
        macs, params_count = get_model_complexity_info(
            model,
            input_res,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        macs_value = int(macs)
        rows.append(
            {
                "arch_name": arch_name,
                "input_res": "x".join(str(dim) for dim in input_res),
                "flops": int(macs_value * 2),
                "macs": macs_value,
                "params": int(params_count),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["arch_name", "input_res", "flops", "macs", "params"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved FLOPs summary to {output_path}")
    return output_path


def main() -> None:
    """Parse config, build the requested model, and execute the selected mode."""

    params = get_params()
    ensure_output_dirs(params)
    set_seed(params["seed"])

    device = torch.device(
        params["device"]
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Run: {params['run_name']}")
    print(f"Device: {device}")

    if params["mode"] == "flops":
        write_flops_summary(params, device)
        return

    model = build_model(params).to(device)

    train_summary: dict[str, float | int | str] | None = None
    if params["mode"] in ("train", "both"):
        train_summary = run_training(model, params, device)

    if params["mode"] in ("test", "both"):
        run_test(model, params, device, train_summary=train_summary)


if __name__ == "__main__":
    main()
