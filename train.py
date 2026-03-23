"""Training utilities for scratch, transfer, and distillation experiments."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.ResNet import BasicBlock, ResNet
from parameters import ExperimentConfig


RUN_CSV_COLUMNS = [
    "run_name",
    "epoch",
    "split",
    "loss",
    "accuracy",
    "lr",
    "seed",
    "model",
    "dataset",
    "train_mode",
    "teacher_run",
    "label_smoothing",
    "temperature",
    "alpha",
    "input_size",
]

SUMMARY_COLUMNS = [
    "run_name",
    "model",
    "dataset",
    "train_mode",
    "teacher_run",
    "checkpoint_path",
    "label_smoothing",
    "temperature",
    "alpha",
    "input_size",
    "best_epoch",
    "best_val_loss",
    "best_val_accuracy",
    "final_train_loss",
    "final_train_accuracy",
    "final_val_loss",
    "final_val_accuracy",
    "test_loss",
    "test_accuracy",
    "seed",
]


def ensure_output_dirs(params: ExperimentConfig) -> None:
    """Create the required artifact directories."""

    Path(params["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(params["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(params["run_dir"]).mkdir(parents=True, exist_ok=True)


def get_teacher_run_name(params: ExperimentConfig) -> str:
    """Infer the teacher run name from the checkpoint path if present."""

    teacher_checkpoint = params["teacher_checkpoint"]
    return Path(teacher_checkpoint).stem if teacher_checkpoint else ""


def get_train_mode(params: ExperimentConfig) -> str:
    """Map config flags to the compact train mode label used in logs."""

    if params["distill_mode"] == "kd":
        return "kd"
    if params["distill_mode"] == "teacher_trueclass_prob":
        return "teacher_prob"
    if params["transfer_mode"] != "none" or params["pretrained"]:
        return "transfer"
    return "scratch"


def _resolve_mean_std(params: ExperimentConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if params["transfer_mode"] == "resize_freeze":
        return params["imagenet_mean"], params["imagenet_std"]
    return params["mean"], params["std"]


def get_transforms(params: ExperimentConfig, train: bool = True) -> transforms.Compose:
    """Build dataset transforms while keeping transfer branches minimal."""

    mean, std = _resolve_mean_std(params)

    if params["dataset"] == "mnist":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    if params["transfer_mode"] == "resize_freeze":
        steps: list[Any] = [transforms.Resize((params["input_size"], params["input_size"]))]
        if train:
            steps.append(transforms.RandomHorizontalFlip())
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return transforms.Compose(steps)

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(params["input_size"], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _build_datasets(params: ExperimentConfig) -> tuple[Subset, Subset, torch.utils.data.Dataset]:
    """Create reproducible train/val splits plus the held-out test dataset."""

    train_tf = get_transforms(params, train=True)
    eval_tf = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        train_aug_ds = datasets.MNIST(params["data_dir"], train=True, download=True, transform=train_tf)
        train_eval_ds = datasets.MNIST(params["data_dir"], train=True, download=True, transform=eval_tf)
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=eval_tf)
    else:
        train_aug_ds = datasets.CIFAR10(params["data_dir"], train=True, download=True, transform=train_tf)
        train_eval_ds = datasets.CIFAR10(params["data_dir"], train=True, download=True, transform=eval_tf)
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=eval_tf)

    total_items = len(train_aug_ds)
    val_size = int(total_items * params["val_split"])
    if val_size <= 0 or val_size >= total_items:
        raise ValueError("val_split must leave at least one sample in both train and val sets.")

    generator = torch.Generator().manual_seed(params["seed"])
    indices = torch.randperm(total_items, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_ds = Subset(train_aug_ds, train_indices)
    val_ds = Subset(train_eval_ds, val_indices)
    return train_ds, val_ds, test_ds


def get_loaders(params: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train, validation, and test loaders."""

    train_ds, val_ds, test_ds = _build_datasets(params)
    loader_kwargs = {
        "batch_size": params["batch_size"],
        "num_workers": params["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def reset_run_csv(params: ExperimentConfig) -> None:
    """Create or overwrite the per-run CSV with a single header row."""

    csv_path = Path(params["csv_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_CSV_COLUMNS)
        writer.writeheader()


def append_run_row(params: ExperimentConfig, row: dict[str, Any]) -> None:
    """Append one metrics row to the current run CSV."""

    if not params["save_csv"]:
        return
    with Path(params["csv_path"]).open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_CSV_COLUMNS)
        writer.writerow(row)


def upsert_summary_row(params: ExperimentConfig, row: dict[str, Any]) -> None:
    """Replace or append a single run row in the compact summary CSV."""

    summary_path = Path(params["summary_path"])
    existing_rows: list[dict[str, Any]] = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_rows = list(csv.DictReader(csv_file))

    filtered_rows = [existing for existing in existing_rows if existing.get("run_name") != params["run_name"]]
    filtered_rows.append({column: row.get(column, "") for column in SUMMARY_COLUMNS})

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(filtered_rows)


def build_log_row(
    params: ExperimentConfig,
    epoch: int,
    split: str,
    loss: float,
    accuracy: float,
    lr: float,
) -> dict[str, Any]:
    """Build one compact CSV metrics row."""

    temperature = ""
    alpha = ""
    if params["distill_mode"] == "kd":
        temperature = params["temperature"]
        alpha = params["alpha"]
    elif params["distill_mode"] == "teacher_trueclass_prob":
        temperature = 1.0

    return {
        "run_name": params["run_name"],
        "epoch": epoch,
        "split": split,
        "loss": f"{loss:.6f}",
        "accuracy": f"{accuracy:.6f}",
        "lr": f"{lr:.8f}",
        "seed": params["seed"],
        "model": params["model"],
        "dataset": params["dataset"],
        "train_mode": get_train_mode(params),
        "teacher_run": get_teacher_run_name(params),
        "label_smoothing": params["label_smoothing"],
        "temperature": temperature,
        "alpha": alpha,
        "input_size": params["input_size"],
    }


def load_teacher_model(params: ExperimentConfig, device: torch.device) -> nn.Module:
    """Load the fixed ResNet teacher used by the distillation experiments."""

    teacher_checkpoint = params["teacher_checkpoint"]
    if not teacher_checkpoint:
        raise ValueError("A teacher checkpoint is required for distillation modes.")

    teacher = ResNet(BasicBlock, params["resnet_layers"], num_classes=params["num_classes"]).to(device)
    state_dict = torch.load(teacher_checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute the standard compact KD objective."""

    hard_loss = criterion(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    )
    return (1.0 - alpha) * hard_loss + alpha * (temperature ** 2) * soft_loss


def teacher_trueclass_probability_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Use the teacher's true-class probability to define soft targets."""

    teacher_probs = F.softmax(teacher_logits, dim=1)
    true_class_prob = teacher_probs.gather(1, labels.unsqueeze(1))
    off_class_prob = (1.0 - true_class_prob) / (student_logits.size(1) - 1)
    soft_targets = off_class_prob.expand(-1, student_logits.size(1)).clone()
    soft_targets.scatter_(1, labels.unsqueeze(1), true_class_prob)
    return -(soft_targets * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    params: ExperimentConfig,
    teacher_model: nn.Module | None = None,
) -> tuple[float, float]:
    """Train the model for one epoch under the configured supervision mode."""

    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        student_logits = model(imgs)

        if params["distill_mode"] == "kd":
            if teacher_model is None:
                raise ValueError("teacher_model is required for KD training.")
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)
            loss = kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=params["temperature"],
                alpha=params["alpha"],
                criterion=criterion,
            )
        elif params["distill_mode"] == "teacher_trueclass_prob":
            if teacher_model is None:
                raise ValueError("teacher_model is required for teacher_trueclass_prob training.")
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)
            loss = teacher_trueclass_probability_loss(student_logits, teacher_logits, labels)
        else:
            loss = criterion(student_logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct += student_logits.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

        if params["log_interval"] and (batch_idx + 1) % params["log_interval"] == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] loss={total_loss / n:.4f} acc={correct / n:.4f}")

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate loss and accuracy on a validation or test split."""

    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.detach().item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

    return total_loss / n, correct / n


def run_training(
    model: nn.Module,
    params: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Run the main training loop and save only the best validation checkpoint."""

    ensure_output_dirs(params)
    train_loader, val_loader, _ = get_loaders(params)
    if params["save_csv"]:
        reset_run_csv(params)

    train_criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss()
    teacher_model = load_teacher_model(params, device) if params["distill_mode"] != "none" else None

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = -1.0
    best_epoch = 0
    best_val_loss = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    final_train_loss = 0.0
    final_train_acc = 0.0
    final_val_loss = 0.0
    final_val_acc = 0.0

    for epoch in range(1, params["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=train_criterion,
            device=device,
            params=params,
            teacher_model=teacher_model,
        )
        val_loss, val_acc = evaluate(model, val_loader, eval_criterion, device)
        scheduler.step()

        append_run_row(params, build_log_row(params, epoch, "train", train_loss, train_acc, current_lr))
        append_run_row(params, build_log_row(params, epoch, "val", val_loss, val_acc, current_lr))

        print(
            f"Epoch {epoch:02d}/{params['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        final_train_loss = train_loss
        final_train_acc = train_acc
        final_val_loss = val_loss
        final_val_acc = val_acc

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params["save_path"])

    model.load_state_dict(best_weights)
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_acc,
        "final_train_loss": final_train_loss,
        "final_train_accuracy": final_train_acc,
        "final_val_loss": final_val_loss,
        "final_val_accuracy": final_val_acc,
    }
