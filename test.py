"""Evaluation utilities for the compact CIFAR/MNIST training pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from parameters import ExperimentConfig
from train import append_run_row, build_log_row, get_loaders, get_teacher_run_name, get_train_mode, upsert_summary_row


@torch.no_grad()
def run_test(
    model: torch.nn.Module,
    params: ExperimentConfig,
    device: torch.device,
    train_summary: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Load the best checkpoint, evaluate on the held-out test set, and update logs."""

    _, _, test_loader = get_loaders(params)
    state_dict = torch.load(params["save_path"], map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.detach().item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)

    test_loss = total_loss / total
    test_accuracy = correct / total
    append_run_row(
        params,
        build_log_row(
            params=params,
            epoch=params["epochs"],
            split="test",
            loss=test_loss,
            accuracy=test_accuracy,
            lr=0.0,
        ),
    )

    summary = train_summary or {}
    upsert_summary_row(
        params,
        {
            "run_name": params["run_name"],
            "model": params["model"],
            "dataset": params["dataset"],
            "train_mode": get_train_mode(params),
            "teacher_run": get_teacher_run_name(params),
            "checkpoint_path": params["save_path"],
            "label_smoothing": params["label_smoothing"],
            "temperature": params["temperature"] if params["distill_mode"] == "kd" else (1.0 if params["distill_mode"] == "teacher_trueclass_prob" else ""),
            "alpha": params["alpha"] if params["distill_mode"] == "kd" else "",
            "input_size": params["input_size"],
            "best_epoch": summary.get("best_epoch", ""),
            "best_val_loss": summary.get("best_val_loss", ""),
            "best_val_accuracy": summary.get("best_val_accuracy", ""),
            "final_train_loss": summary.get("final_train_loss", ""),
            "final_train_accuracy": summary.get("final_train_accuracy", ""),
            "final_val_loss": summary.get("final_val_loss", ""),
            "final_val_accuracy": summary.get("final_val_accuracy", ""),
            "test_loss": f"{test_loss:.6f}",
            "test_accuracy": f"{test_accuracy:.6f}",
            "seed": params["seed"],
        },
    )

    print(f"Test | loss={test_loss:.4f} acc={test_accuracy:.4f}")
    return {"test_loss": test_loss, "test_accuracy": test_accuracy}
