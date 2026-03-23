#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p artifacts/checkpoints artifacts/runs

python3 main.py --mode both --dataset cifar10 --model resnet --epochs 10 --seed 42 --device cuda --batch_size 32 --run_name transfer_resize_freeze --pretrained --transfer_mode resize_freeze --freeze_backbone true --output_dir artifacts
python3 main.py --mode both --dataset cifar10 --model resnet --epochs 10 --seed 42 --device cuda --batch_size 64 --run_name transfer_modify_finetune --pretrained --transfer_mode modify_finetune --output_dir artifacts
python3 main.py --mode both --dataset cifar10 --model cnn --epochs 10 --seed 42 --device cuda --batch_size 64 --run_name simplecnn_scratch --output_dir artifacts
python3 main.py --mode both --dataset cifar10 --model resnet --epochs 10 --seed 42 --device cuda --batch_size 64 --run_name resnet_scratch_no_ls --label_smoothing 0.0 --output_dir artifacts
python3 main.py --mode both --dataset cifar10 --model resnet --epochs 10 --seed 42 --device cuda --batch_size 64 --run_name resnet_scratch_ls --label_smoothing 0.1 --output_dir artifacts

teacher_run="$(
python3 - <<'PY'
import csv
from pathlib import Path

summary_path = Path("artifacts/summary.csv")
if not summary_path.exists():
    raise SystemExit("artifacts/summary.csv was not found.")

target_runs = {"resnet_scratch_no_ls", "resnet_scratch_ls"}
with summary_path.open("r", newline="", encoding="utf-8") as csv_file:
    rows = [row for row in csv.DictReader(csv_file) if row["run_name"] in target_runs]

if len(rows) != 2:
    raise SystemExit("Expected both ResNet scratch runs in artifacts/summary.csv.")

best_row = max(rows, key=lambda row: float(row["best_val_accuracy"]))
print(best_row["run_name"])
PY
)"

teacher_checkpoint="artifacts/checkpoints/${teacher_run}.pth"
echo "Selected teacher: ${teacher_run}"

python3 main.py --mode both --dataset cifar10 --model cnn --epochs 10 --seed 42 --device cuda --batch_size 64 --run_name simplecnn_kd_from_best_resnet --distill_mode kd --teacher_checkpoint "${teacher_checkpoint}" --temperature 4.0 --alpha 0.7 --output_dir artifacts
python3 main.py --mode both --dataset cifar10 --model mobilenet --epochs 10 --seed 42 --device cuda --batch_size 64 --run_name mobilenet_teacher_trueclass_prob --distill_mode teacher_trueclass_prob --teacher_checkpoint "${teacher_checkpoint}" --output_dir artifacts
python3 main.py --mode flops --dataset cifar10 --model resnet --device cuda --run_name flops_summary --output_dir artifacts
