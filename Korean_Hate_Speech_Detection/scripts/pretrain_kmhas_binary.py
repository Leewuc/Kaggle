from __future__ import annotations

import argparse
import copy
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from runtime_profiles import apply_runtime_profile


LABELS = ["not_hate", "hate"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


class BinaryTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.text = frame["text"].fillna("").tolist()
        self.labels = frame["hate_binary"].tolist()

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int):
        return {"text": self.text[idx], "label": int(self.labels[idx])}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [row["text"] for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        return encoded

    return collate


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, data_loader, device: torch.device, fp16: bool, bf16: bool) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    use_autocast = device.type == "cuda" and (fp16 or bf16)
    autocast_dtype = torch.float16 if fp16 else torch.bfloat16
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            labels = batch.pop("labels")
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
            )
            with autocast_ctx:
                logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            trues.extend(labels.cpu().tolist())
    return {
        "weighted_f1": f1_score(trues, preds, average="weighted"),
        "macro_f1": f1_score(trues, preds, average="macro"),
        "accuracy": accuracy_score(trues, preds),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--valid-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--gpu-profile", choices=["auto", "cpu", "quality", "safe", "medium", "full40"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = apply_runtime_profile(args, device)

    train_df = pd.read_csv(args.train_path)
    valid_df = pd.read_csv(args.valid_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(BinaryTextDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(BinaryTextDataset(valid_df), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    criterion = None
    if args.use_class_weights:
        counts = train_df["hate_binary"].value_counts()
        weights = torch.tensor(
            [len(train_df) / (2 * counts[idx]) for idx in [0, 1]],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)

    best_state = None
    best_metrics = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            labels = batch.pop("labels")
            use_autocast = device.type == "cuda" and (args.fp16 or args.bf16)
            autocast_dtype = torch.float16 if args.fp16 else torch.bfloat16
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
            )
            with autocast_ctx:
                outputs = model(**batch)
                if criterion is None:
                    loss = nn.functional.cross_entropy(outputs.logits, labels)
                else:
                    loss = criterion(outputs.logits, labels)
                loss = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * args.gradient_accumulation_steps
            if step % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "avg_loss": total_loss / step}, ensure_ascii=False))

        metrics = evaluate(model, valid_loader, device, args.fp16, args.bf16)
        print(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False, indent=2))
        if best_metrics is None or metrics["weighted_f1"] > best_metrics["weighted_f1"]:
            best_metrics = metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_payload = {
        "device": str(device),
        "best_epoch": best_epoch,
        "validation": best_metrics,
    }
    metrics_path = args.output_dir / "kmhas_binary_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
    print(f"Saved checkpoint to {args.output_dir}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
