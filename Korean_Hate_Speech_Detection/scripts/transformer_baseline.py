from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from runtime_profiles import apply_runtime_profile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--klue--roberta-small"
    / "snapshots"
    / "b6b4c36d827e0293ae2fcf04d527072f10a23064"
)
LABELS = ["none", "offensive", "hate"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


class HateSpeechDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.comments = frame["comments"].fillna("").tolist()
        self.titles = frame["news_title"].fillna("").tolist()
        self.labels = None
        if "label" in frame.columns:
            self.labels = [LABEL_TO_ID[label] for label in frame["label"].tolist()]

    def __len__(self) -> int:
        return len(self.comments)

    def __getitem__(self, idx: int):
        item = {
            "comments": self.comments[idx],
            "news_title": self.titles[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        comments = [row["comments"] for row in batch]
        titles = [row["news_title"] for row in batch]
        encoded = tokenizer(
            comments,
            titles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            encoded["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        return encoded

    return collate


def build_model(model_path: Path):
    return AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, data_loader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    use_autocast = (args_device_type := device.type) == "cuda"
    autocast_dtype = torch.float16
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            labels = batch.pop("labels")
            autocast_ctx = (
                torch.autocast(device_type=args_device_type, dtype=autocast_dtype) if use_autocast else nullcontext()
            )
            with autocast_ctx:
                logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            trues.extend(labels.cpu().tolist())

    pred_labels = [ID_TO_LABEL[idx] for idx in preds]
    true_labels = [ID_TO_LABEL[idx] for idx in trues]
    return {
        "weighted_f1": f1_score(true_labels, pred_labels, average="weighted"),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro"),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }


def train_model(
    frame: pd.DataFrame,
    tokenizer,
    model_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    eval_frame: pd.DataFrame | None = None,
) -> tuple[torch.nn.Module, dict[str, float] | None, int, list[dict[str, float]]]:
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        HateSpeechDataset(frame),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = None
    if eval_frame is not None:
        eval_loader = DataLoader(
            HateSpeechDataset(eval_frame),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    model = build_model(model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    criterion = None
    if args.use_class_weights:
        counts = frame["label"].value_counts()
        weights = torch.tensor(
            [len(frame) / (len(LABELS) * counts[label]) for label in LABELS],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)

    best_state = None
    best_metrics = None
    best_epoch = 0
    history: list[dict[str, float]] = []

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
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "avg_loss": total_loss / step,
                        },
                        ensure_ascii=False,
                    )
                )

        if eval_loader is None:
            continue

        metrics = evaluate(model, eval_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            **metrics,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False, indent=2))
        if best_metrics is None or metrics["weighted_f1"] > best_metrics["weighted_f1"]:
            best_metrics = metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_epoch == 0:
        best_epoch = args.epochs

    return model, best_metrics, best_epoch, history


def predict(model, frame: pd.DataFrame, tokenizer, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    data_loader = DataLoader(
        HateSpeechDataset(frame),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return pd.DataFrame({"id": frame["id"], "label": [ID_TO_LABEL[idx] for idx in preds]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--skip-full-retrain", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gpu-profile", choices=["auto", "cpu", "quality", "safe", "medium", "full40"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = apply_runtime_profile(args, device)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    fit_train_df, fit_valid_df = train_test_split(
        train_df,
        test_size=args.validation_size,
        stratify=train_df["label"],
        random_state=args.random_state,
    )

    model, best_metrics, best_epoch, _ = train_model(
        frame=fit_train_df.reset_index(drop=True),
        eval_frame=fit_valid_df.reset_index(drop=True),
        tokenizer=tokenizer,
        model_path=args.model_path,
        args=args,
        device=device,
    )

    result = {
        "device": str(device),
        "model_path": str(args.model_path),
        "best_epoch": best_epoch,
        "validation": best_metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    final_model = model
    if not args.skip_full_retrain:
        final_args = copy.deepcopy(args)
        final_args.epochs = best_epoch
        final_model, _, _, _ = train_model(
            frame=train_df.reset_index(drop=True),
            eval_frame=None,
            tokenizer=tokenizer,
            model_path=args.model_path,
            args=final_args,
            device=device,
        )

    submission = predict(final_model, test_df, tokenizer, args, device)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)

    metrics_path = args.output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"Saved submission to {args.output_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
