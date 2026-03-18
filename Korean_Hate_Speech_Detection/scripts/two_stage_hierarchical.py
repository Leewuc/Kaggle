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
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from runtime_profiles import apply_runtime_profile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "KcELECTRA-base-v2022-kmhas-binary"
FINAL_LABELS = ["none", "offensive", "hate"]
FINAL_LABEL_TO_ID = {label: idx for idx, label in enumerate(FINAL_LABELS)}
STAGE1_LABELS = ["none", "toxic"]
STAGE2_LABELS = ["offensive", "hate"]


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label_to_id: dict[str, int] | None, use_title: bool) -> None:
        self.comments = frame["comments"].fillna("").tolist()
        self.titles = frame["news_title"].fillna("").tolist() if use_title else [""] * len(frame)
        self.labels = None
        if label_to_id is not None and "label" in frame.columns:
            self.labels = [label_to_id[label] for label in frame["label"].tolist()]

    def __len__(self) -> int:
        return len(self.comments)

    def __getitem__(self, idx: int) -> dict[str, str | int]:
        item: dict[str, str | int] = {
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
        encoded = tokenizer(
            [row["comments"] for row in batch],
            [row["news_title"] for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            encoded["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        return encoded

    return collate


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def score_labels(true_labels: pd.Series, pred_labels: list[str]) -> dict[str, float]:
    return {
        "weighted_f1": f1_score(true_labels, pred_labels, average="weighted"),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro"),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }


def build_model(model_path: Path, labels: list[str]):
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )


def evaluate(
    model,
    data_loader,
    device: torch.device,
    labels: list[str],
    fp16: bool,
    bf16: bool,
) -> dict[str, float]:
    preds: list[int] = []
    trues: list[int] = []
    use_autocast = device.type == "cuda" and (fp16 or bf16)
    autocast_dtype = torch.float16 if fp16 else torch.bfloat16
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            truth = batch.pop("labels")
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
            )
            with autocast_ctx:
                logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            trues.extend(truth.cpu().tolist())
    pred_labels = [labels[idx] for idx in preds]
    true_labels = [labels[idx] for idx in trues]
    return score_labels(pd.Series(true_labels), pred_labels)


def train_model(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    tokenizer,
    model_path: Path,
    labels: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, float], int, list[dict[str, float]]]:
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        TextDataset(train_frame, label_to_id=label_to_id, use_title=args.use_title),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        TextDataset(valid_frame, label_to_id=label_to_id, use_title=args.use_title),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model(model_path, labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    criterion = None
    if args.use_class_weights:
        counts = train_frame["label"].value_counts()
        weights = torch.tensor(
            [len(train_frame) / (len(labels) * counts[label]) for label in labels],
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
            labels_tensor = batch.pop("labels")
            use_autocast = device.type == "cuda" and (args.fp16 or args.bf16)
            autocast_dtype = torch.float16 if args.fp16 else torch.bfloat16
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
            )
            with autocast_ctx:
                outputs = model(**batch)
                if criterion is None:
                    loss = nn.functional.cross_entropy(outputs.logits, labels_tensor)
                else:
                    loss = criterion(outputs.logits, labels_tensor)
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
                print(json.dumps({"stage_labels": labels, "epoch": epoch, "step": step, "avg_loss": total_loss / step}))

        metrics = evaluate(model, valid_loader, device, labels, args.fp16, args.bf16)
        epoch_summary = {"epoch": epoch, "train_loss": total_loss / len(train_loader), **metrics}
        history.append(epoch_summary)
        print(json.dumps({"stage_labels": labels, **epoch_summary}, ensure_ascii=False, indent=2))
        if best_metrics is None or metrics["weighted_f1"] > best_metrics["weighted_f1"]:
            best_metrics = metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, best_metrics, best_epoch, history


def predict_proba(model, frame: pd.DataFrame, tokenizer, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    loader = DataLoader(
        TextDataset(frame, label_to_id=None, use_title=args.use_title),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    outputs = []
    use_autocast = device.type == "cuda" and (args.fp16 or args.bf16)
    autocast_dtype = torch.float16 if args.fp16 else torch.bfloat16
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
            )
            with autocast_ctx:
                logits = model(**batch).logits
            outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
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
    parser.add_argument("--use-title", action="store_true")
    parser.add_argument("--gpu-profile", choices=["auto", "cpu", "quality", "safe", "medium", "full40"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = apply_runtime_profile(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    train_df = pd.read_csv(args.train_path).reset_index(drop=True)
    test_df = pd.read_csv(args.test_path).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    final_oof_probs = np.zeros((len(train_df), len(FINAL_LABELS)), dtype=np.float32)
    final_test_probs = np.zeros((len(test_df), len(FINAL_LABELS)), dtype=np.float32)
    fold_metrics = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_df, train_df["label"]), start=1):
        print(json.dumps({"fold": fold_idx, "status": "start"}, ensure_ascii=False))
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)

        stage1_train = fold_train.copy()
        stage1_valid = fold_valid.copy()
        stage1_train["label"] = stage1_train["label"].replace({"offensive": "toxic", "hate": "toxic"})
        stage1_valid["label"] = stage1_valid["label"].replace({"offensive": "toxic", "hate": "toxic"})
        stage1_model, stage1_metrics, stage1_best_epoch, stage1_history = train_model(
            stage1_train,
            stage1_valid,
            tokenizer=tokenizer,
            model_path=args.model_path,
            labels=STAGE1_LABELS,
            args=args,
            device=device,
        )

        toxic_train = fold_train[fold_train["label"].isin(STAGE2_LABELS)].reset_index(drop=True)
        toxic_valid = fold_valid[fold_valid["label"].isin(STAGE2_LABELS)].reset_index(drop=True)
        stage2_model, stage2_metrics, stage2_best_epoch, stage2_history = train_model(
            toxic_train,
            toxic_valid,
            tokenizer=tokenizer,
            model_path=args.model_path,
            labels=STAGE2_LABELS,
            args=args,
            device=device,
        )

        stage1_valid_probs = predict_proba(stage1_model, fold_valid, tokenizer, args, device)
        stage2_valid_probs = predict_proba(stage2_model, fold_valid, tokenizer, args, device)
        fold_final_probs = np.column_stack(
            [
                stage1_valid_probs[:, 0],
                stage1_valid_probs[:, 1] * stage2_valid_probs[:, 0],
                stage1_valid_probs[:, 1] * stage2_valid_probs[:, 1],
            ]
        )
        final_oof_probs[valid_idx] = fold_final_probs

        stage1_test_probs = predict_proba(stage1_model, test_df, tokenizer, args, device)
        stage2_test_probs = predict_proba(stage2_model, test_df, tokenizer, args, device)
        fold_test_probs = np.column_stack(
            [
                stage1_test_probs[:, 0],
                stage1_test_probs[:, 1] * stage2_test_probs[:, 0],
                stage1_test_probs[:, 1] * stage2_test_probs[:, 1],
            ]
        )
        final_test_probs += fold_test_probs / args.folds

        fold_pred_labels = [FINAL_LABELS[idx] for idx in fold_final_probs.argmax(axis=1)]
        final_metrics = score_labels(fold_valid["label"], fold_pred_labels)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "stage1_best_epoch": stage1_best_epoch,
                "stage1_metrics": stage1_metrics,
                "stage1_history": stage1_history,
                "stage2_best_epoch": stage2_best_epoch,
                "stage2_metrics": stage2_metrics,
                "stage2_history": stage2_history,
                "final_metrics": final_metrics,
            }
        )
        print(json.dumps(fold_metrics[-1], ensure_ascii=False, indent=2))

    oof_pred_labels = [FINAL_LABELS[idx] for idx in final_oof_probs.argmax(axis=1)]
    oof_metrics = score_labels(train_df["label"], oof_pred_labels)
    test_pred_labels = [FINAL_LABELS[idx] for idx in final_test_probs.argmax(axis=1)]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test_df["id"], "label": test_pred_labels})
    submission.to_csv(args.output_path, index=False)

    oof_probs_df = pd.DataFrame(final_oof_probs, columns=[f"prob_{label}" for label in FINAL_LABELS])
    oof_probs_df.insert(0, "id", train_df["id"].tolist())
    oof_probs_path = args.output_path.with_suffix(".oof_probs.csv")
    oof_probs_df.to_csv(oof_probs_path, index=False)

    test_probs_df = pd.DataFrame(final_test_probs, columns=[f"prob_{label}" for label in FINAL_LABELS])
    test_probs_df.insert(0, "id", test_df["id"].tolist())
    test_probs_path = args.output_path.with_suffix(".test_probs.csv")
    test_probs_df.to_csv(test_probs_path, index=False)

    metrics_payload = {
        "device": str(device),
        "folds": args.folds,
        "oof_metrics": oof_metrics,
        "fold_metrics": fold_metrics,
        "use_title": args.use_title,
    }
    metrics_path = args.output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
    print(f"Saved submission to {args.output_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved OOF probabilities to {oof_probs_path}")
    print(f"Saved test probabilities to {test_probs_path}")


if __name__ == "__main__":
    main()
