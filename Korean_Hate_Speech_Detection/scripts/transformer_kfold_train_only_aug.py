from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from augment_weak_korean_comments import augment_frame
from runtime_profiles import apply_runtime_profile
from transformer_baseline import (
    DEFAULT_MODEL_PATH,
    HateSpeechDataset,
    ID_TO_LABEL,
    LABELS,
    make_collate_fn,
    move_to_device,
    set_seed,
    train_model,
)


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
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--aug-copies-per-row", type=int, default=2)
    parser.add_argument("--aug-labels", nargs="+", default=["offensive", "hate"])
    parser.add_argument("--gpu-profile", choices=["auto", "cpu", "quality", "safe", "medium", "full40"], default="auto")
    return parser.parse_args()


def predict_proba(model, frame: pd.DataFrame, tokenizer, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    loader = DataLoader(
        HateSpeechDataset(frame),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            outputs.append(probs)
    return np.vstack(outputs)


def score_labels(true_labels: pd.Series, pred_ids: np.ndarray) -> dict[str, float]:
    pred_labels = [ID_TO_LABEL[idx] for idx in pred_ids]
    return {
        "weighted_f1": f1_score(true_labels, pred_labels, average="weighted"),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro"),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = apply_runtime_profile(args, device)

    train_df = pd.read_csv(args.train_path).reset_index(drop=True)
    test_df = pd.read_csv(args.test_path).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
    test_probs = np.zeros((len(test_df), len(LABELS)), dtype=np.float32)
    fold_metrics = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_df, train_df["label"]), start=1):
        print(json.dumps({"fold": fold_idx, "status": "start"}, ensure_ascii=False))
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)
        fold_train_aug = augment_frame(
            fold_train,
            copies_per_row=args.aug_copies_per_row,
            labels=args.aug_labels,
            seed=args.seed + fold_idx,
        )

        model, metrics, best_epoch, history = train_model(
            frame=fold_train_aug,
            eval_frame=fold_valid,
            tokenizer=tokenizer,
            model_path=args.model_path,
            args=args,
            device=device,
        )
        valid_probs = predict_proba(model, fold_valid, tokenizer, args, device)
        oof_probs[valid_idx] = valid_probs
        test_probs += predict_proba(model, test_df, tokenizer, args, device) / args.folds

        fold_result = {
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "augmented_train_rows": len(fold_train_aug),
            **(metrics or {}),
            "history": history,
        }
        fold_metrics.append(fold_result)
        print(json.dumps(fold_result, ensure_ascii=False, indent=2))

    oof_pred_ids = oof_probs.argmax(axis=1)
    oof_metrics = score_labels(train_df["label"], oof_pred_ids)
    test_pred_ids = test_probs.argmax(axis=1)

    submission = pd.DataFrame({"id": test_df["id"], "label": [ID_TO_LABEL[idx] for idx in test_pred_ids]})
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)

    if args.save_probs:
        oof_probs_df = pd.DataFrame(oof_probs, columns=[f"prob_{label}" for label in LABELS])
        oof_probs_df.insert(0, "id", train_df["id"].tolist())
        oof_probs_path = args.output_path.with_suffix(".oof_probs.csv")
        oof_probs_df.to_csv(oof_probs_path, index=False)

        test_probs_df = pd.DataFrame(test_probs, columns=[f"prob_{label}" for label in LABELS])
        test_probs_df.insert(0, "id", test_df["id"].tolist())
        test_probs_path = args.output_path.with_suffix(".test_probs.csv")
        test_probs_df.to_csv(test_probs_path, index=False)

    metrics_payload = {
        "device": str(device),
        "folds": args.folds,
        "fold_metrics": fold_metrics,
        "oof_metrics": oof_metrics,
        "aug_copies_per_row": args.aug_copies_per_row,
        "aug_labels": args.aug_labels,
        "save_probs": args.save_probs,
    }
    metrics_path = args.output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
    print(f"Saved submission to {args.output_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
