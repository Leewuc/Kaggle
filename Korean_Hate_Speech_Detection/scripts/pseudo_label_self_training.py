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


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "korean_hate_speech_detection_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gpu-profile", choices=["auto", "cpu", "quality", "safe", "medium", "full40"], default="auto")
    parser.add_argument("--pseudo-threshold", type=float, default=0.97)
    parser.add_argument("--max-pseudo-per-class", type=int, default=500)
    parser.add_argument("--unlabeled-sample-size", type=int, default=6000)
    parser.add_argument("--include-kaggle-test", action="store_true")
    return parser.parse_args()


def predict_proba(model, frame: pd.DataFrame, tokenizer, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    loader = DataLoader(
        HateSpeechDataset(frame),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


def score_labels(true_labels: pd.Series, pred_ids: np.ndarray) -> dict[str, float]:
    pred_labels = [ID_TO_LABEL[idx] for idx in pred_ids]
    return {
        "weighted_f1": f1_score(true_labels, pred_labels, average="weighted"),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro"),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }


def load_external_unlabeled(sample_size: int, include_kaggle_test: bool, seed: int) -> pd.DataFrame:
    frames = []

    if include_kaggle_test:
        kaggle_test = pd.read_csv(SOURCE_DIR / "test.hate.no_label.csv").copy()
        with (SOURCE_DIR / "test.news_title.txt").open(encoding="utf-8") as handle:
            kaggle_test["news_title"] = [line.rstrip("\n") for line in handle]
        kaggle_test.insert(0, "id", [f"EXT_TEST_{idx:05d}" for idx in range(len(kaggle_test))])
        kaggle_test["source"] = "kaggle_test"
        frames.append(kaggle_test[["id", "comments", "news_title", "source"]])

    total_unlabeled = min(
        sum(1 for _ in open(SOURCE_DIR / "unlabeled_comments.txt", encoding="utf-8")),
        sum(1 for _ in open(SOURCE_DIR / "unlabeled_comments.news_title.txt", encoding="utf-8")),
    )
    chosen = np.random.default_rng(seed).choice(total_unlabeled, size=min(sample_size, total_unlabeled), replace=False)
    chosen_set = set(int(idx) for idx in chosen)
    comments = []
    titles = []
    with (SOURCE_DIR / "unlabeled_comments.txt").open(encoding="utf-8") as comment_handle, (
        SOURCE_DIR / "unlabeled_comments.news_title.txt"
    ).open(encoding="utf-8") as title_handle:
        for idx, (comment, title) in enumerate(zip(comment_handle, title_handle)):
            if idx not in chosen_set:
                continue
            comments.append(comment.rstrip("\n"))
            titles.append(title.rstrip("\n"))
    unlabeled = pd.DataFrame(
        {
            "id": [f"EXT_UNLABELED_{idx:05d}" for idx in range(len(comments))],
            "comments": comments,
            "news_title": titles,
            "source": "unlabeled_sample",
        }
    )
    frames.append(unlabeled)

    return pd.concat(frames, ignore_index=True)


def select_pseudo_labels(pool: pd.DataFrame, probs: np.ndarray, threshold: float, max_per_class: int) -> pd.DataFrame:
    pred_ids = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    frame = pool.copy()
    frame["pseudo_label"] = [ID_TO_LABEL[idx] for idx in pred_ids]
    frame["confidence"] = confidences
    frame = frame[frame["confidence"] >= threshold].copy()
    frame = frame.sort_values(["pseudo_label", "confidence"], ascending=[True, False])
    selected = frame.groupby("pseudo_label", group_keys=False).head(max_per_class).reset_index(drop=True)
    return selected


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = apply_runtime_profile(args, device)

    train_df = pd.read_csv(args.train_path).reset_index(drop=True)
    test_df = pd.read_csv(args.test_path).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    external_pool = load_external_unlabeled(
        sample_size=args.unlabeled_sample_size,
        include_kaggle_test=args.include_kaggle_test,
        seed=args.seed,
    )

    teacher_skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
    pool_probs = np.zeros((len(external_pool), len(LABELS)), dtype=np.float32)
    teacher_fold_metrics = []

    for fold_idx, (train_idx, valid_idx) in enumerate(teacher_skf.split(train_df, train_df["label"]), start=1):
        print(json.dumps({"stage": "teacher", "fold": fold_idx, "status": "start"}, ensure_ascii=False))
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)

        model, metrics, best_epoch, _ = train_model(
            frame=fold_train,
            eval_frame=fold_valid,
            tokenizer=tokenizer,
            model_path=args.model_path,
            args=args,
            device=device,
        )
        oof_probs[valid_idx] = predict_proba(model, fold_valid, tokenizer, args, device)
        pool_probs += predict_proba(model, external_pool, tokenizer, args, device) / args.folds
        teacher_fold_metrics.append({"fold": fold_idx, "best_epoch": best_epoch, **(metrics or {})})

    teacher_oof_metrics = score_labels(train_df["label"], oof_probs.argmax(axis=1))
    pseudo_df = select_pseudo_labels(
        pool=external_pool,
        probs=pool_probs,
        threshold=args.pseudo_threshold,
        max_per_class=args.max_pseudo_per_class,
    )
    print(
        json.dumps(
            {
                "stage": "pseudo_selection",
                "selected": len(pseudo_df),
                "by_label": pseudo_df["pseudo_label"].value_counts().sort_index().to_dict(),
                "by_source": pseudo_df["source"].value_counts().sort_index().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if pseudo_df.empty:
        raise ValueError("No pseudo-labeled rows were selected. Lower the threshold or increase the sample size.")

    augmented_train = pd.concat(
        [
            train_df[["id", "comments", "news_title", "label"]],
            pseudo_df.rename(columns={"pseudo_label": "label"})[
                ["id", "comments", "news_title", "label"]
            ],
        ],
        ignore_index=True,
    )

    student_skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    student_oof_probs = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
    test_probs = np.zeros((len(test_df), len(LABELS)), dtype=np.float32)
    student_fold_metrics = []

    real_mask = augmented_train["id"].str.startswith("KHSD_")
    real_labels = augmented_train.loc[real_mask, "label"].reset_index(drop=True)

    for fold_idx, (real_train_idx, real_valid_idx) in enumerate(
        student_skf.split(train_df, train_df["label"]),
        start=1,
    ):
        print(json.dumps({"stage": "student", "fold": fold_idx, "status": "start"}, ensure_ascii=False))
        real_train_ids = set(train_df.iloc[real_train_idx]["id"])
        real_valid_frame = train_df.iloc[real_valid_idx].reset_index(drop=True)
        fold_train = augmented_train[
            augmented_train["id"].isin(real_train_ids) | ~augmented_train["id"].str.startswith("KHSD_")
        ].reset_index(drop=True)

        model, metrics, best_epoch, _ = train_model(
            frame=fold_train,
            eval_frame=real_valid_frame,
            tokenizer=tokenizer,
            model_path=args.model_path,
            args=args,
            device=device,
        )
        student_oof_probs[real_valid_idx] = predict_proba(model, real_valid_frame, tokenizer, args, device)
        test_probs += predict_proba(model, test_df, tokenizer, args, device) / args.folds
        student_fold_metrics.append({"fold": fold_idx, "best_epoch": best_epoch, **(metrics or {})})

    student_oof_metrics = score_labels(train_df["label"], student_oof_probs.argmax(axis=1))
    submission = pd.DataFrame({"id": test_df["id"], "label": [ID_TO_LABEL[idx] for idx in test_probs.argmax(axis=1)]})
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)

    pseudo_path = args.output_path.with_suffix(".pseudo.csv")
    pseudo_df.to_csv(pseudo_path, index=False)

    metrics_payload = {
        "device": str(device),
        "teacher_oof_metrics": teacher_oof_metrics,
        "student_oof_metrics": student_oof_metrics,
        "teacher_fold_metrics": teacher_fold_metrics,
        "student_fold_metrics": student_fold_metrics,
        "pseudo_selected": len(pseudo_df),
        "pseudo_by_label": pseudo_df["pseudo_label"].value_counts().sort_index().to_dict(),
        "pseudo_by_source": pseudo_df["source"].value_counts().sort_index().to_dict(),
        "pseudo_threshold": args.pseudo_threshold,
        "max_pseudo_per_class": args.max_pseudo_per_class,
    }
    metrics_path = args.output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))
    print(f"Saved submission to {args.output_path}")
    print(f"Saved pseudo labels to {pseudo_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
