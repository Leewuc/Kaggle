from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


MINORITY_LABELS = {"offensive", "hate"}
LAUGH_PATTERN = re.compile(r"(ㅋ{2,}|ㅎ{2,}|ㅠ{2,}|ㅜ{2,})")
PUNCT_PATTERN = re.compile(r"[!?.,~]{2,}")
MULTISPACE_PATTERN = re.compile(r"\s{2,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copies-per-row", type=int, default=2)
    parser.add_argument("--labels", nargs="+", default=sorted(MINORITY_LABELS))
    return parser.parse_args()


def normalize_laughs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return token[0] * 2

    return LAUGH_PATTERN.sub(repl, text)


def stretch_laughs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        base = token[0]
        return base * random.choice([3, 4, 5])

    return LAUGH_PATTERN.sub(repl, text)


def simplify_punct(text: str) -> str:
    return PUNCT_PATTERN.sub(lambda m: m.group(0)[0], text)


def stretch_punct(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        ch = match.group(0)[0]
        return ch * random.choice([2, 3])

    return PUNCT_PATTERN.sub(repl, text)


def perturb_spacing(text: str) -> str:
    text = MULTISPACE_PATTERN.sub(" ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*\.\.\.\s*", "... ", text)
    return text.strip()


def add_light_suffix_noise(text: str) -> str:
    suffixes = ["", " ㅠㅠ", " ㅋㅋ", " ..", " !", " ~"]
    if len(text) > 3 and random.random() < 0.5:
        return f"{text}{random.choice(suffixes)}".strip()
    return text


def augment_text(text: str, variant_idx: int) -> str:
    augmented = text
    if variant_idx % 4 == 0:
        augmented = normalize_laughs(augmented)
    elif variant_idx % 4 == 1:
        augmented = stretch_laughs(augmented)
    elif variant_idx % 4 == 2:
        augmented = simplify_punct(augmented)
    else:
        augmented = stretch_punct(augmented)
    augmented = perturb_spacing(augmented)
    augmented = add_light_suffix_noise(augmented)
    return augmented


def augment_frame(frame: pd.DataFrame, copies_per_row: int, labels: list[str], seed: int) -> pd.DataFrame:
    random.seed(seed)
    targets = set(labels)
    augmented_rows: list[dict] = []

    for _, row in frame.iterrows():
        if row["label"] not in targets:
            continue
        original_comment = str(row["comments"])
        for copy_idx in range(copies_per_row):
            new_row = row.to_dict()
            new_row["id"] = f"{row['id']}_aug{copy_idx + 1}"
            new_row["comments"] = augment_text(original_comment, copy_idx)
            new_row["source_split"] = f"{row.get('source_split', 'unknown')}_aug"
            if new_row["comments"] == original_comment:
                new_row["comments"] = f"{original_comment} {random.choice(['..', 'ㅋㅋ', 'ㅠㅠ'])}".strip()
            augmented_rows.append(new_row)

    if not augmented_rows:
        return frame.copy()
    return pd.concat([frame, pd.DataFrame(augmented_rows)], ignore_index=True)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_path)
    output = augment_frame(frame, copies_per_row=args.copies_per_row, labels=args.labels, seed=args.seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)

    summary = {
        "input_rows": len(frame),
        "augmented_rows": len(output) - len(frame),
        "output_rows": len(output),
        "labels": args.labels,
        "copies_per_row": args.copies_per_row,
        "label_distribution": output["label"].value_counts().to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved augmented dataset to {args.output_path}")


if __name__ == "__main__":
    main()
