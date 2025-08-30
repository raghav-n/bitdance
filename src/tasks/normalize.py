"""Normalization task: build training input by concatenating original + synthetic.

This task now runs AFTER annotation and augmentation. It:
- Loads annotated dataset (original reviews with LLM-derived labels).
- Loads augmented synthetic dataset (policy-specific violating reviews).
- Maps both into a unified, training-ready schema with label columns
  matching `models.LABELS` and simple metadata fields.
- Writes:
  - Parquet combined dataset to `data/interim/{dataset}/reviews.parquet` (for analysis).
  - JSONL train/val/test splits (70/15/15) to
    `data/processed/{dataset}/train.jsonl|val.jsonl|test.jsonl`.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..orchestrator import task
from ..orchestrator.utils import data_dir, dataset_slug, seed


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df


def _has_url(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r"(https?://|www\.)", text, flags=re.IGNORECASE))


def _bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col in df.columns:
        return df[col].astype(bool)
    # default length-aligned series
    return pd.Series([default] * len(df), index=df.index, dtype=bool)


@task(
    name="normalize",
    inputs=lambda p: [
        "configs/base.yaml",
        f"{data_dir(p)}/annotated/{dataset_slug(p)}/annotations.parquet",
        f"{data_dir(p)}/augmented/{dataset_slug(p)}/reviews.parquet",
    ],
    outputs=lambda p: [
        f"{data_dir(p)}/interim/{dataset_slug(p)}/reviews.parquet",
        f"{data_dir(p)}/processed/{dataset_slug(p)}/train.jsonl",
        f"{data_dir(p)}/processed/{dataset_slug(p)}/val.jsonl",
        f"{data_dir(p)}/processed/{dataset_slug(p)}/test.jsonl",
    ],
)
def normalize(params: Dict):
    """Concatenate annotated original + synthetic into a unified training dataset."""
    base_dir = Path(data_dir(params))
    ds = dataset_slug(params)

    ann_path = base_dir / "annotated" / ds / "annotations.parquet"
    aug_path = base_dir / "augmented" / ds / "reviews.parquet"

    out_parquet = base_dir / "interim" / ds / "reviews.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    proc_dir = base_dir / "processed" / ds
    proc_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = proc_dir / "train.jsonl"
    val_jsonl = proc_dir / "val.jsonl"
    test_jsonl = proc_dir / "test.jsonl"

    # Load annotated original dataset
    ann_df = pd.read_parquet(ann_path)
    # Map policy columns -> label columns
    # Annotate created: is_noadvertisement, is_noirrelevantcontent, is_norantwithoutvisit
    ann_df = _ensure_cols(
        ann_df,
        [
            "is_noadvertisement",
            "is_noirrelevantcontent",
            "is_norantwithoutvisit",
        ],
    )
    ann_df_m = pd.DataFrame(
        {
            "business_name": ann_df.get("place_name", "").astype(str),
            "author_name": ann_df.get("user_name", "").astype(str),
            "text": ann_df.get("text", "").astype(str),
            "language": ann_df.get("language", "").astype(str),
            "rating": ann_df.get("rating", 0).astype(int),
            "has_photo": _bool_series(ann_df, "has_image", False),
            "has_url": ann_df.get("text", "").astype(str).apply(_has_url).astype(bool),
            "advertisement": ann_df["is_noadvertisement"].astype(int),
            "irrelevant_content": ann_df["is_noirrelevantcontent"].astype(int),
            "review_without_visit": ann_df["is_norantwithoutvisit"].astype(int),
            "source": "original",
        }
    )

    # Load synthetic augmented dataset
    aug_df = pd.read_parquet(aug_path)

    def _policy_to_labels(policy: str) -> Dict[str, int]:
        policy = (policy or "").strip()
        lab = {
            "advertisement": 0,
            "irrelevant_content": 0,
            "review_without_visit": 0,
            "fraudulent_review": 0,
        }
        if policy == "NoAdvertisement":
            lab["advertisement"] = 1
        elif policy == "NoIrrelevantContent":
            lab["irrelevant_content"] = 1
        elif policy == "NoRantWithoutVisit":
            lab["review_without_visit"] = 1
        return lab

    def _parse_meta(md) -> Dict:
        if isinstance(md, dict):
            return md
        if isinstance(md, str) and md.strip():
            try:
                return json.loads(md)
            except Exception:
                return {}
        return {}

    aug_meta = aug_df.get("metadata")
    aug_policies = []
    if aug_meta is not None:
        for v in aug_meta:
            md = _parse_meta(v)
            aug_policies.append(md.get("policy", ""))
    else:
        aug_policies = [""] * len(aug_df)

    aug_labels = [_policy_to_labels(p) for p in aug_policies]
    aug_labels_df = pd.DataFrame(aug_labels)

    aug_df_m = pd.DataFrame(
        {
            "business_name": aug_df.get("place_name", "").astype(str),
            "author_name": aug_df.get("user_name", "").astype(str),
            "text": aug_df.get("text", "").astype(str),
            "language": aug_df.get("language", "").astype(str),
            "rating": aug_df.get("rating", 0).astype(int),
            "has_photo": _bool_series(aug_df, "has_image", False),
            "has_url": aug_df.get("text", "").astype(str).apply(_has_url).astype(bool),
            "source": "synthetic",
        }
    ).reset_index(drop=True)
    # Attach label columns
    for c in ["advertisement", "irrelevant_content", "review_without_visit"]:
        if c in aug_labels_df.columns:
            aug_df_m[c] = aug_labels_df[c].astype(int)
        else:
            aug_df_m[c] = 0

    # Concatenate
    combined = pd.concat([ann_df_m, aug_df_m], ignore_index=True)

    # Persist Parquet for EDA and downstream
    combined.to_parquet(out_parquet, index=False)

    # Create train/val/test split by assigning entire businesses to a single split
    # Target ratios: 70/15/15 (by number of reviews), matched as closely as possible.
    # Keep randomness via seed-controlled shuffles with multiple trials; pick best fit to targets.

    # Build a business key: prefer business_id, else business_name; if empty, use row-unique key
    if "business_id" in combined.columns:
        bid = combined["business_id"].astype(str)
    else:
        bid = pd.Series([""] * len(combined), index=combined.index)
    bname = combined.get("business_name", pd.Series([""] * len(combined), index=combined.index)).astype(str)
    biz_key = bid.where(bid.str.strip().ne("") & bid.str.lower().ne("nan"), bname)
    # Fill any remaining empties with row-unique surrogate ids to avoid lumping unrelated rows together
    empty_mask = biz_key.str.strip().eq("") | biz_key.str.lower().eq("nan")
    if empty_mask.any():
        biz_key.loc[empty_mask] = [f"__row_{i}" for i in biz_key.index[empty_mask]]

    # Count reviews per business
    sizes = biz_key.value_counts().to_dict()
    total = int(len(combined))
    # Targets
    t_train = int(round(0.70 * total))
    t_val = int(round(0.15 * total))
    # ensure exact total
    t_test = total - t_train - t_val

    rng = random.Random(int(seed(params)))
    items_all = list(sizes.items())
    targets = {"train": t_train, "val": t_val, "test": t_test}

    def _assign_for_order(order: List[tuple[str, int]]) -> tuple[dict[str, str], dict[str, int]]:
        assign: dict[str, str] = {}
        c = {"train": 0, "val": 0, "test": 0}

        def _pick_bucket(sz: int) -> str:
            deficits = {k: targets[k] - c[k] for k in ("train", "val", "test")}
            pos = [k for k, d in deficits.items() if d > 0]
            if pos:
                max_def = max(deficits[k] for k in pos)
                winners = [k for k in pos if deficits[k] == max_def]
                return rng.choice(winners)
            # All buckets full/over: choose bucket that minimizes overflow after adding sz
            over = {k: c[k] + sz - targets[k] for k in ("train", "val", "test")}
            min_over = min(over.values())
            winners = [k for k, v in over.items() if v == min_over]
            return rng.choice(winners)

        for biz, sz in order:
            bucket = _pick_bucket(int(sz))
            assign[biz] = bucket
            c[bucket] += int(sz)
        return assign, c

    def _score_counts(c: dict[str, int]) -> int:
        # Lower is better; squared error against targets
        return sum((c[k] - targets[k]) ** 2 for k in ("train", "val", "test"))

    trials = int((params.get("normalize", {}) or {}).get("split_trials", 64))
    best_assign: dict[str, str] | None = None
    best_counts: dict[str, int] | None = None
    best_err: int | None = None

    for t in range(max(1, trials)):
        order = items_all[:]
        rng.shuffle(order)
        a, cnt = _assign_for_order(order)
        err = _score_counts(cnt)
        if best_err is None or err < best_err:
            best_err = err
            best_assign = a
            best_counts = cnt

    assign = best_assign or {}

    # Map assignments back to rows
    combined = combined.copy()
    combined["__biz_key"] = biz_key
    combined["__split"] = combined["__biz_key"].map(assign)
    train_df = combined[combined["__split"] == "train"].drop(columns=["__biz_key", "__split"])  # type: ignore[index]
    val_df = combined[combined["__split"] == "val"].drop(columns=["__biz_key", "__split"])  # type: ignore[index]
    test_df = combined[combined["__split"] == "test"].drop(columns=["__biz_key", "__split"])  # type: ignore[index]

    def _to_jsonl(df: pd.DataFrame, path: Path) -> None:
        records = df.to_dict(orient="records")
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _to_jsonl(train_df, train_jsonl)
    _to_jsonl(val_df, val_jsonl)
    _to_jsonl(test_df, test_jsonl)

    return
