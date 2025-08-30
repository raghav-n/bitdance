from __future__ import annotations

"""Fraud similarity scan model.

Implements a lightweight "training" step that computes per-business similarity
metrics and flags suspicious high-similarity subclusters. Artifacts are written
under the model directory to integrate with the existing train pipeline.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .fraud_group_detector import _get_model


@dataclass
class FraudScanConfig:
    min_reviews: int = 5
    topk: int = 5
    dup_threshold: float = 0.92
    sim_threshold: float = 0.85
    high_sim_threshold: float = 0.92
    min_cluster_size: int = 3
    overall_threshold: float = 0.82


def _embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    )


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    return X @ X.T


def _cluster_labels(X: np.ndarray, sim_threshold: float, min_samples: int) -> np.ndarray:
    eps = float(max(0.0, min(1.0, 1.0 - sim_threshold)))
    return DBSCAN(eps=eps, min_samples=int(min_samples), metric="cosine", n_jobs=-1).fit_predict(X)


def _cluster_stats_from_labels(S: np.ndarray, labels: np.ndarray) -> tuple[int, float]:
    best_size = 0
    best_mean = 0.0
    for lbl in set(labels) - {-1}:
        idx = np.where(labels == lbl)[0]
        if idx.size <= 1:
            continue
        sub = S[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, np.nan)
        m = float(np.nanmean(sub)) if np.isfinite(sub).any() else 0.0
        if idx.size > best_size or (idx.size == best_size and m > best_mean):
            best_size = int(idx.size)
            best_mean = float(m)
    return best_size, best_mean


def _biz_key_series(df: pd.DataFrame) -> pd.Series:
    if "business_id" in df.columns:
        bid = df["business_id"].astype(str)
    else:
        bid = pd.Series([""] * len(df), index=df.index)
    bname = df.get("business_name", pd.Series([""] * len(df), index=df.index)).astype(str)
    biz_key = bid.where(bid.str.strip().ne("") & bid.str.lower().ne("nan"), bname)
    return biz_key


def _business_scores(
    texts: List[str],
    *,
    topk: int,
    dup_threshold: float,
    sim_threshold: float,
    min_cluster_size: int,
    high_sim_threshold: float,
) -> Dict[str, float | int]:
    X = _embed_texts(texts)
    n = int(X.shape[0])
    S = _cosine_sim_matrix(X)
    S2 = S.copy()
    np.fill_diagonal(S2, -np.inf)
    k = int(max(1, min(topk, n - 1)))
    topk_vals = np.partition(S2, -k, axis=1)[:, -k:]
    mean_topk = float(np.mean(topk_vals)) if topk_vals.size else 0.0
    dup_cnt = int(np.sum(S2 > float(dup_threshold)))
    denom = max(1, n * (n - 1))
    dup_rate = float(dup_cnt) / float(denom)

    labels_nom = _cluster_labels(X, sim_threshold=sim_threshold, min_samples=min_cluster_size)
    max_cluster_size, max_cluster_mean = _cluster_stats_from_labels(S, labels_nom)

    labels_hi = _cluster_labels(X, sim_threshold=high_sim_threshold, min_samples=min_cluster_size)
    max_hi_size, max_hi_mean = _cluster_stats_from_labels(S, labels_hi)

    return {
        "n_reviews": n,
        "mean_topk": float(mean_topk),
        "dup_rate": float(dup_rate),
        "max_cluster_size": int(max_cluster_size),
        "max_cluster_mean": float(max_cluster_mean),
        "max_high_cluster_size": int(max_hi_size),
        "max_high_cluster_mean": float(max_hi_mean),
    }


def train_fraud_similarity(
    out_dir: str,
    data_parquet: str,
    config_overrides: dict | None = None,
) -> None:
    """Run the fraud similarity scan and persist artifacts under out_dir.

    - Reads normalized interim parquet and restricts to original reviews.
    - Groups by business and computes cohesion + subcluster metrics.
    - Writes business_scores.jsonl and clusters.jsonl.
    - Writes a manifest model.bin for orchestrator compatibility.
    """
    cfg = FraudScanConfig()
    for k, v in (config_overrides or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_parquet)
    # Prefer original reviews to avoid synthetic augmentation affecting similarity
    if "source" in df.columns:
        df = df[df["source"].astype(str) == "original"]
    text_series = df.get("text", pd.Series([""] * len(df), index=df.index)).astype(str)
    biz_key = _biz_key_series(df)
    valid = biz_key.str.strip().ne("") & text_series.str.strip().ne("")
    df = df.loc[valid].copy()
    text_series = text_series.loc[df.index]
    biz_key = biz_key.loc[df.index]

    groups = df.groupby(biz_key)

    scores_path = out / "business_scores.jsonl"
    clusters_path = out / "clusters.jsonl"
    suspicious_total = 0
    with open(scores_path, "w", encoding="utf-8") as fs, open(clusters_path, "w", encoding="utf-8") as fc:
        for biz, idx in groups.groups.items():
            idx_list = list(idx)
            if len(idx_list) < int(cfg.min_reviews):
                continue
            texts = text_series.loc[idx_list].tolist()
            stats = _business_scores(
                texts,
                topk=int(cfg.topk),
                dup_threshold=float(cfg.dup_threshold),
                sim_threshold=float(cfg.sim_threshold),
                min_cluster_size=int(cfg.min_cluster_size),
                high_sim_threshold=float(cfg.high_sim_threshold),
            )
            suspicious = bool(
                stats["mean_topk"] >= float(cfg.overall_threshold)
                or (
                    stats["max_high_cluster_size"] >= int(cfg.min_cluster_size)
                    and stats["max_high_cluster_mean"] >= float(cfg.high_sim_threshold)
                )
            )
            reason = (
                "overall"
                if stats["mean_topk"] >= float(cfg.overall_threshold)
                else (
                    "subcluster"
                    if stats["max_high_cluster_size"] >= int(cfg.min_cluster_size)
                    and stats["max_high_cluster_mean"] >= float(cfg.high_sim_threshold)
                    else ""
                )
            )
            fs.write(
                json.dumps(
                    {
                        "business": str(biz),
                        **stats,
                        "suspicious": suspicious,
                        "reason": reason,
                    }
                )
                + "\n"
            )
            if suspicious:
                suspicious_total += 1
                fc.write(
                    json.dumps(
                        {
                            "business": str(biz),
                            "n_reviews": int(stats["n_reviews"]),
                            "max_cluster_size": int(stats["max_cluster_size"]),
                            "max_cluster_mean": float(stats["max_cluster_mean"]),
                            "max_high_cluster_size": int(stats["max_high_cluster_size"]),
                            "max_high_cluster_mean": float(stats["max_high_cluster_mean"]),
                            "reason": reason,
                        }
                    )
                    + "\n"
                )

    # Write a simple manifest so orchestrator recognizes primary artifact
    (out / "model.bin").write_text(
        json.dumps(
            {
                "note": "Fraud similarity scan artifacts: see business_scores.jsonl and clusters.jsonl",
                "config": asdict(cfg),
            }
        ),
        encoding="utf-8",
    )

    # A label_map isn't used, but some tools expect it; write minimal file
    (out / "label_map.json").write_text(
        json.dumps({"irrelevant_content": 0, "advertisement": 1, "review_without_visit": 2}, indent=2),
        encoding="utf-8",
    )

