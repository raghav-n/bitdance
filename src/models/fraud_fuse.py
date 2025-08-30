from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import LABELS
from .fraud_group_detector import (
    EMB_MODEL,
    business_cohesion_metrics,
    score_fraud_for_business,
    fuse_fraud_scores,
)


@dataclass
class FraudFuseConfig:
    """Configuration for fraud-fusion wrapper around a base text model.

    base_family: one of ("encoder", "baseline", "sft")
    base_model_rel: relative path under models_dir to the base model directory
    """

    base_family: str
    base_model_rel: str
    emb_model: str = EMB_MODEL
    sim_threshold: float = 0.85
    min_samples: int = 2
    topk: int = 5
    dup_threshold: float = 0.92
    w_text: float = 0.7
    w_fraud: float = 0.3


def _load_rows(jsonl_path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _biz_key(r: Dict) -> str:
    # Prefer an explicit id if present, else name; fallback to empty string
    bid = r.get("business_id")
    if bid is None or str(bid) == "":
        bid = r.get("business_name", "")
    return str(bid)


def train_fraud_fuse(
    out_dir: str,
    base_model_rel: str,
    base_family: str,
    cfg_overrides: Optional[dict] = None,
) -> None:
    """"Train" a fraud-fusion wrapper by writing its config manifest.

    This is a lightweight wrapper without weights. It references the base text model
    and stores fusion parameters and thresholds.
    """
    cfg_overrides = cfg_overrides or {}
    fam = base_family.strip().lower()
    if fam not in {"encoder", "baseline", "sft"}:
        raise ValueError("base_family must be one of: encoder | baseline | sft")
    cfg = FraudFuseConfig(base_family=fam, base_model_rel=str(base_model_rel))
    for k, v in cfg_overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Persist config
    (out / "fraud_fuse.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    # Write a manifest as model.bin so downstream tooling finds a primary artifact
    manifest = {
        "note": "Fraud-fusion wrapper; see fraud_fuse.json",
        "base_family": cfg.base_family,
        "base_model_rel": cfg.base_model_rel,
    }
    (out / "model.bin").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _predict_base_probs(
    base_dir: Path, fam: str, jsonl_path: str, batch_size: int = 32
) -> np.ndarray:
    fam = fam.strip().lower()
    if fam == "encoder":
        from .bert_classifier import predict_probs_bert

        return predict_probs_bert(str(base_dir), jsonl_path, batch_size=batch_size)
    if fam == "baseline":
        from .rnn_classifier import load_rnn, predict_probs_rnn

        model = load_rnn(str(base_dir / "model.pt"))
        return predict_probs_rnn(model, jsonl_path, batch_size=batch_size)
    if fam == "sft":
        from .sft_lora import predict_probs_sft

        return predict_probs_sft(str(base_dir), jsonl_path, batch_size=batch_size)
    raise ValueError(f"Unsupported base_family: {fam}")


def predict_probs_fraud_fuse(
    model_dir: str,
    jsonl_path: str,
    batch_size: int = 32,
    write_artifacts_dir: Optional[str] = None,
    models_root: Optional[str] = None,
) -> np.ndarray:
    """Predict probabilities with base model and compute fraud scores as side output.

    Returns y_prob with the same shape as base model (len(LABELS)). Fraud signals are
    written to write_artifacts_dir if provided.
    """
    mdir = Path(model_dir)
    cfg_path = mdir / "fraud_fuse.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing fraud_fuse.json under {model_dir}")
    cfg = FraudFuseConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))

    rows = _load_rows(jsonl_path)
    # Base predictions
    base_root = Path(models_root) if models_root else Path("models")
    base_dir = base_root / cfg.base_model_rel
    if not base_dir.exists():
        # Also try absolute path in case config stored full path
        abs_try = Path(cfg.base_model_rel)
        if abs_try.exists():
            base_dir = abs_try
        else:
            raise FileNotFoundError(
                f"Base model directory not found: {base_dir} (from {cfg.base_model_rel})"
            )
    y_prob = _predict_base_probs(base_dir, cfg.base_family, jsonl_path, batch_size=batch_size)

    # Group by business and compute fraud scores + cohesion
    biz_to_indices: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        biz_to_indices.setdefault(_biz_key(r), []).append(i)

    fraud_scores = np.zeros(len(rows), dtype=float)
    biz_metrics: Dict[str, Dict[str, float]] = {}
    for biz, idxs in biz_to_indices.items():
        group = [rows[i] for i in idxs]
        # Per-review cluster score
        scores = score_fraud_for_business(
            group,
            sim_threshold=float(cfg.sim_threshold),
            min_samples=int(cfg.min_samples),
        )
        # Business cohesion metrics
        metrics = business_cohesion_metrics(
            group, topk=int(cfg.topk), dup_threshold=float(cfg.dup_threshold)
        )
        biz_metrics[biz] = metrics
        # Scale scores by cohesion (mean_topk) to emphasize very cohesive businesses
        scale = float(metrics.get("mean_topk", 0.0))
        scaled = np.clip(np.asarray(scores) * scale, 0.0, 1.0)
        for j, i in enumerate(idxs):
            fraud_scores[i] = float(scaled[j])

    # Optionally write artifacts
    if write_artifacts_dir:
        outd = Path(write_artifacts_dir)
        outd.mkdir(parents=True, exist_ok=True)
        # Review-level fraud scores
        with open(outd / "fraud_scores.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(
                    json.dumps(
                        {
                            "i": int(i),
                            "business": _biz_key(r),
                            "fraud_score": float(fraud_scores[i]),
                        }
                    )
                    + "\n"
                )
        # Business-level cohesion metrics
        (outd / "cohesion.json").write_text(
            json.dumps(biz_metrics, indent=2), encoding="utf-8"
        )

        # If a "fraudulent_review" label existed, we could fuse here; preserve shape
        # in this codebase (LABELS excludes it), so we only write the fraud score sidecar.

    return y_prob

