from typing import List, Dict, Tuple
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_text(r: Dict) -> str:
    """Build text for embedding.

    Important: avoid including business/author fields here to prevent leakage that
    inflates within-business similarity. Focus on the review text itself.
    """
    return str(r.get("text", "") or "")


@lru_cache(maxsize=1)
def _get_model(name: str = EMB_MODEL) -> SentenceTransformer:
    return SentenceTransformer(name)


def score_fraud_for_business(
    reviews: List[Dict],
    sim_threshold: float = 0.85,
    min_samples: int = 2,
    batch_size: int = 64,
) -> List[float]:
    """Score per-review suspiciousness from within-business similarity clusters.

    Uses DBSCAN over cosine metric directly on normalized embeddings and computes
    each review's mean similarity to other members in its cluster (excluding self).
    Noise points receive 0.0. Scores are clipped to [0,1] but not min–max scaled
    per-group, preserving absolute similarity signal.
    """
    if not reviews:
        return []
    model = _get_model()
    X = model.encode(
        [make_text(r) for r in reviews],
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    # DBSCAN on cosine metric avoids explicit NxN distances
    eps = max(0.0, min(1.0, 1.0 - float(sim_threshold)))
    labels = DBSCAN(
        eps=eps, min_samples=int(min_samples), metric="cosine", n_jobs=-1
    ).fit_predict(X)

    scores = np.zeros(len(reviews), dtype=float)
    # Compute cluster-wise similarities
    for lbl in set(labels) - {-1}:
        idx = np.where(labels == lbl)[0]
        Xi = X[idx]
        # Cosine sim since embeddings are normalized
        S = Xi @ Xi.T
        # exclude self-similarity
        np.fill_diagonal(S, np.nan)
        cluster_scores = np.nanmean(S, axis=1)
        # Clip scores to [0,1]
        scores[idx] = np.clip(cluster_scores, 0.0, 1.0)
    return scores.tolist()


def business_cohesion_metrics(
    reviews: List[Dict],
    topk: int = 5,
    dup_threshold: float = 0.92,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute business-level cohesion metrics from review embeddings.

    - mean_topk: mean of top-k cosine similarities per review (excluding self)
    - dup_rate: fraction of (i,j) pairs with cosine similarity > dup_threshold
    """
    n = len(reviews)
    if n == 0:
        return {"mean_topk": 0.0, "dup_rate": 0.0}
    model = _get_model()
    X = model.encode(
        [make_text(r) for r in reviews],
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    # Cosine sim matrix via dot product as embeddings are normalized
    S = X @ X.T
    # Exclude self from top-k
    np.fill_diagonal(S, -np.inf)
    k = int(max(1, min(topk, n - 1)))
    # Take top-k per row
    topk_sim = np.partition(S, -k, axis=1)[:, -k:]
    mean_topk = float(np.mean(topk_sim)) if topk_sim.size else 0.0
    # dup rate over all ordered pairs (i != j)
    dup_cnt = int(np.sum(S > float(dup_threshold)))
    denom = max(1, n * (n - 1))
    dup_rate = float(dup_cnt) / float(denom)
    return {"mean_topk": mean_topk, "dup_rate": dup_rate}


def fuse_fraud_scores(
    p_text: np.ndarray,
    fraud_scores: List[float],
    w_text: float = 0.7,
    w_fraud: float = 0.3,
) -> np.ndarray:
    """Fuse text probability for fraudulent_review with group fraud score.
    Args:
        p_text: shape (N,) text-only probability for fraudulent_review
        fraud_scores: shape (N,) group similarity-derived score in [0,1]
    Returns: fused probability in [0,1]
    """
    p_text = np.asarray(p_text).reshape(-1)
    fs = np.asarray(fraud_scores).reshape(-1)
    if p_text.shape != fs.shape:
        raise ValueError(f"Shape mismatch: p_text{p_text.shape} vs fraud_scores{fs.shape}")
    w = w_text + w_fraud
    return np.clip((w_text * p_text + w_fraud * fs) / (w if w > 0 else 1.0), 0.0, 1.0)
