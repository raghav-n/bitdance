from __future__ import annotations

"""Small helpers for building artifact paths from config params."""

from typing import Dict


def _get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def slugify(s: str) -> str:
    return (
        (s or "").strip().lower().replace(" ", "_").replace("/", "-").replace("\\", "-")
    )


def data_dir(p: Dict) -> str:
    return _get(p, "project", "data_dir", default="data")


def models_dir(p: Dict) -> str:
    return _get(p, "project", "models_dir", default="models")


def seed(p: Dict) -> int:
    return int(
        _get(p, "featurize", "seed", default=_get(p, "project", "seed", default=42))
    )


def dataset_slug(p: Dict) -> str:
    return slugify(_get(p, "dataset", "slug", default="base"))


def augment_slug(p: Dict) -> str:
    return slugify(_get(p, "augment", "slug", default="none"))


def model_family(p: Dict) -> str:
    return slugify(_get(p, "train", "family", default="encoder"))


def model_name_slug(p: Dict) -> str:
    return slugify(_get(p, "train", "model_name", default="xlm-roberta-base"))


def train_run_name(p: Dict) -> str:
    return slugify(
        _get(
            p, "train", "run_name", default=_get(p, "runtime", "run_id", default="run")
        )
    )
