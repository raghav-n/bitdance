"""LLM-driven augmentation task.

Generates synthetic reviews per policy class using Gemini, guided by the
policy definitions in `annotate.py` (PolicyDetector + few-shot examples).

Inputs:
- data/interim/{dataset.slug}/reviews.parquet (optional; used for sampling metadata)
- configs/base.yaml

Outputs:
- data/augmented/{dataset.slug}/{augment.slug}/reviews.parquet
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..orchestrator.logging import get_logger

from ..orchestrator import task
from ..orchestrator.utils import data_dir, dataset_slug, augment_slug

from dotenv import load_dotenv

load_dotenv()

logger = get_logger("tasks.augment")


def _load_base_reviews(interim_path: Path) -> pd.DataFrame:
    try:
        if interim_path.exists():
            return pd.read_parquet(interim_path)
    except Exception:
        pass
    # Fallback: empty frame with expected columns
    return pd.DataFrame(
        columns=[
            "review_id",
            "place_name",
            "user_name",
            "text",
            "language",
            "rating",
            "has_image",
            "image_path",
            "metadata",
        ]
    )


def _pick_meta(base_df: pd.DataFrame) -> dict:
    """Pick plausible metadata for a synthetic sample from base data or fabricate."""
    if len(base_df) > 0:
        r = base_df.sample(1).iloc[0].to_dict()
        return {
            "place_name": r.get("place_name") or "synthetic_place",
            "user_name": r.get("user_name") or f"user_{random.randint(1000, 9999)}",
        }
    return {
        "place_name": "synthetic_place",
        "user_name": f"user_{random.randint(1000, 9999)}",
    }


def _ensure_imports():
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "google-generativeai is required for augmentation. Add it to requirements and install."
        ) from e
    return genai


def _safe_text(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t.strip("`")
    if t.endswith("```"):
        t = t[:-3].strip()
    return t.strip().strip('"')


def _build_gen_prompt(
    policy_name: str,
    policy_description: str,
    language: str,
    place_name: str,
    example_review: str | None,
) -> str:
    prompt = (
        "You are generating a single synthetic Google-style location review.\n"
        f"Target policy to VIOLATE: {policy_name}.\n"
        f"Policy description: {policy_description}\n"
    )
    if example_review:
        prompt += (
            "Here is ONE example of a violating review (for guidance only):\n"
            f"Example: {example_review}\n"
            "Do NOT copy or paraphrase this example. Produce a distinct review.\n"
        )
    prompt += (
        "Write a plausible review that clearly violates this policy while remaining realistic.\n"
        "Constraints:\n"
        f"- Language: {language}\n"
        f"- Mention the place name '{place_name}' naturally.\n"
        "- Length: 1-3 sentences.\n"
        "- No disclaimers, no meta commentary, no quotes or code blocks.\n"
        "Output: Only the review text."
    )
    return prompt


@task(
    name="augment",
    inputs=lambda p: [
        "data/annotated/restaurant_reviews/annotations.parquet",
        "data/annotated/restaurant_reviews/annotations.csv",
        "configs/base.yaml",
    ],
    outputs=lambda p: [
        "data/augmented/restaurant_reviews/reviews.parquet",
        "data/augmented/restaurant_reviews/reviews.parquet",
    ],
)
def augment(params: dict):
    """Create synthetic reviews to balance underrepresented classes/languages.

    Config under `augment:` supports:
    - slug: str
    - classes: mapping of policy_name -> count
    - total: optional int (informational)
    - languages: list[str] (defaults to ["en"])
    - llm: optional overrides { gemini_api_key, gemini_model, temperature, rate_limit }
    If llm.gemini_api_key/model absent, falls back to annotate.llm.*
    """

    # Resolve paths
    interim_path = Path("data/annotated/restaurant_reviews/annotations.parquet")
    out_path = Path("data/augmented/restaurant_reviews/reviews.parquet")
    out_path_csv = out_path.with_suffix(".csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load base data (if present) for sampling place/user names
    base_df = _load_base_reviews(interim_path)

    # Resolve config
    aug_cfg: Dict = params.get("augment") or {}
    classes: Dict[str, int] = aug_cfg.get("classes", {}) or {}
    languages: List[str] = aug_cfg.get("languages", ["en"]) or ["en"]

    # LLM config: prefer augment.llm; fallback to annotate.llm
    llm_cfg = aug_cfg.get("llm") or {}
    ann_llm_cfg = (params.get("annotate") or {}).get("llm") or {}
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or llm_cfg.get("gemini_api_key")
        or ann_llm_cfg.get("gemini_api_key")
    )
    model_name = (
        llm_cfg.get("gemini_model")
        or ann_llm_cfg.get("gemini_model")
        or "gemini-2.5-flash-lite"
    )
    temperature = float(llm_cfg.get("temperature", 0.7))
    rate_limit = float(llm_cfg.get("rate_limit", ann_llm_cfg.get("rate_limit", 0.2)))

    if not api_key:
        raise ValueError(
            "Missing Gemini API key. Set augment.llm.gemini_api_key or annotate.llm.gemini_api_key in config."
        )

    # Import policy registry from annotate.py for consistency
    from .annotate import (
        POLICY_REGISTRY,
    )  # local import to avoid circulars at import time

    genai = _ensure_imports()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    rows = []
    rng = random.Random(int(params.get("project", {}).get("seed", 42)))

    # Iterate over requested classes
    for policy_name, n in classes.items():
        if policy_name not in POLICY_REGISTRY:
            raise ValueError(
                f"Unknown policy in augment.classes: {policy_name}. Known: {list(POLICY_REGISTRY.keys())}"
            )
        detector = POLICY_REGISTRY[policy_name]
        desc = detector.description()

        produced = 0
        attempts = 0
        max_attempts = n * 5  # generous cap with verification/retries
        while produced < n and attempts < max_attempts:
            attempts += 1
            lang = rng.choice(languages)
            meta = _pick_meta(base_df)
            # Pick a single few-shot example for guidance; do not copy
            ex = None
            try:
                exs = detector.few_shot_examples() or []
                if exs:
                    ex = rng.choice(exs).get("review")
            except Exception:
                ex = None
            prompt = _build_gen_prompt(policy_name, desc, lang, meta["place_name"], ex)
            try:
                resp = model.generate_content(
                    prompt, generation_config={"temperature": temperature}
                )
                text = _safe_text(resp.text or "")
            except Exception as exc:
                logger.exception("LLM error on attempt {attempt}")
                logger.exception(exc)
                # Back off slightly on transient API errors
                time.sleep(max(rate_limit, 0.1))
                continue

            review_id = f"aug-{augment_slug(params)}-{policy_name}-{produced + 1:05d}"
            rating = rng.randint(1, 5)
            row = {
                "review_id": review_id,
                "place_name": meta["place_name"],
                "user_name": meta["user_name"],
                "text": text,
                "language": lang,
                "rating": rating,
                "has_image": False,
                "image_path": "",
                "metadata": json.dumps(
                    {
                        "source": "synthetic",
                        "augment_slug": augment_slug(params),
                        "policy": policy_name,
                        "attempt": attempts,
                        "model": model_name,
                    }
                ),
            }
            rows.append(row)
            produced += 1

            time.sleep(rate_limit)

    df = pd.DataFrame(rows)
    # Ensure stable dtypes
    for col in [
        "review_id",
        "place_name",
        "user_name",
        "text",
        "language",
        "image_path",
        "metadata",
    ]:
        df[col] = df[col].astype(str)
    df["rating"] = df["rating"].astype(int)
    df["has_image"] = df["has_image"].astype(bool)

    df.to_csv(out_path_csv, index=False)
    df.to_parquet(out_path, index=False)
    return
