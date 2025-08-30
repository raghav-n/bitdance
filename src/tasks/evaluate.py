"""Evaluation task.

Evaluates one or more trained models on the test split produced by normalize.
Accepts model directories relative to `models_dir`, e.g. ["encoder/20250830-112017"].
Infers trainer by inspecting the model path (encoder | baseline | sft).
Exports per-label confusion matrices and metrics to reports/metrics/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..orchestrator import task
from ..orchestrator.logging import get_logger
from ..orchestrator.utils import (
    data_dir,
    models_dir,
    dataset_slug,
    model_family,
    train_run_name,
    slugify,
)
from ..models import LABELS
from ..models.eval_utils import evaluate_probs

from dotenv import load_dotenv

load_dotenv()


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _y_true_from_rows(rows: List[Dict]) -> np.ndarray:
    Y = []
    for r in rows:
        Y.append([float(bool(r.get(name, False))) for name in LABELS])
    return np.array(Y, dtype=float)


def _confusion_per_label(
    y_true: np.ndarray, y_pred: np.ndarray
) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(int)
        yp = y_pred[:, i].astype(int)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        out.append({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return out


def _infer_family(model_rel: str) -> str:
    # heuristics by prefix
    if model_rel.startswith("encoder/"):
        return "encoder"
    if model_rel.startswith("baseline/"):
        return "baseline"
    if model_rel.startswith("sft/") or model_rel.startswith("sft-"):
        return "sft"
    if model_rel.startswith("fraud_fuse/") or model_rel.startswith("fraud/"):
        return "fraud_fuse"
    # fallback: look for characteristic files
    fam = model_rel.split("/")[0]
    return fam


def _predict_probs(
    model_dir: Path,
    fam: str,
    jsonl_path: Path,
    eval_cfg: dict | None = None,
    logger=None,
) -> np.ndarray:
    eval_cfg = eval_cfg or {}
    # Configure batch sizes per family with sensible defaults
    if fam == "encoder":
        from ..models.bert_classifier import predict_probs_bert  # lazy import

        bs = int(eval_cfg.get("encoder_batch_size", eval_cfg.get("batch_size", 32)))
        if logger:
            logger.info("Encoding in batches: size=%d", bs)
        return predict_probs_bert(str(model_dir), str(jsonl_path), batch_size=bs)
    if fam == "baseline":
        from ..models.rnn_classifier import load_rnn, predict_probs_rnn

        bs = int(eval_cfg.get("rnn_batch_size", eval_cfg.get("batch_size", 128)))
        log_every = int(eval_cfg.get("log_every", 50))
        last = {"n": 0}

        def _progress(n: int, total: int):
            if logger is None:
                return
            if n == total or (n - last["n"]) >= log_every:
                last["n"] = n
                logger.info("RNN eval progress: %d/%d", n, total)

        model = load_rnn(str(model_dir / "model.pt"))
        return predict_probs_rnn(model, str(jsonl_path), batch_size=bs, progress_callback=_progress)
    if fam == "sft":
        from ..models.sft_lora import predict_probs_sft

        bs = int(eval_cfg.get("sft_batch_size", eval_cfg.get("batch_size", 8)))
        log_every = int(eval_cfg.get("log_every", 10))
        last = {"n": 0}

        def _progress(n: int, total: int):
            if logger is None:
                return
            if n == total or (n - last["n"]) >= log_every:
                last["n"] = n
                logger.info("SFT eval progress: %d/%d", n, total)

        if logger:
            logger.info("Generating in batches: size=%d", bs)
        return predict_probs_sft(
            str(model_dir), str(jsonl_path), batch_size=bs, progress_callback=_progress
        )
    if fam == "fraud_fuse":
        from ..models.fraud_fuse import predict_probs_fraud_fuse  # lazy import

        bs = int(eval_cfg.get("batch_size", 32))
        # Write fraud artifacts next to predictions
        out_dir = eval_cfg.get("_write_dir")
        return predict_probs_fraud_fuse(
            str(model_dir), str(jsonl_path), batch_size=bs, write_artifacts_dir=out_dir, models_root=str(models_dir({}))
        )
    raise ValueError(f"Unknown model family for evaluation: {fam}")


def _outputs_from_models(p: dict, model_list: List[str]) -> List[str]:
    outs: List[str] = []
    for rel in model_list:
        ident = slugify(rel.replace("/", "-"))
        outs.append(f"reports/metrics/{ident}/metrics.json")
    return outs


@task(
    name="evaluate",
    inputs=lambda p: [
        f"{data_dir(p)}/processed/{dataset_slug(p)}/test.jsonl",
        "configs/base.yaml",
    ]
    + [
        f"{models_dir(p)}/{rel}"
        for rel in (
            p.get("evaluate", {}).get("models")
            or [f"{model_family(p)}/{train_run_name(p)}"]
        )
    ],
    outputs=lambda p: _outputs_from_models(
        p,
        (
            p.get("evaluate", {}).get("models")
            or [f"{model_family(p)}/{train_run_name(p)}"]
        ),
    ),
)
def evaluate(params: dict):
    """Evaluate specified models on the normalize-produced test set and write metrics."""
    eval_cfg = params.get("evaluate", {}) or {}
    threshold = float(eval_cfg.get("threshold", 0.5))
    logger = get_logger("tasks.evaluate")
    model_list: List[str] = eval_cfg.get("models") or [
        f"{model_family(params)}/{train_run_name(params)}"
    ]
    test_path = (
        Path(data_dir(params)) / "processed" / dataset_slug(params) / "test.jsonl"
    )
    logger.info(
        "Evaluating %d model(s) on %s (threshold=%.2f)", len(model_list), str(test_path), threshold
    )

    # Modal evaluation path
    use_modal = bool(eval_cfg.get("use_modal", False))
    if use_modal:
        # Evaluate on Modal using remote models in the Modal volume.
        # Launch one container per model concurrently (like training).
        try:
            import modal  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Modal not installed; set evaluate.use_modal=false or install modal."
            ) from e

        volume_name = str(
            eval_cfg.get(
                "modal_volume_name",
                (params.get("train", {}) or {}).get("modal_volume_name", "bitdance-models"),
            )
        )
        timeout_s = int(eval_cfg.get("modal_timeout_seconds", 7200))
        gpu_spec = str(eval_cfg.get("modal_gpu", "T4"))
        test_text = test_path.read_text(encoding="utf-8")

        logger.info(
            "Evaluating remotely on Modal: %d model(s), volume=%s, gpu=%s, timeout=%ds",
            len(model_list), volume_name, gpu_spec, timeout_s,
        )

        def _evaluate_one_remote(rel: str) -> tuple[str, dict, bytes | None]:
            image = (
                modal.Image.debian_slim()
                .pip_install_from_requirements("requirements.txt")
                .pip_install_from_requirements("requirements_models.txt")
                .pip_install("huggingface_hub[hf_transfer]")
                .env(
                    {
                        "HF_HUB_ENABLE_HF_TRANSFER": "1",
                        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                    }
                )
                .add_local_dir(".", "/workspace", ignore=[".git", ".DS_Store", "__pycache__"])
            )
            ident = slugify(rel.replace("/", "-"))
            app = modal.App(name=f"bitdance-eval-{ident}", image=image)
            vol = modal.Volume.from_name(volume_name, create_if_missing=True)

            @app.function(
                gpu=gpu_spec,
                volumes={"/vol": vol},
                serialized=True,
                timeout=timeout_s,
            )
            def remote_eval_one(model_rel: str, test_jsonl: str, thr: float, batch_size: int) -> str:
                import json as _json
                import numpy as _np
                import sys as _sys
                import os

                _sys.path.insert(0, "/workspace")
                from ..models import LABELS as _LABELS  # type: ignore
                from ..models.eval_utils import evaluate_probs as _evaluate_probs  # type: ignore
                from sklearn.metrics import precision_recall_fscore_support as _prf

                def _infer_family(_rel: str) -> str:
                    if _rel.startswith("encoder/"):
                        return "encoder"
                    if _rel.startswith("baseline/"):
                        return "baseline"
                    if _rel.startswith("sft/") or _rel.startswith("sft-"):
                        return "sft"
                    return _rel.split("/")[0]

                fam = _infer_family(model_rel)
                base = "/vol/" + (
                    "models/" + model_rel if not model_rel.startswith("models/") else model_rel
                )
                ident_local = model_rel.replace("/", "-")
                pred_dir = f"/vol/reports/metrics/{ident_local}"
                os.makedirs(pred_dir, exist_ok=True)

                # Prepare test set
                rows = [_json.loads(ln) for ln in test_jsonl.splitlines() if ln.strip()]
                y_true = _np.array(
                    [[float(bool(r.get(name, False))) for name in _LABELS] for r in rows],
                    dtype=float,
                )
                # Write to temp path for predictors that expect a filepath
                with open("/tmp/test.jsonl", "w", encoding="utf-8") as _f:
                    _f.write(test_jsonl)

                if fam == "encoder":
                    from ..models.bert_classifier import predict_probs_bert as _ppb
                    y_prob = _ppb(base, "/tmp/test.jsonl", batch_size=batch_size)
                    # Write predictions JSONL (no raw text)
                    with open(f"{pred_dir}/predictions.jsonl", "w", encoding="utf-8") as _wf:
                        for i in range(y_prob.shape[0]):
                            row = {
                                "i": int(i),
                                "probs": [float(x) for x in y_prob[i].tolist()],
                                "pred": [int(x) for x in (y_prob[i] >= thr).astype(int).tolist()],
                            }
                            _wf.write(_json.dumps(row) + "\n")
                elif fam == "baseline":
                    from ..models.rnn_classifier import (
                        load_rnn as _load_rnn,
                        predict_probs_rnn as _ppr,
                    )

                    model = _load_rnn(base + "/model.pt")
                    y_prob = _ppr(model, "/tmp/test.jsonl", batch_size=batch_size)
                    with open(f"{pred_dir}/predictions.jsonl", "w", encoding="utf-8") as _wf:
                        for i in range(y_prob.shape[0]):
                            row = {
                                "i": int(i),
                                "probs": [float(x) for x in y_prob[i].tolist()],
                                "pred": [int(x) for x in (y_prob[i] >= thr).astype(int).tolist()],
                            }
                            _wf.write(_json.dumps(row) + "\n")
                elif fam == "sft":
                    from ..models.sft_lora import predict_probs_sft as _pps
                    # Write raw outputs while predicting
                    y_prob = _pps(base, "/tmp/test.jsonl", batch_size=batch_size, raw_out_path=f"{pred_dir}/predictions.jsonl")
                elif fam == "fraud_fuse":
                    from ..models.fraud_fuse import predict_probs_fraud_fuse as _ppf
                    # Write label predictions and fraud sidecars
                    y_prob = _ppf(base, "/tmp/test.jsonl", batch_size=batch_size, write_artifacts_dir=pred_dir, models_root="/vol/models")
                else:
                    raise ValueError(f"Unknown family: {fam}")

                y_pred = (y_prob >= thr).astype(int)
                per_label = _evaluate_probs(y_true, y_prob, _LABELS)
                p_macro, r_macro, f1_macro, _ = _prf(
                    y_true, y_pred, average="macro", zero_division=0
                )
                p_micro, r_micro, f1_micro, _ = _prf(
                    y_true, y_pred, average="micro", zero_division=0
                )
                conf = []
                for i in range(y_true.shape[1]):
                    yt = y_true[:, i].astype(int)
                    yp = y_pred[:, i].astype(int)
                    tp = int((_np.logical_and(yt == 1, yp == 1)).sum())
                    tn = int((_np.logical_and(yt == 0, yp == 0)).sum())
                    fp = int((_np.logical_and(yt == 0, yp == 1)).sum())
                    fn = int((_np.logical_and(yt == 1, yp == 0)).sum())
                    conf.append({"tn": tn, "fp": fp, "fn": fn, "tp": tp})

                out = {
                    "model": model_rel,
                    "family": fam,
                    "labels": _LABELS,
                    "threshold": thr,
                    "n_samples": int(y_true.shape[0]),
                    "per_label": {
                        _LABELS[i]: {
                            **per_label[_LABELS[i]],
                            "support": int((y_true[:, i] == 1).sum()),
                            "confusion": conf[i],
                        }
                        for i in range(len(_LABELS))
                    },
                    "macro": {
                        "precision": float(p_macro),
                        "recall": float(r_macro),
                        "f1": float(f1_macro),
                    },
                    "micro": {
                        "precision": float(p_micro),
                        "recall": float(r_micro),
                        "f1": float(f1_micro),
                    },
                }
                return _json.dumps(out)

            @app.function(volumes={"/vol": vol}, serialized=True, timeout=timeout_s)
            def read_predictions(model_rel: str) -> bytes:
                ident_local = model_rel.replace("/", "-")
                path = f"/vol/reports/metrics/{ident_local}/predictions.jsonl"
                try:
                    with open(path, "rb") as f:
                        return f.read()
                except Exception:
                    return b""

            with app.run():
                bs = int(
                    eval_cfg.get(
                        "batch_size",
                        8,
                    )
                )
                payload = remote_eval_one.remote(rel, test_text, threshold, bs)
                preds_bytes = read_predictions.remote(rel)
            return rel, json.loads(payload), preds_bytes

        # Fan out across models using a thread pool
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: dict[str, dict] = {}
        preds_blobs: dict[str, bytes] = {}
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_evaluate_one_remote, rel) for rel in model_list]
            for fut in as_completed(futures):
                rel, metrics, blob = fut.result()
                results[rel] = metrics
                preds_blobs[rel] = blob or b""
                logger.info("Remote eval completed: %s", rel)

        for rel, metrics in results.items():
            ident = slugify(rel.replace("/", "-"))
            out_dir = Path("reports/metrics") / ident
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            # Write predictions if available
            blob = preds_blobs.get(rel)
            if blob:
                (out_dir / "predictions.jsonl").write_bytes(blob)
        return

    # Local evaluation path
    rows = _load_jsonl(test_path)
    y_true = _y_true_from_rows(rows)
    logger.info("Loaded test set: %d rows", len(rows))
    for rel in model_list:
        fam = _infer_family(rel)
        logger.info("Evaluating model: %s (family=%s)", rel, fam)
        model_dir = Path(models_dir(params)) / rel
        # Prepare output dir early to store predictions
        ident = slugify(rel.replace("/", "-"))
        out_dir = Path("reports/metrics") / ident
        out_dir.mkdir(parents=True, exist_ok=True)
        if fam == "sft":
            from ..models.sft_lora import predict_probs_sft as _pps  # type: ignore
            bs = int(eval_cfg.get("sft_batch_size", eval_cfg.get("batch_size", 8)))
            y_prob = _pps(str(model_dir), str(test_path), batch_size=bs, raw_out_path=str(out_dir / "predictions.jsonl"))
        else:
            # Provide output dir to predictors that may write side artifacts (fraud_fuse)
            _cfg = dict(eval_cfg)
            _cfg["_write_dir"] = str(out_dir)
            y_prob = _predict_probs(model_dir, fam, test_path, eval_cfg=_cfg, logger=logger)
            # For non-SFT, write a simple predictions file (no raw text)
            with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
                for i in range(y_prob.shape[0]):
                    row = {
                        "i": int(i),
                        "probs": [float(x) for x in y_prob[i].tolist()],
                        "pred": [int(x) for x in (y_prob[i] >= threshold).astype(int).tolist()],
                    }
                    f.write(json.dumps(row) + "\n")
        y_pred = (y_prob >= threshold).astype(int)
        per_label = evaluate_probs(y_true, y_prob, LABELS)
        from sklearn.metrics import precision_recall_fscore_support

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        conf = _confusion_per_label(y_true, y_pred)
        metrics = {
            "model": rel,
            "family": fam,
            "labels": LABELS,
            "threshold": threshold,
            "n_samples": int(y_true.shape[0]),
            "per_label": {
                LABELS[i]: {
                    **per_label[LABELS[i]],
                    "support": int(np.sum(y_true[:, i] == 1)),
                    "confusion": conf[i],
                }
                for i in range(len(LABELS))
            },
            "macro": {
                "precision": float(p_macro),
                "recall": float(r_macro),
                "f1": float(f1_macro),
            },
            "micro": {
                "precision": float(p_micro),
                "recall": float(r_micro),
                "f1": float(f1_micro),
            },
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        logger.info(
            "Wrote metrics to %s (macro F1=%.4f, micro F1=%.4f)",
            str(out_dir / "metrics.json"),
            float(metrics["macro"]["f1"]),
            float(metrics["micro"]["f1"]),
        )
