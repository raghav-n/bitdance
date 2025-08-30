"""Training task implementation.

Integrates baseline (RNN), encoder (BERT family), and SFT (7B + LoRA)
training into the pipeline using config under `train:` in configs/base.yaml.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Optional

from ..orchestrator import task
from ..orchestrator.logging import get_logger
from ..orchestrator.utils import (
    data_dir,
    models_dir,
    seed,
    model_family,
    train_run_name,
    dataset_slug,
    slugify,
)

# Import model trainers from top-level models package
from ..models import LABELS

from dotenv import load_dotenv

load_dotenv()


def _ensure_label_map(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map = {name: i for i, name in enumerate(LABELS)}
    path = out_dir / "label_map.json"
    path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    return path


def _pick_jsonl_from_split(split_file: Path) -> Optional[Path]:
    """Attempt to read a split file (txt) which may contain a path to a JSONL file.
    Returns the first non-empty line as Path if it exists.
    """
    try:
        lines = [
            ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines()
        ]
        for ln in lines:
            if not ln:
                continue
            p = Path(ln)
            if p.exists():
                return p
    except FileNotFoundError:
        return None
    return None


def _copy_primary_weight(source_dir: Path, target_bin: Path) -> None:
    """Copy a primary model weight file into target_bin.
    Tries common filenames produced by various trainers.
    """
    import shutil

    candidates = [
        source_dir / "pytorch_model.bin",
        source_dir / "adapter_model.safetensors",
        source_dir / "model.safetensors",
        source_dir / "model.pt",
    ]
    for c in candidates:
        if c.exists():
            shutil.copy2(c, target_bin)
            return
    # Fallback: write a manifest if no known weight file exists
    manifest = {
        "note": "No known weight file found; refer to this directory for model artifacts.",
        "dir": str(source_dir),
    }
    target_bin.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _train_via_modal(
    fam: str,
    train_jsonl_path: str,
    val_jsonl_path: str,
    rel_subdir: str,
    out_dir_local: Path,
    tcfg: dict,
    gpu_spec: str,
    app_name: str = "bitdance-train",
    volume_name: str = "bitdance-models",
):
    """Run training remotely on Modal with a GPU and persist artifacts to a Modal Volume.

    Writes a local manifest model.bin pointing to remote artifacts in the Modal volume.
    """
    try:
        import modal  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Modal is not installed. Set train.use_modal=false or install modal."
        ) from e
    # Logger
    from ..orchestrator.logging import (
        get_logger as _get_logger,
    )  # local import to avoid circulars

    log = _get_logger("tasks.train")

    # Modal 1.0: add local source via Image.add_local_dir / add_local_python_source
    image = (
        modal.Image.debian_slim()
        .pip_install_from_requirements("requirements.txt")
        .pip_install_from_requirements("requirements_models.txt")
        .pip_install("huggingface_hub[hf_transfer]")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_TOKEN": os.getenv("HF_TOKEN", "")})
        .add_local_dir(".", "/workspace", ignore=[".git", ".DS_Store", "__pycache__"])
    )
    app = modal.App(name=app_name, image=image)
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)

    timeout_s = int(tcfg.get("modal_timeout_seconds", 86400))  # default 24h

    @app.function(
        gpu=gpu_spec, volumes={"/vol": vol}, serialized=True, timeout=timeout_s
    )
    def run_remote(
        train_text: str, val_text: str, fam: str, tcfg_payload: str, rel_subdir: str
    ) -> str:
        import os, sys, json

        sys.path.insert(0, "/workspace")
        os.makedirs("/tmp/data", exist_ok=True)
        with open("/tmp/data/train.jsonl", "w", encoding="utf-8") as f:
            f.write(train_text)
        with open("/tmp/data/val.jsonl", "w", encoding="utf-8") as f:
            f.write(val_text)
        out_dir = os.path.join("/vol", rel_subdir)
        os.makedirs(out_dir, exist_ok=True)
        tcfg = json.loads(tcfg_payload)
        patience = int(tcfg.get("early_stopping_patience", 2))
        min_delta = float(tcfg.get("early_stopping_min_delta", 0.0))
        if fam == "baseline":
            from ..models.rnn_classifier import train_rnn_jsonl  # type: ignore

            train_rnn_jsonl(
                "/tmp/data/train.jsonl",
                "/tmp/data/val.jsonl",
                os.path.join(out_dir, "model.pt"),
                epochs=int(tcfg.get("epochs", 5)),
                batch_size=int(tcfg.get("batch_size", 64)),
                lr=float(tcfg.get("lr", 1e-3)),
                early_stopping_patience=patience,
                early_stopping_min_delta=min_delta,
            )
        elif fam == "encoder":
            from ..models.bert_classifier import train_bert, BertConfig  # type: ignore

            cfg = BertConfig(
                model_name=tcfg.get("model_name", "distilbert-base-uncased"),
                max_length=int(tcfg.get("max_length", 256)),
                lr=float(tcfg.get("lr", 2e-5)),
                epochs=int(tcfg.get("epochs", 3)),
                batch_size=int(tcfg.get("batch_size", 32)),
            )
            train_bert(
                "/tmp/data/train.jsonl",
                "/tmp/data/val.jsonl",
                out_dir,
                cfg,
                early_stopping_patience=patience,
                early_stopping_min_delta=min_delta,
            )
        elif fam == "sft":
            from ..models.sft_lora import train_sft, SFTCfg  # type: ignore

            cfg = SFTCfg()
            if tcfg.get("model_name"):
                cfg.base_model = tcfg["model_name"]
            if "epochs" in tcfg:
                cfg.epochs = int(tcfg["epochs"])
            if "lr" in tcfg:
                cfg.lr = float(tcfg["lr"])
            if "oversample" in tcfg:
                cfg.oversample = bool(tcfg["oversample"])
            train_sft(
                "/tmp/data/train.jsonl",
                "/tmp/data/val.jsonl",
                out_dir,
                cfg,
                early_stopping_patience=patience,
                early_stopping_min_delta=min_delta,
            )
        else:
            raise ValueError(f"Unknown train.family: {fam}")
        # Return remote out dir for discovery
        return out_dir

    train_text = Path(train_jsonl_path).read_text(encoding="utf-8")
    val_text = Path(val_jsonl_path).read_text(encoding="utf-8")

    import json as _json

    # Run the function within an active Modal app context
    @app.function(
        volumes={"/vol": vol}, serialized=True, timeout=max(900, min(timeout_s, 7200))
    )
    def fetch_remote_tar(rel_subdir: str) -> bytes:
        import io, os, tarfile

        base = os.path.join("/vol", rel_subdir)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(base, arcname=os.path.basename(base))
        buf.seek(0)
        return buf.read()

    log.info(
        "Submitting remote training to Modal (GPU=%s) for %s", gpu_spec, rel_subdir
    )
    # Opt-in download: only fetch artifacts if explicitly requested in config
    download_artifacts = bool(tcfg.get("modal_download_artifacts", False))
    with app.run():
        remote_out_dir = run_remote.remote(
            train_text, val_text, fam, _json.dumps(tcfg or {}), rel_subdir
        )
        if download_artifacts:
            log.info(
                "Remote training finished. Fetching artifacts from %s", remote_out_dir
            )
            tar_bytes = fetch_remote_tar.remote(rel_subdir)

    if download_artifacts:
        # Extract tarball locally into out_dir_local
        import io as _io, tarfile as _tarfile, tempfile as _tempfile, shutil as _shutil, os as _os

        with _tempfile.TemporaryDirectory() as tmpd:
            tmp_tar = Path(tmpd) / "artifacts.tar.gz"
            tmp_tar.write_bytes(tar_bytes)
            with _tarfile.open(tmp_tar, mode="r:gz") as tar:
                tar.extractall(path=tmpd)
            # Find extracted directory (first top-level)
            entries = [p for p in Path(tmpd).iterdir() if p.is_dir()]
            if not entries:
                raise RuntimeError("No artifacts found in remote tarball")
            extracted_dir = entries[0]
            # Copy contents into out_dir_local
            out_dir_local.mkdir(parents=True, exist_ok=True)
            for item in extracted_dir.iterdir():
                dst = out_dir_local / item.name
                if item.is_dir():
                    if dst.exists():
                        _shutil.rmtree(dst)
                    _shutil.copytree(item, dst)
                else:
                    _shutil.copy2(item, dst)

        # Ensure model.bin points to a primary weight if present
        _copy_primary_weight(out_dir_local, out_dir_local / "model.bin")
    else:
        # No download: write a manifest as model.bin for downstream reference
        out_dir_local.mkdir(parents=True, exist_ok=True)
        (out_dir_local / "model.bin").write_text(
            _json.dumps(
                {
                    "note": "Remote artifacts only; enable train.modal_download_artifacts to fetch locally",
                    "remote_out_dir": remote_out_dir,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Write a manifest file recording remote location
    out_dir_local.mkdir(parents=True, exist_ok=True)
    manifest = {
        "modal": True,
        "app": app_name,
        "volume": volume_name,
        "gpu": gpu_spec,
        "remote_out_dir": remote_out_dir,
    }
    (out_dir_local / "remote_manifest.json").write_text(
        _json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log.info("Pulled remote artifacts to %s", out_dir_local)


def _write_fraud_fuse_to_modal(
    rel_subdir: str,
    cfg_payload: str,
    label_map_json: str,
    app_name: str = "bitdance-train",
    volume_name: str = "bitdance-models",
) -> None:
    """Create a fraud_fuse wrapper directory in the Modal Volume with config + labels."""
    try:
        import modal  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Modal is not installed. Cannot write fraud_fuse wrapper to remote."
        ) from e

    app = modal.App(name=app_name)
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)

    @app.function(volumes={"/vol": vol}, serialized=True, timeout=600)
    def write_remote(rel_subdir: str, cfg_payload: str, label_map_json: str) -> str:
        import os, json

        out_dir = os.path.join("/vol", rel_subdir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "fraud_fuse.json"), "w", encoding="utf-8") as f:
            f.write(cfg_payload)
        with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
            f.write(label_map_json)
        manifest = {
            "note": "Fraud-fusion wrapper; see fraud_fuse.json",
        }
        with open(os.path.join(out_dir, "model.bin"), "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2))
        return out_dir

    with app.run():
        write_remote.remote(rel_subdir, cfg_payload, label_map_json)


def _pull_modal_artifacts(
    rel_subdir: str,
    out_dir_local: Path,
    app_name: str = "bitdance-train",
    volume_name: str = "bitdance-models",
) -> None:
    """Pull an existing model directory from a Modal Volume to local out_dir_local."""
    try:
        import modal  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Modal is not installed. Cannot pull remote artifacts."
        ) from e
    from ..orchestrator.logging import get_logger as _get_logger

    log = _get_logger("tasks.train")

    app = modal.App(name=app_name)
    vol = modal.Volume.from_name(volume_name, create_if_missing=False)

    @app.function(volumes={"/vol": vol}, serialized=True, timeout=3600)
    def fetch_remote_tar(rel_subdir: str) -> bytes:
        import io, os, tarfile

        base = os.path.join("/vol", rel_subdir)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(base, arcname=os.path.basename(base))
        buf.seek(0)
        return buf.read()

    log.info("Fetching existing remote artifacts from %s", rel_subdir)
    with app.run():
        tar_bytes = fetch_remote_tar.remote(rel_subdir)

    import tarfile as _tarfile, tempfile as _tempfile, shutil as _shutil

    with _tempfile.TemporaryDirectory() as tmpd:
        tmp_tar = Path(tmpd) / "artifacts.tar.gz"
        tmp_tar.write_bytes(tar_bytes)
        with _tarfile.open(tmp_tar, mode="r:gz") as tar:
            tar.extractall(path=tmpd)
        entries = [p for p in Path(tmpd).iterdir() if p.is_dir()]
        if not entries:
            raise RuntimeError("No artifacts found in remote tarball")
        extracted_dir = entries[0]
        out_dir_local.mkdir(parents=True, exist_ok=True)
        for item in extracted_dir.iterdir():
            dst = out_dir_local / item.name
            if item.is_dir():
                if dst.exists():
                    _shutil.rmtree(dst)
                _shutil.copytree(item, dst)
            else:
                _shutil.copy2(item, dst)
    _copy_primary_weight(out_dir_local, out_dir_local / "model.bin")
    log.info("Pulled remote artifacts to %s", out_dir_local)
    # label_map written earlier by caller


def _train_outputs(p: dict) -> list[str]:
    """Build output paths for single or multi-model configuration."""
    base_models_dir = models_dir(p)
    ds_slug = dataset_slug(p)  # keep for potential future use
    tcfg = p.get("train", {}) or {}
    items = tcfg.get("models")
    out: list[str] = []
    if isinstance(items, list) and items:
        base_run = train_run_name(p)
        for i, spec in enumerate(items):
            fam = slugify(
                str((spec or {}).get("family", tcfg.get("family", "encoder")).lower())
            )
            run_name = spec.get("run_name")
            if not run_name:
                mslug = spec.get("model_name") or spec.get("base_model") or f"m{i + 1}"
                run_name = f"{base_run}-{fam}-{slugify(str(mslug))}"
            run_name = slugify(str(run_name))
            out_dir = Path(base_models_dir) / fam / run_name
            out += [str(out_dir / "model.bin"), str(out_dir / "label_map.json")]
        return out
    # Fallback to single-model outputs
    fam = model_family(p)
    run = train_run_name(p)
    out_dir = Path(base_models_dir) / fam / run
    return [str(out_dir / "model.bin"), str(out_dir / "label_map.json")]


@task(
    name="train",
    inputs=lambda p: [
        f"{data_dir(p)}/processed/{dataset_slug(p)}/train.jsonl",
        f"{data_dir(p)}/processed/{dataset_slug(p)}/val.jsonl",
        "configs/base.yaml",
    ],
    outputs=_train_outputs,
)
def train(params: dict):
    """Train the selected model family and persist artifacts under models/.

    Looks for explicit JSONL paths in params.train.{train_jsonl,val_jsonl}. If not
    provided, attempts to read first path from split files. Saves a model weight
    snapshot to `model.bin` and label mapping to `label_map.json`.
    """
    log = get_logger("tasks.train")

    tcfg = params.get("train", {}) or {}

    def _trigger_evaluation(
        base_params: dict, fam: str, run_name: str, use_modal: bool
    ) -> None:
        try:
            from .evaluate import evaluate as _eval
        except Exception:  # noqa: BLE001
            log.warning("Evaluate task not available; skipping automatic evaluation.")
            return
        # Clone params and set the evaluate scope for this model
        p = dict(base_params)
        ev = dict(p.get("evaluate", {}) or {})
        ev["models"] = [f"{fam}/{run_name}"]
        if use_modal:
            ev["use_modal"] = True
            # Propagate Modal GPU and Volume if present in train config
            ev.setdefault("modal_gpu", str(tcfg.get("modal_gpu", "T4")))
            ev.setdefault(
                "modal_volume_name",
                str(tcfg.get("modal_volume_name", "bitdance-models")),
            )
        else:
            ev["use_modal"] = False
        p["evaluate"] = ev
        log.info("Evaluating model %s/%s (use_modal=%s)", fam, run_name, use_modal)
        _eval(params=p)

    def _resolve_data_paths(spec: dict) -> tuple[str, str]:
        t_j = spec.get("train_jsonl") or tcfg.get("train_jsonl")
        v_j = spec.get("val_jsonl") or tcfg.get("val_jsonl")
        if t_j and v_j:
            return str(t_j), str(v_j)
        # Default to normalize outputs
        ds = dataset_slug(params)
        base = Path(data_dir(params)) / "processed" / ds
        t_path = base / "train.jsonl"
        v_path = base / "val.jsonl"
        if t_path.exists() and v_path.exists():
            return str(t_path), str(v_path)
        # Fallback: attempt split-file hints if present
        splits_root = Path(data_dir(params)) / "splits" / str(seed(params))
        t_hint = _pick_jsonl_from_split(splits_root / "train.txt")
        v_hint = _pick_jsonl_from_split(splits_root / "val.txt")
        if t_hint and v_hint:
            return str(t_hint), str(v_hint)
        raise FileNotFoundError(
            "Training requires JSONL paths. Provide train.train_jsonl and train.val_jsonl in config, "
            "or ensure splits contain JSONL paths."
        )

    def _derive_run_name(spec: dict, idx: int) -> str:
        base_run = train_run_name(params)
        rn = spec.get("run_name")
        if rn:
            return slugify(str(rn))
        fam = slugify(str(spec.get("family", tcfg.get("family", "encoder")).lower()))
        mslug = spec.get("model_name") or spec.get("base_model") or f"m{idx + 1}"
        return slugify(f"{base_run}-{fam}-{mslug}")

    def _train_one(spec: dict, idx: int) -> None:
        fam = slugify(str(spec.get("family", tcfg.get("family", "encoder")).lower()))
        run_name = _derive_run_name(spec, idx)
        out_dir = Path(models_dir(params)) / fam / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _ensure_label_map(out_dir)

        patience = int(
            spec.get("early_stopping_patience", tcfg.get("early_stopping_patience", 2))
        )
        min_delta = float(
            spec.get(
                "early_stopping_min_delta", tcfg.get("early_stopping_min_delta", 0.0)
            )
        )
        train_jsonl, val_jsonl = _resolve_data_paths(spec)
        log.info(
            "Training [%d] family=%s run=%s -> %s", idx + 1, fam, run_name, out_dir
        )

        # Modal options
        use_modal = bool(spec.get("use_modal", tcfg.get("use_modal", False)))
        modal_gpu = str(spec.get("modal_gpu", tcfg.get("modal_gpu", "T4")))
        modal_vol = str(
            spec.get(
                "modal_volume_name", tcfg.get("modal_volume_name", "bitdance-models")
            )
        )
        modal_app = str(
            spec.get("modal_app_name", spec.get("run_name", tcfg.get("modal_app_name", "bitdance") + "-train"))
        )
        # fraud_scan is CPU-only and local; ignore use_modal for it
        if fam == "fraud_scan" and use_modal:
            log.info("family=fraud_scan ignores use_modal; running locally on CPU")
            use_modal = False

        if use_modal:
            rel_subdir = str(Path("models") / fam / run_name)
            if fam in {"fraud", "fraud_fuse"}:
                # Deploy wrapper to Modal volume (no GPU needed)
                from ..models import LABELS as _LBL  # local import to get labels
                label_map = {name: i for i, name in enumerate(_LBL)}
                overrides = {}
                for k in [
                    "emb_model",
                    "sim_threshold",
                    "min_samples",
                    "topk",
                    "dup_threshold",
                    "w_text",
                    "w_fraud",
                ]:
                    if k in spec:
                        overrides[k] = spec[k]
                    elif k in tcfg:
                        overrides[k] = tcfg[k]
                base_rel = str(
                    spec.get("base_model")
                    or spec.get("base_model_rel")
                    or tcfg.get("base_model")
                    or tcfg.get("base_model_rel", "")
                )
                if not base_rel:
                    raise ValueError(
                        "fraud_fuse requires base_model (relative to models_dir), e.g. 'encoder/enc-xlm'"
                    )
                base_fam = str(
                    spec.get("base_family")
                    or tcfg.get("base_family")
                    or (base_rel.split("/")[0] if "/" in base_rel else "encoder")
                )
                import json as _json
                cfg = {
                    "base_family": base_fam,
                    "base_model_rel": base_rel,
                    "emb_model": overrides.get("emb_model", "sentence-transformers/all-MiniLM-L6-v2"),
                    "sim_threshold": float(overrides.get("sim_threshold", 0.85)),
                    "min_samples": int(overrides.get("min_samples", 2)),
                    "topk": int(overrides.get("topk", 5)),
                    "dup_threshold": float(overrides.get("dup_threshold", 0.92)),
                    "w_text": float(overrides.get("w_text", 0.7)),
                    "w_fraud": float(overrides.get("w_fraud", 0.3)),
                }
                _write_fraud_fuse_to_modal(
                    rel_subdir=rel_subdir,
                    cfg_payload=_json.dumps(cfg, indent=2),
                    label_map_json=_json.dumps(label_map, indent=2),
                    app_name=modal_app,
                    volume_name=modal_vol,
                )
                # Also write local manifest outputs for downstream references
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "fraud_fuse.json").write_text(
                    _json.dumps(cfg, indent=2), encoding="utf-8"
                )
                (out_dir / "label_map.json").write_text(
                    _json.dumps(label_map, indent=2), encoding="utf-8"
                )
                (out_dir / "model.bin").write_text(
                    _json.dumps(
                        {
                            "note": "Remote wrapper; use evaluate.use_modal=true to run",
                            "remote_out_dir": rel_subdir,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                # Evaluate on Modal referencing the shared volume
                _trigger_evaluation(params, fam, run_name, use_modal=True)
                return
            else:
                if bool(spec.get("modal_pull_only", tcfg.get("modal_pull_only", False))):
                    _pull_modal_artifacts(
                        rel_subdir=rel_subdir,
                        out_dir_local=out_dir,
                        app_name=modal_app,
                        volume_name=modal_vol,
                    )
                    _trigger_evaluation(params, fam, run_name, use_modal=False)
                    return
                _train_via_modal(
                    fam=fam,
                    train_jsonl_path=train_jsonl,
                    val_jsonl_path=val_jsonl,
                    rel_subdir=rel_subdir,
                    out_dir_local=out_dir,
                    tcfg=spec or tcfg,
                    gpu_spec=modal_gpu,
                    app_name=modal_app,
                    volume_name=modal_vol,
                )
                # If we did not download artifacts, evaluate on Modal directly using shared volume
                if not bool(
                    spec.get(
                        "modal_download_artifacts",
                        tcfg.get("modal_download_artifacts", False),
                    )
                ):
                    _trigger_evaluation(params, fam, run_name, use_modal=True)
                else:
                    _trigger_evaluation(params, fam, run_name, use_modal=False)
                return

        # Local training per family
        if fam in {"fraud", "fraud_fuse"}:
            # Create a lightweight wrapper referencing a base model
            from ..models.fraud_fuse import train_fraud_fuse as _train_fuse

            base_rel = str(
                spec.get("base_model")
                or spec.get("base_model_rel")
                or tcfg.get("base_model")
                or tcfg.get("base_model_rel", "")
            )
            if not base_rel:
                raise ValueError(
                    "fraud_fuse requires base_model (relative to models_dir), e.g. 'encoder/enc-xlm'"
                )
            # Infer base family from prefix if not provided explicitly
            base_fam = str(
                spec.get("base_family")
                or tcfg.get("base_family")
                or (base_rel.split("/")[0] if "/" in base_rel else "encoder")
            )
            overrides = {}
            for k in [
                "emb_model",
                "sim_threshold",
                "min_samples",
                "topk",
                "dup_threshold",
                "w_text",
                "w_fraud",
            ]:
                if k in spec:
                    overrides[k] = spec[k]
                elif k in tcfg:
                    overrides[k] = tcfg[k]
            _train_fuse(str(out_dir), base_rel, base_fam, overrides)
            # Write a minimal label map if missing
            _ensure_label_map(out_dir)

        elif fam == "fraud_scan":
            # Run similarity scan over original reviews from normalized dataset
            from ..orchestrator.utils import data_dir as _data_dir, dataset_slug as _dataset_slug
            from ..models.fraud_similarity import train_fraud_similarity as _train_scan

            # Path to interim parquet built by normalize
            base_dir = Path(_data_dir(params))
            ds = _dataset_slug(params)
            data_parquet = base_dir / "interim" / ds / "reviews.parquet"
            overrides = {}
            for k in [
                "min_reviews",
                "topk",
                "dup_threshold",
                "sim_threshold",
                "high_sim_threshold",
                "min_cluster_size",
                "overall_threshold",
            ]:
                if k in spec:
                    overrides[k] = spec[k]
                elif k in tcfg:
                    overrides[k] = tcfg[k]
            _train_scan(str(out_dir), str(data_parquet), overrides)
            # Ensure a minimal label map
            _ensure_label_map(out_dir)

        elif fam == "baseline":
            from ..models.rnn_classifier import train_rnn_jsonl

            ckpt = out_dir / "model.pt"
            train_rnn_jsonl(
                train_jsonl,
                val_jsonl,
                str(ckpt),
                epochs=int(spec.get("epochs", tcfg.get("epochs", 5))),
                batch_size=int(spec.get("batch_size", tcfg.get("batch_size", 64))),
                lr=float(spec.get("lr", tcfg.get("lr", 1e-3))),
                early_stopping_patience=patience,
                early_stopping_min_delta=min_delta,
            )
            _copy_primary_weight(out_dir, out_dir / "model.bin")

        elif fam == "encoder":
            from ..models.bert_classifier import train_bert, BertConfig

            cfg = BertConfig(
                model_name=spec.get(
                    "model_name", tcfg.get("model_name", "distilbert-base-uncased")
                ),
                max_length=int(spec.get("max_length", tcfg.get("max_length", 256))),
                lr=float(spec.get("lr", tcfg.get("lr", 2e-5))),
                epochs=int(spec.get("epochs", tcfg.get("epochs", 3))),
                batch_size=int(spec.get("batch_size", tcfg.get("batch_size", 32))),
            )
            train_bert(
                train_jsonl,
                val_jsonl,
                str(out_dir),
                cfg,
                early_stopping_patience=patience,
                early_stopping_min_delta=min_delta,
            )
            _copy_primary_weight(out_dir, out_dir / "model.bin")

        elif fam == "sft":
            from ..models.sft_lora import train_sft, SFTCfg

            cfg = SFTCfg()
            if spec.get("model_name") or tcfg.get("model_name"):
                cfg.base_model = spec.get("model_name", tcfg.get("model_name"))
            if "epochs" in spec or "epochs" in tcfg:
                cfg.epochs = int(spec.get("epochs", tcfg.get("epochs")))
            if "lr" in spec or "lr" in tcfg:
                cfg.lr = float(spec.get("lr", tcfg.get("lr")))
            if "oversample" in spec or "oversample" in tcfg:
                cfg.oversample = bool(spec.get("oversample", tcfg.get("oversample")))
            train_sft(
                train_jsonl,
                val_jsonl,
                str(out_dir),
                cfg,
                early_stopping_patience=patience,
                early_stopping_min_delta=min_delta,
            )
            _copy_primary_weight(out_dir, out_dir / "model.bin")

        else:
            raise ValueError(f"Unknown train.family: {fam}")

        log.info("Training complete: %s", out_dir)
        # Evaluate locally post-training
        if fam == "fraud_scan":
            # No classifier to evaluate; artifacts are written during training.
            log.info("Skipping evaluation for fraud_scan: no classifier to evaluate (%s/%s)", fam, run_name)
        elif fam in {"fraud", "fraud_fuse"}:
            try:
                import json as _json
                from ..orchestrator.utils import models_dir as _models_dir

                cfg_path = out_dir / "fraud_fuse.json"
                if cfg_path.exists():
                    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                    base_rel = cfg.get("base_model_rel", "")
                    base_dir = Path(_models_dir({})) / str(base_rel)
                    if base_dir.exists():
                        _trigger_evaluation(params, fam, run_name, use_modal=False)
                    else:
                        log.info(
                            "Skipping immediate evaluation for %s/%s: base model not found at %s",
                            fam,
                            run_name,
                            str(base_dir),
                        )
                else:
                    log.info(
                        "Skipping evaluation: missing fraud_fuse.json for %s/%s", fam, run_name
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("Unable to evaluate fraud_fuse: %s", e)
        else:
            _trigger_evaluation(params, fam, run_name, use_modal=False)

    # Determine if multiple models are requested
    items = tcfg.get("models")
    if isinstance(items, list) and items:
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, spec in enumerate(items):
                futures.append(executor.submit(_train_one, spec or {}, i))

            results = []
            for f in as_completed(futures):
                results.append(f.result())  # will raise if _train_one failed

        return

    # Single-model fallback (compat)
    _train_one(tcfg, 0)
