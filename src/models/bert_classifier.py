from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from . import LABELS


@dataclass
class BertConfig:
    model_name: str = "distilbert-base-uncased"  # or bert-base-uncased
    max_length: int = 256
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 32


class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_text(example: Dict) -> str:
    parts = [
        f"BUSINESS: {example.get('business_name','')}\n",
        f"AUTHOR: {example.get('author_name','')}\n",
        f"RATING: {example.get('rating', '')}\n",
        f"TEXT: {example.get('text','')}",
    ]
    if example.get("has_url"):
        parts.append("\nHAS_URL: true")
    if example.get("has_photo"):
        parts.append("\nHAS_PHOTO: true")
    return "".join(parts)


def make_dataset(jsonl_path: str) -> Dataset:
    # For a single JSONL file, the HF "json" builder exposes only a "train" split.
    # Always read the provided file with split="train" regardless of role (train/val/test).
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    ds = ds.map(
        lambda x: {
            "inputs": build_text(x),
            "labels": [float(bool(x.get(name, False))) for name in LABELS],
        }
    )
    return ds.remove_columns([c for c in ds.column_names if c not in ["inputs", "labels"]])


def train_bert(
    jsonl_train: str,
    jsonl_val: str,
    out_dir: str,
    cfg: BertConfig = BertConfig(),
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 0.0,
):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = make_dataset(jsonl_train)
    val_ds = make_dataset(jsonl_val)

    def tok_fn(batch):
        return tokenizer(batch["inputs"], truncation=True, max_length=cfg.max_length)

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["inputs"]).with_format("torch")
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["inputs"]).with_format("torch")

    def to_float_tensor(batch):
        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float32)
        return batch

    train_ds = train_ds.map(to_float_tensor)
    val_ds = val_ds.map(to_float_tensor)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
    )
    # Ensure multi-label classification setting is applied even on older versions
    try:
        model.config.problem_type = "multi_label_classification"
    except Exception:
        pass

    # Build TrainingArguments with compatibility across transformers versions
    from inspect import signature

    sig = signature(TrainingArguments.__init__)
    def supports(name: str) -> bool:
        return name in sig.parameters

    args_kwargs = {
        "output_dir": out_dir,
        "learning_rate": cfg.lr,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "num_train_epochs": cfg.epochs,
    }
    set_eval_save = False
    if supports("eval_strategy"):
        args_kwargs["eval_strategy"] = "epoch"
        set_eval_save = True
    elif supports("evaluate_during_training"):
        args_kwargs["evaluate_during_training"] = True
    if supports("save_strategy") and set_eval_save:
        args_kwargs["save_strategy"] = "epoch"
    elif supports("save_steps") and not set_eval_save:
        args_kwargs["save_steps"] = 500
    if set_eval_save and supports("load_best_model_at_end"):
        args_kwargs["load_best_model_at_end"] = True
        # Use validation loss for early stopping/best model selection
        if supports("metric_for_best_model"):
            args_kwargs["metric_for_best_model"] = "eval_loss"
        if supports("greater_is_better"):
            args_kwargs["greater_is_better"] = False
    if supports("report_to"):
        args_kwargs["report_to"] = ["none"]
    if supports("logging_strategy"):
        args_kwargs["logging_strategy"] = "epoch"

    args = TrainingArguments(**args_kwargs)
    # Backward-compatibility: some transformers versions may not expose these
    # as constructor args. Ensure attributes exist for EarlyStoppingCallback.
    if not getattr(args, "load_best_model_at_end", False):
        setattr(args, "load_best_model_at_end", True)
    if getattr(args, "metric_for_best_model", None) in (None, ""):
        setattr(args, "metric_for_best_model", "eval_loss")
    if getattr(args, "greater_is_better", None) is None:
        setattr(args, "greater_is_better", False)

    from sklearn.metrics import precision_recall_fscore_support

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"precision_macro": p, "recall_macro": r, "f1_macro": f1}

    trainer = MultiLabelTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Early stopping if available
    try:
        from transformers.trainer_callback import EarlyStoppingCallback  # type: ignore
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_min_delta,
            )
        )
    except Exception:
        pass

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Collect per-epoch training/eval losses (and metrics if present)
    hist: Dict[str, Dict] = {}
    for rec in trainer.state.log_history:
        ep = rec.get("epoch")
        if ep is None:
            continue
        key = f"{float(ep):.0f}"
        entry = hist.setdefault(key, {"epoch": float(ep)})
        if "loss" in rec:
            entry["train_loss"] = float(rec["loss"])
        if "eval_loss" in rec:
            entry["val_loss"] = float(rec["eval_loss"])
        for k in ("f1_macro", "precision_macro", "recall_macro"):
            if k in rec:
                entry[k] = float(rec[k])
    # Order by epoch
    out_list: List[Dict] = [hist[k] for k in sorted(hist.keys(), key=lambda x: float(x))]
    import json as _json
    from pathlib import Path as _Path
    _Path(out_dir).mkdir(parents=True, exist_ok=True)
    (_Path(out_dir) / "training_history.json").write_text(_json.dumps(out_list, indent=2), encoding="utf-8")


@torch.no_grad()
def infer_bert(model_dir: str, example: Dict, threshold: float = 0.5) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    text = build_text(example)
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    logits = model(**batch).logits.squeeze(0).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    return {label: bool(preds[i]) for i, label in enumerate(LABELS)}


@torch.no_grad()
def predict_probs_bert(
    model_dir: str,
    jsonl_path: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Return probabilities array shape (N, L) for dataset from jsonl_path."""
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    ds = make_dataset(jsonl_path)
    # Tokenize; remove raw columns so the collator only sees token fields
    ds = ds.map(
        lambda x: tokenizer(x["inputs"], truncation=True, max_length=256),
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("input_ids", "attention_mask")],
    )
    # Use dynamic padding like in training
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=max(1, int(batch_size)), collate_fn=collator)
    probs_all = []
    for batch in loader:
        inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        logits = model(**inputs).logits.cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        probs_all.append(probs)
    return np.vstack(probs_all)
