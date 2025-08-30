from dataclasses import dataclass
from typing import Dict, Callable, List, Optional, Tuple
import json
import re
import math
from pathlib import Path
import contextlib

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
try:  # optional; fallback if bitsandbytes not present
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # noqa: BLE001
    BitsAndBytesConfig = None  # type: ignore
from transformers.generation import (
    LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList,
)
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, AutoPeftModelForCausalLM

import numpy as np

from . import LABELS


@dataclass
class SFTCfg:
    base_model: str = "mistral-7b-instruct"  # placeholder; replace with your 7B instruct
    max_length: int = 768
    lr: float = 1e-4
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 16
    # Expanded LoRA by default; rank bumped for better capacity
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    # Whether to oversample positives to roughly 1:1 in training
    oversample: bool = True


SYS_PROMPT = (
    "You are a policy checker for Google reviews. "
    "Given a review with fields, output JSON with boolean keys: "
    "irrelevant_content, advertisement, review_without_visit. Only output JSON."
)


def _has_url(text: str) -> bool:
    return ("http://" in text) or ("https://" in text)


def format_example(x: Dict, tokenizer: AutoTokenizer) -> str:
    """Training conversation using each model's native chat template.

    Includes a gold assistant turn containing the JSON answer.
    """
    user = (
        f"BUSINESS: {x.get('business_name','')}\n"
        f"AUTHOR: {x.get('author_name','')}\n"
        f"RATING: {x.get('rating','')}\n"
        f"TEXT: {_get_review_text(x)}\n"
        f"HAS_URL: {str(_has_url(x.get('text',''))).lower()}\n"
        f"HAS_PHOTO: {str(bool(x.get('has_photo', False))).lower()}"
    )
    target = {
        "irrelevant_content": bool(x.get("irrelevant_content", False)),
        "advertisement": bool(x.get("advertisement", False)),
        "review_without_visit": bool(x.get("review_without_visit", False)),
    }
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(target)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def format_prompt(x: Dict, tokenizer: AutoTokenizer) -> str:
    """Build only the prompt for inference (no target JSON), using chat template."""
    user = (
        f"BUSINESS: {x.get('business_name','')}\n"
        f"AUTHOR: {x.get('author_name','')}\n"
        f"RATING: {x.get('rating','')}\n"
        f"TEXT: {_get_review_text(x)}\n"
        f"HAS_URL: {str(_has_url(x.get('text',''))).lower()}\n"
        f"HAS_PHOTO: {str(bool(x.get('has_photo', False))).lower()}"
    )
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------- Constrained decoding helpers ----------

def _canonical_json_examples() -> List[str]:
    """Return canonical JSON strings to seed allowed-token sets."""
    obj_true = {k: True for k in LABELS}
    obj_false = {k: False for k in LABELS}
    # two spacing variants
    a = json.dumps(obj_true, separators=(", ", ": "))
    b = json.dumps(obj_true, separators=(",", ":"))
    c = json.dumps(obj_false, separators=(", ", ": "))
    d = json.dumps(obj_false, separators=(",", ":"))
    return [a, b, c, d]


def _json_allowed_token_ids(tokenizer: AutoTokenizer) -> Tuple[set, List[int]]:
    """Compute a permissive allowed token-id set to emit JSON with our keys.

    We collect all token ids that appear when tokenizing several canonical
    JSON realizations so the model can toggle booleans but stay in-format.
    Returns (allowed_ids, close_brace_ids_for_stop).
    """
    allowed: set = set()
    for s in _canonical_json_examples():
        ids = tokenizer.encode(s, add_special_tokens=False)
        allowed.update(ids)
    # Always allow eos and pad so generation can terminate cleanly
    if tokenizer.eos_token_id is not None:
        allowed.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        allowed.add(tokenizer.pad_token_id)
    close_ids = tokenizer.encode("}", add_special_tokens=False)
    return allowed, close_ids


class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: set):
        self.allowed = allowed_token_ids

    def __call__(self, input_ids, scores):
        # scores: [batch, vocab]
        # Operate in float32 for broad kernel support, then cast back
        # to original dtype to avoid Float vs BF16/Half mismatches on some backends.
        import torch

        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        orig_dtype = scores.dtype
        device = scores.device
        scores_f32 = scores.float()
        mask = torch.full_like(scores_f32, float("-inf"))
        allowed_idx = torch.tensor(list(self.allowed), device=device, dtype=torch.long)
        # Place original scores at allowed indices; others -inf
        mask.scatter_(1, allowed_idx.unsqueeze(0).expand(scores_f32.size(0), -1), 0.0)
        scores_out = scores_f32 + mask
        return scores_out.to(orig_dtype)


class StopOnSubsequence(StoppingCriteria):
    def __init__(self, subseq: List[int]):
        self.subseq = subseq or []

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.subseq:
            return False
        import torch

        seq = input_ids[0].tolist() if isinstance(input_ids, torch.Tensor) else input_ids[0]
        n = len(self.subseq)
        if len(seq) < n:
            return False
        return seq[-n:] == self.subseq


def train_sft(
    jsonl_train: str,
    jsonl_val: str,
    out_dir: str,
    cfg: SFTCfg = SFTCfg(),
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 0.0,
):
    # Load datasets
    ds_train = load_dataset("json", data_files=jsonl_train, split="train")
    ds_val = load_dataset("json", data_files=jsonl_val, split="train")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-rebalance: oversample rows with any positive label to reach ~1:1
    def _any_pos(x: Dict) -> bool:
        return any(bool(x.get(k, False)) for k in LABELS)

    if bool(getattr(cfg, "oversample", True)):
        try:
            ds_pos = ds_train.filter(_any_pos)
            ds_neg = ds_train.filter(lambda x: not _any_pos(x))
            n_pos = len(ds_pos)
            n_neg = len(ds_neg)
            if n_pos > 0 and n_pos < n_neg:
                reps = int(math.ceil(n_neg / float(n_pos)) - 1)
                if reps > 0:
                    parts = [ds_train] + [ds_pos] * reps
                    ds_train = concatenate_datasets(parts)
                    ds_train = ds_train.shuffle(seed=42)
        except Exception:
            # If filtering fails for any reason, proceed without rebalancing
            pass

    # Map to text using correct chat template; also preserve original 'text' as 'review_text'
    ds_train = ds_train.map(lambda x: {"text": format_example(x, tokenizer), "review_text": x.get("text", "")})
    ds_val = ds_val.map(lambda x: {"text": format_example(x, tokenizer), "review_text": x.get("text", "")})

    # Ensure consistent compute dtype for 4-bit to avoid bf16/float mismatches (seen on Mistral)
    import torch
    quantization_config = None
    # Choose a global compute dtype
    if torch.cuda.is_available():
        supports_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    else:
        compute_dtype = torch.float32
    if cfg.load_in_4bit and BitsAndBytesConfig is not None:
        # Prefer bf16 compute on GPUs that support it (e.g., L4/Ampere+), else fp16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        load_in_4bit=bool(cfg.load_in_4bit and BitsAndBytesConfig is None),
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    try:
        # Ensure lm_head matches compute dtype (prevents Float vs BF16 mismatch)
        model.lm_head.to(compute_dtype)
    except Exception:
        pass

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            # attention projections
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # MLP projections (SwiGLU variants will partially match)
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )
    sft_args = SFTConfig(
        output_dir=out_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        max_length=cfg.max_length,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config=lora,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        args=sft_args,
    )
    # After each eval (epoch), dump raw generations on validation set
    class _ValDumpCallback(TrainerCallback):
        def __init__(self, tok: AutoTokenizer, val_rows, batch_size: int, out_dir: str):
            self.tokenizer = tok
            self.rows = val_rows
            self.bs = max(1, int(batch_size))
            self.out_dir = Path(out_dir) / "predictions"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.allowed_ids, close_ids = _json_allowed_token_ids(self.tokenizer)
            self.lp = LogitsProcessorList([AllowedTokensLogitsProcessor(self.allowed_ids)])
            self.sc = StoppingCriteriaList([StopOnSubsequence(close_ids)])

        def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
            import torch
            from torch import autocast

            model = kwargs.get("model")
            if model is None:
                model = trainer.model
            model.eval()
            out_path = self.out_dir / f"epoch-{int(state.epoch or 0)}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                with torch.no_grad():
                    # Disable autocast here so compute dtype aligns with 4-bit linear compute
                    amp_ctx = autocast(device_type='cuda', dtype=kwargs.get('model', trainer.model).dtype) if torch.cuda.is_available() else contextlib.nullcontext()
                    with amp_ctx:
                        for i in range(0, len(self.rows), self.bs):
                            batch = self.rows[i : i + self.bs]
                            prompts = [format_prompt(r, self.tokenizer) for r in batch]
                            inputs = self.tokenizer(
                                prompts, return_tensors="pt", padding=True, truncation=True
                            ).to(model.device)
                            out = model.generate(
                                **inputs,
                                max_new_tokens=64,
                                do_sample=False,
                                pad_token_id=self.tokenizer.eos_token_id,
                                logits_processor=self.lp,
                                stopping_criteria=self.sc,
                            )
                            texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
                            for j, txt in enumerate(texts):
                                rec = {
                                    "index": i + j,
                                    "prompt": prompts[j],
                                    "output": txt,
                                }
                                f.write(json.dumps(rec) + "\n")

    # Prepare rows of val set as dicts for the callback
    val_rows: List[Dict] = [
        {k: v for k, v in r.items() if k != "text"} for r in ds_val
    ]
    trainer.add_callback(
        _ValDumpCallback(tokenizer, val_rows, cfg.batch_size, out_dir)
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
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Collect and write training history
    hist = {}
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
    out_list = [hist[k] for k in sorted(hist.keys(), key=lambda x: float(x))]
    import json as _json
    from pathlib import Path as _Path
    _Path(out_dir).mkdir(parents=True, exist_ok=True)
    (_Path(out_dir) / "training_history.json").write_text(_json.dumps(out_list, indent=2), encoding="utf-8")


def _extract_last_json(text: str) -> Dict:
    m = re.search(r"\{.*\}$", text.strip(), re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def _load_sft_model_and_tokenizer(model_dir: str):
    """Load a PEFT SFT model checkpoint robustly for inference.

    Prefers AutoPeftModelForCausalLM (adapter-aware). Falls back to
    AutoModelForCausalLM if PEFT auto loader is unavailable. Uses 4-bit loading
    on GPU to avoid OOM and disables a late model.to() that can conflict with
    accelerate offloading.
    """
    import torch

    # Load tokenizer first and ensure a pad token exists
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Match compute dtype across modules
    if torch.cuda.is_available():
        supports_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    else:
        compute_dtype = torch.float32

    load_in_4bit = bool(torch.cuda.is_available())  # 4-bit only when GPU present

    quantization_config = None
    if load_in_4bit and BitsAndBytesConfig is not None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Try adapter-aware loader first
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=compute_dtype,
            load_in_4bit=bool(load_in_4bit and BitsAndBytesConfig is None),
            quantization_config=quantization_config,
        )
    except Exception:
        # Fallback: standard auto model loader
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=compute_dtype,
            load_in_4bit=bool(load_in_4bit and BitsAndBytesConfig is None),
            quantization_config=quantization_config,
        )
    try:
        model.lm_head.to(compute_dtype)
    except Exception:
        pass

    return model, tokenizer


def infer_sft(model_dir: str, example: Dict, max_new_tokens=64) -> Dict[str, bool]:
    # Load per-call for API compatibility; heavy, but kept for backward use.
    model, tokenizer = _load_sft_model_and_tokenizer(model_dir)
    # Only provide prompt; do not include a pre-filled JSON answer.
    prompt = format_prompt(example, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Constrained generation
    allowed_ids, close_ids = _json_allowed_token_ids(tokenizer)
    lp = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_ids)])
    sc = StoppingCriteriaList([StopOnSubsequence(close_ids)])
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=lp,
        stopping_criteria=sc,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    payload = _extract_last_json(text)
    return {k: bool(payload.get(k, False)) for k in LABELS}


def predict_probs_sft(
    model_dir: str,
    jsonl_path: str,
    batch_size: int = 8,
    max_new_tokens: int = 64,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    raw_out_path: Optional[str] = None,
) -> np.ndarray:
    """Greedy decode and parse JSON to boolean, return float probs 0/1.
    Loads the model once and reuses it for all rows. Performs batched generation
    to improve throughput.

    Parameters
    - model_dir: path to SFT model directory
    - jsonl_path: dataset file with one JSON sample per line
    - batch_size: number of prompts to generate per batch
    - max_new_tokens: generation length for the assistant reply
    - progress_callback: optional callable(completed:int, total:int) for progress
    """
    import torch

    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)
    if progress_callback:
        progress_callback(0, total)

    model, tokenizer = _load_sft_model_and_tokenizer(model_dir)
    allowed_ids, close_ids = _json_allowed_token_ids(tokenizer)
    lp = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_ids)])
    sc = StoppingCriteriaList([StopOnSubsequence(close_ids)])
    preds: List[List[float]] = []
    writer = None
    if raw_out_path:
        Path(raw_out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = open(raw_out_path, "w", encoding="utf-8")
    # Use a conservative context limit for safety
    try:
        max_ctx = int(getattr(tokenizer, "model_max_length", 2048))
        if max_ctx <= 0 or max_ctx > 4096:
            max_ctx = 2048
    except Exception:
        max_ctx = 2048

    # Ensure eval mode and no grad
    model.eval()
    completed = 0
    idx = 0
    with torch.no_grad():
        for i in range(0, total, max(1, int(batch_size))):
            batch_rows = rows[i : i + batch_size]
            prompts = [format_prompt(r, tokenizer) for r in batch_rows]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_ctx,
            )
            # Move inputs to the same device shards as the model
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=lp,
                stopping_criteria=sc,
            )
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            for t in texts:
                payload = _extract_last_json(t)
                parsed = {k: bool(payload.get(k, False)) for k in LABELS}
                preds.append([float(parsed[k]) for k in LABELS])
                if writer is not None:
                    rec = {"i": idx, "output": t, "parsed": parsed}
                    writer.write(json.dumps(rec) + "\n")
                idx += 1
            completed = min(total, i + len(batch_rows))
            if progress_callback:
                progress_callback(completed, total)

    # Free GPU memory if applicable
    try:
        if torch.cuda.is_available():
            del model
            torch.cuda.empty_cache()
    except Exception:
        pass
    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass

    return np.array(preds, dtype=float)
def _get_review_text(x: Dict) -> str:
    """Return the original review text, preferring 'review_text' if present.

    During training we overwrite the 'text' field with chat-formatted content,
    so we preserve the original input in 'review_text'. For raw rows (eval), the
    original content is under 'text'.
    """
    return str(x.get("review_text", x.get("text", "")))
