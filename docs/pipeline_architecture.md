# Review Quality & Relevancy Pipeline Architecture

This document proposes an implementation-ready, modular pipeline to assess review relevancy and enforce policy violations for location reviews (Google, Yelp, TripAdvisor, etc.). It is designed for a 5-person team on a short timeline: easy to split, easy to run end-to-end or by component, and reproducible.

The architecture prioritizes:
- Modularity: clear, typed interfaces between steps.
- Orchestration: simple to run whole flows or individual tasks.
- Reproducibility: versioned data and configs, seeded randomness, logged runs.
- Practicality: minimal complexity for hackathon velocity, scalable later.

The document includes viable variations for key choices with pros/cons so you can tailor based on time, infra, and skillset.

---

## Goals & Scope

- Assess review relevancy to a venue.
- Flag policy violations (e.g., advertisements, rants-without-visit).
- Support multilingual content (English + Chinese at minimum).
- Produce explainable metrics and an interactive dashboard.
- Allow selective re-runs (e.g., re-annotate only, retrain only on a new sample).

Non-goals for hackathon phase (optional later): real-time streaming ingestion; complex MLOps with full CI/CD.

---

## High-Level Architecture

Flow (batch-oriented, orchestrated):

1) Ingest → 2) Normalize/Clean → 3) EDA → 4) Augment → 5) Annotate → 6) Feature/Vectorize → 7) Train → 8) Evaluate → 9) Dashboard

Artifacts produced at each step are versioned and reusable. Each component reads typed inputs and writes typed outputs described below.

---

## Orchestration (Lightweight, In-Repo)

We will build a minimal orchestrator in-repo for clarity and speed. It provides:
- Task abstraction with declared inputs/outputs (static lists or callables of config).
- Simple DAG scheduling with topological sort and cycle detection.
- Content-addressed caching (skip if inputs+config unchanged).
- Run records, logging, and resumable execution.
- Typer-based CLI to invoke pipelines or individual tasks.

Why this choice
- Pros: zero heavy dependencies, fully transparent, hackathon-friendly, easy to extend.
- Cons: fewer built-in bells/whistles than Prefect/Dagster (no UI, limited retries/ops).

Package layout (proposed):
```
orchestrator/
  __init__.py
  core.py          # Task, Pipeline, DAG, hashing, cycle detection
  cache.py         # artifact hashing, cache read/write, file locks
  cli.py           # Typer commands for steps and pipelines
  logging.py       # standard logging setup
tasks/
  ingest.py        # defines @task functions/classes
  normalize.py
  eda.py
  augment.py
  annotate.py
  featurize.py
  train.py
  evaluate.py
``` 

Core concepts
- Task: declares `name`, `inputs`/`outputs` as either static path lists or callables that take `params` and return lists; `run(params)`.
- Pipeline: list of tasks + dependencies. Executes via topological order, with caching.
- Artifact cache: compute a stable hash from input file content (and their metadata), task code version (git commit + file hash), and task-relevant config; store under `runs/<pipeline>/<run_id>/state.json` plus per-artifact `.hash` files next to outputs.

Minimal interfaces (Python sketch):
```
# orchestrator/core.py
@dataclass
class TaskSpec:
    name: str
    inputs: list[str] | Callable[[dict], list[str]]
    outputs: list[str] | Callable[[dict], list[str]]
    fn: Callable[..., None]

class Pipeline:
    def __init__(self, tasks: dict[str, TaskSpec], edges: list[tuple[str, str]]):
        self.tasks = tasks
        self.edges = edges
        self.order = topo_sort(tasks.keys(), edges)

    def run(self, params: dict, force: set[str] = set(), from_step: str | None = None, only_step: str | None = None):
        # resolve subset, check caches, execute, record state
        ...
```

Caching rules
- For each task, compute `task_hash = hash(inputs bytes + resolved paths + config subset + code hash)`. If unchanged and all outputs exist, skip.
- Store per-output `.hash` next to files and record in `runs/<pipeline>/<run_id>/state.json`.
- `--force <task>` bypasses cache for one or more tasks.

Resumability & partial runs
- CLI supports `--from-step <name>`, `--only-step <name>`, and `--until-step <name>`.
- Tasks can be run directly (e.g., re-run annotate with a different provider) while reusing prior artifacts by path.

Concurrency
- Optional `--workers N` to parallelize independent tasks; default sequential to keep the implementation simple and predictable.

Failures & retries
- Default: fail-fast; optionally `--retries K` per task with exponential backoff.

Logging
- Python `logging` with rotating file handlers per run under `runs/<pipeline>/<run_id>/*.log`.

Security & reproducibility
- No network calls unless inside tasks. All secrets via `.env`. Log redaction for secrets.

---

## Configuration Management

Recommended: YAML configs with Pydantic models for validation (+ environment overrides)
- Pros: explicit schemas, early validation, simple to extend; familiar YAML.
- Cons: not as powerful for config composition as Hydra.

Alternative: Hydra
- Pros: powerful config composition and overrides.
- Cons: slightly heavier mental model; can slow down a brand-new team.

Decision: Pydantic + YAML now; optionally migrate to Hydra if complexity grows.

Config resolution order:
1) Base YAML per component (e.g., `configs/train.yaml`).
2) Environment overrides via `.env` and/or CLI params.
3) Runtime defaults embedded in Pydantic models (validated at startup).

---

## Config‑Driven Artifact Paths (Plug‑and‑Play)

To support parallel experiments without collisions, artifact paths derive from config at runtime. Key knobs:

- `dataset.slug`: logical dataset id used under `data/interim/`, `data/augmented/`, `data/annotated/`.
- `augment.slug`: augmentation scheme id under `data/augmented/` and `data/annotated/`.
- `featurize.family`: `encoder` (HF tokenization) or `tfidf` (classical features); determines `features/{featurizer}/...`.
- `featurize.seed`: split seed; determines `data/splits/{seed}/...`.
- `train.run_name`: human-friendly run id for model/eval artifacts; if omitted, the pipeline timestamp `run_id` is used.

Conventions:
- Normalized data: `data/interim/{dataset.slug}/reviews.parquet`.
- Augmented data: `data/augmented/{dataset.slug}/{augment.slug}/reviews.parquet`.
- Annotations: `data/annotated/{dataset.slug}/{augment.slug}/annotations.parquet`.
- Features: `data/features/{featurizer}/tokenized.arrow` where `{featurizer}` is `hf-{train.model_name}` (slashes replaced) for encoders or a custom name for TF‑IDF.
- Splits: `data/splits/{seed}/(train|val|test).txt`.
- Models: `models/{train.family}/{train.run_name or run_id}/...`.
- Metrics: `reports/metrics/{train.run_name or run_id}/metrics.json`.

Result: You can swap models, datasets, augmentation schemes, or seeds by editing config alone. The orchestrator’s caching includes resolved paths + config, enabling multiple variants to coexist.

Implementation detail: task specs accept callables of `params` to compute inputs/outputs at runtime.

---

## Data & Artifacts Layout

```
project/
├─ data/
│  ├─ raw/                 # Original dumps, read-only
│  ├─ interim/             # Cleaned/normalized outputs (scoped by dataset slug)
│  ├─ augmented/           # Synthetic/augmented sets (scoped by dataset + augment slug)
│  ├─ annotated/           # Labeled datasets (scoped by dataset + augment slug)
│  ├─ features/            # Tokenized datasets or TF‑IDF (scoped by featurizer slug)
│  └─ splits/              # Train/val/test indices (scoped by split seed)
├─ models/
│  ├─ baseline/            # TF-IDF + LR, etc.
│  ├─ encoder/             # Fine-tuned (DistilBERT/XLM-R)
│  └─ sft/                 # LoRA adapters for small LLMs
├─ runs/                   # Per-run logs, configs snapshot, metrics
└─ reports/
   ├─ eda/                 # EDA notebooks/exports
   └─ metrics/             # Metrics per run_name
```

Variant: add DVC to version `data/` and `models/` (recommended if collaborating heavily). Pros: reproducible data lineage; Cons: adds setup time.

---

## Core Schemas

Review record (JSON/Parquet schema):
```
{
  "review_id": str,
  "place_id": str,
  "user_id": str | null,
  "text": str,
  "language": str,                 # e.g., "en", "zh"
  "rating": int | null,            # 1..5 if available
  "created_at": str | null,        # ISO 8601
  "metadata": {                    # optional, source-specific
    "source": "google|yelp|tripadvisor",
    "url": str | null,
    "gps_proximity": float | null,
    "user_review_count": int | null
  }
}
```

Annotation schema (multi-label):
```
{
  "review_id": str,
  "relevant": bool | float,        # binary or score [0,1]
  "policies": {
    "advertisement": 0|1,
    "rant_without_visit": 0|1,
    # extendable list
  },
  "annotator": "llm|rule|human",
  "explanations": [str]            # optional rationale strings
}
```

Dataset contract:
- All inter-component data are stored as Parquet (preferred) or JSONL for streaming friendliness.
- UTF-8 encoding, normalized newlines.
- Language field normalized to ISO-639-1.
- Timestamps in ISO-8601, timezone-aware if available.

---

## Components

### 1) Ingestion

Inputs:
- Public datasets (Kaggle/GoogleLocal/Yelp) and optional scraped reviews.

Outputs:
- `data/raw/{source}/reviews.parquet`

Steps:
- Source readers: adapters per provider (GoogleLocal parquet/JSON, Yelp CSV/JSON, etc.).
- De-duplication by `(place_id, user_id, text_hash)`.
- Light PII scrubbing (remove emails/URLs if required at this stage or leave for policy detection).

Variants:
- Minimal: single-source loader (fastest start).
- Full: multi-source merge with source provenance retained.

### 2) Normalize & Clean

Inputs:
- `data/raw/*/*.parquet`

Outputs:
- `data/interim/{dataset.slug}/reviews.parquet` (unified schema; `dataset.slug` identifies the dataset variant).

Steps:
- Normalize fields to Core Schemas; set `language` using fast langid (e.g., fasttext/langdetect) when missing.
- Text cleaning: control chars, excessive whitespace, optional lowercasing; keep emojis.
- Filter or mark extremely short texts (e.g., < 5 tokens).

Variants:
- Aggressive cleaning vs minimal cleaning. Pros of aggressive: fewer spurious tokens; Cons: risk of losing signals.

### 3) EDA

Inputs:
- `data/interim/{dataset.slug}/reviews.parquet`

Outputs:
- `reports/eda/*.html|png`, optional notebook artifacts under `reports/eda/`.

Scope:
- Language mix, length distribution, rating distributions, time trends.
- Keyword clouds by language; n-gram frequencies.
- Preliminary rule signals (presence of URLs, phone numbers) as hints for ads.

Tooling:
- Jupyter/Polars/Pandas + Plotly/Altair; export static assets for sharing.

### 4) Augmentation (Optional but recommended for underrepresented classes/languages)

Inputs:
- `data/interim/{dataset.slug}/reviews.parquet`

Outputs:
- `data/augmented/{dataset.slug}/{augment.slug}/reviews.parquet`

Strategies:
- Back-translation (en ↔ zh) for relevancy-preserving augmentation.
- Template-based generation for specific policies (ads, no-visit rants).
- LLM-guided augmentation with constraints (length, tone, keywords).

Variants:
- Offline with open models (e.g., Qwen2-7B, Llama3-8B via vLLM/Ollama/Modal) vs API (OpenAI/Anthropic). Pros of offline: cost control; Cons: setup time, GPU required.

### 5) Annotation (LLM + Rules + Human)

Inputs:
- `data/interim/{dataset.slug}/reviews.parquet` (+ `data/augmented/{dataset.slug}/{augment.slug}/reviews.parquet` if present)

Outputs:
- `data/annotated/{dataset.slug}/{augment.slug}/annotations.parquet` (aligned by `review_id`)

Approach:
- Rule pre-labels: simple regex/features for ads (URLs, phone numbers, phrases like "promo", "discount"), and no-visit rants (patterns like "never been", language-specific variants).
- LLM labeling: prompt to classify relevancy and policy flags with short rationale. Use temperature=0, system prompts defining policies.
- Human spot-check: sample stratum per class/language to estimate label quality and correct a validation set.

Quality control:
- Triangulation: If rules + LLM disagree with low confidence, send to manual queue (CSV or simple Streamlit labeling page).
- Record annotator provenance and confidence (LLM logprob if available).

Variants:
- API LLM (fastest) vs local small LLM on Modal with LoRA adapters for domain style.
- Single-pass labeling vs two-pass (coarse then refine disagreements).

### 6) Feature Engineering / Tokenization

Inputs:
- `data/interim/{dataset.slug}/reviews.parquet`, `data/annotated/{dataset.slug}/{augment.slug}/annotations.parquet`

Outputs:
- `data/features/{featurizer}/tokenized.arrow` (HF datasets; `{featurizer}` is `hf-{train.model_name}` with slashes replaced, or a custom TF‑IDF name)
- `data/splits/{seed}/(train|val|test).txt` (review_id lists) or parquet partitions

Options:
- Baseline: TF-IDF + n-grams; concatenate simple metadata (length, rating, url_presence) into features.
- Encoders: HuggingFace tokenization (DistilBERT, XLM-R for multilingual).

### 7) Model Training

Inputs:
- Tokenized datasets or features; `data/splits/*`.

Outputs:
- `models/{family}/{run_name}/...` (model weights, label map, config snapshot). `run_name` is `train.run_name` if provided, else the pipeline `run_id` (timestamp).

Model families:
- Baseline: Logistic Regression / Linear SVM for relevancy + policy flags (one-vs-rest). Pros: very fast baseline; Cons: ceiling performance.
- Encoder fine-tuning: DistilBERT (English) + XLM-R (multilingual) multi-label heads. Pros: strong accuracy; Cons: needs GPU for speed.
- Small LLM SFT (4–7B) with LoRA for instruction-style classification. Pros: unified text understanding and rationale; Cons: heavier infra, longer runs.

Training modes:
- Single-task heads (relevancy, advertisement, rant_without_visit) vs multi-task with shared encoder. Multi-task Pros: shared signal; Cons: tune task balance.

Infra:
- Local GPU if available; else run training on Modal (recommended for SFT and encoder FT). Start with small batch sizes and gradient accumulation.

### 8) Evaluation & Reporting

Inputs:
- `models/{family}/{run_name}`, `data/splits/{seed}`, holdout annotations.

Outputs:
- `reports/metrics/{run_name}/metrics.json` (precision/recall/F1 per class), confusion matrices, PR curves.

Scope:
- Per-class, macro/micro F1, AUROC for binary tasks, calibration plots.
- Language-sliced metrics (en vs zh) and by review length.
- Error analysis export with example false positives/negatives and rationales.

### 9) Dashboard

Goal:
- Explore dataset, model metrics, and allow interactive classification of sample reviews.

Recommended: Streamlit app
- Reads latest artifacts from `reports/metrics/` and `models/*/latest`.
- Simple interface to paste a review, run inference, show policy flags + rationale (if model provides), and language detection.

Alternative: Gradio (faster to scaffold a single inference widget) or a lightweight FastAPI + React dashboard (more work).

---

## Pipelines and Tasks (Lightweight Orchestrator)

Pipelines are DAGs composed of the following tasks. Each task declares inputs/outputs and is cacheable.

- `task: ingest_all(sources: list[str]) -> data/raw/*`
  - steps: `read_source`, `dedupe`, `write_raw`

- `task: normalize() -> data/interim/reviews.parquet`
 - `task: normalize() -> data/interim/{dataset.slug}/reviews.parquet`
  - steps: `load_raw`, `normalize_schema`, `clean_text`, `detect_language`

- `task: eda() -> reports/eda/*`
  - steps: `compute_stats`, `plot_*`, `write_reports`

 - `task: augment(config) -> data/augmented/{dataset.slug}/{augment.slug}/reviews.parquet`
  - steps: `back_translate`, `template_generate`, `llm_augment`, `merge_augmented`

 - `task: annotate(config) -> data/annotated/{dataset.slug}/{augment.slug}/annotations.parquet`
  - steps: `rule_label`, `llm_label_batch`, `resolve_disagreements`, `write_annotations`

- `task: featurize(model_family) -> data/features/*, data/splits/*`
  - steps: `build_splits`, `tokenize_or_vectorize`, `persist_features`

 - `task: train(model_family, hyperparams) -> models/{family}/{run_name}`
  - steps: `load_features`, `train_model`, `save_model`, `log_metrics`

 - `task: evaluate(model_path) -> reports/metrics/{run_name}`
  - steps: `inference_on_holdout`, `compute_metrics`, `write_reports`

- `pipeline: full_run(params) -> Artifacts`
  - order: `ingest_all` → `normalize` → (`eda`) → `augment` → `annotate` → `featurize` → `train` → `evaluate`

Execution patterns:
- Run end-to-end with a single command, or run individual tasks with cached inputs.
- Task parameters accept explicit artifact paths to reuse previous outputs (e.g., retrain with a different split without re-annotating).

Example task and pipeline assembly
```
# orchestrator/core.py (continued)
def task(name: str, inputs, outputs):  # inputs/outputs can be callables of params
    def deco(fn):
        return TaskSpec(name=name, inputs=inputs, outputs=outputs, fn=fn)
    return deco

# tasks/ingest.py
@task(
    name="ingest_all",
    inputs=["configs/base.yaml", "data/raw/*"],
    outputs=["data/raw/googlelocal/reviews.parquet", "data/raw/yelp/reviews.parquet"],
)
def ingest_all(params: dict):
    ...  # read sources, dedupe, write parquet

# orchestrator/cli.py
app = Typer()

@app.command()
def full_run(config: str = "configs/base.yaml", force: list[str] = Option([], "--force")):
    pipe = Pipeline(tasks={
        "ingest_all": tasks.ingest.ingest_all,
        "normalize": tasks.normalize.normalize,
        "eda": tasks.eda.eda,
        "augment": tasks.augment.augment,
        "annotate": tasks.annotate.annotate,
        "featurize": tasks.featurize.featurize,
        "train": tasks.train.train,
        "evaluate": tasks.evaluate.evaluate,
    }, edges=[
        ("ingest_all", "normalize"),
        ("normalize", "eda"),
        ("normalize", "augment"),
        ("normalize", "annotate"),
        ("annotate", "featurize"),
        ("featurize", "train"),
        ("train", "evaluate"),
    ])
    params = load_config(config)
    pipe.run(params=params, force=set(force))
```

---

## Config Structure (YAML + Pydantic)

Example `configs/base.yaml` (selected fields):
```
project:
  seed: 42
  device: "auto"               # "cpu" | "cuda:0" | "mps"
  data_dir: "data"
  models_dir: "models"
  runs_dir: "runs"

dataset:
  slug: base                   # logical dataset id

augment:
  slug: none                   # augmentation scheme id

featurize:
  family: encoder              # encoder | tfidf (defaults to train.family)
  seed: 42                     # overrides project.seed for splits

ingest:
  sources:
    - name: "googlelocal"
      path: "data/raw/googlelocal/*.parquet"
    - name: "yelp"
      path: "data/raw/yelp/*.jsonl"

annotate:
  llm:
    provider: "openai"         # or "local"
    model: "gpt-4o-mini"       # or local HF model name
    max_concurrency: 4
    temperature: 0.0
  rules:
    url_regex: true
    phone_regex: true
    never_been_patterns:
      en: ["never been", "haven't visited"]
      zh: ["没去过", "从来没去"]

train:
  family: "encoder"            # "baseline" | "encoder" | "sft"
  model_name: "xlm-roberta-base"
  batch_size: 16
  lr: 3e-5
  epochs: 3
  multilabel: true
  # run_name: xlmr_a           # optional; defaults to runtime run_id if omitted

evaluate:
  threshold: 0.5
```

Pydantic example (pseudo-type):
```
class TrainConfig(BaseModel):
    family: Literal["baseline", "encoder", "sft"]
    model_name: str
    batch_size: int = 16
    lr: float = 3e-5
    epochs: int = 3
    multilabel: bool = True
    run_name: str | None = None
```

---

## Inference Contract

Standard predictor interface:
```
def predict(batch: list[Review]) -> list[{
  "review_id": str,
  "relevant": float,                   # probability
  "policies": {label: float},          # probabilities per policy
  "explanations": list[str] | None
}]
```

This contract is used by evaluation and the dashboard. For rule-augmented models, explanations may include matched rule names.

---

## Modal (GPU) Integration Options

Use Modal for:
- Encoder fine-tuning (GPU) and SFT (LoRA) jobs.
- Batched LLM inference for annotation/augmentation.

Approach:
- Build a Modal Image with Python deps + model weights.
- Mount a persistent volume or sync `data/` via S3/GCS.
- Expose a function per job (e.g., `train_encoder`, `llm_label_batch`).

Pros: elastic GPUs, reproducible builds. Cons: requires setup and secrets management.

Fallback: local training on CPU/MPS with smaller batches; or colab.

---

## Tracking, Logging, and Reproducibility

- Runs: each task/pipeline writes under `runs/{task_or_pipeline}/{run_id}` a snapshot of config, code ref (git commit + file hash), and logs.
- Metrics: JSON + CSV under `reports/metrics/{run_name}` (where `run_name` is `train.run_name` or the pipeline `run_id`); optionally use MLflow or Weights & Biases.
- Seeds: global seed in config; set in NumPy, PyTorch, Python `random`.
- Environments: `.venv` with pinned `requirements.txt` or `pyproject.toml`. Optional Dockerfile for Modal/local parity.

Variant: MLflow tracking server. Pros: strong experiment management; Cons: setup time.

---

## Team Split (5 people)

- Data & Ingestion: build loaders, normalization, language detection, dedupe.
- Annotation & Rules: prompts, rule patterns, disagreement resolver, quality checks.
- Modeling: baselines + encoder fine-tunes (and optional SFT on Modal).
- Evaluation & Dashboard: metrics pipeline, error analysis, Streamlit app.
- Orchestration & Infra: lightweight orchestrator (core/cache/CLI), configs, run scripts, data layout, optional DVC/MLflow.

---

## Minimal Viable Path (2–3 days)

Day 1:
- Ingest 1–2 sources; normalize; quick EDA; draft rule patterns.
- Baseline TF-IDF + LR; initial thresholds; Streamlit skeleton.

Day 2:
- LLM annotation pass on 20–50k reviews; refine rules; build validation set.
- Train DistilBERT (en) and/or XLM-R (multilingual). Evaluate and compare.

Day 3:
- Error analysis; refine prompts/rules; threshold tuning.
- Dashboard polish (metrics, per-language slices); package end-to-end flow.

Stretch:
- Augment underrepresented classes (Chinese ads, no-visit rants); try SFT LoRA.

---

## Pros/Cons Summary of Key Variations

- Orchestrator: In-repo lightweight (transparent, fastest) vs Prefect (balanced, extra dep) vs Make/CLI-only (simplest, least features).
- Config: Pydantic+YAML (simple, validated) vs Hydra (composable, heavier).
- Data versioning: Plain folders (fastest) vs DVC (traceability, overhead).
- Models: Baseline (fast, lower ceiling) vs Encoder FT (best balance) vs SFT (flexible, heavier).
- Annotation: API LLM (fast) vs Local LLM on Modal (control, setup) vs Human (gold, slow/limited).
- Dashboard: Streamlit (fast) vs Gradio (simplest UI) vs Web app (custom, heavy).

---

## Implementation Checklist

- [ ] Create folders: `data/`, `models/`, `runs/`, `reports/`, `configs/`.
- [ ] Write Pydantic models for configs; load `configs/base.yaml` with `.env` overrides.
- [ ] Scaffold lightweight orchestrator (`orchestrator/core.py`, `cache.py`, `cli.py`) and task modules under `tasks/`.
- [ ] Implement loaders for chosen sources; normalize and write Parquet.
- [ ] Implement rules and LLM annotator with batched inference.
- [ ] Build baseline + encoder trainers (HuggingFace `Trainer` or Lightning).
- [ ] Write evaluation script to produce metrics + plots.
- [ ] Build Streamlit dashboard reading latest artifacts.
- [ ] Add CLI commands to run pipelines end-to-end and by stage.
- [ ] Optional: DVC/MLflow wiring; Modal GPU functions for training/LLM.

---

## Example CLIs (Typer) for Local Development

```
# End-to-end
python -m cli full-run --config configs/base.yaml

# Run individual steps
python -m cli ingest --sources googlelocal yelp
python -m cli normalize
python -m cli annotate --provider openai --model gpt-4o-mini
python -m cli featurize --family encoder --model xlm-roberta-base
python -m cli train --family encoder --epochs 3
python -m cli evaluate --run-id <id>
python -m cli dashboard  # launches Streamlit
```

These commands are thin wrappers that call the lightweight orchestrator pipelines or the underlying functions directly for quick iteration.

---

## Notes on Multilingual Support (EN + ZH)

- Prefer XLM-R or mBERT for multilingual classification.
- Maintain language-specific rule phrases; expand with EDA findings.
- Use back-translation augmentation to balance classes.
- Evaluate per-language and calibrate thresholds separately if needed.

---

## Security & Compliance Notes

- Respect ToS for any scraping; prefer public datasets.
- Keep API keys in `.env` and never commit them.
- If using LLM APIs, avoid sending PII; scrub URLs or emails if required by policy.

---

With the above, the team can implement the pipeline step-by-step, swap variants where needed, and run either locally or on Modal for GPU-backed tasks.
