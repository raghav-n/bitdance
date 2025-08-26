# Lightweight Orchestrator and CLI

This repo includes a lightweight, in-repo orchestrator and CLI to run a modular pipeline for review relevancy and policy violation detection.

The orchestrator favors clarity and hackathon velocity: no heavy dependencies, a simple DAG scheduler, content-addressed caching, and typed task interfaces.

## Quick Start

1) Activate the virtualenv and install deps
```
source .venv/bin/activate
pip install -r requirements.txt
```

2) Inspect the CLI (src layout)
```
python -m src.orchestrator.cli --help
python -m src.orchestrator.cli list
```

3) Add tasks under `src/tasks/`
- Create modules like `src/tasks/ingest.py`, `src/tasks/normalize.py`, etc.
- Decorate task functions with the `@orchestrator.task(...)` decorator.
- Do not change orchestrator internals unless you need to extend scheduling/caching.

Example skeleton:
```python
# src/tasks/ingest.py
from ..orchestrator import task

@task(
    name="ingest_all",
    inputs=["configs/base.yaml", "data/raw/*"],
    outputs=["data/raw/googlelocal/reviews.parquet"],
)
def ingest_all(params: dict):
    raise NotImplementedError("Implement ingestion logic")
```

4) Define config
- Place a YAML config at `configs/base.yaml` with sections like `project`, `ingest`, `annotate`, `train`, `evaluate`.
- The CLI loads this file and passes the parsed dict to each task as `params`.

5) Run a pipeline (after tasks are implemented)
```
python -m src.orchestrator.cli full_run --config configs/base.yaml
```

Use `--from-step`, `--until-step`, or `--force` to control execution and caching:
```
python -m src.orchestrator.cli full_run --from-step annotate --force featurize,train
```

## Concepts

- Task: Declares name, inputs (paths/globs or callables of `params`), outputs (paths or callables), and a function `fn(params)`.
- Pipeline: A DAG of tasks with dependencies. Runs in topological order.
- Cache: The orchestrator hashes input files, task code, and config to determine if outputs are up-to-date. Hashes are written next to outputs as `.hash` files.
- Runs: The orchestrator writes run state to `runs/<pipeline>/<run_id>/state.json` and logs events to standard output.

## Project Structure (suggested)

```
src/
  orchestrator/
    core.py        # Task, Pipeline, DAG, hashing
    cache.py       # content-addressed cache utilities
    cli.py         # Typer CLI
    logging.py     # logging helpers
  tasks/
    ingest.py      # implement later
    normalize.py   # implement later
    eda.py         # implement later
    augment.py     # implement later
    annotate.py    # implement later
    featurize.py   # implement later
    train.py       # implement later
    evaluate.py    # implement later
configs/
  base.yaml        # (recommended) configuration file
data/
  ...              # datasets and artifacts
```

## Notes

- This scaffold ships without task implementations. The CLI will report missing tasks until you add them.
- Keep tasks pure wrt inputs/outputs. If a task writes additional files, include them in `outputs` so caching remains correct.
- For secrets and API keys, use a `.env` file and load in your task code (do not commit secrets).

## Config-driven Artifact Paths

To enable plug-and-play experiments (multiple datasets, tokenizers, and model runs), task inputs/outputs are now derived from config at runtime.

- Featurized datasets: `data/features/{featurizer}/tokenized.arrow` where `{featurizer}` is `hf-{train.model_name}` for encoders (slashes replaced), or a custom name for TF-IDF.
- Splits: `data/splits/{seed}/(train|val|test).txt` where `{seed}` is `featurize.seed` or fallback to `project.seed`.
- Normalized data: `data/interim/{dataset.slug}/reviews.parquet`.
- Augmented data: `data/augmented/{dataset.slug}/{augment.slug}/reviews.parquet`.
- Annotations: `data/annotated/{dataset.slug}/{augment.slug}/annotations.parquet`.
- Model artifacts: `models/{train.family}/{train.run_name or runtime.run_id}/...`.
- Metrics: `reports/metrics/{train.run_name or runtime.run_id}/metrics.json`.

Add these optional sections to `configs/base.yaml` (defaults shown):

```
dataset:
  slug: base
augment:
  slug: none
featurize:
  family: encoder   # or tfidf
  seed: 42          # overrides project.seed for splits
train:
  run_name: null    # optional; otherwise uses the pipeline run_id timestamp
```

Because inputs/outputs are callables of `params`, you can swap models, datasets, or augmentation schemes by editing the config and re-running without changing code. Caching keys include the resolved paths and config, so parallel variants can coexist.
