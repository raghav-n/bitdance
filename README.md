# Bitdance Review Moderation Pipeline

This document describes the end‑to‑end pipeline for moderating Google business reviews sourced from a Kaggle dataset (https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews). It explains how the pipeline is orchestrated, what each stage produces, which models are trained, how development and infrastructure tools are used (with a detailed look at Modal), and what improvements are planned next.

## 1) Pipeline Overview

The pipeline is implemented with a lightweight in‑repository orchestrator under `src/orchestrator` and task modules under `src/tasks`. It represents the workflow as a small directed acyclic graph (DAG) with explicit edges, content‑addressed caching, and a simple Typer‑based command‑line interface. The default execution order is: ingest_all, annotate, augment, normalize, train, and evaluate.

This design works well for this project because it is easy to understand and modify, and because it encourages rapid iteration. Tasks declare their inputs, outputs, and code, and the orchestrator hashes these to decide whether a step can be skipped safely. This makes it fast to tweak a stage and rerun only what is necessary. The approach avoids heavyweight workflow systems and keeps the runtime surface small while still offering useful features such as selective execution (for example, starting at a step or stopping after a step) and persistent run metadata.

The ingestion step reads the raw CSV of Google reviews, validates the presence of expected columns, and constructs a unified schema. When image paths are available, it loads, converts, and resizes the images, and it deduplicates reviews by a combination of place, user, and text content. The step writes Parquet datasets for text‑only and image‑annotated reviews, as well as a pickle containing the image arrays, under `data/raw/restaurant_reviews/`.

The annotation step applies policy labels using Google’s Gemini models through few‑shot prompting. For each review, the process asks whether it violates the “No Advertisement,” “No Irrelevant Content,” or “No Rant Without Visit” policies. It returns structured JSON from the model, extracts the decision, and writes one column per policy (`is_noadvertisement`, `is_noirrelevantcontent`, and `is_norantwithoutvisit`). The outputs are written to `data/annotated/restaurant_reviews/annotations.parquet` and a mirrored CSV.

The augmentation step generates synthetic reviews that intentionally violate the specified policies in order to balance the training data. It uses Gemini with lightweight prompts to create short, realistic reviews in the desired language and associates each with sampled metadata such as the place and a user handle. The step produces `data/augmented/restaurant_reviews/reviews.parquet` and a CSV with the same content.

The normalization step merges the annotated original reviews and the synthetic reviews into a single, training‑ready dataset with consistent columns. It derives simple features such as a `has_url` flag and maps the annotations to the supervised label names used by the model code. To avoid business leakage across splits, it assigns entire businesses to a single split and targets a 70/15/15 ratio for train, validation, and test sets. It writes an interim Parquet for exploratory analysis and JSONL files for the three splits under `data/processed/{dataset}/`.

The training step runs one or more model families depending on the configuration. It supports a baseline recurrent neural network, an encoder‑based BERT‑family classifier, and a supervised fine‑tuned (SFT) large language model using LoRA adapters. Each training run uses early stopping on validation loss where available and writes artifacts under `models/{family}/{run_name}/`, including a primary model file (`model.bin` or an equivalent manifest) and a `label_map.json` file.

The evaluation step scores trained models on the test split produced by normalization. It calculates per‑label precision, recall, F1 score, and average precision, as well as macro and micro aggregates and per‑label confusion statistics. It writes machine‑readable results under `reports/metrics/{model-id}/metrics.json` and emits predictions to `predictions.jsonl` for further analysis.

Operationally, the orchestrator caches each task by hashing its inputs, relevant source code, and configuration so that it can skip a step when nothing that affects its outputs has changed. It exposes a small Typer CLI that can list discovered tasks, run an individual task by name, or run the full graph with options to force steps or truncate the range of steps. Each invocation writes a `runs/{pipeline}/{run_id}/state.json` file that records which steps ran, which were cached, and any errors encountered.

## Running the Pipeline

Before you run the pipeline, you should ensure that your Python environment is ready and that the configuration references the correct dataset paths and API keys. If the virtual environment has not yet been created, you can create it and install dependencies by activating `.venv` and using `uv` for installation. You can then run the full pipeline or individual steps using the Typer CLI.

To prepare the environment, activate the virtual environment and install the requirements with `uv pip`:

```
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt -r requirements_models.txt
```

Next, verify that `configs/base.yaml` points to your Kaggle Google reviews CSV. The path is read from `ingest.sources[0].path`. If you plan to run annotation or augmentation, you should also provide a Gemini API key either via environment variable or in the config. For example, you can export the keys in your shell before running:

```
export GEMINI_API_KEY=...      # required for annotate/augment
export HF_TOKEN=...            # optional, for Hugging Face model downloads
```

You can list all discovered tasks to confirm the CLI is wired correctly:

```
python -m src.orchestrator.cli list
```

You can run the full pipeline using the base configuration:

```
python -m src.orchestrator.cli full_run --config configs/base.yaml
```

You can also run a subset of steps. For example, you can skip directly to normalization and training, or you can force specific steps to re‑run even if cached:

```
python -m src.orchestrator.cli full_run --config configs/base.yaml \
  --from-step normalize --until-step train

python -m src.orchestrator.cli full_run --config configs/base.yaml \
  --force annotate,augment
```

You can run a single task for faster iteration. This is useful when you are working on one stage at a time:

```
python -m src.orchestrator.cli run_task normalize --config configs/base.yaml
python -m src.orchestrator.cli run_task train --config configs/base.yaml --retries 1
```

If a training run did not automatically trigger evaluation, you can evaluate explicitly. The evaluation task reads the test split from the normalization step and either evaluates the most recent training run or the models listed in `evaluate.models` within the config:

```
python -m src.orchestrator.cli run_task evaluate --config configs/base.yaml
```

If you prefer to use Modal for remote training or evaluation, you can enable it in the configuration by setting `train.use_modal: true` and/or `evaluate.use_modal: true`. You can also select the GPU type (for example, `T4`), provide a Modal volume name for persistent storage, and choose whether to download remote artifacts back to your local `models/` directory. When Modal is enabled, the pipeline uploads the necessary inputs, runs the job in a reproducible container, and stores outputs in the configured volume, while keeping your local environment lightweight.

## 2) Training Details

The supervised label space consists of three policy‑violation labels: `irrelevant_content`, `advertisement`, and `review_without_visit`. The models treat this as a multi‑label classification problem where each label is an independent binary prediction. The JSONL input produced by the normalization step includes compact, structured text that concatenates fields such as the business name, author name, rating, review text, and flags like `has_url` or `has_photo` to give the model simple contextual cues.

The baseline recurrent model uses a small vocabulary with tokenization, a bidirectional LSTM encoder, max pooling, and a feed‑forward head trained with binary cross‑entropy with logits. It trains with early stopping on the validation set and saves a `model.pt` checkpoint along with a training‑history JSON file. This model provides a fast, transparent baseline that is easy to train and deploy.

The encoder model uses Hugging Face Transformers with `AutoModelForSequenceClassification` and a tokenizer from the chosen encoder family (for example, DistilBERT or BERT). It is configured for multi‑label classification and uses a data collator with padding. The training loop enables evaluation each epoch, supports early stopping, and writes a `training_history.json` file for later inspection. A helper function computes probabilities for evaluation and inference, which allows the evaluation task to call it in a uniform way.

The SFT‑LoRA model fine‑tunes an instruction‑tuned causal language model using TRL’s `SFTTrainer` and PEFT’s LoRA adapters. The training data is formatted as short chat exchanges with a system instruction and a user message containing the structured review fields, followed by a gold JSON response with boolean values for each label. At inference time, constrained decoding is applied using an allowed‑token filter and an early stopping criterion when the closing brace is generated. This forces the model to output valid JSON that can be parsed into boolean predictions reliably. On GPUs, the model can run in 4‑bit quantization to reduce memory usage, and the trainer optionally oversamples positive examples to address class imbalance.

In addition to the supervised training, the repository includes an unsupervised similarity analysis that operates per business. The goal of this analysis is to detect unusually similar groups of reviews that may indicate templated content or coordinated manipulation. It computes normalized sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`, clusters them with DBSCAN using cosine distance, and summarizes each business with cohesion metrics such as the mean of the top‑k pairwise similarities, the fraction of near‑duplicate pairs, and statistics for the largest high‑similarity subclusters. The artifacts are written as `business_scores.jsonl` and `clusters.jsonl` under the model directory with a small manifest file for compatibility with the orchestrator.

The codebase also provides a fusion utility that combines base supervised probabilities with the unsupervised fraud‑similarity scores for exploratory analysis. It can scale the per‑review similarity score by business‑level cohesion and blend it with the base model’s output, although the primary labels do not include an explicit “fraudulent_review” class in this project. In its current form, the unsupervised method is not very good and requires substantial tuning of thresholds and cluster parameters. There was not enough time to perform that tuning, so the results should be treated as leads rather than final decisions.

The evaluation procedure supports all of the model families mentioned above. It reads the test set produced by normalization, obtains probabilities from the appropriate prediction function, applies a default threshold of 0.5 to form binary predictions, and computes per‑label and aggregate metrics. It writes the metrics and the raw predictions to disk so that the results can be visualized or inspected later without re‑running inference.

## Results Summary

- Dataset: 211 test samples; 3 labels; decision threshold 0.5.
- Top overall: `encoder/enc-bert-large-cased` — micro F1 0.920, macro F1 0.920.
- Runner‑up: `encoder/enc-xlm-roberta-base` — micro F1 0.913, macro F1 0.915.
- Best SFT: `sft/sft-gemma-3-12b` — micro F1 0.857, macro F1 0.865.
- Others: `sft/sft-gemma-3-4b` — micro F1 0.786, macro F1 0.790; `sft/sft-ministral-8b` — micro F1 0.763, macro F1 0.785.

Per‑label highlights

- Advertisement: strongest label overall — F1 0.875–1.00; perfect 1.00 for both encoders and `gemma-3-12b`.
- Irrelevant content: hardest label — F1 0.655–0.872; encoders lead on precision (1.00 for `enc-bert`), with recall around 0.77–0.82.
- Review without visit: F1 0.722–0.889; `gemma-3-12b` has highest recall (0.933), encoders have highest precision (1.00) with recall ~0.80.

Raw metrics (reports/metrics)

- `reports/metrics/encoder-enc-bert-large-cased/metrics.json` — micro F1 0.920, macro F1 0.920; per‑label F1: [irrelevant 0.872, advertisement 1.000, review_without_visit 0.889].
- `reports/metrics/encoder-enc-xlm-roberta-base/metrics.json` — micro F1 0.913, macro F1 0.915; per‑label F1: [irrelevant 0.857, advertisement 1.000, review_without_visit 0.889].
- `reports/metrics/sft-sft-gemma-3-12b/metrics.json` — micro F1 0.857, macro F1 0.865; per‑label F1: [irrelevant 0.773, advertisement 1.000, review_without_visit 0.824].
- `reports/metrics/sft-sft-gemma-3-4b/metrics.json` — micro F1 0.786, macro F1 0.790; per‑label F1: [irrelevant 0.773, advertisement 0.875, review_without_visit 0.722].
- `reports/metrics/sft-sft-ministral-8b/metrics.json` — micro F1 0.763, macro F1 0.785; per‑label F1: [irrelevant 0.655, advertisement 0.944, review_without_visit 0.757].

Notes

- Encoder classifiers outperform SFT models on this task and dataset size; both encoder variants achieve perfect F1 on the advertisement label and the best overall micro/macro F1.
- SFT models tend to trade higher recall for lower precision on some labels, leading to more false positives, especially for `irrelevant_content`.

## 3) Development Tools, APIs, and Libraries

The project is developed primarily in Visual Studio Code with standard Python tooling, and ad‑hoc analysis and metric inspection are performed in Jupyter notebooks such as `eda_analysis.ipynb`. The orchestrator exposes a small Typer CLI, so you can run the full pipeline with a command such as `python -m src.orchestrator.cli full_run --config configs/base.yaml`, or you can run an individual step with `python -m src.orchestrator.cli run_task <name>`.

The annotation and augmentation steps use the Gemini API via the `google-generativeai` library. The code formats few‑shot prompts, requests a concise JSON response, and sanitizes the output to extract the model’s decision. The API key is resolved from configuration and environment variables so that the steps can run locally or remotely with minimal changes.

The project relies on a standard stack of data and machine‑learning libraries, including pandas and NumPy for data manipulation, scikit‑learn for evaluation metrics, PyTorch for the baseline model, Hugging Face Transformers and Datasets for the encoder model, TRL and PEFT for SFT with LoRA, and sentence‑transformers for the embedding‑based similarity analysis. The orchestrator and configuration helpers are implemented with Typer, PyYAML, and dotenv, and the logging is configured for readable progress during runs.

Modal is used extensively for remote training and evaluation. When Modal is enabled in the configuration, the training and evaluation code constructs a container image from the project’s requirements, sets environment variables such as `HF_HUB_ENABLE_HF_TRANSFER` and `HF_TOKEN`, and runs the job on a GPU selected by configuration (for example, a T4). The code mounts a persistent Modal Volume (by default named `bitdance-models`) at `/vol`, writes models under `/vol/models/...`, and writes evaluation artifacts under `/vol/reports/...`. For some workflows, the code downloads a tarball of the artifacts from the volume and extracts it into the local `models/{family}/{run_name}` directory; in other cases, it writes a small local manifest that points to the remote location. This setup keeps the local environment lightweight while making it easy to resume or reuse artifacts across jobs, and it enables concurrent fan‑out when training or evaluating multiple models in parallel.

## 4) Next Steps and Improvements

The supervised models can benefit from per‑label threshold tuning and probability calibration so that operating points better reflect the desired precision‑recall trade‑offs. It would also be sensible to try stronger or domain‑specific encoders and multilingual variants, and to experiment with loss functions such as class‑balanced or focal losses that can improve learning on imbalanced datasets. The SFT prompting can be refined and the constrained decoding grammar can be expanded so that the model is both more faithful and more robust; the resulting behavior could then be distilled into smaller encoder models for faster inference.

The unsupervised fraud‑similarity analysis should be treated as an exploratory component that needs more work. The similarity thresholds, minimum cluster sizes, and top‑k parameters should be tuned by business size, and alternative clustering methods such as HDBSCAN may provide better behavior on variable‑density data. It would also help to incorporate time windows and author‑level features, to look for cross‑business duplication or rings, and to surface exemplar pairs so that reviewers can quickly assess the evidence. Lightweight near‑duplicate detection, such as MinHash or locality‑sensitive hashing, could provide a complementary signal.

The data and labels can be improved by expanding and cleaning the annotations, and by adding an active‑learning loop that prioritizes uncertain or borderline cases for manual review. Synthetic augmentation can be broadened in both content and language coverage, and the generation prompts can be adjusted to minimize repetitive templates.

Finally, the pipeline itself can be strengthened by adding experiment tracking for runs and metrics, by enriching the evaluation artifacts and dashboards under `reports/metrics/`, and by extending the orchestrator with optional parallel execution and more detailed cache provenance. The Modal integration can also be deepened with job queues for larger grid searches, automated artifact synchronization, and guardrails for long‑running SFT jobs.
