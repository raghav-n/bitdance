"""Evaluation task stubs.

Implement metrics, PR curves, and error analysis exports here.
"""

from ..orchestrator import task
from ..orchestrator.utils import data_dir, models_dir, featurizer_slug, seed, model_family, train_run_name


@task(
    name="evaluate",
    inputs=lambda p: [
        f"{models_dir(p)}/{model_family(p)}/{train_run_name(p)}/model.bin",
        f"{models_dir(p)}/{model_family(p)}/{train_run_name(p)}/label_map.json",
        f"{data_dir(p)}/splits/{seed(p)}/test.txt",
        f"{data_dir(p)}/features/{featurizer_slug(p)}/tokenized.arrow",
        "configs/base.yaml",
    ],
    outputs=lambda p: [
        f"reports/metrics/{train_run_name(p)}/metrics.json",
    ],
)
def evaluate(params: dict):
    """Evaluate on a holdout set and write metrics under reports/metrics/.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError("Implement evaluation and metrics export")
