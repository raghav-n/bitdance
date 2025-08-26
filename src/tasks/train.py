"""Training task stubs.

Implement baseline models, encoder fine-tuning, and optional SFT here.
"""

from ..orchestrator import task
from ..orchestrator.utils import data_dir, models_dir, featurizer_slug, seed, model_family, train_run_name


@task(
    name="train",
    inputs=lambda p: [
        f"{data_dir(p)}/features/{featurizer_slug(p)}/tokenized.arrow",
        f"{data_dir(p)}/splits/{seed(p)}/train.txt",
        f"{data_dir(p)}/splits/{seed(p)}/val.txt",
        "configs/base.yaml",
    ],
    outputs=lambda p: [
        f"{models_dir(p)}/{model_family(p)}/{train_run_name(p)}/model.bin",
        f"{models_dir(p)}/{model_family(p)}/{train_run_name(p)}/label_map.json",
    ],
)
def train(params: dict):
    """Train the selected model family and persist artifacts under models/.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError(
        "Implement training (baseline/encoder/SFT) + save artifacts"
    )
