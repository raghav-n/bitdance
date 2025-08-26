"""Feature engineering / tokenization task stubs.

Implement TF-IDF or HF tokenization and dataset splits here.
"""

from ..orchestrator import task
from ..orchestrator.utils import data_dir, featurizer_slug, seed, dataset_slug, augment_slug


@task(
    name="featurize",
    inputs=lambda p: [
        f"{data_dir(p)}/annotated/{dataset_slug(p)}/{augment_slug(p)}/annotations.parquet",
        f"{data_dir(p)}/interim/{dataset_slug(p)}/reviews.parquet",
        "configs/base.yaml",
    ],
    outputs=lambda p: [
        f"{data_dir(p)}/features/{featurizer_slug(p)}/tokenized.arrow",
        f"{data_dir(p)}/splits/{seed(p)}/train.txt",
        f"{data_dir(p)}/splits/{seed(p)}/val.txt",
        f"{data_dir(p)}/splits/{seed(p)}/test.txt",
    ],
)
def featurize(params: dict):
    """Build features/tokenized datasets and train/val/test splits.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError(
        "Implement featurization (tokenization/TF-IDF) and data splits"
    )
