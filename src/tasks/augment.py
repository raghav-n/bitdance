"""Augmentation task stubs.

Implement back-translation or templated/LLM augmentation here.
"""

from ..orchestrator import task
from ..orchestrator.utils import data_dir, dataset_slug, augment_slug


@task(
    name="augment",
    inputs=lambda p: [
        f"{data_dir(p)}/interim/{dataset_slug(p)}/reviews.parquet",
        "configs/base.yaml",
    ],
    outputs=lambda p: [
        f"{data_dir(p)}/augmented/{dataset_slug(p)}/{augment_slug(p)}/reviews.parquet",
    ],
)
def augment(params: dict):
    """Create synthetic reviews to balance underrepresented classes/languages.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError(
        "Implement augmentation (back-translation, templated, LLM)"
    )
