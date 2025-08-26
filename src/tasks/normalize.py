"""Normalization & cleaning task stubs.

Implement schema normalization, language detection, and text cleaning here.
"""

from ..orchestrator import task
from ..orchestrator.utils import data_dir, dataset_slug


@task(
    name="normalize",
    inputs=[
        "configs/base.yaml",
        "data/raw/googlelocal/reviews.parquet",
        "data/raw/yelp/reviews.parquet",
    ],
    outputs=lambda p: [f"{data_dir(p)}/interim/{dataset_slug(p)}/reviews.parquet"],
)
def normalize(params: dict):
    """Normalize/clean raw sources into a unified schema parquet.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError(
        "Implement normalization (schema mapping, cleaning, lang detect)"
    )
