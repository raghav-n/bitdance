"""Annotation task stubs.

Implement rule-based and LLM-based annotators and disagreement resolution.
"""

from ..orchestrator import task
from ..orchestrator.utils import data_dir, dataset_slug, augment_slug


@task(
    name="annotate",
    inputs=lambda p: [
        f"{data_dir(p)}/interim/{dataset_slug(p)}/reviews.parquet",
        f"{data_dir(p)}/augmented/{dataset_slug(p)}/{augment_slug(p)}/reviews.parquet",
        "configs/base.yaml",
    ],
    outputs=lambda p: [
        f"{data_dir(p)}/annotated/{dataset_slug(p)}/{augment_slug(p)}/annotations.parquet",
    ],
)
def annotate(params: dict):
    """Apply rules and/or LLMs to label relevancy and policy violations.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError("Implement annotation (rules + LLM + resolution)")
