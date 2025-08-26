"""Ingestion task stubs.

Implement source loaders and dedup logic here.
"""

from ..orchestrator import task


@task(
    name="ingest_all",
    inputs=["configs/base.yaml", "data/raw/*"],
    outputs=[
        "data/raw/googlelocal/reviews.parquet",
        "data/raw/yelp/reviews.parquet",
    ],
)
def ingest_all(params: dict):
    """Read sources, merge, and deduplicate into raw parquet files.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError(
        "Implement ingestion logic (source readers, merge, dedupe)"
    )
