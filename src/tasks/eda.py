"""EDA task stubs.

Implement dataset statistics and plots here.
"""

from ..orchestrator import task
from ..orchestrator.utils import data_dir, dataset_slug


@task(
    name="eda",
    inputs=lambda p: [f"{data_dir(p)}/interim/{dataset_slug(p)}/reviews.parquet", "configs/base.yaml"],
    outputs=["reports/eda/index.html"],
)
def eda(params: dict):
    """Generate EDA artifacts (HTML/PNGs) under reports/eda/.

    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError("Implement EDA plots and statistics exports")
