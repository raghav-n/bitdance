"""Lightweight in-repo orchestrator for the review pipeline.

Provides Task and Pipeline primitives, simple DAG scheduling, caching, and a Typer CLI.
Implementation is minimal to keep hackathon velocity high.
"""

from .core import TaskSpec, Pipeline, task  # re-export for convenience

__all__ = ["TaskSpec", "Pipeline", "task"]
