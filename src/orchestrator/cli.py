from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict

import typer
import yaml

from .core import Pipeline, TaskSpec
from .logging import get_logger


app = typer.Typer(add_completion=False, help="Lightweight pipeline orchestrator CLI")
log = get_logger("orchestrator.cli")


def load_config(path: str | Path) -> dict:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_tasks() -> Dict[str, TaskSpec]:
    """Import all modules in `tasks` package and collect decorated functions."""
    tasks_pkg = "src.tasks"
    specs: Dict[str, TaskSpec] = {}
    try:
        pkg = importlib.import_module(tasks_pkg)
    except ModuleNotFoundError:
        log.warning("No tasks package found.")
        return specs
    for m in pkgutil.iter_modules(pkg.__path__, prefix=f"{tasks_pkg}."):
        try:
            mod = importlib.import_module(m.name)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to import %s: %s", m.name, e)
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            spec = getattr(obj, "_task_spec", None)
            if isinstance(spec, TaskSpec):
                specs[spec.name] = spec
    return specs


@app.command("list")
def list_tasks():
    """List discovered tasks."""
    specs = discover_tasks()
    if not specs:
        typer.echo(
            "No tasks discovered. Create modules under `tasks/` and decorate functions with @task()."
        )
        raise typer.Exit(code=0)
    typer.echo("Discovered tasks:")
    for name in sorted(specs.keys()):
        typer.echo(f"- {name}")


@app.command()
def run_task(
    name: str = typer.Argument(..., help="Task name to run"),
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
    force: bool = typer.Option(False, help="Ignore cache for this task"),
    retries: int = typer.Option(0, help="Retries on failure"),
):
    """Run a single task by name."""
    specs = discover_tasks()
    if name not in specs:
        typer.echo(f"Task not found: {name}")
        raise typer.Exit(code=1)
    spec = specs[name]
    params = load_config(config)
    pipe = Pipeline(tasks={name: spec}, edges=[], name=f"task.{name}")
    pipe.run(
        params=params, force={name} if force else set(), only_step=name, retries=retries
    )


@app.command()
def full_run(
    config: str = typer.Option("configs/base.yaml", help="Path to YAML config"),
    force: str = typer.Option("", help="Comma-separated tasks to force"),
    from_step: str = typer.Option("", help="Start from this step name"),
    until_step: str = typer.Option("", help="Stop after this step name"),
    retries: int = typer.Option(0, help="Retries per task on failure"),
):
    """Run the full pipeline, if tasks are defined in `tasks/` package."""
    specs = discover_tasks()
    required = [
        "ingest_all",
        "annotate",
        "augment",
        "normalize",
        "train",
    ]
    missing = [r for r in required if r not in specs]
    if missing:
        typer.echo(
            "Missing required tasks: "
            + ", ".join(missing)
            + "\nCreate them under src/tasks/ and decorate with @task(name=..., inputs=[...], outputs=[...])."
        )
        raise typer.Exit(code=1)

    # Streamlined pipeline: go straight from normalize to train.
    edges = [
        ("ingest_all", "annotate"),
        ("annotate", "augment"),
        ("annotate", "normalize"),
        ("augment", "normalize"),
        ("normalize", "train"),
    ]

    params = load_config(config)
    pipe = Pipeline(tasks={k: specs[k] for k in required}, edges=edges, name="full_run")
    force_set = set([x.strip() for x in force.split(",") if x.strip()])
    pipe.run(
        params=params,
        force=force_set,
        from_step=from_step or None,
        until_step=until_step or None,
        retries=retries,
    )


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
