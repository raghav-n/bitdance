from __future__ import annotations

import fnmatch
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Union, List

from .logging import get_logger
from . import cache as cache_mod


# Allow static lists or callables that build paths from params
PathSpec = Union[List[str], Callable[[dict], List[str]]]


@dataclass
class TaskSpec:
    name: str
    inputs: PathSpec
    outputs: PathSpec
    fn: Callable[..., None]


def task(name: str, inputs: PathSpec, outputs: PathSpec):
    """Decorator to declare a task on a function.

    The wrapped function should accept a single dict `params` (parsed config),
    and optionally keyword overrides.
    """

    def deco(fn: Callable[..., None]):
        spec = TaskSpec(name=name, inputs=inputs, outputs=outputs, fn=fn)
        setattr(fn, "_task_spec", spec)
        return fn

    return deco


def topo_sort(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[str]:
    nodes = list(nodes)
    incoming = {n: set() for n in nodes}
    outgoing = {n: set() for n in nodes}
    for u, v in edges:
        if u not in incoming or v not in incoming:
            raise ValueError(f"Edge references unknown node: {(u, v)}")
        outgoing[u].add(v)
        incoming[v].add(u)
    ordered: list[str] = []
    roots = [n for n in nodes if not incoming[n]]
    while roots:
        n = roots.pop()
        ordered.append(n)
        for m in list(outgoing[n]):
            incoming[m].discard(n)
            outgoing[n].discard(m)
            if not incoming[m]:
                roots.append(m)
    if any(incoming[n] for n in nodes):
        raise ValueError("Cycle detected in DAG")
    return ordered


class Pipeline:
    def __init__(
        self,
        tasks: dict[str, TaskSpec],
        edges: list[tuple[str, str]],
        name: str = "pipeline",
    ):
        self.name = name
        self.tasks = tasks
        self.edges = edges
        self.order = topo_sort(tasks.keys(), edges)
        self.logger = get_logger(f"orchestrator.{self.name}")

    def _select_subset(
        self, from_step: str | None, until_step: str | None, only_step: str | None
    ) -> list[str]:
        if only_step:
            if only_step not in self.tasks:
                raise KeyError(f"Unknown step: {only_step}")
            return [only_step]
        ordered = self.order
        if from_step:
            if from_step not in self.tasks:
                raise KeyError(f"Unknown step: {from_step}")
            start_idx = ordered.index(from_step)
            ordered = ordered[start_idx:]
        if until_step:
            if until_step not in self.tasks:
                raise KeyError(f"Unknown step: {until_step}")
            end_idx = ordered.index(until_step)
            ordered = ordered[: end_idx + 1]
        return ordered

    def run(
        self,
        params: dict,
        force: set[str] | None = None,
        from_step: str | None = None,
        until_step: str | None = None,
        only_step: str | None = None,
        retries: int = 0,
    ) -> None:
        force = force or set()
        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_dir = (
            Path(params.get("project", {}).get("runs_dir", "runs")) / self.name / run_id
        )
        os.makedirs(run_dir, exist_ok=True)

        # Expose runtime metadata to tasks for dynamic path construction
        params = dict(params)
        params.setdefault("runtime", {})
        params["runtime"]["run_id"] = run_id

        selected = self._select_subset(from_step, until_step, only_step)
        self.logger.info("Selected steps: %s", " → ".join(selected))

        state = {
            "pipeline": self.name,
            "run_id": run_id,
            "steps": [],
            "python": sys.version,
        }

        for step_name in selected:
            spec = self.tasks[step_name]
            step_logger = get_logger(f"orchestrator.{self.name}.{step_name}")
            code_path = Path(inspect.getsourcefile(spec.fn) or "")

            # Resolve dynamic path specs then expand globs
            resolved_inputs = _resolve_paths(spec.inputs, params)
            resolved_outputs = _resolve_paths(spec.outputs, params)
            expanded_inputs = _expand_globs(resolved_inputs)
            expanded_outputs = _expand_globs(resolved_outputs, create_dirs=True)

            task_hash = cache_mod.compute_task_hash(
                name=spec.name,
                input_paths=expanded_inputs,
                code_paths=[code_path] if code_path.exists() else [],
                config=params,
            )

            skip = False
            if step_name not in force:
                skip = cache_mod.is_cached(task_hash, expanded_outputs)
            if skip:
                step_logger.info("Skip (cached): %s", step_name)
                state["steps"].append(
                    {"name": step_name, "status": "cached", "hash": task_hash}
                )
                continue

            attempt = 0
            while True:
                try:
                    step_logger.info("Run: %s", step_name)
                    spec.fn(params=params)
                    cache_mod.write_hash_files(task_hash, expanded_outputs)
                    state["steps"].append(
                        {"name": step_name, "status": "ok", "hash": task_hash}
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    attempt += 1
                    step_logger.exception(
                        "Step failed (%s), attempt %d/%d",
                        step_name,
                        attempt,
                        retries + 1,
                    )
                    if attempt > retries:
                        state["steps"].append(
                            {"name": step_name, "status": "error", "error": str(e)}
                        )
                        _write_state(run_dir, state)
                        raise
            _write_state(run_dir, state)


def _write_state(run_dir: Path, state: dict) -> None:
    with open(run_dir / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _resolve_paths(paths_spec: PathSpec, params: dict) -> list[str]:
    """Resolve a static list of paths or a callable(PathSpec) into a list[str].

    Callables receive the full params dict and must return a list of path strings.
    """
    if callable(paths_spec):
        paths = paths_spec(params)
    else:
        paths = paths_spec
    if paths is None:
        return []
    # Normalize to strings
    out: list[str] = []
    for p in paths:
        out.append(str(p))
    return out


def _expand_globs(patterns: list[str], create_dirs: bool = False) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            base = Path(pat.split("*")[0]).parent if "*" in pat else Path(".")
            for root, _, files in os.walk(base):
                for file in files:
                    p = Path(root) / file
                    if fnmatch.fnmatch(str(p), pat):
                        paths.append(p)
        else:
            p = Path(pat)
            if create_dirs and (p.suffix or p.name):
                p.parent.mkdir(parents=True, exist_ok=True)
            paths.append(p)
    return paths
