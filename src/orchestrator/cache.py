from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_stat(path: Path) -> dict:
    try:
        st = path.stat()
        return {"size": st.st_size, "mtime": st.st_mtime}
    except FileNotFoundError:
        return {"size": None, "mtime": None}


def compute_task_hash(
    name: str, input_paths: Iterable[Path], code_paths: Iterable[Path], config: dict
) -> str:
    payload: dict = {"name": name, "inputs": [], "code": [], "config": config}
    for p in sorted({str(p) for p in input_paths}):
        pp = Path(p)
        entry = {"path": p}
        if pp.exists() and pp.is_file():
            entry["digest"] = file_digest(pp)
            entry["stat"] = safe_stat(pp)
        else:
            entry["digest"] = None
            entry["stat"] = safe_stat(pp)
        payload["inputs"].append(entry)
    for cp in sorted({str(p) for p in code_paths}):
        cpp = Path(cp)
        entry = {"path": cp}
        if cpp.exists() and cpp.is_file():
            entry["digest"] = file_digest(cpp)
            entry["stat"] = safe_stat(cpp)
        else:
            entry["digest"] = None
            entry["stat"] = safe_stat(cpp)
        payload["code"].append(entry)
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return sha256_bytes(data)


def _hash_file_for_output(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".hash")


def is_cached(task_hash: str, output_paths: Iterable[Path]) -> bool:
    outputs = list(output_paths)
    if not outputs:
        return False
    # All outputs must exist and match hash
    for p in outputs:
        if not p.exists():
            return False
        hf = _hash_file_for_output(p)
        if not hf.exists():
            return False
        try:
            cached = hf.read_text(encoding="utf-8").strip()
        except Exception:
            return False
        if cached != task_hash:
            return False
    return True


def write_hash_files(task_hash: str, output_paths: Iterable[Path]) -> None:
    for p in output_paths:
        # Ensure parent exists for both output and hash file
        p.parent.mkdir(parents=True, exist_ok=True)
        hf = _hash_file_for_output(p)
        try:
            hf.write_text(task_hash, encoding="utf-8")
        except FileNotFoundError:
            # If output doesn't exist (e.g., task decided to skip writing), we still write hash next to intended path
            hf.parent.mkdir(parents=True, exist_ok=True)
            hf.write_text(task_hash, encoding="utf-8")
