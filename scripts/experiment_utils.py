"""Reproducibility helpers shared by research experiment entry points."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def portable_path(path: Path, root: Path) -> str:
    """Prefer a repository-relative POSIX path in persisted metadata."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(root: Path) -> dict[str, Any]:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in [
                "torch",
                "transformers",
                "sentence-transformers",
                "datasets",
                "faiss-cpu",
                "faiss-gpu",
                "rank-bm25",
                "numpy",
                "scikit-learn",
            ]
        },
        "git_commit": git_commit(root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda": None,
        "gpus": [],
    }
    try:
        import torch

        environment["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            environment["gpus"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except (ImportError, OSError, RuntimeError) as error:
        environment["torch_runtime_error"] = str(error)
    return environment


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except (ImportError, OSError, RuntimeError):
        pass


def seed_worker(worker_id: int) -> None:
    del worker_id
    try:
        import numpy as np
        import torch

        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    except (ImportError, OSError, RuntimeError):
        pass


def write_json_atomic(path: Path, value: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=indent), encoding="utf-8")
    temporary.replace(path)
