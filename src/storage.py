"""Storage helpers that work transparently for local paths and remote URIs.

Every artifact path in the project is a plain string. When ``STORAGE_BACKEND=s3``
the paths become ``s3://bucket/prefix/...`` URIs; otherwise they are local
filesystem paths. Routing all reads/writes through this module means the rest of
the codebase never has to care which backend is active.

Remote access uses ``fsspec``/``s3fs`` (and pandas' native ``s3://`` support for
parquet). ``fsspec`` is imported lazily so a purely local run does not require it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _fsspec():
    try:
        import fsspec
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without s3 extras
        raise RuntimeError(
            "fsspec/s3fs is required for remote storage paths. Install with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return fsspec


def is_remote(path: str | os.PathLike[str]) -> bool:
    """Return True for URIs such as ``s3://...`` and False for local paths."""
    return "://" in str(path)


def join(root: str | os.PathLike[str], *parts: str) -> str:
    """Join path parts with forward slashes, preserving any URI scheme on ``root``."""
    base = str(root).rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part)
    return f"{base}/{suffix}" if suffix else base


def _ensure_parent(path: str) -> None:
    if is_remote(path):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def exists(path: str | os.PathLike[str]) -> bool:
    path = str(path)
    if is_remote(path):
        fs, _, paths = _fsspec().get_fs_token_paths(path)
        return bool(fs.exists(paths[0]))
    return Path(path).exists()


def read_parquet(path: str | os.PathLike[str]) -> pd.DataFrame:
    return pd.read_parquet(str(path))


def write_parquet(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    path = str(path)
    _ensure_parent(path)
    df.to_parquet(path, index=False)


def read_text(path: str | os.PathLike[str]) -> str:
    path = str(path)
    if is_remote(path):
        with _fsspec().open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | os.PathLike[str], text: str) -> None:
    path = str(path)
    _ensure_parent(path)
    if is_remote(path):
        with _fsspec().open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return
    Path(path).write_text(text, encoding="utf-8")


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(read_text(path))


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    write_text(path, json.dumps(data, indent=2, sort_keys=True, default=str))


def save_figure(fig: Any, path: str | os.PathLike[str], dpi: int = 150) -> str:
    """Save a matplotlib figure to local disk or a remote URI and return the path."""
    path = str(path)
    _ensure_parent(path)
    fmt = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else "png"
    if is_remote(path):
        with _fsspec().open(path, "wb") as handle:
            fig.savefig(handle, dpi=dpi, format=fmt)
    else:
        fig.savefig(path, dpi=dpi)
    return path
