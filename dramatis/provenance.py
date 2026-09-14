"""Small helpers for recording what produced an artifact: git state, file hashes, timestamps."""

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def git_dirty(paths: list[str] | None = None) -> bool:
    """True if there are uncommitted changes (optionally restricted to the given paths)."""
    cmd = ["git", "status", "--porcelain", "--"] + (paths or [])
    return bool(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())


def file_sha256(path: str | Path, chunk_size: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
