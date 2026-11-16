"""Build identity reported by the running service."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from config.settings import PROJECT_ROOT

_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "tests",
    "venv",
}
_SOURCE_SUFFIXES = {".css", ".js", ".jsx", ".py", ".ts", ".tsx"}


def build_fingerprint(root: Path = PROJECT_ROOT) -> dict[str, str | int]:
    """Hash tracked-style Python source files without including local runtime data."""
    digest = hashlib.sha256()
    file_count = 0
    for current_root, directories, filenames in os.walk(root):
        directories[:] = sorted(
            name for name in directories
            if name not in _EXCLUDED_DIRS and not name.startswith(".")
        )
        for filename in sorted(filenames):
            if Path(filename).suffix not in _SOURCE_SUFFIXES:
                continue
            path = Path(current_root) / filename
            relative_path = path.relative_to(root).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
            file_count += 1
    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "source_files": file_count,
    }


RUNTIME_BUILD = build_fingerprint()
RUNTIME_PROCESS_ID = os.getpid()
