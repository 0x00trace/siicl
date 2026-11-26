"""Where SIICL writes its output, and the guard that keeps it out of the repo."""

from __future__ import annotations

import os
from pathlib import Path


class OutputPathError(ValueError):
    pass


def project_root() -> Path | None:
    """The SIICL source checkout this module runs from, if any."""
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file() and 'name = "siicl"' in pyproject.read_text(errors="replace"):
            return parent
    return None


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "siicl"


def ensure_outside_repo(path: Path, root: Path | None = None) -> Path:
    """Refuse an output folder inside the SIICL repository (see CLAUDE.md)."""
    resolved = path.expanduser().resolve()
    root = root or project_root()
    if root is not None and resolved.is_relative_to(root):
        raise OutputPathError(
            f"{resolved} is inside the SIICL repository ({root}). Output from a game dump must stay "
            "outside the repository; choose another folder."
        )
    return resolved
