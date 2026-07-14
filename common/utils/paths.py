"""Project path resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_MARKER_FILES = ("pyproject.toml", "README.md")


def _find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by searching for marker files."""
    current = (start or Path(__file__).resolve()).parent
    for path in (current, *current.parents):
        if any((path / marker).is_file() for marker in _MARKER_FILES):
            return path
    raise FileNotFoundError("Could not locate project root from current path.")


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths used across the research repository."""

    root: Path
    configs: Path
    data: Path
    checkpoints: Path
    logs: Path
    papers: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> ProjectPaths:
        """Build project paths relative to the repository root.

        Args:
            root: Optional repository root. When omitted, paths are resolved
                automatically from the project layout.

        Returns:
            Resolved project paths.
        """
        project_root = Path(root).resolve() if root is not None else _find_project_root()
        if not any((project_root / marker).is_file() for marker in _MARKER_FILES):
            raise FileNotFoundError(
                f"Could not locate project root from current path: {project_root}"
            )

        return cls(
            root=project_root,
            configs=project_root / "configs",
            data=project_root / "data",
            checkpoints=project_root / "checkpoints",
            logs=project_root / "logs",
            papers=project_root / "papers",
        )

    def ensure_runtime_dirs(self) -> None:
        """Create runtime directories that may not exist in version control."""
        for path in (self.data, self.checkpoints, self.logs):
            path.mkdir(parents=True, exist_ok=True)
