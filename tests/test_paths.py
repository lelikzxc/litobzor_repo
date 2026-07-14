"""Tests for common.utils.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.utils.paths import ProjectPaths

ROOT = Path(__file__).resolve().parents[1]


def test_from_root_auto_detects_repository_root() -> None:
    paths = ProjectPaths.from_root()
    assert paths.root == ROOT
    assert paths.configs == ROOT / "configs"
    assert paths.papers == ROOT / "papers"


def test_from_root_with_explicit_path() -> None:
    paths = ProjectPaths.from_root(ROOT)
    assert paths.root == ROOT.resolve()
    assert paths.data == ROOT / "data"
    assert paths.checkpoints == ROOT / "checkpoints"
    assert paths.logs == ROOT / "logs"


def test_ensure_runtime_dirs_creates_directories(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tmp'\n", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_runtime_dirs()

    assert paths.data.is_dir()
    assert paths.checkpoints.is_dir()
    assert paths.logs.is_dir()


def test_from_root_missing_marker_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not locate project root from current path"):
        ProjectPaths.from_root(tmp_path)
