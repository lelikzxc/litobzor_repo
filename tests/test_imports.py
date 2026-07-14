"""Smoke tests for package imports and project structure."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PAPER_PACKAGES = [
    "papers.vit_tiny",
    "papers.ctm_yolov10",
    "papers.vmamba",
    "papers.transformer_segmentation",
    "papers.semiwafernet",
]

COMMON_MODULES = [
    "common",
    "common.utils",
    "common.utils.logger",
    "common.utils.seed",
    "common.utils.config",
    "common.utils.paths",
    "common.metrics",
    "common.metrics.metrics",
    "common.losses",
    "common.datasets",
    "common.visualization",
]


@pytest.mark.parametrize("module_name", COMMON_MODULES + PAPER_PACKAGES)
def test_import_module(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_project_structure() -> None:
    required_dirs = [
        "common/utils",
        "common/metrics",
        "common/losses",
        "common/datasets",
        "common/visualization",
        "configs",
        "docs",
        "scripts",
        "tests",
        "papers/vit_tiny",
        "papers/ctm_yolov10",
        "papers/vmamba",
        "papers/transformer_segmentation",
        "papers/semiwafernet",
    ]
    for rel_path in required_dirs:
        assert (ROOT / rel_path).is_dir(), f"Missing directory: {rel_path}"


@pytest.mark.parametrize("script", ["train.py", "evaluate.py", "predict.py"])
def test_entrypoint_scripts(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Not implemented" in result.stdout
