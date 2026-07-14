"""Tests for common.utils.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from common.utils.config import Config

ROOT = Path(__file__).resolve().parents[1]


def test_from_yaml_loads_default_config() -> None:
    config = Config.from_yaml(ROOT / "configs" / "default.yaml")
    assert config.get("project.name") == "litobzor"
    assert config.get("project.seed") == 42


def test_from_yaml_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(ROOT / "configs" / "missing.yaml")


def test_get_dot_notation_and_default() -> None:
    config = Config({"training": {"batch_size": 16, "optimizer": "adam"}})
    assert config.get("training.batch_size") == 16
    assert config.get("training.momentum") is None
    assert config.get("training.momentum", 0.9) == 0.9


def test_get_missing_nested_key_returns_default() -> None:
    config = Config({"project": {"name": "litobzor"}})
    assert config.get("project.seed", 0) == 0
    assert config.get("logging.level") is None


def test_from_yaml_empty_file(tmp_path: Path) -> None:
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("", encoding="utf-8")
    config = Config.from_yaml(empty_config)
    assert config.raw == {}


def test_from_yaml_invalid_root_type(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(yaml.dump(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(TypeError, match="Expected YAML mapping"):
        Config.from_yaml(invalid_config)
