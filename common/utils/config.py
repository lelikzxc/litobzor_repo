"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Base configuration container for YAML-based experiment configs."""

    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed configuration object.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
            TypeError: If the root YAML document is not a mapping.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(f"Expected YAML mapping at root, got {type(data).__name__}")

        return cls(raw=data)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by dot-separated key.

        Args:
            key: Dot-separated configuration key (e.g. ``training.batch_size``).
            default: Value returned when the key is missing.

        Returns:
            Configuration value or default.
        """
        node: Any = self.raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
