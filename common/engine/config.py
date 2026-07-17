"""YAML-based configuration with dot-separated nested access.

``EngineConfig`` wraps a raw ``dict`` and provides:

- ``from_yaml()`` / ``from_dict()`` class methods
- ``get(key, default)`` with dot-separated key support
- ``__getitem__`` / ``__contains__`` for dict-like access
- ``merge()`` / ``merge_deep()`` for combining configs
- ``to_dict()`` / ``to_yaml()`` for serialisation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class EngineConfig:
    """Configuration container with dot-separated nested access.

    Args:
        raw: Nested dictionary of configuration values.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw: dict[str, Any] = raw

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed ``EngineConfig`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            TypeError: If the YAML root is not a mapping.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected YAML mapping at root, got {type(data).__name__}"
            )
        return cls(raw=data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineConfig:
        """Create an ``EngineConfig`` from an existing dictionary.

        Args:
            d: Nested configuration dictionary.

        Returns:
            New ``EngineConfig`` instance.
        """
        return cls(raw=d.copy())

    # ── Accessors ─────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by dot-separated key.

        Args:
            key: Dot-separated path (e.g. ``"training.batch_size"``).
            default: Value returned when the key is missing.

        Returns:
            The configuration value or *default*.
        """
        node: Any = self._raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        """Dict-style access via ``config["training.batch_size"]``.

        Raises:
            KeyError: If the key is not found.
        """
        value = self.get(key, _sentinel := object())
        if value is _sentinel:
            raise KeyError(f"Key {key!r} not found in config")
        return value

    def __contains__(self, key: str) -> bool:
        """Check if a key exists (``"key" in config``)."""
        return self.get(key, _sentinel := object()) is not _sentinel

    # ── Mutation ──────────────────────────────────────────────────────────

    def merge(self, other: dict[str, Any] | EngineConfig) -> EngineConfig:
        """Shallow-merge *other* into this config (top-level keys only).

        Args:
            other: Dictionary or ``EngineConfig`` to merge in.

        Returns:
            ``self`` for chaining.
        """
        if isinstance(other, EngineConfig):
            other = other._raw
        self._raw.update(other)
        return self

    def merge_deep(self, other: dict[str, Any] | EngineConfig) -> EngineConfig:
        """Deep-merge *other* into this config (recursive dict update).

        Args:
            other: Dictionary or ``EngineConfig`` to merge in.

        Returns:
            ``self`` for chaining.
        """
        if isinstance(other, EngineConfig):
            other = other._raw
        self._deep_merge(self._raw, other)
        return self

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
        """Recursively merge *override* into *base*."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                EngineConfig._deep_merge(base[key], value)
            else:
                base[key] = value

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the raw configuration dictionary."""
        return _deep_copy(self._raw)

    def to_yaml(self, path: str | Path | None = None) -> str | None:
        """Serialise config to YAML string or file.

        Args:
            path: Optional file path. If provided, writes to file.

        Returns:
            YAML string if *path* is ``None``, otherwise ``None``.
        """
        yaml_str = yaml.dump(
            self._raw, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
        if path is not None:
            Path(path).write_text(yaml_str, encoding="utf-8")
            return None
        return yaml_str

    def __repr__(self) -> str:
        return f"EngineConfig({self._raw!r})"


def _deep_copy(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively copy a nested dictionary."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out[key] = _deep_copy(value)
        elif isinstance(value, list):
            out[key] = [v.copy() if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out