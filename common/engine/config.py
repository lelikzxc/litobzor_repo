"""Configuration helpers for the engine.

Supports loading YAML, nested dictionary access, default values,
and merging user overrides. No CLI support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class EngineConfig:
    """Configuration container with nested access, defaults, and merging.

    Args:
        data: Raw configuration dictionary.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data) if data else {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A new ``EngineConfig`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
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

        return cls(data=data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        """Create configuration from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            A new ``EngineConfig`` instance.
        """
        return cls(data=dict(data))

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by dot-separated key.

        Args:
            key: Dot-separated key (e.g. ``"training.batch_size"``).
            default: Value returned when the key is missing.

        Returns:
            The configuration value, or ``default``.
        """
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access via dot-separated key.

        Args:
            key: Dot-separated key.

        Returns:
            The configuration value.

        Raises:
            KeyError: If the key is not found.
        """
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(f"Key not found: '{key}'")
        return value

    def __contains__(self, key: str) -> bool:
        """Check if a key exists."""
        return self.get(key, _MISSING) is not _MISSING

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(self, overrides: dict[str, Any] | EngineConfig) -> EngineConfig:
        """Merge overrides into this configuration (shallow merge).

        Args:
            overrides: Dictionary or ``EngineConfig`` with override values.

        Returns:
            ``self`` for chaining.
        """
        if isinstance(overrides, EngineConfig):
            overrides = overrides._data
        self._data.update(overrides)
        return self

    def merge_deep(self, overrides: dict[str, Any] | EngineConfig) -> EngineConfig:
        """Deep-merge overrides into this configuration.

        Nested dictionaries are merged recursively rather than replaced.

        Args:
            overrides: Dictionary or ``EngineConfig`` with override values.

        Returns:
            ``self`` for chaining.
        """
        if isinstance(overrides, EngineConfig):
            overrides = overrides._data
        self._data = _deep_merge(self._data, dict(overrides))
        return self

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the raw configuration dictionary.

        Returns:
            Deep copy of the internal data.
        """
        return _deep_copy(self._data)

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.dump(self._data, handle, default_flow_style=False)

    def __repr__(self) -> str:
        return f"EngineConfig({self._data!r})"


# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

class _MISSING_TYPE:
    """Sentinel for missing values."""
    _instance: _MISSING_TYPE | None = None

    def __new__(cls) -> _MISSING_TYPE:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MISSING_TYPE()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = {}
    all_keys = set(base) | set(override)
    for key in all_keys:
        if key in base and key in override:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                result[key] = _deep_merge(base[key], override[key])
            else:
                result[key] = override[key]
        elif key in base:
            result[key] = _deep_copy(base[key])
        else:
            result[key] = _deep_copy(override[key])
    return result


def _deep_copy(value: Any) -> Any:
    """Deep-copy a value (handles nested dicts)."""
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value