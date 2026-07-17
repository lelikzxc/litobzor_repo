"""Common engine infrastructure for model training, evaluation, and inference.

Provides:
    - Registry: lightweight registration for models, datasets, optimizers, etc.
    - EngineConfig: YAML-based configuration with dot-separated nested access.
    - Builder: factory that constructs all components from a config.
    - Engine: high-level fit/validate/test/predict API.
"""

from common.engine.registry import (
    build_model,
    is_registered,
    list_registered,
    register_model,
)
from common.engine.config import EngineConfig
from common.engine.builder import Builder
from common.engine.engine import Engine

__all__ = [
    "Engine",
    "Builder",
    "EngineConfig",
    "register_model",
    "build_model",
    "is_registered",
    "list_registered",
]