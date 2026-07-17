"""High-level engine infrastructure for the litobzor research repository.

The engine connects common/data, common/training, and common/inference
into a unified API. Provides:

- Engine: high-level fit/validate/test/predict/save/load API
- Builder: factory for constructing all components from config
- Registry: lightweight registration for models, datasets, optimizers, etc.
- EngineConfig: YAML loading, nested access, defaults, merging
- EngineState: mutable state tracking (epoch, metrics, steps)
"""

from __future__ import annotations

from common.engine.builder import Builder
from common.engine.config import EngineConfig
from common.engine.engine import Engine
from common.engine.registry import (
    build_dataset,
    build_loss,
    build_metric,
    build_model,
    build_optimizer,
    build_scheduler,
    is_registered,
    list_registered,
    register_dataset,
    register_loss,
    register_metric,
    register_model,
    register_optimizer,
    register_scheduler,
)
from common.engine.state import EngineState

__all__ = [
    # Engine
    "Engine",
    # Builder
    "Builder",
    # Config
    "EngineConfig",
    # State
    "EngineState",
    # Registry
    "register_model",
    "register_dataset",
    "register_optimizer",
    "register_scheduler",
    "register_loss",
    "register_metric",
    "build_model",
    "build_dataset",
    "build_optimizer",
    "build_scheduler",
    "build_loss",
    "build_metric",
    "list_registered",
    "is_registered",
]