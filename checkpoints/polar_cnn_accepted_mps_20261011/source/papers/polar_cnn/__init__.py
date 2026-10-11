"""Polar-sector CNN reproduction, Electronics 2024, 13(7), 1360."""

from contextlib import suppress

from common.engine.registry import register_model

from .models import PolarCNN

with suppress(ValueError):
    register_model("polar_cnn", PolarCNN)

__all__ = ["PolarCNN"]
