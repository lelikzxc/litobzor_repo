"""Semi-supervised training components for SemiWaferNet.

Modules:
- EMATeacher: exponential moving average teacher model
- PseudoLabelGenerator: confidence-thresholded pseudo-label generation
- ConsistencyLoss: consistency regularization between student and teacher
- MonteCarloDropout: uncertainty estimation via MC Dropout
- AdaptiveThreshold: per-class adaptive confidence thresholding
- UncertaintyFilter: uncertainty-based pseudo-label filtering
- StageManager: three-stage training schedule manager
- Trainer: high-level training orchestrator
"""

from __future__ import annotations

from papers.semiwafernet.training.ema import EMATeacher
from papers.semiwafernet.training.pseudo_label import PseudoLabelGenerator
from papers.semiwafernet.training.consistency import ConsistencyLoss
from papers.semiwafernet.training.mc_dropout import MonteCarloDropout
from papers.semiwafernet.training.adaptive_threshold import AdaptiveThreshold
from papers.semiwafernet.training.uncertainty import UncertaintyFilter
from papers.semiwafernet.training.stage_manager import StageManager
from papers.semiwafernet.training.trainer import Trainer

__all__ = [
    "EMATeacher",
    "PseudoLabelGenerator",
    "ConsistencyLoss",
    "MonteCarloDropout",
    "AdaptiveThreshold",
    "UncertaintyFilter",
    "StageManager",
    "Trainer",
]