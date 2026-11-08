"""A model-frame estimate and the observations that produced it."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    name: str
    category: int
    pose: np.ndarray
    center: np.ndarray
    size: np.ndarray
    confidence: float
    residual: float
    views: int
    orientation: str
    valid: bool
    clipped: bool
    points: int
    instances: tuple
