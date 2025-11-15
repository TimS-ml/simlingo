"""PyTorch Lightning callbacks for SimLingo-Base training.

This package contains custom callback implementations for monitoring and
visualizing training progress.

Callbacks:
    VisualiseCallback: Generates and logs visualization of model predictions.
"""

from .visualise import VisualiseCallback

__all__ = [
    "VisualiseCallback",
]
