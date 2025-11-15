"""Callbacks package for training visualization and monitoring.

This package contains PyTorch Lightning callbacks used during SimLingo training
to visualize predictions, monitor training progress, and log artifacts to wandb.

The main callback is VisualiseCallback which periodically generates and logs:
- Waypoint trajectory predictions vs ground truth
- Route planning visualizations
- Language generation outputs (commentary/QA)

These visualizations help monitor model convergence and identify failure modes
during training without manual intervention.
"""

from .visualise import VisualiseCallback

__all__ = [
    "VisualiseCallback",
]
