"""Visualization callback for monitoring SimLingo-Base model training.

This module provides a PyTorch Lightning callback that periodically generates
visualizations of model predictions during training. It creates plots comparing
ground truth and predicted waypoints/routes.

Key Components:
    - VisualiseCallback: Main callback for periodic visualization
    - once_per_step: Decorator ensuring callback runs once per optimization step
    - visualise_waypoints: Function to create waypoint comparison plots
    - fig_to_np: Helper to convert matplotlib figures to numpy arrays

The visualizations help monitor training progress and identify potential issues
with waypoint or route prediction.
"""

from typing import Any, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from typing import Dict, Callable
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
from functools import wraps


from simlingo_base_training.utils.custom_types import DrivingExample
from simlingo_base_training.utils.projection import get_camera_intrinsics, project_points


# Global dictionary to track first batch index for each training step
_STEPS_TO_FIRST_IDX: Dict[int, int] = {}


def once_per_step(function: Callable[[Callback, Trainer, LightningModule, Any, Any, int], None]) -> Callable:
    """Decorator to ensure callback runs only once per optimization step.

    PyTorch Lightning calls `on_train_batch_end` multiple times per optimization step
    when using gradient accumulation. This decorator ensures the wrapped function
    executes only on the first call for each global step.

    Args:
        function: The callback function to wrap. Must be from a pl.Callback class
                  with signature (self, trainer, pl_module, outputs, batch, batch_idx).

    Returns:
        Wrapped function that executes only once per optimization step.

    Note:
        Uses a global dictionary to track which batch_idx was seen first for each step.
        Not thread-safe but callbacks are expected to run sequentially per process.
    """
    # Implementation note: `on_before_optimizer_step` exists but gets called before a step is finished,
    # potentially leading to unexpected behavior when measuring timing across steps.
    # The `_STEPS_TO_FIRST_IDX` global dict is not threadsafe, but `on_train_batch_end` is expected
    # to only be called sequentially per process.
    @wraps(function)
    def only_on_first_go(
        self: Callback, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not all(
            (
                isinstance(self, Callback),
                isinstance(trainer, Trainer),
                isinstance(pl_module, LightningModule),
                isinstance(batch_idx, int),
            )
        ):
            raise ValueError(
                "Only use this decorator on `pl.Callback`'s `on_train_batch_end` function!",
            )
        global_step = trainer.global_step
        if global_step not in _STEPS_TO_FIRST_IDX:
            _STEPS_TO_FIRST_IDX[global_step] = batch_idx

        if _STEPS_TO_FIRST_IDX[global_step] == batch_idx:
            return function(self, trainer, pl_module, outputs, batch, batch_idx)
        return None

    return only_on_first_go


class VisualiseCallback(Callback):
    """PyTorch Lightning callback for visualizing model predictions.

    Periodically generates and logs comparison plots of predicted vs ground truth
    waypoints and routes. Helps monitor training progress visually.

    Attributes:
        interval: Number of training steps between visualizations.
    """

    def __init__(self, interval: int):
        """Initialize the visualization callback.

        Args:
            interval: How often to generate visualizations (in training steps).
        """
        super().__init__()
        self.interval = interval

    @once_per_step
    @torch.no_grad
    def on_train_batch_end(  # pylint: disable=too-many-statements
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: DrivingExample,
        batch_idx: int,
    ):
        """Called at the end of each training batch.

        Generates visualizations at specified intervals and logs them to W&B or TensorBoard.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
            outputs: Outputs from the training step.
            batch: Current batch of training data.
            batch_idx: Index of the current batch.

        Returns:
            None

        Side Effects:
            - Logs waypoint and route visualizations to logger
            - Clears cache if model has clear_cache method
        """
        if trainer.global_step % self.interval != 0:
            return

        with torch.cuda.amp.autocast(enabled=True):
            # Forward with sampling
            speed_wps, route = pl_module.forward(batch.driving_input)

        try:
            self._visualise_training_examples(batch, speed_wps, trainer, pl_module, 'waypoints')
            self._visualise_training_examples(batch, route, trainer, pl_module, 'route')

            # visualise_cameras(batch, pl_module, trainer, route, speed_wps)
            
            print("visualised_training_example")
            # _LOGGER.info("visualised_training_example")
        except Exception as e:  # pylint: disable=broad-except
            print("visualise_training_examples", e)
            pass
            # _LOGGER.exception("visualise_training_examples", e)
        if hasattr(pl_module, "clear_cache"):
            print("clearing_cache")
            # Clear cache associated with decoding language model
            # _LOGGER.info("clearing_cache")
            pl_module.clear_cache()

    @rank_zero_only
    def _visualise_training_examples(
        self,
        batch: DrivingExample,
        waypoints,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        name: str,
    ):
        """Generate and log visualization of waypoints or routes (rank 0 only).

        Creates comparison plots between predicted and ground truth trajectories.
        Only executes on the main process in distributed training.

        Args:
            batch: Batch of driving examples with labels.
            waypoints: Predicted waypoints or routes from the model.
            trainer: PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
            name: Name for the visualization ('waypoints' or 'route').

        Returns:
            None

        Side Effects:
            Logs visualization images to the configured logger.
        """
        if not pl_module.logger:
            return

        # only visualise max 16 examples
        # if len(batch.run_id) > 16:
        if name == 'waypoints':
            waypoint_vis = visualise_waypoints(batch, waypoints)
        elif name == 'route':
            waypoint_vis = visualise_waypoints(batch, waypoints, route=True)
        # pl_module.logger.log_image("visualise/images", images=[Image.fromarray(si_vis)], step=trainer.global_step)
        pl_module.logger.log_image(
            f"visualise/{name}", images=[Image.fromarray(waypoint_vis)], step=trainer.global_step
        )

        plt.close("all")


def fig_to_np(fig):
    """Convert a matplotlib figure to a numpy array.

    Args:
        fig: Matplotlib figure object.

    Returns:
        numpy.ndarray: RGB image array of shape (height, width, 3) with dtype uint8.
    """
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

@torch.no_grad()
def visualise_waypoints(batch: DrivingExample, waypoints, route=False):
    """Create comparison plot of predicted vs ground truth waypoints/routes.

    Generates a grid of subplots (up to 16 examples) showing both predicted and
    ground truth trajectories in bird's-eye view.

    Args:
        batch: Batch of driving examples containing ground truth labels.
        waypoints: Predicted waypoints or routes from the model, shape [B, N, 2].
        route: If True, visualizes routes (20 points); if False, waypoints (11 points).

    Returns:
        numpy.ndarray: RGB image array containing the visualization grid.

    Note:
        - Predicted trajectories shown in blue
        - Ground truth trajectories shown in green
        - Coordinates are in vehicle's local frame (x: forward, y: left)
    """
    assert batch.driving_label is not None

    n = 11
    if route:
        n = 20
    fig = plt.figure(figsize=(10, 10))
    if route:
        gt_waypoints = batch.driving_label.route_adjusted[:, :n, :].cpu().numpy()
    else:
        gt_waypoints = batch.driving_label.poses[..., :n, :2, 3].cpu().numpy()
    pred_waypoints = waypoints[:, :n].cpu().numpy()
    b = gt_waypoints.shape[0]
    # visualise max 16 examples
    b = min(b, 16)
    rows = int(np.ceil(b / 4))
    cols = min(b, 4)


    # add space for text
    fig.subplots_adjust(hspace=0.8)
    for i in range(b):
        ax = fig.add_subplot(rows, cols, i + 1)
        # Predicted waypoints
        ax.scatter(-pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], marker="o", c="b")
        ax.plot(-pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], c="b")
        # Ground truth waypoints (i.e. ideal waypoints)
        ax.scatter(-gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], marker="x", c="g")
        ax.plot(-gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], c="g")
        ax.set_title(f"waypoints {i}")
        ax.grid()
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1.5)

    return fig_to_np(fig)
