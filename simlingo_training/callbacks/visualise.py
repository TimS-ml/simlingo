"""Visualization callback for monitoring SimLingo training progress.

This module provides a PyTorch Lightning callback that generates visual artifacts
during training and validation to monitor model performance. It creates:

1. Waypoint Trajectory Plots:
   - Predicted waypoints (blue)
   - Ground truth waypoints (green)
   - Original/input waypoints (red, if available)

2. Route Planning Visualizations:
   - Full route predictions vs ground truth
   - Shows longer-term planning capabilities

3. Language Output Text:
   - Ground truth language (commentary/answers)
   - Model predictions
   - Side-by-side comparison

The visualizations are logged to wandb at configurable intervals, enabling:
- Quick debugging of model behavior
- Monitoring convergence across tasks
- Identifying mode collapse or failure cases
- Visual verification of multi-task learning

Key Features:
- Rank-zero only logging (avoids duplicate logs in distributed training)
- Automatic memory cleanup (clears model cache)
- Configurable logging intervals for train/val/predict
- Handles gradient accumulation correctly with once_per_step decorator
"""

import textwrap
from functools import wraps
from typing import Any, Callable, Dict

import matplotlib
matplotlib.use('agg')  # Use non-interactive backend for headless training
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from hydra.utils import get_original_cwd

from simlingo_training.utils.custom_types import DrivingExample

# Global dict to track first batch index per step (for gradient accumulation)
_STEPS_TO_FIRST_IDX: Dict[int, int] = {}


def get_1d_wps(wps):
    """Convert 2D waypoints to 1D cumulative distance representation.

    This function transforms waypoints from (x, y) coordinates to a 1D
    representation where the x-coordinate is cumulative distance traveled
    and y is always 0. Useful for longitudinal planning.

    Args:
        wps: Array of shape (N, 2) with waypoints as (x, y) coordinates

    Returns:
        Array of shape (N, 2) with (cumulative_distance, 0) format

    The conversion:
    1. Computes distances between consecutive waypoints
    2. Cumulative sum to get total distance traveled
    3. Prepends (0, 0) for starting position
    """
    # Compute Euclidean distances between consecutive waypoints
    waypoints_1d = [np.linalg.norm(wps[i+1] - wps[i]) for i in range(len(wps)-1)]

    # Cumulative sum to get distance from start
    waypoints_1d = np.cumsum(waypoints_1d)

    # Format as (distance, 0) coordinates
    waypoints_1d = [[x, 0] for x in waypoints_1d]

    # Prepend starting position (0, 0)
    waypoints_1d = [[0, 0]] + waypoints_1d

    return np.array(waypoints_1d).reshape(-1, 2)


def once_per_step(function: Callable[[Callback, Trainer, LightningModule, Any, Any, int], None]) -> Callable:
    """Decorator to ensure callback only runs once per optimization step.

    PyTorch Lightning calls `on_train_batch_end` multiple times per step when
    using gradient accumulation. This decorator ensures the wrapped function
    only executes once per actual optimizer step, not per forward pass.

    Args:
        function: Callback function to wrap (must be on_train_batch_end signature)

    Returns:
        Wrapped function that only executes on first batch of each step

    Technical Details:
    - Tracks first batch_idx seen for each global_step
    - Only executes function when current batch_idx matches first
    - Uses global dict (not thread-safe, but callbacks are sequential)
    - trainer.global_step is already incremented when on_train_batch_end is called

    Note:
        This is a workaround because on_before_optimizer_step runs before
        the step completes, which can cause timing/logging issues.
    """
    @wraps(function)
    def only_on_first_go(
        self: Callback, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        # Validate this is being used correctly
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

        # Track first batch_idx for this global_step
        global_step = trainer.global_step
        if global_step not in _STEPS_TO_FIRST_IDX:
            _STEPS_TO_FIRST_IDX[global_step] = batch_idx

        # Only execute if this is the first batch of the step
        if _STEPS_TO_FIRST_IDX[global_step] == batch_idx:
            return function(self, trainer, pl_module, outputs, batch, batch_idx)
        return None

    return only_on_first_go


class VisualiseCallback(Callback):
    """PyTorch Lightning callback for visualizing driving predictions.

    This callback periodically generates and logs visualizations of the model's
    predictions during training and validation. It creates side-by-side comparisons
    of predicted vs ground truth waypoints/routes and language outputs.

    Attributes:
        interval: How often to visualize during training (in steps)
        val_interval: How often to visualize during validation (in steps)
        predict_interval: How often to visualize during prediction (in batches)

    The callback:
    - Runs model inference with sampling (not teacher forcing)
    - Visualizes up to 16 examples per batch
    - Logs to wandb (or other Lightning logger)
    - Clears model cache to prevent OOM errors
    - Only runs on rank 0 in distributed training
    """
    def __init__(self, interval: int, val_interval: int = 10, predict_interval: int = 1):
        """Initialize visualization callback.

        Args:
            interval: Training visualization interval (steps)
            val_interval: Validation visualization interval (steps)
            predict_interval: Prediction visualization interval (batches)
        """
        super().__init__()
        self.interval = interval
        self.val_interval = val_interval
        self.predict_interval = predict_interval

    @torch.no_grad
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: DrivingExample,
        batch_idx: int,
    ):
        """Visualize predictions at end of validation batch.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The SimLingo model being trained
            outputs: Model outputs (unused)
            batch: Input batch containing images, labels, etc.
            batch_idx: Index of current batch

        This method:
        1. Checks if visualization should run (based on interval)
        2. Runs model forward pass with autocast
        3. Generates waypoint and route visualizations
        4. Logs to wandb
        5. Clears model cache to free memory
        """
        # Only visualize at specified intervals
        if trainer.global_step % self.val_interval != 0:
            return

        print("Validation visualization!")

        # Run forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=True):
            # Forward with sampling (not teacher forcing)
            # Returns: speed_waypoints, route, language_predictions
            speed_wps, route, language = pl_module.forward(batch, return_language=True)

        try:
            # Create waypoint visualization (predicted vs ground truth)
            self._visualise_training_examples(batch, speed_wps, trainer, pl_module, 'val_waypoints')

            # Create route visualization (longer-term planning)
            self._visualise_training_examples(batch, route, trainer, pl_module, 'val_route')

            print("visualised_val_example")
        except Exception as e:
            # Catch visualization errors to prevent training crashes
            print("visualise_training_examples", e)

        # Clear language model KV cache to prevent OOM
        if hasattr(pl_module, "clear_cache"):
            print("clearing_cache")
            pl_module.clear_cache()

    @once_per_step  # Only run once per optimization step (handles grad accumulation)
    @torch.no_grad
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: DrivingExample,
        batch_idx: int,
    ):
        """Visualize predictions at end of training batch.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The SimLingo model being trained
            outputs: Model outputs (unused)
            batch: Input batch containing images, labels, etc.
            batch_idx: Index of current batch

        This is similar to validation visualization but:
        - Uses @once_per_step to handle gradient accumulation
        - Includes language predictions in visualization
        - Only runs at training interval (typically less frequent)
        """
        # Only visualize at specified intervals
        if trainer.global_step % self.interval != 0:
            return

        # Run forward pass with mixed precision and sampling
        with torch.cuda.amp.autocast(enabled=True):
            speed_wps, route, language = pl_module.forward(batch, return_language=True)

        # Visualize waypoints with language predictions
        self._visualise_training_examples(
            batch, speed_wps, trainer, pl_module, 'waypoints', language_pred=language
        )

        # Visualize routes with language predictions
        self._visualise_training_examples(
            batch, route, trainer, pl_module, 'route', language_pred=language
        )

        print("visualised_training_example")

        # Clear language model cache
        if hasattr(pl_module, "clear_cache"):
            print("clearing_cache")
            pl_module.clear_cache()

    @rank_zero_only  # Only run on rank 0 process (avoid duplicate logs in DDP)
    def _visualise_training_examples(
        self,
        batch: DrivingExample,
        waypoints,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        name: str,
        language_pred = None,
    ):
        """Create and log visualizations to wandb.

        Args:
            batch: Input batch with ground truth labels
            waypoints: Predicted waypoints/routes (tensor)
            trainer: PyTorch Lightning trainer (for global_step)
            pl_module: Model (for logger access)
            name: Name prefix for logged images ('waypoints' or 'route')
            language_pred: Optional language predictions to display

        This method:
        1. Generates matplotlib plots of trajectories
        2. Creates text visualization of language outputs
        3. Logs both as images to wandb
        4. Cleans up matplotlib figures to prevent memory leaks
        """
        # Skip if no logger configured
        if not pl_module.logger:
            return

        # Generate appropriate visualization based on name
        if 'waypoints' in name:
            # Short-term trajectory prediction (11 points)
            waypoint_vis, prompt_img = visualise_waypoints(
                batch, waypoints, language_pred=language_pred
            )
        elif 'route' in name:
            # Long-term route planning (20 points)
            waypoint_vis, prompt_img = visualise_waypoints(
                batch, waypoints, language_pred=language_pred, route=True
            )

        # Log both trajectory plot and text visualization to wandb
        pl_module.logger.log_image(
            f"visualise/{name}",
            images=[Image.fromarray(waypoint_vis), prompt_img],
            step=trainer.global_step
        )

        # Clean up matplotlib to prevent memory leaks
        plt.close("all")


def fig_to_np(fig):
    """Convert matplotlib figure to numpy array for logging.

    Args:
        fig: Matplotlib figure object

    Returns:
        Numpy array of shape (H, W, 3) with RGB image data

    This enables logging matplotlib plots as images to wandb or other loggers.
    """
    # Apply tight layout for better spacing
    fig.tight_layout()

    # Render figure to RGB buffer
    fig.canvas.draw()

    # Convert buffer to numpy array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


@torch.no_grad()
def visualise_waypoints(batch: DrivingExample, waypoints, route=False, language_pred=None):
    """Create comprehensive visualization of waypoint predictions.

    This function generates a multi-panel plot showing predicted vs ground truth
    waypoints/routes, along with language outputs (if available).

    Args:
        batch: Input batch containing ground truth labels
        waypoints: Predicted waypoints tensor of shape (B, N, 2)
        route: If True, visualize route (20 points) instead of waypoints (11 points)
        language_pred: Optional list of predicted language strings

    Returns:
        Tuple of (trajectory_plot_array, text_image):
            - trajectory_plot_array: RGB numpy array of matplotlib plot
            - text_image: PIL Image with ground truth and predicted text

    The visualization includes:
    - Blue circles/lines: Predicted waypoints
    - Green crosses/lines: Ground truth waypoints
    - Red circles/lines: Original/input waypoints (if available)
    - Text panel: Ground truth and predicted language side-by-side

    Up to 16 examples are visualized in a 4-column grid.
    """
    assert batch.driving_label is not None
    repo_root = get_original_cwd()

    # Number of waypoints to visualize
    n = 11  # Short-term waypoints
    if route:
        n = 20  # Long-term route

    # Create matplotlib figure
    fig = plt.figure(figsize=(10.24, 10.24))

    # Extract ground truth waypoints
    if route:
        gt_waypoints = batch.driving_label.path[:, :n, :].cpu().numpy()
    else:
        gt_waypoints = batch.driving_label.waypoints[:, :n].cpu().numpy()

    # Extract original waypoints from input prompts (if available)
    # These are placeholder values in the language model input
    org_wps = []
    if len(batch.driving_input.prompt.placeholder_values) > 0:
        for b in batch.driving_input.prompt.placeholder_values:
            if 32007 in b:  # Waypoint token ID
                org_wps.append(np.array(b[32007]))
            elif 32005 in b:  # Alternative waypoint token ID
                org_wps.append(np.array(b[32005]))

    # Get predicted waypoints
    pred_waypoints = waypoints[:, :n].cpu().numpy()

    # Limit to max 16 examples for visualization
    b = gt_waypoints.shape[0]
    b = min(b, 16)

    # Compute grid layout (4 columns max)
    rows = int(np.ceil(b / 4))
    cols = min(b, 4)

    # Create white image for text visualization
    white_pil = Image.new("RGB", (1024, 1024), "white")
    white_draw = ImageDraw.Draw(white_pil)

    # Add vertical space between subplots for text
    fig.subplots_adjust(hspace=0.8)

    # Draw language outputs (GT and predictions)
    y_curr = 10
    for i in range(b):
        # Draw language outputs if available
        if batch.driving_label.answer is not None:
            language = batch.driving_label.answer.language_string[i]

            # Wrap text to fit in image
            wrapped_text = textwrap.fill(language, width=80)
            wrapped_pred_text = textwrap.fill(language_pred[i], width=80) if language_pred is not None else ""

            # Count lines for vertical spacing
            lines_wrap = len(textwrap.wrap(wrapped_text, width=80))
            lines_wrap_pred = len(textwrap.wrap(wrapped_pred_text, width=80))

            # Draw ground truth text
            white_draw.text(
                (10, y_curr),
                f'{i} GT: {wrapped_text}',
                fill="black",
                font=ImageFont.truetype(f"{repo_root}/simlingo_training/arial.ttf", 20)
            )
            y_curr += 20 * lines_wrap

            # Draw predicted text
            white_draw.text(
                (10, y_curr),
                f'{i} Pred: {wrapped_pred_text}',
                fill="black",
                font=ImageFont.truetype(f"{repo_root}/simlingo_training/arial.ttf", 20)
            )
            y_curr += 20 * lines_wrap_pred + 20

        # Create subplot for this example
        ax = fig.add_subplot(rows, cols, i + 1)

        # Plot predicted waypoints (blue)
        ax.scatter(pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], marker="o", c="b", label="Predicted")
        ax.plot(pred_waypoints[i, :, 1], pred_waypoints[i, :, 0], c="b")

        # Plot ground truth waypoints (green)
        ax.scatter(gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], marker="x", c="g", label="Ground Truth")
        ax.plot(gt_waypoints[i, :, 1], gt_waypoints[i, :, 0], c="g")

        # Plot original/input waypoints if available (red)
        if len(org_wps) > 0:
            ax.scatter(org_wps[i][:, 1], org_wps[i][:, 0], marker="o", c="r", label="Input")
            ax.plot(org_wps[i][:, 1], org_wps[i][:, 0], c="r")

        # Format subplot
        ax.set_title(f"waypoints {i}")
        ax.grid()
        ax.set_aspect("equal", adjustable="box")  # Equal aspect ratio for realistic trajectory view
        ax.set_box_aspect(1.5)  # Tall plots for forward-looking trajectories

    return fig_to_np(fig), white_pil
