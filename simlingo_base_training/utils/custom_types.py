"""Custom type definitions for SimLingo-Base training.

This module defines NamedTuple types used throughout the codebase for
type-safe data structures and clear interfaces between components.

Key Types:
    - DrivingInput: Model input containing images, speed, and route
    - DrivingLabel: Ground truth labels for training
    - DrivingExample: Complete training example with input and label
    - DrivingOutput: Model predictions
    - TrainingOutput: Training metrics and losses
    - ParamGroup: Optimizer parameter group specification

These types enforce structure and improve code readability by making
data flow explicit.
"""

from typing import Dict, List, NamedTuple, Optional
import torch
from torch import Tensor

class DrivingOutput(NamedTuple):
    """Model output predictions.

    Attributes:
        time_delta_sec: Time deltas for predicted waypoints, shape [B, F].
        waypoints: Predicted waypoints in vehicle frame, shape [B, F, 2].
        language_tokens: Language generation tokens (unused in base model).
        trajectory_tokens: Trajectory tokens (unused in base model).
    """
    time_delta_sec: Tensor  # [B, F] float32
    waypoints: Tensor  # [B, F, 2] float32
    # Auxiliary outputs (MUST be at the end):
    language_tokens: Tensor  # [B, max(len(tokens))]
    trajectory_tokens: Tensor  # [B, F, max(len(tokens))]


class TrainingOutput(NamedTuple):
    """Training step output with loss information.

    Attributes:
        loss: Total combined loss.
        loss_averages: Per-loss-type averages.
        loss_values: Per-sample loss values.
        loss_counts: Per-sample loss counts.
        driving_output: Optional model predictions.
    """
    loss: Tensor  # [] floating

    loss_averages: Dict[str, Tensor]  # [] floating
    loss_values: Dict[str, Tensor]  # [B] floating
    loss_counts: Dict[str, Tensor]  # [B] int64

    driving_output: Optional[DrivingOutput] = None

class ParamGroup(NamedTuple):
    """Optimizer parameter group specification.

    Attributes:
        pattern: Regex pattern matching parameter names.
        lr: Learning rate for this group.
        weight_decay: L2 regularization weight.
    """
    pattern: str
    lr: float
    weight_decay: float

class DrivingInput(NamedTuple):
    """Model input data.

    Attributes:
        camera_images: Camera images, shape [B, T, N, C, H, W].
        image_sizes: Image sizes for LLaVA-Next processing.
        camera_intrinsics: Camera calibration matrices, shape [B, N, 3, 3].
        camera_extrinsics: Camera pose matrices, shape [B, N, 4, 4].
        vehicle_speed: Current speed in m/s, shape [B, 1].
        map_route: Navigation route representation.
        target_point: Target navigation point, shape [B, 2].
    """
    camera_images: torch.Tensor  # [B, T, N, C, H, W] uint8 [0, 255]
    image_sizes: torch.Tensor
    camera_intrinsics: torch.Tensor  # [B, N, 3, 3] float32
    camera_extrinsics: torch.Tensor  # [B, N, 4, 4] float32
    vehicle_speed: torch.Tensor  # [B, S] float32 m/s
    map_route: torch.Tensor  # [B, 3, RH, RW] uint8 [0, 255]
    target_point: torch.Tensor  # [B, 2] float32

class DrivingLabel(NamedTuple):
    """Ground truth labels for training.

    Attributes:
        time_delta_sec: Time deltas for waypoints, shape [B, F].
        waypoints: Future waypoints in 2D, shape [B, F, 2].
        waypoints_1d: Future waypoints as distances, shape [B, F, 2].
        route_adjusted: Adjusted route points for prediction.
    """
    time_delta_sec: Tensor  # [B, F] 0-2 sec
    waypoints: Tensor  # [B, F, 2] 11 future waypoints 0.2s apart
    waypoints_1d: Tensor  # [B, F, 2] 11 future waypoints 0.2s apart
    route_adjusted: Tensor

class DrivingExample(NamedTuple):
    """Complete training example.

    Attributes:
        driving_input: Model input data.
        driving_label: Ground truth labels.
        run_id: Unique identifier for this sample.
        timestamp: Unix timestamp of capture.
    """
    driving_input: DrivingInput
    driving_label: DrivingLabel
    run_id: List[str]
    timestamp: Tensor  # unix timestamp of ff cam