"""Configuration module for SimLingo-Base model training.

This module defines all configuration dataclasses for the SimLingo-Base training pipeline,
which is a vision-only, smaller autonomous driving model using LLaVA-Next or ResNet vision
encoders and small Llama language models.

Main Components:
    - LLaVAnextEncoderConfig: Configuration for LLaVA-Next vision encoder
    - ResnetEncoderConfig: Configuration for ResNet vision encoder
    - LanguageModelConfig: Configuration for Llama language model
    - DrivingModelConfig: Main model configuration combining vision and language
    - DrivingDataModuleConfig: Data loading and preprocessing configuration
    - TrainConfig: Overall training configuration

Differences from Full SimLingo:
    - Uses smaller language models (x-small, tiny variants)
    - Vision-only training (no language commentary)
    - Simplified prediction targets (waypoints and routes only)
    - Faster training with fewer parameters

Dependencies:
    - Hydra: For configuration management
    - Dataclasses: For structured configuration objects
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore

@dataclass
class LLaVAnextEncoderConfig:
    """Configuration for LLaVA-Next vision encoder.

    LLaVA-Next is a vision-language model that extracts visual features from images.
    This encoder processes camera images into token embeddings for the language model.

    Attributes:
        variant: HuggingFace model ID for the LLaVA-Next model.
        embed_dim: Dimension of output embeddings after projection.
        freeze: Whether to freeze encoder weights during training.
        downsample_feature_grid_factor: Downsampling factor for feature grid (reduces memory).
        use_global_img: Whether to use global image patch in addition to local patches.
        _target_: Hydra target path for instantiation.
    """
    variant: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    embed_dim: int = 512
    freeze: bool = False
    downsample_feature_grid_factor: Optional[int] = 2
    use_global_img: bool = False

    _target_: str = "simlingo_base_training.models.encoder.llavanext.LLaVAnextEncoderModel"

@dataclass
class ResnetEncoderConfig:
    """Configuration for ResNet vision encoder.

    ResNet is a convolutional neural network that extracts visual features from images.
    This is an alternative to LLaVA-Next, offering faster processing with simpler architecture.

    Attributes:
        variant: HuggingFace model ID for the ResNet model (e.g., 'microsoft/resnet-34').
        embed_dim: Dimension of output embeddings after projection.
        freeze: Whether to freeze encoder weights during training.
        downsample_feature_grid_factor: Downsampling factor for feature grid (reduces memory).
        use_global_img: Whether to use global image representation.
        _target_: Hydra target path for instantiation.
    """
    variant: str = 'microsoft/resnet-34'
    embed_dim: int = 512
    freeze: bool = False
    downsample_feature_grid_factor: Optional[int] = 2
    use_global_img: bool = True

    _target_: str = "simlingo_base_training.models.encoder.resnet.ResnetEncoderModel"


@dataclass
class LanguageModelConfig:
    """Configuration for Llama language model backbone.

    The language model processes combined vision and route embeddings to predict
    future waypoints. Uses smaller Llama variants for efficient training.

    Attributes:
        variant: Model size variant ('x-small', 'tiny', 'small', etc.).
        lora: Whether to use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
        _target_: Hydra target path for instantiation.
    """
    variant: str = "x-small"
    lora: bool = False
    _target_: str = "simlingo_base_training.models.language_model.llama.Llama"


@dataclass
class DrivingModelConfig:
    """Main configuration for the driving model.

    Combines vision encoder, language model, and training hyperparameters.
    The model predicts waypoints and optionally routes for autonomous driving.

    Attributes:
        vision_model: Configuration for vision encoder (LLaVA-Next or ResNet).
        language_model: Configuration for language model backbone.
        lr: Base learning rate for most parameters.
        vision_lr: Separate learning rate for vision encoder (if different from lr).
        weight_decay: L2 regularization weight.
        betas: Adam optimizer beta parameters.
        pct_start: Percentage of training for learning rate warmup.
        speed_wps_mode: Mode for speed/waypoint prediction ('1d' or '2d').
        predict_route_as_wps: Whether to predict route as waypoints.
        speed_as_input: Whether to include current speed as input.
        new_layer_norm_minmax: Whether to use updated normalization ranges.
        _target_: Hydra target path for instantiation.
    """
    vision_model: Any
    language_model: Any

    lr: float = 1e-4
    vision_lr: Optional[float] = 1e-4

    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = '2d'
    predict_route_as_wps: bool = False
    speed_as_input: bool = True
    new_layer_norm_minmax: bool = False

    _target_: str = "simlingo_base_training.models.driving.DrivingModel"


@dataclass
class DrivingDataModuleConfig:
    """Configuration for data loading and preprocessing.

    Controls how CARLA simulation data is loaded, processed, and augmented for training.
    Includes settings for temporal sequences, augmentation, and data partitioning.

    Attributes:
        batch_size: Number of examples per batch.
        num_workers: Number of worker processes for data loading.
        data_path: Path to main dataset directory.
        bucket_path: Path to data buckets (filtered subsets by driving scenario).
        encoder: Type of vision encoder to use ('llavanext' or 'resnet').
        train_partitions: Optional dict mapping bucket names to sampling weights.
        cut_bottom_quarter: Whether to crop bottom of images (removes vehicle hood).
        use_global_img: Whether to include global image patch.
        skip_first_n_frames: Number of initial frames to skip per route.
        pred_len: Number of future waypoints to predict (including current).
        hist_len: Number of historical frames to use as input (including current).
        image_enhancing: Whether to apply histogram equalization.
        img_augmentation: Whether to apply image augmentations (blur, noise, etc.).
        img_augmentation_prob: Probability of applying image augmentation.
        img_shift_augmentation: Whether to use camera shift augmentation.
        img_shift_augmentation_prob: Probability of using shifted camera view.
        num_route_points: Number of route points to include.
        use_town13: Whether to use Town13 data for training.
        use_old_towns: Whether to use older town maps.
        route_as: How to represent route ('coords', 'image', 'target_point').
        _target_: Hydra target path for instantiation.
    """
    batch_size: int = 16
    num_workers: int = 10
    data_path: str = "database/simlingo"
    bucket_path: str = "database/bucketsv2_simlingo"
    encoder: str = "llavanext"  # "resnet"
    train_partitions: Optional[Dict[str, float]] = None
    cut_bottom_quarter: bool = False

    use_global_img: bool = False

    skip_first_n_frames: int = 10
    pred_len: int = 11  # including the current time step
    hist_len: int = 3  # including the current time step

    image_enhancing: bool = False
    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5  # 80% of the data that contains augmented views -> only a small portion of the dataset contains augmented views

    num_route_points: int = 20

    use_town13: bool = True
    use_old_towns: bool = True

    route_as: str = 'target_point'  # coords, image, target_point
    _target_: str = "simlingo_base_training.dataloader.datamodule.DataModule"


@dataclass
class TrainConfig:
    """Main training configuration.

    Top-level configuration that combines model, data, and training settings.
    Controls the overall training loop, logging, and checkpointing behavior.

    Attributes:
        model: DrivingModelConfig for model architecture.
        data_module: DataModule configuration for data loading.
        seed: Random seed for reproducibility.
        gpus: Number of GPUs to use.
        resume: Whether to resume from checkpoint.
        resume_path: Path to checkpoint for resuming.
        debug: Debug mode (offline logging, single GPU).
        overfit: Number of batches to overfit on (0 = normal training).
        submit: Whether to save checkpoints during training.
        fp16_loss_scale: Loss scaling for mixed precision (0.0 = dynamic).
        enable_wandb: Whether to enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_name: W&B run name (auto-generated with timestamp).
        name: Experiment name.
        max_epochs: Maximum number of training epochs.
        precision: Training precision ('16-mixed' for mixed precision).
        strategy: Distributed strategy ('deepspeed_stage_2' or 'ddp').
        accumulate_grad_batches: Number of batches for gradient accumulation.
        devices: Device specification ('auto' or specific count).
        val_every_n_epochs: Validation frequency in epochs.
        checkpoint: Path to checkpoint to load weights from.
        weights: Path to weights to load (without optimizer state).
    """
    model: DrivingModelConfig
    data_module: Any

    seed: int = 42
    gpus: int = 8

    resume: bool = False
    resume_path: Optional[str] = None


    debug: bool = False
    overfit: int = 0
    submit: bool = True  # whether to checkpoint and submit the model during training
    fp16_loss_scale: float = 32.0 # 0.0 means dynamic loss scaling, only used with deepspeed

    enable_wandb: bool = True
    wandb_project: Optional[str] = "simlingo_base"
    if debug:
        wandb_name: Optional[str] = f"debug"
        gpus: int = 1
    else:
        # wandb_name: Optional[str] = f"debug"
        name: Optional[str] = "test"
        wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    

    # max_steps: int = 100_000
    max_epochs: int = 20
    precision: str = "16-mixed"
    strategy: str = "deepspeed_stage_2" # deepspeed_stage_2 ddp
    accumulate_grad_batches: int = 1
    devices: Union[str, int] = "auto"
    # val_check_interval: int = 5000
    val_every_n_epochs: int = 1

    checkpoint: Optional[str] = None
    weights: Optional[str] = None  # same as checkpoint, except we don't load optimizer


def register_configs():
    """Register all configuration schemas with Hydra's ConfigStore.

    This function makes all configuration dataclasses available to Hydra for
    composition and validation. Called at module import time to set up the
    configuration system.

    The configurations are organized hierarchically:
    - train_base: Root training configuration
    - data_module/driving: Data loading configuration
    - model/driving: Model architecture configuration
    - model/vision_model/{llavanext,resnet}: Vision encoder options
    - model/language_model/llm: Language model configuration

    Returns:
        None
    """
    cs = ConfigStore.instance()
    cs.store(name="train_base", node=TrainConfig)
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model/vision_model", name="llavanext", node=LLaVAnextEncoderConfig)
    cs.store(group="model/vision_model", name="resnet", node=ResnetEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)


register_configs()
