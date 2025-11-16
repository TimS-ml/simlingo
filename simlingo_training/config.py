"""Configuration module for SimLingo training pipeline.

This module defines all configuration dataclasses used throughout the SimLingo training system.
It uses Hydra for configuration management, enabling flexible and composable configuration of:
- Vision-language models (VLM encoders)
- Language models (LLMs with LoRA)
- Driving models
- Dataset configurations
- Data module settings
- Training hyperparameters

The configuration system supports:
- Multiple dataset types (driving, dreamer, QA, instruction evaluation)
- Flexible data augmentation options
- Multi-task learning configurations
- DeepSpeed and DDP training strategies
- Wandb logging integration

All dataclasses are registered with Hydra's ConfigStore for easy composition and overriding.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore

@dataclass
class VLMEncoderConfig:
    """Configuration for vision-language model encoder.

    This config controls the vision encoder component that processes camera images
    and extracts visual features for the driving model. Typically uses InternVL2
    architecture for joint vision-language understanding.

    Attributes:
        variant: HuggingFace model identifier for the VLM encoder
        embed_dim: Dimension of output embeddings from the encoder
        freeze: Whether to freeze encoder weights during training (for transfer learning)
        _target_: Hydra target path to the VLMEncoderModel class
    """
    variant: str = 'OpenGVLab/InternVL2-1B'
    embed_dim: int = 512
    freeze: bool = False

    _target_: str = "simlingo_training.models.encoder.vlm.VLMEncoderModel"


@dataclass
class LanguageModelConfig:
    """Configuration for language model with LoRA fine-tuning.

    This config controls the language model used for generating driving commentary,
    answering questions, and instruction following. Uses LoRA (Low-Rank Adaptation)
    for efficient fine-tuning of large language models.

    Attributes:
        variant: HuggingFace model identifier for the LLM
        lora: Whether to enable LoRA fine-tuning
        lora_alpha: LoRA scaling parameter (controls adaptation strength)
        lora_r: LoRA rank (lower rank = fewer parameters, faster training)
        lora_dropout: Dropout rate for LoRA layers (regularization)
        _target_: Hydra target path to the LLM class
    """
    variant: str = 'OpenGVLab/InternVL2-1B'
    lora: bool = True
    lora_alpha: int = 64
    lora_r: int = 32
    lora_dropout: float = 0.1

    _target_: str = "simlingo_training.models.language_model.llm.LLM"


@dataclass
class DrivingModelConfig:
    """Configuration for the main driving model.

    This is the top-level model config that combines vision and language models
    for autonomous driving with natural language capabilities. It includes both
    model architecture settings and optimization hyperparameters.

    Attributes:
        vision_model: Configuration for vision encoder (VLMEncoderConfig)
        language_model: Configuration for language model (LanguageModelConfig)
        lr: Learning rate for optimizer
        weight_decay: L2 regularization coefficient
        betas: Beta parameters for Adam optimizer (momentum terms)
        pct_start: Percentage of training for learning rate warmup
        speed_wps_mode: Format for speed waypoints ('2d' for x,y coordinates)
        predict_route_as_wps: Whether to predict route as waypoints (vs. command)
        _target_: Hydra target path to the DrivingModel class
    """
    vision_model: Any
    language_model: Any

    lr: float = 5e-2

    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = '2d'
    predict_route_as_wps: bool = True

    _target_: str = "simlingo_training.models.driving.DrivingModel"


@dataclass
class DatasetBaseConfig:
    """Base configuration for all dataset types.

    This config contains common settings shared across all dataset variants
    (driving, dreamer, QA, etc.). It controls data loading, preprocessing,
    augmentation, and temporal window settings.

    Attributes:
        data_path: Path pattern to raw data files (supports wildcards)
        bucket_path: Path to data bucketing information
        cut_bottom_quarter: Whether to crop bottom 25% of images (removes ego vehicle hood)
        use_1d_wps: Whether to use 1D waypoints (longitudinal only) vs. 2D
        use_commentary: Enable commentary generation task
        use_qa: Enable question-answering task
        qa_augmentation: Apply augmentation to QA samples
        commentary_augmentation: Apply augmentation to commentary samples
        use_old_towns: Include data from older CARLA town versions
        use_only_old_towns: Use exclusively old town data
        use_town13: Include Town13 data
        skip_first_n_frames: Skip initial frames (unstable vehicle dynamics)
        pred_len: Prediction horizon length (timesteps into future)
        hist_len: History length for driving prediction (timesteps)
        hist_len_commentary: History length for commentary generation (timesteps)
        img_augmentation: Enable image augmentation (color jitter, etc.)
        img_augmentation_prob: Probability of applying image augmentation
        img_shift_augmentation: Enable spatial shift augmentation
        img_shift_augmentation_prob: Probability of applying shift augmentation
        use_safety_flag: Include safety flags for dreamer evaluation
        num_route_points: Number of route waypoints to provide as input
        route_as: Route representation format (target_point_command/target_point/command)
        use_lmdrive_commands: Use LMDrive-style navigation commands
    """
    data_path: str = "/home/katrinrenz/coding/wayve_carla/database/expertv3_2*"
    bucket_path: str = "data/buckets"

    cut_bottom_quarter: bool = False
    use_1d_wps: bool = False

    use_commentary: bool = False
    use_qa: bool = False
    qa_augmentation: bool = True
    commentary_augmentation: bool = True
    use_old_towns: bool = False
    use_only_old_towns: bool = False
    use_town13: bool = False

    skip_first_n_frames: int = 10
    pred_len: int = 11  # including the current time step
    hist_len: int = 1  # including the current time step
    hist_len_commentary: int = 5  # including the current time step

    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5

    use_safety_flag: bool = False

    num_route_points: int = 20

    route_as: str = 'target_point_command'  # target_point_command, target_point, command
    use_lmdrive_commands: bool = True

@dataclass
class DrivingDatasetConfig:
    """Configuration for driving trajectory prediction dataset.

    Points to the Data_Driving class which loads driving data for waypoint prediction.
    """
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_driving.Data_Driving"

@dataclass
class DreamerDatasetConfig:
    """Configuration for dreamer instruction following dataset.

    Points to the Data_Dreamer class for instruction-conditioned driving.
    """
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_dreamer.Data_Dreamer"

@dataclass
class QADatasetConfig:
    """Configuration for QA and commentary evaluation dataset.

    Points to the Data_Eval class for evaluating language understanding.
    """
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_eval_qa_comm.Data_Eval"

@dataclass
class InstEvalDatasetConfig:
    """Configuration for instruction evaluation dataset.

    Points to the Eval_Dreamer class for evaluating instruction following.
    """
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_eval_dreamer.Eval_Dreamer"

@dataclass
class DrivingDataModuleConfig:
    """Configuration for PyTorch Lightning data module.

    This config orchestrates multiple datasets for multi-task learning. It manages
    data loading, batching, and sampling across different tasks (driving, dreamer,
    QA, instruction evaluation).

    Attributes:
        base_dataset: Shared base configuration for all datasets
        driving_dataset: Config for driving waypoint prediction task
        dreamer_dataset: Config for instruction following task
        qa_dataset: Config for QA/commentary evaluation
        insteval_dataset: Config for instruction evaluation
        batch_size: Batch size for training/evaluation
        num_workers: Number of parallel data loading workers
        train_partitions: Dict mapping dataset names to sampling probabilities
        train_partitions_dreamer: Separate partitions for dreamer dataset
        use_global_img: Whether to use global (map) images in addition to camera
        _target_: Hydra target path to the DataModule class
    """

    base_dataset: DatasetBaseConfig

    driving_dataset:Optional[ DrivingDatasetConfig] = field(default_factory=DrivingDatasetConfig)
    dreamer_dataset: Optional[DreamerDatasetConfig] = field(default_factory=DreamerDatasetConfig)
    qa_dataset: Optional[QADatasetConfig] = field(default_factory=QADatasetConfig)
    insteval_dataset: Optional[InstEvalDatasetConfig] = field(default_factory=InstEvalDatasetConfig)

    batch_size: int = 16
    num_workers: int = 10

    train_partitions: Optional[Dict[str, float]] = None
    train_partitions_dreamer: Optional[Dict[str, float]] = None
    use_global_img: bool = False

    _target_: str = "simlingo_training.dataloader.datamodule.DataModule"


@dataclass
class TrainConfig:
    """Top-level training configuration.

    This is the main config that brings together all components for training.
    It controls hardware settings, training loop behavior, logging, and checkpointing.

    Attributes:
        model: Configuration for the driving model
        data_module: Configuration for data loading
        seed: Random seed for reproducibility
        gpus: Number of GPUs to use for training
        resume: Whether to resume from checkpoint
        resume_path: Path to checkpoint for resuming
        debug: Enable debug mode (reduces GPUs, disables wandb upload)
        overfit: Number of batches to overfit on (0 = normal training)
        fp16_loss_scale: Loss scaling for mixed precision (DeepSpeed)
        enable_wandb: Whether to enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_name: W&B run name (auto-generated from timestamp)
        name: Experiment name
        max_epochs: Maximum number of training epochs
        precision: Training precision (16-mixed for mixed precision)
        strategy: Distributed training strategy (deepspeed_stage_2 or ddp)
        val_every_n_epochs: Validation frequency in epochs
        checkpoint: Path to pretrained checkpoint to load
    """
    model: DrivingModelConfig
    data_module: Any

    seed: int = 42
    gpus: int = 8

    resume: bool = False
    resume_path: Optional[str] = None

    debug: bool = False
    overfit: int = 0
    fp16_loss_scale: float = 32.0  # 0.0 means dynamic loss scaling, only used with deepspeed

    enable_wandb: bool = True
    wandb_project: Optional[str] = "simlingo"
    if debug:
        wandb_name: Optional[str] = f"debug"
        gpus: int = 1
    else:
        # wandb_name: Optional[str] = f"debug"
        name: Optional[str] = 'test'
        wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"

    # max_steps: int = 100_000
    max_epochs: int = 20
    precision: str = "16-mixed"
    strategy: str = "deepspeed_stage_2"  # deepspeed_stage_2 ddp
    # val_check_interval: int = 5000
    val_every_n_epochs: int = 1

    checkpoint: Optional[str] = None


def register_configs():
    """Register all configuration dataclasses with Hydra's ConfigStore.

    This function registers configuration nodes with Hydra, enabling:
    - Hierarchical configuration composition
    - Type checking at runtime
    - Command-line overrides with validation
    - Multi-run sweeps over configuration spaces

    The configs are organized in groups for modular composition:
    - train_base: Top-level training config
    - data_module: Data loading configurations
    - model: Model architecture configurations
    """
    cs = ConfigStore.instance()
    cs.store(name="train_base", node=TrainConfig)
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="data_module/base_dataset", name="dataset", node=DatasetBaseConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model/vision_model", name="vlm", node=VLMEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)


# Register all configs on module import
register_configs()
