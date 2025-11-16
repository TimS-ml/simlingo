"""Main training script for SimLingo autonomous driving model.

This module implements the complete training pipeline for SimLingo, a vision-language
model for autonomous driving with natural language capabilities. It orchestrates:
- Multi-GPU distributed training with DeepSpeed or DDP
- Mixed precision training (FP16/BF16)
- Multi-task learning (driving, commentary, QA)
- Wandb experiment tracking
- Model checkpointing and resuming

The training loop is managed by PyTorch Lightning, which handles the distributed
training boilerplate, gradient accumulation, and device management.

Usage:
    python train.py [config overrides]
    python train.py model.lr=1e-4 data_module.batch_size=32

Key Features:
    - Hydra-based configuration system for easy experimentation
    - DeepSpeed ZeRO Stage 2 for memory-efficient multi-GPU training
    - Automatic checkpointing and wandb logging
    - Debug mode for rapid iteration
    - Support for overfitting on small data subsets
"""

import os
import hydra

from omegaconf import OmegaConf
import torch
import wandb

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from transformers import AutoProcessor

from simlingo_training.utils.logging_project import setup_logging, sync_wandb

from simlingo_training.config import TrainConfig
from simlingo_training.callbacks.visualise import VisualiseCallback


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    """Main training function orchestrating the complete pipeline.

    This function:
    1. Sets up environment and random seeds
    2. Initializes processor and model components
    3. Configures loggers and callbacks
    4. Creates PyTorch Lightning trainer
    5. Launches training loop

    Args:
        cfg: Hydra configuration object containing all training parameters

    The function handles:
    - Debug mode configuration (offline wandb, reduced GPUs)
    - Checkpoint loading from DeepSpeed or regular PyTorch formats
    - Resume from interrupted training runs
    - Distributed training strategy setup
    """
    # Enable TensorFloat-32 for faster training on Ampere GPUs (A100, etc.)
    torch.set_float32_matmul_precision("high")

    # Set random seeds for reproducibility across all workers
    pl.seed_everything(cfg.seed, workers=True)

    # Turn off wandb uploading when in debug mode (saves bandwidth and keeps logs local)
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"

    # Create unique wandb run name combining timestamp and experiment name
    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"

    # Load the vision-language processor (handles image preprocessing and tokenization)
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = None  # Could cache pretrained weights: f"pretrained/{(model_type_name)}"
    
    # Instantiate data module with all datasets (driving, dreamer, QA, etc.)
    # _recursive_=False prevents Hydra from instantiating nested configs
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False
    )

    # Instantiate the main driving model (vision encoder + language model)
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    # Load pretrained checkpoint if specified
    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            # DeepSpeed ZeRO checkpoint (distributed across multiple files)
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            # Regular PyTorch checkpoint (single file)
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

        
    # Print full configuration for debugging and reproducibility
    print(OmegaConf.to_yaml(cfg))

    # Disable wandb code tracking (keeps logs cleaner)
    os.environ["WANDB_DISABLE_CODE"] = "True"

    # Setup overfitting mode if requested (for debugging on small data)
    if cfg.overfit > 0:
        overfit = cfg.overfit

    # Setup logging directories and wandb initialization
    setup_logging(cfg)

    # Configure checkpoint resuming
    resume_path = cfg.resume_path
    resume_wandb = False

    # Determine if we should resume wandb logging
    # Resume if: path doesn't exist (new run) OR path exists and resume flag is set
    if resume_path is not None and not os.path.exists(resume_path):
        resume_wandb = True
    elif resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    # Only use resume_path if checkpoint actually exists and resume is enabled
    if resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None

    # Setup PyTorch Lightning loggers
    loggers = []

    # Wandb logger for experiment tracking (metrics, visualizations, hyperparameters)
    wandblogger = WandbLogger(
        project=cfg.wandb_project,
        id=cfg.wandb_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=resume_wandb,
    )
    # Watch model gradients and parameters
    wandblogger.watch(model)
    loggers.append(wandblogger)

    # Configure distributed training strategy
    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        # DeepSpeed ZeRO Stage 2: shards optimizer states and gradients across GPUs
        # This reduces memory usage significantly for large models
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2,
            loss_scale=cfg.fp16_loss_scale,  # FP16 loss scaling (0.0 = dynamic)
            logging_batch_size_per_gpu=cfg.data_module.batch_size
        )

    # Checkpoint callback: saves model weights periodically
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,  # Save all checkpoints (no pruning)
        monitor=None,  # Don't monitor any metric (just save at intervals)
        dirpath="./checkpoints",
        filename="{epoch:03d}",  # Checkpoint naming: epoch=001.ckpt
        save_last=True,  # Always save latest checkpoint as 'last.ckpt'
        every_n_epochs=cfg.val_every_n_epochs,
    )

    # Monitor and log learning rate changes
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Print model architecture summary (useful for debugging)
    model_summary = ModelSummary(max_depth=3)

    # Setup callbacks for training loop
    callbacks=[
        checkpoint_callback,
        model_summary,
        # ThroughputMonitor can track samples/sec if needed
        VisualiseCallback(interval=1000, val_interval=1000)  # Visualize predictions periodically
    ]
    if not cfg.debug:
        callbacks.append(lr_monitor)
    
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0  # Reset overfit (will use cfg.overfit if set)

    if cfg.gpus >= 1:
        # Create PyTorch Lightning Trainer with all training configurations
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,  # Enable cudnn benchmarking for faster training
            callbacks=callbacks,
            devices=cfg.gpus,  # Number of GPUs to use
            gradient_clip_val=0.3,  # Clip gradients to prevent exploding gradients
            logger=loggers,
            precision=cfg.precision,  # Mixed precision training (16-mixed or 32)
            strategy=strategy,  # DeepSpeed or DDP
            sync_batchnorm=True,  # Synchronize batch norm stats across GPUs
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,  # For debugging: overfit on N batches
            check_val_every_n_epoch=cfg.val_every_n_epochs,  # Validation frequency
        )

    # Start training loop (resumes from resume_path if provided)
    trainer.fit(model, data_module, ckpt_path=resume_path)

    # Clean up wandb
    wandb.finish()

if __name__ == "__main__":
    main()