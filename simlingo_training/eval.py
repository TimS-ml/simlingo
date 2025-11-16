"""Evaluation script for SimLingo model on different tasks.

This module implements the evaluation pipeline for SimLingo, supporting three
evaluation modes:
1. QA (Question Answering): Evaluate the model's ability to answer questions about driving scenarios
2. Commentary: Evaluate natural language commentary generation
3. Dreaming: Evaluate instruction following for safety-critical scenarios

The evaluation uses PyTorch Lightning's predict mode to generate predictions,
which are then saved to JSON files for offline metric computation.

Usage:
    python eval.py [config overrides]

The script automatically:
- Loads checkpoints from training runs
- Disables data augmentation for fair evaluation
- Configures appropriate datasets based on eval_mode
- Saves predictions for later analysis
"""

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoProcessor, AutoTokenizer

from simlingo_training.config import TrainConfig
from simlingo_training.utils.logging_project import setup_logging
# from simlingo_training.callbacks.visualise import VisualiseCallback

@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    """Main evaluation function for SimLingo model.

    This function:
    1. Configures evaluation mode (QA, commentary, or dreaming)
    2. Loads model checkpoint
    3. Disables augmentation for fair evaluation
    4. Runs prediction on test data
    5. Saves results to JSON for metric computation

    Args:
        cfg: Hydra configuration object (can be overridden from command line)

    The function modifies config based on eval_mode:
    - Disables unused datasets to save memory
    - Enables appropriate task flags (use_qa, use_commentary, use_safety_flag)
    - Disables augmentation for deterministic evaluation
    """
    
    # Enable TensorFloat-32 for faster inference on Ampere GPUs
    torch.set_float32_matmul_precision("high")

    # Set seed for reproducible evaluation
    pl.seed_everything(42)

    # Evaluation mode selection (uncomment the desired mode)
    # eval_mode = "QA"  # Question answering evaluation
    # eval_mode = "commentary"  # Commentary generation evaluation
    eval_mode = "Dreaming"  # Instruction following (safety) evaluation

    # Save original dataset configs before loading checkpoint config
    qa_dataset = cfg.data_module.qa_dataset
    insteval_dataset = cfg.data_module.insteval_dataset

    # Path to trained checkpoint
    load_path = '/YOUR_PATH/outputs/simlingo/checkpoints/epoch=013.ckpt'
    if load_path is not None:
        # Load config from training run (ensures model architecture matches)
        load_path_config = Path(load_path).parent.parent / '.hydra/config.yaml'
        cfg = OmegaConf.load(load_path_config)
    
    # Restore dataset configs
    cfg.data_module.qa_dataset = qa_dataset
    cfg.data_module.insteval_dataset = insteval_dataset

    # Override config for evaluation (reduce resources)
    cfg.gpus = 1  # Evaluation typically uses single GPU
    cfg.data_module.num_workers = 8
    cfg.data_module.batch_size = 64  # Can use larger batch for inference

    print(f'Eval mode: {eval_mode}')
    print(f'Checkpoint: {load_path}')
    print(f"Using {cfg.gpus} GPUs")

    # Configure which datasets to use based on eval mode
    if eval_mode == "QA" or eval_mode == "commentary":
        # For language tasks, disable driving datasets
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.insteval_dataset = None
    elif eval_mode == "Dreaming":
        # For instruction following, disable language datasets
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None

    # Configure task-specific flags
    if eval_mode == "QA":
        cfg.data_module.base_dataset.use_commentary = False
        cfg.data_module.base_dataset.use_qa = True
    elif eval_mode == "commentary":
        cfg.data_module.base_dataset.use_commentary = True
        cfg.data_module.base_dataset.use_qa = False
    elif eval_mode == "Dreaming":
        cfg.data_module.base_dataset.use_safety_flag = True

    # Disable image augmentation for deterministic evaluation
    cfg.data_module.base_dataset.img_augmentation = False

    # Disable spatial shift augmentation
    cfg.data_module.base_dataset.img_shift_augmentation = False
    
    # Load appropriate processor based on model size
    # 2B models use different tokenizer than smaller variants
    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(
            cfg.model.language_model.variant,
            trust_remote_code=True,
            use_fast=False
        )
    else:
        processor = AutoProcessor.from_pretrained(
            cfg.model.language_model.variant,
            trust_remote_code=True,
            use_fast=False
        )

    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = f"pretrained/{(model_type_name)}"

    # Instantiate data module in prediction mode
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=True,  # Enable prediction mode (no labels required)
        _recursive_=False
    )

    # Instantiate model with same architecture as training
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    # Load checkpoint weights
    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            # DeepSpeed distributed checkpoint
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            # Standard PyTorch checkpoint
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

        
    # Print full configuration for debugging
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"

    # Setup logging directories
    setup_logging(cfg)

    # Configure checkpoint path for resuming (if needed)
    resume_path = "./checkpoints/last.ckpt"

    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None

    # No loggers needed for evaluation (predictions saved to file)
    loggers = []

    # Configure distributed strategy (same as training for consistency)
    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2,
            loss_scale=cfg.fp16_loss_scale,
            logging_batch_size_per_gpu=cfg.data_module.batch_size
        )

    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0

    if cfg.gpus >= 1:
        # Create trainer for prediction
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,  # Enable cudnn benchmarking
            devices=cfg.gpus,
            gradient_clip_val=0.3,
            log_every_n_steps=20,
            logger=loggers,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
        )

    # Run prediction and save results
    if load_path is not None:
        trainer.predict(model, data_module, ckpt_path=f"{load_path}/")
    else:
        trainer.predict(model, data_module)

if __name__ == "__main__":
    main()