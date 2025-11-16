"""Dataloader package for SimLingo multi-task learning.

This package contains all dataset and data loading components for SimLingo training.
It implements:
- PyTorch Lightning DataModule for orchestrating multiple datasets
- Base dataset class with CARLA data loading and preprocessing
- Task-specific datasets (driving, dreamer, QA, commentary)
- Data augmentation and temporal sampling
- Multi-task batching and collation

The dataloader supports flexible multi-task learning by combining:
- Driving waypoint prediction
- Instruction following (dreamer)
- Question answering
- Commentary generation

All datasets share a common base class and are combined via the DataModule
for efficient multi-task sampling during training.
"""
