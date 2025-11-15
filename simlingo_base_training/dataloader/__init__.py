"""Data loading utilities for SimLingo-Base training.

This package provides dataset implementations for loading CARLA simulation data
and creating PyTorch dataloaders for model training.

Main Components:
    - CARLA_Data: Dataset class for CARLA driving data
    - BaseDataset: Base class with common functionality
    - DataModule: PyTorch Lightning DataModule for training

The dataloaders handle:
    - Loading camera images and measurements from disk
    - Data augmentation (camera shifts, image transforms)
    - Waypoint and route processing
    - Batch collation for model input
"""
