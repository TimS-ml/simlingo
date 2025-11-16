"""CARLA dataset loader for SimLingo-Base training.

This module provides the main dataset class for loading CARLA autonomous driving
simulation data. It handles loading images, measurements, and labels from disk,
applying augmentations, and preparing batches for training.

Key Features:
    - Loads camera images and vehicle measurements from CARLA recordings
    - Applies data augmentation (camera shifts, image transforms)
    - Processes waypoints and routes for training
    - Handles temporal sequences of frames

Attribution:
    Partially adapted from https://github.com/autonomousvision/carla_garage
    (MIT License)

Dependencies:
    - OpenCV: Image loading and processing
    - NumPy: Array operations
    - BaseDataset: Parent class with core functionality
"""

import numpy as np
import random
import cv2

from simlingo_base_training.dataloader.dataset_base import BaseDataset

VIZ_DATA = False  # Debug flag for data visualization

class CARLA_Data(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """PyTorch Dataset for CARLA driving simulation data.

    Loads and processes driving data from CARLA simulator recordings, including
    camera images, vehicle measurements, and navigation routes. Inherits common
    functionality from BaseDataset.

    Attributes:
        All attributes inherited from BaseDataset including:
        - images: List of image file paths
        - measurements: List of measurement file paths
        - sample_start: Starting frame indices for each sample
        - augment_exists: Flags indicating if augmented data exists

    Note:
        This class is specifically for the base SimLingo training pipeline,
        which focuses on vision-only waypoint prediction.
    """

    def __init__(self,
            **cfg,
        ):
        """Initialize the CARLA dataset.

        Args:
            **cfg: Configuration parameters passed to BaseDataset including:
                - data_path: Path to dataset directory
                - split: 'train' or 'val'
                - hist_len: Number of historical frames
                - pred_len: Number of future waypoints to predict
                - img_augmentation: Whether to apply image augmentations
                - And many more configuration options
        """
        super().__init__(**cfg, base=True)

    def __getitem__(self, index):
        """Get a single training example by index.

        Loads all necessary data for one training sample including camera images,
        measurements, waypoints, and routes. Applies augmentation if configured.

        Args:
            index: Index of the sample to load.

        Returns:
            dict: Dictionary containing:
                - 'rgb': Processed camera images, shape [T, N, C, H, W]
                - 'speed': Current vehicle speed
                - 'waypoints': Future waypoints for prediction
                - 'route': Navigation route points
                - 'target_point': Target waypoint for navigation
                - 'map_route': Route representation for model input
                - And additional metadata

        Note:
            Disables OpenCV threading for compatibility with PyTorch dataloaders.
        """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        boxes = self.boxes[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]

        ######################################################
        ######## load current and future measurements ########
        ######################################################
        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        
        data['measurement_path'] = measurement_file_current

        # Determine whether the augmented camera or the normal camera is used.
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']
            aug_translation = current_measurement['augmentation_translation']
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0

        data['augment_sample'] = augment_sample
        data['aug_rotation'] = aug_rotation
        data['aug_translation'] = aug_translation


        ######################################################
        ################## load waypoints ####################
        ######################################################
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
       
        data['speed'] = current_measurement['speed']

        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        
        data['target_point'] = target_point
        data['next_target_point'] = next_target_point
        ######################################################
        ######## load current and past images ########
        ######################################################
        data = self.load_images(data, images, augment_sample=augment_sample)

        if self.route_as == 'coords':
            map_route = route[:20]
            data['map_route'] = map_route
        elif self.route_as == 'target_point':
            tp = [target_point, next_target_point]
            tp = np.array(tp)
            data['map_route'] = tp
        else:
            raise ValueError(f"Unknown route_as: {self.route_as}")

        return data


if __name__ == "__main__":
    from hydra import compose, initialize

    initialize(config_path="../config")
    cfg = compose(config_name="config")

    print('Test Dataset')
    dataset = CARLA_Data(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        data_path=cfg.dataset.data_path,
        bucket_path=cfg.dataset.bucket_path,
        hist_len=cfg.dataset.hist_len,
        pred_len=cfg.dataset.pred_len,
        skip_first_n_frames=cfg.dataset.skip_first_n_frames,
        num_route_points=cfg.dataset.num_route_points,
        split="train",
        bucket_name="all",
    )

    for i in range(len(dataset)):
        data = dataset[i]
        print(data)
        if i == 10:
            break
