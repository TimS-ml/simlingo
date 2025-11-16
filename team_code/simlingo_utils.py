"""SimLingo Utility Functions - Camera Geometry and 3D Projections.

This module provides utility functions for camera geometry computations used in
the SimLingo agent for visualizing predicted waypoints on camera images.

The functions handle:
1. 3D to 2D projection: Project 3D waypoints onto camera image plane
2. Camera calibration: Compute intrinsic and extrinsic camera matrices
3. Coordinate transformations: Convert between different coordinate frames

These utilities are primarily used for:
- Visualizing predicted waypoints on RGB images during debugging
- Rendering route overlays on camera frames
- Converting ego-relative waypoints to pixel coordinates

Functions:
    project_points: Project 3D points to 2D image coordinates
    get_rotation_matrix: Compute 3D rotation matrix from roll, pitch, yaw
    get_camera_intrinsics: Compute camera intrinsic calibration matrix
    get_camera_extrinsics: Get camera extrinsic transformation matrix
"""

import numpy as np
import math
import cv2
import torch


def project_points(points2D_list, K, tvec=None, rvec=None):
    """Project 3D points (in ego-vehicle coordinates) to 2D image coordinates.

    Uses OpenCV camera projection with calibration matrix to map 3D waypoints
    to pixel coordinates for visualization overlay on camera images.

    Args:
        points2D_list: List of 2D points in ego coordinates [(x, y), ...]
            x is forward distance, y is lateral offset
        K: Camera intrinsic matrix (3x3) from get_camera_intrinsics()
        tvec: Camera translation vector (optional), default is [0.0, 2.0, 1.5]
        rvec: Camera rotation vector (optional), default is zeros

    Returns:
        list: List of 2D pixel coordinates [(u, v), ...] for each input point
    """
    all_points_2d = []
    # Handle rotation vector - convert from agent convention to OpenCV convention
    if rvec is None:
        rvec_new = np.zeros((3, 1), np.float32)
    else:
        # Coordinate frame transformation for CARLA camera
        rvec_new = np.array([[-rvec[1], rvec[2], rvec[0]]], np.float32)
    if tvec is None:
        # Default camera position: slightly forward, at hood height
        tvec = np.array([[0.0, 2.0, 1.5]], np.float32)

  # print(f"rvec_new: {rvec_new}")
  for point in  points2D_list:
    pos_3d = np.array([point[1], 0, point[0]+tvec[0][2]])
    # Define the distortion coefficients 
    dist_coeffs = np.zeros((5, 1), np.float32) 
    points_2d, _ = cv2.projectPoints(pos_3d, 
                        rvec=rvec_new, tvec=tvec, 
                        cameraMatrix=K, 
                        distCoeffs=dist_coeffs)
    all_points_2d.append(points_2d[0][0])
        
  return all_points_2d

def get_rotation_matrix(roll, pitch, yaw):
    """Compute 3D rotation matrix from Euler angles.

    Creates a rotation matrix by composing rotations around the three axes.
    The result is the inverse (transpose) of the standard rotation for camera
    coordinate transformations.

    Args:
        roll: Roll angle in degrees (rotation around x-axis)
        pitch: Pitch angle in degrees (rotation around y-axis)
        yaw: Yaw angle in degrees (rotation around z-axis)

    Returns:
        np.matrix: 3x3 rotation matrix (transposed for camera frame)
    """
    # Convert degrees to radians
    roll = roll * np.pi / 180.0
    pitch = pitch * np.pi / 180.0
    yaw = yaw * np.pi / 180.0

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = pitchMatrix * yawMatrix * rollMatrix

    #inverse rotation
    R = R.T

    return R

def get_camera_intrinsics(w, h, fov):
  """
  Get camera intrinsics matrix from width, height and fov.
  Returns:
    K: A float32 tensor of shape ``[3, 3]`` containing the intrinsic calibration matrices for
      the carla camera.
  """
  focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
  K = np.identity(3)
  K[0, 0] = K[1, 1] = focal
  K[0, 2] = w / 2.0
  K[1, 2] = h / 2.0

  K = torch.tensor(K, dtype=torch.float32)
  return K

def get_camera_extrinsics():
    """
    Get camera extrinsics matrix for the carla camera.
    extrinsics: A float32 tensor of shape ``[4, 4]`` containing the extrinic calibration matrix for
      the carla camera. The extriniscs are specified as homogeneous matrices of the form ``[R t; 0 1]``
    """
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    extrinsics[3, 3] = 1.0
    extrinsics[:3, :3] = np.eye(3)
    extrinsics[:3, 3] = [-1.5, 0.0, 2.0]
    extrinsics = torch.tensor(extrinsics, dtype=torch.float32)

    return extrinsics

