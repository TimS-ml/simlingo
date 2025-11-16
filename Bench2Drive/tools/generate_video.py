"""Video Generator with Telemetry Overlay for CARLA Driving Data.

This script creates annotated videos from CARLA simulation frames by overlaying
vehicle telemetry data (speed, steering, throttle, brake) on each frame. Useful
for visualizing driving behavior and analyzing agent performance.

Usage:
    Modify the configuration variables at the bottom of the script, then:
    python generate_video.py
"""

import cv2
import os
import numpy as np
import json
from tqdm import trange


def create_video(images_folder, output_video, fps, font_scale, text_color, text_position):
    """Create an annotated video from simulation frames with telemetry overlay.

    Reads RGB front camera images and corresponding metadata files to create
    a video with vehicle control information displayed on each frame.

    Args:
        images_folder: Path to folder containing 'rgb_front' and 'meta' subdirectories.
        output_video: Path for the output video file (.mp4).
        fps: Frames per second for the output video.
        font_scale: Font size scale factor for overlay text.
        text_color: RGB tuple for text color (e.g., (255, 255, 255) for white).
        text_position: (x, y) tuple for text position in pixels.
    """
    # Collect all image files from the rgb_front directory
    images = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front'))
              if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    # Read first frame to get video dimensions
    frame = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), images[0]))
    height, width, layers = frame.shape

    # Initialize video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Process each frame with telemetry overlay
    for i in trange(1, len(images)):
        image = images[i]

        # Load metadata for this frame
        f = open(os.path.join(images_folder, f'meta/{i:04}.json'), 'r')
        meta = json.load(f)

        # Extract telemetry values
        steer = float(meta['steer'])
        throttle = float(meta['throttle'])
        brake = float(meta['brake'])
        # command = float(meta['command'])  # Optional: driving command
        # command_list = ["VOID", "LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT",  "CHANGE LANE RIGHT"]
        speed = float(meta['speed'])

        # Format telemetry text for overlay
        text = f'speed: {round(speed,2)}, steer: {round(steer,2)}, throttle: {round(throttle,2)}, brake: {round(brake,2)}'

        # Read the image and add text overlay
        img = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), image))
        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, text_color, 2, cv2.LINE_AA)

        # Write annotated frame to video
        video.write(img)

    video.release()

# Configuration - Update these paths before running
images_folder = ''  # Path to folder containing rgb_front/ and meta/ subdirectories
output_video = ''   # Output video filename (e.g., 'driving_video.mp4')
fps = 15            # Frames per second
font_scale = 1      # Text size multiplier
text_color = (255, 255, 255)  # White text
text_position = (50, 50)      # Top-left corner position

# Generate the video
create_video(images_folder, output_video, fps, font_scale, text_color, text_position)
