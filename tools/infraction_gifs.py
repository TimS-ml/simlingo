"""Infraction GIF Generator for CARLA Evaluation Results.

This script processes CARLA evaluation results and generates animated GIFs showing
the moments when traffic infractions occurred. For each infraction, it creates a
GIF from frames surrounding the infraction event (50 frames before and after).

Usage:
    Modify the 'base' variable to point to your evaluation results directory, then:
    python infraction_gifs.py

The script generates GIFs in an 'infractions' subdirectory, organized by infraction type.
"""

import os
import ujson
import imageio
from pathlib import Path

# Base directory containing evaluation results
base = "eval/cvprCKPT_simlingo_base/routes_validation/1/"

# List of infraction types to visualize
inspect_infractions = [
    "yield_emergency_vehicle_infractions",
    "collisions_pedestrian",
    "collisions_vehicle",
    "collisions_layout",
    "red_light",
    "stop_infraction",
    "scenario_timeouts",
    "outside_route_lanes",
    "vehicle_blocked",
    "route_dev"
]

# Process each result file in the results directory
for result in os.listdir(base + "res"):
    # Extract route index from filename
    res_idx = result.split("_")[0]

    # Load the result JSON file
    with open(base+"res/"+result) as f:
        res = ujson.load(f)

    # Skip incomplete routes
    if res["_checkpoint"]["progress"][0] < res["_checkpoint"]["progress"][1]:
        continue

    # Process each type of infraction
    for infraction_name in inspect_infractions:
        # Iterate through all infractions of this type for this route
        for i, infraction in enumerate(res["_checkpoint"]["records"][0]["infractions"][infraction_name]):
            # Extract the frame number where the infraction occurred
            infraction_frame = int(infraction.split("at Frame: ")[-1])

            # Collect frames surrounding the infraction (±50 frames)
            infraction_frames = []
            frames = os.listdir(base+"viz/"+res_idx+"/images/")
            for frame in range(infraction_frame-50, infraction_frame+51):
                # Check for both 4-digit and variable-length frame naming
                if f"{frame:04}.png" in frames:
                    infraction_frames.append(f"{base}viz/{res_idx}/images/{frame:04}.png")
                elif f"{frame}.png" in frames:
                    infraction_frames.append(f"{base}viz/{res_idx}/images/{frame}.png")

            # Create output directory for this infraction type
            os.makedirs(base+f"infractions/{infraction_name}", exist_ok=True)

            # Load all frames into memory
            images = []
            for filename in infraction_frames:
                images.append(imageio.imread(filename))

            # Create animated GIF (2 FPS, infinite loop)
            imageio.mimsave(f'{base}infractions/{infraction_name}/{res_idx}_{i}.gif', images, fps=2, loop=0)