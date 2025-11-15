"""Data Cleaning: Remove Failed CARLA Simulation Runs

This module cleans up the CARLA simulation dataset by removing duplicate or failed runs.
When a CARLA route is executed multiple times (e.g., due to failures or crashes), multiple
folders with the same route identifier but different timestamps are created. This script
identifies such duplicate runs and retains only the most recent (successful) run while
deleting older attempts.

The script operates in dry-run mode by default (delete=False) to allow inspection before
actual deletion.

Typical usage:
    python delete_failed_runs.py

Directory structure example:
    Town01_route123_2024-01-01_10-30-00/
    Town01_route123_2024-01-01_11-45-00/  <- Kept (newest)
    Town01_route123_2024-01-01_09-15-00/  <- Deleted (older failed run)
"""

import glob
import shutil
import pathlib


# Path to the root directory containing all CARLA simulation data
dataset_path = 'database/simlingo/data'

# Find all folders matching Town* pattern recursively (CARLA town simulation folders)
all_data_folders = glob.glob(f'{dataset_path}/**/Town*', recursive=True)

# Safety flag: Set to True to actually delete folders, False for dry-run
delete = False

# Track which route groups we've already processed to avoid redundant checks
# When multiple folders only differ by timestamp (last part of path), delete all but the newest
already_checked_root = []
num_deleted = 0

for data_folder in all_data_folders:
    data_folder = pathlib.Path(data_folder)
    data_folder_name = data_folder.name

    # Extract the route identifier (everything after 'route')
    # Example: "Town01_route123_2024-01-01_10-30-00" -> "123_2024-01-01_10-30-00"
    data_folder_parts = data_folder_name.split('route')[-1]

    # Extract only the timestamp portion (everything after the first underscore)
    # This gives us the date/time that differs between duplicate runs
    data_folder_date_time = '_'.join(data_folder_parts.split('_')[1:])

    # Get the path without the timestamp to identify the route group
    # All runs of the same route will have the same path_without_date_time prefix
    path_without_date_time = str(data_folder).split(data_folder_date_time)[0]

    # Skip if we've already processed this route group
    if path_without_date_time in already_checked_root:
        continue
    already_checked_root.append(path_without_date_time)

    # Find all folders for this route (same route, different timestamps)
    all_data_folders_without_date_time = glob.glob(f'{path_without_date_time}*')

    # If there are multiple runs of the same route
    if len(all_data_folders_without_date_time) > 1:
        # Sort by path (which includes timestamp) - newest will be last
        all_data_folders_without_date_time.sort()

        # Delete all but the last (newest) folder
        for folder in all_data_folders_without_date_time[:-1]:
            print(f"Deleting {folder}")
            num_deleted += 1
            if delete:
                shutil.rmtree(folder)

print(f"Deleted {num_deleted} folders out of {len(all_data_folders)}")