"""Data Cleaning: Remove or Mark Failed Simulation Runs with Infractions

This module identifies and removes or renames CARLA simulation runs that failed or had
significant infractions. A run is considered failed if:
1. The simulation crashed or the agent couldn't be set up
2. The results.json.gz file is missing
3. The composed score is less than 100% (excluding runs where the only infraction is min_speed)

The script can operate in three modes:
- RENAME: Adds 'FAILED_' prefix to failed run folders
- DELETE: Permanently removes failed run folders (with ML cloud-optimized deletion)
- UNDO_RENAMING: Removes 'FAILED_' prefix to restore original names

Quality control commands:
    # Check dataset size excluding failed runs
    du -hs data --exclude=FAILED_*

    # Count remaining frames
    find data/*/[0-9]*/rgb | wc -l

    # Check scenario distribution
    echo "Number of succesful runs per scenario type";
    for dir in data/*/; do
        echo $(ls "$dir" | grep -v FAILED_* | wc -l) - $dir;
    done | sort -rn -t " " -k 1

Typical usage:
    python delete_infraction_routes.py
"""

import glob
import json
import gzip
import math
import os
from pathlib import Path
import subprocess
import tqdm


# Configuration flags for script behavior
RENAME = False  # Set to True to rename failed run folders with 'FAILED_' prefix
PREFIX = 'FAILED_'  # Prefix added to failed run folder names
DELETE = False  # Set to True to permanently delete failed runs
UNDO_RENAMING = False  # Set to True to remove 'FAILED_' prefix from folders

if __name__ == "__main__":

    # Root directory containing all CARLA simulation data
    data_save_root = 'database/simlingo/data'

    # Find all Town* folders (CARLA simulation runs) recursively
    runs = glob.glob(f'{data_save_root}/**/*Town*', recursive=True)
    num_failed = 0

    # Process each simulation run
    for run in tqdm.tqdm(runs):
        if UNDO_RENAMING:
            # Undo mode: Remove 'FAILED_' prefix from folder names
            p = Path(run)
            if p.stem.startswith(PREFIX):
                p.rename(Path(p.parent, p.stem[len(PREFIX):]))
        else:
            # Normal mode: Identify and mark/delete failed runs
            failed_flag = False
            results_file = run + '/results.json.gz'

            # Check if results file exists
            if not os.path.exists(results_file):
                failed_flag = True
            else:
                # Read and parse the compressed results JSON file
                with gzip.open(results_file, 'r') as fin:
                    json_bytes = fin.read()  # Read compressed bytes
                json_str = json_bytes.decode('utf-8')  # Decode to string

                try:
                    data = json.loads(json_str)  # Parse JSON data

                    # Check for various failure statuses
                    if data['status'] == 'Failed - Agent couldn\'t be set up':
                        failed_flag = True
                    if data['status'] == 'Failed':
                        failed_flag = True
                    if data['status'] == 'Failed - Simulation crashed':
                        failed_flag = True
                    if data['status'] == 'Failed - Agent crashed':
                        failed_flag = True

                    # Check for imperfect runs (score < 100%)
                    # We tolerate runs where the only infraction is minimum speed violations
                    if data['scores']['score_composed'] < 100.0:
                        # Check if route completion is 100%
                        cond1 = math.isclose(data['scores']['score_route'], 100)
                        # Check if all infractions are only min_speed violations
                        cond2 = data['num_infractions'] == len(data['infractions']['min_speed_infractions'])

                        # If conditions are not met, mark as failed
                        # (i.e., there are infractions other than min_speed)
                        if not (cond1 and cond2):
                            failed_flag = True
                except:
                    print(f'Error while loading json in file {results_file}')
                    failed_flag = True

            # Handle failed runs based on configuration
            if failed_flag:
                num_failed = num_failed + 1
                p = Path(run)

                if DELETE:
                    # Use ML cloud-optimized deletion: truncate files before removing
                    # This is more efficient on some distributed filesystems
                    print(f'Deleting {run}')
                    # First, truncate all non-empty files to zero size in parallel
                    subprocess.call(['bash', '-c', f'find "{run}" -type f ! -size 0c | parallel -X --progress truncate -s0'])
                    # Then remove the directory
                    subprocess.call(['bash', '-c', f'rm -rf "{run}"'])
                elif RENAME and not p.stem.startswith(PREFIX):
                    # Rename folder by adding 'FAILED_' prefix
                    p.rename(Path(p.parent, PREFIX+p.stem))

    # Print summary statistics
    if not UNDO_RENAMING:
        print(f'Failed runs: {num_failed} (out of {len(runs)} in total)')
        print(f'Successful runs: {len(runs) - num_failed}')