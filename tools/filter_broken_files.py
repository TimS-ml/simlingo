"""Broken File Detector for Compressed JSON Database.

This script identifies corrupted or unreadable compressed JSON files in a database.
It uses multi-threading to quickly scan large directories and identify files that
cannot be properly loaded or parsed.

Usage:
    python filter_broken_files.py

The script is configured to scan the 'database/simlingo/dreamer' directory for
.json.gz files and reports any that fail to load.
"""

import glob
import json
import gzip
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
path = 'database/simlingo/dreamer'
file_ending = '.json.gz'

# Recursively find all compressed JSON files in the directory
all_files = glob.glob(f'{path}/**/*{file_ending}', recursive=True)

# Print the total number of files found
print(len(all_files))

def check_file(file):
    """Verify that a compressed JSON file can be loaded successfully.

    Args:
        file: Path to the .json.gz file to check.

    Returns:
        str or None: Returns the file path if it's broken, None if it loads successfully.
    """
    try:
        with gzip.open(file, 'rt') as f:
            data = json.load(f)
        return None  # File is fine
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        return file  # File is broken

# Find broken files using multi-threaded processing for speed
broken_files = []
with ThreadPoolExecutor(max_workers=20) as executor:
    # Submit all file checks to the thread pool
    futures = {executor.submit(check_file, file): file for file in all_files}
    # Process results as they complete with a progress bar
    for future in tqdm.tqdm(as_completed(futures), total=len(all_files)):
        result = future.result()
        if result:
            broken_files.append(result)

# Report the number of broken files found
print(len(broken_files))
breakpoint()  # Pause for inspection of results