"""Data Cleaning: Remove Corrupted or Malformed JSON Files

This module identifies and removes corrupted or invalid compressed JSON files in the
dataset. It validates each .json.gz file by attempting to parse it, and deletes any
files that fail to load properly. The validation process runs in parallel across all
available CPU cores for efficiency.

Common causes of corrupted JSON files:
- Incomplete writes due to process crashes
- Disk errors or filesystem issues
- Encoding problems during file generation

The script operates with a safety breakpoint before deletion to allow inspection.

Typical usage:
    python delete_defect_jsons.py
"""

import glob
import gzip
import ujson
import os
import tqdm
import multiprocessing
from functools import partial


def check_json_file(json_file):
    """Validate a compressed JSON file by attempting to parse it.

    This function tries to load and parse a gzipped JSON file. If the file is
    valid JSON and can be successfully decompressed and parsed, it returns None.
    If any error occurs during reading or parsing, the file path is returned
    to indicate it should be deleted.

    Args:
        json_file (str): Path to the .json.gz file to validate.

    Returns:
        str or None: The file path if the file is corrupted/invalid, None if valid.
    """
    try:
        # Attempt to open and parse the compressed JSON file
        with gzip.open(json_file, 'rt') as fin:
            ujson.load(fin)
        return None  # File is valid
    except Exception as e:
        # File is corrupted or malformed
        print(f'Error loading {json_file}: {e}')
        return json_file  # Return file path to mark for deletion


def main():
    """Main function to scan and delete corrupted JSON files.

    Scans the specified directory for all .json.gz files, validates them in parallel
    using multiprocessing, and removes any corrupted files. Includes a breakpoint
    before deletion for safety.
    """
    # Target directory to scan for JSON files
    # Note: The second assignment overrides the first, so only 'dreamer' is processed
    path = 'database/simlingo/commentary'
    path = 'database/simlingo/dreamer'

    # Find all compressed JSON files recursively
    all_jsons = glob.glob(f'{path}/**/*.json.gz', recursive=True)
    print(f'Found {len(all_jsons)} json files')

    # Use all available CPU cores for parallel processing
    num_cores = multiprocessing.cpu_count()
    print(f'Using {num_cores} cores for processing')

    # Process files in parallel to validate JSON integrity
    with multiprocessing.Pool(processes=num_cores) as pool:
        # imap returns results in order as they complete
        results = list(tqdm.tqdm(
            pool.imap(check_json_file, all_jsons),
            total=len(all_jsons),
            desc="Validating JSON files"
        ))

    # Extract list of corrupted files (non-None results)
    to_delete = [file for file in results if file is not None]

    print(f'Deleting {len(to_delete)} json files')

    # Safety breakpoint: pause execution to allow manual inspection before deletion
    breakpoint()

    # Delete all corrupted files
    for json_file in to_delete:
        os.remove(json_file)
        print(f'Deleted {json_file}')


if __name__ == "__main__":
    main()