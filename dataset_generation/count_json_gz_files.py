"""Dataset Statistics: Count Compressed JSON Files

This module counts the number of compressed JSON (.json.gz) files in each major
data category of the SimLingo dataset. It provides a quick overview of the dataset
size across different language label types: commentary, dreamer, and drivelm.

The script is useful for:
- Verifying data generation completeness
- Tracking dataset growth during generation
- Quick sanity checks before training

Typical usage:
    python count_json_gz_files.py

Output example:
    Found 1500 files in database/simlingo/commentary
    Found 800 files in database/simlingo/dreamer
    Found 1200 files in database/simlingo/drivelm
"""

import json
import gzip
import glob

# Paths to the main data directories containing language labels
paths = [
    'database/simlingo/commentary',  # Driving commentary annotations
    'database/simlingo/dreamer',     # Dreamer model data
    'database/simlingo/drivelm',     # DriveLM VQA annotations
]


# Count and report files in each directory
for path in paths:
    # Recursively find all compressed JSON files in subdirectories
    files = glob.glob(f'{path}/**/*.json.gz', recursive=True)

    # Print count of files found
    print(f'Found {len(files)} files in {path}')

