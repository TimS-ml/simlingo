"""Data Buckets: Generate Statistics from Bucket Paths

This module processes bucket path pickle files to generate comprehensive statistics
about data bucket distribution. Buckets are categories of driving scenarios/conditions
used for balanced training.

The script:
1. Loads bucket paths from a pickle file
2. Counts the number of data samples in each bucket
3. Calculates the total number of unique data samples across all buckets
4. Saves statistics as a JSON file for analysis and visualization

Bucket statistics help understand:
- Data distribution across different driving scenarios
- Whether certain scenarios are under/over-represented
- Which scenarios need more data collection

Typical usage:
    python get_bucket_stats.py

Input:
    - buckets_paths.pkl: Dictionary mapping bucket names to lists of file paths

Output:
    - buckets_stats.json: Statistics showing sample counts per bucket
"""

import pickle
import ujson

# Path to the bucket paths pickle file
bucket_path = 'database/buckets_expertv4_0/buckets_paths.pkl'
save_path = 'database/buckets_expertv4_0'
buckets_stats = {}

# Load the bucket paths dictionary
with open(bucket_path, 'rb') as f:
    buckets = pickle.load(f)

# Count the number of samples in each bucket
for key, value in buckets.items():
    buckets_stats[key] = len(value)

# Calculate total unique samples across all buckets
# (samples may appear in multiple buckets, so we need unique count)
unique_bucket_values = set()
for key, value in buckets.items():
    unique_bucket_values.update(value)

# Store total unique sample count
buckets_stats['total'] = len(unique_bucket_values)

# Save bucket statistics as a JSON file
with open(f'{save_path}/buckets_stats.json', 'w') as f:
    ujson.dump(buckets_stats, f, indent=4)