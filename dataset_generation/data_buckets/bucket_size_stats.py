"""Data Buckets: Calculate and Display Relative Bucket Sizes

This module analyzes bucket statistics to compute the relative size (percentage)
of each bucket compared to the total dataset. This helps understand the distribution
of data across different driving scenarios and conditions.

The script:
1. Loads bucket statistics from a JSON file
2. Calculates the percentage of total samples in each bucket
3. Displays the relative proportions in a human-readable format

This analysis is useful for:
- Understanding dataset composition
- Identifying over/under-represented scenarios
- Planning data collection or rebalancing strategies
- Configuring weighted sampling during training

Typical usage:
    python bucket_size_stats.py

Output example:
    junction: 15.42%
    red_light: 8.73%
    green_light: 11.25%
    ...
"""

import json

# Path to the bucket statistics JSON file
file = 'database/bucketsv2_simlingo/buckets_stats.json'

# Load bucket statistics
with open(file, 'r') as f:
    data = json.load(f)

# Calculate percentage of each bucket relative to total dataset
bucket_relatives = {}
total = data['total']
for key in data.keys():
    if key == 'total':
        continue  # Skip the 'total' entry itself
    bucket_relatives[key] = data[key] / total

# Display bucket percentages in a readable format
for key in bucket_relatives.keys():
    print(f'{key}: {bucket_relatives[key]*100:.2f}%')