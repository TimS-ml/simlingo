"""Route Results JSON Merger for Bench2Drive Evaluation.

This script merges multiple individual route result JSON files into a single
consolidated file with aggregate metrics. It computes overall driving scores,
success rates, and filters out crashed routes.

Usage:
    python merge_route_json.py -f <results_folder>

Example:
    python merge_route_json.py -f eval/bench2drive/results/res

The merged results are saved as 'merged.json' in the specified folder.
"""

import json
import glob
import argparse
import os

def merge_route_json(folder_path):
    """Merge individual route JSON files into a single consolidated result.

    Combines all route records from separate JSON files, computes aggregate
    metrics (driving score, success rate), and excludes crashed routes.
    The Bench2Drive benchmark expects 220 routes total.

    Args:
        folder_path: Path to folder containing individual route JSON files.
    """
    # Find all JSON files in the folder
    file_paths = glob.glob(f'{folder_path}/*.json')

    merged_records = []
    driving_score = []
    success_num = 0

    # Process each JSON file
    for file_path in file_paths:
        # Skip previously merged file
        if 'merged.json' in file_path:
            continue

        with open(file_path) as file:
            data = json.load(file)
            records = data['_checkpoint']['records']

            for rd in records:
                # Skip routes where agent crashed
                if rd['status'] == 'Failed - Agent crashed':
                    continue

                # Remove index field (not needed in merged file)
                rd.pop('index')

                merged_records.append(rd)
                driving_score.append(rd['scores']['score_composed'])

                # Check if route completed successfully without infractions
                if rd['status']=='Completed' or rd['status']=='Perfect':
                    success_flag = True
                    # Check for any infractions (except min_speed which is less critical)
                    for k,v in rd['infractions'].items():
                        if len(v)>0 and k != 'min_speed_infractions':
                            success_flag = False
                            break
                    if success_flag:
                        success_num += 1
                        print(rd['route_id'])  # Print successful route IDs

    # Warn if route count doesn't match Bench2Drive benchmark
    if len(merged_records) != 220:
        print(f"-----------------------Warning: there are {len(merged_records)} routes in your json, "
              f"which does not equal to 220. All metrics (Driving Score, Success Rate, Ability) are inaccurate!!!")

    # Sort records by route ID in descending order
    merged_records = sorted(merged_records, key=lambda d: d['route_id'], reverse=True)

    # Create merged checkpoint structure
    _checkpoint = {
        "records": merged_records
    }

    # Create merged data with aggregate metrics
    merged_data = {
        "_checkpoint": _checkpoint,
        "driving score": sum(driving_score) / len(driving_score) if driving_score else 0,
        "success rate": success_num / len(driving_score) if driving_score else 0,
        "eval num": len(driving_score),
    }

    # Write merged results to file
    with open(os.path.join(folder_path, 'merged.json'), 'w') as file:
        json.dump(merged_data, file, indent=4)

if __name__ == '__main__':
    import glob
    parser = argparse.ArgumentParser(description='Merge route result JSON files')
    parser.add_argument('-f', '--folder',
                       help='Folder containing result JSON files',
                       default='eval_results/Bench2Drive/reproCVPTsimlingo_2025_05_02_04_08_01_simlingo_withaugmentation_seed2/bench2drive/1/res')
    args = parser.parse_args()

    # Validate folder exists and merge
    if os.path.isdir(args.folder):
        merge_route_json(args.folder)
