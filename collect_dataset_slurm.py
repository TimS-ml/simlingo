"""
CARLA Dataset Collection Script for SLURM Clusters

This script orchestrates distributed autonomous driving data collection on SLURM clusters
for the SimLingo project. It submits parallel jobs to collect driving data from CARLA
simulator across multiple routes and scenarios.

Main Features:
- Parallelizes data collection across multiple SLURM nodes (one route per node)
- Monitors running jobs and automatically resubmits crashed/failed processes
- Handles port allocation for CARLA servers (world, streaming, traffic manager)
- Tracks job progress through result JSON files
- Designed to run in a tmux terminal for long-running sessions

Workflow:
1. Scans route files (.xml) from the specified route folder
2. Creates SLURM job submission scripts for each route
3. Submits jobs respecting the max parallel job limit
4. Monitors job status and automatically resubmits failed jobs (up to 3 retries)
5. Continues until all routes are successfully collected

Author: SimLingo Team
"""

from datetime import datetime
import os
import subprocess
import time
import glob
import json
from pathlib import Path
import random
import re


def make_bash(code_dir, route_file_number, agent_name, route_file, ckeckpoint_endpoint, save_pth, seed, carla_root, town, repetition):
    """
    Generate the main execution bash script for CARLA data collection.

    This function creates a shell script that:
    - Sets up CARLA environment variables (paths, ports, seeds)
    - Launches the CARLA simulator in headless mode
    - Executes the leaderboard evaluator to collect driving data

    Args:
        code_dir (str): Root directory of the SimLingo codebase
        route_file_number (str): Unique identifier for this route (e.g., "22_0")
        agent_name (str): Path to the data collection agent script
        route_file (str): Path to the route XML file to execute
        ckeckpoint_endpoint (str): Path where result JSON will be saved
        save_pth (str): Directory path for saving collected data
        seed (int): Random seed for traffic manager (ensures different traffic patterns)
        carla_root (str): Path to CARLA installation directory
        town (str): CARLA town name (e.g., "Town01", "Town12")
        repetition (int): Repetition number for this route

    Returns:
        str: Path to the created bash script file
    """
    save_slurm = save_pth.replace("data/", "slurm/")

    jobfile = f"{save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh"
    # create folder
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)
    run_command = "python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${FREE_WORLD_PORT} \
        --traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} --routes=${ROUTES} --repetitions=${REPETITIONS} \
            --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} \
                --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=600"

    qsub_template = f"""#!/bin/bash
export SCENARIO_RUNNER_ROOT={code_dir}/scenario_runner_autopilot
export LEADERBOARD_ROOT={code_dir}/leaderboard_autopilot

# carla
export CARLA_ROOT={carla_root}
export CARLA_SERVER={carla_root}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:leaderboard_autopilot
export PYTHONPATH=$PYTHONPATH:scenario_runner_autopilot
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT={agent_name}
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES={route_file}
export TOWN={town}
export REPETITION={repetition}
export TM_SEED={seed}

export CHECKPOINT_ENDPOINT={ckeckpoint_endpoint}
export TEAM_CONFIG={route_file}
export RESUME=1
export DATAGEN=1
export SAVE_PATH={save_pth}

echo "Start python"

export FREE_STREAMING_PORT=$1
export FREE_WORLD_PORT=$2
export TM_PORT=$3

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"

bash {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &

sleep 180

{run_command}    
"""

    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile

def get_running_jobs(jobname, user_name):
    """
    Query SLURM for currently running jobs belonging to the user.

    Uses squeue command to retrieve job information and parses the output
    to extract job IDs, route numbers, and process IDs.

    Args:
        jobname (str): Name pattern of jobs to search for (e.g., "collect")
        user_name (str): Username to filter jobs by

    Returns:
        tuple: Contains three elements:
            - currently_num_running_jobs (int): Number of matching running jobs
            - routefile_number_list (list[str]): List of route identifiers from job names
            - pid_list (list[str]): List of SLURM job IDs (process IDs)

    Example:
        For a job named "eval_julian_4170_0", this returns:
        - Route number: "4170_0"
        - PID: "4767364"
    """
    job_list = subprocess.check_output(
            (
                f"SQUEUE_FORMAT2='jobid:10,username:{len(username)},name:130' squeue --sort V | grep {user_name} | \
                    grep {jobname} || true"
            ),
            shell=True,
        ).decode("utf-8").splitlines()
    currently_num_running_jobs = len(job_list)
    # Parse job list - example line format: "4767364   gwb791 eval_julian_4170_0   "
    # Extract route number from job name (last two underscore-separated parts)
    routefile_number_list = [line.split("_")[-2] + "_" + line.split("_")[-1].strip() for line in job_list]
    # Extract SLURM job ID (first column)
    pid_list = [line.split(" ")[0] for line in job_list]
    return currently_num_running_jobs, routefile_number_list, pid_list

def get_last_line_from_file(filepath):
    """
    Efficiently read the last line from a file without loading the entire file.

    This function seeks to the end of the file and reads backwards to find the last line,
    which is useful for checking log files for error messages without reading gigabytes of logs.

    Args:
        filepath (str): Path to the file to read

    Returns:
        str: The last line of the file, or empty string if file is empty or cannot be read

    Note:
        Uses binary mode and seeks from end for memory efficiency with large log files
    """
    try:
        with open(filepath, "rb", encoding="utf-8") as f:
            try:
                # Seek to 2 bytes before end of file
                f.seek(-2, os.SEEK_END)
                # Read backwards until we find a newline character
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                # File is too small or empty, seek to beginning
                f.seek(0)
            last_line = f.readline().decode()
    except:
        # Return empty string if any error occurs (file not found, permission denied, etc.)
        last_line=""
    return last_line

def cancel_jobs_with_err_in_log(logroot, jobname, user_name):
    """
    Monitor running jobs and cancel those with fatal errors in their log files.

    Scans the last line of each job's output log to detect common CARLA/simulation errors
    and automatically cancels jobs that have crashed or hung. This prevents wasted compute
    resources on jobs that will never complete successfully.

    Fatal errors detected:
    - "Actor ... not found!" - CARLA actor management error
    - "Watchdog exception - Timeout" - Job timed out without progress
    - "Engine crash handling finished; re-raising signal 11" - Unreal Engine crash

    Args:
        logroot (str): Root directory containing job log files
        jobname (str): Job name pattern to search for
        user_name (str): Username to filter jobs by

    Side Effects:
        Cancels SLURM jobs via `scancel` command if fatal errors are detected
    """
    # Scan all running jobs for error patterns
    print("Checking logs for errors...")
    _, routefile_number_list, pid_list = get_running_jobs(jobname, user_name)
    for i, rf_num in enumerate(routefile_number_list):
        logfile_path = os.path.join(logroot, f"run_files/logs/qsub_out{rf_num}.log")
        last_line = get_last_line_from_file(logfile_path)
        terminate = False
        # Check for various fatal error patterns
        if "Actor" in last_line and "not found!" in last_line:
            terminate = True
        if "Watchdog exception - Timeout" in last_line:
            terminate = True
        if "Engine crash handling finished; re-raising signal 11" in last_line:
            terminate = True
        if terminate:
            print(f"Terminating route {rf_num} with pid {pid_list[i]} due to error in logfile.")
            subprocess.check_output(f"scancel {pid_list[i]}", shell=True)

def wait_for_jobs_to_finish(logroot, jobname, user_name, max_n_parallel_jobs):
    """
    Block until the number of running jobs drops below the maximum allowed.

    Polls the job queue every 5 seconds and periodically checks for errors in logs.
    This function is used to throttle job submission to respect cluster resource limits.

    Args:
        logroot (str): Root directory containing job log files
        jobname (str): Job name pattern to search for
        user_name (str): Username to filter jobs by
        max_n_parallel_jobs (int): Maximum number of jobs allowed to run concurrently

    Behavior:
        - Checks logs for errors every 4th iteration (every ~20 seconds)
        - Sleeps for 5 seconds between checks
        - Returns when job count drops below the max limit
    """
    currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
    print(f"{currently_running_jobs}/{max_n_parallel_jobs} jobs are running...")
    counter = 0
    while currently_running_jobs >= max_n_parallel_jobs:
        # Check for errors periodically (every 4th iteration)
        if counter == 0:
            cancel_jobs_with_err_in_log(logroot, jobname, user_name)
        time.sleep(5)
        currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
        counter = (counter + 1) % 4  # Cycle counter from 0 to 3

def get_num_jobs(job_name, username):
    """
    Get the current number of running jobs and the maximum allowed parallel jobs.

    Queries SLURM for running jobs matching the job name and username, then reads
    the maximum job limit from a configuration file.

    Args:
        job_name (str): Job name pattern to search for
        username (str): Username to filter jobs by

    Returns:
        tuple: (num_running_jobs, max_num_parallel_jobs)
            - num_running_jobs (int): Current number of matching jobs in queue
            - max_num_parallel_jobs (int): Maximum jobs allowed (from max_num_jobs.txt, default 1)

    Note:
        Reads max job limit from 'max_num_jobs.txt' file. If file doesn't exist or
        cannot be read, defaults to 1 job at a time.
    """
    len_usrn = len(username)
    # Query SLURM and count matching jobs
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        ).decode('utf-8').replace('\n', ''))

    # Read maximum parallel jobs from config file
    try:
        with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())
    except:
        # Default to 1 if file not found or invalid
        max_num_parallel_jobs = 1

    return num_running_jobs, max_num_parallel_jobs

def get_which_partition(default):
    """
    Determine which SLURM partition to use for job submission.

    Reads partition name from 'partition.txt' file and validates it against
    known cluster partitions. Falls back to default if invalid or file missing.

    Args:
        default (str): Default partition name to use if file not found or invalid

    Returns:
        str: Validated partition name to use for SLURM job submission

    Valid Partitions:
        - "a100-galvani" - A100 GPU partition
        - "2080-galvani" - RTX 2080 GPU partition
        - "2080-preemptable-galvani" - Preemptable RTX 2080 partition (lower priority)
        - "a100-preemptable-galvani" - Preemptable A100 partition (lower priority)
    """
    try:
        with open('partition.txt', 'r', encoding='utf-8') as f:
            partition_name = f.read()
            # Validate partition name against known partitions
            if partition_name not in ["a100-galvani", "2080-galvani", "2080-preemptable-galvani", "a100-preemptable-galvani"]:
                partition_name = default
    except:
        print("partition.txt not found. Using default partition.")
        partition_name = default

    return partition_name



def make_jobsub_file(save_path_data, jobname, route_file_number, partition_name, repetition, timeout="0-02:00"):
    """
    Create a SLURM job submission script for a data collection job.

    Generates a bash script with SLURM directives that:
    - Specifies resource requirements (GPU, CPU, memory, time limit)
    - Finds available network ports for CARLA communication
    - Logs git branch and commit information for reproducibility
    - Executes the main data collection bash script

    Args:
        save_path_data (str): Base directory where data will be saved
        jobname (str): Name for the SLURM job (used for identification in queue)
        route_file_number (str): Unique identifier for this route
        partition_name (str): SLURM partition to submit job to
        repetition (int): Repetition number for this route
        timeout (str, optional): SLURM time limit in format "D-HH:MM". Default: "0-02:00" (2 hours)

    Returns:
        str: Path to the created job submission script

    Resource Requirements:
        - 1 node, 1 task, 8 CPUs, 40GB RAM, 1 GPU
        - Default timeout: 2 hours (can be overridden)
    """
    save_slurm = save_path_data.replace("data/", "slurm/")
    # Create necessary directory structure for logs and scripts
    os.makedirs(f"{save_slurm}/run_files/logs", exist_ok=True)
    os.makedirs(f"{save_slurm}/run_files/job_files", exist_ok=True)
    os.makedirs(f"{save_slurm}/run_files/start_files", exist_ok=True)
    jobfile = f"{save_slurm}/run_files/job_files/{route_file_number}_Rep{repetition}.sh"
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={jobname}_{route_file_number}
#SBATCH --partition={partition_name}
#SBATCH -o {save_slurm}/run_files/logs/qsub_out{route_file_number}.log
#SBATCH -e {save_slurm}/run_files/logs/qsub_err{route_file_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time={timeout}
#SBATCH --gres=gpu:1
# -------------------------------

echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
scontrol show job $SLURM_JOB_ID

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo "Current branch:"
git branch
echo "Current commit:"
git log -1
echo "Current hash:"
git rev-parse HEAD


export FREE_STREAMING_PORT=`comm -23 <(seq 10000 10400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
export FREE_WORLD_PORT=`comm -23 <(seq 20000 20400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
export TM_PORT=`comm -23 <(seq 30000 30400 | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

sleep 2

echo "start python"
pwd
bash {save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh $FREE_STREAMING_PORT $FREE_WORLD_PORT $TM_PORT
"""
    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile

if __name__ == "__main__":
    """
    Main execution block for distributed CARLA dataset collection.

    Configuration Steps:
    1. Set up paths and user credentials
    2. Scan for route XML files
    3. Submit SLURM jobs for each route
    4. Monitor job execution and resubmit failed jobs
    5. Continue until all routes complete successfully

    User Configuration Required:
        - default_partition: Your SLURM partition name
        - username: Your cluster username
        - code_root: Path to simlingo repository
        - carla_root: Path to CARLA installation
        - root_folder: Base directory for saving datasets
    """
    # ==================== Configuration ====================
    # Job execution parameters
    repetitions = 1  # Number of times to repeat each route
    repetition_start = 0  # Starting repetition index
    default_partition = "YOUR_PARTITION"  # TODO: Set your SLURM partition
    job_name = "collect"  # Name prefix for all submitted jobs
    username = "YOUR_USER"  # TODO: Set your username

    # Path configuration
    code_root = r"/path/to/simlingo"  # TODO: Set path to this repository
    carla_root = "/path/to/CARLA/root"  # TODO: Set path to CARLA installation

    # Dataset naming and storage
    date = datetime.today().strftime("%Y_%m_%d")
    dataset_name = "simlingo_v2_" + date  # Dataset name includes date
    root_folder = r"database/"  # Base folder for all datasets (with ending slash)
    data_save_directory = root_folder + dataset_name
    log_root = f"{data_save_directory}/slurm"

    # ==================== Route Discovery ====================
    route_folder = f"{code_root}/data/simlingo"

    # Find all route XML files - includes both balanced scenarios and leaderboard routes
    routes = glob.glob(f"{route_folder}/**/*balanced*/*.xml", recursive=True)  # Balanced scenario routes
    routes_lb1 = glob.glob(f"{route_folder}/**/*lb1*/**/*.xml", recursive=True)  # Leaderboard 1.0 routes

    routes = routes + routes_lb1

    # ==================== Job Submission Setup ====================
    port_offset = 0
    job_number = 1
    meta_jobs = {}  # Dictionary to track job status: {job_id: (finished, job_file, result_file, resubmit_count)}

    # Shuffle routes for better load balancing across different scenarios
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(routes)

    # Traffic manager seed counter - each route gets a unique seed for traffic variation
    seed_counter = 1000000 * repetition_start - 1  # Incremented for each route

    # ==================== Job Submission Loop ====================
    num_routes = len(routes)
    for repetition in range(repetition_start, repetitions):
        for route in routes:
            seed_counter += 1

            # Extract town name from route file path
            try:
                town = re.search('Town(\\d+)', route).group(0)
            except:
                # Fallback for routes without town in filename
                if 'validation' in route:
                    town = 'Town13'  # CARLA validation town
                elif 'training' in route:
                    town = 'Town12'  # CARLA training town
                else:
                    print(f"Town not found in route {route}")
                    continue


            scenario_type = route.split("/")[-5:-1]
            scenario_type = "/".join(scenario_type)
            routefile_number = route.split("/")[-1].split(".")[0]  # this is the number in the xml file name, e.g. 22_0.xml
            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"

            save_path = f"{code_root}/{data_save_directory}/data/{scenario_type}"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            agent = f"{code_root}/team_code/data_agent.py"

            partition_name = get_which_partition(default_partition)

            bash_file = make_bash(code_root, routefile_number, agent, route,
                                  ckpt_endpoint, save_path, seed_counter, carla_root, town, repetition)
            job_file = make_jobsub_file(save_path, job_name, routefile_number, partition_name, repetition, "0-04:00")

            # Wait until submitting new jobs that the #jobs are at below max
            num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name, username=username)
            print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
            while num_running_jobs >= max_num_parallel_jobs:
                num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name, username=username)
                time.sleep(0.05)

            print(f"Submitting job {job_number}/{num_routes}: {job_name}_{routefile_number}. ", end="")
            time.sleep(1)
            jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8") \
                                                            .strip().rsplit(" ", maxsplit=1)[-1]
            print(f"Jobid: {jobid}")
            meta_jobs[jobid] = (False, job_file, ckpt_endpoint, 0)  # job_finished, job_file, result_file, resubmitted
            job_number += 1

    # ==================== Job Monitoring and Resubmission Loop ====================
    time.sleep(1)
    training_finished = False
    while not training_finished:
        num_running_jobs, _, _ = get_running_jobs(job_name, username)
        print(f"{num_running_jobs} jobs are running... Job: {job_name}")
        cancel_jobs_with_err_in_log(log_root, job_name, username)
        time.sleep(20)

        # Check all submitted jobs and resubmit failed ones (up to 3 retries)
        for k in list(meta_jobs.keys()):
            job_finished, job_file, result_file, resubmitted = meta_jobs[k]
            need_to_resubmit = False

            # Only check unfinished jobs that haven't exceeded retry limit
            if not job_finished and resubmitted < 3:
                # Check if job is still in SLURM queue
                if int(subprocess.check_output(f"squeue | grep {k} | wc -l", shell=True).decode("utf-8").strip()) == 0:
                    # Job is not running - check if it completed successfully
                    if os.path.exists(result_file):
                        with open(result_file, "r", encoding="utf-8") as f_result:
                            evaluation_data = json.load(f_result)
                        progress = evaluation_data["_checkpoint"]["progress"]

                        # Check if route execution is incomplete
                        if len(progress) < 2 or progress[0] < progress[1]:
                            need_to_resubmit = True
                        else:
                            # Check each route record for failure status
                            for record in evaluation_data["_checkpoint"]["records"]:
                                if record["scores"]["score_route"] <= 0.00000000001:
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Agent couldn\'t be set up":
                                    need_to_resubmit = True
                                if record["status"] == "Failed":
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Simulation crashed":
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Agent crashed":
                                    need_to_resubmit = True

                        if not need_to_resubmit:
                            # Job completed successfully
                            print(f"Finished job {job_file}")
                            meta_jobs[k] = (True, None, None, 0)

                    else:
                        # Result file doesn't exist - job failed to start or crashed early
                        need_to_resubmit = True

            if need_to_resubmit:
                # rename old error files to still access it
                routefile_number = Path(job_file).stem
                print(f"Resubmit job {routefile_number} (previous id: {k}). Waiting for jobs to finish...")

                with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
                    max_num_parallel_jobs = int(f.read())
                wait_for_jobs_to_finish(log_root, job_name, username, max_num_parallel_jobs)

                time_now_log = time.time()
                os.system(f'mkdir -p "{log_root}/run_files/logs_{routefile_number}_{time_now_log}"')
                os.system(f"cp {log_root}/run_files/logs/qsub_err{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}")
                os.system(f"cp {log_root}/run_files/logs/qsub_out{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}")

                jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8").strip() \
                                                                                .rsplit(" ", maxsplit=1)[-1]
                meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
                meta_jobs[k] = (True, None, None, 0)
                print(f"resubmitted job {routefile_number}. (new id: {jobid})")

        time.sleep(10)

        if num_running_jobs == 0:
            training_finished = True
