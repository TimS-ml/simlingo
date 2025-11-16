#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""CARLA Challenge Evaluator Routes - Main Evaluation Entry Point

This module provides the main evaluation framework for the CARLA Autonomous Driving
Leaderboard. It orchestrates the complete evaluation pipeline including:

- Setting up the CARLA simulation environment (client, world, traffic manager)
- Loading and configuring autonomous agents from user-provided implementations
- Executing driving routes with scenarios and traffic
- Managing sensors and data collection
- Computing performance statistics and metrics
- Handling crashes, timeouts, and error conditions

The LeaderboardEvaluator class is the main entry point that coordinates all evaluation
activities across multiple routes and repetitions. It handles both the simulation
setup/teardown and the execution of individual route scenarios.

Typical usage:
    python leaderboard_evaluator.py --routes=routes.xml --agent=path/to/agent.py

This is the server-based version that connects to a CARLA server instance.
For local evaluation, see leaderboard_evaluator_local.py.
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import carla
import signal

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration
from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer


# Mapping from CARLA sensor types to their icon identifiers for visualization.
# This is used to track and display which sensors an agent is using during evaluation.
sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',        # RGB camera sensor
    'sensor.lidar.ray_cast':    'carla_lidar',         # LiDAR ray-casting sensor
    'sensor.other.radar':       'carla_radar',         # Radar sensor
    'sensor.other.gnss':        'carla_gnss',          # GPS/GNSS sensor
    'sensor.other.imu':         'carla_imu',           # Inertial Measurement Unit
    'sensor.opendrive_map':     'carla_opendrive_map', # HD map in OpenDRIVE format
    'sensor.speedometer':       'carla_speedometer'    # Vehicle speedometer
}

class LeaderboardEvaluator(object):
    """Main orchestrator for CARLA Leaderboard evaluation.

    This class is the central coordinator for the entire evaluation pipeline. It manages:
    - CARLA simulation client connection and world loading
    - Agent instantiation and lifecycle management
    - Scenario execution across multiple routes
    - Sensor validation and configuration
    - Statistics collection and error handling
    - Traffic manager configuration

    The evaluator runs routes sequentially, setting up the environment for each route,
    executing the scenario with the agent, collecting metrics, and cleaning up afterwards.
    It handles various failure modes including agent crashes, simulation timeouts, and
    invalid sensor configurations.

    Attributes:
        client_timeout (float): Maximum time in seconds to wait for CARLA client responses
        frame_rate (float): Simulation update frequency in Hz (20 Hz = 0.05s per tick)
        world: Reference to the CARLA world object
        manager: ScenarioManager instance that handles scenario execution
        sensors: List of sensor configurations requested by the agent
        agent_instance: The instantiated autonomous agent being evaluated
        route_scenario: Current RouteScenario being executed
        statistics_manager: Tracks and computes evaluation metrics
    """

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """Initialize the leaderboard evaluator and set up the simulation environment.

        This constructor performs the following initialization steps:
        1. Connects to the CARLA server and configures synchronous mode
        2. Sets up the traffic manager with deterministic behavior
        3. Dynamically loads the agent module from the provided path
        4. Creates the ScenarioManager for handling route execution
        5. Initializes watchdogs and signal handlers for timeout management

        Args:
            args: Parsed command-line arguments containing:
                - host: CARLA server IP address
                - port: CARLA server port number
                - traffic_manager_port: Port for the traffic manager
                - timeout: Maximum allowed time for operations
                - agent: Path to the agent Python file
                - debug: Debug level for verbose output
            statistics_manager: StatisticsManager instance for tracking metrics

        Raises:
            ImportError: If CARLA version is incompatible (< 0.9.10.1)
            ConnectionError: If unable to connect to CARLA server
        """
        self.world = None
        self.manager = None
        self.sensors = None
        self.sensors_initialized = False
        self.sensor_icons = []
        self.agent_instance = None
        self.route_scenario = None

        self.statistics_manager = statistics_manager

        # This is the ROS1 bridge server instance. This is not encapsulated inside the ROS1 agent because the same
        # instance is used on all the routes (i.e., the server is not restarted between routes). This is done
        # to avoid reconnection issues between the server and the roslibpy client.
        self._ros1_server = None

        # Setup the simulation
        self.client, self.client_timeout, self.traffic_manager = self._setup_simulation(args)

        # Verify CARLA version compatibility (except for leaderboard custom builds)
        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Dynamically load the agent module from the provided path
        # This allows users to provide custom agent implementations
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, self.statistics_manager, args.debug)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Prepare the agent timer
        self._agent_watchdog = None
        signal.signal(signal.SIGINT, self._signal_handler)

        self._client_timed_out = False

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt.
        Either the agent initialization watchdog is triggered, or the runtime one at scenario manager
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took longer than {}s to setup".format(self.client_timeout))
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._agent_watchdog:
            return self._agent_watchdog.get_status()
        return False

    def _cleanup(self):
        """Clean up all resources after route execution.

        This method performs comprehensive cleanup to ensure no resources are leaked
        between routes. It handles:
        - Stopping and destroying the agent instance
        - Removing all scenario actors from the world
        - Cleaning up the CarlaDataProvider
        - Stopping any active watchdog timers
        - Destroying all remaining sensor actors

        The cleanup is performed in a specific order to prevent dependencies issues.
        Errors during cleanup are caught and logged but don't prevent the rest of
        the cleanup from proceeding.

        Note:
            This method should be called after each route execution, whether it
            succeeded or failed, to ensure a clean slate for the next route.
        """
        CarlaDataProvider.cleanup()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        try:
            if self.agent_instance:
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        if self.route_scenario:
            self.route_scenario.remove_all_actors()
            self.route_scenario = None
            if self.statistics_manager:
                self.statistics_manager.remove_scenario()

        if self.manager:
            self._client_timed_out = not self.manager.get_running_status()
            self.manager.cleanup()

        # Make sure no sensors are left streaming
        alive_sensors = self.world.get_actors().filter('*sensor*')
        for sensor in alive_sensors:
            sensor.stop()
            sensor.destroy()

    def _setup_simulation(self, args):
        """Prepare and configure the CARLA simulation environment.

        Sets up the simulation with synchronous mode for deterministic execution.
        This includes:
        - Creating a CARLA client connection to the server
        - Configuring synchronous mode with fixed time steps
        - Enabling deterministic ragdoll physics for consistency
        - Setting up the traffic manager in hybrid physics mode

        Args:
            args: Command-line arguments containing:
                - host: CARLA server hostname or IP
                - port: CARLA server port
                - traffic_manager_port: Port for traffic manager
                - timeout: Client timeout value in seconds

        Returns:
            tuple: (client, client_timeout, traffic_manager)
                - client: Connected CARLA client instance
                - client_timeout: Configured timeout value
                - traffic_manager: Configured TrafficManager instance

        Note:
            Synchronous mode ensures the simulation only advances when explicitly
            ticked, providing deterministic behavior crucial for fair evaluation.
        """
        client = carla.Client(args.host, args.port)
        if args.timeout:
            client_timeout = args.timeout
        client.set_timeout(client_timeout)

        settings = carla.WorldSettings(
            synchronous_mode = True,
            fixed_delta_seconds = 1.0 / self.frame_rate,
            deterministic_ragdolls = True,
            spectator_as_ego = False
        )
        client.get_world().apply_settings(settings)

        traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)

        return client, client_timeout, traffic_manager

    def _reset_world_settings(self):
        """
        Changes the modified world settings back to asynchronous
        """
        # Has simulation failed?
        if self.world and self.manager and not self._client_timed_out:
            # Reset to asynchronous mode
            self.world.tick()  # TODO: Make sure all scenario actors have been destroyed
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.deterministic_ragdolls = False
            settings.spectator_as_ego = True
            self.world.apply_settings(settings)

            # Make the TM back to async
            self.traffic_manager.set_synchronous_mode(False)
            self.traffic_manager.set_hybrid_physics_mode(False)

    def _load_and_wait_for_world(self, args, town):
        """Load a CARLA town/map and configure it for the scenario.

        This method loads the specified CARLA map and configures it with the
        necessary settings for large maps and provides world data to the
        CarlaDataProvider singleton. It also:
        - Resets all traffic lights to their default state
        - Configures tile streaming for large maps
        - Sets the traffic manager random seed for reproducibility
        - Validates the loaded map matches the expected one

        Args:
            args: Command-line arguments containing traffic_manager_seed
            town: Name of the CARLA town/map to load (e.g., 'Town01', 'Town02')

        Raises:
            Exception: If the loaded map doesn't match the requested town name.
                This can happen if the CARLA server has the wrong map loaded.

        Note:
            Large map settings (tile_stream_distance, actor_active_distance) are
            reset to 650 as they get overridden when loading a new world.
        """
        self.world = self.client.load_world(town, reset_settings=False)

        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)

        # Wait for the world to be ready
        self.world.tick()

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            " This scenario requires the use of map {}".format(town))

    def _register_statistics(self, route_index, entry_status, crash_message=""):
        """Compute and save evaluation statistics for the current route.

        This method finalizes the statistics collection for a route by computing
        metrics based on the scenario execution data. It saves both the route's
        completion status and detailed performance metrics.

        Args:
            route_index (int): Index of the route in the route configuration file
            entry_status (str): Final status of the route execution, one of:
                - "Started": Route began execution
                - "Finished": Route completed successfully
                - "Failed": Route failed due to errors
            crash_message (str, optional): Description of any crash/error that occurred.
                Defaults to empty string for successful executions.

        Note:
            This method should be called exactly once per route execution, after
            the route has finished (successfully or not).
        """
        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_entry_status(entry_status)
        self.statistics_manager.compute_route_statistics(
            route_index, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message
        )

    def _load_and_run_scenario(self, args, config):
        """Load and execute a single route scenario with the autonomous agent.

        This is the main execution method for a single route. It handles the complete
        lifecycle of a route execution:

        1. World Loading: Loads the CARLA map specified in the config
        2. Scenario Creation: Creates the RouteScenario with waypoints and events
        3. Agent Setup: Instantiates and configures the autonomous agent
        4. Sensor Validation: Validates agent's sensor configuration against track rules
        5. Scenario Execution: Runs the agent through the route
        6. Statistics Collection: Records performance metrics
        7. Cleanup: Destroys actors and cleans up resources

        The method implements comprehensive error handling at each stage. Depending on
        the type of failure, it either:
        - Continues to the next route (agent initialization failures)
        - Stops the entire evaluation (simulation crashes)

        Args:
            args: Command-line arguments containing:
                - host, port: CARLA connection parameters
                - agent_config: Path to agent configuration file
                - track: Competition track (SENSORS or MAP)
                - debug: Debug output level
                - record: Path to save CARLA recording
            config: RouteConfiguration object containing:
                - name: Route identifier
                - town: CARLA map name
                - index: Route index in the configuration file
                - repetition_index: Repetition number for this route

        Returns:
            bool: True if simulation crashed (should stop all evaluation),
                  False otherwise (can continue to next route)

        Raises:
            The method catches and handles all exceptions internally, converting
            them to appropriate status messages and return values.
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========\033[0m".format(config.name, config.repetition_index))

        # Prepare the statistics of the route
        route_name = f"{config.name}_rep{config.repetition_index}"
        self.statistics_manager.create_route_data(route_name, config.index)

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args, config.town)
            self.route_scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self.statistics_manager.set_scenario(self.route_scenario)

        except Exception:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        print("\033[1m> Setting up the agent\033[0m")

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            # Start watchdog timer to detect if agent setup hangs or takes too long
            self._agent_watchdog = Watchdog(args.timeout)
            self._agent_watchdog.start()

            # Dynamically get the agent class name and instantiate it
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            agent_class_obj = getattr(self.module_agent, agent_class_name)

            # Start the ROS1 bridge server only for ROS1 based agents.
            # The server is shared across all routes to avoid reconnection issues.
            if getattr(agent_class_obj, 'get_ros_version')() == 1 and self._ros1_server is None:
                from leaderboard.autoagents.ros1_agent import ROS1Server
                self._ros1_server = ROS1Server()
                self._ros1_server.start()

            # Create agent instance and provide it with the route plan
            self.agent_instance = agent_class_obj(args.host, args.port, args.debug)
            self.agent_instance.set_global_plan(self.route_scenario.gps_route, self.route_scenario.route)
            self.agent_instance.setup(args.agent_config)

            # Check and store the sensors (only once for the first route)
            # Sensor configuration is assumed to be the same across all routes
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                # Validate that the sensor suite is legal for the selected track
                validate_sensor_configuration(self.sensors, track, args.track)

                # Save sensor configuration for statistics reporting
                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons)
                self.statistics_manager.write_statistics()

                self.sensors_initialized = True

            # Stop the watchdog - agent setup completed successfully
            self._agent_watchdog.stop()
            self._agent_watchdog = None

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print(f"{e}\033[0m\n")

            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        except Exception:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return False

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(self.route_scenario, self.agent_instance, config.index, config.repetition_index)
            self.manager.run_scenario()

        except AgentError:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]

        except Exception:
            print("\n\033[91mError during the simulation:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config.index, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()

        except Exception:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print(f"\n{traceback.format_exc()}\033[0m")

            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # If the simulation crashed, stop the leaderboard, for the rest, move to the next route
        return crash_message == "Simulation crashed"

    def run(self, args):
        """Execute the complete leaderboard evaluation across all routes.

        This is the main evaluation loop that processes all routes specified in the
        configuration file. It handles:
        - Route indexing and iteration
        - Resume functionality from checkpoints
        - Sequential execution of all routes and their repetitions
        - Progress tracking and statistics writing
        - Global statistics computation
        - Graceful shutdown including ROS1 bridge cleanup

        The evaluation runs until either:
        - All routes have been completed
        - A simulation crash occurs (unrecoverable error)

        Args:
            args: Command-line arguments containing:
                - routes: Path to XML file with route configurations
                - repetitions: Number of times to repeat each route
                - routes_subset: Optional subset of routes to execute
                - resume: Whether to resume from a checkpoint
                - checkpoint: Path to checkpoint file for resume/statistics

        Returns:
            bool: True if evaluation was stopped due to a crash, False if completed
                  successfully. This is used for the exit code of the program.

        Note:
            The method automatically saves progress after each route, allowing
            resumption from the last completed route if the evaluation is interrupted.
        """
        route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)

        if args.resume:
            resume = route_indexer.validate_and_resume(args.checkpoint)
        else:
            resume = False

        if resume:
            self.statistics_manager.add_file_records(args.checkpoint)
        else:
            self.statistics_manager.clear_records()
        self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
        self.statistics_manager.write_statistics()

        crashed = False
        while route_indexer.peek() and not crashed:

            # Run the scenario
            config = route_indexer.get_next_config()
            crashed = self._load_and_run_scenario(args, config)

            # Save the progress and write the route statistics
            self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
            self.statistics_manager.write_statistics()

        # Shutdown ROS1 bridge server if necessary
        if self._ros1_server is not None:
            self._ros1_server.shutdown()

        # Go back to asynchronous mode
        self._reset_world_settings()

        if not crashed:
            # Save global statistics
            print("\033[1m> Registering the global statistics\033[0m")
            self.statistics_manager.compute_global_statistics()
            self.statistics_manager.validate_and_write_statistics(self.sensors_initialized, crashed)

        return crashed

def main():
    """Main entry point for the CARLA leaderboard evaluation system.

    This function handles command-line argument parsing and orchestrates the complete
    evaluation process. It:
    1. Parses command-line arguments for CARLA connection, routes, and agent config
    2. Creates the StatisticsManager for tracking evaluation metrics
    3. Initializes the LeaderboardEvaluator
    4. Runs the evaluation across all configured routes
    5. Returns appropriate exit code based on success/failure

    Exit codes:
        0: Evaluation completed successfully
        -1: Evaluation stopped due to simulation crash

    Command-line arguments include:
        --host, --port: CARLA server connection
        --routes: XML file with route configurations
        --agent: Path to agent implementation
        --checkpoint: Path for saving/loading progress
        See argument parser below for complete list
    """
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--traffic-manager-port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=0, type=int,
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int,
                        help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default=300.0, type=float,
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes', required=True,
                        help='Name of the routes file to be executed.')
    parser.add_argument('--routes-subset', default='', type=str,
                        help='Execute a specific set of routes')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str,
                        help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str,
                        help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS',
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--debug-checkpoint", type=str, default='./live_results.txt',
                        help="Path to checkpoint used for saving live results")

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager(arguments.checkpoint, arguments.debug_checkpoint)
    leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
    crashed = leaderboard_evaluator.run(arguments)

    del leaderboard_evaluator

    if crashed:
        sys.exit(-1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
