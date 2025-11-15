#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""CARLA Challenge Evaluator Routes - Local Evaluation Version

This module provides the local evaluation framework for the CARLA Autonomous Driving
Leaderboard. It is a variant of the standard leaderboard_evaluator.py with modifications
for local/offline evaluation scenarios.

Key differences from the standard evaluator:
- Uses local versions of ScenarioManager and StatisticsManager
- Includes automatic port detection to avoid port conflicts
- Supports data generation mode (DATAGEN environment variable)
- Modified agent initialization with route-specific naming
- Enhanced cleanup handling with result passing

The evaluation pipeline remains similar:
- Setting up the CARLA simulation environment
- Loading and configuring autonomous agents
- Executing driving routes with scenarios
- Managing sensors and collecting data
- Computing performance statistics
- Handling errors and timeouts

Typical usage:
    python leaderboard_evaluator_local.py --routes=routes.xml --agent=path/to/agent.py

This version is optimized for local development and batch data collection.
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import importlib
import os
import sys
import signal
import socket

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper_local import AgentError, validate_sensor_configuration
from leaderboard.utils.statistics_manager_local import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer

import pathlib


# Mapping from CARLA sensor types to their icon identifiers for visualization.
# This is used to track and display which sensors an agent is using during evaluation.
# Extended to include data generation sensors (semantic segmentation, depth).
sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',        # RGB camera sensor
    'sensor.lidar.ray_cast':    'carla_lidar',         # LiDAR ray-casting sensor
    'sensor.other.radar':       'carla_radar',         # Radar sensor
    'sensor.other.gnss':        'carla_gnss',          # GPS/GNSS sensor
    'sensor.other.imu':         'carla_imu',           # Inertial Measurement Unit
    'sensor.opendrive_map':     'carla_opendrive_map', # HD map in OpenDRIVE format
    'sensor.speedometer':       'carla_speedometer',   # Vehicle speedometer
    'sensor.camera.semantic_segmentation': 'carla_camera', # Semantic segmentation (datagen)
    'sensor.camera.depth':      'carla_camera',        # Depth camera (datagen)
}

class LeaderboardEvaluator(object):
    """Main orchestrator for local CARLA Leaderboard evaluation.

    This is the local evaluation version with enhanced features for development and
    data generation. Key differences from the standard evaluator:

    1. Automatic port detection: Finds free ports to avoid conflicts
    2. Data generation support: Handles DATAGEN environment variable
    3. Route-specific naming: Creates unique identifiers for each route run
    4. Enhanced cleanup: Passes results to agent destroy method
    5. Local managers: Uses local versions of Scenario and Statistics managers

    The class manages the complete evaluation pipeline with the same core functionality
    as the standard evaluator but with optimizations for batch processing and local
    development workflows.

    Attributes:
        client_timeout (float): Maximum time in seconds for CARLA client operations
        frame_rate (float): Simulation update frequency in Hz (20 Hz default)
        world: CARLA world object reference
        manager: Local ScenarioManager instance
        sensors: Agent's sensor configuration
        agent_instance: Instantiated autonomous agent
        route_scenario: Current RouteScenario being executed
        statistics_manager: Local StatisticsManager for metrics
        traffic_manager_port: Dynamically allocated traffic manager port
    """

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
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
        self.client, self.client_timeout, self.traffic_manager, self.traffic_manager_port = self._setup_simulation(args)

        # Load agent
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

    def _cleanup(self, results=None):
    # def _cleanup(self):
        """
        Remove and destroy all actors
        """
        CarlaDataProvider.cleanup()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        try:
            if self.agent_instance:
                self.agent_instance.destroy(results)
                del self.agent_instance
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


    def find_free_port(self, start_port=2_000, end_port=40_000):
        """Find an available network port for the traffic manager.

        Scans through a range of ports to find one that is not currently in use.
        This is useful for running multiple evaluations in parallel or avoiding
        conflicts with other services.

        The method attempts to bind to each port in sequence. If binding succeeds,
        the port is available and returned. If binding fails with OSError (port
        already in use), it tries the next port.

        Args:
            start_port (int, optional): First port to try. Defaults to 2000.
            end_port (int, optional): Last port to try. Defaults to 40000.

        Returns:
            int or None: First available port number in the range, or None if
                        no ports are available in the entire range.

        Note:
            The socket is configured with SO_REUSEADDR to handle TIME_WAIT states.
            This prevents issues with ports that were recently released.
        """
        for port in range(start_port, end_port + 1):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            try:
                s.bind(('localhost', port))
                return port
            except OSError as e:  # Address already in use (Linux/Windows)
                pass
            finally:
                s.close()

        return None

    def _setup_simulation(self, args):
        """Prepare and configure the CARLA simulation environment with dynamic port allocation.

        Similar to the standard evaluator but with automatic port detection for the
        traffic manager. This enables running multiple evaluation instances in parallel
        without port conflicts.

        Args:
            args: Command-line arguments containing:
                - host: CARLA server hostname or IP
                - port: CARLA server port
                - timeout: Client timeout value in seconds

        Returns:
            tuple: (client, client_timeout, traffic_manager, traffic_manager_port)
                - client: Connected CARLA client instance
                - client_timeout: Configured timeout value
                - traffic_manager: Configured TrafficManager instance
                - traffic_manager_port: Dynamically allocated port number

        Note:
            The traffic manager port is found dynamically using find_free_port(),
            rather than using a fixed port from arguments.
        """
        client = carla.Client(args.host, args.port)
        if args.timeout:
            client_timeout = args.timeout
        client.set_timeout(client_timeout)

        # Configure synchronous mode for deterministic simulation
        settings = carla.WorldSettings(
            synchronous_mode = True,
            fixed_delta_seconds = 1.0 / self.frame_rate,
            deterministic_ragdolls = True,
            spectator_as_ego = False
        )
        client.get_world().apply_settings(settings)

        # Find an available port dynamically to avoid conflicts
        traffic_manager_port = self.find_free_port()
        traffic_manager = client.get_trafficmanager(traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)

        return client, client_timeout, traffic_manager, traffic_manager_port

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
        """
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """

        self.world = self.client.load_world(town, reset_settings=False)

        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(self.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_random_seed(args.traffic_manager_seed)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)

        # Wait for the world to be ready
        self.world.tick()

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            " This scenario requires the use of map {}".format(town))

    def _register_statistics(self, route_date_string, route_index, entry_status, crash_message=""):
        """
        Computes and saves the route statistics
        """
        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_entry_status(entry_status)
        current_stats_record = self.statistics_manager.compute_route_statistics(
            route_date_string, route_index, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message
        )
        return current_stats_record

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
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
            # Create a unique identifier for this route run with timestamp
            # Format: <routes_filename>_route<index>_MM_DD_HH_MM_SS
            now = datetime.now()
            route_string = pathlib.Path(args.routes).stem + '_'
            route_string += f'route{config.index}'
            route_date_string = route_string + '_' + '_'.join(
                map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second))
            )

            # Start watchdog timer to detect if agent setup hangs
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

            # Create agent instance with different parameters based on mode
            # DATAGEN mode: pass route index for data collection batches
            # Normal mode: pass unique route identifier with timestamp
            if int(os.environ.get('DATAGEN', 0))==1:
                self.agent_instance = agent_class_obj(args.agent_config, config.index)
            else:
                self.agent_instance = agent_class_obj(args.agent_config, route_date_string)

            # Provide the agent with the global route plan
            self.agent_instance.set_global_plan(self.route_scenario.gps_route, self.route_scenario.route)
            # Setup agent with config, route identifier, and traffic manager reference
            self.agent_instance.setup(args.agent_config, route_date_string, self.traffic_manager)

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons)
                self.statistics_manager.write_statistics()

                self.sensors_initialized = True

            self._agent_watchdog.stop()
            self._agent_watchdog = None

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print(f"{e}\033[0m\n")

            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)
            self._cleanup(result)
            return True

        except Exception:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)
            self._cleanup(result)
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
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup(result)

        except Exception:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print(f"\n{traceback.format_exc()}\033[0m")

            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # If the simulation crashed, stop the leaderboard, for the rest, move to the next route
        return crash_message == "Simulation crashed"

    def run(self, args):
        """
        Run the challenge mode
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
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--traffic-manager-port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=100, type=int,
                        help='Seed used by the TrafficManager (default: 100)')
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
    parser.add_argument('--resume', type=int, default=False,
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--debug-checkpoint", type=str, default='./live_results.txt',
                        help="Path to checkpoint used for saving live results")

    arguments = parser.parse_args()

    pathlib.Path(arguments.checkpoint).parent.mkdir(parents=True, exist_ok=True)

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
