#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""CARLA Scenario Runner - Main Entry Point

This module provides the main entry point for executing CARLA scenarios. It orchestrates
the entire scenario lifecycle from initialization to cleanup, supporting multiple scenario
types and execution modes.

Key Features:
    - Supports standard Python-based scenarios
    - Compatible with OpenSCENARIO format (ASAM standard)
    - Route-based scenario execution for autonomous driving benchmarks
    - Autonomous agent integration for testing driving policies
    - Synchronous and asynchronous simulation modes
    - Comprehensive result reporting (stdout, file, JUnit, JSON)
    - CARLA recorder integration for playback and analysis

Typical Usage:
    Execute a standard scenario:
        $ python scenario_runner.py --scenario FollowLeadingVehicle_1 --reloadWorld

    Run a route-based scenario with an agent:
        $ python scenario_runner.py --route routes.xml --agent my_agent.py --sync

    Execute an OpenSCENARIO file:
        $ python scenario_runner.py --openscenario example.xosc

Architecture:
    The ScenarioRunner class manages the complete scenario execution pipeline:
    1. CARLA client/server connection and version validation
    2. World loading and synchronization setup
    3. Ego vehicle spawning/attachment
    4. Scenario instantiation from configuration
    5. ScenarioManager coordination for scenario execution
    6. Result analysis and reporting
    7. Cleanup and resource deallocation

Classes:
    ScenarioRunner: Main orchestrator for scenario execution and management

Functions:
    main: Command-line interface and program entry point
"""

from __future__ import print_function

import glob
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version # Python 2 fallback
import importlib
import inspect
import os
import signal
import sys
import time
import json

try:
    # requires Python 3.8+
    from importlib.metadata import metadata
    def get_carla_version():
        return Version(metadata("carla")["Version"])
except ModuleNotFoundError:
    # backport checking for older Python versions; module is deprecated
    import pkg_resources
    def get_carla_version():
        return Version(pkg_resources.get_distribution("carla").version)

import carla

from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser

# Version of scenario_runner
VERSION = '0.9.13'

# Minimum version of CARLA that is required
MIN_CARLA_VERSION = '0.9.12'


class ScenarioRunner(object):
    """Core scenario execution orchestrator for CARLA simulation.

    This class manages the complete lifecycle of scenario execution, from CARLA
    connection setup through scenario instantiation, execution, and cleanup. It
    serves as the primary interface between the user's scenario definitions and
    the CARLA simulator.

    The ScenarioRunner handles:
        - CARLA client/server connection management
        - World loading and simulation mode configuration (sync/async)
        - Ego vehicle spawning or attachment to existing vehicles
        - Scenario type detection and instantiation (standard/route/OpenSCENARIO)
        - Autonomous agent integration and lifecycle management
        - Traffic manager configuration and seed control
        - Result collection, analysis, and export
        - Graceful cleanup and resource deallocation

    Attributes:
        ego_vehicles (list): List of ego vehicle actors controlled by the agent/scenario
        client_timeout (float): CARLA client connection timeout in seconds (default: 10.0)
        wait_for_world (float): Maximum time to wait for world readiness in seconds (default: 20.0)
        frame_rate (float): Simulation frame rate in Hz for synchronous mode (default: 20.0)
        world (carla.World): Active CARLA world instance
        manager (ScenarioManager): Scenario execution manager handling behavior trees
        finished (bool): Flag indicating scenario completion status
        agent_instance (AutonomousAgent): Instance of the autonomous agent being tested
        module_agent (module): Imported Python module containing agent implementation

    Example:
        >>> args = parse_arguments()
        >>> runner = ScenarioRunner(args)
        >>> success = runner.run()
        >>> runner.destroy()
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds - Maximum time to wait for CARLA server responses
    wait_for_world = 20.0  # in seconds - Maximum time to wait for world initialization
    frame_rate = 20.0      # in Hz - Fixed timestep for synchronous simulation mode

    # CARLA world and scenario handlers
    world = None
    manager = None

    finished = False

    additional_scenario_module = None

    agent_instance = None
    module_agent = None

    def __init__(self, args):
        """Initialize the ScenarioRunner with CARLA client and configuration.

        Sets up the connection to the CARLA server, validates the CARLA version,
        loads the autonomous agent module (if specified), creates the scenario
        manager, and registers signal handlers for graceful shutdown.

        Args:
            args (argparse.Namespace): Parsed command-line arguments containing:
                - host (str): CARLA server hostname/IP
                - port (int): CARLA server port
                - timeout (float): Client timeout in seconds
                - agent (str): Path to autonomous agent Python file
                - debug (bool): Enable debug output
                - sync (bool): Use synchronous simulation mode

        Raises:
            ImportError: If CARLA version is older than MIN_CARLA_VERSION
            ImportError: If agent module cannot be loaded

        Note:
            Signal handlers are registered for SIGINT, SIGTERM, and SIGHUP (Unix only)
            to enable graceful shutdown and cleanup.
        """
        self._args = args

        if args.timeout:
            self.client_timeout = float(args.timeout)

        # Create CARLA client connection
        # The client sends requests to the simulator server
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)

        # Validate CARLA version compatibility
        carla_version = get_carla_version()
        if carla_version < Version(MIN_CARLA_VERSION):
            raise ImportError("CARLA version {} or newer required. CARLA version found: {}".format(MIN_CARLA_VERSION, carla_version))

        # Load autonomous agent module if specified
        # The agent implements the driving policy to be tested
        if self._args.agent is not None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager to handle scenario execution
        self.manager = ScenarioManager(self._args.debug, self._args.sync, self._args.timeout)

        # Register signal handlers for graceful shutdown
        self._shutdown_requested = False
        if sys.platform != 'win32':
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._start_wall_time = datetime.now()

    def destroy(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world
        if self.client is not None:
            del self.client

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown.

        Called when the process receives SIGINT (Ctrl+C), SIGTERM, or SIGHUP signals.
        Initiates a graceful shutdown by stopping the scenario and setting shutdown flag.

        Args:
            signum (int): Signal number received
            frame: Current stack frame (unused)
        """
        self._shutdown_requested = True
        if self.manager:
            self.manager.stop_scenario()

    def _get_scenario_class_or_fail(self, scenario):
        """Dynamically load and return scenario class by name.

        Searches through all Python files in the scenarios directory and the
        additional scenario file (if provided) to find a class matching the
        scenario name. Uses dynamic import and introspection.

        Args:
            scenario (str): Name of the scenario class to load

        Returns:
            class: The scenario class matching the given name

        Raises:
            SystemExit: If scenario class is not found (exits with code -1)

        Note:
            Searches in srunner/scenarios/*.py and --additionalScenario file
        """

        # Collect all scenario Python files from standard location and additional scenarios
        scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        scenarios_list.append(self._args.additionalScenario)

        for scenario_file in scenarios_list:

            # Import the scenario module dynamically
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            scenario_module = importlib.import_module(module_name)

            # Search through all classes in the module
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                if scenario in member:
                    return member[1]

            # Clean up sys.path to avoid pollution
            sys.path.pop(0)

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def _cleanup(self):
        """Clean up and destroy all scenario actors and reset simulation state.

        Performs comprehensive cleanup including:
        - Resetting synchronous mode to asynchronous (if applicable)
        - Cleaning up scenario manager resources
        - Cleaning up CarlaDataProvider singleton state
        - Destroying ego vehicles (unless waitForEgo mode is active)
        - Destroying autonomous agent instances

        Note:
            This method is idempotent - calling it multiple times is safe.
            In waitForEgo mode, ego vehicles are NOT destroyed as they're
            externally managed.
        """
        if self.finished:
            return

        self.finished = True

        # Reset synchronous mode if it was enabled
        # This ensures CARLA returns to normal asynchronous operation
        if self.world is not None and self._args.sync:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                self.client.get_trafficmanager(int(self._args.trafficManagerPort)).set_synchronous_mode(False)
            except RuntimeError:
                sys.exit(-1)

        # Clean up scenario manager resources
        self.manager.cleanup()

        # Clean up global data provider state
        CarlaDataProvider.cleanup()

        # Destroy ego vehicles (unless externally managed)
        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if not self._args.waitForEgo and self.ego_vehicles[i] is not None and self.ego_vehicles[i].is_alive:
                    print("Destroying ego vehicle {}".format(self.ego_vehicles[i].id))
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        # Clean up agent instance
        if self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles):
        """Spawn new ego vehicles or attach to existing ones.

        Two modes of operation:
        1. Normal mode (--waitForEgo not set): Spawns new ego vehicles at specified locations
        2. Wait mode (--waitForEgo set): Searches for existing vehicles with matching role names
           and attaches to them, waiting if necessary until they appear

        Args:
            ego_vehicles (list): List of ActorConfiguration objects specifying ego vehicle
                                 properties (model, transform, rolename, color, category)

        Note:
            In wait mode, vehicles are reset to specified transform and zero velocity.
            After preparation, performs a world tick/wait to synchronize state.
        """

        if not self._args.waitForEgo:
            # Spawn new ego vehicles
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             actor_category=vehicle.category))
        else:
            # Wait for and attach to existing ego vehicles
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            # Reset ego vehicle state to scenario starting conditions
            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)
                self.ego_vehicles[i].set_target_velocity(carla.Vector3D())
                self.ego_vehicles[i].set_target_angular_velocity(carla.Vector3D())
                self.ego_vehicles[i].apply_control(carla.VehicleControl())
                CarlaDataProvider.register_actor(self.ego_vehicles[i], ego_vehicles[i].transform)

        # Synchronize world state
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _analyze_scenario(self, config):
        """
        Provide feedback about success/failure of a scenario
        """

        # Create the filename
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        json_filename = None
        config_name = config.name
        if self._args.outputDir != '':
            config_name = os.path.join(self._args.outputDir, config_name)

        if self._args.junit:
            junit_filename = config_name + current_time + ".xml"
        if self._args.json:
            json_filename = config_name + current_time + ".json"
        filename = None
        if self._args.file:
            filename = config_name + current_time + ".txt"

        if not self.manager.analyze_scenario(self._args.output, filename, junit_filename, json_filename):
            print("All scenario tests were passed successfully!")
        else:
            print("Not all scenario tests were successful")
            if not (self._args.output or filename or junit_filename):
                print("Please run with --output for further information")

    def _record_criteria(self, criteria, name):
        """
        Filter the JSON serializable attributes of the criterias and
        dumps them into a file. This will be used by the metrics manager,
        in case the user wants specific information about the criterias.
        """
        file_name = name[:-4] + ".json"

        # Filter the attributes that aren't JSON serializable
        with open('temp.json', 'w', encoding='utf-8') as fp:

            criteria_dict = {}
            for criterion in criteria:

                criterion_dict = criterion.__dict__
                criteria_dict[criterion.name] = {}

                for key in criterion_dict:
                    if key != "name":
                        try:
                            key_dict = {key: criterion_dict[key]}
                            json.dump(key_dict, fp, sort_keys=False, indent=4)
                            criteria_dict[criterion.name].update(key_dict)
                        except TypeError:
                            pass

        os.remove('temp.json')

        # Save the criteria dictionary into a .json file
        with open(file_name, 'w', encoding='utf-8') as fp:
            json.dump(criteria_dict, fp, sort_keys=False, indent=4)

    def _load_and_wait_for_world(self, town, ego_vehicles=None):
        """Load CARLA world and configure simulation mode.

        Either loads a new world (if --reloadWorld is set) or uses the existing world.
        Configures synchronous mode if requested and initializes the CarlaDataProvider
        singleton with world references.

        Args:
            town (str): Name of the CARLA town/map to load (e.g., 'Town01', 'Town02')
            ego_vehicles (list, optional): List of ego vehicle configurations for wait mode

        Returns:
            bool: True if world loaded successfully and map matches requirements,
                  False if map mismatch detected

        Note:
            - In reloadWorld mode: Loads the specified map from scratch
            - In waitForEgo mode: Waits for ego vehicles to appear before proceeding
            - Synchronous mode is configured with fixed_delta_seconds based on frame_rate
            - CarlaDataProvider is initialized as the global data access point
        """

        if self._args.reloadWorld:
            # Load a fresh instance of the requested map
            self.world = self.client.load_world(town)
        else:
            # Use existing world, but wait for ego vehicles if required
            ego_vehicle_found = False
            if self._args.waitForEgo:
                while not ego_vehicle_found and not self._shutdown_requested:
                    vehicles = self.client.get_world().get_actors().filter('vehicle.*')
                    for ego_vehicle in ego_vehicles:
                        ego_vehicle_found = False
                        for vehicle in vehicles:
                            if vehicle.attributes['role_name'] == ego_vehicle.rolename:
                                ego_vehicle_found = True
                                break
                        if not ego_vehicle_found:
                            print("Not all ego vehicles ready. Waiting ... ")
                            time.sleep(1)
                            break

        self.world = self.client.get_world()

        # Configure synchronous mode if requested
        # In sync mode, simulation only advances when world.tick() is called
        if self._args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            self.world.apply_settings(settings)

        # Initialize global data provider with world references
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        # Perform initial tick to ensure world is ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Validate that the loaded map matches scenario requirements
        map_name = CarlaDataProvider.get_map().name.split('/')[-1]
        if map_name not in (town, "OpenDriveMap"):
            print("The CARLA server uses the wrong map: {}".format(map_name))
            print("This scenario requires to use map: {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, config):
        """Execute a complete scenario from initialization to cleanup.

        This is the main scenario execution pipeline that:
        1. Loads the CARLA world and validates map compatibility
        2. Instantiates the autonomous agent (if specified)
        3. Configures traffic manager with deterministic seed
        4. Prepares ego vehicles (spawn or attach)
        5. Instantiates the appropriate scenario type (OpenSCENARIO/Route/Standard)
        6. Executes the scenario via ScenarioManager
        7. Analyzes and reports results
        8. Cleans up all actors and resources

        Args:
            config (ScenarioConfiguration): Configuration object containing:
                - town: Map name to load
                - ego_vehicles: List of ego vehicle configurations
                - name: Scenario identifier
                - type: Scenario class name (for standard scenarios)

        Returns:
            bool: True if scenario completed successfully, False on any error

        Note:
            Supports three scenario types:
            - OpenSCENARIO (.xosc files using ASAM standard)
            - Route scenarios (XML-defined waypoint routes)
            - Standard Python scenarios (custom scenario classes)

            If --record is enabled, saves full simulation log and criteria data.
        """
        result = False

        # Load the world and validate map compatibility
        if not self._load_and_wait_for_world(config.town, config.ego_vehicles):
            self._cleanup()
            return False

        # Instantiate autonomous agent if specified
        if self._args.agent:
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            try:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig)
                config.agent = self.agent_instance
            except Exception as e:          # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return False

        # Configure traffic manager for background vehicles
        CarlaDataProvider.set_traffic_manager_port(int(self._args.trafficManagerPort))
        tm = self.client.get_trafficmanager(int(self._args.trafficManagerPort))
        tm.set_random_device_seed(int(self._args.trafficManagerSeed))  # Deterministic behavior
        if self._args.sync:
            tm.set_synchronous_mode(True)

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            # Spawn or attach ego vehicles
            self._prepare_ego_vehicles(config.ego_vehicles)

            # Instantiate scenario based on type
            if self._args.openscenario:
                # OpenSCENARIO: ASAM standard scenario format
                scenario = OpenScenario(world=self.world,
                                        ego_vehicles=self.ego_vehicles,
                                        config=config,
                                        config_file=self._args.openscenario,
                                        timeout=100000)
            elif self._args.route:
                # Route scenario: Waypoint-based navigation challenge
                scenario = RouteScenario(world=self.world,
                                         config=config,
                                         debug_mode=self._args.debug)
            else:
                # Standard scenario: Python-defined scenario class
                scenario_class = self._get_scenario_class_or_fail(config.type)
                scenario = scenario_class(world=self.world,
                                          ego_vehicles=self.ego_vehicles,
                                          config=config,
                                          randomize=self._args.randomize,
                                          debug_mode=self._args.debug)
        except Exception as exception:                  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return False

        try:
            # Start CARLA recorder if requested
            if self._args.record:
                recorder_name = "{}/{}/{}.log".format(
                    os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.record, config.name)
                self.client.start_recorder(recorder_name, True)

            # Execute scenario via behavior tree
            self.manager.load_scenario(scenario, self.agent_instance)
            self.manager.run_scenario()

            # Generate result reports
            self._analyze_scenario(config)

            # Cleanup actors and stop recording
            scenario.remove_all_actors()
            if self._args.record:
                self.client.stop_recorder()
                self._record_criteria(self.manager.scenario.get_criteria(), recorder_name)

            result = True

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self._cleanup()
        return result

    def _run_scenarios(self):
        """
        Run conventional scenarios (e.g. implemented using the Python API of ScenarioRunner)
        """
        result = False

        # Load the scenario configurations provided in the config file
        scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(
            self._args.scenario,
            self._args.configFile)
        if not scenario_configurations:
            print("Configuration for scenario {} cannot be found!".format(self._args.scenario))
            return result

        # Execute each configuration
        for config in scenario_configurations:
            for _ in range(self._args.repetitions):
                self.finished = False
                result = self._load_and_run_scenario(config)

            self._cleanup()
        return result

    def _run_route(self):
        """
        Run the route scenario
        """
        result = False

        # retrieve routes
        route_configurations = RouteParser.parse_routes_file(self._args.route, self._args.route_id)

        for config in route_configurations:
            for _ in range(self._args.repetitions):
                result = self._load_and_run_scenario(config)

                self._cleanup()
        return result

    def _run_openscenario(self):
        """
        Run a scenario based on OpenSCENARIO
        """

        # Load the scenario configurations provided in the config file
        if not os.path.isfile(self._args.openscenario):
            print("File does not exist")
            self._cleanup()
            return False

        openscenario_params = {}
        if self._args.openscenarioparams is not None:
            for entry in self._args.openscenarioparams.split(','):
                [key, val] = [m.strip() for m in entry.split(':')]
                openscenario_params[key] = val
        config = OpenScenarioConfiguration(self._args.openscenario, self.client, openscenario_params)

        result = self._load_and_run_scenario(config)
        self._cleanup()
        return result

    def run(self):
        """
        Run all scenarios according to provided commandline args
        """
        result = True
        if self._args.openscenario:
            result = self._run_openscenario()
        elif self._args.route:
            result = self._run_route()
        else:
            result = self._run_scenarios()

        print("No more scenarios .... Exiting")
        return result


def main():
    """
    main function
    """
    description = ("CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n"
                   "Current version: " + VERSION)

    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--timeout', default="10.0",
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--sync', action='store_true',
                        help='Forces the simulation to run synchronously')
    parser.add_argument('--list', action="store_true", help='List all supported scenarios and exit')

    parser.add_argument(
        '--scenario', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    parser.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')
    parser.add_argument('--openscenarioparams', help='Overwrited for OpenSCENARIO ParameterDeclaration')
    parser.add_argument('--route', help='Run a route as a scenario', type=str)
    parser.add_argument('--route-id', help='Run a specific route inside that \'route\' file', default='', type=str)
    parser.add_argument(
        '--agent', help="Agent used to execute the route. Not compatible with non-route-based scenarios.")
    parser.add_argument('--agentConfig', type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument('--output', action="store_true", help='Provide results on stdout')
    parser.add_argument('--file', action="store_true", help='Write results into a txt file')
    parser.add_argument('--junit', action="store_true", help='Write results into a junit file')
    parser.add_argument('--json', action="store_true", help='Write results into a JSON file')
    parser.add_argument('--outputDir', default='', help='Directory for output files (default: this directory)')

    parser.add_argument('--configFile', default='', help='Provide an additional scenario configuration file (*.xml)')
    parser.add_argument('--additionalScenario', default='', help='Provide additional scenario implementations (*.py)')

    parser.add_argument('--debug', action="store_true", help='Run with debug output')
    parser.add_argument('--reloadWorld', action="store_true",
                        help='Reload the CARLA world before starting a scenario (default=True)')
    parser.add_argument('--record', type=str, default='',
                        help='Path were the files will be saved, relative to SCENARIO_RUNNER_ROOT.\nActivates the CARLA recording feature and saves to file all the criteria information.')
    parser.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    parser.add_argument('--repetitions', default=1, type=int, help='Number of scenario executions')
    parser.add_argument('--waitForEgo', action="store_true", help='Connect the scenario to an existing ego vehicle')

    arguments = parser.parse_args()
    # pylint: enable=line-too-long

    if arguments.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(arguments.configFile), sep='\n')
        return 1

    if not arguments.scenario and not arguments.openscenario and not arguments.route:
        print("Please specify either a scenario or use the route mode\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.route and (arguments.openscenario or arguments.scenario):
        print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.agent and (arguments.openscenario or arguments.scenario):
        print("Agents are currently only compatible with route scenarios'\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.openscenarioparams and not arguments.openscenario:
        print("WARN: Ignoring --openscenarioparams when --openscenario is not specified")

    if arguments.route:
        arguments.reloadWorld = True

    if arguments.agent:
        arguments.sync = True

    scenario_runner = None
    result = True
    try:
        scenario_runner = ScenarioRunner(arguments)
        result = scenario_runner.run()
    except Exception:   # pylint: disable=broad-except
        traceback.print_exc()

    finally:
        if scenario_runner is not None:
            scenario_runner.destroy()
            del scenario_runner
    return not result


if __name__ == "__main__":
    sys.exit(main())
