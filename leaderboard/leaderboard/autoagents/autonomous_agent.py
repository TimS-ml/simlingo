#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Base class for autonomous driving agents in the CARLA Leaderboard.

This module defines the abstract base class that all user-submitted autonomous agents
must inherit from. It provides the interface contract between the leaderboard evaluation
system and custom agent implementations.

The AutonomousAgent class handles:
- Sensor configuration and data collection
- Route planning and waypoint management
- Control command generation
- Agent lifecycle management (setup/destroy)
- Performance timing and statistics

Users must implement their own derived classes that override key methods like:
- setup(): Initialize the agent (load models, configure parameters)
- sensors(): Define required sensor suite (cameras, LiDAR, etc.)
- run_step(): Core decision-making logic that returns vehicle controls

Example:
    class MyAgent(AutonomousAgent):
        def setup(self, path_to_conf_file):
            # Load your model, initialize components
            pass

        def sensors(self):
            # Return list of required sensors
            return [{'type': 'sensor.camera.rgb', ...}]

        def run_step(self, input_data, timestamp):
            # Process sensors and return control commands
            control = carla.VehicleControl()
            # ... your logic here ...
            return control
"""

from __future__ import print_function

from enum import Enum

import carla
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface


class Track(Enum):
    """Competition tracks for the CARLA Autonomous Driving Leaderboard.

    This enumeration defines the different competition tracks, each with different
    constraints on which sensors and information sources agents can use:

    Attributes:
        SENSORS: Standard track allowing only physical sensors (cameras, LiDAR, RADAR,
                GPS, IMU, speedometer). No access to HD map or privileged information.

        MAP: Advanced track that additionally allows access to OpenDRIVE HD maps,
            providing detailed lane geometry and topology information.

        SENSORS_QUALIFIER: Qualifier round for the SENSORS track, may have different
                          evaluation criteria or route sets.

        MAP_QUALIFIER: Qualifier round for the MAP track, may have different
                      evaluation criteria or route sets.

    The track selection affects which sensor configurations are valid during evaluation.
    Agents must declare their track in the __init__ method via self.track.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'
    SENSORS_QUALIFIER = 'SENSORS_QUALIFIER'
    MAP_QUALIFIER = 'MAP_QUALIFIER'


class AutonomousAgent(object):
    """Abstract base class for autonomous driving agents.

    This class defines the interface that all custom agent implementations must follow.
    It provides the infrastructure for:

    1. Sensor Management: Declares required sensors and receives their data
    2. Route Following: Receives global route plans and waypoints
    3. Control Generation: Produces steering/throttle/brake commands
    4. Timing Metrics: Tracks real-time vs. simulation time ratios

    Subclasses must implement:
        - sensors(): Return list of required sensor configurations
        - run_step(): Core decision-making logic
        - setup(): Agent initialization (models, parameters, etc.)

    Optional overrides:
        - destroy(): Custom cleanup logic
        - set_global_plan(): Custom route processing

    Attributes:
        track (Track): Competition track (SENSORS or MAP)
        sensor_interface (SensorInterface): Manages sensor data collection
        _global_plan: GPS coordinates of route waypoints (downsampled)
        _global_plan_world_coord: CARLA world coordinates of waypoints
        wallclock_t0: Reference wallclock time for performance tracking
    """

    def __init__(self, carla_host, carla_port, debug=False):
        """Initialize the autonomous agent.

        Args:
            carla_host (str): CARLA server hostname or IP address
            carla_port (int): CARLA server port number
            debug (bool, optional): Enable debug output. Defaults to False.

        Note:
            Subclasses should override this to add custom initialization,
            but must call super().__init__() to initialize base components.
        """
        self.track = Track.SENSORS  # Default to SENSORS track
        # Current global route plan to reach the destination
        self._global_plan = None  # GPS coordinates
        self._global_plan_world_coord = None  # CARLA world coordinates

        # Sensor interface manages all sensor data collection and callbacks
        self.sensor_interface = SensorInterface()

        # Reference time for computing simulation speed metrics
        self.wallclock_t0 = None

    def setup(self, path_to_conf_file):
        """Initialize agent-specific components and configuration.

        This method is called once before the first route execution. Use it to:
        - Load neural network models and weights
        - Initialize planners, controllers, or other components
        - Parse configuration files
        - Set self.track to the appropriate Track enum value
        - Allocate buffers or data structures

        Args:
            path_to_conf_file (str): Path to agent configuration file (e.g., JSON, YAML).
                                    Can be empty if agent doesn't need configuration.

        Note:
            Must set self.track to indicate which competition track:
                - Track.SENSORS: Only physical sensors allowed
                - Track.MAP: Physical sensors + OpenDRIVE HD map allowed

            This method should NOT create the sensor actors - sensors are created
            automatically based on the sensors() method return value.
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """Define the sensor suite required by the agent.

        This method specifies which sensors to attach to the ego vehicle and their
        configurations. The leaderboard system will create these sensors and provide
        their data to run_step() via the input_data parameter.

        Returns:
            list: List of sensor configuration dictionaries. Each dictionary specifies:
                - type (str): Sensor type from CARLA (e.g., 'sensor.camera.rgb')
                - x, y, z (float): Position relative to vehicle center (meters)
                - roll, pitch, yaw (float): Orientation in degrees
                - id (str): Unique identifier for this sensor instance
                - Additional sensor-specific parameters (width, height, fov, etc.)

        Available sensor types:
            - sensor.camera.rgb: RGB camera
            - sensor.camera.semantic_segmentation: Semantic segmentation camera
            - sensor.lidar.ray_cast: LiDAR point cloud
            - sensor.other.radar: RADAR sensor
            - sensor.other.gnss: GPS receiver
            - sensor.other.imu: Inertial measurement unit
            - sensor.speedometer: Vehicle speedometer
            - sensor.opendrive_map: HD map (only for MAP track)

        Example:
            [
                {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60,
                 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'width': 300, 'height': 200,
                 'fov': 100, 'id': 'Left'},

                {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60,
                 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'id': 'LIDAR'}
            ]

        Note:
            Sensor configurations will be validated against the declared track.
            Requesting invalid sensors for your track will result in disqualification.
        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """Execute one step of navigation and return vehicle control commands.

        This is the core decision-making method called at every simulation step (20 Hz).
        It receives sensor data and must return control commands within the time budget.

        Args:
            input_data (dict): Dictionary mapping sensor IDs to sensor data.
                Keys are the 'id' strings from sensors() method.
                Values are tuples of (frame_number, sensor_data) where sensor_data
                type depends on the sensor (numpy array for cameras, point cloud for
                LiDAR, etc.)

            timestamp (float): Current simulation time in seconds since episode start.

        Returns:
            carla.VehicleControl: Vehicle control object with fields:
                - throttle (float): [0.0, 1.0] for acceleration
                - steer (float): [-1.0, 1.0] for steering (- left, + right)
                - brake (float): [0.0, 1.0] for braking
                - hand_brake (bool): Emergency handbrake
                - reverse (bool): Reverse gear
                - manual_gear_shift (bool): Manual gear control (set to False)
                - gear (int): Gear number if manual_gear_shift is True

        Note:
            This method must execute within the time budget to avoid penalties.
            The default implementation returns zero control (vehicle remains stationary).
            Agents should override this with their decision-making logic.
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """Clean up agent resources before termination.

        Called once after all routes have been completed or when evaluation is stopped.
        Use this method to:
        - Close open files or databases
        - Save final results or logs
        - Release GPU memory
        - Stop background threads
        - Clean up temporary files

        Returns:
            None

        Note:
            The base implementation does nothing. Override if your agent needs cleanup.
            This is called even if the agent crashes, so implement defensively.
        """
        pass

    def __call__(self, sensors=None):
        """Execute agent decision-making for one simulation step (callable interface).

        This method is called by the leaderboard framework at each simulation tick.
        It handles timing metrics, retrieves sensor data, calls run_step(), and
        ensures proper control formatting.

        The method automatically:
        - Retrieves sensor data for the current frame
        - Tracks wallclock vs simulation time ratios
        - Prints performance statistics
        - Ensures manual_gear_shift is disabled

        Args:
            sensors: Unused parameter (kept for compatibility)

        Returns:
            carla.VehicleControl: Vehicle control commands with manual_gear_shift=False
        """
        # Retrieve sensor data for the current simulation frame
        input_data = self.sensor_interface.get_data(GameTime.get_frame())

        # Get current simulation timestamp
        timestamp = GameTime.get_time()

        # Initialize wallclock reference time on first call
        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()

        # Calculate performance metrics: real-time vs simulation time ratio
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp/wallclock_diff

        # Print timing statistics (ratio > 1.0 means faster than real-time)
        print('=== [Agent] -- Wallclock = {} -- System time = {} -- Game time = {} -- Ratio = {}x'.format(
            str(wallclock)[:-3], format(wallclock_diff, '.3f'), format(timestamp, '.3f'), format(sim_ratio, '.3f')))

        # Execute agent's decision-making logic
        control = self.run_step(input_data, timestamp)

        # Disable manual gear shifting (automatic transmission only)
        control.manual_gear_shift = False

        return control

    @staticmethod
    def get_ros_version():
        """Indicate which ROS version the agent uses (if any).

        This static method allows the leaderboard to identify ROS-based agents
        and start the appropriate bridge server.

        Returns:
            int: ROS version indicator:
                -1: No ROS (default for Python agents)
                 1: ROS1 (Robot Operating System 1)
                 2: ROS2 (Robot Operating System 2)

        Note:
            Override this method in ROS-based agent subclasses to return 1 or 2.
            The default implementation returns -1 (no ROS).
        """
        return -1

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Set the global route plan for the agent to follow.

        Called once per route before execution starts. The route is provided in both
        GPS and CARLA world coordinate formats. The base implementation downsamples
        the route to reduce memory usage while maintaining route fidelity.

        Args:
            global_plan_gps (list): List of (lat, lon) GPS waypoint tuples defining
                the route in geographic coordinates.

            global_plan_world_coord (list): List of (carla.Location, road_option) tuples
                defining the route in CARLA world coordinates (meters). road_option
                indicates maneuvers like STRAIGHT, LEFT, RIGHT, etc.

        Note:
            The base implementation downsamples waypoints to ~200 meters spacing using
            downsample_route(). Override this method if you need:
            - Denser waypoints for precise trajectory following
            - Custom waypoint preprocessing
            - Alternative route representations

            Downsampled routes are stored in:
            - self._global_plan: GPS coordinates
            - self._global_plan_world_coord: World coordinates
        """
        # Downsample route to approximately 200m spacing between waypoints
        ds_ids = downsample_route(global_plan_world_coord, 200)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
