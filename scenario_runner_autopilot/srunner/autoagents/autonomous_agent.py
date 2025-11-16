#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""AutonomousAgent - Base class for implementing autonomous driving agents.

This module defines the AutonomousAgent interface that all user-implemented driving
agents must follow. The agent receives sensor data each timestep and returns vehicle
control commands to navigate the scenario route.

Agent Lifecycle:
    1. Instantiation: Agent.__init__(path_to_config) loads configuration
    2. Sensor Setup: sensors() defines required sensor suite (cameras, lidar, etc.)
    3. Sensor Attachment: Framework attaches sensors to ego vehicle
    4. Route Setting: set_global_plan() provides the global navigation route
    5. Execution Loop: __call__() invoked each tick to get control commands
       - Sensor data collected automatically via SensorInterface
       - run_step() processes data and computes controls
    6. Cleanup: destroy() releases resources when scenario ends

Sensor Data:
    Sensor data is collected automatically by the SensorInterface and provided to
    run_step() as a dictionary:
        {
            'sensor_id': {
                'data': sensor_data (e.g., numpy array for images/lidar),
                'timestamp': carla.Timestamp
            },
            ...
        }

    Common sensor types:
        - sensor.camera.rgb: RGB camera image
        - sensor.camera.depth: Depth map
        - sensor.camera.semantic_segmentation: Semantic segmentation
        - sensor.lidar.ray_cast: 3D point cloud
        - sensor.other.gnss: GPS coordinates
        - sensor.other.imu: Inertial measurement
        - sensor.speedometer: Vehicle speed

Control Output:
    run_step() must return a carla.VehicleControl with:
        - throttle (float [0, 1]): Acceleration
        - steer (float [-1, 1]): Steering angle
        - brake (float [0, 1]): Brake intensity
        - hand_brake (bool): Emergency brake
        - reverse (bool): Reverse gear
        - manual_gear_shift (bool): Manual transmission control

Global Plan:
    The route is provided as two representations:
    1. GPS coordinates: List of (lat, lon) tuples
    2. World coordinates: List of (waypoint, RoadOption) tuples
       - waypoint: carla.Waypoint with transform and lane info
       - RoadOption: Enum (LANE_FOLLOW, LEFT, RIGHT, STRAIGHT, CHANGE_LEFT, etc.)

Example Implementation:
    >>> class MyAgent(AutonomousAgent):
    ...     def setup(self, config_path):
    ...         self.model = load_model(config_path)
    ...
    ...     def sensors(self):
    ...         return [
    ...             {'type': 'sensor.camera.rgb', 'id': 'Center',
    ...              'x': 0.7, 'y': 0.0, 'z': 1.60, 'width': 800, 'height': 600},
    ...             {'type': 'sensor.lidar.ray_cast', 'id': 'LIDAR',
    ...              'x': 0.0, 'y': 0.0, 'z': 2.5}
    ...         ]
    ...
    ...     def run_step(self, input_data, timestamp):
    ...         rgb = input_data['Center']['data']
    ...         lidar = input_data['LIDAR']['data']
    ...         control = self.model.predict(rgb, lidar, self._global_plan)
    ...         return control

See Also:
    - SensorInterface: Manages sensor data collection
    - RouteScenario: Provides routes for agent navigation
    - AgentWrapper: Internal wrapper used by ScenarioManager
"""

from __future__ import print_function

import carla

from srunner.autoagents.sensor_interface import SensorInterface
from srunner.scenariomanager.timer import GameTime
from srunner.tools.route_manipulation import downsample_route


class AutonomousAgent(object):
    """Abstract base class for autonomous driving agents.

    All user-implemented driving agents must inherit from this class and implement
    the required methods: setup(), sensors(), and run_step(). The agent receives
    sensor data each tick and returns vehicle control commands.

    Attributes:
        sensor_interface (SensorInterface): Manages sensor data collection
        _global_plan (list): GPS route as list of (lat, lon) tuples
        _global_plan_world_coord (list): Route as list of (waypoint, RoadOption) tuples

    Required Methods to Implement:
        sensors(): Return list of sensor specifications
        run_step(input_data, timestamp): Compute control from sensor data

    Optional Methods to Override:
        setup(path_to_conf_file): Custom initialization logic
        destroy(): Custom cleanup logic

    Note:
        Do not override __call__() - it manages the sensor interface automatically.
    """

    def __init__(self, path_to_conf_file):
        """Initialize agent with configuration file.

        Args:
            path_to_conf_file (str): Path to agent configuration file (JSON, YAML, etc.)

        Note:
            Automatically calls setup(path_to_conf_file) for custom initialization.
        """
        # Global route for navigation
        self._global_plan = None
        self._global_plan_world_coord = None

        # Sensor data interface
        self.sensor_interface = SensorInterface()

        # Custom agent initialization
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        """Initialize agent-specific resources and configuration.

        Override this method to load models, parameters, planners, etc.

        Args:
            path_to_conf_file (str): Path to configuration file

        Example:
            >>> def setup(self, path_to_conf_file):
            ...     with open(path_to_conf_file) as f:
            ...         self.config = json.load(f)
            ...     self.model = torch.load(self.config['model_path'])
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """Define the sensor suite to attach to the ego vehicle.

        Returns:
            list: Sensor specifications as dictionaries with keys:
                - type (str): Sensor type (e.g., 'sensor.camera.rgb')
                - id (str): Unique sensor identifier for data retrieval
                - x, y, z (float): Position relative to vehicle center (meters)
                - roll, pitch, yaw (float): Orientation in degrees
                - Additional sensor-specific parameters (width, height, fov, etc.)

        Example:
            >>> def sensors(self):
            ...     return [
            ...         {'type': 'sensor.camera.rgb', 'id': 'Center',
            ...          'x': 0.7, 'y': 0.0, 'z': 1.60,
            ...          'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            ...          'width': 800, 'height': 600, 'fov': 100},
            ...         {'type': 'sensor.lidar.ray_cast', 'id': 'LIDAR',
            ...          'x': 0.0, 'y': 0.0, 'z': 2.5,
            ...          'channels': 64, 'range': 50, 'points_per_second': 100000}
            ...     ]

        Note:
            Sensors are automatically attached and data is collected by SensorInterface.
        """
        sensors = []
        return sensors

    def run_step(self, input_data, timestamp):
        """Compute vehicle control commands from sensor data.

        This is the main method called each simulation tick. Implement your driving
        policy here.

        Args:
            input_data (dict): Sensor data dictionary with structure:
                {
                    'sensor_id': {
                        'data': sensor_data,  # numpy array or appropriate type
                        'timestamp': carla.Timestamp
                    }
                }
            timestamp (float): Current simulation time in seconds

        Returns:
            carla.VehicleControl: Control commands with fields:
                - throttle (float [0, 1]): Acceleration
                - steer (float [-1, 1]): Steering angle (left negative, right positive)
                - brake (float [0, 1]): Brake intensity
                - hand_brake (bool): Emergency brake
                - reverse (bool): Reverse gear

        Example:
            >>> def run_step(self, input_data, timestamp):
            ...     rgb = input_data['Center']['data']
            ...     steer, throttle = self.model.predict(rgb)
            ...     control = carla.VehicleControl()
            ...     control.steer = steer
            ...     control.throttle = throttle
            ...     control.brake = 0.0
            ...     return control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        return control

    def destroy(self):
        """Clean up agent resources.

        Override this method to release models, close files, etc.

        Example:
            >>> def destroy(self):
            ...     if hasattr(self, 'model'):
            ...         del self.model
            ...     torch.cuda.empty_cache()
        """
        pass

    def __call__(self):
        """Execute one agent step (called automatically by framework).

        Collects sensor data, calls run_step(), and returns control commands.
        Do not override this method.

        Returns:
            carla.VehicleControl: Vehicle control commands

        Note:
            This method is called by AgentWrapper each simulation tick.
        """
        # Collect sensor data
        input_data = self.sensor_interface.get_data()

        # Get timing information
        timestamp = GameTime.get_time()
        wallclock = GameTime.get_wallclocktime()
        print('======[Agent] Wallclock_time = {} / Sim_time = {}'.format(wallclock, timestamp))

        # Compute control
        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Set the navigation route for the agent.

        Called automatically by RouteScenario to provide the route.

        Args:
            global_plan_gps (list): Route as GPS coordinates [(lat, lon), ...]
            global_plan_world_coord (list): Route as waypoints [(waypoint, RoadOption), ...]

        Note:
            Routes are automatically downsampled to 1-meter spacing.
            Access via self._global_plan and self._global_plan_world_coord.
        """
        # Downsample to 1-meter spacing
        ds_ids = downsample_route(global_plan_world_coord, 1)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1])
                                         for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
