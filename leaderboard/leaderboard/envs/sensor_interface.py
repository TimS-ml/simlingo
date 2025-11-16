"""Sensor interface and data management for CARLA autonomous agents.

This module provides the infrastructure for managing sensor data collection and delivery
in the CARLA Leaderboard. It handles:

1. Sensor Registration: Creating and attaching sensors to the ego vehicle
2. Data Collection: Receiving sensor callbacks from CARLA
3. Data Synchronization: Ensuring all sensors provide data for the same frame
4. Data Format: Converting CARLA sensor data to numpy arrays
5. Pseudo-sensors: Handling special sensors like speedometer and HD map

Key Components:
    - SensorInterface: Main class that agents use to receive sensor data
    - CallBack: Handles sensor callbacks and data parsing
    - BaseReader: Base class for pseudo-sensors (speedometer, HD map)
    - SensorConfigurationInvalid: Exception for invalid sensor configurations
    - SensorReceivedNoData: Exception for sensor timeout issues

The sensor interface uses a queue-based system to collect data from all sensors and
ensures frame synchronization before delivering data to the agent. This guarantees
that all sensor readings correspond to the same simulation timestep.
"""

import copy
import logging
import numpy as np
import os
import time
from threading import Thread

from queue import Queue
from queue import Empty

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def threaded(fn):
    """Decorator to run a function in a daemon thread.

    This decorator wraps a function to execute it in a separate daemon thread,
    which is useful for background sensor reading without blocking the main thread.

    Args:
        fn (callable): Function to execute in a thread

    Returns:
        callable: Wrapper function that starts fn in a daemon thread
    """
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)  # Thread exits when main program exits
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):
    """Exception raised when sensor configuration violates competition rules.

    This exception is thrown when:
    - Sensors requested by the agent are not allowed for the declared track
    - Duplicate sensor tags are detected
    - Sensor parameters are invalid or missing

    For example, requesting 'sensor.opendrive_map' while on SENSORS track
    will raise this exception.
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """Exception raised when a sensor fails to provide data within the timeout period.

    This indicates that a sensor took longer than the configured timeout
    (default 10 seconds) to send its data. Possible causes:
    - CARLA server performance issues
    - Network communication problems
    - Sensor malfunction or misconfiguration
    - Simulation overload

    This typically results in route failure and requires investigation.
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()

    def __call__(self):
        pass

    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class SpeedometerReader(BaseReader):
    """Pseudo-sensor that provides vehicle forward speed measurements.

    Unlike physical CARLA sensors (cameras, LiDAR), the speedometer is a pseudo-sensor
    that queries the vehicle's physics state directly. It computes forward speed by
    projecting the velocity vector onto the vehicle's forward direction.

    The speedometer runs in a background thread and provides measurements at the
    configured frequency (default 1 Hz). Connection attempts are retried up to 10 times
    to handle temporary CARLA server communication issues.

    Attributes:
        MAX_CONNECTION_ATTEMPTS (int): Maximum retries for vehicle state queries

    Returns:
        dict: {'speed': float} where speed is in meters/second (m/s)
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None):
        """Convert vehicle velocity to forward speed along the heading direction.

        Projects the 3D velocity vector onto the vehicle's forward direction vector
        to get the scalar forward speed. This accounts for vehicle pitch and yaw.

        Args:
            transform (carla.Transform, optional): Vehicle transform. If None, queries vehicle.
            velocity (carla.Vector3D, optional): Vehicle velocity. If None, queries vehicle.

        Returns:
            float: Forward speed in meters/second. Positive = forward, negative = reverse.
        """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        return {'speed': self._get_forward_speed(transform=transform, velocity=velocity)}


class OpenDriveMapReader(BaseReader):
    def __call__(self):
        return {'opendrive': CarlaDataProvider.get_map().to_opendrive()}


class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


class SensorInterface(object):
    """Central manager for sensor data collection and synchronization.

    This class is the main interface between agents and CARLA sensors. It handles:
    - Registering sensors and creating callbacks
    - Collecting data from multiple sensors asynchronously
    - Synchronizing data to ensure all sensors provide data for the same frame
    - Delivering sensor data bundles to agents

    The interface uses a thread-safe queue to collect sensor data as it arrives,
    then waits until all sensors have reported data for the requested frame before
    returning. This ensures temporal consistency across all sensor modalities.

    Frame Synchronization:
        The interface blocks until ALL registered sensors provide data for the
        requested frame. The timeout (default 10s) prevents indefinite blocking
        if a sensor fails. The OpenDRIVE map sensor is treated specially as it
        doesn't update per-frame.

    Attributes:
        _sensors_objects (dict): Registered sensors keyed by tag
        _data_buffers (Queue): Thread-safe queue for incoming sensor data
        _queue_timeout (int): Maximum seconds to wait for sensor data
        _opendrive_tag (str): Tag of HD map sensor (if any)

    Example:
        # Typically used inside AutonomousAgent
        sensor_interface = SensorInterface()
        # Sensors are registered automatically during setup
        # Get synchronized data for current frame
        data = sensor_interface.get_data(GameTime.get_frame())
        # data = {'Left': (frame, numpy_array), 'LIDAR': (frame, points), ...}
    """

    def __init__(self):
        """Initialize the sensor interface with empty sensor registry and data queue."""
        self._sensors_objects = {}  # Registered sensors
        self._data_buffers = Queue()  # Thread-safe queue for sensor data
        self._queue_timeout = 10  # Max seconds to wait for sensor data

        # Only sensor that doesn't get the data on tick, needs special treatment
        # HD map is static and doesn't need frame synchronization
        self._opendrive_tag = None

    def register_sensor(self, tag, sensor_type, sensor):
        """Register a sensor with the interface.

        This method is called when a sensor is created and attached to the vehicle.
        It stores the sensor reference and handles special cases like the HD map sensor.

        Args:
            tag (str): Unique identifier for this sensor (from sensors() method)
            sensor_type (str): CARLA sensor type (e.g., 'sensor.camera.rgb')
            sensor (object): CARLA sensor actor or pseudo-sensor object

        Raises:
            SensorConfigurationInvalid: If a sensor with the same tag already exists
        """
        if tag in self._sensors_objects:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor

        # HD map sensor is static and doesn't provide per-frame updates
        if sensor_type == 'sensor.opendrive_map':
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, frame):
        """Receive and queue sensor data from a callback.

        This method is called by sensor callbacks when new data arrives. It adds
        the data to the queue along with the frame number for synchronization.

        Args:
            tag (str): Sensor identifier
            data: Sensor measurement (numpy array, point cloud, etc.)
            frame (int): Simulation frame number when data was captured

        Raises:
            SensorConfigurationInvalid: If sensor tag is not registered
        """
        if tag not in self._sensors_objects:
            raise SensorConfigurationInvalid("The sensor with tag [{}] has not been created!".format(tag))

        # Add to thread-safe queue: (tag, frame, data)
        self._data_buffers.put((tag, frame, data))

    def get_data(self, frame):
        """Retrieve synchronized sensor data for a specific frame.

        Blocks until all registered sensors provide data for the requested frame.
        This ensures that agents receive temporally consistent sensor readings.

        The method continuously polls the data queue until it has collected data
        from all sensors (except the HD map) for the specified frame number.

        Args:
            frame (int): Simulation frame number to retrieve data for

        Returns:
            dict: Mapping from sensor tags to (frame, data) tuples:
                {
                    'Left': (frame_num, numpy_array),
                    'LIDAR': (frame_num, point_cloud),
                    'GPS': (frame_num, lat_lon_alt),
                    ...
                }

        Raises:
            SensorReceivedNoData: If any sensor fails to provide data within timeout

        Note:
            The HD map sensor (if present) is excluded from synchronization as it's
            static and doesn't update per frame.
        """
        try:
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):
                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    break

                sensor_data = self._data_buffers.get(True, self._queue_timeout)
                if sensor_data[1] != frame:
                    continue
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")

        return data_dict
