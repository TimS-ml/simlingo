"""Route Management: Filter Duplicate CARLA Routes

This module filters out duplicate routes from the CARLA simulation dataset based on
spatial proximity of start and end waypoints. Routes with start/end points closer
than a threshold distance (min_dist) are considered duplicates, and only one is kept.

The script:
1. Connects to a running CARLA server
2. Parses route XML files
3. Computes interpolated traces using GlobalRoutePlanner
4. Identifies spatially duplicate routes
5. Copies unique routes to a filtered output directory
6. Generates visualization plots showing the filtered route network

Prerequisites:
    - CARLA server must be running on the specified port
    - Start CARLA with: cd $CARLA_ROOT && ./CarlaUE4.sh --world-port=2500 -RenderOffScreen -nosound
    - Use the garage environment to execute the script

Typical usage:
    python filter_duplicate_routes.py
"""

import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import carla
import os
import shutil
import glob
import pathlib
from agents.navigation.global_route_planner import GlobalRoutePlanner
import argparse
import time
from pathlib import Path
from IPython.display import display, clear_output
import tqdm


# Configuration parameters
TOWN = 'Town13'  # CARLA town/map to load
MAIN_DIR = 'data/longxall_val/routes_validation'  # Source directory for route XML files
MAIN_DESTINATION = 'data/benchmarks/longxall_train_filtered'  # Destination for filtered routes
Path(MAIN_DESTINATION).mkdir(parents=True, exist_ok=True)

# Minimum distance (in meters) between route start/end points to be considered unique
min_dist = 0.2

# Connect to CARLA server
client = carla.Client('localhost', 2500)
client.set_timeout(240)
world = client.load_world(TOWN)

# Initialize map and route planner
carla_map = world.get_map()
grp_1 = GlobalRoutePlanner(carla_map, 1.0)  # 1.0 meter sampling resolution
print('loaded.')


class Route():
    """Represents a CARLA route with waypoints, scenarios, and interpolated trace.

    This class parses route XML data, creates an interpolated trace along the route
    using CARLA's GlobalRoutePlanner, and integrates scenario trigger points into
    the trace.

    Attributes:
        weather_params (list): Names of weather parameters for the route
        weather_values_begin (list): Weather parameter values at route start
        weather_values_end (list): Weather parameter values at route end
        route_town (str): Name of the CARLA town/map for this route
        waypoints (np.ndarray): Array of key waypoints (x, y, z) defining the route
        scenarios (list): List of scenario XML elements
        scenario_locations (np.ndarray): Array of scenario trigger point locations
        trace (np.ndarray): Full interpolated trace including waypoints and scenarios
        trace_type (np.ndarray): Type of each trace point ('waypoint', 'trace', or 'scenario')
        trace_elem (list): Associated XML element for each trace point (or None)
    """
    weather_params = ["route_percentage", "cloudiness", "precipitation", "precipitation_deposits",
                      "wetness", "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle", "fog_density"]
    weather_values_begin = [0, 5.0, 0.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0]
    weather_values_end = [100, 5.0, 0.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0]

    def __init__(self, route_tree):
        """Initialize Route from XML tree.

        Args:
            route_tree (xml.etree.ElementTree.Element): XML element containing route data
        """
        self.parse_route_tree(route_tree)
        self.create_trace()
        self.sort_scenarios_in()

    def parse_route_tree(self, route_tree):
        """Parse route XML tree to extract waypoints and scenarios.

        Extracts the town name, waypoint positions, and scenario trigger points
        from the route XML structure.

        Args:
            route_tree (xml.etree.ElementTree.Element): XML element containing route data
        """
        # Extract town/map name from route attributes
        self.route_town = route_tree.attrib['town']
        self.waypoints = []

        # Parse all waypoint positions
        for waypoint in route_tree.find('waypoints').iter('position'):
            loc = [waypoint.attrib['x'], waypoint.attrib['y'], waypoint.attrib['z']]
            self.waypoints.append(loc)

        self.waypoints = np.array(self.waypoints).astype('float')

        # Parse all scenario trigger points
        self.scenarios = []
        self.scenario_locations = []
        for scenario in route_tree.find('scenarios').iter('scenario'):
            p = scenario.find('trigger_point')
            loc = [p.attrib['x'], p.attrib['y'], p.attrib['z']]

            self.scenario_locations.append(loc)
            self.scenarios.append(scenario)

        self.scenario_locations = np.array(self.scenario_locations).astype('float')

    def create_trace(self):
        """Create interpolated route trace using CARLA's GlobalRoutePlanner.

        Generates a detailed trace by interpolating between consecutive waypoints
        using CARLA's road network. The trace includes the original waypoints plus
        interpolated points along valid driving paths.
        """
        self.trace = []
        self.trace_type = []
        self.trace_elem = []

        # Interpolate between each pair of consecutive waypoints
        for i in range(len(self.waypoints)-1):
            p = self.waypoints[i]
            p_next = self.waypoints[i+1]

            # Convert numpy arrays to CARLA Location objects
            waypoint = carla.Location(x=p[0], y=p[1], z=p[2])
            waypoint_next = carla.Location(x=p_next[0], y=p_next[1], z=p_next[2])

            # Use GlobalRoutePlanner to get interpolated path between waypoints
            interpolated_trace = grp_1.trace_route(waypoint, waypoint_next)

            # Extract location coordinates from route trace
            interpolated_trace = [x[0].transform.location for x in interpolated_trace]
            interpolated_trace = [[x.x, x.y, x.z] for x in interpolated_trace]

            # Add waypoint and interpolated points to trace
            self.trace += [p] + interpolated_trace
            self.trace_type += ['waypoint'] + ['trace'] * len(interpolated_trace)
            self.trace_elem += [None] + [None] * len(interpolated_trace)

        # Add the final waypoint
        self.trace.append(p_next)
        self.trace_type.append('waypoint')
        self.trace_elem.append(None)

        self.trace = np.array(self.trace)
        self.trace_type = np.array(self.trace_type)

    def sort_scenarios_in(self):
        """Insert scenario trigger points into the route trace at nearest positions.

        For each scenario, finds the closest point in the interpolated trace
        (excluding waypoints) and inserts the scenario trigger point there.
        This ensures scenarios are triggered at appropriate locations along the route.
        """
        for scenario, scenario_location in zip(self.scenarios, self.scenario_locations):
            # Calculate distance from scenario to all trace points
            diff = self.trace - scenario_location[None]
            diff = np.linalg.norm(diff, axis=1)

            # Exclude waypoints from consideration (only insert at trace points)
            diff[self.trace_type == 'waypoint'] = 1e9
            min_idx = np.argmin(diff)

            # Insert scenario at the nearest trace point
            self.trace = np.concatenate([self.trace[:min_idx], [scenario_location], self.trace[min_idx:]])
            self.trace_type = np.concatenate([self.trace_type[:min_idx], ['scenario'], self.trace_type[min_idx:]])
            self.trace_elem = self.trace_elem[:min_idx] + [scenario] + self.trace_elem[min_idx:]

# Generate background waypoints from the map for visualization
# These represent all drivable road positions in the CARLA town
waypoint_list = carla_map.generate_waypoints(5.0)  # Sample every 5 meters
waypoint_list = [x.transform.location for x in waypoint_list]
waypoint_list = [[x.x, x.y, x.z] for x in waypoint_list]
waypoint_list = np.array(waypoint_list)

# Process each scenario directory
dirs = glob.glob(MAIN_DIR + '/*/')
for DIR in dirs:
    # Skip the 'allroutes' aggregation directory
    if "allroutes" in DIR:
        continue

    # Create destination directory for filtered routes
    destination = MAIN_DESTINATION + '/' + DIR.split("/")[-2]
    Path(destination).mkdir(parents=True, exist_ok=True)

    l_routes = []  # List of Route objects
    l_routes_idx = []  # List of route identifiers (XML filenames without extension)

    # Find all route XML files in this scenario directory
    all_xml_paths = glob.glob(f'{DIR}/**/*.xml', recursive=True)

    # Parse all route XML files
    for pth in tqdm.tqdm(all_xml_paths):
        tree = ET.parse(pth)

        # Each XML file may contain multiple <route> elements
        for i, route_tree in enumerate(tree.iter("route")):
            l_routes.append(Route(route_tree))
            l_routes_idx.append(pth.split('/')[-1].split('.')[0])

    # Initialize visualization: plot map background with all drivable waypoints
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(waypoint_list[:, 0], waypoint_list[:, 1], c='lightgray', s=1)

    # Lists to track unique (non-duplicate) routes
    saved_wps = []  # Waypoints of unique routes
    saved_wps_idx = []  # Identifiers of unique routes

    # Filter duplicate routes based on spatial proximity
    for i, route in tqdm.tqdm(enumerate(l_routes)):
        # Extract different components of the route trace
        trace = route.trace[route.trace_type == 'trace']  # Interpolated points
        scenario = route.trace[route.trace_type == 'scenario']  # Scenario triggers
        waypoint = route.trace[route.trace_type == 'waypoint']  # Key waypoints

        # Check if a route with similar start/end points already exists
        # A route is duplicate if both start AND end points are within min_dist
        duplicate = False
        for saved_route in saved_wps:
            # Compare first waypoint to first waypoint AND last to last
            if (np.linalg.norm(saved_route[0] - waypoint[0]) < min_dist and
                np.linalg.norm(saved_route[-1] - waypoint[-1]) < min_dist):
                duplicate = True
                break

            # Alternative duplicate checks (commented out):
            # - Could also check for reversed routes (start-to-end vs end-to-start)
            # - Currently only checking for routes in the same direction

        # Skip duplicate routes
        if duplicate:
            continue

        # Save this unique route
        saved_wps.append(waypoint)
        saved_wps_idx.append(l_routes_idx[i])

        # Add route to visualization
        ax.scatter(trace[:, 0], trace[:, 1], c='black', s=1)  # Route trace (black)
        ax.scatter(scenario[:, 0], scenario[:, 1], c='orange', s=20)  # Scenarios (orange)
        ax.scatter(waypoint[:, 0], waypoint[:, 1], c='green', s=40, marker='x')  # Waypoints (green X)
        ax.annotate(str(l_routes_idx[i]), (waypoint[0, 0]+40, waypoint[0, 1]+40))  # Route label

        clear_output(wait=True)  # Update display in Jupyter notebook

    # Save visualization showing all filtered routes
    plt.savefig(f"{destination}/{DIR.split('/')[-2]}_{min_dist}.png")

    # Copy unique route XML files to destination directory
    for routes in saved_wps_idx:
        shutil.copy(os.path.join(DIR, routes+'.xml'), destination)

    # Print filtering statistics
    print(f"Number of selected routes: {len(saved_wps_idx)}")
    print(f"Number of scenarios: {len(l_routes_idx)}")

    plt.clf()  # Clear figure for next iteration