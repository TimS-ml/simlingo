"""Route Management: Balance Scenario Distribution via Upsampling

This module balances the CARLA route dataset by ensuring each scenario type has
approximately the same number of route instances. This is crucial for training
models that don't overfit to common scenarios while underperforming on rare ones.

The script:
1. Analyzes existing route XML files to count scenarios by type
2. Upsamples underrepresented scenarios through random selection with replacement
3. Randomizes weather conditions for each upsampled route
4. Adds random variance to scenario parameters (e.g., distances)
5. Generates visualization plots comparing original vs balanced distributions

Key features:
- Ensures NUM_SAMPLES routes per scenario type
- Randomizes weather on upsampled routes to increase diversity
- Adds parameter variation to prevent exact duplicates
- Tracks unique route usage to limit over-replication
- Handles special scenarios (e.g., ControlLoss, HardBreakRoute) with downsampling

The upsampling process:
- Routes containing rare scenarios are duplicated
- Each duplicate gets new randomized weather
- Scenario parameters (like distance) are varied within 10%
- Multiple scenarios in one route all count toward their respective quotas

Adapted from https://github.com/autonomousvision/carla_garage

Typical usage:
    python balance_scenarios.py --path-in data/routes --num-samples 150
"""

import argparse
import glob
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt


argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', default=1, type=int, help='Seed for random number generator.')
argparser.add_argument('--num-samples', default=150, type=int, help='Number of samples per scenario.')
argparser.add_argument('--routes-per-file', default=40, type=int, help='Maximum number of routes per xml file.')
argparser.add_argument('--path-in', default='/home/karenz/coding/simlingo_cleanup/data/training_1_scenario/routes_training/random_weather_easy_weather_seed_1', type=str, help='Input path of the route that should be split.')
argparser.add_argument('--max-scenarios', default=1, type=int, help='Maximum number of scenarios per xml file.')
argparser.add_argument('--easy-weather', default=False, action='store_true', help='Use easy weather.')
args = argparser.parse_args()

# Create output directory with balanced dataset
args.save_path = f'{args.path_in}_balanced_{args.num_samples}'
Path(args.save_path).mkdir(parents=True, exist_ok=True)
random.seed(args.seed)

def get_random_weather_values():
    """Generate randomized weather parameter values for route diversity.

    Selects random weather values from predefined ranges to create varied driving
    conditions. Supports both 'easy' weather (limited variation) and full weather
    (wide range of conditions including extreme weather).

    Returns:
        tuple: (weather_values_begin, weather_values_end, weather_params)
            - weather_values_begin: List of weather values at route start
            - weather_values_end: List of weather values at route end
            - weather_params: List of parameter names
    """

    weather_params = ["route_percentage", "cloudiness", "precipitation", "precipitation_deposits", "wetness", "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle", "fog_density"]
    if args.easy_weather:
        ranges_per_param = {
            'cloudiness': [0.0, 2.0, 5.0, 10.0, 15.0, 20.0],
            'precipitation': [0.0, 2.0, 4.0, 6.0],
            'precipitation_deposits': [0.0, 4.0, 8.0],
            'wetness': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            'wind_intensity': [5.0, 10.0],
            'sun_azimuth_angle': [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0],
            'sun_altitude_angle': [10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 70.0, 80.0, 90.0],
            'fog_density': [0.0, 0.0, 1.0, 1.0, 2.0]
        }
    else:
        ranges_per_param = {
            'cloudiness': [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 40.0, 50.0, 60.0, 80.0, 100.0],
            'precipitation': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
            'precipitation_deposits': [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
            'wetness': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            'wind_intensity': [5.0, 10.0, 25.0, 30.0, 50.0, 60.0, 80.0, 100.0],
            'sun_azimuth_angle': [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0],
            'sun_altitude_angle': [-90.0, -45.0, -30.0, -10, -15.0, 5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 70.0, 80.0, 90.0],
            'fog_density': [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0, 70.0, 100.0]
        }
    weather_values_begin = [0]
    weather_values_end = [100]
    for param in weather_params[1:]:
        ranges = ranges_per_param[param]
        weather_values_begin.append(random.choice(ranges))
        weather_values_end.append(random.choice(ranges))

    return weather_values_begin, weather_values_end, weather_params

# Find all route XML files in the input directory
all_xml_paths = glob.glob(f'{args.path_in}/**/*.xml', recursive=True)

# Dictionary to track scenario counts and which routes contain each scenario
scenarios_count = {}  # scenario_name -> count
scenarios_routes = {}  # scenario_name -> list of route file paths

num_xmls = 0  # Counter for output XML files

# First pass: Count existing scenarios and group routes by scenario type
for route in all_xml_paths:
    tree = ET.parse(route)
    root = tree.getroot()

    # Iterate over all route elements in the XML file
    for route_id in root.iter('route'):
        # Handle routes with no scenarios
        if len(route_id.find('scenarios')) == 0:
            scenario_name = 'None'
            if scenario_name in scenarios_count:
                scenarios_count[scenario_name] += 1
                scenarios_routes[scenario_name].append(route)
            else:
                scenarios_count[scenario_name] = 1
                scenarios_routes[scenario_name] = [route]

        # Count each scenario in the route
        for scenario in route_id.find('scenarios').iter('scenario'):
            scenario_name = scenario.attrib['type']
            if scenario_name in scenarios_count:
                scenarios_count[scenario_name] += 1
                scenarios_routes[scenario_name].append(route)
            else:
                scenarios_count[scenario_name] = 1
                scenarios_routes[scenario_name] = [route]

# Print original scenario distribution (sorted by count, descending)
for scenario_name, count in sorted(scenarios_count.items(), key=lambda x: x[1], reverse=True):
    print(f'{scenario_name}: {count}')



# Tracking dictionaries for upsampling process
upsampled_scenario_nums = {}  # Track how many times each scenario has been added
paths_per_scenario = {}  # Track which route paths are used for each scenario
none_already_upsampled = False  # Flag to ensure 'None' scenario is only processed once
count_paths = {}  # Track how many times each unique route file is used (limit: args.num_samples)

# Sort scenarios by number of routes (ascending) to process rarest scenarios first
# This ensures rare scenarios get priority access to routes
scenarios_routes = {k: v for k, v in sorted(scenarios_routes.items(), key=lambda item: len(item[1]), reverse=False)}

# Main upsampling loop: Process each scenario type to reach target count
for scenario_name, routes in scenarios_routes.items():
    print(f'Upsampling {scenario_name}')
    print(f'Number of routes: {len(routes)}')
    # if scenarios_count[scenario_name] >= args.num_samples:
    #     print(f'Skipping {scenario_name}')
    #     continue

    if scenario_name == 'None' and not none_already_upsampled:
        #ranodmly draw args.num_samples routes, allow duplicates
        upsampled_routes = random.choices(routes, k=args.num_samples)
        paths_per_scenario[scenario_name] = upsampled_routes
        none_already_upsampled = True
        for route in upsampled_routes:
            tree = ET.parse(route)
            num_xmls += 1
            ET.ElementTree(tree.getroot()).write(f'{args.save_path}/{num_xmls}.xml')

    elif scenario_name != 'None':

        root_new = ET.Element('routes')
        new_route_id = 0

        added_routes = 0 #scenarios_count[scenario_name]
        tries = 0
        while (scenario_name not in upsampled_scenario_nums or upsampled_scenario_nums[scenario_name] < args.num_samples) and tries < 30000*len(routes):
            tries += 1
            route = random.choice(routes)
            if route in count_paths and count_paths[route] >= args.num_samples:
                continue

            tree = ET.parse(route)
            root = tree.getroot()
            # iterate over all routes
            for route_id in root.iter('route'):
                skip = False
                for scenario in route_id.find('scenarios').iter('scenario'):
                    scenario_name_tmp1 = scenario.attrib['type']
                    if scenario_name_tmp1 == 'ControlLoss' or scenario_name_tmp1 == 'HardBreakRoute':
                        if random.random() < 0.99:
                            skip = True
                            break
                if skip:
                    continue
                
                all_scenarios = [scenario.attrib['type'] for scenario in route_id.find('scenarios').iter('scenario')]

                for scenario in route_id.find('scenarios').iter('scenario'):
                    scenario_name_tmp = scenario.attrib['type']

                    if scenario_name_tmp == scenario_name:
                        for scenario_all_tmp in all_scenarios:
                            if scenario_all_tmp in upsampled_scenario_nums:
                                upsampled_scenario_nums[scenario_all_tmp] += 1
                            else:
                                upsampled_scenario_nums[scenario_all_tmp] = 1
                            if scenario_all_tmp in paths_per_scenario:
                                paths_per_scenario[scenario_all_tmp].append(route)
                            else:
                                paths_per_scenario[scenario_all_tmp] = [route]

                        route_id.set('id', f'{new_route_id}')

                        # if <distance value= is present, change it
                        distance_elem = scenario.find('distance')
                        if distance_elem is not None:
                            # get value
                            distance = float(distance_elem.attrib['value'])
                            # change value random +- 10%
                            distance = distance + distance * random.uniform(-0.1, 0.1)
                            # set new value
                            distance_elem.set('value', '{:.1f}'.format(distance))


                        # change weather values
                        weathers_elem = route_id.find('weathers')
                        # get all weather elems in weathers
                        weather_elems = weathers_elem.findall('weather')
                        weather_elem_begin = weather_elems[0]
                        weather_elem_end = weather_elems[1]

                        weather_values_begin, weather_values_end, weather_params = get_random_weather_values()

                        for param_name, value_begin, value_end in zip(weather_params, weather_values_begin, weather_values_end):
                            weather_elem_begin.set(param_name, '{:.1f}'.format(value_begin))
                            weather_elem_end.set(param_name, '{:.1f}'.format(value_end))

                        root_new.append(route_id)
                        added_routes += 1
                        new_route_id += 1

                        if new_route_id >= args.routes_per_file-1:
                            if route not in count_paths:
                                count_paths[route] = 1
                            else:
                                count_paths[route] += 1

                            num_xmls += 1
                            ET.ElementTree(root_new).write(f'{args.save_path}/{num_xmls}.xml')
                            root_new = ET.Element('routes')
                            new_route_id = 0
                        
                        continue
        

        if new_route_id > 0:
            num_xmls += 1
            ET.ElementTree(root_new).write(f'{args.save_path}/{num_xmls}.xml')
            root_new = ET.Element('routes')
            new_route_id = 0



# Generate the plot with the number of scenarios
paths = [
    args.path_in,
    args.save_path,
]
res = []

for path in paths:
    routes = sorted(glob.glob(f'{path}/**/*.xml', recursive=True))

    scenarios_count = {}
    for scenario_name in scenarios_routes.keys():
        scenarios_count[scenario_name] = 0

    for route in routes:
        try:
            tree = ET.parse(route)
        except ET.ParseError:
            print(f'Error parsing {route} in {path}')
            continue
        root = tree.getroot()
        # iterate over all routes
        for route_id in root.iter('route'):
            # get all information as dict
            if len(route_id.find('scenarios')) == 0:
                scenario_name = 'None'
                if scenario_name in scenarios_count:
                    scenarios_count[scenario_name] += 1
                else:
                    scenarios_count[scenario_name] = 1
            else:
                for scenario in route_id.find('scenarios').iter('scenario'):
                    scenario_name = scenario.attrib['type']
                    if scenario_name in scenarios_count:
                        scenarios_count[scenario_name] += 1
                    else:
                        scenarios_count[scenario_name] = 1


    scenarios_count = {k: v for k, v in sorted(scenarios_count.items(), key=lambda item: item[1], reverse=True)}
    # pretty print sorted by number of scenarios
    for scenario_name, count in sorted(scenarios_count.items(), key=lambda x: x[1], reverse=True):
        print(f'{scenario_name}: {count}')

    res.append(scenarios_count)

# breakpoint()
import matplotlib.pyplot as plt
import numpy as np

# Extract keys and values
keys = list(res[0].keys())
values1 = list(res[0].values())
values2 = [res[1][key] for key in keys] 

# Define the position of bars
x = np.arange(len(keys))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values1, width, label='Original')
rects2 = ax.bar(x + width/2, values2, width, label='Upsampled')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Scenario type')
ax.set_ylabel('# of routes')
# ax.set_title('Values by key for two dictionaries')
# x labels rotate 90 degrees

ax.set_xticks(x)
ax.set_xticklabels(keys, rotation=90)
ax.legend()

fig.tight_layout()

plt.savefig(f'{paths[1]}/scenario_types.png')


# get unique paths per scenario
paths_per_scenario = {k: list(set(v)) for k, v in paths_per_scenario.items()}

# create and save figure with bar plot
fig, ax = plt.subplots()
keys = list(paths_per_scenario.keys())
values = [len(v) for v in paths_per_scenario.values()]
ax.bar(keys, values)
ax.set_xlabel('Scenario type')
ax.set_ylabel('# of unique routes')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{paths[1]}/unique_routes_per_scenario.png')
