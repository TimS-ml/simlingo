"""Ability Benchmark Evaluator for Bench2Drive.

This script evaluates autonomous driving performance across 5 key driving abilities:
1. Overtaking - Navigating around obstacles and slow vehicles
2. Merging - Joining traffic flows and lane changes
3. Emergency_Brake - Reacting to sudden hazards
4. Give_Way - Yielding to priority vehicles
5. Traffic_Signs - Following traffic signals and signs

The evaluator processes route results and computes ability-specific success rates
using CARLA simulation and route planning.

Usage:
    python ability_benchmark.py -f <routes_file> -r <result_file> -t <host> -p <port>

Example:
    python ability_benchmark.py -f leaderboard/data/bench2drive220.xml \\
                               -r eval/results.json -p 4000
"""

import json
import carla
import argparse
import xml.etree.ElementTree as ET
from agents.navigation.global_route_planner import GlobalRoutePlanner
import os
import atexit
import subprocess
import time
import random

# Mapping of driving abilities to their associated scenario types
Ability = {
    "Overtaking": ['Accident', 'AccidentTwoWays', 'ConstructionObstacle', 'ConstructionObstacleTwoWays',
                   'HazardAtSideLaneTwoWays', 'HazardAtSideLane', 'ParkedObstacleTwoWays', 'ParkedObstacle',
                   'VehicleOpensDoorTwoWays'],
    "Merging": ['CrossingBicycleFlow', 'EnterActorFlow', 'HighwayExit', 'InterurbanActorFlow', 'HighwayCutIn',
                'InterurbanAdvancedActorFlow', 'MergerIntoSlowTrafficV2', 'MergerIntoSlowTraffic',
                'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow',
                'ParkingExit', 'SequentialLaneChange', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn',
                'SignalizedJunctionLeftTurnEnterFlow'],
    "Emergency_Brake": ['BlockedIntersection', 'DynamicObjectCrossing', 'HardBreakRoute', 'OppositeVehicleTakingPriority',
                        'OppositeVehicleRunningRedLight', 'ParkingCutIn', 'PedestrianCrossing', 'ParkingCrossingPedestrian',
                        'StaticCutIn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'ControlLoss'],
    "Give_Way": ['InvadingTurn', 'YieldToEmergencyVehicle'],
    "Traffic_Signs": ['BlockedIntersection', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight',
                      'PedestrianCrossing', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'EnterActorFlow',
                      'CrossingBicycleFlow', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn',
                      'NonSignalizedJunctionLeftTurnEnterFlow', 'OppositeVehicleTakingPriority',
                      'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'SignalizedJunctionLeftTurn',
                      'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow', 'T_Junction',
                      'VanillaNonSignalizedTurn', 'VanillaSignalizedTurnEncounterGreenLight',
                      'VanillaSignalizedTurnEncounterRedLight', 'VanillaNonSignalizedTurnEncounterStopsign',
                      'VehicleTurningRoute', 'VehicleTurningRoutePedestrian']
}

def get_infraction_status(record):
    """Check if a route record has any infractions (excluding minor speed infractions).

    Args:
        record: Route record dictionary containing infraction data.

    Returns:
        bool: True if the route has any significant infractions, False otherwise.
    """
    for infraction, value in record['infractions'].items():
        # Skip minor speed infractions
        if infraction == "min_speed_infractions":
            continue
        elif len(value) > 0:
            return True
    return False

def update_Ability(scenario_name, Ability_Statistic, status):
    """Update ability statistics based on scenario success status.

    Args:
        scenario_name: Name of the scenario type.
        Ability_Statistic: Dictionary tracking [successes, total_attempts] for each ability.
        status: Boolean indicating if the scenario was successful.
    """
    for ability, scenarios in Ability.items():
        if scenario_name in scenarios:
            Ability_Statistic[ability][1] += 1  # Increment total attempts
            if status:
                Ability_Statistic[ability][0] += 1  # Increment successes

def update_Success(scenario_name, Success_Statistic, status):
    """Update per-scenario success statistics.

    Args:
        scenario_name: Name of the scenario type.
        Success_Statistic: Dictionary tracking [successes, total_attempts] per scenario.
        status: Boolean indicating if the scenario was successful.
    """
    if scenario_name not in Success_Statistic:
        if status:
            Success_Statistic[scenario_name] = [1, 1]
        else:
            Success_Statistic[scenario_name] = [0, 1]
    else:
        Success_Statistic[scenario_name][1] += 1
        if status:
            Success_Statistic[scenario_name][0] += 1

def get_position(xml_route):
    waypoints_elem = xml_route.find('waypoints')
    keypoints = waypoints_elem.findall('position')
    return [carla.Location(float(pos.get('x')), float(pos.get('y')), float(pos.get('z'))) for pos in keypoints]

def get_route_result(records, route_id):
    for record in records:
        record_route_id = record['route_id'].split('_')[1]
        if route_id == record_route_id:
            return record
    return None

def get_waypoint_route(locs, grp):
    route = []
    for i in range(len(locs) - 1):
        loc = locs[i]
        loc_next = locs[i + 1]
        interpolated_trace = grp.trace_route(loc, loc_next)
        for wp, _ in interpolated_trace:
            route.append(wp)
    return route

def main(args):
    routes_file = args.file 
    result_file = args.result_file
    Ability_Statistic = {}
    crash_route_list = []
    for key in Ability:
        Ability_Statistic[key] = [0, 0.]
    Success_Statistic = {}
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    records = data["_checkpoint"]["records"]
                    
    tree = ET.parse(routes_file)
    root = tree.getroot()
    routes = root.findall('route')
    sorted_routes = sorted(routes, key=lambda x: x.get('town'))
    
    carla_path = os.environ["CARLA_ROOT"]
    cmd1 = f"{os.path.join(carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={args.port}"
    server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
    print(cmd1, server.returncode, flush=True)
    time.sleep(60)
    client = carla.Client(args.host, args.port)
    client.set_timeout(300)
    
    current_town = sorted_routes[0].get('town')
    world = client.load_world(current_town)
    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, 1.0)
    for route in sorted_routes:
        scenarios = route.find('scenarios')
        scenario_name = scenarios.find('scenario').get("type")
        route_id = route.get('id')
        route_record = get_route_result(records, route_id)
        if route_record is None:
            crash_route_list.append((scenario_name, route_id))
            print('No result record of route', route_id, "in the result file")
            continue
        if route_record["status"] == 'Completed' or route_record["status"] == "Perfect":
            if get_infraction_status(route_record):
                record_success_status = False
            else:
                record_success_status = True
        else:
            record_success_status = False
        update_Ability(scenario_name, Ability_Statistic, record_success_status)
        update_Success(scenario_name, Success_Statistic, record_success_status)
        # if scenario_name in Ability["Traffic_Signs"] and (scenario_name in Ability["Merging"] or scenario_name in Ability["Emergency_Brake"]):
        # Only these three 'Ability's intersect
        if scenario_name in Ability["Traffic_Signs"]:
            # Only these three 'Ability's intersect
            if route.get('town') != current_town:
                current_town = route.get('town')
                print("Loading the town:", current_town)
                world = client.load_world(current_town)
                print("successfully load the town:", current_town)
            carla_map = world.get_map()
            grp = GlobalRoutePlanner(carla_map, 1.0)
            location_list = get_position(route)
            waypoint_route = get_waypoint_route(location_list, grp)
            count = 0
            for wp in waypoint_route:
                count += 1
                if wp.is_junction:
                    break 
            if not wp.is_junction:
                raise RuntimeError("This route does not contain any junction-waypoint!")
            # +8 to ensure the ego pass the trigger volume
            junction_completion = float(count+8) / float(len(waypoint_route))
            record_completion = route_record["scores"]["score_route"] / 100.0
            stop_infraction = route_record["infractions"]["stop_infraction"]
            red_light_infraction = route_record["infractions"]["red_light"]
            if record_completion > junction_completion and not stop_infraction and not red_light_infraction:
                Ability_Statistic['Traffic_Signs'][0] += 1
                Ability_Statistic['Traffic_Signs'][1] += 1
            else:
                Ability_Statistic['Traffic_Signs'][1] += 1
        else:
            pass
        
    Ability_Res = {}
    for ability, statis in Ability_Statistic.items():
        Ability_Res[ability] = float(statis[0])/float(statis[1])
        
    for key, value in Ability_Res.items():
        print(key, ": ", value)
    
    Ability_Res['mean'] = sum(list(Ability_Res.values())) / 5
    Ability_Res['crashed'] = crash_route_list
    with open(f"{result_file.split('.')[0]}_ability.json", 'w') as file:
        json.dump(Ability_Res, file, indent=4)
        
    Success_Res = {}
    Route_num = 0
    Succ_Route_num = 0
    for scenario, statis in Success_Statistic.items():
        Success_Res[scenario] = float(statis[0])/float(statis[1])
        Succ_Route_num += statis[0]
        Route_num += statis[1]
    assert len(crash_route_list) == 220 - float(Route_num)
    print(f"Mean:{Ability_Res['mean']}")
    print(f'Crashed Route num: {len(crash_route_list)}, Crashed Route ID: {crash_route_list}')
    print('Finished!')

if __name__=='__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-f', '--file', nargs=None, default="leaderboard/data/bench2drive220.xml", help='route file')
    argparser.add_argument('-r', '--result_file', nargs=None, default="", help='result json file')
    argparser.add_argument('-t', '--host', default='localhost', help='IP of the host server (default: localhost)')
    argparser.add_argument('-p', '--port', nargs=1, default=4000, help='carla rpc port')
    args = argparser.parse_args()
    main(args)
    