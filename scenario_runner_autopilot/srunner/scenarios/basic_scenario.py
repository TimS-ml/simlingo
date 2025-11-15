#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""BasicScenario - Base class for all scenario implementations.

This module provides the BasicScenario class, which serves as the foundation for all
scenario definitions in the scenario runner framework. It handles the common scenario
structure via py_trees behavior trees.

A scenario is defined as a behavior tree with the following structure:
    Parallel (SUCCESS_ON_ONE policy)
    ├── Lights behavior (optional)
    ├── Weather behavior (optional)
    ├── Main scenario behavior sequence
    │   ├── Trigger condition (wait for ego to reach starting point)
    │   ├── User-defined scenario behavior (implemented in subclass)
    │   └── End condition (signal completion in route mode)
    ├── Criteria tree (success/failure evaluation)
    ├── Timeout node (scenario deadline)
    └── UpdateAllActorControls (apply physics updates)

Key Concepts:
    - Behavior Trees: Hierarchical trees of actions, conditions, and control flow nodes
    - Criteria: Success/failure conditions evaluated in parallel with scenario execution
    - Trigger/End: Synchronization points for route-based scenario sequences
    - Timeout: Maximum allowed duration for scenario execution

Classes:
    BasicScenario: Abstract base class for all scenario implementations

Usage:
    Subclass BasicScenario and implement:
    - _create_behavior(): Define scenario action sequence
    - _create_test_criteria(): Define evaluation criteria
    - _initialize_actors() (optional): Custom actor setup
    - _create_weather_behavior() (optional): Dynamic weather changes
    - _create_lights_behavior() (optional): Street light control
"""

from __future__ import print_function

import operator
import py_trees

import carla

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (WaitForBlackboardVariable,
                                                                               InTimeToArrivalToLocation)
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import WaitForever
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import TimeOut
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import UpdateAllActorControls
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion


class BasicScenario(object):
    """Abstract base class for all scenario implementations.

    BasicScenario provides the common infrastructure for scenario definition via
    py_trees behavior trees. It handles environment setup, actor initialization,
    behavior tree construction, criteria integration, and resource management.

    Subclasses must implement:
        _create_behavior(): Returns py_trees behavior defining scenario actions
        _create_test_criteria(): Returns list of Criterion objects or composite tree

    Optionally override:
        _initialize_actors(config): Custom actor spawning logic
        _initialize_environment(world): Custom weather/friction setup
        _create_weather_behavior(): Dynamic weather changes during scenario
        _create_lights_behavior(): Street light control during scenario
        _setup_scenario_trigger(config): Custom trigger conditions
        _setup_scenario_end(config): Custom end conditions
        _create_timeout_behavior(): Custom timeout behavior

    Attributes:
        name (str): Scenario identifier
        ego_vehicles (list): Ego vehicle actors
        other_actors (list): Scenario-spawned actors
        config: Scenario configuration from XML or route file
        world (carla.World): CARLA world instance
        scenario_tree: Root of complete behavior tree (parallel composite)
        behavior_tree: Main scenario behavior sequence
        criteria_tree: Criteria evaluation tree (parallel composite)
        timeout_node: Timeout behavior node
        route_mode (bool): True if scenario is part of a route sequence

    Example Subclass:
        >>> class MyScenario(BasicScenario):
        ...     def _create_behavior(self):
        ...         return py_trees.composites.Sequence([
        ...             ActorTransformSetter(self.other_actors[0], transform),
        ...             WaitUntilInFront(self.ego_vehicles[0], self.other_actors[0])
        ...         ])
        ...
        ...     def _create_test_criteria(self):
        ...         return [CollisionTest(self.ego_vehicles[0])]
    """

    def __init__(self, name, ego_vehicles, config, world,
                 debug_mode=False, terminate_on_failure=False, criteria_enable=False):
        """Initialize the scenario with behavior tree construction.

        Sets up environment (weather, friction), initializes actors, constructs the
        behavior tree structure, integrates criteria, and prepares for execution.

        Args:
            name (str): Scenario identifier
            ego_vehicles (list): List of ego vehicle actors
            config: Scenario configuration with trigger_points, route, weather, friction, etc.
            world (carla.World): CARLA world instance
            debug_mode (bool): If True, enables py_trees debug logging
            terminate_on_failure (bool): If True, scenario stops on first criterion failure
            criteria_enable (bool): If True, creates and evaluates criteria tree

        Note:
            - Automatically calls _initialize_environment() and _initialize_actors()
            - Constructs complete behavior tree via template method pattern
            - Default timeout is 60 seconds if not set in subclass
            - In route mode, trigger/end conditions synchronize with route execution
        """
        self.name = name
        self.ego_vehicles = ego_vehicles
        self.other_actors = []
        self.parking_slots = []
        self.config = config
        self.world = world
        self.debug_mode = debug_mode
        self.terminate_on_failure = terminate_on_failure
        self.criteria_enable = criteria_enable

        self.route_mode = bool(config.route)
        self.behavior_tree = None
        self.criteria_tree = None

        # Set default timeout if not specified by subclass
        if not hasattr(self, 'timeout'):
            self.timeout = 60
        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        # Initialize environment (weather, friction) unless in route mode
        if not self.route_mode:
            self._initialize_environment(world)

        # Spawn scenario actors
        self._initialize_actors(config)

        # Synchronize world state
        if CarlaDataProvider.is_runtime_init_mode():
            world.wait_for_tick()
        elif CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

        # Build main scenario tree (parallel: first child to succeed terminates)
        self.scenario_tree = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # Build main behavior sequence with trigger/scenario/end
        self.behavior_tree = py_trees.composites.Sequence()

        # Add trigger condition (wait for ego to reach start or route variable)
        trigger_behavior = self._setup_scenario_trigger(config)
        if trigger_behavior:
            self.behavior_tree.add_child(trigger_behavior)

        # Add user-defined scenario behavior (implemented by subclass)
        scenario_behavior = self._create_behavior()
        self.behavior_tree.add_child(scenario_behavior)
        self.behavior_tree.name = scenario_behavior.name

        # Add end condition (reset route variable and wait forever)
        end_behavior = self._setup_scenario_end(config)
        if end_behavior:
            self.behavior_tree.add_child(end_behavior)

        # Add optional lights behavior
        lights = self._create_lights_behavior()
        if lights:
            self.scenario_tree.add_child(lights)

        # Add optional weather behavior
        weather = self._create_weather_behavior()
        if weather:
            self.scenario_tree.add_child(weather)

        # Add main behavior sequence to tree
        self.scenario_tree.add_child(self.behavior_tree)

        # Build criteria tree if enabled
        if self.criteria_enable:
            criteria = self._create_test_criteria()

            # Criteria returned as pre-built composite tree
            if isinstance(criteria, py_trees.composites.Composite):
                self.criteria_tree = criteria

            # Criteria returned as list - wrap in parallel composite
            elif isinstance(criteria, list):
                for criterion in criteria:
                    criterion.terminate_on_failure = terminate_on_failure

                self.criteria_tree = py_trees.composites.Parallel(name="Test Criteria",
                                                                  policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
                self.criteria_tree.add_children(criteria)
                self.criteria_tree.setup(timeout=1)

            else:
                raise ValueError("WARNING: Scenario {} couldn't be setup, make sure the criteria is either "
                                 "a list or a py_trees.composites.Composite".format(self.name))

            self.scenario_tree.add_child(self.criteria_tree)

        # Add timeout node (scenario deadline)
        self.timeout_node = self._create_timeout_behavior()
        if self.timeout_node:
            self.scenario_tree.add_child(self.timeout_node)

        # Add actor control update node
        self.scenario_tree.add_child(UpdateAllActorControls())

        # Setup behavior tree (initializes all nodes)
        self.scenario_tree.setup(timeout=1)

    def _initialize_environment(self, world):
        """
        Default initialization of weather and road friction.
        Override this method in child class to provide custom initialization.
        """

        # Set the appropriate weather conditions
        world.set_weather(self.config.weather)

        # Set the appropriate road friction
        if self.config.friction is not None:
            friction_bp = world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(self.config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            world.spawn_actor(friction_bp, transform)

    def _initialize_actors(self, config):
        """
        Default initialization of other actors.
        Override this method in child class to provide custom initialization.
        """
        if config.other_actors:
            new_actors = CarlaDataProvider.request_new_actors(config.other_actors)
            if not new_actors:
                raise Exception("Error: Unable to add actors")

            for new_actor in new_actors:
                self.other_actors.append(new_actor)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        if config.trigger_points and config.trigger_points[0]:
            start_location = config.trigger_points[0].location
        else:
            return None

        # Scenario is not part of a route, wait for the ego to move
        if not self.route_mode or config.route_var_name is None:
            return InTimeToArrivalToLocation(self.ego_vehicles[0], 2.0, start_location)

        # Scenario is part of a route.
        check_name = "WaitForBlackboardVariable: {}".format(config.route_var_name)
        return WaitForBlackboardVariable(config.route_var_name, True, False, name=check_name)

    def _setup_scenario_end(self, config):
        """
        This function adds and additional behavior to the scenario, which is triggered
        after it has ended. The Blackboard variable is set to False to indicate the scenario has ended.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        if not self.route_mode or config.route_var_name is None:
            return None

        # Scenario is part of a route.
        end_sequence = py_trees.composites.Sequence()
        name = "Reset Blackboard Variable: {} ".format(config.route_var_name)
        end_sequence.add_child(py_trees.blackboard.SetBlackboardVariable(name, config.route_var_name, False))
        end_sequence.add_child(WaitForever())  # scenario can't stop the route

        return end_sequence

    def _create_behavior(self):
        """Create and return the scenario behavior tree.

        This abstract method must be implemented by all scenario subclasses.
        It defines the sequence of actions, conditions, and control flow that
        constitute the scenario's behavior.

        Returns:
            py_trees.behaviour.Behaviour: Root node of scenario behavior (often
                a Sequence or Parallel composite containing atomic behaviors)

        Example:
            >>> def _create_behavior(self):
            ...     sequence = py_trees.composites.Sequence("MyScenario")
            ...     sequence.add_child(ActorTransformSetter(...))
            ...     sequence.add_child(DriveDistance(...))
            ...     return sequence

        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def _create_test_criteria(self):
        """Create and return scenario evaluation criteria.

        This abstract method must be implemented by all scenario subclasses.
        It defines the success/failure conditions that determine if the ego
        vehicle passed the scenario.

        Returns:
            list or py_trees.composites.Composite: Either a list of Criterion
                objects or a pre-built composite tree of criteria

        Example:
            >>> def _create_test_criteria(self):
            ...     return [
            ...         CollisionTest(self.ego_vehicles[0]),
            ...         DrivenDistanceTest(self.ego_vehicles[0], 100),
            ...         ReachedRegionTest(self.ego_vehicles[0], region)
            ...     ]

        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def _create_weather_behavior(self):
        """
        Default empty initialization of the weather behavior,
        responsible of controlling the weather during the simulation.
        Override this method in child class to provide custom initialization.
        """
        pass

    def _create_lights_behavior(self):
        """
        Default empty initialization of the lights behavior,
        responsible of controlling the street lights during the simulation.
        Override this method in child class to provide custom initialization.
        """
        pass

    def _create_timeout_behavior(self):
        """
        Default initialization of the timeout behavior.
        Override this method in child class to provide custom initialization.
        """
        return TimeOut(self.timeout, name="TimeOut")  # Timeout node

    def change_control(self, control):  # pylint: disable=no-self-use
        """
        This is a function that changes the control based on the scenario determination
        :param control: a carla vehicle control
        :return: a control to be changed by the scenario.

        Note: This method should be overriden by the user-defined scenario behavior
        """
        return control

    def get_criteria(self):
        """
        Return the list of test criteria, including all the leaf nodes.
        Some criteria might have trigger conditions, which have to be filtered out.
        """
        criteria = []
        if not self.criteria_tree:
            return criteria

        criteria_nodes = self._extract_nodes_from_tree(self.criteria_tree)
        for criterion in criteria_nodes:
            if isinstance(criterion, Criterion):
                criteria.append(criterion)

        return criteria

    def _extract_nodes_from_tree(self, tree):  # pylint: disable=no-self-use
        """
        Returns the list of all nodes from the given tree
        """
        node_list = [tree]
        more_nodes_exist = True
        while more_nodes_exist:
            more_nodes_exist = False
            for node in node_list:
                if node.children:
                    node_list.remove(node)
                    more_nodes_exist = True
                    for child in node.children:
                        node_list.append(child)

        if len(node_list) == 1 and isinstance(node_list[0], py_trees.composites.Parallel):
            return []

        return node_list

    def terminate(self):
        """
        This function sets the status of all leaves in the scenario tree to INVALID
        """
        # Get list of all nodes in the tree
        node_list = self._extract_nodes_from_tree(self.scenario_tree)

        # Set status to INVALID
        for node in node_list:
            node.terminate(py_trees.common.Status.INVALID)

        # Cleanup all instantiated controllers
        actor_dict = {}
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass
        for actor_id in actor_dict:
            actor_dict[actor_id].reset()
        py_trees.blackboard.Blackboard().set("ActorsWithController", {}, overwrite=True)

    def remove_all_actors(self):
        """
        Remove all actors
        """
        if not hasattr(self, 'other_actors'):
            return
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if CarlaDataProvider.actor_id_exists(self.other_actors[i].id):
                    CarlaDataProvider.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []

    def get_parking_slots(self):
        """
        Returns occupied parking slots.
        """
        return self.parking_slots
