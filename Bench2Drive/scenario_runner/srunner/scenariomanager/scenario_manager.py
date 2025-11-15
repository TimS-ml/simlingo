#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Scenario Manager - Core execution engine for CARLA scenarios.

This module provides the ScenarioManager class, which is responsible for orchestrating
scenario execution via behavior trees. It handles the simulation loop, agent integration,
timing, and result analysis.

The ScenarioManager operates on py_trees behavior trees, which define the scenario logic
as a hierarchical tree of behaviors (actions, conditions, sequences, parallels, etc.).
Each simulation tick, the behavior tree is evaluated, scenarios progress, and criteria
are checked.

Key Responsibilities:
    - Managing the scenario behavior tree execution lifecycle
    - Coordinating agent sensor setup and control application
    - Synchronizing simulation ticks in synchronous mode
    - Tracking scenario timing (system time and game time)
    - Monitoring scenario health via watchdog
    - Analyzing results against success/failure criteria
    - Generating standardized result reports

Classes:
    ScenarioManager: Main class for scenario execution and management

Note:
    This module is designed as a reference implementation and should not be modified
    for normal scenario development. Extend scenarios via scenario classes instead.
"""

from __future__ import print_function
import sys
import time

import py_trees

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog


class ScenarioManager(object):
    """Core manager for scenario execution via behavior trees.

    The ScenarioManager is the execution engine that runs scenarios defined as py_trees
    behavior trees. It manages the simulation loop, ticks the behavior tree each frame,
    coordinates with autonomous agents, and tracks scenario completion.

    Execution Flow:
        1. Initialization: Create manager with debug/sync/timeout settings
        2. Load: Load scenario and optional agent via load_scenario()
        3. Execute: Run simulation loop via run_scenario() until completion
        4. Analyze: Evaluate criteria and generate reports via analyze_scenario()
        5. Cleanup: Release resources via cleanup() or stop_scenario()

    Behavior Tree Execution:
        Each simulation tick, the manager:
        - Updates game time and actor information
        - Calls agent to get control commands (if agent present)
        - Applies control to ego vehicle
        - Ticks behavior tree once (evaluates all active nodes)
        - Checks tree status (RUNNING, SUCCESS, or FAILURE)
        - Ticks CARLA world if in synchronous mode

    Attributes:
        scenario: The loaded scenario instance
        scenario_tree: Root of the py_trees behavior tree
        ego_vehicles: List of ego vehicle actors
        other_actors: List of scenario-spawned actors
        scenario_duration_system: Total wall-clock time in seconds
        scenario_duration_game: Total simulation time in seconds

    Example:
        >>> manager = ScenarioManager(debug_mode=True, sync_mode=True)
        >>> manager.load_scenario(my_scenario, agent_instance)
        >>> manager.run_scenario()
        >>> manager.analyze_scenario(stdout=True)

    Note:
        This is a reference implementation. Do not modify for scenario development.
        Create custom scenario classes instead.
    """

    def __init__(self, debug_mode=False, sync_mode=False, timeout=2.0):
        """Initialize the ScenarioManager with execution parameters.

        Args:
            debug_mode (bool): If True, prints behavior tree state each tick
            sync_mode (bool): If True, manager controls world ticking (synchronous mode)
            timeout (float): Watchdog timeout in seconds for detecting stalls
        """
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._sync_mode = sync_mode
        self._watchdog = None
        self._timeout = timeout

        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

    def _reset(self):
        """Reset all internal state for a fresh scenario run.

        Clears timing information, running status, and restarts the game time clock.
        Called automatically by load_scenario() before loading a new scenario.
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def cleanup(self):
        """Perform cleanup and resource deallocation.

        Stops watchdog, terminates scenario behavior tree, cleans up agent resources,
        and cleans up CarlaDataProvider global state. Should be called after scenario
        completion or on error.

        Note:
            This is automatically called at the end of run_scenario().
        """

        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None

        if self.scenario is not None:
            self.scenario.terminate()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

        CarlaDataProvider.cleanup()

    def load_scenario(self, scenario, agent=None):
        """Load a scenario and optional autonomous agent.

        Resets internal state, wraps the agent, stores scenario references, and
        sets up agent sensors. If an agent is provided, automatically enables
        synchronous mode for deterministic execution.

        Args:
            scenario: Scenario instance with scenario_tree, ego_vehicles, other_actors
            agent: Optional autonomous agent implementing the control policy

        Note:
            Agent is wrapped in AgentWrapper which handles sensor setup and data flow.
            Sensors are attached to ego_vehicles[0] (the primary ego vehicle).
        """
        self._reset()
        self._agent = AgentWrapper(agent) if agent else None
        if self._agent is not None:
            self._sync_mode = True
        self.scenario = scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        if self._agent is not None:
            self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """Execute the loaded scenario until completion or failure.

        Runs the main simulation loop, ticking the scenario behavior tree each frame
        until the tree status changes from RUNNING to SUCCESS or FAILURE. Tracks both
        system time (wall clock) and game time (simulation time).

        The loop:
            1. Gets current world snapshot and timestamp
            2. Ticks scenario (agent + behavior tree) via _tick_scenario()
            3. Repeats until scenario finishes or watchdog triggers
            4. Performs cleanup
            5. Records total duration

        Note:
            Automatically calls cleanup() at the end.
            Prints termination reason if scenario fails.
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        # Start watchdog to detect simulation stalls
        self._watchdog = Watchdog(float(self._timeout))
        self._watchdog.start()
        self._running = True

        # Main simulation loop
        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

        self.cleanup()

        # Record duration metrics
        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    def _tick_scenario(self, timestamp):
        """Execute one simulation tick: update state, run agent, tick behavior tree.

        This is the core per-frame logic that:
        1. Updates game time from CARLA timestamp
        2. Updates actor information in CarlaDataProvider
        3. Calls agent to compute control command
        4. Applies control to ego vehicle
        5. Ticks behavior tree once
        6. Checks if scenario completed
        7. Ticks CARLA world if in synchronous mode

        Args:
            timestamp: CARLA world snapshot timestamp with elapsed_seconds

        Note:
            Only processes if timestamp is newer than last tick (avoids double-ticking).
            In synchronous mode, world.tick() is called at the end to advance simulation.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            # Update watchdog to prevent timeout
            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update global game time and actor cache
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            # Get agent control command
            if self._agent is not None:
                ego_action = self._agent()  # pylint: disable=not-callable

            # Apply control to ego vehicle
            if self._agent is not None:
                self.ego_vehicles[0].apply_control(ego_action)

            # Tick the behavior tree (evaluates all nodes)
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            # Check if scenario finished
            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

        # Advance CARLA simulation in synchronous mode
        if self._sync_mode and self._running and self._watchdog.get_status():
            CarlaDataProvider.get_world().tick()

    def get_running_status(self):
        """Check if watchdog detected a simulation stall.

        Returns:
            bool: False if watchdog exception occurred, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """Stop scenario execution immediately.

        Sets the running flag to False, which will cause the simulation loop in
        run_scenario() to exit. Typically called by signal handlers (Ctrl+C, etc.)
        for graceful shutdown.
        """
        self._running = False

    def analyze_scenario(self, stdout, filename, junit, json):
        """Analyze scenario results and generate reports.

        Evaluates all scenario criteria (success/failure conditions) and generates
        output in requested formats. Determines overall scenario result based on:
        - Criterion status (SUCCESS, ACCEPTABLE, or FAILURE)
        - Optional vs. required criteria
        - Timeout detection

        Args:
            stdout (bool): If True, print results to stdout
            filename (str): Path for text file output, or None
            junit (str): Path for JUnit XML output, or None
            json (str): Path for JSON output, or None

        Returns:
            bool: True if any failure or timeout occurred, False if all criteria passed

        Note:
            Result priority: FAILURE > TIMEOUT > ACCEPTABLE > SUCCESS
            Optional criteria do not affect overall pass/fail status.
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        # Collect all scenario criteria
        criteria = self.scenario.get_criteria()
        if len(criteria) == 0:
            print("Nothing to analyze, this scenario has no criteria")
            return True

        # Check each criterion for failure
        for criterion in criteria:
            if (not criterion.optional and
                    criterion.test_status != "SUCCESS" and
                    criterion.test_status != "ACCEPTABLE"):
                failure = True
                result = "FAILURE"
            elif criterion.test_status == "ACCEPTABLE":
                result = "ACCEPTABLE"

        # Check for timeout
        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        # Generate output reports
        output = ResultOutputProvider(self, result, stdout, filename, junit, json)
        output.write()

        return failure or timeout
